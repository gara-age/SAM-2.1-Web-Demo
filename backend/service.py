import base64
from collections import deque
import threading
import time
import cv2
from .camera import Camera
from .geometry import measure, png_mask
from .model import SamEngine
from .motion import MotionTracker

COLORS = ['#48ddac','#69a6ff','#ffb866','#ce8eff','#ff788e','#73dce8']


def frame_packet(frame):
    ok,jpeg = cv2.imencode('.jpg',frame.image,[cv2.IMWRITE_JPEG_QUALITY,82])
    if not ok:
        raise RuntimeError('영상 인코딩 실패')
    h,w = frame.image.shape[:2]
    return dict(id=frame.id,timestamp=frame.timestamp,width=w,height=h,
                source_index=frame.source_index,source_time=frame.source_time,
                jpeg=base64.b64encode(jpeg).decode('ascii'))


class TrackingService:
    def __init__(self):
        self.camera = Camera()
        self.engine = SamEngine()
        self.model_name = 'sam2.1'
        self.lock = threading.RLock()
        self.engine_lock=threading.RLock()
        self.tracking_mode='smooth'
        self.revision=0
        self.motion=None
        self.ai_seed=None
        self.ai_busy=False
        self.ai_updates=0
        self.ai_retry_at=0
        self.ai_error=None
        self.motion_fps=None
        self.last_ai_frame=None
        self.shutdown_event = threading.Event()
        self.running = False
        self.editing = False
        self.objects = {}
        self.next_id = 1
        self.anchor = None
        self.masks = {}
        self.history = deque(maxlen=12)
        self.packet = None
        self.version = 0
        self.error = None
        self.inference_ms = None
        self.inference_fps = None
        self.worker = threading.Thread(target=self._loop,daemon=True)
        self.worker.start()
        self.motion_worker=threading.Thread(target=self._motion_loop,daemon=True)
        self.motion_worker.start()

    def status(self):
        return dict(server_version='0.3.0',model_name=self.model_name,
                    tracking_mode=self.tracking_mode,ai_busy=self.ai_busy,ai_updates=self.ai_updates,
                    ai_error=self.ai_error,
                    motion_fps=self.motion_fps,last_ai_frame=self.last_ai_frame,
                    model_precision=getattr(self.engine,'precision',None) if self.model_name=='sam3.1' else 'FP32',
                    camera_connected=self.camera.connected,camera_error=self.camera.error,
                    source=self.camera.source,camera_fps=round(self.camera.fps,1),
                    input_kind=self.camera.input_kind,source_name=self.camera.source_name,
                    media_index=self.camera.media_index,media_total=self.camera.media_total,
                    media_fps=self.camera.media_fps,media_ended=self.camera.media_ended,
                    model_status=self.engine.status,model_error=self.engine.error,
                    device=self.engine.device,running=self.running,editing=self.editing,
                    inference_ms=self.inference_ms,inference_fps=self.inference_fps,
                    error=self.error,version=self.version,
                    active_objects=sum(1 for m in self.masks.values() if m.any()))

    def load(self):
        with self.lock,self.engine_lock:
            self.engine.load()
        return self.status()

    def release_model(self):
        import gc
        import torch
        self.running=False
        with self.lock,self.engine_lock:
            if self.model_name=='sam3.1':self.engine.close()
            else:
                self.engine.predictor=None
                self.engine.status='not_loaded'
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
        return self.status()

    def select_model(self,name):
        if name not in ('sam2.1','sam3.1'):raise ValueError('지원하지 않는 모델입니다.')
        self.running=False
        self.camera.pause_media()
        with self.lock:
            if name!=self.model_name:
                self.release_model();self.reset()
                if name=='sam3.1':
                    from .sam31 import Sam31Engine
                    self.engine=Sam31Engine()
                else:self.engine=SamEngine()
                self.model_name=name
        return self.status()

    def select_text(self,frame_id,text):
        from .prompts import concept_prompt
        prompt=concept_prompt(text)
        if self.model_name!='sam3.1':raise ValueError('텍스트 선택은 SAM 3.1 모드에서 사용할 수 있습니다.')
        self.running=False
        self.camera.pause_media();self.revision+=1
        with self.lock:
            frame=self.camera.find(frame_id)
            if frame is None:
                record=next((r for r in reversed(self.history) if r[0].id==frame_id),None)
                frame=record[0] if record else None
            if frame is None:raise ValueError('프레임이 만료됐습니다. 최신 영상으로 다시 선택하세요.')
            start=time.perf_counter()
            with self.engine_lock:predictions=self.engine.segment_text(frame,prompt)
            predictions={i:m for i,m in predictions.items() if m.any()}
            if not predictions:raise ValueError(f'"{prompt}"에 해당하는 객체를 찾지 못했습니다.')
            self.objects.clear();self.masks.clear();self.history.clear()
            for mask in list(predictions.values())[:4]:
                obj_id=self.next_id;self.next_id+=1
                self.objects[obj_id]=dict(id=obj_id,name=f'{prompt} {obj_id:02}',
                    color=COLORS[(obj_id-1)%len(COLORS)],points=[],path=[],source_prompt=prompt)
                self.masks[obj_id]=mask
            self.anchor=frame;self.editing=True;self.error=None
            self.inference_ms=round((time.perf_counter()-start)*1000,1)
            self._publish(frame,self.masks,False)
            return dict(packet=self.packet,prompt=prompt,count=len(self.objects),model='sam3.1')

    def start_camera(self, **kwargs):
        self.running = False
        with self.lock:
            self.reset()
            self.camera.start(**kwargs)
        return self.status()

    def open_media(self,path,name):
        self.running=False
        with self.lock:
            self.camera.open_media(path,name)
            self.reset()
        return self.status()

    def rewind_media(self):
        if self.camera.input_kind not in ('image','video') or self.camera.media_path is None:
            raise ValueError('먼저 이미지나 동영상을 업로드하세요.')
        return self.open_media(self.camera.media_path,self.camera.source_name)

    def stop_camera(self):
        self.running = False
        with self.lock:
            self.camera.stop()
            self.editing = False
        return self.status()

    def reset(self):
        self.running = False
        self.camera.pause_media()
        with self.lock:
            self.revision+=1;self.motion=None;self.ai_seed=None
            self.ai_updates=0;self.motion_fps=None;self.last_ai_frame=None
            self.ai_error=None;self.ai_retry_at=0
            self.objects.clear()
            self.masks.clear()
            self.history.clear()
            self.anchor = None
            self.packet = None
            self.editing = False
            self.error = None
            self.inference_ms = None
            self.inference_fps = None
            self.version += 1
        return self.status()

    def pin(self, frame_id):
        self.running = False
        self.camera.pause_media()
        self.revision+=1
        with self.lock:
            self.motion=None;self.ai_seed=None
            record = next((r for r in reversed(self.history) if r[0].id==frame_id),None)
            if self.objects and not record:
                raise ValueError('객체 보정은 분석 프레임에서 가능합니다. 분석 보기를 선택하고 다시 시도하세요.')
            frame = record[0] if record else self.camera.find(frame_id)
            if frame is None:
                raise ValueError('선택한 프레임이 버퍼에서 만료되었습니다. 최신 영상을 다시 선택하세요.')
            self.anchor = frame
            self.masks = {i:m for i,m in record[1].items() if i in self.objects} if record else {}
            for obj in self.objects.values():
                obj['path'] = [p for p in obj['path'] if p['frame_id']<=frame_id]
            self.editing = True
            self._publish(frame,self.masks,append_path=False)
            return self.packet

    def new_object(self):
        with self.lock:
            if not self.editing or self.anchor is None:
                raise ValueError('먼저 프레임을 고정하세요.')
            if self.model_name=='sam3.1' and len(self.objects)>=4:
                raise ValueError('현재 SAM 3.1 실험 모드는 최대 4개 객체를 지원합니다.')
            obj_id = self.next_id
            self.next_id += 1
            self.objects[obj_id] = dict(id=obj_id,name=f'Object {obj_id:02}',
                                       color=COLORS[(obj_id-1)%len(COLORS)],points=[],path=[])
            self._publish(self.anchor,self.masks,False)
            return dict(id=obj_id,packet=self.packet)

    def set_points(self, obj_id,frame_id,points):
        with self.lock:
            if not self.editing or self.anchor is None or self.anchor.id!=frame_id:
                raise ValueError('현재 고정된 프레임과 선택 프레임이 다릅니다.')
            if obj_id not in self.objects:
                raise ValueError('객체가 존재하지 않습니다.')
            h,w = self.anchor.image.shape[:2]
            if any(not (0<=p['x']<w and 0<=p['y']<h) for p in points):
                raise ValueError('좌표가 원본 영상 범위를 벗어났습니다.')
            if points and not any(p['label']==1 for p in points):
                raise ValueError('하나 이상의 포함 지점이 필요합니다.')
            if points:
                start = time.perf_counter()
                with self.engine_lock:outputs = self.engine.infer([self.anchor],self.masks,target=obj_id,points=points)
                self.masks = outputs[-1][1]
                self.inference_ms = round((time.perf_counter()-start)*1000,1)
            else:
                self.masks.pop(obj_id,None)
            self.objects[obj_id]['points'] = [dict(**p,frame_id=frame_id) for p in points]
            self.error = None
            self._publish(self.anchor,self.masks,False)
            return self.packet

    def delete(self,obj_id):
        self.running = False
        self.camera.pause_media();self.revision+=1
        with self.lock:
            if obj_id not in self.objects:
                raise ValueError('객체가 존재하지 않습니다.')
            self.objects.pop(obj_id)
            self.masks.pop(obj_id,None)
            if self.anchor:
                self._publish(self.anchor,self.masks,False)
        return self.status()

    def control(self,action):
        if action=='pause':
            self.running = False
            self.camera.pause_media();self.revision+=1
            with self.lock:
                return self.status()
        if action=='stop':
            return self.reset()
        with self.lock:
            if not self.camera.connected:
                raise ValueError('카메라를 먼저 시작하세요.')
            if not any(m.any() for m in self.masks.values()):
                raise ValueError('포함 지점을 추가하여 객체를 분할한 뒤 추적하세요.')
            self.editing = False
            self.error = None
            self.revision+=1
            if self.tracking_mode=='smooth' and self.camera.input_kind!='image':
                self.motion=MotionTracker(self.anchor,self.masks)
                self.ai_seed=(self.anchor,{i:m.copy() for i,m in self.masks.items()})
            self.camera.play_media(self.tracking_mode=='smooth')
            self.running = self.camera.input_kind!='image'
        return self.status()

    def clear_paths(self):
        with self.lock:
            for obj in self.objects.values():
                obj['path'] = []
            if self.anchor:
                self._publish(self.anchor,self.masks,False)
        return self.status()

    def _publish(self,frame,masks,append_path=True):
        objects = []
        for obj_id,obj in self.objects.items():
            mask = masks.get(obj_id)
            geometry = measure(mask) if mask is not None else dict(
                area=0,area_percent=0,bbox=None,width=None,height=None,center=None,corners=[],contours=[])
            if append_path and geometry['center']:
                obj['path'] = (obj['path'] + [dict(frame_id=frame.id,center=geometry['center'])])[-120:]
            objects.append(dict(**obj,**geometry,status=('Active' if geometry['area'] else
                              'Lost' if obj['points'] or obj.get('source_prompt') or obj_id in masks else 'Pending'),frame_id=frame.id,
                              timestamp=frame.timestamp,
                              mask_png=png_mask(mask) if mask is not None else None))
        self.packet = dict(frame=frame_packet(frame),objects=objects,
                          tracking_method='optical-flow+SAM' if self.motion is not None and self.motion.frame.id==frame.id and self.tracking_mode=='smooth' else 'SAM',
                          model=self.model_name)
        self.history.append((frame,dict(masks)))
        self.version += 1

    def _loop(self):
        while not self.shutdown_event.wait(.02):
            if not self.running:
                continue
            if not self.camera.connected:
                self.running = False
                self.error = self.camera.error or '카메라 연결이 해제되었습니다.'
                continue
            try:
                if self.tracking_mode=='smooth':
                    self._smooth_ai()
                    continue
                with self.lock:
                    if not self.running or self.anchor is None:
                        continue
                    frames = self.camera.after(self.anchor.id,limit=3)
                    if not frames:
                        if self.camera.media_ended:self.running=False
                        continue
                    start = time.perf_counter()
                    with self.engine_lock:outputs = self.engine.infer([self.anchor]+frames,self.masks)
                    elapsed = time.perf_counter()-start
                    self.inference_ms = round(elapsed*1000,1)
                    self.inference_fps = round(len(frames)/elapsed,2)
                    for frame,masks in outputs[1:]:
                        self.anchor,self.masks = frame,masks
                        self._publish(frame,masks)
            except Exception as exc:
                self.error = f'추론 실패: {exc}'
                self.running = False

    def select_tracking_mode(self,mode):
        if mode not in ('smooth','precise'):raise ValueError('지원하지 않는 추적 방식입니다.')
        self.control('pause')
        with self.lock:self.tracking_mode=mode
        return self.status()

    def _smooth_ai(self):
        with self.lock:
            if not self.running or self.motion is None or self.ai_seed is None or time.monotonic()<self.ai_retry_at:return
            seed,masks=self.ai_seed;target=self.motion.frame
            if target.id==seed.id or not any(m.any() for m in masks.values()):return
            motion=self.motion;revision=self.revision;base=motion.snapshot()
            self.ai_busy=True
        try:
            start=time.perf_counter()
            with self.engine_lock:outputs=self.engine.infer([seed,target],masks)
            elapsed=time.perf_counter()-start
            with self.lock:
                if not self.running or revision!=self.revision or self.motion is not motion:return
                self.inference_ms=round(elapsed*1000,1);self.inference_fps=round(1/elapsed,2)
                if outputs and motion.correct(outputs[-1][1],base):
                    self.masks={i:m.copy() for i,m in motion.masks.items()}
                    self.ai_updates+=1;self.last_ai_frame=target.id
                self.ai_error=None
                self.ai_seed=(motion.frame,{i:m.copy() for i,m in motion.masks.items()})
        except Exception as exc:
            with self.lock:
                if revision==self.revision:
                    self.ai_error=f'SAM 보정 실패 · 광학흐름만 진행 중: {exc}'
                    self.ai_retry_at=time.monotonic()+5
        finally:self.ai_busy=False

    def _motion_loop(self):
        last_time=time.monotonic()
        while not self.shutdown_event.wait(.015):
            if not self.running or self.tracking_mode!='smooth':continue
            try:
                with self.lock:
                    if not self.running or self.motion is None:continue
                    frame=self.camera.latest()
                    if frame is not None and frame.id!=self.motion.frame.id:
                        self.masks=dict(self.motion.advance(frame))
                        self.anchor=frame
                        self._publish(frame,self.masks)
                        now=time.monotonic();self.motion_fps=round(1/max(.001,now-last_time),1);last_time=now
                    elif self.camera.media_ended:self.running=False
            except Exception as exc:
                self.error=f'연속 추적 실패: {exc}';self.running=False;self.camera.pause_media()

    def close(self):
        self.running = False
        self.shutdown_event.set()
        self.worker.join(timeout=30)
        self.motion_worker.join(timeout=5)
        self.camera.stop()
        if self.model_name=='sam3.1':self.engine.close()
