import threading
import time
from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path


@dataclass
class Frame:
    id: int
    timestamp: float
    image: np.ndarray
    source_index: int | None = None
    source_time: float | None = None


class Camera:
    def __init__(self):
        self.frames = deque(maxlen=120)
        self.frame_event=threading.Event()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.connected = False
        self.error = None
        self.fps = 0.0
        self.next_id = 1
        self.source = None
        self.input_kind = 'live'
        self.source_name = None
        self.media_cap = None
        self.media_path = None
        self.media_index = 0
        self.media_total = 0
        self.media_fps = 0.0
        self.media_ended = False
        self.play_event=threading.Event()
        self.continuous=False

    @staticmethod
    def discover():
        devices = []
        for index in range(5):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                devices.append({'id': index, 'name': f'USB / Camera {index}'})
            cap.release()
        return devices

    def start(self, device, width, height, fps, synthetic=False, example=False):
        self.stop()
        cap = None
        sample=None
        if example:
            path=Path(__file__).resolve().parents[1]/'vendor'/'sam3'/'assets'/'videos'/'0001'/'0.jpg'
            sample=cv2.imread(str(path))
            if sample is None:raise ValueError('공식 예제 이미지가 없습니다. setup-sam31.ps1을 실행하세요.')
        elif not synthetic:
            cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                raise ValueError('카메라를 열 수 없습니다. 다른 앱의 카메라 사용 여부를 확인하세요.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
        with self.lock:
            self.frames.clear()
        self.error = None
        self.fps = 0
        self.connected = True
        self.source = 'example-static' if example else 'synthetic' if synthetic else str(device)
        self.input_kind = 'example' if example else 'live'
        self.source_name = None
        self.media_ended=False;self.media_index=0;self.media_total=0;self.media_fps=0
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, args=(cap,width,height,fps,sample), daemon=True)
        self.thread.start()

    def open_media(self,path,name):
        path=Path(path)
        image_extensions={'.jpg','.jpeg','.png','.bmp','.webp'}
        kind='image' if path.suffix.lower() in image_extensions else 'video'
        cap=None
        try:
            if kind=='image':
                image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            else:
                cap=cv2.VideoCapture(str(path))
                ok,image=cap.read()
                if not ok:image=None
            if image is None:raise ValueError('파일을 읽을 수 없습니다. 지원되는 이미지나 동영상 코덱인지 확인하세요.')
            h,w=image.shape[:2]
            if h*w>16_000_000 or max(h,w)>8192:
                raise ValueError('입력은 최대 1,600만 픽셀, 한 변 8192px까지 지원합니다. 해상도를 줄여주세요.')
            self.stop()
            self.media_cap=cap;cap=None
            self.media_path=path;self.source_name=name;self.input_kind=kind
            self.source='upload:'+path.stem
            self.media_index=0;self.media_ended=False
            count=self.media_cap.get(cv2.CAP_PROP_FRAME_COUNT) if self.media_cap else 1
            self.media_total=max(0,int(count)) if np.isfinite(count) else 0
            rate=self.media_cap.get(cv2.CAP_PROP_FPS) if self.media_cap else 0
            self.media_fps=float(rate) if np.isfinite(rate) and rate>0 else (30.0 if kind=='video' else 0.0)
            self.error=None;self.fps=0;self.connected=True
            with self.lock:
                self.frames.clear()
                self.frames.append(Frame(self.next_id,time.time(),image,0,0.0))
                self.next_id+=1
            self.frame_event.set()
            if kind=='video':
                self.stop_event.clear()
                self.thread=threading.Thread(target=self._video_loop,daemon=True)
                self.thread.start()
        finally:
            if cap is not None:cap.release()

    def play_media(self,continuous):
        self.continuous=continuous
        if continuous and self.input_kind=='video':self.play_event.set()

    def pause_media(self):
        self.play_event.clear()

    def _video_loop(self):
        while not self.stop_event.wait(.001):
            if not self.play_event.wait(.03):continue
            start=time.monotonic()
            with self.lock:
                if not self.play_event.is_set() or self.media_cap is None:continue
                ok,image=self.media_cap.read()
                if not ok:
                    self.media_ended=True;self.play_event.clear();continue
                self.media_index+=1
                self.frames.append(Frame(self.next_id,time.time(),image,self.media_index,self.media_index/self.media_fps))
                self.next_id+=1
                self.fps=self.media_fps
                self.frame_event.set()
            self.stop_event.wait(max(0,1/self.media_fps-(time.monotonic()-start)))

    def _loop(self, cap, width, height, fps,sample=None):
        times = deque(maxlen=60)
        try:
            while not self.stop_event.is_set():
                start = time.monotonic()
                if sample is not None:
                    image=sample.copy()
                elif cap is None:
                    image = np.full((height,width,3), (30,24,18), np.uint8)
                    x = int(width*.45 + np.sin(time.monotonic()*.4)*width*.12)
                    cv2.rectangle(image,(x-60,height//2-60),(x+60,height//2+60),(45,165,240),-1)
                    cv2.circle(image,(width//4,height//3),40,(200,90,60),-1)
                    triangle = np.array([[int(width*.78),int(height*.3)],
                                         [int(width*.68),int(height*.55)],
                                         [int(width*.88),int(height*.55)]],np.int32)
                    cv2.fillPoly(image,[triangle],(95,205,105))
                    cv2.putText(image,'SYNTHETIC INPUT - SAM 2.1 inference',(15,height-20),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)
                else:
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError('카메라 프레임 수신이 중단되었습니다.')
                frame = Frame(self.next_id, time.time(), image)
                self.next_id += 1
                with self.lock:
                    self.frames.append(frame)
                self.frame_event.set()
                times.append(time.monotonic())
                if len(times)>1:
                    self.fps = (len(times)-1)/(times[-1]-times[0])
                self.stop_event.wait(max(0,1/fps-(time.monotonic()-start)))
        except Exception as exc:
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            self.connected = False
            self.fps = 0

    def latest(self):
        with self.lock:
            return self.frames[-1] if self.frames else None

    def find(self, frame_id):
        with self.lock:
            return next((f for f in reversed(self.frames) if f.id==frame_id), None)

    def after(self, frame_id, limit=4):
        if self.input_kind in ('image','video') and not self.continuous:
            # File playback advances only when inference requests the next frames.
            # Slow models never skip video frames or run past a paused selection.
            with self.lock:
                candidates=[f for f in self.frames if f.id>frame_id]
                while self.media_cap is not None and not self.media_ended and len(candidates)<limit:
                    ok,image=self.media_cap.read()
                    if not ok:
                        self.media_ended=True
                        break
                    self.media_index+=1
                    frame=Frame(self.next_id,time.time(),image,self.media_index,self.media_index/self.media_fps)
                    self.next_id+=1;self.frames.append(frame);candidates.append(frame)
                return candidates[:limit]
        with self.lock:
            candidates = [f for f in self.frames if f.id>frame_id]
        if len(candidates)<=limit:
            return candidates
        # Sample the retained interval evenly, always including its latest frame.
        return [candidates[i] for i in np.linspace(0,len(candidates)-1,limit,dtype=int)]

    def stop(self):
        self.frame_event.set()
        self.pause_media();self.continuous=False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=4)
            if self.thread.is_alive():
                raise RuntimeError('카메라 종료 대기 중입니다. 잠시 후 다시 시도하세요.')
        self.thread = None
        if self.media_cap is not None:
            self.media_cap.release();self.media_cap=None
        self.connected = False
