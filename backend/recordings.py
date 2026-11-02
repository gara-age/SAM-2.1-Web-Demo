"""Persistent, frame-aligned SAM video results encoded as browser-compatible H.264."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def overlay_frame(frame,masks,objects,model):
    image=frame.image.copy()
    for obj_id,mask in masks.items():
        obj=objects.get(obj_id)
        if obj is None or not mask.any():continue
        color=obj.get('color','#48ddac').lstrip('#')
        bgr=tuple(int(color[i:i+2],16) for i in (4,2,0))
        image[mask]=(image[mask]*.65+np.array(bgr)*.35).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,bgr,2)
        ys,xs=np.nonzero(mask);x,y=int(xs.min()),int(ys.min())
        cv2.rectangle(image,(x,y),(int(xs.max()),int(ys.max())),bgr,1)
        cv2.putText(image,f'OBJ {obj_id:02}',(x,max(17,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,bgr,1,cv2.LINE_AA)
    label=f'{model.upper()} | frame {frame.source_index} | {frame.source_time or 0:.2f}s'
    cv2.rectangle(image,(0,0),(min(image.shape[1]-1,510),28),(15,22,28),-1)
    cv2.putText(image,label,(9,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,245,245),1,cv2.LINE_AA)
    return image


class RecordingStore:
    def __init__(self,root=None):
        self.root=Path(root) if root is not None else ROOT/'data'/'recordings'
        self.lock=threading.RLock();self.active=None;self.current=None;self.jobs=[];self.owned=set()

    def folder(self,key):
        if not re.fullmatch('[0-9a-f]{32}',key):raise ValueError('잘못된 결과 ID입니다.')
        return self.root/key

    def _save(self,data):
        with self.lock:
            folder=self.folder(data['id']);folder.mkdir(parents=True,exist_ok=True)
            temporary=folder/'result.tmp'
            temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
            for attempt in range(10):
                try:
                    temporary.replace(folder/'result.json');break
                except PermissionError:
                    if attempt==9:raise
                    time.sleep(.02)

    def public(self,data):
        if data['state'] in ('recording','paused','finalizing') and data['id'] not in self.owned:
            data=dict(data,state='failed',error='이전 서버 실행이 중단되어 결과를 완료하지 못했습니다.')
        result=dict(data)
        ready=data['state'] in ('completed','interrupted') and (self.folder(data['id'])/'tracking.mp4').is_file()
        result['video_url']=f"/api/recordings/{data['id']}/video" if ready else None
        result['duration']=round(data['frame_count']/data['fps'],3)
        return result

    def status(self):
        return self.public(self.current) if self.current else None

    def list(self):
        with self.lock:
            results=[]
            for path in self.root.glob('*/result.json'):
                try:results.append(self.public(json.loads(path.read_text(encoding='utf8'))))
                except (OSError,ValueError,KeyError):continue
            return sorted(results,key=lambda r:r['created_at'],reverse=True)

    def get(self,key):
        with self.lock:
            path=self.folder(key)/'result.json'
            if not path.is_file():raise ValueError('결과 영상을 찾을 수 없습니다.')
            return self.public(json.loads(path.read_text(encoding='utf8')))

    def start(self,camera,frame,objects,model):
        with self.lock:
            if self.active:
                self.current['state']='recording';self._save(self.current)
                return
            ffmpeg=shutil.which('ffmpeg')
            if not ffmpeg:raise ValueError('결과 영상 저장에는 FFmpeg가 필요합니다. FFmpeg 설치와 PATH를 확인하세요.')
            h,w=frame.image.shape[:2];key=uuid.uuid4().hex
            data=dict(id=key,model=model,source=camera.source,input_name=camera.source_name,
                      state='recording',frame_count=0,fps=camera.media_fps or 30,width=w,height=h,
                      first_source_index=frame.source_index,last_source_index=None,
                      expected_frames=max(0,camera.media_total-(frame.source_index or 0)),
                      created_at=time.time(),error=None,tracking_method='SAM sequential',audio=False)
            folder=self.folder(key);folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'encoder.log').open('wb')
            args=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
                  '-s',f'{w}x{h}','-r',str(data['fps']),'-i','pipe:0','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
                  '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
                  '-movflags','+faststart',str(folder/'tracking.mp4')]
            try:process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,
                                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except Exception:log.close();raise
            self.owned.add(key);self.current=data;self.active=(process,log,dict(objects));self._save(data)

    def write(self,frame,masks):
        with self.lock:
            if not self.active:return
            data=self.current
            index=frame.source_index
            if data['last_source_index'] is not None:
                if index==data['last_source_index']:return
                if index!=data['last_source_index']+1:
                    raise ValueError('결과 영상의 원본 프레임 순서가 끊겼습니다. 처음부터 새로 분석하세요.')
            process,_,objects=self.active
            image=overlay_frame(frame,masks,objects,data['model'])
            try:process.stdin.write(image.tobytes())
            except (BrokenPipeError,OSError) as exc:raise RuntimeError('결과 영상 인코딩이 중단됐습니다. encoder.log를 확인하세요.') from exc
            data['frame_count']+=1;data['last_source_index']=index;self._save(data)

    def pause(self):
        with self.lock:
            if self.active:self.current['state']='paused';self._save(self.current)

    def finish(self,complete=False,error=None):
        with self.lock:
            if not self.active:return
            process,log,_=self.active;data=self.current;self.active=None
            if complete and data['expected_frames'] and data['frame_count']!=data['expected_frames']:
                error=f"프레임 수가 맞지 않습니다: {data['frame_count']} / {data['expected_frames']}"
            data['state']='finalizing';self._save(data)
            try:process.stdin.close()
            except (BrokenPipeError,OSError):pass
            def finalize():
                failure=error
                try:
                    code=process.wait(timeout=60)
                    if code!=0:failure=f'인코더가 오류 코드 {code}로 종료했습니다. encoder.log를 확인하세요.'
                except subprocess.TimeoutExpired:
                    process.kill();process.wait();failure='결과 영상 마무리 시간이 초과됐습니다.'
                finally:log.close()
                with self.lock:
                    data.update(state='failed' if failure else 'completed' if complete else 'interrupted',error=failure)
                    self._save(data)
            job=threading.Thread(target=finalize,daemon=True);self.jobs.append(job);job.start()

    def close(self):
        self.finish()
        for job in self.jobs:job.join(timeout=65)
