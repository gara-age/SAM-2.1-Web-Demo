"""On-demand monocular relative-depth point clouds; never feeds back into SAM."""
import base64
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def point_cloud(image, mask, inverse_depth):
    """Back-project relative inverse depth with assumed intrinsics, not metres."""
    h,w=mask.shape
    if image.shape[:2]!=(h,w) or inverse_depth.shape!=(h,w):
        raise ValueError('깊이·마스크·영상 프레임 크기가 다릅니다.')
    valid=np.isfinite(inverse_depth)&(inverse_depth>0)
    mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid
    if np.count_nonzero(mask)<12:raise ValueError('표시할 객체 표면이 없습니다. 대상을 다시 선택하세요.')
    lo,hi=np.percentile(inverse_depth[valid],[2,98])
    relative=np.clip((inverse_depth-lo)/max(float(hi-lo),1e-6),0,1)
    z=1/(.3+.7*relative)
    step=max(1,int(np.ceil(max(h,w)/320)))
    yy,xx=np.mgrid[0:h:step,0:w:step];keep=mask[yy,xx]
    yy,xx=yy[keep],xx[keep]
    if len(xx)<3:raise ValueError('객체가 너무 작아 점군을 만들 수 없습니다.')
    zz=z[yy,xx];f=.9*w
    xyz=np.column_stack(((xx-(w-1)/2)*zz/f,((h-1)/2-yy)*zz/f,-zz))
    # Display framing is relative; preserve perspective and the depth variation.
    span=max(float(np.ptp(xyz[:,0])),float(np.ptp(xyz[:,1])),1e-5)
    xyz=(xyz-np.median(xyz,axis=0))*1.8/span
    rgb=image[yy,xx,::-1].copy()
    return dict(count=len(xx),positions=base64.b64encode(xyz.astype('<f4').tobytes()).decode('ascii'),
                colors=base64.b64encode(rgb.tobytes()).decode('ascii'),
                depth_range=round(float(np.ptp(xyz[:,2])),4),
                units='relative',intrinsics='assumed',model='Depth Anything V2 Small',input_size=518)


class DepthEngine:
    def __init__(self):self.model=None

    def infer(self,image):
        import torch
        if self.model is None:
            vendor=ROOT/'vendor'/'depth-anything-v2'
            weights=ROOT/'models'/'depth_anything_v2_vits.pth'
            if not vendor.is_dir() or not weights.is_file():
                raise ValueError('깊이 모델이 없습니다. setup-depth.ps1을 실행하세요.')
            if str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
            from depth_anything_v2.dpt import DepthAnythingV2
            self.device='cuda' if torch.cuda.is_available() else 'cpu'
            model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
            self.model=model.to(self.device).eval()
        with torch.inference_mode():
            return self.model.infer_image(image,input_size=518)


class DepthPreview:
    def __init__(self,service,engine=None):
        self.service=service;self.engine=engine or DepthEngine()
        self.lock=threading.RLock();self.stop=threading.Event()
        self.target=None;self.lease=0;self.result=None;self.error=None;self.state='idle'
        self.last=None
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()

    def request(self,object_id,since=None):
        with self.service.lock:
            revision=self.service.revision
            packet=self.service.packet
            available=packet and any(o['id']==object_id and o['area']>0 for o in packet['objects'])
        with self.lock:
            key=(revision,object_id)
            if not available:
                self.target=None;self.result=None;self.last=None;self.state='waiting';self.error=None
            else:
                if key!=self.target:
                    self.target=key;self.result=None;self.last=None;self.error=None;self.state='waiting'
                self.lease=time.monotonic()+2
            result=self.result if self.target==key else None
            return dict(state=self.state,error=self.error,result=result if result and result['version']!=since else None,
                        frame_id=result['frame_id'] if result else None)

    def _loop(self):
        while not self.stop.wait(.03):
            with self.lock:
                target=self.target;active=time.monotonic()<self.lease
            if not target or not active:continue
            with self.service.lock:
                if self.service.revision!=target[0] or not self.service.history:continue
                frame,masks=self.service.history[-1];mask=masks.get(target[1])
                if mask is None or not mask.any():continue
                version=self.service.version
                token=(*target,frame.id,version)
                if token==self.last:continue
                image=frame.image;mask=mask.copy()
            # SAM and depth share the GPU. Never enqueue behind SAM inference.
            if not self.service.engine_lock.acquire(blocking=False):continue
            started=time.monotonic()
            try:
                with self.lock:self.state='loading' if getattr(self.engine,'model',True) is None else 'processing'
                try:depth=self.engine.infer(image)
                finally:self.service.engine_lock.release()
                cloud=point_cloud(image,mask,depth)
                cloud.update(frame_id=frame.id,object_id=target[1],timestamp=frame.timestamp,
                             version=version,
                             processing_ms=round((time.monotonic()-started)*1000,1))
                with self.service.lock:current=self.service.revision==target[0]
                with self.lock:
                    if self.target==target and current and time.monotonic()<self.lease:
                        self.result=cloud;self.last=token;self.error=None;self.state='ready'
            except Exception as exc:
                with self.lock:
                    if self.target==target:self.error=str(exc);self.state='error';self.last=token
            self.stop.wait(max(0,.2-(time.monotonic()-started)))

    def close(self):
        self.stop.set();self.thread.join(timeout=30)

