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


class Camera:
    def __init__(self):
        self.frames = deque(maxlen=120)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.connected = False
        self.error = None
        self.fps = 0.0
        self.next_id = 1
        self.source = None

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
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, args=(cap,width,height,fps,sample), daemon=True)
        self.thread.start()

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
        with self.lock:
            candidates = [f for f in self.frames if f.id>frame_id]
        if len(candidates)<=limit:
            return candidates
        # Sample the retained interval evenly, always including its latest frame.
        return [candidates[i] for i in np.linspace(0,len(candidates)-1,limit,dtype=int)]

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=4)
            if self.thread.is_alive():
                raise RuntimeError('카메라 종료 대기 중입니다. 잠시 후 다시 시도하세요.')
        self.thread = None
        self.connected = False
