"""Real CUDA inference on generated frames; no simulated SAM results."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from backend.camera import Frame
from backend.geometry import measure
from backend.model import SamEngine

engine=SamEngine()
frames=[]
for i in range(4):
    image=np.full((480,640,3),25,np.uint8)
    cv2.rectangle(image,(220+i*8,160),(350+i*8,300),(50,170,240),-1)
    cv2.circle(image,(120,120),40,(210,80,60),-1)
    frames.append(Frame(i+1,time.time(),image))
start=time.perf_counter()
output=engine.infer(frames[:1],{},target=1,points=[{'x':280,'y':230,'label':1}])
first=measure(output[0][1][1])
assert first['area']>1000 and first['center'] is not None,first
selection_ms=(time.perf_counter()-start)*1000
start=time.perf_counter()
output2=engine.infer(frames,output[0][1])
last=measure(output2[-1][1][1])
assert last['area']>1000 and last['center'][0]>first['center'][0]+10,(first,last)
tracking_seconds=time.perf_counter()-start
# A second object and negative-point correction use the same official API.
second=engine.infer(frames[-1:],output2[-1][1],target=2,points=[{'x':120,'y':120,'label':1},{'x':200,'y':120,'label':0}])
assert set(second[-1][1])=={1,2}
assert measure(second[-1][1][2])['area']>1000
report=dict(device=engine.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            selection_ms=round(selection_ms,1),tracking_seconds=round(tracking_seconds,3),
            tracking_fps=round(3/tracking_seconds,2),first_center=first['center'],last_center=last['center'],
            first_area=first['area'],last_area=last['area'],objects=len(second[-1][1]),
            peak_gpu_mb=round(torch.cuda.max_memory_allocated()/1024**2,1) if engine.device=='cuda' else None)
target=Path(__file__).resolve().parents[1]/'logs'/'model-smoke.json'
target.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
