"""Real SAM 3.1 text selection and short video propagation, isolated from UI."""
from pathlib import Path
import sys
import time
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from backend.camera import Frame
from backend.sam31 import Sam31Engine

root=Path(__file__).resolve().parents[1]
paths=sorted((root/'vendor'/'sam3'/'assets'/'videos'/'0001').glob('*.jpg'),key=lambda p:int(p.stem))
assert paths,'Official example frames missing'
frames=[Frame(i,time.time(),cv2.imread(str(p))) for i,p in enumerate(paths[:3])]
engine=Sam31Engine()
try:
    start=time.perf_counter();engine.load();load_seconds=time.perf_counter()-start
    print('Model loaded in '+str(round(load_seconds,2))+'s',flush=True)
    start=time.perf_counter();masks=engine.segment_text(frames[0],'person');selection_seconds=time.perf_counter()-start
    assert masks and any(mask.any() for mask in masks.values()),'No person masks from official example'
    print('Text selection: '+str(len(masks))+' objects in '+str(round(selection_seconds,2))+'s',flush=True)
    start=time.perf_counter();output=engine.infer(frames,{i+1:m for i,m in enumerate(list(masks.values())[:1])});tracking_seconds=time.perf_counter()-start
    print('Tracked mask areas: '+str([{i:int(m.sum()) for i,m in objects.items()} for _,objects in output]),flush=True)
    assert len(output)==3 and any(m.any() for m in output[-1][1].values()),'No propagated masks'
    report=dict(model='SAM 3.1',precision=engine.precision,load_seconds=round(load_seconds,2),
                selection_seconds=round(selection_seconds,2),tracking_seconds=round(tracking_seconds,2),
                selected_objects=len(masks),tracked_frames=len(output),source='Meta official example video 0001')
    (root/'logs'/'sam31-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
finally:
    engine.close()
