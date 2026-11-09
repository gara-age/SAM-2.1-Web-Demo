import time
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.camera import Frame
from backend.service import TrackingService
from backend.app import app
import backend.app as module


def scene(index):
    image=np.zeros((120,200,3),np.uint8)
    image[30:90,30+2*index:90+2*index]=np.random.default_rng(32).integers(30,240,(60,60,3),dtype=np.uint8)
    mask=np.zeros((120,200),bool);mask[30:90,30+2*index:90+2*index]=True
    return Frame(index+1,time.time(),image),mask


def prepare(root):
    service=TrackingService(recordings_root=root)
    service.shutdown_event.set();service.worker.join();service.motion_worker.join()
    service.camera.connected=True;service.camera.source='0'
    service.camera.frames.append(scene(0)[0])
    return service


@pytest.mark.parametrize('model',['sam2.1','sam3.1'])
def test_live_click_captures_exact_frame_catches_up_and_starts_without_editing(tmp_path,model):
    service=prepare(tmp_path);service.model_name=model;calls=[]
    def infer(frames,masks,target=None,points=None):
        calls.append((frames[0].id,points))
        # Camera reception continues while SAM computes the clicked frame.
        service.camera.frames.extend([scene(1)[0],scene(2)[0]])
        return [(frames[0],{target:scene(0)[1]})]
    try:
        service.engine=SimpleNamespace(status='ready',device='cpu',error=None,precision='test',infer=infer,close=lambda:None)
        result=service.select_live(1,55,55)
        assert calls==[(1,[dict(x=55,y=55,label=1)])]
        assert result['clicked_frame_id']==1 and result['packet']['frame']['id']==3
        assert result['status']['running'] and not result['status']['editing']
        assert not result['status']['selecting_live']
        obj=result['packet']['objects'][0]
        assert obj['points'][0]['frame_id']==1 and obj['center'][0]>62
        assert service.ai_seed[0].id==1 and service.anchor.id==3
    finally:service.close()


def test_cancelled_or_expired_live_selection_never_commits_old_mask(tmp_path):
    service=prepare(tmp_path)
    def infer(frames,masks,target=None,points=None):
        service.reset()
        return [(frames[0],{target:scene(0)[1]})]
    try:
        service.engine=SimpleNamespace(status='ready',device='cpu',error=None,infer=infer)
        with pytest.raises(ValueError,match='만료'):service.select_live(99,55,55)
        with pytest.raises(ValueError,match='좌표'):service.select_live(1,201,55)
        with pytest.raises(ValueError,match='상태가 변경'):service.select_live(1,55,55)
        assert not service.objects and not service.running and not service.selecting_live
        def delayed(frames,masks,target=None,points=None):
            service.camera.frames.clear();service.camera.frames.append(scene(3)[0])
            return [(frames[0],{target:scene(0)[1]})]
        service.engine.infer=delayed
        with pytest.raises(ValueError,match='버퍼보다'):service.select_live(1,55,55)
    finally:service.close()


def test_live_selection_api_origin_and_input_guards(tmp_path):
    with TestClient(app) as client:
        service=module.service
        service.engine.status='ready'
        assert client.post('/api/objects/live',json={'frame_id':1,'x':0,'y':0}).status_code==400
        assert client.post('/api/objects/live',json={'frame_id':1,'x':-1,'y':0}).status_code==422
        assert client.post('/api/objects/live',json={'frame_id':1,'x':0,'y':0},headers={'origin':'https://other.example'}).status_code==403
