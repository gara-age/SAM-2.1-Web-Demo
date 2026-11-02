import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.camera import Camera
from backend.app import app
import backend.app as module


def video_file(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for value in (20,60,100,140,180):
        writer.write(np.full((48,64,3),value,np.uint8))
    writer.release()
    return path


def test_video_demand_playback_keeps_every_frame_and_timeline(tmp_path):
    path=video_file(tmp_path/'clip.avi');camera=Camera()
    try:
        camera.open_media(path,'한글 동영상.avi')
        first=camera.latest();time.sleep(.15)
        assert camera.latest().id==first.id  # No advance while selecting or paused.
        assert first.source_index==0 and camera.media_total==5
        outputs=[first]+camera.after(first.id,3)
        outputs+=camera.after(outputs[-1].id,3)
        assert [f.source_index for f in outputs]==list(range(5))
        assert [f.source_time for f in outputs]==[0,.1,.2,.3,.4]
        assert all(abs(f.image.mean()-value)<4 for f,value in zip(outputs,(20,60,100,140,180)))
        assert camera.media_ended and camera.connected
        assert camera.after(outputs[-1].id,3)==[]
        old_id=outputs[-1].id
        camera.open_media(path,'한글 동영상.avi')
        assert camera.latest().id>old_id and camera.latest().source_index==0
        assert not camera.media_ended
    finally:camera.stop()
    assert camera.media_cap is None


def test_upload_image_video_rewind_invalid_files_and_exports(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    image=np.full((48,64,3),100,np.uint8)
    encoded=cv2.imencode('.png',image)[1].tobytes()
    with TestClient(app) as client:
        response=client.post('/api/media/upload',params={'filename':'../테스트.png'},content=encoded)
        assert response.status_code==200,response.text
        data=response.json();service=module.service
        assert data['status']['input_kind']=='image'
        assert data['status']['source_name']=='테스트.png'
        assert data['frame']['width']==64 and data['frame']['height']==48
        assert client.get('/api/cameras').json()==[]
        assert service.camera.thread is None
        first=service.camera.latest();service.pin(first.id)
        obj_id=service.new_object()['id']
        service.masks={obj_id:np.ones((48,64),bool)}
        service._publish(first,service.masks)
        service.control('start')
        assert not service.running and not service.editing
        assert client.get('/api/export/json').json()['frame']['source_time']==0
        assert client.post('/api/scans',json={'object_id':obj_id}).status_code==400
        assert client.post('/api/media/upload',params={'filename':'bad.exe'},content=encoded).status_code==400
        assert client.post('/api/media/upload',params={'filename':'empty.png'},content=b'').status_code==400
        assert client.post('/api/media/upload',params={'filename':'broken.png'},content=b'broken').status_code==400
        monkeypatch.setattr(module,'MAX_UPLOAD_BYTES',8)
        assert client.post('/api/media/upload',params={'filename':'large.png'},content=encoded).status_code==413
        monkeypatch.setattr(module,'MAX_UPLOAD_BYTES',250*1024*1024)
        assert service.camera.latest() is first and obj_id in service.objects
        path=video_file(tmp_path/'input.avi')
        result=client.post('/api/media/upload',params={'filename':'영상.avi'},content=path.read_bytes()).json()
        assert result['status']['input_kind']=='video' and not service.objects
        frames=service.camera.after(result['frame']['id'],2)
        assert [f.source_index for f in frames]==[1,2]
        rewind=client.post('/api/media/rewind').json()
        assert rewind['frame']['source_index']==0
        assert rewind['frame']['id']>frames[-1].id
        assert len(list((tmp_path/'data'/'uploads').iterdir()))==2
        assert client.post('/api/media/upload',params={'filename':'x.png'},content=encoded,headers={'origin':'https://other.example'}).status_code==403


def test_tracking_worker_reaches_video_end_without_skipping(tmp_path):
    from backend.service import TrackingService
    from types import SimpleNamespace
    service=TrackingService(recordings_root=tmp_path/'results');seen=[]
    service.select_tracking_mode('precise')
    def infer(frames,masks):
        time.sleep(.1)
        seen.extend(f.source_index for f in frames[1:])
        return [(f,{k:m.copy() for k,m in masks.items()}) for f in frames]
    try:
        service.engine=SimpleNamespace(infer=infer,status='ready',error=None,device='cpu')
        service.open_media(video_file(tmp_path/'clip.avi'),'clip.avi')
        frame=service.camera.latest();service.pin(frame.id)
        obj_id=service.new_object()['id'];service.masks={obj_id:np.ones((48,64),bool)}
        service._publish(frame,service.masks)
        service.control('start')
        deadline=time.monotonic()+4
        while service.running and time.monotonic()<deadline:time.sleep(.02)
        assert not service.running and service.camera.media_ended
        assert seen==[1,2,3,4]
        assert service.packet['frame']['source_index']==4
        assert service.packet['objects'][0]['area']==48*64
    finally:service.close()
