import json
import shutil
import time
import threading
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.camera import Frame
from backend.recordings import RecordingStore,overlay_frame
from backend.service import TrackingService
from backend.app import app
import backend.app as module


def inputs():
    frames=[Frame(i+1,time.time(),np.full((48,64,3),60,np.uint8),i,i/10) for i in range(5)]
    mask=np.zeros((48,64),bool);mask[30:43,20:40]=True
    camera=SimpleNamespace(source='upload:test',source_name='한글.avi',media_fps=10,media_total=5)
    return frames,{1:mask},{1:{'color':'#48ddac'}},camera


def wait_result(store,key):
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        result=store.get(key)
        if result['state'] not in ('recording','paused','finalizing'):return result
        time.sleep(.05)
    raise AssertionError('Encoder did not finish')


@pytest.mark.parametrize('model',['sam2.1','sam3.1'])
def test_real_h264_has_every_frame_fps_overlay_and_survives_restart(tmp_path,model):
    assert shutil.which('ffmpeg'),'FFmpeg required for result-video tests'
    frames,masks,objects,camera=inputs();store=RecordingStore(tmp_path)
    store.start(camera,frames[0],objects,model);key=store.current['id']
    try:
        store.write(frames[0],masks);store.pause()
        assert store.status()['state']=='paused' and not store.status()['video_url']
        store.start(camera,frames[0],objects,model)
        store.write(frames[0],masks)  # Resume must not duplicate the anchor frame.
        for frame in frames[1:]:store.write(frame,masks)
        store.finish(complete=True);result=wait_result(store,key)
        assert result['state']=='completed',result
        assert result['frame_count']==5 and result['duration']==.5
        assert RecordingStore(tmp_path).get(key)['video_url']==result['video_url']
        cap=cv2.VideoCapture(str(store.folder(key)/'tracking.mp4'))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==5
        assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<.01
        ok,image=cap.read();cap.release();assert ok
        # In-mask pixel is tinted, outside the overlay/header remains original.
        assert image[35,28,1]>image[35,28,2]+20
        assert abs(float(image[35,5].mean())-60)<10
    finally:store.close()


def test_gaps_incomplete_and_old_sessions_never_claim_completed(tmp_path):
    frames,masks,objects,camera=inputs();store=RecordingStore(tmp_path)
    try:
        store.start(camera,frames[0],objects,'sam2.1');key=store.current['id']
        store.write(frames[0],masks)
        with pytest.raises(ValueError,match='순서'):store.write(frames[2],masks)
        store.finish(complete=True)
        assert wait_result(store,key)['state']=='failed'
        store.start(camera,frames[0],objects,'sam2.1');partial=store.current['id']
        store.write(frames[0],masks);store.finish()
        assert wait_result(store,partial)['state']=='interrupted'
        assert store.get(partial)['video_url']
        with pytest.raises(ValueError):store.folder('../elsewhere')
        stale=dict(store.current,state='recording');store._save(stale)
        assert RecordingStore(tmp_path).get(partial)['state']=='failed'
    finally:store.close()


def test_video_http_range_and_download(tmp_path):
    frames,masks,objects,camera=inputs();store=RecordingStore(tmp_path)
    store.start(camera,frames[0],objects,'sam3.1');key=store.current['id']
    for frame in frames:store.write(frame,masks)
    store.finish(complete=True);result=wait_result(store,key)
    with TestClient(app) as client:
        module.service.recordings=store
        assert client.get('/api/recordings').json()[0]['id']==key
        response=client.get(result['video_url'],headers={'Range':'bytes=0-127'})
        assert response.status_code==206 and len(response.content)==128
        assert response.headers['content-type']=='video/mp4'
        assert response.headers['content-range'].startswith('bytes 0-127/')
        download=client.get(result['video_url']+'?download=true')
        assert 'attachment' in download.headers['content-disposition']
        assert client.get('/api/recordings/bad/video').status_code==404
        assert client.get('/api/recordings/'+('0'*32)+'/video').status_code==404


def test_listing_and_progress_writes_do_not_conflict_on_windows(tmp_path):
    frames,masks,objects,camera=inputs();store=RecordingStore(tmp_path)
    store.start(camera,frames[0],objects,'sam3.1');key=store.current['id']
    stop=threading.Event();errors=[]
    def reader():
        try:
            while not stop.wait(.001):
                assert store.list()[0]['id']==key
                assert store.get(key)['id']==key
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=reader);thread.start()
    try:
        for frame in frames:store.write(frame,masks)
        store.finish(complete=True)
        assert wait_result(store,key)['state']=='completed'
        assert not errors
    finally:stop.set();thread.join();store.close()


@pytest.mark.parametrize('model',['sam2.1','sam3.1'])
def test_precise_pipeline_finishes_recording_for_both_model_settings(tmp_path,model):
    path=tmp_path/'input.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for _ in range(5):writer.write(np.full((48,64,3),60,np.uint8))
    writer.release()
    service=TrackingService(recordings_root=tmp_path/'results')
    try:
        service.engine=SimpleNamespace(status='ready',error=None,device='cpu',precision='test',close=lambda:None,
            infer=lambda frames,masks:[(f,masks) for f in frames])
        service.model_name=model;service.select_tracking_mode('precise')
        service.open_media(path,'input.avi');frame=service.camera.latest();service.pin(frame.id)
        obj_id=service.new_object()['id'];service.masks={obj_id:inputs()[1][1]}
        service._publish(frame,service.masks);service.control('start')
        key=service.recordings.current['id'];result=wait_result(service.recordings,key)
        assert result['state']=='completed' and result['frame_count']==5
        assert result['model']==model and not service.running
    finally:service.close()
