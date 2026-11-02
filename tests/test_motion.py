import time
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from backend.camera import Frame
from backend.motion import MotionTracker
from backend.service import TrackingService


def scene(index):
    rng=np.random.default_rng(32)
    image=np.zeros((120,200,3),np.uint8)
    image[30:90,30+index*2:90+index*2]=rng.integers(30,240,(60,60,3),dtype=np.uint8)
    mask=np.zeros((120,200),bool);mask[30:90,30+index*2:90+index*2]=True
    return Frame(index+1,time.time(),image,index,index/10),mask


def test_measured_motion_and_delayed_correction_are_frame_aligned():
    frame,mask=scene(0);tracker=MotionTracker(frame,{1:mask})
    base=tracker.snapshot()
    for index in range(1,6):
        frame,expected=scene(index);actual=tracker.advance(frame)[1]
        assert np.logical_and(actual,expected).sum()/np.logical_or(actual,expected).sum()>.9
    assert tracker.correct({1:mask},base)==1
    assert np.logical_and(tracker.masks[1],expected).sum()/np.logical_or(tracker.masks[1],expected).sum()>.9


def test_no_features_means_lost_instead_of_static_fake_mask():
    frame=Frame(1,time.time(),np.zeros((120,200,3),np.uint8))
    tracker=MotionTracker(frame,{1:np.ones((120,200),bool)})
    assert not tracker.advance(Frame(2,time.time(),frame.image))[1].any()


def test_both_models_play_while_slow_sam_runs_and_pause_is_responsive(tmp_path):
    for model in ('sam2.1','sam3.1'):
        service=TrackingService()
        def infer(frames,masks):
            time.sleep(.6)
            # Test only concurrency and versioning, not SAM inference quality.
            return [(frame,{i:m.copy() for i,m in masks.items()}) for frame in frames]
        engine=SimpleNamespace(infer=infer,status='ready',error=None,device='cpu',precision='test',close=lambda:None)
        path=tmp_path/(model+'.avi')
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(200,120))
        assert writer.isOpened()
        for index in range(30):writer.write(scene(index)[0].image)
        writer.release()
        try:
            service.engine=engine;service.model_name=model
            service.open_media(path,path.name)
            frame=service.camera.latest();service.pin(frame.id)
            obj_id=service.new_object()['id'];service.masks={obj_id:scene(0)[1]}
            service._publish(frame,service.masks)
            service.control('start')
            deadline=time.monotonic()+2
            while not service.ai_busy and time.monotonic()<deadline:time.sleep(.01)
            initial=service.version;time.sleep(.2)
            assert service.version>initial and service.camera.media_index>=2
            start=time.monotonic();service.control('pause')
            assert time.monotonic()-start<.3
            version=service.version;time.sleep(.7)
            assert service.version==version and not service.running
            assert service.packet['tracking_method']=='optical-flow+SAM'
        finally:service.close()


@pytest.mark.parametrize('model',['sam2.1','sam3.1'])
def test_live_input_keeps_results_updating_during_slow_ai(model):
    service=TrackingService()
    def infer(frames,masks):
        time.sleep(.8)
        return [(frame,masks) for frame in frames]
    try:
        service.engine=SimpleNamespace(infer=infer,status='ready',error=None,device='cpu',precision='test',close=lambda:None)
        service.model_name=model
        service.start_camera(device=0,width=640,height=480,fps=15,synthetic=True)
        deadline=time.monotonic()+1
        while service.camera.latest() is None and time.monotonic()<deadline:time.sleep(.01)
        frame=service.camera.latest();service.pin(frame.id)
        obj_id=service.new_object()['id']
        # The fixed blue circle avoids the clock-dependent rectangle/circle
        # occlusion in the synthetic camera. Moving geometry is tested above.
        mask=np.all(frame.image==[200,90,60],axis=2)
        service.masks={obj_id:mask};service._publish(frame,service.masks)
        service.control('start')
        deadline=time.monotonic()+1.5
        while service.packet['frame']['id']<frame.id+4 and time.monotonic()<deadline:time.sleep(.02)
        assert service.packet['frame']['id']>=frame.id+4
        assert service.packet['objects'][0]['area']>0
        assert service.packet['tracking_method']=='optical-flow+SAM'
        service.control('pause');version=service.version
        time.sleep(.15)
        assert service.version==version
        assert service.camera.latest().id>service.packet['frame']['id']
    finally:service.close()
