import base64
from types import SimpleNamespace
import threading
import time
import numpy as np
import pytest
from backend.camera import Frame
from backend.depth import point_cloud,DepthPreview


def sample():
    image=np.zeros((60,80,3),np.uint8);image[:]=[10,20,200]
    mask=np.zeros((60,80),bool);mask[10:50,20:60]=True
    depth=np.tile(np.linspace(1,5,80),(60,1)).astype(np.float32)
    return image,mask,depth


def test_depth_points_use_mask_color_and_real_depth_variation():
    image,mask,depth=sample();cloud=point_cloud(image,mask,depth)
    points=np.frombuffer(base64.b64decode(cloud['positions']),'<f4').reshape(-1,3)
    colors=np.frombuffer(base64.b64decode(cloud['colors']),np.uint8).reshape(-1,3)
    assert cloud['count']==38*38 and np.isfinite(points).all()
    assert np.all(colors==[200,20,10]) and np.ptp(points[:,2])>.1
    # Row-major camera top becomes positive Y, near depths become positive Z.
    assert points[0,1]>points[-1,1] and points[37,2]>points[0,2]
    flat=point_cloud(image,mask,np.ones((60,80),np.float32))
    assert flat['depth_range']==0


def test_depth_invalid_or_empty_mask_is_rejected():
    image,mask,depth=sample()
    with pytest.raises(ValueError):point_cloud(image,mask,np.zeros_like(depth))
    with pytest.raises(ValueError):point_cloud(image,np.zeros_like(mask),depth)
    with pytest.raises(ValueError):point_cloud(image,mask,depth[:20])


def test_depth_stale_result_discarded_and_exact_history_used():
    image,mask,depth=sample();entered=threading.Event();release=threading.Event()
    class Engine:
        def infer(self,source):
            assert source is image
            entered.set();release.wait(2);return depth
    frame=Frame(8,time.time(),image)
    service=SimpleNamespace(lock=threading.RLock(),engine_lock=threading.RLock(),revision=1,version=1,
        packet={'objects':[{'id':1,'area':100}]},history=[(frame,{1:mask})])
    worker=DepthPreview(service,Engine())
    try:
        worker.request(1);assert entered.wait(2)
        # Changing input invalidates in-flight inference, even if object ID is reused.
        with service.lock:service.revision+=1
        release.set();time.sleep(.15)
        assert worker.result is None
        assert worker.request(1)['result'] is None
        deadline=time.monotonic()+2
        while worker.result is None and time.monotonic()<deadline:
            worker.request(1);time.sleep(.02)
        assert worker.result['frame_id']==8 and worker.result['object_id']==1
        assert worker.request(1,since=1)['result'] is None
        # A correction to a static frame must recompute even with the same frame ID.
        with service.lock:service.version=2;service.history=[(frame,{1:np.ones_like(mask)})]
        deadline=time.monotonic()+2
        while worker.result['version']!=2 and time.monotonic()<deadline:
            worker.request(1);time.sleep(.02)
        assert worker.result['version']==2 and worker.request(1,since=1)['result'] is not None
        service.packet={'objects':[]}
        assert worker.request(1)['frame_id'] is None
    finally:release.set();worker.close()
