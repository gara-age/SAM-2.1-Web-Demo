import time
import cv2
import numpy as np
from backend.camera import Frame
from backend.service import TrackingService,frame_packet
from backend.geometry import measure


def test_frame_jpeg_is_encoded_once_for_multiple_consumers(monkeypatch):
    frame=Frame(1,time.time(),np.zeros((80,120,3),np.uint8))
    original=cv2.imencode;calls=[]
    def encode(*args,**kwargs):calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(cv2,'imencode',encode)
    assert frame_packet(frame)==frame_packet(frame)
    assert calls==['.jpg']


def test_compressed_display_contours_keep_exact_geometry_and_full_export(tmp_path):
    service=TrackingService(recordings_root=tmp_path)
    service.shutdown_event.set();service.worker.join();service.motion_worker.join()
    try:
        frame=Frame(1,time.time(),np.zeros((120,160,3),np.uint8))
        mask=np.zeros((120,160),bool);mask[20:100,30:130]=True;mask[40:70,60:90]=False
        service.objects={1:dict(id=1,name='test',color='#48ddac',points=[],path=[])}
        service.running=True;service.masks={1:mask};service.anchor=frame
        with service.lock:service._publish(frame,service.masks)
        display=service.packet['objects'][0];full=service.export_packet()['objects'][0]
        assert display['contours_simplified'] and not full['contours_simplified']
        assert sum(len(c['points']) for c in display['contours'])<sum(len(c['points']) for c in full['contours'])
        for key in ('area','center','bbox','width','height'):assert display[key]==full[key]
        assert full['contours']==measure(mask)['contours']
        # Rebuilding exact exports must never mutate the streamed packet.
        assert service.packet['objects'][0]['contours_simplified']
    finally:service.close()


def test_websocket_can_disable_duplicate_preview_and_enable_it_again():
    from fastapi.testclient import TestClient
    from backend.app import app
    import backend.app as module
    with TestClient(app) as client:
        module.service.start_camera(device=0,width=640,height=480,fps=10,synthetic=True)
        with client.websocket_connect('/ws') as socket:
            socket.send_json({'type':'preview','enabled':False})
            while True:
                message=socket.receive_json()
                if message['type']=='status' and not message['data']['stream_preview']:break
            # After acknowledgement, no raw preview should be sent while disabled.
            message=socket.receive_json();assert message['type']=='status'
            socket.send_json({'type':'preview','enabled':True})
            while True:
                message=socket.receive_json()
                if message['type']=='preview':break
            assert message['data']['width']==640
