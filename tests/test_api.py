import csv
import io
import time
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app
from backend.camera import Frame
import backend.app as module


def test_validation_frame_binding_export_and_reset():
    with TestClient(app) as client:
        service=module.service
        frame=Frame(100,time.time(),np.zeros((80,100,3),np.uint8))
        service.camera.frames.append(frame)
        assert client.post('/api/frame/pin',json={'frame_id':99}).status_code==400
        assert client.post('/api/frame/pin',json={'frame_id':100}).status_code==200
        obj_id=client.post('/api/objects').json()['id']
        point={'x':30,'y':20,'label':1}
        endpoint=f'/api/objects/{obj_id}/points'
        assert client.post(endpoint,json={'frame_id':99,'points':[point]}).status_code==400
        assert client.post(endpoint,json={'frame_id':100,'points':[dict(point,x=100)]}).status_code==400
        assert client.post(endpoint,json={'frame_id':100,'points':[dict(point,label=0)]}).status_code==400
        # Inject a known mask to verify geometry/export, independently of GPU availability.
        mask=np.zeros((80,100),bool); mask[10:30,20:60]=True
        service.objects[obj_id]['points']=[dict(point,frame_id=100)]
        service.masks={obj_id:mask}
        service._publish(frame,service.masks)
        data=client.get('/api/export/json').json()
        assert data['frame']['id']==100 and 'jpeg' not in data['frame']
        assert data['objects'][0]['area']==800
        assert data['objects'][0]['center']==[39.5,19.5]
        assert 'mask_png' not in data['objects'][0]
        rows=list(csv.DictReader(io.StringIO(client.get('/api/export/csv').content.decode('utf-8-sig'))))
        assert rows[0]['frame_id']=='100' and rows[0]['width_px']=='40'
        assert client.post('/api/reset').status_code==200
        assert client.get('/api/export/json').status_code==400
        assert client.post('/api/objects').status_code==400


def test_local_origin_guard_and_request_bounds():
    with TestClient(app) as client:
        assert client.post('/api/reset',headers={'Origin':'https://other.example'}).status_code==403
        assert client.post('/api/reset',headers={'Origin':'http://127.0.0.1:8000'}).status_code==200
        assert client.post('/api/camera/start',json={'width':1280,'height':480}).status_code==400
        assert client.post('/api/camera/start',json={'device':-1}).status_code==422
        assert client.post('/api/tracking',json={'action':'start'}).status_code==400


def test_pinning_history_does_not_restore_deleted_masks_or_future_paths():
    with TestClient(app) as client:
        service=module.service
        frame=Frame(10,time.time(),np.zeros((20,20,3),np.uint8))
        service.camera.frames.append(frame)
        service.pin(10)
        obj_id=service.new_object()['id']
        service.masks={obj_id:np.ones((20,20),bool)}
        service._publish(frame,service.masks)
        later=Frame(20,time.time(),frame.image)
        service._publish(later,service.masks)
        service.pin(10)
        assert all(p['frame_id']<=10 for p in service.objects[obj_id]['path'])
        service.delete(obj_id)
        service.pin(20)
        assert obj_id not in service.masks
