import json
import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from backend.camera import Frame
from backend.scans import ScanStore,MIN_VIEWS


def tracking_fixture():
    rng=np.random.default_rng(19)
    image=rng.integers(20,235,(80,100,3),dtype=np.uint8)
    mask=np.zeros((80,100),bool);mask[10:70,15:85]=True
    frame=Frame(1,time.time(),image)
    return SimpleNamespace(lock=threading.RLock(),objects={1:{'name':'Object 01'}},
                           masks={1:mask},anchor=frame,history=[(frame,{1:mask})],
                           camera=SimpleNamespace(connected=True,source='0'))


def test_capture_binds_frame_mask_and_persists_restart(tmp_path):
    tracking=tracking_fixture();store=ScanStore(tracking,tmp_path)
    session=store.create(1);key=session['id']
    data=store.capture(key,1)
    assert data['count']==1 and data['captures'][0]['frame_id']==1
    folder=store.folder(key)
    image=cv2.imread(str(folder/'images'/'00001.jpg'))
    mask=cv2.imread(str(folder/'masks'/'00001.jpg.png'),0)
    assert mask[0,0]==0 and mask[30,30]==255
    assert image[0,0].max()<10
    assert (folder/'originals'/'00001.jpg').is_file()
    restarted=ScanStore(tracking,tmp_path)
    assert restarted.get(key)['count']==1
    with pytest.raises(ValueError,match='이미 촬영한'):
        restarted.capture(key,1)
    with pytest.raises(ValueError,match='만료'):
        restarted.capture(key,999)
    with pytest.raises(ValueError,match='최소'):
        restarted.reconstruct(key)


def test_removed_capture_is_retained_and_name_not_reused(tmp_path):
    tracking=tracking_fixture();store=ScanStore(tracking,tmp_path)
    key=store.create(1)['id'];store.capture(key,1)
    store.remove_capture(key,'00001.jpg')
    assert (store.folder(key)/'originals'/'00001.jpg').exists()
    frame=Frame(2,time.time(),np.roll(tracking.anchor.image,10,axis=1))
    tracking.history.append((frame,tracking.masks))
    data=store.capture(key,2)
    assert data['captures'][0]['name']=='00002.jpg'


def test_capture_rejects_lost_object_changed_source_and_bad_ids(tmp_path):
    tracking=tracking_fixture();store=ScanStore(tracking,tmp_path)
    key=store.create(1)['id']
    with pytest.raises(ValueError):store.folder('../outside')
    tracking.camera.source='1'
    with pytest.raises(ValueError,match='장치'):store.capture(key,1)
    tracking.camera.source='0';tracking.objects.clear()
    with pytest.raises(ValueError,match='소실'):store.capture(key,1)


def test_restart_marks_active_jobs_interrupted(tmp_path):
    store=ScanStore(tracking_fixture(),tmp_path)
    data=store.create(1);data.update(state='reconstructing')
    store._save(data)
    restarted=ScanStore(None,tmp_path)
    assert restarted.get(data['id'])['state']=='interrupted'


def test_flat_blurry_object_rejected(tmp_path):
    tracking=tracking_fixture();store=ScanStore(tracking,tmp_path);key=store.create(1)['id']
    tracking.anchor.image[:]=100
    with pytest.raises(ValueError,match='흐리거나'):store.capture(key,1)


def test_scan_http_capture_thumbnail_and_failed_start(tmp_path):
    from fastapi.testclient import TestClient
    import backend.app as module
    with TestClient(module.app) as client:
        service=module.service
        fixture=tracking_fixture()
        service.anchor=fixture.anchor;service.masks=fixture.masks
        service.history.extend(fixture.history)
        service.objects={1:{'id':1,'name':'Object 01','color':'#48ddac','points':[],'path':[]}}
        service.camera.connected=True;service.camera.source='0'
        module.scans=ScanStore(service,tmp_path)
        result=client.post('/api/scans',json={'object_id':1})
        assert result.status_code==200,result.text
        key=result.json()['id']
        assert service.running
        result=client.post(f'/api/scans/{key}/capture',json={'frame_id':1})
        assert result.status_code==200,result.text
        thumbnail=result.json()['captures'][0]['thumbnail']
        response=client.get(thumbnail)
        assert response.status_code==200
        assert response.headers['content-type']=='image/jpeg'
        assert client.get(f'/api/scans/{key}/model.ply').status_code==404
        assert client.post(f'/api/scans/{key}/reconstruct').status_code==400
        assert client.get('/api/scans').json()[0]['count']==1
        assert client.delete(f'/api/scans/{key}/captures/00001.jpg').status_code==200
        assert client.get(f'/api/scans/{key}').json()['count']==0
