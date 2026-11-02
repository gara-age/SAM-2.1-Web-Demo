"""End-to-end test using real SAM, camera worker, HTTP and WebSocket.

The synthetic source is explicitly labeled and only replaces the USB input.
Inference and geometry use the production implementation without mocks.
"""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.app import app
import backend.app as module

report={}
with TestClient(app) as client:
    def post(path,body=None):
        response=client.post('/api'+path,json=body)
        assert response.status_code==200,(path,response.status_code,response.text)
        return response.json()

    post('/camera/start',{'synthetic':True,'fps':15})
    with client.websocket_connect('/ws') as ws:
        for _ in range(20):
            event=ws.receive_json()
            if event['type']=='preview':
                frame=event['data'];break
        else:
            raise AssertionError('No camera preview over WebSocket')
        post('/frame/pin',{'frame_id':frame['id']})
        service=module.service
        image=service.anchor.image
        # Derive the moving orange rectangle's location from the known input.
        import numpy as np
        ys,xs=np.where((image[:,:,2]>230)&(image[:,:,1]>150)&(image[:,:,1]<180))
        center=[int(xs.mean()),int(ys.mean())]
        prompts=[center,[160,160],[499,220]]
        ids=[]
        for x,y in prompts:
            obj_id=post('/objects')['id'];ids.append(obj_id)
            packet=post(f'/objects/{obj_id}/points',{'frame_id':frame['id'],
                         'points':[{'x':x,'y':y,'label':1}]})
            assert packet['objects'][-1]['area']>1000
        # Multiple positive/negative prompts, point removal and reset/reselection.
        packet=post(f'/objects/{ids[0]}/points',{'frame_id':frame['id'],'points':[
            {'x':center[0],'y':center[1],'label':1},
            {'x':center[0]+10,'y':center[1],'label':1},
            {'x':10,'y':10,'label':0}]})
        assert len(packet['objects'][0]['points'])==3
        packet=post(f'/objects/{ids[0]}/points',{'frame_id':frame['id'],'points':[
            {'x':center[0],'y':center[1],'label':1}]})
        assert len(packet['objects'][0]['points'])==1
        packet=post(f'/objects/{ids[2]}/points',{'frame_id':frame['id'],'points':[]})
        assert packet['objects'][2]['status']=='Pending'
        packet=post(f'/objects/{ids[2]}/points',{'frame_id':frame['id'],'points':[{'x':499,'y':220,'label':1}]})
        post('/tracking',{'action':'start'})
        deadline=time.monotonic()+30
        tracked=None
        while time.monotonic()<deadline:
            event=ws.receive_json()
            if event['type']=='result' and event['data'] and event['data']['frame']['id']>frame['id']:
                tracked=event['data']
                if all(o['status']=='Active' for o in tracked['objects']):
                    break
        assert tracked and all(o['status']=='Active' for o in tracked['objects']),tracked
        paused=post('/tracking',{'action':'pause'})
        data=client.get('/api/export/json').json()
        assert len(data['objects'])==3
        assert all(o['frame_id']==data['frame']['id'] for o in data['objects'])
        assert all(o['path'] for o in data['objects'])
        assert len(client.get('/api/export/csv').content)>100
        # Correction on an analyzed frame then resume.
        pinned=post('/frame/pin',{'frame_id':data['frame']['id']})
        post('/tracking',{'action':'start'})
        post('/tracking',{'action':'pause'})
        assert client.delete(f'/api/objects/{ids[1]}').status_code==200
        # Re-pinning a history entry must not resurrect deleted object masks.
        post('/frame/pin',{'frame_id':pinned['frame']['id']})
        assert ids[1] not in service.masks
        assert client.get('/api/export/json').json()['objects']
        post('/paths/clear')
        assert all(not o['path'] for o in client.get('/api/export/json').json()['objects'])
        report=dict(source='synthetic',model='SAM 2.1 Tiny',device=service.engine.device,
                    tracked_objects=3,frame_id=data['frame']['id'],
                    fps=paused['inference_fps'],chunk_ms=paused['inference_ms'],
                    checks=['WebSocket preview/results','3-object segmentation and tracking',
                            'positive/negative prompts','point deletion/reset',
                            'synchronized measurements','JSON/CSV export','pause/resume',
                            'object deletion/history filtering','path reset'])
        post('/camera/stop')
        post('/reset')
        assert client.get('/api/export/json').status_code==400
target=Path(__file__).resolve().parents[1]/'logs'/'workflow-smoke.json'
target.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
