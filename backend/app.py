import asyncio
from contextlib import asynccontextmanager
import csv
import io
import json
import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .service import TrackingService, frame_packet
from .scans import ScanStore

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'http://127.0.0.1:8000','http://localhost:8000',
           'http://127.0.0.1:5173','http://localhost:5173'}
PORT = int(os.environ.get('SAM_PORT','8000'))
ALLOWED.update({f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'})
service = None
scans = None


@asynccontextmanager
async def lifespan(app):
    global service,scans
    service = TrackingService()
    scans = ScanStore(service)
    yield
    await asyncio.to_thread(scans.close)
    await asyncio.to_thread(service.close)


app = FastAPI(title='SAM 2.1 Vision Tracking',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=list(ALLOWED),allow_methods=['GET','POST','DELETE'],allow_headers=['Content-Type'])


@app.middleware('http')
async def local_write_guard(request: Request,call_next):
    origin = request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin and origin not in ALLOWED:
        return Response('Origin not allowed',status_code=403)
    return await call_next(request)


async def invoke(func,*args,**kwargs):
    try:
        return await asyncio.to_thread(func,*args,**kwargs)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503,str(exc)) from exc


class CameraStart(BaseModel):
    device: int = Field(0,ge=0,le=10)
    width: Literal[640,1280] = 640
    height: Literal[480,720] = 480
    fps: int = Field(30,ge=1,le=30)
    synthetic: bool = False


class Pin(BaseModel):
    frame_id: int


class Point(BaseModel):
    x: int
    y: int
    label: Literal[0,1]


class Points(BaseModel):
    frame_id: int
    points: list[Point] = Field(max_length=64)


class Control(BaseModel):
    action: Literal['start','pause','stop']


class ScanCreate(BaseModel):
    object_id: int = Field(ge=1)


@app.get('/api/scans')
async def list_scans():
    return await invoke(scans.list)


@app.post('/api/scans')
async def create_scan(body:ScanCreate):
    result=await invoke(scans.create,body.object_id)
    await invoke(service.control,'start')
    return result


@app.get('/api/scans/{scan_id}')
async def get_scan(scan_id:str):
    return await invoke(scans.get,scan_id)


@app.post('/api/scans/{scan_id}/capture')
async def capture_scan(scan_id:str,body:Pin):
    return await invoke(scans.capture,scan_id,body.frame_id)


@app.delete('/api/scans/{scan_id}/captures/{name}')
async def remove_scan_capture(scan_id:str,name:str):
    return await invoke(scans.remove_capture,scan_id,name)


@app.post('/api/scans/{scan_id}/reconstruct')
async def reconstruct_scan(scan_id:str):
    await invoke(service.control,'pause')
    return await invoke(scans.reconstruct,scan_id)


@app.post('/api/scans/{scan_id}/cancel')
async def cancel_scan(scan_id:str):
    return await invoke(scans.cancel,scan_id)


@app.get('/api/scans/{scan_id}/thumb/{name}')
async def scan_thumbnail(scan_id:str,name:str):
    import re
    if not re.fullmatch(r'\d{5}\.jpg',name):
        raise HTTPException(404)
    folder=await invoke(scans.folder,scan_id)
    path=folder/'thumbs'/name
    if not path.is_file(): raise HTTPException(404)
    return FileResponse(path,media_type='image/jpeg')


@app.get('/api/scans/{scan_id}/{filename}')
async def scan_result_file(scan_id:str,filename:Literal['model.ply','points.ply','log.txt']):
    data=await invoke(scans.get,scan_id)
    folder=await invoke(scans.folder,scan_id)
    if filename=='log.txt':
        path=folder/'reconstruction.log'
        if not path.is_file(): raise HTTPException(404)
        return FileResponse(path,media_type='text/plain',filename='reconstruction.log')
    path=folder/filename
    if not data['model_url'] or not path.is_file(): raise HTTPException(404)
    return FileResponse(path,media_type='application/octet-stream',filename=filename,
                        headers={'Cache-Control':'no-cache'})


@app.get('/api/status')
def status():
    return service.status()


@app.get('/api/cameras')
async def cameras():
    if service.camera.connected:
        return [] if service.camera.source=='synthetic' else [{'id':int(service.camera.source),'name':f'Camera {service.camera.source} (connected)'}]
    return await invoke(service.camera.discover)


@app.post('/api/camera/start')
async def camera_start(body:CameraStart):
    if (body.width,body.height) not in [(640,480),(1280,720)]:
        raise HTTPException(400,'지원하지 않는 해상도')
    return await invoke(service.start_camera,**body.model_dump())


@app.post('/api/camera/stop')
async def camera_stop():
    return await invoke(service.stop_camera)


@app.post('/api/model/load')
async def load_model():
    return await invoke(service.load)


@app.post('/api/frame/pin')
async def pin(body:Pin):
    return await invoke(service.pin,body.frame_id)


@app.post('/api/objects')
async def new_object():
    return await invoke(service.new_object)


@app.post('/api/objects/{obj_id}/points')
async def set_points(obj_id:int,body:Points):
    return await invoke(service.set_points,obj_id,body.frame_id,[p.model_dump() for p in body.points])


@app.delete('/api/objects/{obj_id}')
async def delete_object(obj_id:int):
    return await invoke(service.delete,obj_id)


@app.post('/api/tracking')
async def tracking(body:Control):
    return await invoke(service.control,body.action)


@app.post('/api/reset')
async def reset():
    return await invoke(service.reset)


@app.post('/api/paths/clear')
async def clear_paths():
    return await invoke(service.clear_paths)


@app.get('/api/export/{kind}')
async def export(kind:Literal['json','csv']):
    # The immutable published packet binds all values to one source frame.
    packet = service.packet
    if not packet:
        raise HTTPException(400,'내보낼 분석 결과가 없습니다.')
    clean = dict(frame={k:v for k,v in packet['frame'].items() if k!='jpeg'},
                 coordinates='origin: top-left; x: right; y: down; bbox: inclusive; units: px, px^2',
                 objects=[{k:v for k,v in o.items() if k!='mask_png'} for o in packet['objects']])
    if kind=='json':
        return Response(json.dumps(clean,ensure_ascii=False,indent=2),media_type='application/json',
                        headers={'Content-Disposition':'attachment; filename=tracking.json'})
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(['frame_id','timestamp','image_width','image_height','object_id','status',
                     'width_px','height_px','area_px2','area_percent','center_x','center_y',
                     'x_min','y_min','x_max','y_max','points_json'])
    for o in clean['objects']:
        writer.writerow([o['frame_id'],o['timestamp'],clean['frame']['width'],clean['frame']['height'],
                         o['id'],o['status'],o['width'],o['height'],o['area'],o['area_percent'],
                         *(o['center'] or [None,None]),*(o['bbox'] or [None]*4),json.dumps(o['points'])])
    return Response(out.getvalue().encode('utf-8-sig'),media_type='text/csv',
                    headers={'Content-Disposition':'attachment; filename=tracking.csv'})


@app.websocket('/ws')
async def websocket(ws:WebSocket):
    if ws.headers.get('origin') and ws.headers['origin'] not in ALLOWED:
        await ws.close(code=1008)
        return
    await ws.accept()
    last_frame,last_version,last_status = -1,-1,0
    try:
        while True:
            now = asyncio.get_running_loop().time()
            if now-last_status>.5:
                await ws.send_json(dict(type='status',data=service.status()))
                last_status = now
            version,packet = service.version,service.packet
            if version!=last_version:
                await ws.send_json(dict(type='result',data=packet))
                last_version = version
            frame = service.camera.latest()
            if service.camera.connected and frame and frame.id!=last_frame:
                data = await asyncio.to_thread(frame_packet,frame)
                await ws.send_json(dict(type='preview',data=data))
                last_frame = frame.id
            await asyncio.sleep(1/30)
    except (WebSocketDisconnect,RuntimeError,OSError):
        pass


DIST = ROOT/'frontend'/'dist'
if DIST.is_dir():
    app.mount('/assets',StaticFiles(directory=DIST/'assets'),name='assets')

    @app.get('/')
    def index():
        return FileResponse(DIST/'index.html')
