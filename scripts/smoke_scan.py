"""Real COLMAP reconstruction from rendered views of a textured 3D cube.

Only the input scene/masks are synthetic; no reconstruction outputs are mocked.
The fixture is labeled in the persisted session and is not a user's scan.
"""
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from backend.scans import ScanStore


def render_views(count=16):
    rng=np.random.default_rng(42)
    vertices=np.array([[-.8,-.8,-.8],[.8,-.8,-.8],[.8,.8,-.8],[-.8,.8,-.8],
                       [-.8,-.8,.8],[.8,-.8,.8],[.8,.8,.8],[-.8,.8,.8]])
    faces=[([0,3,2,1],[0,0,-1]),([4,5,6,7],[0,0,1]),
           ([0,4,7,3],[-1,0,0]),([1,2,6,5],[1,0,0]),
           ([0,1,5,4],[0,-1,0]),([3,7,6,2],[0,1,0])]
    textures=[]
    for k in range(6):
        tex=np.full((512,512,3),200,np.uint8)
        for i in range(300):
            x,y=rng.integers(8,504,2);r=int(rng.integers(3,12));color=tuple(map(int,rng.integers(15,240,3)))
            cv2.circle(tex,(int(x),int(y)),r,color,-1,cv2.LINE_AA)
        for y in range(30,510,60):
            for x in range(10,490,80):
                cv2.putText(tex,f'{k}{x//80}{y//60}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
        textures.append(tex)
    for index,yaw in enumerate(np.linspace(-70,70,count)):
        angle=np.deg2rad(yaw)
        center=np.array([np.sin(angle)*4.2,1.1+0.15*np.sin(angle),np.cos(angle)*4.2])
        forward=-center/np.linalg.norm(center)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        rot=np.stack([right,down,forward])
        camera=(vertices-center)@rot.T
        uv=camera[:,:2]/camera[:,2:]*650+np.array([320,240])
        image=np.zeros((480,640,3),np.uint8);mask=np.zeros((480,640),np.uint8)
        visible=[(k,idx) for k,(idx,n) in enumerate(faces) if np.dot(np.array(n),center-vertices[idx].mean(0))>0]
        visible.sort(key=lambda item:camera[item[1],2].mean(),reverse=True)
        for k,idx in visible:
            polygon=uv[idx].astype(np.float32)
            transform=cv2.getPerspectiveTransform(np.array([[0,0],[511,0],[511,511],[0,511]],np.float32),polygon)
            warped=cv2.warpPerspective(textures[k],transform,(640,480))
            face_mask=np.zeros_like(mask);cv2.fillConvexPoly(face_mask,np.round(polygon).astype(np.int32),255)
            image[face_mask>0]=warped[face_mask>0];mask[face_mask>0]=255
        yield image,mask>0,index


def main():
    store=ScanStore(None)
    scan_id=uuid.uuid4().hex
    data=dict(id=scan_id,object_id=1,object_name='검증용 합성 3D 큐브',source='synthetic-3d-fixture',
              width=640,height=480,created_at=time.time(),state='capturing',stage='검증 데이터 준비',
              error=None,captures=[],registered_views=0,vertices=0,faces=0,model_url=None,
              units='relative',version=0)
    store._save(data)
    for image,mask,index in render_views():
        with store.lock:
            store._add_capture(scan_id,image,mask,index,time.time())
    print('SCAN_ID='+scan_id,flush=True)
    store.reconstruct(scan_id)
    previous=None;start=time.monotonic()
    while store.worker.is_alive():
        status=store.get(scan_id)
        if status['stage']!=previous:
            print(status['stage'],flush=True);previous=status['stage']
        if time.monotonic()-start>1200:
            store.cancel(scan_id)
            raise TimeoutError('3D smoke exceeded 20 minutes')
        time.sleep(1)
    result=store.get(scan_id)
    assert result['state']=='complete',result
    assert result['faces']>100 and result['vertices']>100,result
    assert result['registered_views']>=12,result
    from plyfile import PlyData
    mesh=PlyData.read(str(store.folder(scan_id)/'model.ply'))
    xyz=np.column_stack([mesh['vertex'][k] for k in ('x','y','z')])
    assert np.isfinite(xyz).all()
    assert np.linalg.matrix_rank(xyz-xyz.mean(0))==3,'Not a 3D surface'
    report={k:v for k,v in result.items() if k!='captures'}
    report['seconds']=round(time.monotonic()-start,1)
    report['input']='16 rendered views of a textured cube; ground-truth masks; actual COLMAP pipeline'
    target=Path(__file__).resolve().parents[1]/'logs'/'scan-smoke.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
