"""Local, persistent, object-masked photogrammetry sessions.

All captures and reconstruction intermediates stay under data/scans on D:.
COLMAP is invoked with argument arrays, never user-provided commands.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIN_VIEWS = 12
MAX_VIEWS = 120


class ScanCancelled(Exception):
    pass


class ScanStore:
    def __init__(self, tracking, root=None, executable=None):
        self.tracking = tracking
        self.root = Path(root or ROOT/'data'/'scans')
        self.root.mkdir(parents=True,exist_ok=True)
        self.executable = Path(executable or ROOT/'vendor'/'colmap'/'bin'/'colmap.exe')
        self.lock = threading.RLock()
        self.worker = None
        self.cancel_event = threading.Event()
        self.current_job = None
        # A killed server must never leave a session permanently "running".
        for path in self.root.glob('*/session.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if data['state']=='reconstructing':
                    data.update(state='interrupted',stage='서버 재시작으로 복원 중단',error='다시 복원할 수 있습니다.')
                    self._save(data)
            except (ValueError,KeyError,OSError):
                continue

    def folder(self,scan_id):
        if not re.fullmatch(r'[0-9a-f]{32}',scan_id):
            raise ValueError('올바르지 않은 스캔 ID입니다.')
        return self.root/scan_id

    def _load(self,scan_id):
        path = self.folder(scan_id)/'session.json'
        if not path.is_file():
            raise ValueError('스캔 세션을 찾을 수 없습니다.')
        return json.loads(path.read_text(encoding='utf-8'))

    def _save(self,data):
        folder = self.folder(data['id'])
        folder.mkdir(exist_ok=True)
        temp = folder/'session.json.tmp'
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(temp,folder/'session.json')

    def public(self,data):
        result = dict(data)
        result['minimum_views'] = MIN_VIEWS
        result['maximum_views'] = MAX_VIEWS
        result['count'] = len(data['captures'])
        result['engine_available'] = self.executable.is_file()
        return result

    def get(self,scan_id):
        with self.lock:
            return self.public(self._load(scan_id))

    def list(self):
        with self.lock:
            sessions=[]
            for path in self.root.glob('*/session.json'):
                try:
                    sessions.append(self.public(self._load(path.parent.name)))
                except (OSError,ValueError,KeyError):
                    continue
            return sorted(sessions,key=lambda s:s['created_at'],reverse=True)

    def create(self,obj_id):
        with self.tracking.lock:
            if getattr(self.tracking.camera,'input_kind',None) in ('image','example'):
                raise ValueError('정지 이미지 한 장으로는 3D 스캔을 만들 수 없습니다. 카메라나 여러 시점이 담긴 동영상을 사용하세요.')
            obj = self.tracking.objects.get(obj_id)
            mask = self.tracking.masks.get(obj_id)
            frame = self.tracking.anchor
            if obj is None or mask is None or not mask.any() or frame is None:
                raise ValueError('먼저 추적할 객체를 선택하고 분할하세요.')
            if not self.tracking.camera.connected:
                raise ValueError('카메라를 먼저 시작하세요.')
            data = dict(id=uuid.uuid4().hex,object_id=obj_id,object_name=obj['name'],
                        tracking_model=getattr(self.tracking,'model_name','sam2.1'),
                        source=self.tracking.camera.source,width=frame.image.shape[1],
                        height=frame.image.shape[0],created_at=time.time(),state='capturing',
                        stage='서로 다른 각도의 분석 프레임을 촬영하세요.',error=None,
                        captures=[],registered_views=0,vertices=0,faces=0,model_url=None,
                        units='relative',version=0)
        with self.lock:
            self._save(data)
        return self.public(data)

    def capture(self,scan_id,frame_id):
        # Hold only the tracking lock while copying; never during disk/CPU work.
        with self.tracking.lock:
            record=next((r for r in reversed(self.tracking.history) if r[0].id==frame_id),None)
            if record is None:
                raise ValueError('분석 프레임이 만료되었습니다. 최신 분석 프레임을 촬영하세요.')
            with self.lock:
                data=self._load(scan_id)
            mask=record[1].get(data['object_id'])
            if data['object_id'] not in self.tracking.objects or mask is None or not mask.any():
                raise ValueError('선택 객체가 소실되었거나 삭제되었습니다. 객체 마스크를 보정하세요.')
            if self.tracking.camera.source!=data['source']:
                raise ValueError('촬영 중 입력 장치를 변경했습니다. 새 스캔을 시작하세요.')
            if getattr(self.tracking,'model_name','sam2.1')!=data.get('tracking_model','sam2.1'):
                raise ValueError('선택·추적 모델이 변경되었습니다. 새 스캔을 시작하세요.')
            image=record[0].image.copy()
            mask=mask.copy()
            timestamp=record[0].timestamp
        with self.lock:
            return self._add_capture(data['id'],image,mask,frame_id,timestamp)

    def _add_capture(self,scan_id,image,mask,frame_id,timestamp):
        data=self._load(scan_id)
        if data['state']=='reconstructing':
            raise ValueError('복원 중에는 촬영 이미지를 변경할 수 없습니다.')
        if len(data['captures'])>=MAX_VIEWS:
            raise ValueError(f'스캔당 최대 {MAX_VIEWS}장까지 촬영할 수 있습니다.')
        if image.shape[:2]!=(data['height'],data['width']):
            raise ValueError('영상 해상도가 달라졌습니다. 새 스캔을 시작하세요.')
        if any(c['frame_id']==frame_id for c in data['captures']):
            raise ValueError('이미 촬영한 프레임입니다. 추적을 재개하고 다른 각도에서 촬영하세요.')
        mask=mask.astype(np.uint8)*255
        area=int(np.count_nonzero(mask))
        if area<500:
            raise ValueError('객체가 너무 작습니다. 대상에 가까이 촬영하세요.')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        inner=cv2.erode(mask,np.ones((5,5),np.uint8))>0
        sharpness=float(cv2.Laplacian(gray,cv2.CV_64F)[inner].var()) if inner.any() else 0
        if sharpness<12:
            raise ValueError('객체 영상이 흐리거나 표면 특징이 부족합니다. 초점과 조명을 확인하세요.')
        folder=self.folder(scan_id)
        if data['captures']:
            previous=cv2.imread(str(folder/'originals'/data['captures'][-1]['name']))
            delta=float(np.abs(cv2.resize(gray,(160,120)).astype(float)-
                        cv2.resize(cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),(160,120))).mean())
            if delta<1.2:
                raise ValueError('직전 촬영과 거의 같은 영상입니다. 카메라 위치를 조금 이동하세요.')
        for name in ['originals','images','masks','thumbs']:
            (folder/name).mkdir(exist_ok=True)
        index=data.get('next_capture_index',max([c['index'] for c in data['captures']],default=0)+1)
        name=f'{index:05d}.jpg'
        masked=image.copy();masked[mask==0]=0
        thumb=cv2.resize(masked,(160,round(160*data['height']/data['width'])))
        for path,array in [(folder/'originals'/name,image),(folder/'images'/name,masked),
                           (folder/'masks'/(name+'.png'),mask),(folder/'thumbs'/name,thumb)]:
            if not cv2.imwrite(str(path),array):
                raise RuntimeError('D드라이브에 촬영 이미지를 저장할 수 없습니다.')
        data['captures'].append(dict(index=int(name.split('.')[0]),name=name,frame_id=frame_id,
                                     timestamp=timestamp,sharpness=round(sharpness,1),area=area,
                                     thumbnail=f'/api/scans/{scan_id}/thumb/{name}'))
        data['next_capture_index']=index+1
        data.update(state='capturing',error=None,stage='다른 각도에서 촬영을 이어가세요.')
        self._save(data)
        return self.public(data)

    def remove_capture(self,scan_id,name):
        with self.lock:
            data=self._load(scan_id)
            if data['state']=='reconstructing':
                raise ValueError('복원 중에는 촬영 이미지를 변경할 수 없습니다.')
            if not any(c['name']==name for c in data['captures']):
                raise ValueError('촬영 이미지를 찾을 수 없습니다.')
            data['captures']=[c for c in data['captures'] if c['name']!=name]
            # Keep originals recoverable locally; next build uses a manifest list.
            data.update(state='capturing',stage='촬영 목록에서 제외했습니다.',error=None)
            self._save(data)
            return self.public(data)

    def reconstruct(self,scan_id):
        with self.lock:
            data=self._load(scan_id)
            if self.worker and self.worker.is_alive():
                raise ValueError('다른 3D 복원 작업이 진행 중입니다.')
            if len(data['captures'])<MIN_VIEWS:
                raise ValueError(f'서로 다른 각도의 사진을 최소 {MIN_VIEWS}장 촬영하세요. 권장: 30~60장.')
            if not self.executable.is_file():
                raise ValueError('COLMAP이 없습니다. setup-scan.ps1을 실행하세요.')
            data.update(state='reconstructing',stage='복원 준비',error=None,progress=0)
            self._save(data)
            self.cancel_event.clear()
            self.current_job=scan_id
            self.worker=threading.Thread(target=self._reconstruct,args=(scan_id,),daemon=True)
            self.worker.start()
            return self.public(data)

    def cancel(self,scan_id):
        with self.lock:
            data=self._load(scan_id)
            if self.current_job==scan_id and data['state']=='reconstructing':
                self.cancel_event.set()
                return dict(self.public(data),stage='중지 요청을 처리하고 있습니다.')
            return self.public(data)

    def _update(self,scan_id,**changes):
        with self.lock:
            data=self._load(scan_id);data.update(changes);self._save(data)

    def _command(self,folder,stage,**options):
        if self.cancel_event.is_set():
            raise ScanCancelled()
        command=[str(self.executable),stage]
        for key,value in options.items():
            command += ['--'+key,str(value)]
        env=dict(os.environ)
        env['PATH']=str(self.executable.parent)+os.pathsep+env.get('PATH','')
        env['QT_PLUGIN_PATH']=str(self.executable.parent.parent/'plugins')
        with (folder/'reconstruction.log').open('a',encoding='utf-8') as log:
            log.write('\n'+' '.join(command)+'\n');log.flush()
            process=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=log,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            start=time.monotonic()
            try:
                while process.poll() is None:
                    if self.cancel_event.wait(.2):
                        raise ScanCancelled()
                    if time.monotonic()-start>1800:
                        raise RuntimeError(f'{stage} 단계가 제한 시간 30분을 초과했습니다.')
                if process.returncode:
                    raise RuntimeError(f'COLMAP {stage} 실행 실패. 로그에서 상세 원인을 확인하세요.')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill();process.wait()

    def _reconstruct(self,scan_id):
        try:
            self._pipeline(scan_id)
        except ScanCancelled:
            self._update(scan_id,state='cancelled',stage='복원 중지됨',error=None)
        except Exception as exc:
            self._update(scan_id,state='failed',stage='복원 실패',error=str(exc))
        finally:
            self.current_job=None

    def _pipeline(self,scan_id):
        import pycolmap
        from plyfile import PlyData
        data=self._load(scan_id);folder=self.folder(scan_id)
        # New run directory avoids mixing stale databases/depths with edited captures.
        run=folder/('run-'+uuid.uuid4().hex[:8]);run.mkdir()
        names=run/'images.txt'
        names.write_text('\n'.join(c['name'] for c in data['captures'])+'\n',encoding='utf-8')
        db=run/'database.db';sparse=run/'sparse';sparse.mkdir()
        dense=run/'dense'
        def stage(label,progress,command,**options):
            self._update(scan_id,stage=label,progress=progress)
            self._command(folder,command,**options)
        stage('1/7 · 객체 표면 특징 추출',5,'feature_extractor',database_path=db,
              image_path=folder/'images',image_list_path=names,
              **{'ImageReader.mask_path':folder/'masks','ImageReader.single_camera':1,
                 'ImageReader.camera_model':'SIMPLE_RADIAL','FeatureExtraction.use_gpu':0,
                 'FeatureExtraction.num_threads':4,'SiftExtraction.max_num_features':6000})
        stage('2/7 · 촬영 이미지 대응점 연결',15,'exhaustive_matcher',database_path=db,
              **{'FeatureMatching.use_gpu':0,'FeatureMatching.num_threads':4,
                 'FeatureMatching.max_num_matches':6000})
        stage('3/7 · 카메라 위치와 3D 구조 복원',25,'mapper',database_path=db,
              image_path=folder/'images',output_path=sparse,
              **{'Mapper.num_threads':4,'Mapper.min_model_size':6,
                 'Mapper.init_min_tri_angle':8,'Mapper.init_min_num_inliers':50})
        models=[]
        for path in sparse.iterdir():
            if path.is_dir() and (path/'cameras.bin').exists():
                rec=pycolmap.Reconstruction(str(path));models.append((rec.num_reg_images(),path,rec))
        if not models:
            raise ValueError('3D 시점을 연결하지 못했습니다. 대상 표면의 무늬·초점·시점 이동·사진 겹침을 확인하세요.')
        registered,model,rec=max(models,key=lambda item:item[0])
        if registered<max(3,len(data['captures'])//2):
            raise ValueError(f'{len(data["captures"])}장 중 {registered}장만 연결됐습니다. 더 촘촘한 각도에서 다시 촬영하세요.')
        self._update(scan_id,registered_views=registered)
        stage('4/7 · 렌즈 왜곡 보정',40,'image_undistorter',image_path=folder/'images',
              input_path=model,output_path=dense,output_type='COLMAP',max_image_size=960)
        stage('5/7 · 여러 시점의 깊이 계산',50,'patch_match_stereo',workspace_path=dense,
              **{'PatchMatchStereo.max_image_size':800,'PatchMatchStereo.cache_size':1,
                 'PatchMatchStereo.num_iterations':3,'PatchMatchStereo.window_step':2,
                 'PatchMatchStereo.num_samples':10,'PatchMatchStereo.geom_consistency':1})
        fused=run/'fused.ply'
        stage('6/7 · 표면 점군 결합',75,'stereo_fusion',workspace_path=dense,
              input_type='geometric',output_path=fused,
              **{'StereoFusion.cache_size':1,'StereoFusion.num_threads':4})
        self._update(scan_id,stage='6/7 · 선택 객체 밖의 점 제거',progress=83)
        filtered=run/'object.ply'
        filter_cloud(fused,filtered,rec,folder/'masks',self.cancel_event)
        points=PlyData.read(str(filtered))['vertex'].count
        if points<100:
            raise ValueError('객체 표면의 복원 점이 부족합니다. 더 가까이, 다양한 각도에서 선명하게 촬영하세요.')
        mesh=run/'model.ply'
        stage('7/7 · 3D 표면 메시 생성',90,'poisson_mesher',input_path=filtered,output_path=mesh,
              **{'PoissonMeshing.depth':8,'PoissonMeshing.trim':7,'PoissonMeshing.num_threads':4})
        if self.cancel_event.is_set(): raise ScanCancelled()
        mesh_data=PlyData.read(str(mesh))
        faces=mesh_data['face'].count if 'face' in mesh_data else 0
        if faces<10:
            raise ValueError('표면 메시를 만들지 못했습니다. 촬영 범위와 대상의 표면 특징을 확인하세요.')
        # Restrict extrapolated Poisson surfaces using the same object silhouettes.
        cleaned=run/'model-clean.ply'
        clean_mesh(mesh,cleaned,rec,folder/'masks',self.cancel_event)
        final=PlyData.read(str(cleaned))
        if final['face'].count<10:
            raise ValueError('마스크 검증 후 유효한 표면이 부족합니다. 다른 각도에서 촬영을 추가하세요.')
        import shutil
        from .scan_geometry import camera_alignment,rotate_ply,reconstruction_diagnostics
        alignment=camera_alignment(rec)
        diagnostics=reconstruction_diagnostics(rec,filtered,folder/'masks',len(data['captures']))
        rotate_ply(cleaned,run/'model-upright.ply',alignment['rotation'])
        rotate_ply(filtered,run/'points-upright.ply',alignment['rotation'])
        shutil.copyfile(run/'model-upright.ply',folder/'model.ply.tmp')
        shutil.copyfile(run/'points-upright.ply',folder/'points.ply.tmp')
        os.replace(folder/'model.ply.tmp',folder/'model.ply')
        os.replace(folder/'points.ply.tmp',folder/'points.ply')
        self._update(scan_id,state='complete',stage='복원 처리 완료 · 형상과 누락을 확인하세요.',progress=100,
                     vertices=final['vertex'].count,faces=final['face'].count,points=points,
                     alignment=alignment,diagnostics=diagnostics,reconstruction_run=run.name,
                     model_url=f'/api/scans/{scan_id}/model.ply',version=data['version']+1,
                     completed_at=time.time(),error=None)

    def close(self):
        self.cancel_event.set()
        if self.worker:
            self.worker.join(timeout=10)


def silhouette_support(xyz,rec,mask_folder,cancel_event=None):
    votes=np.zeros(len(xyz),np.uint16);seen=np.zeros(len(xyz),np.uint16)
    for image in rec.images.values():
        if cancel_event and cancel_event.is_set(): raise ScanCancelled()
        mask=cv2.imread(str(mask_folder/(image.name+'.png')),cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        camera=rec.cameras[image.camera_id]
        transformed=image.cam_from_world()*xyz
        uv=camera.img_from_cam(transformed)
        valid=np.isfinite(uv).all(axis=1)&(transformed[:,2]>0)
        uv=np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)
        pixels=np.floor(uv).astype(np.int64)
        valid &= (pixels[:,0]>=0)&(pixels[:,0]<mask.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<mask.shape[0])
        indices=np.flatnonzero(valid);seen[indices]+=1
        inside=mask[pixels[indices,1],pixels[indices,0]]>0
        votes[indices[inside]]+=1
    return (votes>=2)&(votes>=seen*.7)


def filter_cloud(source,target,rec,mask_folder,cancel_event=None):
    from plyfile import PlyData,PlyElement
    ply=PlyData.read(str(source));vertices=ply['vertex'].data
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')]).astype(np.float64)
    keep=silhouette_support(xyz,rec,mask_folder,cancel_event)
    PlyData([PlyElement.describe(vertices[keep],'vertex')],text=False).write(str(target))


def clean_mesh(source,target,rec,mask_folder,cancel_event=None):
    from plyfile import PlyData,PlyElement
    ply=PlyData.read(str(source));vertices=ply['vertex'].data;faces=ply['face'].data
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')]).astype(np.float64)
    keep=silhouette_support(xyz,rec,mask_folder,cancel_event)
    indices=np.array(list(faces['vertex_indices']),dtype=np.int64)
    good=np.all(keep[indices],axis=1)
    indices=indices[good]
    used=np.unique(indices)
    remap=np.full(len(vertices),-1,np.int64);remap[used]=np.arange(len(used))
    cleaned=np.empty(len(indices),dtype=[('vertex_indices','i4',(3,))])
    cleaned['vertex_indices']=remap[indices]
    PlyData([PlyElement.describe(vertices[used],'vertex'),
             PlyElement.describe(cleaned,'face')],text=False).write(str(target))
