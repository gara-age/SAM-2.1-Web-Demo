"""Orient existing PLY artifacts using the matching COLMAP run; keep backups."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pycolmap
from plyfile import PlyData
from backend.scan_geometry import camera_alignment,rotate_ply,reconstruction_diagnostics


def xyz_sample(path):
    vertices=PlyData.read(str(path))['vertex']
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')])
    return len(xyz),np.concatenate([xyz[:32],xyz[-32:]])


def align(folder):
    metadata=folder/'session.json';original_text=metadata.read_text(encoding='utf-8')
    data=json.loads(original_text)
    if data['state']=='reconstructing':return 'busy'
    if not data.get('model_url') or not (folder/'points.ply').exists():return 'no_model'
    count,sample=xyz_sample(folder/'points.ply')
    matches=[]
    for path in folder.glob('run-*/object.ply'):
        n,other=xyz_sample(path)
        if n==count and np.allclose(other,sample,atol=1e-6):matches.append(path.parent)
        elif n==count and data.get('alignment',{}).get('method')=='camera_up_v1':
            matrix=np.asarray(data['alignment']['rotation'])
            if np.allclose(other@matrix.T,sample,atol=1e-5):return 'already_aligned'
    if len(matches)!=1:
        raise RuntimeError(f'현재 모델과 일치하는 COLMAP 실행이 하나가 아닙니다: {folder.name}')
    run=matches[0];models=[]
    for model in (run/'sparse').iterdir():
        if (model/'cameras.bin').exists():
            rec=pycolmap.Reconstruction(str(model));models.append((rec.num_reg_images(),rec))
    rec=max(models,key=lambda item:item[0])[1]
    alignment=camera_alignment(rec)
    diagnostics=reconstruction_diagnostics(rec,folder/'points.ply',folder/'masks',rec.num_reg_images())
    backup=folder/f'orientation-backup-v{data["version"]}';backup.mkdir(exist_ok=True)
    for name in ['model.ply','points.ply','session.json']:
        if not (backup/name).exists():shutil.copyfile(folder/name,backup/name)
    for name in ['model.ply','points.ply']:
        rotate_ply(folder/name,folder/(name+'.oriented'),alignment['rotation'])
    if metadata.read_text(encoding='utf-8')!=original_text:
        raise RuntimeError('세션이 변경되어 방향 보정 반영을 중지했습니다. 진행 중인 작업을 마친 후 다시 실행하세요.')
    for name in ['model.ply','points.ply']:os.replace(folder/(name+'.oriented'),folder/name)
    data.update(alignment=alignment,diagnostics=diagnostics,reconstruction_run=run.name,version=data['version']+1)
    if data['state']=='complete':data['stage']='복원 처리 완료 · 형상과 누락을 확인하세요.'
    temp=metadata.with_name('session.json.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,metadata)
    return dict(id=data['id'],version=data['version'],alignment=alignment,diagnostics=diagnostics,backup=backup.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scan-id');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'data'/'scans'
    folders=[root/args.scan_id] if args.scan_id else sorted(root.iterdir())
    for folder in folders:
        if folder.is_dir() and (folder/'session.json').exists():
            print(json.dumps(align(folder),ensure_ascii=False),flush=True)
