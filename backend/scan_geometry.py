"""Camera-based orientation and observable reconstruction diagnostics."""
from pathlib import Path
import os
import cv2
import numpy as np


def upright_rotation(rotations):
    """Right-handed world -> viewer rotation; camera image up becomes viewer +Y.

    This aligns to the capture's camera-up, not measured gravity. Camera optical
    frames use +Y down, +Z forward; the viewer uses +Y up and looks along -Z.
    """
    rotations=np.asarray(rotations,dtype=np.float64)
    if rotations.ndim!=3 or rotations.shape[1:]!=(3,3) or not len(rotations):
        raise ValueError('카메라 회전 정보가 없습니다.')
    up=-rotations[:,1,:].mean(axis=0)
    if np.linalg.norm(up)<1e-6: up=-rotations[0,1,:]
    up=up/np.linalg.norm(up)
    front=-rotations[0,2,:]
    front-=np.dot(front,up)*up
    if np.linalg.norm(front)<1e-6:
        up=-rotations[0,1,:];front=-rotations[0,2,:]
    front/=np.linalg.norm(front)
    right=np.cross(up,front);right/=np.linalg.norm(right)
    up=np.cross(front,right)
    return np.stack([right,up,front])


def camera_alignment(rec):
    images=sorted(rec.images.values(),key=lambda image:image.name)
    rotation=upright_rotation([im.cam_from_world().rotation.matrix() for im in images])
    return dict(method='camera_up_v1',rotation=rotation.tolist(),
                reference_image=images[0].name,gravity_calibrated=False)


def rotate_ply(source,target,rotation):
    """Rotate positions and normals with no reflection, scale or topology edits."""
    from plyfile import PlyData,PlyElement
    source,target=Path(source),Path(target)
    ply=PlyData.read(str(source));vertices=ply['vertex'].data.copy()
    matrix=np.asarray(rotation,dtype=np.float64)
    if not np.allclose(matrix@matrix.T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(matrix),1):
        raise ValueError('정상적인 3D 회전 행렬이 아닙니다.')
    for keys in [('x','y','z'),('nx','ny','nz')]:
        if all(key in vertices.dtype.names for key in keys):
            values=np.column_stack([vertices[key] for key in keys])@matrix.T
            for index,key in enumerate(keys): vertices[key]=values[:,index]
    elements=[PlyElement.describe(vertices,'vertex')]+[e for e in ply.elements if e.name!='vertex']
    temp=target.with_name(target.name+'.tmp')
    PlyData(elements,text=False,byte_order='<').write(str(temp))
    os.replace(temp,target)


def reconstruction_diagnostics(rec,points_file,mask_folder,total_views):
    """Projection coverage is a gap diagnostic, never confidence or accuracy."""
    from plyfile import PlyData
    vertices=PlyData.read(str(points_file))['vertex']
    xyz=np.column_stack([vertices[key] for key in ('x','y','z')]).astype(np.float64)
    coverages=[]
    for image in rec.images.values():
        mask=cv2.imread(str(Path(mask_folder)/(image.name+'.png')),cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        h,w=mask.shape
        small=cv2.resize(mask,(160,120),interpolation=cv2.INTER_NEAREST)>0
        transformed=image.cam_from_world()*xyz
        uv=rec.cameras[image.camera_id].img_from_cam(transformed)*np.array([160/w,120/h])
        good=np.isfinite(uv).all(axis=1)&(transformed[:,2]>0)
        uv=np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1).astype(np.int64)
        good&=(uv[:,0]>=0)&(uv[:,0]<160)&(uv[:,1]>=0)&(uv[:,1]<120)
        raster=np.zeros((120,160),np.uint8);raster[uv[good,1],uv[good,0]]=1
        raster=cv2.dilate(raster,np.ones((3,3),np.uint8))>0
        coverages.append(float(np.count_nonzero(raster&small)/max(np.count_nonzero(small),1)))
    error=float(rec.compute_mean_reprojection_error())
    coverage=float(np.median(coverages)) if coverages else None
    warnings=[]
    if coverage is not None and coverage<.75:
        warnings.append('촬영된 객체 영역에 복원 점이 없는 부분이 많습니다. 누락·왜곡을 확인하세요.')
    if error>2:
        warnings.append('사진 사이의 대응점 오차가 큽니다. 흔들림·대상 움직임·초점을 확인하세요.')
    if rec.num_reg_images()<total_views*.85:
        warnings.append('일부 촬영 시점이 복원에 연결되지 않았습니다.')
    if len(vertices)<2000:
        warnings.append('표면 복원 점이 적어 세부 형상이 부족할 수 있습니다.')
    return dict(warnings=warnings,registered_views=rec.num_reg_images(),total_views=total_views,
                mean_reprojection_error_px=round(error,3),
                projected_mask_coverage=round(coverage,3) if coverage is not None else None,
                measurement_note='투영 영역 비율과 대응점 오차는 실제 형상 정확도나 신뢰도가 아닙니다.')
