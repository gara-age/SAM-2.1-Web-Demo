import numpy as np
import pytest
from backend.scan_geometry import upright_rotation,rotate_ply


def test_camera_down_is_viewer_up_and_no_mirror():
    rotation=upright_rotation([np.eye(3)])
    np.testing.assert_allclose(rotation@np.array([0,-1,0]),[0,1,0])
    np.testing.assert_allclose(rotation@np.array([1,0,0]),[1,0,0])
    assert np.linalg.det(rotation)==pytest.approx(1)


def test_rolled_world_keeps_camera_up_and_distances():
    angle=.7
    cam=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    rotation=upright_rotation([cam,cam])
    np.testing.assert_allclose(rotation@(-cam[1]),[0,1,0],atol=1e-8)
    pair=np.array([[3,4,5],[-1,6,2]])
    transformed=pair@rotation.T
    assert np.linalg.norm(transformed[1]-transformed[0])==pytest.approx(np.linalg.norm(pair[1]-pair[0]))


def test_ply_rotates_normals_preserves_colors_faces_and_original(tmp_path):
    from plyfile import PlyData,PlyElement
    v=np.zeros(3,dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('red','u1')])
    v['x']=[0,1,0];v['y']=[0,0,-1];v['ny']=1;v['red']=[10,20,30]
    faces=np.array([([0,1,2],)],dtype=[('vertex_indices','i4',(3,))])
    source=tmp_path/'source.ply';target=tmp_path/'target.ply'
    PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(faces,'face')]).write(str(source))
    rotate_ply(source,target,upright_rotation([np.eye(3)]))
    result=PlyData.read(str(target))
    np.testing.assert_allclose(result['vertex']['y'],[0,0,1])
    np.testing.assert_allclose(result['vertex']['ny'],[-1,-1,-1])
    np.testing.assert_array_equal(result['vertex']['red'],[10,20,30])
    np.testing.assert_array_equal(result['face']['vertex_indices'][0],[0,1,2])
    np.testing.assert_allclose(PlyData.read(str(source))['vertex']['y'],v['y'])
    with pytest.raises(ValueError):rotate_ply(source,target,np.diag([1,-1,1]))
