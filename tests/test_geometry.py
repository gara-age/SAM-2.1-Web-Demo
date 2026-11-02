import cv2
import numpy as np
from backend.geometry import measure


def test_rectangle_inclusive_size_and_pixel_centroid():
    mask = np.zeros((80,100),bool)
    mask[10:30,20:60] = True
    result = measure(mask)
    assert result['bbox']==[20,10,59,29]
    assert (result['width'],result['height'])==(40,20)
    assert result['center']==[39.5,19.5]
    assert result['area']==800
    assert result['area_percent']==10
    assert result['corners']==[[20,10],[59,10],[59,29],[20,29]]


def test_holes_and_disconnected_components_keep_hierarchy():
    mask = np.zeros((100,100),np.uint8)
    mask[10:50,10:50]=1
    mask[20:40,20:40]=0
    mask[70:80,70:80]=1
    result=measure(mask)
    assert result['area']==1300
    assert len(result['contours'])==3
    assert sum(c['parent']>=0 for c in result['contours'])==1


def test_empty_and_single_border_pixel():
    empty = np.zeros((3,4),bool)
    result=measure(empty)
    assert result['area']==0
    assert result['center'] is None and result['bbox'] is None
    assert result['width'] is None and result['contours']==[]
    empty[2,3]=True
    result=measure(empty)
    assert result['center']==[3,2]
    assert result['bbox']==[3,2,3,2]
    assert result['width']==result['height']==result['area']==1
