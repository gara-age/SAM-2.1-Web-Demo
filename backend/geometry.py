"""Pixel geometry. Bounding box endpoints are inclusive pixel indices."""
import base64
import cv2
import numpy as np


def measure(mask: np.ndarray) -> dict:
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return dict(area=0, area_percent=0, bbox=None, corners=[], center=None,
                    width=None, height=None, contours=[])
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return dict(area=int(xs.size), area_percent=round(xs.size / (w*h)*100, 4),
                bbox=[x0, y0, x1, y1], width=x1-x0+1, height=y1-y0+1,
                corners=[[x0,y0], [x1,y0], [x1,y1], [x0,y1]],
                center=[round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
                contours=[dict(points=c.reshape(-1,2).tolist(),
                               parent=int(hierarchy[0,i,3])) for i,c in enumerate(contours)])


def png_mask(mask):
    ok, encoded = cv2.imencode('.png', mask.astype(np.uint8)*255)
    if not ok:
        raise RuntimeError('Mask encoding failed')
    return base64.b64encode(encoded).decode('ascii')
