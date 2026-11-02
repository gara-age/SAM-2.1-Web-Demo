"""CPU optical flow between SAM results; never labels interpolation as SAM inference."""
import cv2
import numpy as np


class MotionTracker:
    def __init__(self,frame,masks):
        self.frame=frame
        self.gray,self.scale=self._gray(frame.image)
        self.masks={i:m.copy() for i,m in masks.items()}
        self.transforms={i:np.eye(3) for i in masks}

    @staticmethod
    def _gray(image):
        h,w=image.shape[:2];scale=min(1,640/w)
        gray=cv2.cvtColor(cv2.resize(image,(round(w*scale),round(h*scale))),cv2.COLOR_BGR2GRAY)
        return gray,scale

    def advance(self,frame):
        gray,scale=self._gray(frame.image)
        h,w=frame.image.shape[:2]
        for obj_id,mask in list(self.masks.items()):
            matrix=None
            small=cv2.resize(mask.astype('uint8'),(self.gray.shape[1],self.gray.shape[0]),interpolation=cv2.INTER_NEAREST)
            points=cv2.goodFeaturesToTrack(self.gray,120,.01,5,mask=small,blockSize=5)
            if points is not None and len(points)>=4:
                moved,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,points,None,winSize=(21,21),maxLevel=3)
                back,valid,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,moved,None,winSize=(21,21),maxLevel=3)
                good=(ok[:,0]>0)&(valid[:,0]>0)&(np.linalg.norm(points[:,0]-back[:,0],axis=1)<1.5)
                if good.sum()>=4:
                    affine,inliers=cv2.estimateAffinePartial2D(points[good,0],moved[good,0],method=cv2.RANSAC,ransacReprojThreshold=2)
                    if affine is not None and inliers.sum()>=4:
                        zoom=np.linalg.norm(affine[:2,0])
                        if .75<zoom<1.33 and np.isfinite(affine).all():
                            matrix=np.eye(3);matrix[:2]=affine;matrix[:2,2]/=scale
            if matrix is None or self.transforms[obj_id] is None:
                self.masks[obj_id]=np.zeros((h,w),bool);self.transforms[obj_id]=None
            else:
                self.masks[obj_id]=cv2.warpAffine(mask.astype('uint8'),matrix[:2],(w,h),flags=cv2.INTER_NEAREST).astype(bool)
                self.transforms[obj_id]=matrix@self.transforms[obj_id]
        self.gray=gray;self.frame=frame
        return self.masks

    def snapshot(self):
        return {i:t.copy() if t is not None else None for i,t in self.transforms.items()}

    def correct(self,masks,base):
        """Move delayed SAM masks to the current frame using measured transforms."""
        h,w=self.frame.image.shape[:2];applied=0
        for obj_id,mask in masks.items():
            current=self.transforms.get(obj_id);previous=base.get(obj_id)
            if current is None or previous is None:continue
            transform=current@np.linalg.inv(previous)
            corrected=cv2.warpAffine(mask.astype('uint8'),transform[:2],(w,h),flags=cv2.INTER_NEAREST).astype(bool)
            self.masks[obj_id]=corrected;applied+=1
        return applied
