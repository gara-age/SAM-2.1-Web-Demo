"""SAM 2.1 public video API adapter using bounded JPEG chunks.

No private predictor state is extended. Each chunk seeds frame zero with the
previous masks, then propagates within that chunk; long-term memory resets.
"""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class SamEngine:
    def __init__(self):
        self.predictor = None
        self.status = 'not_loaded'
        self.error = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint = ROOT/'models'/'sam2.1_hiera_tiny.pt'
        self.temp_dir = ROOT/'.cache'/'frames'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self.predictor is not None:
            return
        self.status = 'loading'
        try:
            from sam2.build_sam import build_sam2_video_predictor
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f'체크포인트가 없습니다: {self.checkpoint}')
            self.predictor = build_sam2_video_predictor(
                'configs/sam2.1/sam2.1_hiera_t.yaml', str(self.checkpoint),
                device=self.device, apply_postprocessing=False)
            self.status = 'ready'
            self.error = None
        except Exception as exc:
            self.status = 'error'
            self.error = str(exc)
            raise

    def precision(self):
        # RTX 2080 has no native BF16. Use FP32 unless the device supports BF16.
        if self.device=='cuda' and torch.cuda.is_bf16_supported():
            return torch.autocast('cuda', dtype=torch.bfloat16)
        return nullcontext()

    def infer(self, frames, masks, target=None, points=None):
        self.load()
        result = []
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as folder:
            for i,frame in enumerate(frames):
                if not cv2.imwrite(str(Path(folder)/f'{i:05d}.jpg'), frame.image,
                                   [cv2.IMWRITE_JPEG_QUALITY,95]):
                    raise RuntimeError('프레임 임시 저장 실패')
            with torch.inference_mode(), self.precision():
                state = self.predictor.init_state(folder, offload_video_to_cpu=True)
                for obj_id,mask in masks.items():
                    if obj_id!=target and mask.any():
                        self.predictor.add_new_mask(state, frame_idx=0,obj_id=obj_id,mask=mask)
                if target is not None:
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=target,
                        points=np.array([[p['x'],p['y']] for p in points],np.float32),
                        labels=np.array([p['label'] for p in points],np.int32))
                if not state['obj_ids']:
                    return [(f,{}) for f in frames]
                for index,ids,logits in self.predictor.propagate_in_video(state):
                    values = (logits[:,0]>0).cpu().numpy()
                    result.append((frames[index],{int(i):m for i,m in zip(ids,values)}))
                del state
        return result
