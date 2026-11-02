"""Isolated Linux worker. Only official SAM 3.1 APIs; no cloud inference."""
from pathlib import Path
import json
import sys
import traceback
import inspect
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
predictor=None
precision=None


def preserve_refined_output(model):
    """Keep actual tracker masks on frames without a prior grounding cache."""
    native_merge=model._build_sam2_output
    def merge(inference_state,frame_idx,refined_obj_id_to_mask=None):
        output=dict(native_merge(inference_state,frame_idx,refined_obj_id_to_mask))
        if refined_obj_id_to_mask:output.update(refined_obj_id_to_mask)
        return output
    model._build_sam2_output=merge


def load():
    global predictor,precision
    if predictor is not None:return {'precision':precision}
    import torch
    if not torch.cuda.is_available():raise RuntimeError('WSL CUDA GPU를 사용할 수 없습니다.')
    from sam3.model_builder import build_sam3_predictor
    old_gpu=torch.cuda.get_device_capability()[0]<8
    if old_gpu:
        # Upstream's RoPE decoder selects FLASH_ATTENTION unconditionally even
        # when use_fa3=False. Turing has no such kernel. Enable PyTorch's math
        # implementation at this dispatch boundary, keeping the same SAM model.
        import sam3.model.decoder as decoder
        from torch.nn.attention import SDPBackend
        native_sdpa_kernel=decoder.sdpa_kernel
        def compatible_sdpa(backends,*args,**kwargs):
            if backends==SDPBackend.FLASH_ATTENTION:backends=[SDPBackend.MATH]
            return native_sdpa_kernel(backends,*args,**kwargs)
        decoder.sdpa_kernel=compatible_sdpa
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        native_attention=torch.nn.functional.scaled_dot_product_attention
        def compatible_attention(q,k,v,*args,**kwargs):
            if q.dtype!=torch.bfloat16:return native_attention(q,k,v,*args,**kwargs)
            # Turing supports efficient FP16 SDPA, but not BF16 FlashAttention.
            # Limit the conversion to attention, preserve the model's output dtype.
            args=list(args)
            if args and isinstance(args[0],torch.Tensor) and args[0].is_floating_point():args[0]=args[0].half()
            mask=kwargs.get('attn_mask')
            if isinstance(mask,torch.Tensor) and mask.is_floating_point():kwargs['attn_mask']=mask.half()
            with native_sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
                return native_attention(q.half(),k.half(),v.half(),*args,**kwargs).to(q.dtype)
        torch.nn.functional.scaled_dot_product_attention=compatible_attention
        torch.set_autocast_cache_enabled(False)
    predictor=build_sam3_predictor(checkpoint_path=str(ROOT/'models'/'sam3.1'/'sam3.1_multiplex.pt'),
                                  version='sam3.1',compile=False,warm_up=False,use_fa3=False,
                                  use_rope_real=True,async_loading_frames=False,max_num_objects=4)
    preserve_refined_output(predictor.model)
    # Pinned upstream base predictor passes this SAM-3-only flag even though
    # multiplex init_state does not accept it. Keep multiplex's actual API and
    # discard only the unsupported False/default argument at this boundary.
    native_init=predictor.model.init_state
    if 'offload_state_to_cpu' not in inspect.signature(native_init).parameters:
        def compatible_init(*args,offload_state_to_cpu=False,**kwargs):
            if offload_state_to_cpu:
                raise ValueError('SAM 3.1 multiplex does not support state CPU offload.')
            return native_init(*args,**kwargs)
        predictor.model.init_state=compatible_init
    precision='BF16 + FP16 attention (Turing compatibility)' if old_gpu else 'BF16 autocast (official)'
    return {'precision':precision,'gpu':torch.cuda.get_device_name(0)}


def seed_points(mask):
    import cv2
    # Multiplex's public API accepts points, not masks. Re-prompt each short chunk
    # from the previous mask's interior, retaining caller-specified object IDs.
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    points=[]
    for _ in range(3):
        y,x=np.unravel_index(np.argmax(distance),distance.shape)
        radius=float(distance[y,x])
        if radius<=0:break
        points.append([int(x),int(y)])
        cv2.circle(distance,(int(x),int(y)),max(8,int(radius*1.5)),0,-1)
    return points


def infer(request):
    import torch
    load();folder=Path(request['folder']);images=folder/'frames'
    from PIL import Image
    with Image.open(images/'00000.jpg') as image:width,height=image.size
    def relative(points):return [[x/width,y/height] for x,y in points]
    session=predictor.handle_request({'type':'start_session','resource_path':str(images),
                                      'offload_video_to_cpu':True})['session_id']
    saved={}
    def accept(index,outputs):
        if not outputs:return
        ids=outputs['out_obj_ids'];masks=outputs['out_binary_masks']
        if isinstance(ids,torch.Tensor):ids=ids.cpu().numpy()
        if isinstance(masks,torch.Tensor):masks=masks.cpu().numpy()
        for obj_id,mask in zip(ids,masks):saved[f'{index}_{int(obj_id)}']=np.asarray(mask,dtype=bool).copy()
    try:
        if request.get('text'):
            response=predictor.handle_request(dict(type='add_prompt',session_id=session,frame_index=0,text=request['text']))
            accept(0,response['outputs'])
        else:
            with np.load(folder/'seeds.npz',allow_pickle=False) as seeds:
                for key in seeds.files:
                    obj_id=int(key)
                    if obj_id==request.get('target'):continue
                    points=seed_points(seeds[key])
                    if points:
                        predictor.handle_request(dict(type='add_prompt',session_id=session,frame_index=0,
                            obj_id=obj_id,points=relative(points),point_labels=[1]*len(points),rel_coordinates=True))
            if request.get('target') is not None:
                points=request['points']
                predictor.handle_request(dict(type='add_prompt',session_id=session,frame_index=0,
                    obj_id=request['target'],points=relative([[p['x'],p['y']] for p in points]),
                    point_labels=[p['label'] for p in points],rel_coordinates=True))
            for response in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=session,
                      propagation_direction='forward',start_frame_index=0)):
                accept(response['frame_index'],response['outputs'])
        np.savez_compressed(folder/'output.npz',**saved)
        return {'objects':len(saved),'precision':precision}
    finally:
        predictor.handle_request({'type':'close_session','session_id':session,'clear_cache_threshold':0})


def main():
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            result=load() if request['operation']=='load' else infer(request)
            response={'id':request['id'],'result':result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response={'id':request.get('id'),'error':f'{type(exc).__name__}: {exc}'}
        print('SAM31_RPC:'+json.dumps(response,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
