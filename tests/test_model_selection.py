import pytest
from backend.service import TrackingService
from backend.sam31 import Sam31Engine


def test_switch_resets_selected_objects_and_preserves_camera():
    service=TrackingService()
    try:
        camera=service.camera
        service.objects[8]={'id':8,'points':[],'path':[]}
        result=service.select_model('sam3.1')
        assert result['model_name']=='sam3.1'
        assert isinstance(service.engine,Sam31Engine)
        assert service.objects=={} and service.camera is camera
        assert service.select_model('sam2.1')['model_name']=='sam2.1'
        with pytest.raises(ValueError):service.select_text(1,'사람 선택해')
        with pytest.raises(ValueError):service.select_model('unknown')
    finally:service.close()


def test_missing_checkpoint_is_an_explicit_error(tmp_path,monkeypatch):
    import backend.sam31 as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    engine=module.Sam31Engine()
    with pytest.raises(ValueError,match='체크포인트'):engine.load()
    assert engine.status=='error'
    assert engine.process is None


def test_sam31_masks_survive_frames_without_grounding_cache():
    import numpy as np
    from backend.sam31_worker import preserve_refined_output
    class Model:
        def _build_sam2_output(self,state,index,refined=None):
            if index not in state:return {}
            return {**state[index],**(refined or {})}
    model=Model();preserve_refined_output(model)
    predicted=np.ones((4,4),bool)
    assert model._build_sam2_output({},1,{7:predicted})[7] is predicted
    assert model._build_sam2_output({},1,{})=={}
    previous=np.zeros((4,4),bool)
    result=model._build_sam2_output({0:{8:previous}},0,{7:predicted})
    assert result[8] is previous and result[7] is predicted
