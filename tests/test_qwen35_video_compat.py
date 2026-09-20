from types import SimpleNamespace,MethodType
import pytest


def test_timestamp_groups_and_original_grid_unchanged():
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    if transformers.__version__!='5.3.0':pytest.skip('Pinned 5.3 regression')
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from spatial_intelligence.qwen35_video_compat import install_video_rope_compat
    inner=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    inner.get_vision_position_ids=MethodType(Qwen3_5Model.get_vision_position_ids,inner)
    inner.get_rope_index=MethodType(Qwen3_5Model.get_rope_index,inner)
    model=SimpleNamespace(model=inner)
    grid=torch.tensor([[2,4,4]])
    ids=torch.ones((1,11),dtype=torch.long)
    types=torch.tensor([[0,2,2,2,2,0,2,2,2,2,0]])
    with pytest.raises(StopIteration):inner.get_rope_index(ids,types,video_grid_thw=grid)
    assert install_video_rope_compat(model)=='transformers-5.3-video-grid-backport-v1'
    positions,deltas=inner.get_rope_index(ids,types,video_grid_thw=grid)
    assert positions.shape==(3,1,11) and deltas.shape==(1,1)
    assert grid.tolist()==[[2,4,4]]
    assert positions[:,0,1:5].tolist()==[[1,1,1,1],[1,1,2,2],[1,2,1,2]]
    assert positions[:,0,6:10].tolist()==[[4,4,4,4],[4,4,5,5],[4,5,4,5]]
