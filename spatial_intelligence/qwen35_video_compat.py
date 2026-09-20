"""Narrow Transformers 5.3 video-RoPE backport; does not change vision grids.

Upstream Qwen3.5 get_rope_index splits each video grid into timestamp-separated
temporal groups. Version 5.3 lacks this step. Keep the original grid for the
vision encoder, expanding only the private copy passed to position indexing.
"""
import inspect
from types import MethodType


def install_video_rope_compat(model):
    import torch
    import transformers
    inner=model.model
    original=inner.get_rope_index
    if getattr(inner,'_eurekasi_video_rope_compat',False):return 'already-installed'
    source=inspect.getsource(original)
    if 'repeat_interleave(video_grid_thw' in source:return 'upstream-native'
    if transformers.__version__!='5.3.0':
        raise RuntimeError('Unverified Qwen3.5 video RoPE implementation; inspect upstream before evaluation')
    def compatible(self,input_ids,mm_token_type_ids,image_grid_thw=None,video_grid_thw=None,attention_mask=None,**kwargs):
        if video_grid_thw is not None:
            video_grid_thw=torch.repeat_interleave(video_grid_thw,video_grid_thw[:,0],dim=0)
            video_grid_thw[:,0]=1
        return original(input_ids,mm_token_type_ids,image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,attention_mask=attention_mask,**kwargs)
    inner.get_rope_index=MethodType(compatible,inner)
    inner._eurekasi_video_rope_compat=True
    return 'transformers-5.3-video-grid-backport-v1'
