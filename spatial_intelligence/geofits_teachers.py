"""Frozen real multi-level geometry extraction and explicit shared-FOV mapping.

Upstream interfaces audited at facebookresearch/vggt a288dd0f1478 and
yyfz/Pi3 9fa3ddb3f8d5. This wrapper does not vendor upstream implementations.
VGGT outputs concatenate frame/global states (2048); Pi3 decoder block outputs
are 1024, NOT its final concatenation of blocks 34/35 (2048).
"""
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from .geofits import FrozenGeometryTeacher


def matching_teacher_canvas(qwen_canvas, frame_ids, *, qwen_patch=16, teacher_patch=14):
    """Resize the SAME letterboxed RGB canvas, never independently crop teachers.

    Inputs are [B,T,3,H,W] unnormalized [0,1] and ordered per-clip frame IDs.
    448/16 -> 392/14 gives identical 28x28 patch-center relative coordinates;
    2x2 spatial pooling then yields 14x14 merged regions for each input image.
    This is a declared resolution adaptation, not identical pixel evidence.
    Qwen processor must preserve the supplied canvas (no additional resizing).
    """
    if qwen_canvas.ndim != 5 or qwen_canvas.shape[2] != 3:
        raise ValueError('Expected unnormalized [B,T,3,H,W] shared letterbox canvas')
    b, t, c, h, w = qwen_canvas.shape
    if t < 1 or t > 32 or h % (2*qwen_patch) or w % (2*qwen_patch):
        raise ValueError('One to 32 frames and exact merged native patch grid required')
    if len(frame_ids) != b or any(len(ids) != t or len(set(ids)) != t for ids in frame_ids):
        raise ValueError('Frame identities must be ordered, unique, and count-aligned')
    if not torch.isfinite(qwen_canvas).all() or qwen_canvas.min() < 0 or qwen_canvas.max() > 1:
        raise ValueError('Canvas must be finite unnormalized RGB in [0,1]')
    th, tw = h // qwen_patch * teacher_patch, w // qwen_patch * teacher_patch
    resized = F.interpolate(qwen_canvas.flatten(0, 1).float(), size=(th, tw),
                            mode='bilinear', align_corners=False, antialias=True)
    receipt = dict(policy='same-letterbox-normalized-coordinates-different-pixel-resolution',
        frame_ids=[list(ids) for ids in frame_ids], qwen_hw=[h,w], teacher_hw=[th,tw],
        qwen_patch=qwen_patch, teacher_patch=teacher_patch,
        patch_grid=[h//qwen_patch,w//qwen_patch], merged_grid=[h//(2*qwen_patch),w//(2*qwen_patch)],
        temporal_group=1, interpolation='bilinear-antialias-align_corners_false')
    return resized.reshape(b,t,c,th,tw), receipt


def patch_grid(tokens, *, batch, frames, height, width, prefix_tokens, feature_width):
    """Restore true intermediate tokens; reject accidental final-head tensors."""
    expected = prefix_tokens + height * width
    shapes = {(batch,frames,expected,feature_width),
              (batch*frames,expected,feature_width), (batch,frames*expected,feature_width)}
    if tuple(tokens.shape) not in shapes:
        raise ValueError('Unexpected intermediate token shape/width/register count')
    if tokens.shape[-1] != feature_width:
        raise ValueError('Feature width is not the declared intermediate width')
    return tokens.reshape(batch,frames,expected,feature_width)[:, :, prefix_tokens:].reshape(
        batch,frames,height,width,feature_width)


class _FrozenLevels(FrozenGeometryTeacher):
    def __init__(self, backbone):
        super().__init__(backbone)

    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self

    def _apply(self, fn, recurse=True):
        constants = {name: getattr(self.backbone,name).float().clone()
                     for name in ('_resnet_mean','_resnet_std','image_mean','image_std')
                     if hasattr(self.backbone,name)}
        result=super()._apply(fn,recurse=recurse)
        for name,value in constants.items():
            setattr(self.backbone,name,value.to(device=getattr(self.backbone,name).device))
        return result

    @staticmethod
    def validate_images(images):
        if images.ndim != 5 or images.shape[2] != 3 or images.shape[-2] % 14 or images.shape[-1] % 14:
            raise ValueError('Teacher expects [B,T,3,H,W], dimensions divisible by patch14')
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('Teacher requires finite RGB in [0,1], not normalized pixels')

    @staticmethod
    def precision(images):
        return torch.autocast('cuda', dtype=torch.bfloat16) if images.is_cuda else nullcontext()


class VGGTMultiLevel(_FrozenLevels):
    levels = (11,17,23)
    feature_width = 2048

    def __init__(self, aggregator):
        super().__init__(aggregator)
        if aggregator.patch_size != 14 or aggregator.depth != 24:
            raise ValueError('Expected pinned VGGT patch14 / 24-block aggregator')
        if hasattr(aggregator,'cached_layer_indices'):
            aggregator.cached_layer_indices = set(self.levels)

    def forward(self, images):
        self.validate_images(images)
        b,t,_,h,w=images.shape
        with torch.no_grad(), self.precision(images):
            outputs,start=self.backbone(images)
            if start != 5: raise ValueError('Unexpected VGGT camera/register prefix')
            result={}
            for level in self.levels:
                if len(outputs) <= level or outputs[level] is None:
                    raise ValueError('Requested true VGGT intermediate missing')
                result[level]=patch_grid(outputs[level],batch=b,frames=t,height=h//14,width=w//14,
                                         prefix_tokens=start,feature_width=self.feature_width)
        return result


class Pi3MultiLevel(_FrozenLevels):
    levels = (17,26,35)
    feature_width = 1024

    def __init__(self, backbone):
        super().__init__(backbone)
        if backbone.patch_size != 14 or len(backbone.decoder) != 36 or backbone.patch_start_idx != 5:
            raise ValueError('Expected Pi3-Large patch14 / 36-block / five-register architecture')
        if backbone.dec_embed_dim != 1024:
            raise ValueError('Pi3 intermediate block width must be 1024, not concatenated final2048')

    def forward(self, images):
        self.validate_images(images)
        b,t,_,h,w=images.shape
        captured={}; handles=[]
        def capture(level):
            def hook(module,args,output):
                captured[level]=patch_grid(output,batch=b,frames=t,height=h//14,width=w//14,
                    prefix_tokens=5,feature_width=self.feature_width)
            return hook
        try:
            for level in self.levels:
                handles.append(self.backbone.decoder[level].register_forward_hook(capture(level)))
            with torch.no_grad(), self.precision(images):
                # Upstream performs normalization BEFORE DINO, then decode runs
                # all alternating within-frame/global layers. No point/camera heads.
                mean=self.backbone.image_mean.float(); std=self.backbone.image_std.float()
                normalized=(images.float()-mean)/std
                encoded=self.backbone.encoder(normalized.flatten(0,1),is_training=True)
                if isinstance(encoded,dict): encoded=encoded['x_norm_patchtokens']
                self.backbone.decode(encoded,t,h,w)
            if set(captured) != set(self.levels): raise ValueError('Pi3 intermediate hooks did not all execute')
            return captured
        finally:
            for handle in handles: handle.remove()


def _source_import(source, package):
    source=Path(source).resolve()
    if not (source/package).is_dir(): raise ValueError('Extracted pinned upstream package missing')
    if package in sys.modules:
        existing=Path(sys.modules[package].__file__).resolve()
        if source not in existing.parents: raise ValueError('Different upstream source already imported')
    sys.path.insert(0,str(source))


def load_vggt_levels(source, weights):
    """Local-only strict released-model load; caller installs source dependencies."""
    _source_import(source,'vggt')
    from vggt.models.vggt import VGGT
    full=VGGT()
    state=torch.load(weights,map_location='cpu',weights_only=True)
    full.load_state_dict(state,strict=True)
    return VGGTMultiLevel(full.aggregator)


def load_pi3_levels(source, weights):
    """Local-only strict load of original Pi3, not Pi3X or downloaded fallback."""
    _source_import(source,'pi3')
    from pi3.models.pi3 import Pi3
    from safetensors.torch import load_file
    full=Pi3(decoder_size='large')
    full.load_state_dict(load_file(str(weights),device='cpu'),strict=True)
    # Validate complete release first, retain only trunk required for extraction.
    for name in ('point_decoder','point_head','conf_decoder','conf_head','camera_decoder','camera_head'):
        delattr(full,name)
    return Pi3MultiLevel(full)
