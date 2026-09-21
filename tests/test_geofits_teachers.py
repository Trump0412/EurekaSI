"""Tiny fake upstream contracts only; not released VGGT/Pi3 GPU acceptance."""
import pytest
import torch
from torch import nn
from spatial_intelligence.geofits_teachers import (matching_teacher_canvas,
    patch_grid,VGGTMultiLevel,Pi3MultiLevel)


def test_common_canvas_preserves_frame_order_and_relative_patch_grid():
    rgb=torch.stack((torch.zeros(3,448,448),torch.ones(3,448,448)))[None]
    out,receipt=matching_teacher_canvas(rgb,[['frame-a','frame-b']])
    assert out.shape==(1,2,3,392,392)
    assert out[0,0].max()==0 and out[0,1].min()==1
    assert receipt['patch_grid']==[28,28] and receipt['merged_grid']==[14,14]
    assert receipt['frame_ids']==[['frame-a','frame-b']]
    with pytest.raises(ValueError,match='identities'):
        matching_teacher_canvas(rgb,[['same','same']])


class FakeVGGT(nn.Module):
    patch_size=14; depth=24
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1));self.cached_layer_indices=set()
        self.register_buffer('_resnet_mean',torch.tensor([.485,.456,.406]))
    def forward(self,images):
        b,t,_,h,w=images.shape
        outputs=[None]*24
        for i in self.cached_layer_indices:
            x=torch.full((b,t,5+(h//14)*(w//14),2048),float(i))
            x[:,:,:5]=-99
            outputs[i]=x
        return outputs,5


def test_vggt_true_level_mapping_and_register_strip_frozen_norm():
    teacher=VGGTMultiLevel(FakeVGGT());teacher.train()
    outputs=teacher(torch.zeros(1,2,3,28,28))
    for level,x in outputs.items():
        assert x.shape==(1,2,2,2,2048)
        assert (x==level).all()
    before=teacher.backbone._resnet_mean.clone();teacher.bfloat16()
    assert torch.equal(before,teacher.backbone._resnet_mean)
    assert teacher.backbone._resnet_mean.dtype==torch.float32
    assert not teacher.backbone.training
    assert not any(p.requires_grad for p in teacher.parameters())


class Offset(nn.Module):
    def __init__(self,index):super().__init__();self.index=index
    def forward(self,x):return torch.full_like(x,float(self.index))


class FakeEncoder(nn.Module):
    def forward(self,x,is_training):
        return {'x_norm_patchtokens':torch.zeros(x.shape[0],(x.shape[-2]//14)*(x.shape[-1]//14),1024)}


class FakePi3(nn.Module):
    patch_size=14;patch_start_idx=5;dec_embed_dim=1024
    def __init__(self):
        super().__init__();self.decoder=nn.ModuleList([Offset(i) for i in range(36)])
        self.encoder=FakeEncoder()
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('image_std',torch.ones(1,3,1,1))
    def decode(self,encoded,frames,h,w):
        bt,n,c=encoded.shape;b=bt//frames
        hidden=torch.cat((torch.zeros(bt,5,c),encoded),1)
        for i,block in enumerate(self.decoder):
            hidden=hidden.reshape(bt,n+5,c) if i%2==0 else hidden.reshape(b,frames*(n+5),c)
            hidden=block(hidden)
        return torch.cat((hidden,hidden),-1),None


def test_pi3_actual_selected_blocks_not_final_concat_and_cleanup():
    teacher=Pi3MultiLevel(FakePi3())
    outputs=teacher(torch.zeros(1,2,3,28,28))
    assert set(outputs)=={17,26,35}
    for level,x in outputs.items():
        assert x.shape==(1,2,2,2,1024)
        assert (x==level).all()
    assert not any(block._forward_hooks for block in teacher.backbone.decoder)


def test_wrong_intermediate_width_is_not_silently_accepted():
    with pytest.raises(ValueError):
        patch_grid(torch.zeros(1,2,9,2048),batch=1,frames=2,height=2,width=2,
                   prefix_tokens=5,feature_width=1024)
