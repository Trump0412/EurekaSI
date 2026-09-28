"""Small CPU hybrid-cache contract; not full-model GPU acceptance."""
import torch
from torch import nn
from transformers import Qwen3_5Config
from spatial_intelligence.qwen3vl_geometry_matrix import Qwen35GeometryMatrix


class Features(nn.Module):
    def forward_batch(self, images, max_batch=1):
        return images


def test_hybrid_geometry_forward_backward_and_cached_generation(monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    # The installed optional FLA package chooses CUDA kernels even on CPU.
    # Exercise the actual native torch reference, not mocked model outputs.
    for name in ('chunk_gated_delta_rule','fused_recurrent_gated_delta_rule',
                 'causal_conv1d_fn','causal_conv1d_update','FusedRMSNormGated'):
        monkeypatch.setattr(implementation,name,None)
    torch.set_num_threads(2)
    config=Qwen3_5Config(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,
        linear_value_head_dim=8,layer_types=['linear_attention']*3+['full_attention'],
        max_position_embeddings=128,rope_parameters=dict(rope_type='default',rope_theta=10000,
        partial_rotary_factor=1.,mrope_section=[1,1,2])),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
            patch_size=2,temporal_patch_size=2,out_hidden_size=32,spatial_merge_size=2),
        image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63,
        pad_token_id=0,eos_token_id=1)
    model=Qwen35GeometryMatrix(config)
    model.geometry_backbone=Features()
    model.geometry_adapter=nn.Linear(32,32)
    inputs=dict(input_ids=torch.tensor([[2,3,0,0,4]]),attention_mask=torch.ones(1,5,dtype=torch.long),
        mm_token_type_ids=torch.zeros(1,5,dtype=torch.long),
        geometry_images=[torch.randn(2,32)],geometry_positions=torch.tensor([[2,3]]))
    model(**inputs,use_cache=False).logits.square().mean().backward()
    assert model.geometry_adapter.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad(): output=model.generate(**inputs,max_new_tokens=2,do_sample=False)
    assert 6<=output.shape[1]<=7
