"""Forced-null optimization: identical slots and actual tiny-Qwen logits."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen3VLConfig
from spatial_intelligence.qwen3vl_geometry import Qwen3VLGeometry, add_geometry_slots


def helper():
    path=Path(__file__).resolve().parents[1]/'scripts/run-qwen3vl-geometry.py'
    spec=importlib.util.spec_from_file_location('null_eval_entry',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.null_geometry_features


def test_placeholder_preserves_extractor_shape():
    x=helper()(3)
    assert x.shape==(3,1024,2048)
    assert x.dtype==torch.bfloat16 and torch.isfinite(x).all()
    assert torch.count_nonzero(x)==0


def test_null_logits_exact_and_layout_unchanged():
    torch.manual_seed(41)
    config=Qwen3VLConfig(
        text_config=dict(vocab_size=48,hidden_size=32,intermediate_size=48,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            head_dim=8,max_position_embeddings=256,pad_token_id=0,
            rope_parameters={'rope_type':'default','rope_theta':10000.,
                             'mrope_section':[1,1,2],'mrope_interleaved':True}),
        vision_config=dict(depth=2,hidden_size=16,intermediate_size=32,
            num_heads=2,in_channels=3,patch_size=2,spatial_merge_size=2,
            temporal_patch_size=1,out_hidden_size=32,num_position_embeddings=16,
            deepstack_visual_indexes=[0,1]),
        image_token_id=6,video_token_id=7,vision_start_token_id=4,
        vision_end_token_id=5,pad_token_id=0,eos_token_id=2)
    config._attn_implementation='sdpa'
    config.geometry_interface=dict(input_dim=8,latent_dim=8,output_dim=32,
                                   num_tokens=64,num_heads=2,dropout=.2)
    model=Qwen3VLGeometry(config).eval()
    ids=torch.tensor([[1,4,6,5,12,13,2]])
    native=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        mm_token_type_ids=torch.tensor([[0,0,1,0,0,0,0]]),
        pixel_values=torch.randn(4,12),image_grid_thw=torch.tensor([[1,2,2]]))
    processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0,convert_tokens_to_ids=lambda _:5))
    random_features=torch.randn(1,2,3,8)
    zeros=helper()(2,input_dim=8,patches=3)[None]
    mask=torch.zeros((1,2,3),dtype=torch.bool)
    before=add_geometry_slots(native,processor,random_features,mask,force_null=True)
    after=add_geometry_slots(native,processor,zeros,mask,force_null=True)
    for key in ['input_ids','attention_mask','mm_token_type_ids','geometry_positions','geometry_force_null']:
        assert torch.equal(before[key],after[key])
    with torch.no_grad():
        expected=model(**before,use_cache=False).logits
        actual=model(**after,use_cache=False).logits
    assert torch.equal(expected,actual), (expected-actual).abs().max().item()
    # Dropout is random only during training; the explicit intervention is
    # independent of these values and must equal the learned-null token bank.
    adapter=model.geometry_adapter
    force=torch.ones(1,dtype=torch.bool)
    for training in [False,True]:
        adapter.train(training)
        with torch.no_grad():
            assert torch.equal(adapter(random_features,mask,force_null=force),adapter.null_tokens[None])
            assert torch.equal(adapter(zeros,mask,force_null=force),adapter.null_tokens[None])
