"""Tiny actual Qwen3-VL integration; not a full released-model GPU gate."""
import pytest
import torch
from transformers import Qwen3VLConfig
from spatial_intelligence.geofits_model import GeoFitsQwen3VLForConditionalGeneration,load_geofits_model
from spatial_intelligence.geofits_recipe import architecture_for_variant,VARIANTS


def setup(variant='full'):
    torch.set_num_threads(4);torch.manual_seed(123)
    config=Qwen3VLConfig(text_config=dict(vocab_size=48,hidden_size=32,intermediate_size=48,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=256,pad_token_id=0,rope_parameters={'rope_type':'default','rope_theta':10000.,
        'mrope_section':[1,1,2],'mrope_interleaved':True}),
        vision_config=dict(depth=4,hidden_size=16,intermediate_size=32,num_heads=2,in_channels=3,
        patch_size=2,spatial_merge_size=2,temporal_patch_size=1,out_hidden_size=32,
        num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
        image_token_id=6,video_token_id=7,vision_start_token_id=4,vision_end_token_id=5,pad_token_id=0,eos_token_id=2)
    config._attn_implementation='sdpa'
    config.geofits_config=dict(hidden_size=32,vggt_width=4,pi3_width=6,temporal_bottleneck=4,
        temporal_group_size=1,temporal_group_reduction='mean',timestamp_encoding='order_only_sincos',
        pooling='average_2x2',retrieval_width=8)
    config.geofits_config=architecture_for_variant(config.geofits_config,variant)
    model=GeoFitsQwen3VLForConditionalGeneration(config)
    ids=torch.tensor([[1,4,6,5,4,6,5,12,13,2]])
    inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        mm_token_type_ids=torch.tensor([[0,0,1,0,0,1,0,0,0,0]]),
        labels=torch.tensor([[-100]*8+[13,2]]),pixel_values=torch.randn(8,12),
        image_grid_thw=torch.tensor([[1,2,2],[1,2,2]]))
    features=[dict(vggt={n:torch.randn(1,2,2,2,4) for n in (11,17,23)},
        pi3={n:torch.randn(1,2,2,2,6) for n in (17,26,35)},native_grid=(2,1,1),
        visual_indices=torch.tensor([2,5]),prefix_length=8,labels=inputs['labels'][0])]
    if variant=='3d_only': features[0]['pi3']={}
    if variant=='4d_only': features[0]['vggt']={}
    return model,inputs,features


@pytest.mark.parametrize('variant',VARIANTS)
def test_all_native_and_fusion_parameters_update_with_checkpointing(variant):
    model,inputs,features=setup(variant);model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    with model.feature_context(features):
        output=model(**inputs,use_cache=False);output.loss.backward()
    optimizer.step()
    changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
    groups=['language_model','visual.blocks','geofits.bank.projectors','geofits.layers']
    if variant!='3d_only': groups.append('geofits.bank.temporal')
    for group in groups:
        assert any(group in n for n in changed),group
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('variant',VARIANTS)
def test_answer_blind_prefix_and_saved_generation(tmp_path,variant):
    model,inputs,features=setup(variant);model.eval()
    with model.feature_context(features),torch.no_grad(): expected=model(**inputs,use_cache=False).logits
    altered=dict(inputs);altered['input_ids']=inputs['input_ids'].clone();altered['input_ids'][0,8:]=torch.tensor([20,21])
    with model.feature_context(features),torch.no_grad(): actual=model(**altered,use_cache=False).logits
    assert torch.equal(expected[:,:8],actual[:,:8])
    model.save_pretrained(tmp_path/'model')
    restored=load_geofits_model(tmp_path/'model',attn_implementation='sdpa').eval()
    with restored.feature_context(features),torch.no_grad():
        reloaded=restored(**inputs,use_cache=False).logits
    assert torch.allclose(expected,reloaded,atol=1e-6)
    prompt={k:v for k,v in inputs.items() if k!='labels'}
    for key in ('input_ids','attention_mask','mm_token_type_ids'): prompt[key]=prompt[key][:,:8]
    features[0].pop('labels')
    with restored.feature_context(features),torch.no_grad():
        cached=restored(**prompt,use_cache=True)
        positions,_=restored.model.get_rope_index(inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],
            attention_mask=inputs['attention_mask'],mm_token_type_ids=inputs['mm_token_type_ids'])
        decoded=restored(input_ids=inputs['input_ids'][:,8:9],attention_mask=torch.ones(1,9,dtype=torch.long),
            position_ids=positions[:,:,8:9],mm_token_type_ids=torch.zeros(1,1,dtype=torch.long),
            past_key_values=cached.past_key_values,use_cache=True)
        assert torch.allclose(decoded.logits[:,0],expected[:,8],atol=1e-5)
        output=restored.generate(**prompt,max_new_tokens=2,min_new_tokens=2,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
    assert output.shape[1]==10
    with pytest.raises(ValueError,match='feature_context'): restored(**prompt)


def test_bf16_roundtrip_and_invalid_layout(tmp_path):
    model,inputs,features=setup();model.bfloat16().eval();inputs['pixel_values']=inputs['pixel_values'].bfloat16()
    with model.feature_context(features),torch.no_grad(): expected=model(**inputs,use_cache=False).logits
    model.save_pretrained(tmp_path/'bf16')
    restored=load_geofits_model(tmp_path/'bf16',dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    with restored.feature_context(features),torch.no_grad(): actual=restored(**inputs,use_cache=False).logits
    assert torch.equal(expected,actual)
    features[0]['visual_indices']=torch.tensor([2,9])
    with restored.feature_context(features),pytest.raises(ValueError,match='precede'): restored(**inputs,use_cache=False)
