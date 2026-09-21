import pytest
import torch
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from spatial_intelligence.georoute import (
    install_georoute,configure_stage,build_graph,GeoRouteQwen3VLForConditionalGeneration,
    make_tip_intervention,
)


@pytest.fixture
def model():
    torch.set_num_threads(4);torch.manual_seed(41)
    config=Qwen3VLConfig(text_config=dict(vocab_size=48,hidden_size=32,intermediate_size=48,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=256,pad_token_id=0,rope_parameters={'rope_type':'default','rope_theta':10000.,
            'mrope_section':[1,1,2],'mrope_interleaved':True}),
        vision_config=dict(depth=4,hidden_size=16,intermediate_size=32,num_heads=2,in_channels=3,
            patch_size=2,spatial_merge_size=2,temporal_patch_size=1,out_hidden_size=32,
            num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
        image_token_id=6,video_token_id=7,vision_start_token_id=4,vision_end_token_id=5,pad_token_id=0,eos_token_id=2)
    config._attn_implementation='sdpa'
    return GeoRouteQwen3VLForConditionalGeneration(config).eval()


def batch():
    ids=torch.tensor([[1,4,6,5,4,6,5,12,13,2]])
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        mm_token_type_ids=torch.tensor([[0,0,1,0,0,1,0,0,0,0]]),
        labels=torch.tensor([[-100]*8+[13,2]]),pixel_values=torch.randn(8,12),
        image_grid_thw=torch.tensor([[1,2,2],[1,2,2]]))


def graph():
    return build_graph([0]*4+[1]*4,[0,1],[4,5],[1.,1.])


def test_all_four_branch_only_exits_identity_then_nonzero_sft_gradients(model):
    x=batch();g=graph()
    with torch.no_grad(): baseline=model(**x,use_cache=False).logits.clone()
    route=install_georoute(model,dict(exits=[0,1,2,3],bottleneck=8))
    with route.graph_context(g,capture=True) as states,torch.no_grad():
        actual=model(**x,use_cache=False).logits
    assert len(states)==4 and torch.equal(actual,baseline)
    original={i:h.clone() for i,h in states.items()}
    for block in route.routes:
        for stb in block: stb.alpha.data.fill_(.1)
    with route.graph_context(g,capture=True) as routed,torch.no_grad():
        changed=model(**x,use_cache=False).logits
    assert all(torch.equal(original[i],routed[i]) for i in range(4)), 'Intermediate branch must not mutate visual main stream'
    assert not torch.equal(changed,baseline)
    configure_stage(model,'sft')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with route.graph_context(g): model(**x,use_cache=False).loss.backward()
    optimizer.step()
    changed_names=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
    assert any('language_model' in n for n in changed_names)
    assert any('model.visual.blocks' in n for n in changed_names)
    for i in range(4): assert any(f'georoute.routes.{i}.' in n for n in changed_names)
    assert all(p.requires_grad for p in model.parameters())


def test_tip_capture_stage_generation_and_checkpoint_reload(model,tmp_path):
    x=batch();g=graph();route=install_georoute(model,dict(exits=[0,1,2,3],bottleneck=8,alpha_init=.1))
    configure_stage(model,'tip')
    with route.graph_context(g,capture=True) as states,torch.no_grad(): model(**x,use_cache=False)
    mask,bad=make_tip_intervention(g,generator=torch.Generator().manual_seed(5))
    loss,_=route.tip_loss([states[i] for i in range(4)],g,mask,bad)
    loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('georoute.'))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in route.parameters())
    model.zero_grad(set_to_none=True)
    with route.graph_context(g):
        result=model(**x,route_tip=True,tip_mask=mask,tip_substituted_graph=bad)
        result.loss.backward()
    assert result.logits is None and torch.isfinite(result.loss)
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('georoute.'))
    model.eval();x.pop('labels')
    with route.graph_context(g),torch.no_grad():
        expected=model(**x,use_cache=False).logits
        generated=model.generate(**x,max_new_tokens=2,min_new_tokens=2,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
    assert generated.shape[1]==x['input_ids'].shape[1]+2
    model.save_pretrained(tmp_path/'model')
    restored=GeoRouteQwen3VLForConditionalGeneration.from_pretrained(tmp_path/'model',attn_implementation='sdpa').eval()
    with restored.georoute.graph_context(g),torch.no_grad(): actual=restored(**x,use_cache=False).logits
    assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-5)
    assert set(route.state_dict())==set(restored.georoute.state_dict())
    with pytest.raises(ValueError,match='explicit graph'): restored(**x,use_cache=False)


@pytest.mark.parametrize('settings',[{'blocks_per_exit':1},{'active_exits':[3]},{'placement':'post'}])
def test_declared_structural_ablation_keeps_native_output_contract(model,settings):
    x=batch();g=graph()
    with torch.no_grad(): baseline=model(**x,use_cache=False).logits
    route=install_georoute(model,dict(exits=[0,1,2,3],bottleneck=8,**settings))
    with route.graph_context(g),torch.no_grad(): actual=model(**x,use_cache=False).logits
    assert torch.equal(actual,baseline)
    configure_stage(model,'sft')
    with route.graph_context(g): model(**x,use_cache=False).loss.backward()
    assert all(p.grad is not None for p in route.parameters())


def test_bfloat16_save_reload_preserves_routing_and_native_rope(model,tmp_path):
    route=install_georoute(model,dict(exits=[0,1,2,3],bottleneck=8,alpha_init=.1))
    before={n:b.clone() for n,b in model.named_buffers() if n.endswith('inv_freq')}
    model.bfloat16();x=batch();x['pixel_values']=x['pixel_values'].bfloat16();g=graph()
    for name,value in before.items():
        assert dict(model.named_buffers())[name].dtype==torch.float32
        assert torch.equal(value,dict(model.named_buffers())[name])
    with route.graph_context(g),torch.no_grad(): expected=model(**x,use_cache=False).logits
    model.save_pretrained(tmp_path/'bf16')
    restored=GeoRouteQwen3VLForConditionalGeneration.from_pretrained(tmp_path/'bf16',dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    with restored.georoute.graph_context(g),torch.no_grad(): actual=restored(**x,use_cache=False).logits
    assert torch.equal(expected,actual)


def test_post_merger_tip_uses_coarsened_destination_mask(model):
    route=install_georoute(model,dict(exits=[0,1,2,3],bottleneck=8,alpha_init=.1,placement='post'))
    configure_stage(model,'tip')
    g=build_graph([0]*16+[1]*16,[0,1],[16,17],[1.,1.])
    with route.graph_context(g):
        assert route._post_graph.num_nodes==8
        mask,bad=make_tip_intervention(route._post_graph,generator=torch.Generator().manual_seed(4))
        result=model(route_tip=True,pixel_values=torch.randn(32,12),image_grid_thw=torch.tensor([[1,4,4],[1,4,4]]),
                     tip_mask=mask,tip_substituted_graph=bad)
        result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.parameters()) for branch in route.routes)
