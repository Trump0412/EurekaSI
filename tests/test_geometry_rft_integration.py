"""Real tiny Qwen + PEFT, fake registered VGGT; CPU only, no accuracy claims."""
from types import SimpleNamespace
import random
import pytest
import torch

from test_geometry_matrix_integration import model, inputs
from spatial_intelligence.geometry_rft import (
    add_policy_adapter, train_mode, repeat_prompt, response_logps, cached_geometry,
    sample_group, append_response, restore_policy_adapter,
)
from spatial_intelligence.qwen3vl_geometry_matrix import Qwen3VLGeometryMatrix, insert_slots
from spatial_intelligence.qwen35_video_compat import install_video_rope_compat


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def make_policy(model):
    model.configure_stage('eval')
    policy = add_policy_adapter(model, rank=4, alpha=8)
    model._matrix_stage = 'rft'
    policy.eval()
    return policy


def prompt():
    batch = inputs()
    batch.pop('labels')
    return batch


def test_peft_scope_reference_restoration_and_real_update(model):
    batch = prompt(); tokens = torch.tensor([13, 2])
    model.configure_stage('eval')
    with torch.no_grad():
        native = model(**batch, use_cache=False).logits.clone()
    policy = make_policy(model)
    names = [n for n,p in policy.named_parameters() if p.requires_grad]
    assert any('lora_B' in n for n in names)
    assert any('geometry_adapter.modules_to_save.default' in n for n in names)
    assert not any('geometry_backbone' in n or '.visual.' in n or 'null_tokens' in n for n in names)
    assert all('language_model' in n or 'geometry_adapter' in n for n in names)
    with torch.no_grad():
        assert torch.allclose(policy(**batch,use_cache=False).logits, native, atol=1e-6)
        with policy.disable_adapter():
            assert torch.allclose(policy(**batch,use_cache=False).logits, native, atol=1e-6)
    before = {n:p.detach().clone() for n,p in policy.named_parameters()}
    optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad],lr=.1)
    train_mode(policy)
    assert not model.model.visual.training and not model.geometry_backbone.training
    loss = -response_logps(policy,batch,tokens).mean()
    loss.backward(); optimizer.step(); policy.eval()
    changed = [n for n,p in policy.named_parameters() if not torch.equal(p,before[n])]
    assert any('lora_B' in n for n in changed)
    assert any('geometry_adapter.modules_to_save.default' in n for n in changed)
    assert all('lora_' in n or 'geometry_adapter.modules_to_save.default' in n for n in changed)
    with torch.no_grad(), policy.disable_adapter():
        assert torch.allclose(policy(**batch,use_cache=False).logits, native, atol=1e-6)


@pytest.mark.parametrize('images',[1,2,4,8])
def test_multimodal_group_repetition_and_cache(model, images):
    ids=torch.tensor([[1]+[4,6,5]*images+[12]])
    types=torch.tensor([[0]+[0,1,0]*images+[0]])
    batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=types,
        pixel_values=torch.randn(4*images,12),image_grid_thw=torch.tensor([[1,2,2]]*images))
    processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0,convert_tokens_to_ids=lambda _:5))
    batch=insert_slots(batch,processor,[images],'query64',max_context=2048)
    batch['geometry_images']=[torch.randn(images,3,4,4)]
    policy=make_policy(model)
    repeated=repeat_prompt(batch,2)
    assert repeated['input_ids'].shape[0]==2
    assert repeated['pixel_values'].shape[0]==8*images
    assert repeated['image_grid_thw'].shape[0]==2*images
    assert repeated['geometry_images'][0] is repeated['geometry_images'][1]
    calls=[]
    hook=model.geometry_backbone.register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad():
        uncached=policy(**batch,use_cache=False).logits
        with cached_geometry(policy,batch):
            cached=policy(**batch,use_cache=False).logits
            grouped=policy(**repeated,use_cache=False).logits
            other=dict(batch,geometry_images=[batch['geometry_images'][0].clone()])
            with pytest.raises(ValueError,match='identity'):
                policy(**other,use_cache=False)
        restored=policy(**batch,use_cache=False).logits
    hook.remove()
    assert torch.allclose(uncached,cached,atol=1e-6)
    assert torch.allclose(grouped[0],uncached[0],atol=1e-6)
    assert torch.allclose(grouped[1],uncached[0],atol=1e-6)
    assert torch.allclose(restored,uncached,atol=1e-6)


def test_save_reload_policy_and_original_reference(model,tmp_path):
    model.configure_stage('eval'); model.save_pretrained(tmp_path/'base')
    policy=make_policy(model); batch=prompt(); tokens=torch.tensor([13,2])
    with torch.no_grad(),policy.disable_adapter():
        reference=response_logps(policy,batch,tokens).clone()
    optimizer=torch.optim.SGD([p for p in policy.parameters() if p.requires_grad],lr=.1)
    train_mode(policy)
    (-response_logps(policy,batch,tokens).mean()).backward(); optimizer.step(); policy.eval()
    with torch.no_grad(): expected=response_logps(policy,batch,tokens).clone()
    policy.save_pretrained(tmp_path/'adapter')
    base=Qwen3VLGeometryMatrix.from_pretrained(tmp_path/'base',attn_implementation='sdpa').eval()
    restored=restore_policy_adapter(base,tmp_path/'adapter',trainable=False).eval()
    with torch.no_grad():
        actual=response_logps(restored,batch,tokens)
        with restored.disable_adapter(): actual_reference=response_logps(restored,batch,tokens)
    assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-5)
    assert torch.allclose(reference,actual_reference,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('micro',[1,2,4,8])
def test_actual_grouped_generation_preserves_geometry(model,micro):
    policy=make_policy(model); batch=prompt()
    processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0),
                              decode=lambda tokens,skip_special_tokens: ','.join(map(str,tokens.tolist())))
    projection_calls=[]
    hook=model.geometry_backbone.projection.register_forward_hook(lambda *args:projection_calls.append(1))
    try:
        with cached_geometry(policy,batch):
            output=sample_group(policy,processor,batch,group_size=8,micro=micro,max_new_tokens=2)
    finally:
        hook.remove()
    assert len(output)==8
    assert len(projection_calls)==1, 'Frozen geometry must be computed once for the complete group'
    assert all(0<len(row['tokens'])<=2 for row in output)
    assert all(isinstance(row['truncated'],bool) for row in output)


def test_response_logps_causal_alignment_and_cache_guard(model):
    policy=make_policy(model); batch=prompt(); tokens=torch.tensor([13,2])
    with torch.no_grad():
        expected=policy(**append_response(batch,tokens),use_cache=False).logits
        start=batch['input_ids'].shape[1]-1
        expected=expected[0,start:start+2].log_softmax(-1).gather(-1,tokens[:,None]).squeeze(-1)
        actual=response_logps(policy,batch,tokens,chunk=1)
    assert torch.allclose(actual,expected,atol=1e-6)
    model.geometry_backbone.requires_grad_(True)
    with pytest.raises(ValueError,match='Trainable VGGT'):
        with cached_geometry(policy,batch): pass


def test_optimizer_scheduler_rng_resume_matches_next_real_update(model,tmp_path):
    model.configure_stage('eval'); model.save_pretrained(tmp_path/'base')
    policy=make_policy(model); batch=prompt(); tokens=torch.tensor([13,2])
    params=lambda p:[x for x in p.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(params(policy),lr=.001,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.5)
    def update(p,opt,sched):
        train_mode(p); opt.zero_grad(set_to_none=True)
        factor=1.+torch.rand(()).item()+random.random()
        loss=-response_logps(p,batch,tokens).mean()*factor
        loss.backward(); opt.step(); sched.step()
        return loss.detach()
    torch.manual_seed(71); random.seed(71)
    update(policy,optimizer,scheduler)
    saved_parameters={n:p.detach().clone() for n,p in policy.named_parameters() if p.requires_grad}
    policy.save_pretrained(tmp_path/'adapter')
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()},tmp_path/'state.pt')
    expected_loss=update(policy,optimizer,scheduler)
    expected={n:p.detach().clone() for n,p in policy.named_parameters() if p.requires_grad}
    base=Qwen3VLGeometryMatrix.from_pretrained(tmp_path/'base',attn_implementation='sdpa').eval()
    restored=restore_policy_adapter(base,tmp_path/'adapter',trainable=True)
    for name,p in restored.named_parameters():
        if 'null_tokens' in name: p.requires_grad_(False)
    opt2=torch.optim.AdamW(params(restored),lr=.001,weight_decay=0.)
    sched2=torch.optim.lr_scheduler.StepLR(opt2,step_size=1,gamma=.5)
    state=torch.load(tmp_path/'state.pt',weights_only=False)
    opt2.load_state_dict(state['optimizer']); sched2.load_state_dict(state['scheduler'])
    for name,p in restored.named_parameters():
        if p.requires_grad: assert torch.equal(saved_parameters[name],p),name
    for actual,expected_state in zip(opt2.state_dict()['state'].values(),state['optimizer']['state'].values()):
        for key,value in actual.items():
            assert torch.equal(value,expected_state[key])
    torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
    actual_loss=update(restored,opt2,sched2)
    assert torch.equal(expected_loss,actual_loss)
    assert optimizer.param_groups[0]['lr']==opt2.param_groups[0]['lr']
    differences={name:float((expected[name]-p).abs().max())
                 for name,p in restored.named_parameters() if p.requires_grad}
    # Saved state/RNG are exact; multithreaded CPU backward after reconstruction
    # may differ by FP32 roundoff (observed maximum 5.96e-8), not bitwise parity.
    assert max(differences.values())<1e-7, sorted(differences.items(),key=lambda item:item[1],reverse=True)[:5]


def test_bfloat16_base_keeps_sub_ulp_fp32_interface_update_on_peft_reload(model,tmp_path):
    model.configure_stage('eval'); model.bfloat16(); model.save_pretrained(tmp_path/'base')
    policy=make_policy(model)
    name,p=next((n,p) for n,p in policy.named_parameters()
                if 'geometry_adapter.modules_to_save.default' in n and p.requires_grad and p.ndim>=2)
    assert p.dtype==torch.float32
    with torch.no_grad(): p.flatten()[0].add_(1e-7)
    expected=p.detach().clone()
    assert not torch.equal(expected,expected.bfloat16().float()), 'Probe must detect lost sub-BF16 precision'
    policy.save_pretrained(tmp_path/'adapter')
    base=Qwen3VLGeometryMatrix.from_pretrained(tmp_path/'base',dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    original={n:p.detach().clone() for n,p in base.geometry_adapter.named_parameters()}
    restored=restore_policy_adapter(base,tmp_path/'adapter',trainable=True)
    actual=dict(restored.named_parameters())[name]
    assert actual.dtype==torch.float32
    assert torch.equal(expected,actual),(name,float((expected-actual).abs().max()))
    for name,p in restored.get_base_model().geometry_adapter.original_module.named_parameters():
        assert p.dtype==torch.bfloat16
        assert torch.equal(original[name],p)


def test_native_video_grouped_generation_and_logps(model):
    # Two timestamp-separated temporal groups, while the vision tower receives
    # the UNSPLIT [T,H,W] video grid and ordered frame geometry.
    ids=torch.tensor([[1,11,4,7,5,12,4,7,5,13]])
    types=torch.tensor([[0,0,0,2,0,0,0,2,0,0]])
    batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=types,
               pixel_values_videos=torch.randn(8,12),video_grid_thw=torch.tensor([[2,2,2]]))
    processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0,convert_tokens_to_ids=lambda _:5),
                              decode=lambda tokens,skip_special_tokens: ','.join(map(str,tokens.tolist())))
    batch=insert_slots(batch,processor,[2],'query64',max_context=2048)
    batch['geometry_images']=[torch.randn(2,3,4,4)]
    policy=make_policy(model); install_video_rope_compat(model)
    repeated=repeat_prompt(batch,2)
    assert repeated['video_grid_thw'].tolist()==[[2,2,2],[2,2,2]]
    assert repeated['pixel_values_videos'].shape==(16,12)
    with cached_geometry(policy,batch),torch.no_grad():
        grouped=policy(**repeated,use_cache=False).logits
        single=policy(**batch,use_cache=False).logits
        output=sample_group(policy,processor,batch,group_size=8,micro=2,max_new_tokens=2)
        logps=response_logps(policy,batch,output[0]['tokens'])
    assert torch.allclose(grouped[0],single[0],atol=1e-6)
    assert torch.allclose(grouped[1],single[0],atol=1e-6)
    assert len(output)==8 and torch.isfinite(logps).all()
