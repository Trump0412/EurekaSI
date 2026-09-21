"""Geometry-preserving HF policy for GSPO; not a vLLM/VERL registration.

Legacy LoRA shares a fixed SFT model and disables PEFT for reference inference.
Explicit full-language RFT instead loads separate immutable SFT language and
interface weights. Only the frozen RGB/VGGT trunks may be shared in that mode.
"""
from contextlib import contextmanager
from pathlib import Path
import json

import torch


FULL_SCOPE = 'language_full_geometry'


def policy_base(policy):
    return policy.get_base_model() if hasattr(policy, 'get_base_model') else policy


def configure_full_policy(base):
    """All language parameters and used interface parameters; no visual updates."""
    if any('lora_' in name for name,_ in base.named_parameters()):
        raise ValueError('Full-language RFT cannot contain LoRA')
    base.requires_grad_(False)
    base.model.language_model.requires_grad_(True)
    base.lm_head.requires_grad_(True)
    base.geometry_adapter.requires_grad_(True)
    for name,p in base.named_parameters():
        if 'null_tokens' in name: p.requires_grad_(False)
        if p.requires_grad: p.data=p.data.float()
    base._rft_policy_scope=FULL_SCOPE
    return base


def policy_checkpoint_name(plan):
    return 'policy' if plan.get('policy_scope') == FULL_SCOPE else 'adapter'


def save_policy(policy, path, plan):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if plan.get('policy_scope') != FULL_SCOPE:
        policy.save_pretrained(path);return
    state={n:p.detach().cpu() for n,p in policy.named_parameters() if p.requires_grad}
    if not state or any('lora_' in n for n in state): raise ValueError('Invalid full-language checkpoint')
    torch.save(state,path/'full_trainable.pt')
    (path/'policy_scope.json').write_text(json.dumps(dict(scope=FULL_SCOPE,
        initial_checkpoint=str(plan['model_checkpoint']),
        frozen_modules=['native_rgb','checkpoint_specific_vggt'],format_version=1)),encoding='utf-8')


def restore_full_policy(base, path, plan):
    meta=json.loads((Path(path)/'policy_scope.json').read_text(encoding='utf-8'))
    if meta.get('scope')!=FULL_SCOPE or Path(meta['initial_checkpoint']).resolve()!=Path(plan['model_checkpoint']).resolve():
        raise ValueError('Full-policy scope or fixed SFT lineage mismatch')
    policy=configure_full_policy(base)
    saved=torch.load(Path(path)/'full_trainable.pt',map_location='cpu',weights_only=True)
    actual={n:p for n,p in policy.named_parameters() if p.requires_grad}
    if set(saved)!=set(actual): raise ValueError('Incomplete full-language/interface checkpoint')
    with torch.no_grad():
        for name,p in actual.items():
            value=saved[name]
            if value.dtype!=p.dtype or value.shape!=p.shape or not torch.isfinite(value).all():
                raise ValueError('Full-policy shape/precision/value mismatch: '+name)
            p.copy_(value)
            if not torch.equal(p.cpu(),value): raise ValueError('Full-policy restore changed values')
    return policy


def load_reference(plan, policy):
    """Separate immutable SFT language/interface; only frozen visual trunks shared."""
    if plan.get('policy_scope') != FULL_SCOPE: return None
    from .qwen3vl_geometry_matrix import load_matrix_model
    from .qwen35_video_compat import install_video_rope_compat
    reference=load_matrix_model(plan['model_checkpoint'],plan['vggt_source'],stage='eval')
    install_video_rope_compat(reference)
    base=policy_base(policy)
    # Shared modules have no trainable state, no dropout and no optimizer owner.
    reference.model.visual=base.model.visual
    reference.geometry_backbone=base.geometry_backbone
    reference.requires_grad_(False).eval()
    if reference.model.language_model is base.model.language_model or reference.geometry_adapter is base.geometry_adapter:
        raise ValueError('Full RFT reference must not share trainable actor parameters')
    return reference


@contextmanager
def reference_context(policy, reference=None):
    if getattr(policy_base(policy),'_rft_policy_scope',None)==FULL_SCOPE:
        if reference is None or reference is policy or any(p.requires_grad for p in reference.parameters()):
            raise ValueError('Full policy requires a distinct frozen SFT reference')
        yield reference
    else:
        if reference is not None: raise ValueError('Unexpected separate legacy reference')
        with policy.disable_adapter(): yield policy


class CPUStateAdamW(torch.optim.AdamW):
    """Explicit synchronous CPU FP32 master/moment offload, not ZeRO sharding.

    Copies one full gradient/weight set per optimizer update. This trades CPU RAM
    and PCIe traffic for GPU memory; actual throughput must pass a runtime gate.
    """
    def __init__(self, parameters, **kwargs):
        self.device_parameters=list(parameters)
        masters=[torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.device_parameters]
        super().__init__(masters,**kwargs)
        self.master_parameters=masters

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise ValueError('CPU offload does not support optimizer closures')
        for device,master in zip(self.device_parameters,self.master_parameters):
            master.grad=None if device.grad is None else device.grad.detach().float().cpu()
            device.grad=None
        result=super().step()
        for device,master in zip(self.device_parameters,self.master_parameters):
            device.copy_(master.to(device=device.device,dtype=device.dtype))
            master.grad=None
        return result

    def zero_grad(self,set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for parameter in self.device_parameters:
            if set_to_none: parameter.grad=None
            elif parameter.grad is not None: parameter.grad.zero_()


PROMPT_VERSION = 'geopsro-original-template-numeric-safe-v2'
# Original GeoPSRO data/formatters.py::psro_prompt instruction. The numeric
# suffix below is an explicit adapter for our strict SpatialLadder interface.
STRUCTURED_INSTRUCTION = '''You should solve the problem using the following format:

<think>
Spatial Observation: write one concise sentence describing the relevant visual-spatial evidence.
Spatial Transition: write one concise sentence describing the key spatial change, state continuity, or multi-frame relation.
Answer Derivation: write one concise sentence explaining how the previous two parts determine the final answer.
</think>
<answer>
Write only the final answer. For multiple-choice questions, write only the option letter.
</answer>

For numeric questions, write a numeric value only without unit text, expressed in the requested units.'''


def add_policy_adapter(base, rank=64, alpha=128):
    from peft import LoraConfig, get_peft_model
    base.requires_grad_(False)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias='none',
        task_type='CAUSAL_LM',
        target_modules=r'.*model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))',
        modules_to_save=['geometry_adapter'])
    policy = get_peft_model(base, config)
    # At LR 1e-6 BF16-only interface updates can round away. Keep trainable
    # state FP32, while frozen base weights and autocast arithmetic stay BF16.
    for name, parameter in policy.named_parameters():
        if 'null_tokens' in name:
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if not any('lora_' in n and p.requires_grad for n,p in policy.named_parameters()):
        raise ValueError('No language LoRA parameters matched')
    if not any('geometry_adapter' in n and p.requires_grad for n,p in policy.named_parameters()):
        raise ValueError('Geometry interface was not made trainable')
    return policy


def restore_policy_adapter(base, adapter_path, trainable=True):
    """Allocate FP32 policy copies BEFORE loading their FP32 learned weights.

    PEFT's generic from_pretrained can copy modules_to_save from a BF16 base,
    then silently round saved FP32 interface weights while loading. Casting
    after that load cannot restore the lost update. The frozen originals must
    remain untouched because they define the fixed SFT reference policy.
    """
    from peft import PeftConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    config=PeftConfig.from_pretrained(adapter_path)
    config.inference_mode=False
    policy=get_peft_model(base,config)
    for name,p in policy.named_parameters():
        if 'null_tokens' in name: p.requires_grad_(False)
        if p.requires_grad: p.data=p.data.float()
    saved=load_peft_weights(str(adapter_path),device='cpu')
    # PEFT remaps keys in place; retain the original serialized namespace for
    # the exact post-load audit rather than comparing against a mutated map.
    result=set_peft_model_state_dict(policy,dict(saved))
    if result.unexpected_keys: raise ValueError(f'Unexpected adapter keys: {result.unexpected_keys}')
    actual=get_peft_model_state_dict(policy)
    if set(saved)!=set(actual):
        raise ValueError(f'Incomplete or foreign saved adapter state: missing={sorted(set(saved)-set(actual))}, extra={sorted(set(actual)-set(saved))}')
    for key,value in saved.items():
        if actual[key].dtype!=value.dtype or not torch.equal(actual[key].detach().cpu(),value.cpu()):
            raise ValueError(f'Adapter load changed learned values/precision: {key}')
    if not trainable: policy.requires_grad_(False)
    return policy


def load_policy(plan, adapter_path=None, trainable=True):
    from .qwen3vl_geometry_matrix import load_matrix_model
    from .qwen35_video_compat import install_video_rope_compat
    base = load_matrix_model(plan['model_checkpoint'], plan['vggt_source'], stage='eval')
    scope=plan.get('policy_scope','language_lora_geometry')
    if scope not in ('language_lora_geometry',FULL_SCOPE): raise ValueError('Unknown RFT policy scope')
    if scope==FULL_SCOPE:
        if plan.get('use_lora') is not False: raise ValueError('Explicit use_lora=false required')
        policy=restore_full_policy(base,adapter_path,plan) if adapter_path else configure_full_policy(base)
    elif adapter_path:
        policy = restore_policy_adapter(base,adapter_path,trainable=trainable)
    else:
        policy = add_policy_adapter(base, plan.get('lora_rank',64), plan.get('lora_alpha',128))
    install_video_rope_compat(base)
    base._matrix_stage = 'rft'
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if not trainable: policy.requires_grad_(False)
    policy.eval()
    return policy


def train_mode(policy):
    policy.train()
    base = policy_base(policy)
    base.model.visual.eval()
    base.geometry_backbone.eval()


def prompt_inputs(processor, row, source, adapter, max_context=16384):
    import numpy as np
    from PIL import Image
    from .qwen35 import Collator
    from .qwen3vl_geometry_matrix import insert_slots, preprocess_geometry
    text = row['question']
    choices = row.get('choices') or row.get('options')
    if choices:
        if not isinstance(choices,dict): raise ValueError('Choices must be a canonical mapping')
        text += '\nOptions:\n'+'\n'.join(f'{k}. {v}' for k,v in choices.items())
    text += '\n'+STRUCTURED_INSTRUCTION
    media = row['media']
    if not media: raise ValueError('Missing visual evidence')
    if row.get('input_mode') == 'video':
        frames=[]
        for path in media:
            with Image.open(path) as im:
                im=im.convert('RGB'); im.thumbnail((448,448),Image.Resampling.LANCZOS)
                frames.append(np.array(im))
        indices=row['frame_indices']; fps=float(row['fps']); total=int(row['total_num_frames'])
        if fps<=0 or len(indices)!=len(frames) or indices!=sorted(set(indices)) or min(indices)<0 or max(indices)>=total:
            raise ValueError('Invalid native video frame/timestamp contract')
        message=[{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}]
        prefix=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        batch=dict(processor(text=[prefix],videos=[np.stack(frames)],
            video_metadata=[{'fps':fps,'total_num_frames':total,'frames_indices':indices}],
            do_sample_frames=False,return_tensors='pt',padding=True))
        if int(batch['video_grid_thw'][0,0])*processor.video_processor.temporal_patch_size!=len(frames):
            raise ValueError('Processor changed prescribed video frames')
    else:
        adjusted=dict(row,question=text,choices=None,instruction='')
        batch=Collator(processor,max_context=max_context,training=False)([adjusted])
    batch=insert_slots(batch,processor,[len(media)],adapter,max_context=max_context)
    batch['geometry_images']=[preprocess_geometry(media,source)]
    return batch


def to_device(batch, device):
    return {k:[x.to(device) for x in v] if isinstance(v,list) else v.to(device) for k,v in batch.items()}


def append_response(prompt, tokens):
    if tokens.ndim!=1 or tokens.numel()==0: raise ValueError('Nonempty single response required')
    result=dict(prompt)
    result['input_ids']=torch.cat((prompt['input_ids'],tokens[None]),dim=1)
    result['attention_mask']=torch.cat((prompt['attention_mask'],torch.ones_like(tokens[None])),dim=1)
    result['mm_token_type_ids']=torch.cat((prompt['mm_token_type_ids'],torch.zeros_like(tokens[None])),dim=1)
    result.pop('position_ids',None); result.pop('cache_position',None)
    return result


def repeat_prompt(prompt, count):
    """Repeat one complete multimodal prompt without dropping geometry inputs."""
    if count<1 or prompt['input_ids'].shape[0]!=1:
        raise ValueError('Repeat requires a single prompt and positive count')
    result={}
    for key,value in prompt.items():
        if key=='geometry_images': result[key]=value*count
        elif isinstance(value,torch.Tensor):
            result[key]=torch.cat([value]*count,dim=0)
        else: raise TypeError(f'Unsupported multimodal repetition field: {key}')
    return result


def sample_group(policy, processor, prompt, group_size=8, micro=1,
                 max_new_tokens=512, temperature=.7, top_p=.9):
    if group_size%micro: raise ValueError('Rollout microbatch must divide logical group')
    policy.eval()
    eos=policy.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [])
    responses=[]
    with torch.no_grad():
        for start in range(0,group_size,micro):
            batch=repeat_prompt(prompt,micro)
            generated=policy.generate(**batch,do_sample=True,temperature=temperature,
                top_p=top_p,top_k=0,max_new_tokens=max_new_tokens,use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id)
            for sequence in generated:
                tokens=sequence[prompt['input_ids'].shape[1]:]
                stop=next((i+1 for i,x in enumerate(tokens.tolist()) if x in eos),len(tokens))
                tokens=tokens[:stop].detach()
                truncated=len(tokens)>=max_new_tokens and (not len(tokens) or int(tokens[-1]) not in eos)
                responses.append({'tokens':tokens,'text':processor.decode(tokens,skip_special_tokens=True),
                                  'truncated':truncated})
    if len(responses)!=group_size: raise ValueError('Incomplete rollout group')
    return responses


def response_logps(policy, prompt, tokens, chunk=64):
    length=prompt['input_ids'].shape[1]
    positions=torch.arange(length-1,length+tokens.numel()-1,device=tokens.device)
    output=policy(**append_response(prompt,tokens),use_cache=False,logits_to_keep=positions)
    logits=output.logits[0]
    if logits.shape[0]!=tokens.numel(): raise ValueError('Completion logit alignment mismatch')
    pieces=[]
    for start in range(0,len(tokens),chunk):
        values=logits[start:start+chunk].float()
        target=tokens[start:start+chunk]
        pieces.append(values.gather(-1,target[:,None]).squeeze(-1)-torch.logsumexp(values,dim=-1))
    return torch.cat(pieces)


@contextmanager
def cached_geometry(policy, prompt):
    """Cache only this prompt's frozen, checkpoint-specific VGGT features.

    No persistent cross-checkpoint cache and no native RGB feature replacement.
    Identity is checked to prevent reuse for a different frame tensor.
    """
    backbone=policy_base(policy).geometry_backbone
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Trainable VGGT cannot use the detached RFT feature cache')
    images=prompt['geometry_images']
    if len(images)!=1: raise ValueError('One prompt per rollout microbatch')
    original=backbone.forward
    with torch.no_grad(): features=original(images[0]).detach()
    def forward(value):
        if value is not images[0]: raise ValueError('Geometry cache prompt identity mismatch')
        return features
    backbone.forward=forward
    try: yield features
    finally: backbone.forward=original


def frozen_inventory(policy):
    return {'trainable':[n for n,p in policy.named_parameters() if p.requires_grad],
            'trainable_parameters':sum(p.numel() for p in policy.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in policy.parameters())}
