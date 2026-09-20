"""Geometry-preserving HF policy for GSPO; not a vLLM/VERL registration.

The fixed SFT model is shared by actor/reference. Disabling PEFT restores both
the original language weights and the original saved geometry interface.
"""
from contextlib import contextmanager
from pathlib import Path
import json

import torch


STRUCTURED_INSTRUCTION = '''Reason from the supplied visual evidence. Do not invent timestamps or measurements.
Use exactly this structure, filling every field with your own reasoning:
<think>
Spatial Observation: relevant entities, views, and spatial states.
Spatial Transition: camera/object motion, visibility or relation changes; for static views, explain the viewpoint change.
Answer Derivation: combine the observations and transitions to resolve the question.
</think>
<answer>one final option letter, or a numeric value only without unit text, expressed in the requested units</answer>'''


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
    if adapter_path:
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
    base = policy.get_base_model()
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
    backbone=policy.get_base_model().geometry_backbone
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
