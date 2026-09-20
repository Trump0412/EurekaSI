"""One isolated RoboRefer-style geometry stage or its paired evaluation.

New experiments only: never mutate the original direct-fusion training recipe.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--processor', required=True)
    p.add_argument('--vggt-source', required=True); p.add_argument('--vggt-weights', required=True)
    p.add_argument('--adapter', choices=['query64', 'downsample'], required=True)
    p.add_argument('--stage', choices=['align', 'sft'], default='align')
    p.add_argument('--train-vggt', action='store_true'); p.add_argument('--name', required=True)
    p.add_argument('--micro', type=int, default=1); p.add_argument('--global-batch', type=int, default=448)
    p.add_argument('--max-steps', type=int, default=-1); p.add_argument('--manifest', type=Path)
    p.add_argument('--profile', action='store_true'); p.add_argument('--deepspeed')
    p.add_argument('--verify-reload', action='store_true')
    p.add_argument('--parity', action='store_true', help='Disposable full-gradient micro1 versus micro2/4 comparison')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--benchmark', choices=['revsi', 'vsibench'])
    p.add_argument('--smoke-per-type', type=int, default=0); p.add_argument('--merge', action='store_true')
    return p.parse_args()


def parity(a):
    import math
    import torch
    from transformers import AutoProcessor, set_seed
    from spatial_intelligence.qwen3vl_geometry_matrix import load_matrix_model, MatrixCollator
    from spatial_intelligence.qwen35 import completion_loss
    from spatial_intelligence.study import read_rows, dump
    if a.micro not in (2,4) or a.manifest is None:
        raise ValueError('Parity requires micro2/4 and an explicit real-data manifest')
    if int(os.environ.get('WORLD_SIZE',1)) != 1:
        raise ValueError('Numerical parity is a single-GPU diagnostic before DDP timing')
    torch.cuda.set_device(0); torch.set_num_threads(4); set_seed(3407)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    rows=read_rows(a.manifest)[:4]
    if len(rows)!=4: raise ValueError('Four fixed real samples required')
    originals={r['id']:r for r in read_rows(a.root/'manifests/sft.train.jsonl')}
    if any(originals.get(r['id'])!=r for r in rows): raise ValueError('Changed diagnostic examples')
    model=load_matrix_model(a.model,a.vggt_source,a.vggt_weights,a.adapter,a.stage,a.train_vggt).cuda()
    collator=MatrixCollator(AutoProcessor.from_pretrained(a.processor),a.vggt_source,a.adapter)
    losses=[]; reference={}; numerator=denominator=0.
    for micro in (1,a.micro):
        model.zero_grad(set_to_none=True); set_seed(3407); total=0.
        for start in range(0,len(rows),micro):
            batch=collator(rows[start:start+micro])
            batch={k:[t.cuda() for t in v] if isinstance(v,list) else v.cuda() for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,_=completion_loss(model,batch,reduction='sample_mean')
            scaled=loss*(micro/len(rows)); scaled.backward(); total+=float(scaled.detach())
        losses.append(total)
        for name,p in model.named_parameters():
            if not p.requires_grad: continue
            grad=p.grad.detach().float().cpu() if p.grad is not None else None
            if grad is not None and not torch.isfinite(grad).all(): raise ValueError('Nonfinite gradient')
            if micro==1: reference[name]=grad
            else:
                ref=reference.pop(name)
                if (ref is None)!=(grad is None): raise ValueError('Gradient connectivity changed')
                if ref is not None:
                    numerator+=float((grad-ref).double().square().sum())
                    denominator+=float(ref.double().square().sum())
        gc.collect()
    relative=math.sqrt(numerator/max(denominator,1e-30))
    loss_delta=abs(losses[1]-losses[0])/max(abs(losses[0]),1e-12)
    accepted=relative<.05 and loss_delta<.01 and denominator>0
    receipt=dict(status='accepted' if accepted else 'rejected',micro=a.micro,
        loss_micro1=losses[0],loss_candidate=losses[1],loss_relative_delta=loss_delta,
        full_gradient_relative_l2=relative,ids=[r['id'] for r in rows],
        scope='single GPU numerical gate; throughput and long-sample fit are separate')
    dump(a.root/'runs'/a.name/'parity.json',receipt)
    print(json.dumps(receipt),flush=True)


def evaluation(a):
    import transformers
    from spatial_intelligence import qwen35
    from spatial_intelligence.qwen3vl_geometry_matrix import load_matrix_model, insert_slots, preprocess_geometry
    def loader(path, **kwargs):
        return load_matrix_model(path, a.vggt_source, adapter=a.adapter, stage='eval')
    qwen35.load_model = loader
    original_processor = transformers.AutoProcessor.from_pretrained
    transformers.AutoProcessor.from_pretrained = classmethod(lambda cls, path, *x, **kw: original_processor(a.processor, *x, **kw))
    spec = importlib.util.spec_from_file_location('matrix_eval', REPO/'scripts/evaluate-spatial.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original_input = module.native_video_inputs
    def native(processor, row, answer_format):
        result = insert_slots(original_input(processor, row, answer_format), processor, [len(row['media'])], a.adapter)
        result['geometry_images'] = preprocess_geometry(row['media'], a.vggt_source)
        return result
    module.native_video_inputs = native
    sys.argv = ['evaluate-spatial.py', '--root', str(a.root), '--model', a.model, '--name', a.name,
                '--benchmark', a.benchmark, '--smoke-per-type', str(a.smoke_per_type)]
    if a.merge: sys.argv.append('--merge')
    module.main()


def verify_reload(a):
    import torch
    from transformers import AutoProcessor
    from spatial_intelligence.qwen3vl_geometry_matrix import load_matrix_model, MatrixCollator
    from spatial_intelligence.study import dump
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    out = a.root/'runs'/a.name
    evidence = torch.load(out/'reload-reference.pt', map_location='cpu', weights_only=False)
    model = load_matrix_model(str(out/'final'), a.vggt_source, adapter=a.adapter).cuda().eval()
    processor = AutoProcessor.from_pretrained(a.processor)
    batch = MatrixCollator(processor, a.vggt_source, a.adapter)([evidence['row']])
    batch = {k: [t.cuda() for t in v] if isinstance(v, list) else v.cuda() for k,v in batch.items()}
    batch.pop('labels')
    if evidence.get('geometry_input_dtype',str(torch.float32)) != str(batch['geometry_images'][0].dtype):
        raise ValueError('Geometry normalization input precision changed on reload')
    for key,attr in (('normalization_mean','_resnet_mean'),('normalization_std','_resnet_std')):
        if key in evidence and not torch.equal(evidence[key],getattr(model.geometry_backbone.aggregator,attr).cpu()):
            raise ValueError('Geometry normalization constants changed on reload')
    actual_buffers=dict(model.named_buffers())
    for name,value in evidence.get('rotary_buffers',{}).items():
        current=actual_buffers[name].cpu()
        if current.dtype!=value.dtype or not torch.equal(value,current):
            raise ValueError('Native rotary frequency changed on reload: '+name)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(**batch, use_cache=False, logits_to_keep=1).logits.float().cpu()
    delta = float((actual - evidence['logits']).abs().max())
    if not torch.allclose(actual, evidence['logits'], atol=.02, rtol=.01):
        raise ValueError(f'Saved model reload mismatch {delta}')
    receipt = json.loads((out/'completion.json').read_text())
    receipt.update(reload_verified=True, reload_max_logit_delta=delta)
    dump(out/'completion.json', receipt)
    print(json.dumps({'reload_verified': True, 'max_logit_delta': delta}), flush=True)


def train(a):
    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from spatial_intelligence.study import read_rows, dump
    from spatial_intelligence.qwen35 import completion_loss
    from spatial_intelligence.qwen3vl_geometry_matrix import load_matrix_model, MatrixCollator
    from spatial_intelligence.throughput_sampler import EffectiveBatchSampler
    from spatial_intelligence.throughput import row_cost
    local, world = int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local); torch.set_num_threads(4); set_seed(3407)
    if a.global_batch % (world*a.micro): raise ValueError('Global batch must divide micro*world')
    if a.max_steps != -1 and not a.profile: raise ValueError('Truncated training must be explicitly diagnostic')
    if a.profile and (not a.name.startswith('diagnostic-') or a.max_steps < 1): raise ValueError('Invalid profile run')
    if a.stage == 'align' and a.train_vggt: raise ValueError('Alignment must freeze VGGT')
    gate = json.loads((a.root/'receipts/train-prepared.json').read_text())
    if gate.get('status') != 'complete' or gate.get('scene_overlap') != 0: raise ValueError('Data gate failed')
    manifest = a.manifest or a.root/'manifests/sft.train.jsonl'
    rows = read_rows(manifest)
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Empty/duplicate manifest')
    if a.manifest:
        if not a.profile: raise ValueError('Custom manifests are diagnostic only')
        originals = {r['id']:r for r in read_rows(a.root/'manifests/sft.train.jsonl')}
        if any(originals.get(r['id']) != r for r in rows): raise ValueError('Diagnostic samples changed')
    class Rows(torch.utils.data.Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i): return rows[i]
    out = a.root/'runs'/a.name
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.processor)
    model = load_matrix_model(a.model, a.vggt_source, a.vggt_weights, a.adapter, a.stage, a.train_vggt)
    ga = a.global_batch // (world*a.micro)
    # Frozen language weights still participate in activation recomputation.
    # ZeRO-3 partitions those nontrainable weights outside recompute hooks in
    # this stack, producing zero-size tensors. Alignment fits replicated DDP;
    # use sharding only for SFT where all language/visual parameters are trained.
    effective_deepspeed=a.deepspeed if a.stage=='sft' else None
    contract = dict(stage=a.stage, adapter=a.adapter, train_vggt=a.train_vggt,
        initial_model=a.model, manifest=str(manifest), rows=len(rows), seed=3407, epochs=1,
        micro=a.micro, world=world, ga=ga, global_batch=a.global_batch,
        layout='balanced', loss='sample_mean', lr=1e-3 if a.stage=='align' else 2e-5,
        weight_decay=0., dropout=0., warmup=.03, deepspeed=effective_deepspeed,
        sharding='zero3' if effective_deepspeed else 'ddp',
        diagnostic=a.profile, max_steps=a.max_steps, version=1)
    path = out/'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract: raise ValueError('Resume contract mismatch')
    last = get_last_checkpoint(str(out))
    if (out/'completion.json').exists(): raise ValueError('Already completed; do not repeat training')
    args = TrainingArguments(output_dir=str(out), num_train_epochs=1, max_steps=a.max_steps,
        per_device_train_batch_size=a.micro, gradient_accumulation_steps=ga,
        learning_rate=contract['lr'], warmup_ratio=.03, lr_scheduler_type='cosine', weight_decay=0.,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False}, dataloader_num_workers=4,
        dataloader_persistent_workers=True, dataloader_pin_memory=True,
        save_steps=100, save_total_limit=2, logging_steps=1, report_to=[],
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        seed=3407, data_seed=3407, optim='adamw_torch_fused', deepspeed=effective_deepspeed)
    class MatrixTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            # HF's DeepSpeed preparation casts *all* floating inputs to bf16.
            # VGGT normalizes RGB before its first projection; preserve the
            # upstream fp32 pixels here, exactly as in standalone inference.
            copied=dict(inputs)
            images=copied.pop('geometry_images',None)
            result=super()._prepare_inputs(copied)
            if images is not None:
                result['geometry_images']=[im.to(self.args.device,dtype=torch.float32,non_blocking=True) for im in images]
            return result
        def _get_train_sampler(self, train_dataset=None):
            return EffectiveBatchSampler(self.train_dataset, [row_cost(r) for r in rows],
                seed=3407, micro_batch=a.micro, world_size=world, accumulation=ga)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = completion_loss(model, inputs, reduction='sample_mean')
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            return (loss, outputs) if return_outputs else loss
    class Telemetry(TrainerCallback):
        def __init__(self):
            self.times=[]; self.previous=None; self.losses=[]; self.updated=False
            self.group_updates={}; self.probes=[]; self.step_probes=[]
            previous=out/'live-eta.json'
            if last and previous.exists():
                prior=json.loads(previous.read_text())
                self.group_updates=prior.get('component_updates',{})
                self.updated=prior.get('nonzero_update',False)
        def on_train_begin(self, args, state, control, **kw):
            self.previous=time.monotonic(); self.started=self.previous
        def on_step_begin(self, args, state, control, model=None, **kw):
            self.step_probes=[]
            # Accelerate's DeepSpeed wrapper may perform engine.step inside
            # backward, before Trainer's pre-optimizer callback. Probe across
            # the whole accumulation step as well, including FP32 master weights.
            for group in ('geometry_adapter','geometry_backbone','model.visual','model.language_model'):
                if self.group_updates.get(group): continue
                for name,p in model.named_parameters():
                    if group not in name or not p.requires_grad: continue
                    if any(x in name for x in ('embed_tokens','lm_head','null_tokens','mask_token')): continue
                    if group=='geometry_backbone' and 'qkv.weight' not in name: continue
                    size=getattr(p,'ds_numel',p.numel())
                    if size<1: continue
                    indices=torch.linspace(0,size-1,min(1024,size),device=p.device).long()
                    self.step_probes.append((group,p,indices,self.master_values(p,indices)))
                    break
        @staticmethod
        def master_values(parameter,indices):
            if hasattr(parameter,'ds_id'):
                from deepspeed.utils import safe_get_full_fp32_param
                value=safe_get_full_fp32_param(parameter)
                if value is not None: return value.detach().flatten()[indices].clone()
            return Telemetry.values(parameter,indices)
        @staticmethod
        def values(parameter, indices):
            from contextlib import nullcontext
            context=nullcontext()
            if hasattr(parameter,'ds_id'):
                import deepspeed
                context=deepspeed.zero.GatheredParameters([parameter])
            with context:
                return parameter.detach().flatten()[indices].clone()
        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize(); now=time.monotonic()
            self.times.append(now-self.previous); self.previous=now
            for group, parameter, indices, before in self.probes:
                changed=not torch.equal(before, self.values(parameter,indices))
                self.group_updates[group]=self.group_updates.get(group,False) or changed
            for group,parameter,indices,before in self.step_probes:
                changed=not torch.equal(before,self.master_values(parameter,indices))
                self.group_updates[group]=self.group_updates.get(group,False) or changed
            self.updated |= self.group_updates.get('geometry_adapter',False)
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            self.probes=[]
            if any(hasattr(p,'ds_id') for p in model.parameters()):
                # DeepSpeed already stepped in backward. Scanning every now-
                # cleared gradient would trigger thousands of collectives;
                # the full-step FP32 master-weight probes are authoritative.
                return
            # Select coordinates with actual gradients, not arbitrary embedding
            # rows which may not occur in this batch. Only until first update.
            for group in ('geometry_adapter','geometry_backbone','model.visual','model.language_model'):
                if self.group_updates.get(group): continue
                for name, parameter in model.named_parameters():
                    if group not in name or not parameter.requires_grad: continue
                    if any(x in name for x in ('embed_tokens','lm_head','null_tokens')): continue
                    if group=='geometry_backbone' and 'qkv.weight' not in name: continue
                    if hasattr(parameter,'ds_id'):
                        from deepspeed.utils import safe_get_full_grad
                        gradient=safe_get_full_grad(parameter)
                    else:
                        gradient=parameter.grad
                    if gradient is None: continue
                    grad=gradient.detach().flatten()
                    if not torch.isfinite(grad).all(): raise ValueError('Nonfinite gradient: '+name)
                    if not torch.count_nonzero(grad): continue
                    indices=grad.abs().topk(min(16,grad.numel())).indices
                    self.probes.append((group,parameter,indices,self.values(parameter,indices)))
                    break
        def on_log(self, args, state, control, logs=None, **kw):
            import math
            if logs and any(not math.isfinite(float(logs[k])) for k in ('loss','grad_norm') if k in logs):
                raise ValueError('Nonfinite loss or gradient norm')
            if logs and 'loss' in logs: self.losses.append(logs['loss'])
            if state.is_world_process_zero:
                elapsed=time.monotonic()-self.started
                done=len(self.times)
                value=dict(step=state.global_step,max_steps=state.max_steps,**(logs or {}),
                    elapsed_seconds=elapsed,remaining_hours=(sum(self.times[1:])/max(1,done-1))*(state.max_steps-state.global_step)/3600,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    component_updates=self.group_updates,nonzero_update=self.updated,
                    eta_status='measured' if done>=3 else 'warming_up')
                dump(out/'live-eta.json',value); print(json.dumps(value),flush=True)
    telemetry=Telemetry()
    trainer=MatrixTrainer(model=model,args=args,train_dataset=Rows(),
        data_collator=MatrixCollator(processor,a.vggt_source,a.adapter),callbacks=[telemetry])
    trainer.model_accepts_loss_kwargs=False
    if trainer.is_world_process_zero(): dump(path,contract)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(out/'final'))
    # All ranks participate in reductions; no false per-rank memory acceptance.
    peak=torch.tensor(torch.cuda.max_memory_reserved()/2**30,device=f'cuda:{local}')
    updated=torch.tensor(int(telemetry.updated),device=peak.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(peak,op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(updated,op=torch.distributed.ReduceOp.MIN)
    if trainer.is_world_process_zero():
        processor.save_pretrained(out/'final')
        measured=sum(telemetry.times[1:]); steps=max(0,len(telemetry.times)-1)
        dump(out/'completion.json',dict(status='complete',steps=trainer.state.global_step,
            rows=len(rows),checkpoint=str(out/'final'),diagnostic_only=a.profile,
            finite_loss=bool(telemetry.losses),nonzero_update=bool(updated.item()),reload_verified=False,
            component_updates=telemetry.group_updates,
            samples_per_second=steps*a.global_batch/measured if measured else None,
            peak_reserved_gib=peak.item(),gpu_total_gib=torch.cuda.get_device_properties(local).total_memory/2**30,
            warmup_steps=1,measured_steps=steps,long_sample_passed=False,elapsed_seconds=sum(telemetry.times)))
    # Reference is taken from the unwrapped trained model; fresh-process
    # --verify-reload validates actual exported VGGT and projector weights.
    unwrapped=trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.eval()
    batch=trainer._prepare_inputs(MatrixCollator(processor,a.vggt_source,a.adapter)([rows[0]]))
    batch.pop('labels')
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        logits=unwrapped(**batch,use_cache=False,logits_to_keep=1).logits.float().cpu()
    if trainer.is_world_process_zero():
        torch.save({'row':rows[0],'logits':logits,
            'geometry_input_dtype':str(batch['geometry_images'][0].dtype),
            'rotary_buffers':{n:v.detach().cpu() for n,v in unwrapped.named_buffers() if 'inv_freq' in n},
            'normalization_mean':unwrapped.geometry_backbone.aggregator._resnet_mean.detach().cpu(),
            'normalization_std':unwrapped.geometry_backbone.aggregator._resnet_std.detach().cpu()},
            out/'reload-reference.pt')


if __name__ == '__main__':
    a=arguments()
    if a.parity: parity(a)
    elif a.evaluate: evaluation(a)
    elif a.verify_reload: verify_reload(a)
    else: train(a)
