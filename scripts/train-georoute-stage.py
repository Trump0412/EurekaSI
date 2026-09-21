"""Full-parameter GeoRoute TIP/SFT stage; real-data GPU acceptance is mandatory.

This new worker does not modify prior SFT/RFT. TIP initializes routing only;
instruction SFT updates all native language/vision/routing weights, without LoRA.
The discrete external geometry teacher remains frozen and is cached separately.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def schedule(rows,stage,global_batch=64,replay_every=15,seed=3407):
    """One deterministic instruction pass; replay is counted in OPTIMIZER batches.

    Tail padding is explicit. Different micro/world choices preserve the exact
    scheduled stream when world*micro divides the global batch.
    """
    rng=random.Random(seed);ordered=list(rows);rng.shuffle(ordered)
    tip=[row for row in ordered if len(row['media'])>1]
    if stage=='tip': ordered=tip
    if not ordered: raise ValueError('No eligible rows')
    original=len(ordered)
    ordered.extend(ordered[i%original] for i in range((-original)%global_batch))
    result=[];tip_cursor=0
    for first in range(0,len(ordered),global_batch):
        result.extend(dict(row,_task=stage) for row in ordered[first:first+global_batch])
        if stage=='sft' and replay_every and ((first//global_batch)+1)%replay_every==0:
            if not tip: raise ValueError('TIP replay requested without multi-view data')
            result.extend(dict(tip[(tip_cursor+i)%len(tip)],_task='tip') for i in range(global_batch))
            tip_cursor+=global_batch
    return result,dict(original_rows=original,tail_padding=len(ordered)-original,tip_replay_rows=tip_cursor,
                       total_scheduled_rows=len(result),unit='15 instruction optimizer updates then 1 TIP update')


def train(plan,stage,micro,diagnostic_steps=0,pause_request=None):
    import torch
    from torch.utils.data import Dataset,SequentialSampler
    from transformers import AutoProcessor,Trainer,TrainingArguments,TrainerCallback,set_seed,Qwen3VLForConditionalGeneration
    from transformers.trainer_utils import get_last_checkpoint
    from spatial_intelligence.georoute import (load_georoute_model,configure_stage,batch_graphs,make_tip_intervention)
    from spatial_intelligence.georoute_inputs import RouteCollator,GraphCache,load_teacher
    from spatial_intelligence.qwen35 import completion_loss
    from spatial_intelligence.cooperative_pause import PauseProtocol,persistent_probe_baselines,validate_full_checkpoint
    data=read(plan['data_receipt'])
    from spatial_intelligence.followup_data_policy import validate_leakage_policy
    validate_leakage_policy(data)
    if data.get('status')!='ready' or data.get('media_verified') is not True:
        raise ValueError('Five-source data are not accepted; downloaded assets are insufficient')
    if plan.get('use_lora') is not False: raise ValueError('Explicit full-parameter recipe required')
    if not diagnostic_steps:
        gate=read(plan['runtime_acceptance'])
        if gate.get('status')!='ready' or gate.get('full_model_verified') is not True:
            raise ValueError('Formal training requires a separate accepted real-model runtime gate')
        if gate.get('architecture')!=plan['architecture'] or gate.get('graph')!=plan['graph']:
            raise ValueError('Runtime gate belongs to a different architecture/graph protocol')
    local=int(os.getenv('LOCAL_RANK',0));world=int(os.getenv('WORLD_SIZE',1))
    torch.cuda.set_device(local);torch.set_num_threads(4);set_seed(plan.get('seed',3407))
    global_batch=64
    baseline=plan.get('variant')=='matched_rgb_sft'
    use_tip=not baseline and plan.get('variant','full')!='no_tip'
    if stage=='tip' and not use_tip: raise ValueError('This ablation must not run TIP')
    if global_batch%(world*micro): raise ValueError('Global batch64 must divide world*micro')
    source=[json.loads(line) for line in Path(data['train_manifest']).read_text(encoding='utf-8').splitlines() if line]
    if len({row['id'] for row in source})!=len(source): raise ValueError('Duplicate training IDs')
    arranged,count=schedule(source,stage,global_batch,plan.get('replay_every',15) if use_tip else 0,plan.get('seed',3407))
    out=Path(plan['root'])/('diagnostic' if diagnostic_steps else 'formal')/stage/f'micro{micro}'
    out.mkdir(parents=True,exist_ok=True)
    initial=plan['model'] if stage=='tip' or not use_tip else (
        plan['diagnostic_tip_checkpoint'] if diagnostic_steps and plan.get('diagnostic_tip_checkpoint') else plan['tip_checkpoint'])
    if stage=='sft' and use_tip:
        if diagnostic_steps and plan.get('diagnostic_tip_checkpoint'):
            initial=plan['diagnostic_tip_checkpoint']
            tip_receipt=read(plan['diagnostic_tip_receipt'])
        else:tip_receipt=read(plan['tip_receipt'])
        if tip_receipt.get('status')!='complete' or tip_receipt.get('accepted') is not True or (not diagnostic_steps and tip_receipt.get('diagnostic') is not False):
            raise ValueError('A complete accepted TIP checkpoint is required; formal SFT requires formal TIP lineage')
        if Path(tip_receipt['checkpoint']).resolve()!=Path(initial).resolve(): raise ValueError('TIP lineage mismatch')
    contract=dict(stage=stage,variant=plan.get('variant','full'),initial_model=initial,training='full_parameter_no_lora',count=count,micro=micro,world=world,
        ga=64//(world*micro),global_batch=64,seed=plan.get('seed',3407),lr=1e-5,warmup=.03,weight_decay=0.,
        loss='sample_mean',diagnostic=bool(diagnostic_steps),max_steps=diagnostic_steps or -1,
        data=data,architecture=plan['architecture'],graph=plan['graph'])
    if (out/'contract.json').exists() and read(out/'contract.json')!=contract: raise ValueError('Changed resume contract')
    if (out/'completion.json').exists(): raise ValueError('Completed run exists; do not overwrite')
    if int(os.getenv('RANK',0))==0: write(out/'contract.json',contract)
    args=TrainingArguments(output_dir=str(out),num_train_epochs=1,max_steps=diagnostic_steps or -1,
        per_device_train_batch_size=micro,gradient_accumulation_steps=contract['ga'],learning_rate=1e-5,
        warmup_ratio=.03,lr_scheduler_type='cosine',weight_decay=0.,bf16=True,tf32=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        dataloader_num_workers=2,dataloader_pin_memory=True,save_steps=100,save_total_limit=2,
        logging_steps=1,report_to=[],remove_unused_columns=False,ddp_find_unused_parameters=True,
        seed=contract['seed'],data_seed=contract['seed'],optim='adamw_torch_fused',
        deepspeed=plan.get('deepspeed') if stage=='sft' else None)
    if baseline:
        model=Qwen3VLForConditionalGeneration.from_pretrained(initial,dtype=torch.bfloat16,attn_implementation='sdpa')
        model.requires_grad_(True);model.train()
    else:
        model=load_georoute_model(initial,route_config=plan['architecture'] if stage=='tip' or not use_tip else None,
            dtype=torch.bfloat16,attn_implementation='sdpa')
        configure_stage(model,stage)
        teacher=load_teacher(plan['vggt_source'],plan['vggt_weights'],torch.device('cuda',local))
        checkpoint=Path(plan['vggt_weights'])
        identity=dict(path=str(checkpoint.resolve()),size=checkpoint.stat().st_size,
                      mtime_ns=checkpoint.stat().st_mtime_ns,source_revision=plan['vggt_revision'])
        cache=GraphCache(plan['graph_cache'],teacher,identity,plan['graph'],torch.device('cuda',local))
    model.config.use_cache=False
    processor=AutoProcessor.from_pretrained(plan['processor'])
    pause=PauseProtocol(pause_request,out)
    collator=RouteCollator(processor,training=True)
    class Rows(Dataset):
        def __len__(self): return len(arranged)
        def __getitem__(self,index): return arranged[index]
    class RouteTrainer(Trainer):
        def _get_train_sampler(self,train_dataset=None): return SequentialSampler(self.train_dataset)
        def _prepare_inputs(self,inputs):
            return super()._prepare_inputs({key:value for key,value in inputs.items() if key!='_route_rows'})
        def training_step(self,wrapped,inputs,num_items_in_batch=None):
            records=inputs['_route_rows'];tasks={row['_task'] for row in records}
            if len(tasks)!=1: raise ValueError('TIP/instruction micro-batch crossed optimizer boundary')
            if baseline:
                return super().training_step(wrapped,{key:value for key,value in inputs.items() if key!='_route_rows'},num_items_in_batch)
            graph=batch_graphs([cache.get(row) for row in records]).to(self.args.device)
            actual=self.accelerator.unwrap_model(wrapped)
            batch={key:value for key,value in inputs.items() if key!='_route_rows'}
            # This context outlives gradient-checkpoint recomputation.
            with actual.georoute.graph_context(graph):
                if next(iter(tasks))=='tip':
                    generator=torch.Generator().manual_seed(contract['seed']+self.state.global_step*world+local)
                    effective=actual.georoute._post_graph if actual.georoute.settings.get('placement')=='post' else graph
                    # A failed support gate stops the synchronized job. Never
                    # skip an individual rank or report a fake zero-loss update.
                    mask,bad=make_tip_intervention(effective.to('cpu'),generator=generator)
                    batch.update(route_tip=True,tip_mask=mask.to(self.args.device),tip_substituted_graph=bad.to(self.args.device))
                return super().training_step(wrapped,batch,num_items_in_batch)
        def compute_loss(self,wrapped,inputs,return_outputs=False,num_items_in_batch=None):
            if inputs.get('route_tip'):
                outputs=wrapped(**inputs);loss=outputs.loss
            else: loss,outputs=completion_loss(wrapped,inputs,reduction='sample_mean')
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            return (loss,outputs) if return_outputs else loss
    class Telemetry(TrainerCallback):
        def __init__(self):
            self.previous=time.monotonic();self.steps=[];self.losses=[];self.probes={};self.updated={}
        @staticmethod
        def probe(parameter):
            if hasattr(parameter,'ds_id'):
                from deepspeed.utils import safe_get_full_fp32_param
                full=safe_get_full_fp32_param(parameter)
                if full is None: raise ValueError('Missing sharded FP32 parameter audit')
            else: full=parameter.detach().float()
            size=full.numel();count=min(size,2048)
            if not size: raise ValueError('Cannot probe an empty parameter shard')
            indices=torch.arange(count,device=full.device,dtype=torch.int64)*(size-1)//max(1,count-1)
            return full.detach().flatten()[indices].cpu().clone()
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            if (out/'finite-loss-history.json').exists():
                self.losses=read(out/'finite-loss-history.json')['losses']
                if not all(math.isfinite(value) for value in self.losses):raise ValueError('Invalid prior loss audit')
            wanted={} if stage=='tip' else {'language':'.language_model.layers.0.self_attn.q_proj.weight',
                                          'native_visual':'.visual.blocks.0.attn.qkv.weight'}
            if not baseline:
                for index in plan['architecture'].get('active_exits',[0,1,2,3]):
                    wanted[f'routing_exit{index}']=f'georoute.routes.{index}.0.alpha'
            for group,suffix in wanted.items():
                matches=[p for name,p in model.named_parameters() if name.endswith(suffix) and p.requires_grad]
                if len(matches)!=1: raise ValueError('Parameter ownership/probe mismatch: '+group)
                self.probes[group]=(matches[0],self.probe(matches[0]));self.updated[group]=False
            initial=persistent_probe_baselines(out/'original-probe-baselines.pt',
                {group:value[1] for group,value in self.probes.items()})
            self.probes={group:(value[0],initial[group]) for group,value in self.probes.items()}
        def on_step_end(self,args,state,control,**kwargs):
            return pause.step_end(control,args.device)
        def on_save(self,args,state,control,**kwargs):
            pause.saved(state.global_step)
        def on_train_end(self,args,state,control,**kwargs):
            for group,(parameter,before) in self.probes.items():
                after=self.probe(parameter)
                if not torch.isfinite(after).all(): raise ValueError('Nonfinite parameter after training')
                self.updated[group]=not torch.equal(before,after)
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs and 'loss' in logs:
                now=time.monotonic();self.steps.append(now-self.previous);self.previous=now
                self.losses.append(float(logs['loss']))
                if not math.isfinite(self.losses[-1]): raise ValueError('Nonfinite optimizer loss')
                if state.is_world_process_zero:
                    write(out/'finite-loss-history.json',dict(losses=self.losses))
                    warm=self.steps[2:] or self.steps
                    write(out/'live-eta.json',dict(step=state.global_step,total_steps=state.max_steps,loss=logs['loss'],
                        seconds_per_update=sum(warm)/len(warm),remaining_seconds=(state.max_steps-state.global_step)*sum(warm)/len(warm),
                        rank0_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,diagnostic=bool(diagnostic_steps)))
    telemetry=Telemetry()
    trainer=RouteTrainer(model=model,args=args,train_dataset=Rows(),data_collator=collator,callbacks=[telemetry])
    # Custom sample-mean loss does not consume num_items_in_batch. Let Trainer
    # perform its ordinary GA scaling, regardless of the model **kwargs signature.
    trainer.model_accepts_loss_kwargs=False
    last=get_last_checkpoint(str(out))
    if last:
        validate_full_checkpoint(last,world)
        if not (out/'original-probe-baselines.pt').exists():
            raise ValueError('Resume lacks original parameter audit; cannot claim whole-run update verification')
        if pause_request and Path(pause_request).exists():
            raise ValueError('Pause request remains active; scheduler must clear it before resuming')
        if trainer.is_world_process_zero():
            write(out/'resume.json',dict(status='resuming',checkpoint=last,scientific_contract_unchanged=True))
    trainer.train(resume_from_checkpoint=last)
    # Preserve a final resumable Trainer checkpoint even for a short diagnostic
    # or a final tail shorter than the periodic checkpoint interval.
    trainer._save_checkpoint(trainer.model,trial=None)
    trainer.save_model(str(out/'final'));trainer.save_state()
    raw=trainer.accelerator.unwrap_model(trainer.model_wrapped);raw.eval()
    row=next((record for record in source if len(record['media'])>1),source[0])
    evidence_inputs=RouteCollator(processor,training=False)([row]);evidence_inputs.pop('_route_rows')
    evidence_inputs={key:value.to(args.device) for key,value in evidence_inputs.items()}
    graph=None if baseline else cache.get(row).to(args.device)
    context=nullcontext() if baseline else raw.georoute.graph_context(graph)
    with context,torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=trainer.model_wrapped(**evidence_inputs,use_cache=False,logits_to_keep=1).logits.float().cpu()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        torch.save(dict(inputs={key:value.cpu() for key,value in evidence_inputs.items()},
            graph=None if graph is None else vars(graph.to('cpu')),logits=logits,row_id=row['id']),out/'reload-evidence.pt')
        processor.save_pretrained(out/'final')
        write(out/'completion.json',dict(status='trained_pending_acceptance',accepted=False,
            diagnostic=bool(diagnostic_steps),checkpoint=str(out/'final'),steps=trainer.state.global_step,
            finite_loss=bool(telemetry.losses) and all(math.isfinite(loss) for loss in telemetry.losses),
            component_updates=telemetry.updated,nonzero_update=bool(telemetry.updated) and all(telemetry.updated.values()),
            resume_checkpoint=str(out/f'checkpoint-{trainer.state.global_step}'),
            pending=['fresh-process reload and inference'],contract=contract))


def verify_saved(plan,stage,micro,diagnostic_steps=0):
    """Separate process; never mark a checkpoint accepted from file existence."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from spatial_intelligence.georoute import load_georoute_model,RouteGraph
    torch.cuda.set_device(0);torch.set_num_threads(4)
    out=Path(plan['root'])/('diagnostic' if diagnostic_steps else 'formal')/stage/f'micro{micro}'
    receipt=read(out/'completion.json')
    if not receipt.get('finite_loss') or not receipt.get('nonzero_update'):
        raise ValueError('Missing actual finite loss/nonzero component updates')
    resume=Path(receipt['resume_checkpoint'])
    if not (resume/'trainer_state.json').is_file(): raise ValueError('Missing resumable Trainer state')
    optimizer_files=list(resume.glob('**/*optim*'))
    rng_files=list(resume.glob('rng_state*.pth'))
    if not optimizer_files or not rng_files: raise ValueError('Missing saved optimizer or RNG states')
    evidence=torch.load(out/'reload-evidence.pt',map_location='cpu',weights_only=False)
    baseline=plan.get('variant')=='matched_rgb_sft'
    loader=Qwen3VLForConditionalGeneration.from_pretrained if baseline else load_georoute_model
    model=loader(str(out/'final'),dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    inputs={key:value.cuda() for key,value in evidence['inputs'].items()}
    graph=None if baseline else RouteGraph(**evidence['graph']).to('cuda')
    context=nullcontext() if baseline else model.georoute.graph_context(graph)
    with context,torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits.float().cpu()
        if not torch.allclose(logits,evidence['logits'],atol=.02,rtol=.01): raise ValueError('Fresh checkpoint logits changed')
        response=model.generate(**inputs,do_sample=False,max_new_tokens=8,use_cache=True)
        if response.shape[1]<=inputs['input_ids'].shape[1]: raise ValueError('Reloaded model did not generate')
    receipt.update(status='complete',accepted=True,reload_verified=True,
        reload_max_logit_delta=float((logits-evidence['logits']).abs().max()),
        generation_smoke_tokens=int(response.shape[1]-inputs['input_ids'].shape[1]),
        optimizer_rng_files_present=True,pending=[],
        acceptance_scope='stage update and fresh reload; resume trajectory parity and benchmark results are separate gates')
    write(out/'completion.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--stage',choices=['tip','sft'],required=True)
    parser.add_argument('--micro',type=int,default=1);parser.add_argument('--diagnostic-steps',type=int,default=0)
    parser.add_argument('--verify-saved',action='store_true')
    parser.add_argument('--pause-request',help='Private request file; acknowledge only after a full optimizer-boundary checkpoint')
    args=parser.parse_args()
    if args.verify_saved:
        verify_saved(read(args.plan),args.stage,args.micro,args.diagnostic_steps)
    else:
        from spatial_intelligence.cooperative_pause import TrainingPaused
        try:train(read(args.plan),args.stage,args.micro,args.diagnostic_steps,args.pause_request)
        except TrainingPaused:raise SystemExit(75)


if __name__=='__main__': main()
