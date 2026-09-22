"""Full-parameter GeoFits SFT worker; formal execution requires runtime acceptance.

No teacher or language LoRA. Real VGGT/Pi3 features are computed online with
frozen teachers, never replaced by zeros. Full, registered-model save/reload is
separate from formal benchmark acceptance. Each ablation changes real modules.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8'); temp.replace(path)


def validate_plan(plan, diagnostic_steps=0):
    from spatial_intelligence.geofits_recipe import validate_variant_architecture
    if plan.get('use_lora') is not False:
        raise ValueError('Only explicit full-parameter GeoFits training is supported')
    validate_variant_architecture(plan['architecture'],plan.get('variant','full'))
    data = read(plan['data_receipt'])
    from spatial_intelligence.followup_data_policy import validate_leakage_policy
    validate_leakage_policy(data)
    if data.get('status') != 'ready' or data.get('media_verified') is not True:
        raise ValueError('Audited media and source-scene data split required')
    if not diagnostic_steps:
        gate = read(plan['runtime_acceptance'])
        if gate.get('status') != 'ready' or gate.get('full_model_verified') is not True:
            raise ValueError('Formal execution requires actual full-model runtime acceptance')
        if gate.get('architecture') != plan['architecture']:
            raise ValueError('Runtime gate belongs to a different GeoFits architecture')
    return data


def load_teachers(plan, device):
    """Load only active frozen trunks; absent branches are not zero placeholders."""
    from spatial_intelligence.geofits_recipe import validate_variant_architecture
    from spatial_intelligence.geofits_teachers import load_vggt_levels,load_pi3_levels
    architecture=validate_variant_architecture(plan['architecture'],plan.get('variant','full'))
    loaders={'vggt':load_vggt_levels,'pi3':load_pi3_levels}
    teachers={name:loaders[name](plan[name+'_source'],plan[name+'_weights']).to(device).eval()
              for name in architecture['bank_sources']}
    if any(p.requires_grad for teacher in teachers.values() for p in teacher.parameters()):
        raise ValueError('Geometry teachers must remain frozen')
    return teachers


def arrange(rows, global_batch=64, seed=3407):
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Nonempty unique training IDs required')
    rows = list(rows); random.Random(seed).shuffle(rows)
    count = len(rows)
    rows.extend(rows[i % count] for i in range((-count) % global_batch))
    return rows, dict(original_rows=count, tail_padding=len(rows)-count, scheduled_rows=len(rows))


def feature_entries(records, inputs, model, teachers, device):
    """True teacher inputs and per-sample ragged image positions, no answer input."""
    import numpy as np
    import torch
    from spatial_intelligence.georoute_inputs import canvas
    from spatial_intelligence.geofits_teachers import matching_teacher_canvas
    entries = []
    image_id = model.config.image_token_id
    labels = inputs.get('labels')
    if not isinstance(teachers,dict):
        # Original full-only caller compatibility; variant loaders return dicts.
        teachers=dict(zip(('vggt','pi3'),teachers))
    expected=set(model.geofits.config.bank_sources)
    if set(teachers)!=expected: raise ValueError('Teacher ownership differs from model bank sources')
    for i, row in enumerate(records):
        images = torch.stack([torch.from_numpy(np.asarray(canvas(path)[0]).copy()).permute(2,0,1).float()/255
                              for path in row['media']]).unsqueeze(0).to(device)
        teacher_images, receipt = matching_teacher_canvas(images, [list(map(str, row['media']))])
        vggt = teachers['vggt'](teacher_images) if 'vggt' in teachers else {}
        pi3 = teachers['pi3'](teacher_images) if 'pi3' in teachers else {}
        visual = (inputs['input_ids'][i] == image_id).nonzero().flatten().to(device)
        if labels is None:
            # Micro1 inference has no left padding. Evaluation adapters with
            # larger batches must retain absolute prefix ends, not token counts.
            valid = inputs['attention_mask'][i].nonzero().flatten()
            prefix = int(valid[-1])+1
        else:
            supervised = (labels[i] != -100).nonzero().flatten()
            if not supervised.numel(): raise ValueError('Missing answer supervision')
            prefix = int(supervised[0])
        timestamps=None
        if model.geofits.config.timestamp_encoding=='normalized_linear_sincos':
            values=row.get('timestamps')
            if values is None or len(values)!=len(row['media']):
                raise ValueError('Measured timestamp recipe requires one real timestamp per frame')
            timestamps=torch.tensor([values],device=device,dtype=torch.float32)
        item = dict(vggt=vggt, pi3=pi3, visual_indices=visual, prefix_length=prefix,
            native_grid=(len(row['media']), *receipt['merged_grid']), timestamps=timestamps)
        if labels is not None: item['labels'] = labels[i].to(device)
        entries.append(item)
    return entries


def train(plan, micro=1, diagnostic_steps=0):
    data = validate_plan(plan, diagnostic_steps)
    import torch
    from torch.utils.data import Dataset, SequentialSampler
    from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from spatial_intelligence.geofits_model import load_geofits_model
    from spatial_intelligence.georoute_inputs import RouteCollator
    from spatial_intelligence.qwen35 import completion_loss
    local = int(os.getenv('LOCAL_RANK', 0)); world = int(os.getenv('WORLD_SIZE', 1))
    if micro < 1 or 64 % (world*micro): raise ValueError('world*micro must divide global batch64')
    torch.cuda.set_device(local); torch.set_num_threads(4); set_seed(plan.get('seed', 3407))
    source = [json.loads(line) for line in Path(data['train_manifest']).read_text(encoding='utf-8').splitlines() if line]
    rows, count = arrange(source, seed=plan.get('seed', 3407))
    out = Path(plan['root'])/('diagnostic' if diagnostic_steps else 'formal')/f'micro{micro}'
    contract = dict(model=plan['model'], data=data, architecture=plan['architecture'], training='full_parameter_no_lora',
        micro=micro, world=world, ga=64//(world*micro), global_batch=64, seed=plan.get('seed',3407),
        lr=1e-5, epochs=1, count=count, diagnostic=bool(diagnostic_steps), max_steps=diagnostic_steps or -1,
        teacher_inputs='shared448-letterbox-resize392', teachers=plan['architecture'].get('bank_sources',['vggt','pi3']),
        variant=plan.get('variant','full'),loss='sample_mean',
        vggt_weights=plan.get('vggt_weights'), pi3_weights=plan.get('pi3_weights'))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completion.json').exists(): raise ValueError('Completed output exists; preserve it')
    if (out/'contract.json').exists() and read(out/'contract.json') != contract: raise ValueError('Changed resume contract')
    if int(os.getenv('RANK',0)) == 0: write(out/'contract.json', contract)
    args = TrainingArguments(output_dir=str(out), num_train_epochs=1, max_steps=diagnostic_steps or -1,
        per_device_train_batch_size=micro, gradient_accumulation_steps=contract['ga'], learning_rate=1e-5,
        warmup_ratio=.03, lr_scheduler_type='cosine', weight_decay=0., bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        dataloader_num_workers=2, dataloader_pin_memory=True, save_steps=100, save_total_limit=2,
        logging_steps=1, report_to=[], remove_unused_columns=False, ddp_find_unused_parameters=True,
        seed=contract['seed'], data_seed=contract['seed'], optim='adamw_torch_fused', deepspeed=plan.get('deepspeed'))
    model = load_geofits_model(plan['model'], architecture=plan['architecture'], dtype=torch.bfloat16, attn_implementation='sdpa')
    if any('lora_' in n for n,_ in model.named_parameters()): raise ValueError('Unexpected LoRA parameters')
    model.requires_grad_(True); model.config.use_cache=False
    device = torch.device('cuda', local)
    teachers = load_teachers(plan,device)
    processor = AutoProcessor.from_pretrained(plan['processor'])
    class Rows(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self,index): return rows[index]
    class FitsTrainer(Trainer):
        def _get_train_sampler(self,train_dataset=None): return SequentialSampler(self.train_dataset)
        def training_step(self,wrapped,inputs,num_items_in_batch=None):
            records=inputs.pop('_route_rows')
            actual=self.accelerator.unwrap_model(wrapped)
            entries=feature_entries(records,inputs,actual,teachers,self.args.device)
            with actual.feature_context(entries):
                return super().training_step(wrapped,inputs,num_items_in_batch)
        def compute_loss(self,wrapped,inputs,return_outputs=False,num_items_in_batch=None):
            loss,outputs=completion_loss(wrapped,inputs,reduction='sample_mean')
            if not torch.isfinite(loss): raise ValueError('Nonfinite GeoFits loss')
            return (loss,outputs) if return_outputs else loss
    class Telemetry(TrainerCallback):
        def __init__(self): self.previous=time.monotonic();self.times=[];self.losses=[];self.probes={};self.updated={}
        @staticmethod
        def probe(parameter):
            if hasattr(parameter,'ds_id'):
                from deepspeed.utils import safe_get_full_fp32_param
                value=safe_get_full_fp32_param(parameter)
                if value is None: raise ValueError('Missing sharded FP32 state')
            else: value=parameter.detach().float()
            value=value.flatten();count=min(value.numel(),2048)
            if not count: raise ValueError('Empty audited parameter')
            indices=torch.arange(count,device=value.device,dtype=torch.long)*(value.numel()-1)//max(1,count-1)
            return value[indices].detach().cpu().clone()
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            wanted={'language':'.language_model.layers.0.self_attn.q_proj.weight',
                'native_visual':'.visual.blocks.0.attn.qkv.weight',
                'geometry_projection':'geofits.bank.projectors.0.weight',
                'geometry_retrieval':'geofits.layers.0.output.weight'}
            if 'pi3' in plan['architecture'].get('bank_sources',['vggt','pi3']):
                wanted['temporal_adapter']='geofits.bank.temporal.0.up.weight'
            if plan['architecture'].get('gate_enabled',True):
                wanted['geometry_gate']='geofits.layers.0.gate.2.weight'
            for key,suffix in wanted.items():
                matches=[p for n,p in model.named_parameters() if n.endswith(suffix) and p.requires_grad]
                if len(matches)!=1: raise ValueError('Trainable ownership mismatch '+key)
                self.probes[key]=(matches[0],self.probe(matches[0]))
        def on_train_end(self,args,state,control,**kwargs):
            for key,(parameter,before) in self.probes.items():
                after=self.probe(parameter)
                self.updated[key]=bool(torch.isfinite(after).all()) and not torch.equal(before,after)
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs and 'loss' in logs:
                now=time.monotonic();self.times.append(now-self.previous);self.previous=now;self.losses.append(float(logs['loss']))
                if not math.isfinite(self.losses[-1]): raise ValueError('Nonfinite logged loss')
                if state.is_world_process_zero:
                    times=self.times[2:] or self.times;seconds=sum(times)/len(times)
                    write(out/'live-eta.json',dict(step=state.global_step,total_steps=state.max_steps,loss=logs['loss'],
                        seconds_per_update=seconds,remaining_seconds=seconds*(state.max_steps-state.global_step),
                        rank0_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,diagnostic=bool(diagnostic_steps)))
    telemetry=Telemetry()
    trainer=FitsTrainer(model=model,args=args,train_dataset=Rows(),data_collator=RouteCollator(processor),callbacks=[telemetry])
    trainer.model_accepts_loss_kwargs=False
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(out)))
    trainer._save_checkpoint(trainer.model,trial=None)
    trainer.save_model(str(out/'final'));trainer.save_state()
    raw=trainer.accelerator.unwrap_model(trainer.model_wrapped);raw.eval()
    row=next((x for x in source if len(x['media'])>1),source[0])
    probe=RouteCollator(processor,training=False)([row]);probe.pop('_route_rows')
    entries=feature_entries([row],probe,raw,teachers,args.device)
    probe={key:value.to(args.device) for key,value in probe.items()}
    with raw.feature_context(entries),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=trainer.model_wrapped(**probe,use_cache=False,logits_to_keep=1).logits.float().cpu()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        def cpu(value):
            if isinstance(value,torch.Tensor): return value.cpu()
            if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
            if isinstance(value,list): return [cpu(v) for v in value]
            return value
        torch.save(dict(inputs=cpu(probe),features=cpu(entries),logits=logits,row_id=row['id']),out/'reload-evidence.pt')
        processor.save_pretrained(out/'final')
        write(out/'completion.json',dict(status='trained_pending_acceptance',accepted=False,diagnostic=bool(diagnostic_steps),
            checkpoint=str(out/'final'),steps=trainer.state.global_step,component_updates=telemetry.updated,
            finite_loss=bool(telemetry.losses) and all(math.isfinite(x) for x in telemetry.losses),
            nonzero_update=bool(telemetry.updated) and all(telemetry.updated.values()),
            resume_checkpoint=str(out/f'checkpoint-{trainer.state.global_step}'),contract=contract))


def verify_saved(plan,micro=1,diagnostic_steps=0):
    import torch
    from spatial_intelligence.geofits_model import load_geofits_model
    torch.cuda.set_device(0);torch.set_num_threads(4)
    out=Path(plan['root'])/('diagnostic' if diagnostic_steps else 'formal')/f'micro{micro}'
    receipt=read(out/'completion.json')
    if not receipt.get('finite_loss') or not receipt.get('nonzero_update'): raise ValueError('Missing actual loss/component updates')
    resume=Path(receipt['resume_checkpoint'])
    if not (resume/'trainer_state.json').exists(): raise ValueError('Missing Trainer state')
    world=int(receipt['contract']['world'])
    rng_paths=[resume/('rng_state.pth' if world==1 else f'rng_state_{rank}.pth') for rank in range(world)]
    for path in rng_paths:
        if not path.is_file(): raise ValueError('Missing per-rank RNG checkpoint: '+path.name)
        state=torch.load(path,map_location='cpu',weights_only=False)
        if not all(key in state for key in ('python','numpy','cpu','cuda')):
            raise ValueError('Incomplete per-rank RNG state')
        del state
    optimizer_files=list(resume.glob('**/*optim*.pt'))
    if not optimizer_files: raise ValueError('Missing optimizer state')
    for path in optimizer_files:
        state=torch.load(path,map_location='cpu',weights_only=False)
        if not isinstance(state,dict) or not any(k in state for k in ('state','optimizer_state_dict')):
            raise ValueError('Invalid optimizer checkpoint payload')
        del state
    scheduler_ok=False
    scheduler_paths=[resume/'scheduler.pt'] if (resume/'scheduler.pt').exists() else list(resume.glob('**/*model_states.pt'))
    for path in scheduler_paths:
        state=torch.load(path,map_location='cpu',weights_only=False)
        scheduler=state if path.name=='scheduler.pt' else state.get('lr_scheduler')
        scheduler_ok |= isinstance(scheduler,dict) and 'last_epoch' in scheduler
        del state
    if not scheduler_ok: raise ValueError('Missing saved scheduler state')
    evidence=torch.load(out/'reload-evidence.pt',map_location='cpu',weights_only=False)
    model=load_geofits_model(out/'final',dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    inputs={key:value.cuda() for key,value in evidence['inputs'].items()}
    with model.feature_context(evidence['features']),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits.float().cpu()
        if not torch.allclose(logits,evidence['logits'],atol=.02,rtol=.01): raise ValueError('Fresh model logits differ')
        output=model.generate(**inputs,use_cache=True,do_sample=False,max_new_tokens=8)
        if output.shape[1]<=inputs['input_ids'].shape[1]: raise ValueError('No generated continuation')
    receipt.update(status='complete',accepted=True,reload_verified=True,
        reload_max_logit_delta=float((logits-evidence['logits']).abs().max()),
        generation_smoke_tokens=int(output.shape[1]-inputs['input_ids'].shape[1]),
        optimizer_rng_files_present=True,optimizer_state_files_checked=len(optimizer_files),
        rng_ranks_checked=world,scheduler_state_present=True,
        acceptance_scope='stage update/reload only; full benchmark and exact resume trajectory remain separate')
    write(out/'completion.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--micro',type=int,default=1)
    parser.add_argument('--diagnostic-steps',type=int,default=0);parser.add_argument('--verify-saved',action='store_true')
    args=parser.parse_args()
    (verify_saved if args.verify_saved else train)(read(args.plan),args.micro,args.diagnostic_steps)


if __name__=='__main__': main()
