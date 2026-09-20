"""Checkpoint-preserving multimodal GSPO with real grouped rollouts.

This inspectable HF/PEFT synchronous backend preserves custom VGGT inputs.
It is not the vanilla VERL/vLLM engine, which cannot silently load this model.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sampled_row(datasets, arm, seed, index):
    """Deterministic 7:3 blocks, shuffled by block; restart/world-size independent."""
    source='4drl'
    if arm['spatial_fraction']:
        if arm['four_d_rl_fraction']!=.7 or arm['spatial_fraction']!=.3:
            raise ValueError('This locked pair uses exactly 70:30 or 100:0')
        block=index//10
        schedule=['4drl']*7+['spatialladder']*3
        random.Random(seed+block).shuffle(schedule)
        source=schedule[index%10]
    pool=datasets[source]
    return pool[random.Random(seed*1000003+index).randrange(len(pool))]


def evaluation_row(row, metadata_reader=None):
    """Adapt released benchmark manifests without changing frames or exposing gold."""
    if row.get('answer_type') and row.get('input_mode'):
        return dict(row)
    if metadata_reader is None:
        from spatial_intelligence.spatial_eval import video_metadata
        metadata_reader=video_metadata
    fps,total=metadata_reader(row['source'])
    return dict(row,answer=row['ground_truth'],
        answer_type='mcq' if row.get('choices') else 'numeric',
        input_mode='video',fps=fps,total_num_frames=total)


def probe_indices(size, count, device):
    import torch
    count=min(count,size)
    if count<=1: return torch.zeros(count,device=device,dtype=torch.int64)
    return torch.arange(count,device=device,dtype=torch.int64)*(size-1)//(count-1)


def synchronize_gradients(parameters, world):
    import torch
    import torch.distributed as dist
    if world==1: return
    # One usage vector collective, then bounded dtype-homogeneous buckets.
    used=torch.tensor([p.grad is not None for p in parameters],device=parameters[0].device,dtype=torch.int32)
    dist.all_reduce(used)
    bucket=[]; size=0
    def flush():
        if not bucket: return
        flat=torch.cat([p.grad.flatten() for p in bucket])
        dist.all_reduce(flat); flat.div_(world)
        offset=0
        for p in bucket:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p)); offset+=p.numel()
    for active,p in zip(used.tolist(),parameters):
        if not active: continue
        if p.grad is None: p.grad=torch.zeros_like(p)
        if bucket and (size+p.numel()*p.element_size()>16*2**20 or bucket[0].dtype!=p.dtype):
            flush(); bucket=[]; size=0
        bucket.append(p); size+=p.numel()*p.element_size()
    flush()


def run(plan, arm_name, mode):
    import torch
    import torch.distributed as dist
    from transformers import AutoProcessor, set_seed
    from spatial_intelligence.geometry_rft import (load_policy,train_mode,prompt_inputs,to_device,
        cached_geometry,sample_group,response_logps,frozen_inventory)
    from spatial_intelligence.geometry_rft_objective import group_standardized_advantages,gspo_loss
    from spatial_intelligence.geometry_rft_reward import score_response,RewardConfig
    rank=int(os.getenv('RANK','0')); local=int(os.getenv('LOCAL_RANK','0')); world=int(os.getenv('WORLD_SIZE','1'))
    torch.cuda.set_device(local); torch.set_num_threads(4)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    def barrier():
        if world>1: dist.barrier()
    def gather(value):
        if world==1: return [value]
        values=[None]*world; dist.all_gather_object(values,value); return values
    seed=int(plan.get('seed',3407)); set_seed(seed)
    arm=next(x for x in plan['jobs'] if x['name']==arm_name)
    recipe=read(plan['scientific_config']) if plan.get('scientific_config') else {}
    cfg={**recipe.get('training',{}),**plan.get('training',{})}
    plan=dict(plan,lora_rank=int(cfg.get('lora_rank',64)),lora_alpha=int(cfg.get('lora_alpha',128)))
    group=int(plan.get('group_size',8)); prompt_batch=int(plan['prompts_per_update'])
    if group!=8 or prompt_batch%world: raise ValueError('Keep G=8 and globally matched prompt batch')
    data=read(plan['data_receipt'])
    if data.get('status')!='ready' or not data.get('media_verified') or not data.get('leakage_checked'):
        raise ValueError('Data/media/heldout gate not accepted')
    sft=read(plan['sft_receipt'])
    if sft.get('status')!='complete' or not sft.get('reload_verified') or sft.get('diagnostic_only') is not False:
        raise ValueError('A verified formal SFT checkpoint is required')
    if Path(sft['checkpoint']).resolve()!=Path(plan['model_checkpoint']).resolve():
        raise ValueError('SFT lineage mismatch')
    out=Path(plan['root'])/'runs'/arm_name/mode; out.mkdir(parents=True,exist_ok=True)
    datasets={k:rows(v) for k,v in plan.get('train_manifests',data['train_manifests']).items()}
    if not datasets.get('4drl') or (arm['spatial_fraction'] and not datasets.get('spatialladder')):
        raise ValueError('Empty declared training source')
    # The same mixed-pool-based total prompt budget is assigned to BOTH arms.
    reference_rows=sum(len(v) for v in datasets.values())
    updates=int(plan.get('updates') or math.ceil(2*reference_rows/prompt_batch))
    if plan.get('prompt_budget') not in (None,updates*prompt_batch): raise ValueError('Prompt budget mismatch')
    if plan.get('accepted_train_rows',reference_rows)!=reference_rows:
        raise ValueError('Snapshotted training row count disagrees with paired budget')
    target=2 if mode=='gate' else updates
    max_new=int(cfg.get('max_new_tokens',512)); context_limit=int(cfg.get('max_context',16384))
    contract={'arm':arm,'initial_checkpoint':str(Path(plan['model_checkpoint']).resolve()),
        'data_receipt':str(Path(plan['data_receipt']).resolve()),'group_size':group,
        'prompts_per_update':prompt_batch,'updates':target,'formal_updates':updates,
        'prompt_budget':target*prompt_batch,'world':world,'seed':seed,'training':cfg,
        'reference_rows':reference_rows,'engine':'HF PEFT synchronous GSPO, not VERL',
        'sampling':'temperature .7 / top_p .9; surrogate uses untempered actor log probabilities',
        'sampling_limit':'truncated sampling differs from raw actor distribution; no exact behavior-density claim',
        'mode':mode,'base_checkpoint_identity':sft.get('checkpoint')}
    path=out/'contract.json'
    if path.exists() and read(path)!=contract: raise ValueError('Changed contract requires a new run')
    if rank==0: write(path,contract)
    processor=AutoProcessor.from_pretrained(plan.get('processor',plan['model_checkpoint']))
    completed=sorted((p for p in out.glob('checkpoint-*') if (p/'complete.json').exists()),
                     key=lambda p:int(p.name.split('-')[-1]))
    checkpoint=completed[-1] if completed else None
    policy=load_policy(plan,checkpoint/'adapter' if checkpoint else None).cuda()
    base=policy.get_base_model(); adapter=base.config.geometry_matrix['adapter']
    inventory=frozen_inventory(policy)
    if any(('geometry_backbone' in n or '.visual.' in n) for n in inventory['trainable']):
        raise ValueError('RFT must freeze native RGB and checkpoint-specific VGGT')
    if rank==0: write(out/'parameter-inventory.json',inventory)
    def prepare(row):
        value=to_device(prompt_inputs(processor,row,plan['vggt_source'],adapter,context_limit),torch.device('cuda',local))
        if value['input_ids'].shape[1]+max_new>context_limit:
            raise ValueError('Prompt plus response exceeds context; no silent truncation')
        return value
    def score(response,row):
        return score_response(response['text'],row['answer'],task_type=row['answer_type'],
            choices=row.get('choices'),truncated=response['truncated'],config=RewardConfig(**cfg.get('reward',{})))
    autocast=lambda: torch.autocast('cuda',dtype=torch.bfloat16)
    if mode=='evaluate':
        evaluate_pair(plan,arm_name,out,policy,processor,prepare,score,autocast,gather,barrier,rank,world)
        if world>1: dist.destroy_process_group()
        return
    parameters=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=float(cfg.get('learning_rate',1e-6)),weight_decay=0.)
    # Constant LR is an explicit adaptation where the RFT schedule is unspecified.
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    start=0; all_variation=0; components={'language_lora':False,'geometry_adapter':False}
    if checkpoint:
        state=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start=state['step']; all_variation=state['within_group_reward_variation']; components=state['component_updates']
        for value in optimizer.state.values():
            for k,v in value.items():
                if isinstance(v,torch.Tensor) and k!='step': value[k]=v.cuda()
        rng=torch.load(checkpoint/f'rng-rank{rank}.pt',map_location='cpu',weights_only=False)
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda']); random.setstate(rng['python'])
    else: set_seed(seed+rank)
    first_row=sampled_row(datasets,arm,seed,rank)
    fixed_prompt=prepare(first_row)
    fixed_tokens=torch.tensor(processor.tokenizer.encode('Spatial Observation: objects are visible. Spatial Transition: viewpoints differ. Answer Derivation: compare the evidence.',add_special_tokens=False),device=local)
    if fixed_tokens.numel()==0: raise ValueError('Empty reference probe')
    policy.eval()
    with cached_geometry(policy,fixed_prompt),autocast(),torch.no_grad():
        with policy.disable_adapter(): reference_initial=response_logps(policy,fixed_prompt,fixed_tokens).cpu()
        actor_initial=response_logps(policy,fixed_prompt,fixed_tokens).cpu()
    initial_parity=bool(torch.allclose(actor_initial,reference_initial,atol=.02,rtol=.01)) if not checkpoint else True
    if not initial_parity: raise ValueError('New actor differs from fixed SFT reference before training')
    del fixed_prompt
    rollout_micro=1
    selection_path=Path(plan['root'])/'runs'/arm_name/'gate'/'rollout-selection.json'
    if mode=='train':
        selected=read(selection_path); rollout_micro=selected['selected_micro']
        if selected.get('group_size')!=8: raise ValueError('Changed group size')
    elif not checkpoint:
        # Pure rollout sizing precedes any optimizer update; same long prompt
        # and seed per candidate. The logical group always contains eight draws.
        longest=max(datasets['4drl']+datasets.get('spatialladder',[]),
                    key=lambda r:(len(r['media']),len(r['question'])))
        candidate_prompt=prepare(longest); candidates=[]
        for micro in (1,2,4,8):
            torch.cuda.reset_peak_memory_stats(); set_seed(seed+rank)
            began=time.monotonic(); success=True; error=None; generated=0
            try:
                with cached_geometry(policy,candidate_prompt),autocast():
                    trials=sample_group(policy,processor,candidate_prompt,group,micro,max_new,
                        temperature=float(cfg.get('temperature',.7)),top_p=float(cfg.get('top_p',.9)))
                    generated=sum(len(r['tokens']) for r in trials)
                del trials
            except torch.OutOfMemoryError as exc:
                success=False; error=type(exc).__name__; torch.cuda.empty_cache()
            torch.cuda.synchronize()
            reports=gather({'micro':micro,'success':success,'error':error,'seconds':time.monotonic()-began,
                'tokens':generated,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'capacity_gib':torch.cuda.get_device_properties(local).total_memory/2**30})
            safe=all(r['success'] and r['peak_reserved_gib']<.92*r['capacity_gib'] for r in reports)
            candidates.append({'micro':micro,'safe':safe,'reports':reports,
                'tokens_per_second':sum(r['tokens'] for r in reports)/max(r['seconds'] for r in reports)})
        accepted=[c for c in candidates if c['safe']]
        if not accepted: raise RuntimeError('No G=8 rollout microbatch fits safely')
        rollout_micro=max(accepted,key=lambda c:c['tokens_per_second'])['micro']
        if rank==0: write(selection_path,{'selected_micro':rollout_micro,'group_size':8,'candidates':candidates,
            'limit':'rollout timing, not optimizer or total-training ETA'})
        del candidate_prompt; set_seed(seed+rank)
    elif selection_path.exists(): rollout_micro=read(selection_path)['selected_micro']
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    step_times=[]; started=time.monotonic()
    last_loss=state.get('last_loss',float('nan')) if checkpoint else 0.
    last_grad=state.get('last_grad',0.) if checkpoint else 0.
    for step in range(start,target):
        began=time.monotonic(); optimizer.zero_grad(set_to_none=True); local_loss=0.; varied=0; generated=0
        timings={'input':0.,'rollout_and_geometry':0.,'old_reference_logprob':0.,'backward':0.,'sync_optimizer':0.}
        source_stats={}
        probes={}
        for group_name,pattern in [('language_lora','lora_B'),('geometry_adapter','geometry_adapter')]:
            for name,p in policy.named_parameters():
                if pattern in name and p.requires_grad and p.ndim>=2:
                    indices=probe_indices(p.numel(),2048,local)
                    probes[group_name]=(p,indices,p.detach().flatten()[indices].clone()); break
        with (out/f'rollouts.rank{rank}.jsonl').open('a',encoding='utf-8') as stream:
            for local_index in range(prompt_batch//world):
                index=step*prompt_batch+local_index*world+rank
                part=time.monotonic()
                row=sampled_row(datasets,arm,seed,index); prompt=prepare(row)
                torch.cuda.synchronize(); timings['input']+=time.monotonic()-part
                part=time.monotonic()
                with cached_geometry(policy,prompt),autocast():
                    responses=sample_group(policy,processor,prompt,group,rollout_micro,max_new,
                        temperature=float(cfg.get('temperature',.7)),top_p=float(cfg.get('top_p',.9)))
                    torch.cuda.synchronize(); timings['rollout_and_geometry']+=time.monotonic()-part
                    scores=[score(r,row) for r in responses]
                    source=row['source']; stats=source_stats.setdefault(source,{'prompts':0,'responses':0,'answer_sum':0.,'reward_sum':0.,'truncated':0})
                    stats['prompts']+=1; stats['responses']+=len(responses)
                    stats['answer_sum']+=sum(r['answer'] for r in scores)
                    stats['reward_sum']+=sum(r['total'] for r in scores)
                    stats['truncated']+=sum(r['truncated'] for r in responses)
                    rewards=torch.tensor([r['total'] for r in scores],device=local)
                    advantages=group_standardized_advantages(rewards).to(local)
                    varied+=int(float(rewards.max()-rewards.min())>0)
                    # Freeze old/reference token probabilities BEFORE any update.
                    old=[]; reference=[]; policy.eval()
                    part=time.monotonic()
                    with torch.no_grad():
                        for response in responses:
                            old.append(response_logps(policy,prompt,response['tokens']).detach())
                        with policy.disable_adapter():
                            for response in responses:
                                reference.append(response_logps(policy,prompt,response['tokens']).detach())
                    torch.cuda.synchronize(); timings['old_reference_logprob']+=time.monotonic()-part
                    train_mode(policy)
                    part=time.monotonic()
                    for response,old_lp,ref_lp,adv in zip(responses,old,reference,advantages):
                        current=response_logps(policy,prompt,response['tokens'])
                        mask=torch.ones_like(current)[None]
                        loss,metrics=gspo_loss(current[None],old_lp[None],ref_lp[None],adv.reshape(1),mask,
                            clip_low=float(cfg.get('clip_low',.0003)),clip_high=float(cfg.get('clip_high',.0004)),
                            beta=float(cfg.get('kl_beta',.02)))
                        if not torch.isfinite(loss): raise ValueError('Nonfinite GSPO loss')
                        (loss/(group*(prompt_batch//world))).backward()
                        local_loss+=float(loss.detach())/(group*(prompt_batch//world))
                    torch.cuda.synchronize(); timings['backward']+=time.monotonic()-part
                    generated+=sum(len(r['tokens']) for r in responses)
                stream.write(json.dumps({'step':step+1,'prompt_index':index,'id':row['id'],
                    'source':row.get('source'),'responses':[{'text':r['text'],'tokens':r['tokens'].tolist(),
                    'truncated':r['truncated'],'reward':s,'advantage':float(a)}
                    for r,s,a in zip(responses,scores,advantages)]},ensure_ascii=False)+'\n'); stream.flush()
                del prompt,responses,old,reference
        part=time.monotonic(); synchronize_gradients(parameters,world)
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(grad): raise ValueError('Nonfinite policy gradient')
        optimizer.step(); scheduler.step()
        torch.cuda.synchronize(); timings['sync_optimizer']+=time.monotonic()-part
        for name,(p,indices,before) in probes.items():
            components[name] |= not torch.equal(before,p.detach().flatten()[indices])
        torch.cuda.synchronize(); seconds=time.monotonic()-began; step_times.append(seconds)
        reports=gather({'loss':local_loss,'grad_norm':float(grad),'variation':varied,'tokens':generated,
            'seconds':seconds,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'components':components,
            'timings':timings,'source_statistics':source_stats})
        all_variation+=sum(x['variation'] for x in reports)
        last_loss=sum(x['loss'] for x in reports)/world; last_grad=max(x['grad_norm'] for x in reports)
        if rank==0:
            value={'step':step+1,'max_steps':target,'loss':last_loss,'grad_norm':last_grad,
                'group_size':8,'rollout_micro':rollout_micro,'within_group_reward_variation':all_variation,
                'component_updates':components,'peak_reserved_gib':max(x['peak_reserved_gib'] for x in reports),
                'seconds':max(x['seconds'] for x in reports),'generated_tokens':sum(x['tokens'] for x in reports),
                'rank_timings':[x['timings'] for x in reports],
                'rank_source_statistics':[x['source_statistics'] for x in reports],
                'remaining_hours':sum(step_times[1:] or step_times)/len(step_times[1:] or step_times)*(target-step-1)/3600,
                'eta_status':'warming_up' if len(step_times)<3 else 'measured'}
            write(out/'live-eta.json',value)
            with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(value)+'\n')
            print(json.dumps(value),flush=True)
        if (step+1)%int(cfg.get('save_every',25))==0 or step+1==target:
            checkpoint_started=time.monotonic()
            checkpoint=out/f'checkpoint-{step+1:06d}'; checkpoint.mkdir(exist_ok=True)
            barrier()
            torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),
                        'python':random.getstate()},checkpoint/f'rng-rank{rank}.pt')
            if rank==0:
                policy.save_pretrained(checkpoint/'adapter'); processor.save_pretrained(checkpoint/'adapter')
                torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step+1,
                    'within_group_reward_variation':all_variation,'component_updates':components,
                    'last_loss':last_loss,'last_grad':last_grad},checkpoint/'optimizer.pt')
                write(checkpoint/'base.json',{'model_checkpoint':plan['model_checkpoint'],'contract':contract})
            barrier()
            if rank==0: write(checkpoint/'complete.json',{'step':step+1,'world':world})
            barrier()
            if rank==0: write(checkpoint/'timing.json',{'seconds':time.monotonic()-checkpoint_started})
    policy.eval(); final_prompt=prepare(first_row)
    with cached_geometry(policy,final_prompt),autocast(),torch.no_grad():
        expected=response_logps(policy,final_prompt,fixed_tokens).cpu()
        with policy.disable_adapter(): ref_after=response_logps(policy,final_prompt,fixed_tokens).cpu()
    reference_fixed=bool(torch.allclose(reference_initial,ref_after,atol=.02,rtol=.01))
    if not reference_fixed: raise ValueError('Reference changed during policy training')
    policy.cpu(); del optimizer,parameters,policy,base
    torch.cuda.empty_cache(); barrier()
    restored=load_policy(plan,checkpoint/'adapter',trainable=False).cuda()
    with cached_geometry(restored,final_prompt),autocast(),torch.no_grad():
        actual=response_logps(restored,final_prompt,fixed_tokens).cpu()
    delta=float((expected-actual).abs().max())
    reload_ok=bool(torch.allclose(expected,actual,atol=.02,rtol=.01))
    reports=gather({'reload_verified':reload_ok,'max_delta':delta,'reference_fixed':reference_fixed,
                    'components':components})
    healthy=(all(x['reload_verified'] and x['reference_fixed'] and all(x['components'].values()) for x in reports)
             and all_variation>0 and (last_grad>0 or start==target))
    receipt={'status':'complete' if healthy else 'failed','checkpoint':str(checkpoint/'adapter'),
        'arm':arm_name,'mode':mode,
        'initial_checkpoint':plan['model_checkpoint'],'updates':target,'prompt_budget':target*prompt_batch,
        'group_size':8,'finite_loss':math.isfinite(last_loss),'nonzero_update':all(components.values()),
        'component_updates':components,'within_group_reward_variation':all_variation>0,
        'groups_with_reward_variation':all_variation,'all_group_responses_verified':True,
        'reference_initial_parity':initial_parity,'reference_fixed':reference_fixed,
        'geometry_preserved':True,'reload_verified':all(x['reload_verified'] for x in reports),
        'reload_max_delta':max(x['max_delta'] for x in reports),'diagnostic':mode=='gate',
        'elapsed_seconds':time.monotonic()-started,'formal_updates':updates}
    receipt['allocated_gpu_hours_excluding_setup']=(time.monotonic()-started)*world/3600
    if rank==0: write(out/'completion.json',receipt)
    barrier()
    if world>1: dist.destroy_process_group()
    if not healthy: raise RuntimeError('GSPO update/reference/reload acceptance failed')


def evaluate_pair(plan, arm, out, policy, processor, prepare, score, autocast, gather, barrier, rank, world):
    import torch
    from spatial_intelligence.geometry_rft import load_policy,cached_geometry,sample_group
    from spatial_intelligence.spatial_eval import score_prediction,summarize
    trained=read(Path(plan['root'])/'runs'/arm/'train'/'completion.json')
    if trained.get('status')!='complete' or not trained.get('reload_verified'): raise ValueError('Training not accepted')
    data=read(plan['data_receipt']); manifests=plan.get('eval_manifests',data['eval_manifests'])
    if not manifests: raise ValueError('No declared evaluation datasets')
    results={}
    for variant in ('sft','rft'):
        if variant=='rft':
            policy.cpu(); del policy; torch.cuda.empty_cache()
            policy=load_policy(plan,trained['checkpoint'],trainable=False).cuda()
        policy.eval()
        for benchmark,manifest in manifests.items():
            selected=rows(manifest); output=out/f'{variant}-{benchmark}.rank{rank}.jsonl'
            official=next((name for name in ('revsi','vsibench') if name in benchmark.lower()),None)
            existing=rows(output) if output.exists() else []; done={r['id'] for r in existing}
            if len(done)!=len(existing) or not done.issubset({r['id'] for r in selected[rank::world]}):
                raise ValueError('Foreign/duplicate evaluation IDs')
            with output.open('a',encoding='utf-8') as stream:
                for row in selected[rank::world]:
                    if row['id'] in done: continue
                    normalized=evaluation_row(row)
                    prompt=prepare(normalized)
                    with cached_geometry(policy,prompt),autocast(),torch.no_grad():
                        generated=policy.generate(**prompt,do_sample=False,max_new_tokens=512,use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id)
                    tokens=generated[0,prompt['input_ids'].shape[1]:]
                    eos=policy.generation_config.eos_token_id; eos=[eos] if isinstance(eos,int) else eos or []
                    response={'text':processor.decode(tokens,skip_special_tokens=True),
                        'truncated':len(tokens)>=512 and int(tokens[-1]) not in eos}
                    record={'id':row['id'],'question_type':row.get('question_type'),
                        'answer_type':normalized['answer_type'],'response':response,'tokens':tokens.tolist(),
                        'frames':len(row['media'])}
                    if official:
                        record['official_score']=score_prediction(row,response['text'],official,response['truncated'])
                    else:
                        record['reward']=score(response,normalized)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush()
            barrier()
            if rank==0:
                merged=[r for j in range(world) for r in rows(out/f'{variant}-{benchmark}.rank{j}.jsonl')]
                if len(merged)!=len(selected) or {r['id'] for r in merged}!={r['id'] for r in selected}:
                    raise ValueError('Incomplete paired benchmark coverage')
                if official:
                    report=summarize([r['official_score'] for r in merged],official)
                    report['prompt_protocol']='Same structured reasoning prompt for SFT and RFT; not leaderboard-equivalent'
                    results.setdefault(benchmark,{})[variant]=report
                    continue
                categories={r.get('question_type','unknown') for r in merged}
                report={'count':len(merged),'mean_answer_reward':sum(r['reward']['answer'] for r in merged)/len(merged),
                    'truncation_rate':sum(r['response']['truncated'] for r in merged)/len(merged),
                    'parse_rate':sum(r['reward']['parsed_answer'] is not None for r in merged)/len(merged),
                    'per_task':{kind:{'count':sum(r.get('question_type','unknown')==kind for r in merged),
                        'mean_answer_reward':sum(r['reward']['answer'] for r in merged if r.get('question_type','unknown')==kind)/sum(r.get('question_type','unknown')==kind for r in merged)} for kind in categories},
                    'metric':'MCQ exact accuracy; numeric SpatialLadder relative-error reward, not exact-match accuracy'}
                results.setdefault(benchmark,{})[variant]=report
    if rank==0: write(out/'completion.json',{'status':'complete','paired':results,'arm':arm,'mode':'evaluate',
        'initial_checkpoint':plan['model_checkpoint'],'paired_ids_verified':True,'reload_verified':True,
        'eval_manifest':plan['eval_manifest'],'eval_manifests':manifests,'checkpoint':trained['checkpoint'],
        'claim':'fixed paired offline QA; no process-truth verification'})
    barrier()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path); parser.add_argument('--arm',required=True)
    parser.add_argument('--mode',required=True,choices=['gate','train','evaluate'])
    arguments=parser.parse_args()
    run(read(arguments.plan),arguments.arm,arguments.mode)
