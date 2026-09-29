"""One-pass, full-language + geometry-interface SFT with the EXACT RFT input path.

RGB/VGGT stay frozen. FP32 policy packs overlay the immutable mature SFT base;
no LoRA, no BF16 round-trip of learned parameters, no teacher-input gold leakage.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from spatial_intelligence.rft_coldstart import read, rows, write


def run(plan, mode):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from transformers import AutoProcessor, set_seed
    from spatial_intelligence.geometry_rft import (load_policy, prompt_inputs, to_device,
        response_logps, train_mode, rft_autocast, CPUStateAdamW, save_policy, resolve_rft_prompt)
    from spatial_intelligence.geometry_rft_reward import score_response, RewardConfig
    spec = importlib.util.spec_from_file_location('rft_worker', REPO / 'scripts/train-geometry-rft.py')
    rft = importlib.util.module_from_spec(spec); spec.loader.exec_module(rft)
    root = Path(plan['root']); acceptance = read(root / 'acceptance.json')
    if acceptance.get('status') != 'accepted': raise ValueError('Annotations not accepted')
    local = int(os.getenv('LOCAL_RANK', '0')); rank = int(os.getenv('RANK', '0'))
    world = int(os.getenv('WORLD_SIZE', '1'))
    torch.cuda.set_device(local); torch.set_num_threads(4); set_seed(3407)
    if world > 1: dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    def barrier():
        if world > 1: dist.barrier()
    def gather(value):
        if world == 1: return [value]
        output = [None] * world; dist.all_gather_object(output, value); return output
    batch = plan['global_batch']
    if batch != 64 or batch % world: raise ValueError('Cold-start global batch must be 64')
    manifest = root / 'manifests/coldstart.train.jsonl'; training = rows(manifest)
    out = root / 'coldstart'; out.mkdir(exist_ok=True)
    contract = dict(stage='coldstart_sft', model_checkpoint=plan['model_checkpoint'],
        manifest=str(manifest), rows=len(training), world=world, global_batch=batch,
        micro=1, ga=batch//world, epochs=1, lr=plan['learning_rate'], seed=3407,
        prompt_recipe=read(plan['scientific_config']), policy_scope='language_full_geometry',
        trainable='all language parameters + geometry adapter; frozen RGB and VGGT', use_lora=False)
    if (out / 'contract.json').exists() and read(out / 'contract.json') != contract:
        raise ValueError('Cold-start restart contract changed')
    if rank == 0: write(out / 'contract.json', contract)
    complete = sorted([p for p in out.glob('checkpoint-*') if (p/'complete.json').exists()])
    checkpoint = complete[-1] if complete else None
    if mode == 'validate':
        receipt = read(out / 'completion.json')
        if not receipt.get('reload_verified'): raise ValueError('Cold-start reload not verified')
        checkpoint = Path(receipt['checkpoint']).parent
    policy = load_policy(plan, checkpoint/'policy' if checkpoint else None, trainable=mode=='train').cuda()
    processor = AutoProcessor.from_pretrained(plan['processor'])
    _, instruction = resolve_rft_prompt(read(plan['scientific_config']), plan)
    adapter = policy.config.geometry_matrix['adapter']
    def prepare(row):
        return to_device(prompt_inputs(processor, row, plan['vggt_source'], adapter,
            instruction=instruction), torch.device('cuda', local))
    if mode == 'validate':
        values = []
        reward = RewardConfig(**read(plan['scientific_config'])['training']['reward'])
        for row in rows(root / 'manifests/coldstart.validation.jsonl')[rank::world]:
            prompt = prepare(row)
            with torch.no_grad(), rft_autocast():
                output = policy.generate(**prompt, do_sample=False, max_new_tokens=512, use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id)
            tokens = output[0,prompt['input_ids'].shape[1]:].tolist()
            eos = policy.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            ended = bool(tokens and tokens[-1] in eos)
            text = processor.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False).strip()
            score = score_response(text, row['answer'], row['answer_type'], row.get('choices'),
                truncated=len(tokens)>=512 and not ended, config=reward)
            values.append(dict(id=row['id'], response=text, **score))
        all_values = sum(gather(values), [])
        if rank == 0:
            from spatial_intelligence.rft_coldstart import write_rows
            write_rows(out / 'validation-predictions.jsonl', all_values)
            rate = sum(x['structure'] for x in all_values)/len(all_values)
            write(out / 'validation.json', dict(status='accepted' if rate >= .9 else 'blocked',
                rows=len(all_values), format_rate=rate,
                mean_answer_reward=sum(x['answer'] for x in all_values)/len(all_values),
                mean_words_reward=sum(x['words'] for x in all_values)/len(all_values),
                limitation='Format acceptance, not downstream improvement or reward advantage evidence'))
        barrier()
        if world > 1: dist.destroy_process_group()
        return
    if (out / 'completion.json').exists(): raise ValueError('Already trained; do not repeat')
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = CPUStateAdamW(parameters, lr=plan['learning_rate'], weight_decay=0.)
    total = math.ceil(len(training)/batch)
    warmup = max(1, math.ceil(total*.03))
    def schedule(step):
        if step < warmup: return (step+1)/warmup
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    order = list(range(len(training))); random.Random(3407).shuffle(order)
    order += order[:total*batch-len(order)]
    start = 0; updates = {'language':False, 'geometry_adapter':False}
    if checkpoint:
        state = torch.load(checkpoint/'optimizer.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start=state['step']; updates=state['updates']
        rng=torch.load(checkpoint/f'rng-rank{rank}.pt', map_location='cpu', weights_only=False)
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda']); random.setstate(rng['python'])
    begun = time.monotonic(); times=[]
    eos = processor.tokenizer.eos_token_id
    if eos is None: raise ValueError('Student EOS missing')
    for step in range(start, total):
        begin = time.monotonic(); optimizer.zero_grad(); train_mode(policy); loss_sum=0.
        for index in order[step*batch+rank:(step+1)*batch:world]:
            row = training[index]; prompt=prepare(row)
            tokens=torch.tensor(processor.tokenizer.encode(row['coldstart_response'], add_special_tokens=False)+[eos],device=local)
            if len(tokens)>512 or prompt['input_ids'].shape[1]+len(tokens)>16384:
                raise ValueError('Cold-start response/context budget exceeded')
            with rft_autocast(): loss=-response_logps(policy,prompt,tokens).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite cold-start loss')
            (loss/(batch//world)).backward(); loss_sum+=float(loss.detach())/(batch//world)
            del prompt, tokens, loss
        rft.synchronize_gradients(parameters, world)
        grad=float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
        if not math.isfinite(grad): raise ValueError('Nonfinite gradient')
        probes=[]
        for label, prefix in [('language','model.language_model.layers'), ('geometry_adapter','geometry_adapter')]:
            for name, parameter in policy.named_parameters():
                if prefix not in name or parameter.grad is None: continue
                flat=parameter.grad.flatten()
                if flat.numel() and torch.count_nonzero(flat):
                    ids=flat.abs().topk(min(16,flat.numel())).indices
                    probes.append((label,parameter,ids,parameter.detach().flatten()[ids].clone())); break
        optimizer.step(); scheduler.step()
        for label,parameter,ids,before in probes:
            updates[label] |= not torch.equal(before,parameter.detach().flatten()[ids])
        reports=gather(dict(loss=loss_sum, grad=grad, updates=updates))
        elapsed=time.monotonic()-begin; times.append(elapsed)
        if rank==0:
            write(out/'live-eta.json', dict(step=step+1,total_steps=total,
                loss=sum(x['loss'] for x in reports)/world, grad_norm=grad,
                component_updates=updates, remaining_hours=(total-step-1)*sum(times)/len(times)/3600,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            print(json.dumps(read(out/'live-eta.json')),flush=True)
        if (step+1)%plan.get('save_every',10)==0 or step+1==total:
            checkpoint=out/f'checkpoint-{step+1:06d}'; checkpoint.mkdir(exist_ok=True); barrier()
            torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate()),
                checkpoint/f'rng-rank{rank}.pt')
            if rank==0:
                save_policy(policy,checkpoint/'policy',plan); processor.save_pretrained(checkpoint/'policy')
                torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    step=step+1,updates=updates),checkpoint/'optimizer.pt')
            barrier()
            if rank==0: write(checkpoint/'complete.json',dict(step=step+1,world=world))
            barrier()
    policy.eval(); sample=training[rank%len(training)]; prompt=prepare(sample)
    tokens=torch.tensor(processor.tokenizer.encode(sample['coldstart_response'],add_special_tokens=False)[:32],device=local)
    with torch.no_grad(),rft_autocast(): expected=response_logps(policy,prompt,tokens).cpu()
    # Remove references retaining parameter storage before reload.
    del optimizer,scheduler,parameters,probes,parameter,policy
    torch.cuda.empty_cache(); barrier()
    restored=load_policy(plan,checkpoint/'policy',trainable=False).cuda()
    with torch.no_grad(),rft_autocast(): actual=response_logps(restored,prompt,tokens).cpu()
    good=bool(torch.allclose(expected,actual,atol=.02,rtol=.01))
    reports=gather(dict(reload=good,delta=float((actual-expected).abs().max()),updates=updates))
    healthy=all(x['reload'] and all(x['updates'].values()) for x in reports)
    if rank==0: write(out/'completion.json',dict(status='complete' if healthy else 'failed',
        checkpoint=str(checkpoint/'policy'),base_checkpoint=plan['model_checkpoint'],
        finite_loss=True,nonzero_update=all(all(x['updates'].values()) for x in reports),
        reload_verified=all(x['reload'] for x in reports),reload_max_delta=max(x['delta'] for x in reports),
        diagnostic_only=False,steps=total,rows=len(training),tail_padding=total*batch-len(training),
        elapsed_seconds=time.monotonic()-begun,gpu_work_finished=True))
    barrier()
    if world>1: dist.destroy_process_group()
    if not healthy: raise RuntimeError('Cold-start update/reload acceptance failed')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',required=True)
    p.add_argument('--mode',choices=['train','validate'],required=True)
    a=p.parse_args(); run(read(a.plan),a.mode)
