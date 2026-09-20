"""Same real global batches, four-rank DDP, fixed effective batch64; disposable updates."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def worker(a):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoProcessor
    from spatial_intelligence.qwen35 import Collator,load_model,completion_loss
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl');assert dist.get_world_size()==4
    torch.manual_seed(3407);torch.backends.cuda.matmul.allow_tf32=True
    rows=[json.loads(s) for s in (a.output/'samples.jsonl').read_text().splitlines()]
    local_rows=rows[rank::4]
    processor=AutoProcessor.from_pretrained(a.root/'models/Qwen3.5-2B')
    loader=torch.utils.data.DataLoader(local_rows,batch_size=a.micro,shuffle=False,
        num_workers=8,pin_memory=True,persistent_workers=True,collate_fn=Collator(processor))
    model=load_model(str(a.checkpoint),training=True).cuda()
    model=DistributedDataParallel(model,device_ids=[local],find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01,fused=True)
    iterator=iter(loader);ga=16//a.micro;records=[]
    for step in range(a.steps+1):
        # Final global batch contains longest real rows, excluded from speed, included in memory gate.
        dist.barrier();torch.cuda.synchronize();start=time.perf_counter();data_seconds=0
        optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for micro in range(ga):
            before=time.perf_counter();cpu_batch=next(iterator);data_seconds+=time.perf_counter()-before
            batch={k:v.cuda(non_blocking=True) for k,v in cpu_batch.items()}
            sync=model.no_sync() if micro<ga-1 else contextlib.nullcontext()
            with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                loss,outputs=completion_loss(model,batch)
                (loss/ga).backward()
            loss_sum+=float(loss.detach())/ga
            del outputs,loss,batch,cpu_batch
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        optimizer.step();torch.cuda.synchronize()
        values=torch.tensor([time.perf_counter()-start,data_seconds,
            torch.cuda.max_memory_allocated()/2**30,torch.cuda.max_memory_reserved()/2**30],device='cuda')
        dist.all_reduce(values,op=dist.ReduceOp.MAX)
        elapsed,data_wait,allocated,reserved=values.tolist()
        record={'step':step+1,'seconds':elapsed,'data_wait_max_seconds':data_wait,
                'peak_allocated_gib':allocated,'peak_reserved_gib':reserved,'rank0_loss':loss_sum,
                'stress_batch':step==a.steps}
        records.append(record)
        if rank==0:
            dump(a.output/f'b{a.micro}-progress.json',record);print(json.dumps(record),flush=True)
    if rank==0:
        measured=records[a.warmup:a.steps];seconds=sum(x['seconds'] for x in measured)
        total=torch.cuda.get_device_properties(local).total_memory/2**30
        dump(a.output/f'b{a.micro}.json',{'status':'complete','micro_batch':a.micro,'ga':ga,
            'effective_batch':64,'world_size':4,'warmup_steps':a.warmup,'timed_steps':len(measured),
            'samples_per_second':64*len(measured)/seconds,
            'median_seconds_per_step':statistics.median(x['seconds'] for x in measured),
            'peak_reserved_gib':max(x['peak_reserved_gib'] for x in records),'gpu_total_gib':total,
            'checkpoint':str(a.checkpoint),'records':records,'diagnostic_only':True,
            'optimizer':'fresh AdamW; warmup materializes optimizer state; no formal checkpoint modified'})
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--micro',type=int)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    if a.detach:
        a.output.mkdir(parents=True,exist_ok=False)
        with (a.output/'supervisor.log').open('w') as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print(json.dumps({'pid':child.pid,'output':str(a.output)}));return
    if a.micro is not None:
        if a.micro not in (1,2,4,8,16):raise ValueError('Effective batch must remain 64')
        worker(a);return
    a.output.mkdir(parents=True,exist_ok=True)
    assert 0<a.warmup<a.steps
    rows=[json.loads(s) for s in (a.root/'manifests/sft.train.jsonl').read_text().splitlines()]
    indices=list(range(len(rows)));random.Random(3407).shuffle(indices)
    selected=[rows[i] for i in indices[:64*a.steps]]
    selected+=sorted(rows,key=lambda x:(len(x['media']),len(x['question'])+len(x['answer'])),reverse=True)[:64]
    with (a.output/'samples.jsonl').open('w') as f:
        for row in selected:f.write(json.dumps(row)+'\n')
    py=str(a.root/'envs/qwen35/bin/python');results=[]
    for micro in (1,2,4,8,16):
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if any(int(x)>500 for x in memory.splitlines()):raise RuntimeError('Allocated GPUs not idle; refuse collision')
        command=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',__file__,
                 '--root',str(a.root),'--output',str(a.output),'--checkpoint',str(a.checkpoint),
                 '--micro',str(micro),'--steps',str(a.steps),'--warmup',str(a.warmup)]
        log_path=a.output/f'b{micro}.log';failure=None
        with log_path.open('w') as log:
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1'))
            deadline=time.monotonic()+900
            while child.poll() is None:
                tail=log_path.read_text(errors='replace')[-32000:]
                if 'OutOfMemoryError' in tail or time.monotonic()>deadline:
                    failure='oom' if 'OutOfMemoryError' in tail else 'timeout'
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(2)
            returncode=child.wait()
        receipt=a.output/f'b{micro}.json'
        if returncode:
            dump(receipt,{'status':'failed','micro_batch':micro,'returncode':returncode,'reason':failure})
        results.append(json.loads(receipt.read_text()))
        if failure=='oom':
            for larger in (1,2,4,8,16):
                if larger>micro:
                    skipped={'status':'skipped','micro_batch':larger,'reason':'smaller candidate OOM; not tested'}
                    dump(a.output/f'b{larger}.json',skipped);results.append(skipped)
            break
    safe=[x for x in results if x['status']=='complete' and x['peak_reserved_gib']<.92*x['gpu_total_gib']]
    if not safe:raise RuntimeError('No safe candidate')
    selected=max(safe,key=lambda x:x['samples_per_second'])
    dump(a.output/'selection.json',{'status':'complete','selected':selected,'candidates':results,
        'scope':'best measured among tested micro-batches on fixed real mixed rows; not a global optimum'})


if __name__=='__main__':main()
