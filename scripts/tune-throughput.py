"""Independent, bounded four-GPU performance study after the current study releases GPUs.

Never interrupts training/evaluation or promotes diagnostic weights. Produces a
recommendation only after numerical gates and repeated real-data timing.
"""
import argparse
import fcntl
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--detach',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (a.output/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],cwd=REPO,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print(json.dumps({'pid':child.pid,'output':str(a.output)}));return
    own=(a.output/'lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.output/'recommendation.json').exists():raise RuntimeError('Finished run; use a new output directory')
    lock=(a.root/'state/study.lock').open('a');deadline=time.monotonic()+48*3600
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            dump(a.output/'status.json',{'status':'waiting_for_current_study','pid':os.getpid(),'updated':time.time()})
            if time.monotonic()>deadline:raise TimeoutError('Study lock still busy after 48h')
            time.sleep(30)
    def idle():
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if len(memory.splitlines())!=4 or any(int(x)>500 for x in memory.splitlines()):
            raise RuntimeError('Four allocated GPUs must be idle; refuse collision')
    idle()
    checkpoint=a.checkpoint or a.root/'runs/sft-spar234k-hound64k/final'
    if not (checkpoint/'config.json').exists():raise RuntimeError('Missing source checkpoint')
    rows=[json.loads(x) for x in (a.root/'manifests/sft.train.jsonl').read_text().splitlines()]
    indices=list(range(len(rows)));random.Random(3407).shuffle(indices);steps=12;warmup=3
    selected=[rows[i] for i in indices[:64*steps]]
    selected+=sorted(rows,key=lambda r:(len(r['media']),len(r['question'])+len(r['answer'])),reverse=True)[:64]
    with (a.output/'samples.jsonl').open('w') as f:
        for row in selected:f.write(json.dumps(row)+'\n')
    py=str(a.root/'envs/qwen35/bin/python');env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    suite_deadline=time.monotonic()+3600
    def run(name,command,timeout=900):
        idle();dump(a.output/'status.json',{'status':'running','stage':name,'pid':os.getpid(),'updated':time.time()})
        log_path=a.output/f'{name}.log'
        with log_path.open('w') as log:
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            until=min(suite_deadline,time.monotonic()+timeout);reason=None
            while child.poll() is None:
                text=log_path.read_text(errors='replace')[-32000:]
                if 'OutOfMemoryError' in text or time.monotonic()>until:
                    reason='oom' if 'OutOfMemoryError' in text else 'timeout'
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(2)
            code=child.wait()
        dump(a.output/f'{name}-process.json',{'returncode':code,'failure':reason})
        return code==0
    parity=a.output/'model-parity.json'
    gate=run('model-parity',[py,str(REPO/'scripts/verify-throughput-model.py'),'--root',str(a.root),
        '--checkpoint',str(checkpoint),'--samples',str(a.output/'samples.jsonl'),'--output',str(parity)])
    if not gate:
        dump(a.output/'status.json',{'status':'blocked_numerical_gate','evidence':str(parity)});return
    variants=[('auto','legacy',1),('reference','legacy',1),('reference','legacy',2),('reference','balanced',2),
              ('fla','legacy',1),('fla','balanced',2),('fla','balanced',4)]
    results={}
    for repeat in (0,1):
        for backend,layout,micro in variants if repeat==0 else reversed(variants):
            if time.monotonic()>suite_deadline:break
            key=f'{backend}-{layout}-b{micro}';tag=f'{key}-r{repeat}'
            cmd=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(REPO/'scripts/benchmark-ddp-batches.py'),
                '--root',str(a.root),'--output',str(a.output),'--checkpoint',str(checkpoint),'--micro',str(micro),
                '--steps',str(steps),'--warmup',str(warmup),'--tag',tag,'--layout',layout,'--backend',backend,
                '--reduction','sample_mean']
            if run(tag,cmd):results.setdefault(key,[]).append(json.loads((a.output/f'{tag}.json').read_text()))
    qualified={k:v for k,v in results.items() if len(v)==2 and all(x['peak_reserved_gib']<.92*x['gpu_total_gib'] for x in v)}
    base=qualified.get('auto-legacy-b1')
    if not base:raise RuntimeError('Missing repeated safe baseline; no recommendation')
    speed=lambda values:statistics.median(x['samples_per_second'] for x in values)
    best=max(qualified,key=lambda k:speed(qualified[k]));gain=speed(qualified[best])/speed(base)
    # Require both repeats to beat the fastest baseline repeat, not one lucky median.
    accepted=gain>=1.05 and min(x['samples_per_second'] for x in qualified[best])>max(x['samples_per_second'] for x in base)
    chosen=qualified[best if accepted else 'auto-legacy-b1'][0]
    dump(a.output/'recommendation.json',{'status':'complete','recommended_key':best if accepted else 'auto-legacy-b1',
        'speed_ratio':gain,'promotion_threshold':1.05,'accepted':accepted,'automatically_changed_formal_training':False,
        'training_options':{'throughput_policy':chosen['layout'],'delta_backend':chosen['backend'],
                            'batch_size':chosen['micro_batch'],'ga':chosen['ga'],'loss_reduction':'sample_mean'},
        'results':results,'checkpoint':str(checkpoint),'scope':'two repeats, fixed train samples; not downstream accuracy'})
    dump(a.output/'status.json',{'status':'complete','finished':time.time()})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1])
            if out.is_dir():dump(out/'failure.json',{'status':'failed','error':repr(exc),'time':time.time(),'pid':os.getpid()})
        raise
