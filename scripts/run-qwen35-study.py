"""Persistent stage-gated baseline -> SFT -> paired ReVSI evaluation supervisor.

Run inside tmux/nohup (or use --detach). Independent downloads are not cancelled
by GPU stage failure. Re-run the same command to resume downloads/checkpoints.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[1]


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def matching_processes(cmd):
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except (OSError,UnicodeError):continue
        if argv==cmd:result.append(int(proc.name))
    return result


def run(root,stage,cmd,env=None):
    marker=root/'state'/f'{stage}.json'
    if marker.exists() and json.loads(marker.read_text()).get('status')=='complete':return
    log=root/'logs'/f'{stage}.log'
    active=matching_processes(cmd)
    if active:
        if 'eval' not in cmd or 'spatial_intelligence.study' not in cmd:
            raise RuntimeError(f'{stage} already has live worker(s) {active}; refuse duplicate stage')
        # A previous supervisor can die while torchrun remains alive. Wait for
        # that exact command, then run resume to verify/complete saved IDs.
        atomic(marker,{'status':'adopting','pids':active,'command':cmd,'log':str(log),'started':time.time()})
        while matching_processes(cmd):time.sleep(5)
    atomic(marker,{'status':'running','command':cmd,'started':time.time(),'log':str(log)})
    with log.open('ab') as f:
        p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        atomic(marker,{'status':'running','pid':p.pid,'command':cmd,'started':time.time(),'log':str(log)})
        result=p.wait()
    atomic(marker,{'status':'complete' if result==0 else 'failed','returncode':result,'finished':time.time(),'log':str(log),'command':cmd})
    if result:raise RuntimeError(f'{stage} failed ({result}); see {log}')


def await_assets(root,names):
    while True:
        pending=[]
        for name in names:
            p=root/'receipts'/f'{name}.json'
            d=json.loads(p.read_text()) if p.exists() else {}
            if d.get('status')=='failed':raise RuntimeError(f'Download failed: {name}: {d.get("error")}')
            if d.get('status')!='complete':pending.append(name)
        if not pending:return
        atomic(root/'state'/('waiting-'+names[0]+'.json'),{'status':'waiting','assets':pending,'updated':time.time()})
        time.sleep(15)


def telemetry(root,stop):
    with (root/'logs/gpu.jsonl').open('a') as f:
        while not stop.is_set():
            r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
            f.write(json.dumps({'unix_time':time.time(),'gpu_csv':r.stdout,'error':r.stderr})+'\n');f.flush()
            stop.wait(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--detach',action='store_true')
    p.add_argument('--attach-downloads',action='store_true',help='Downloads already running externally')
    p.add_argument('--status',action='store_true')
    args=p.parse_args();root=Path(args.root).resolve()
    for name in ['logs','state','receipts','sources']: (root/name).mkdir(parents=True,exist_ok=True)
    if args.status:
        for folder in ['state','receipts']:
            for path in sorted((root/folder).glob('*.json')):
                d=json.loads(path.read_text());print(folder,path.stem,d.get('status'),d.get('error',''))
        return
    if args.detach:
        argv=[x for x in sys.argv[1:] if x!='--detach']
        with (root/'logs/supervisor.log').open('ab') as f:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*argv],stdout=f,stderr=subprocess.STDOUT,
                                  start_new_session=True,stdin=subprocess.DEVNULL)
        print('Supervisor PID',proc.pid);return
    lock=(root/'state/study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(root/'state/study.json',{'status':'running','pid':os.getpid(),'started':time.time()})
    env=dict(os.environ,EUREKASI_ROOT=str(root),HF_ENDPOINT='https://hf-mirror.com',
             HF_HOME=str(root/'cache/huggingface'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    env.pop('PYTHONPATH',None)
    py=str(root/'envs/qwen35/bin/python')
    stop=threading.Event();threading.Thread(target=telemetry,args=(root,stop),daemon=True).start()
    if not args.attach_downloads:
        for name in json.loads((REPO/'configs/qwen35-assets.json').read_text()):
            f=(root/'logs'/f'download-{name}.log').open('ab')
            subprocess.Popen([sys.executable,str(REPO/'scripts/fetch-study-assets.py'),'--root',str(root),'--assets',name],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            f.close()
    try:
        run(root,'fetch-sources',[sys.executable,str(REPO/'scripts/fetch-study-sources.py'),'--root',str(root)],env)
        # Bootstrap may already be running; explicit readiness file avoids pip races.
        ready=root/'receipts/environment.json'
        while True:
            status=json.loads(ready.read_text()).get('status') if ready.exists() else None
            if status=='complete':break
            if status=='failed':raise RuntimeError('Environment setup failed')
            time.sleep(15)
        run(root,'cpu-tests',[py,'-m','pytest','tests','-q'],env)
        run(root,'start-vsibench-download',[sys.executable,str(REPO/'scripts/fetch-study-assets.py'),
            '--root',str(root),'--catalog',str(REPO/'configs/spatial-eval-assets.json'),'--detach'],env)
        await_assets(root,['model','revsi'])
        run(root,'prepare-revsi',[py,'-m','spatial_intelligence.study','prepare-revsi','--root',str(root)],env)
        run(root,'model-gate',[py,'-m','spatial_intelligence.study','verify-model','--root',str(root)],env)
        # CPU/media preparation can overlap the baseline, without using GPUs.
        errors=[]
        def prepare():
            try:
                await_assets(root,['vanilla_annotations','spar_rgbd','hound_media'])
                while not (root/'datasets/geothinker/train/llava_hound_64k.json').exists():time.sleep(15)
                run(root,'prepare-train',[py,'-m','spatial_intelligence.study','prepare-train','--root',str(root)],env)
            except Exception as e:errors.append(str(e));atomic(root/'state/prepare-train.json',{'status':'failed','error':str(e)})
        prep=threading.Thread(target=prepare,daemon=True);prep.start()
        launch=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','spatial_intelligence.study']
        def evaluate(name,model=None,limit=None,benchmark='revsi'):
            name+='-video-final-v1'
            opts=['--root',str(root),'--name',name,'--benchmark',benchmark,'--model',str(model or root/'models/Qwen3.5-2B'),
                  '--answer-format','tagged','--max-new-tokens','512','--smoke-per-type','1' if limit else '0']
            script=str(REPO/'scripts/evaluate-spatial.py')
            run(root,name,launch[:-2]+[script,*opts],env)
            run(root,name+'-score',[py,script,*opts,'--merge'],env)
            report=json.loads((root/'runs'/name/'metrics.json').read_text())
            if limit and (report['parse_rate']<.9 or report['truncation_rate']>.05):
                raise RuntimeError('Evaluation format gate failed; do not expand into a misleading full benchmark')
            return report
        evaluate('baseline-smoke',limit=16)
        evaluate('baseline-revsi32')
        # Validate real-data DDP and checkpoint recovery before the full media wait.
        small=root/'datasets/hound/train_300k/chunk_15.tar.gz'
        frames=root/'datasets/geothinker/train/llava_hound_64k.json'
        while not small.exists() or not frames.exists():
            for asset in ['hound_media','scaled_annotations']:
                receipt=root/'receipts'/f'{asset}.json'
                if receipt.exists() and json.loads(receipt.read_text()).get('status')=='failed':
                    raise RuntimeError(f'Diagnostic prerequisite download failed: {asset}')
            time.sleep(15)
        run(root,'prepare-probe',[py,'-m','spatial_intelligence.study','prepare-probe','--root',str(root)],env)
        diagnostic=launch+['train','--root',str(root),'--name','diagnostic-ddp','--batch-size','1','--ga','1','--diagnostic']
        run(root,'diagnostic-ddp',diagnostic+['--max-steps','2'],env)
        run(root,'diagnostic-resume',diagnostic+['--max-steps','3'],env)
        evaluate('diagnostic-reload',root/'runs/diagnostic-ddp/final',limit=16)
        prep.join()
        if errors:raise RuntimeError(errors[0])
        run(root,'prepare-vsibench',[py,str(REPO/'scripts/prepare-vsibench.py'),'--root',str(root)],env)
        # Auxiliary diagnostics may still own allocated cards. Never collide with full DDP.
        while True:
            usage=subprocess.check_output(['nvidia-smi','--id='+env.get('CUDA_VISIBLE_DEVICES','0,1,2,3'),
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if all(int(v.strip())<500 for v in usage.splitlines()):break
            atomic(root/'state/waiting-gpus.json',{'status':'waiting','memory_mib':usage,'updated':time.time()})
            time.sleep(15)
        # Real mixed-row four-rank measurements, not single-card longest-row proxies.
        visible=env.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
        if len(visible)!=4:raise ValueError('This locked recipe requires exactly four visible GPUs')
        bench_out=root/'runs/diagnostic-ddp-batches-fresh-v1'
        run(root,'profile-ddp-mixed-v1',[sys.executable,str(REPO/'scripts/benchmark-ddp-batches.py'),'--root',str(root),
            '--output',str(bench_out),'--checkpoint',str(root/'models/Qwen3.5-2B')],env)
        selection=json.loads((bench_out/'selection.json').read_text());selected=selection['selected']
        micro=selected['micro_batch'];ga=selected['ga']
        atomic(root/'receipts/batch-selection.json',selection)
        run(root,'sft-smoke',launch+['train','--root',str(root),'--name','sft-smoke','--batch-size',str(micro),'--ga','1','--max-steps','2'],env)
        run(root,'sft',launch+['train','--root',str(root),'--name','sft-spar234k-hound64k','--batch-size',str(micro),'--ga',str(ga)],env)
        evaluate('sft-revsi32',root/'runs/sft-spar234k-hound64k/final')
        before=json.loads((root/'runs/baseline-revsi32-video-final-v1/metrics.json').read_text())
        after=json.loads((root/'runs/sft-revsi32-video-final-v1/metrics.json').read_text())
        atomic(root/'runs/comparison-video-final-v1.json',{'baseline':before,'sft':after,
            'delta_extracted_points':after['extracted']['overall_score']-before['extracted']['overall_score']})
        vsi_before=evaluate('baseline-vsibench',benchmark='vsibench')
        vsi_after=evaluate('sft-vsibench',root/'runs/sft-spar234k-hound64k/final',benchmark='vsibench')
        atomic(root/'runs/comparison-vsibench-video-final-v1.json',{'baseline':vsi_before,'sft':vsi_after,
            'delta_extracted_points':vsi_after['extracted']['overall_score']-vsi_before['extracted']['overall_score']})
        atomic(root/'state/study.json',{'status':'complete','finished':time.time()})
    except Exception as e:
        atomic(root/'state/study.json',{'status':'failed','error':str(e),'time':time.time()});raise
    finally:stop.set()


if __name__=='__main__':main()
