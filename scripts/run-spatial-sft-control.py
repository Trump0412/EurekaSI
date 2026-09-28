"""Two-GPU, matched-prompt SFT control for the Qwen3.5 spatial GSPO pilot.

This is supervised spatial QA, NOT a pure format-only intervention.
It shares 400 prompt draws/global batch 4 with the RL pilot, not compute cost.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--reference-run',required=True)
    parser.add_argument('--name',default='qwen35-spatial-sft-control-v1')
    parser.add_argument('--gpus',default='2,3');parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();reference=Path(args.reference_run).resolve()
    if Path(args.name).name!=args.name:raise ValueError('Single component run name required')
    devices=args.gpus.split(',')
    if len(set(devices))!=2 or len(devices)!=2 or not all(d.isdigit() for d in devices):raise ValueError('Allocate two distinct GPUs')
    if args.detach:
        with (root/'logs'/f'{args.name}.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[s for s in sys.argv[1:] if s!='--detach']],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'name':args.name}));return
    lock=(root/'state/spatial-sft-control.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'runs'/args.name;out.mkdir(exist_ok=True)
    py=str(root/'envs/qwen35/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,OMP_NUM_THREADS='4',PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    def status(phase,**values):
        write(root/'state/spatial-sft-control.json',{'status':phase,'pid':os.getpid(),'updated':time.time(),
            'run':str(out),'devices':devices,**values})
    def execute(stage,command):
        receipt=out/f'{stage}.receipt.json'
        if receipt.exists() and json.loads(receipt.read_text())['status']=='complete':return
        status('running',stage=stage,command=command)
        with (out/f'{stage}.log').open('ab') as log:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(receipt,{'status':'complete' if result.returncode==0 else 'failed','returncode':result.returncode})
        if result.returncode:raise RuntimeError(f'{stage} failed; see {out/stage}.log')
    try:
        status('waiting_for_allocated_gpus');deadline=time.monotonic()+3600
        while True:
            used=subprocess.check_output(['nvidia-smi','--id='+args.gpus,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if all(int(v)<500 for v in used.splitlines()):break
            if time.monotonic()>deadline:raise TimeoutError('GPU allocation stayed occupied')
            time.sleep(10)
        cfg=json.loads((reference/'pilot.json').read_text())
        cfg['train'].update(mode='sft',gradient_accumulation=2,reward_plugin=None)
        cfg['output']=str(out/'training');write(out/'train.json',cfg)
        write(out/'experiment.json',{'reference_run':str(reference),'world_size':2,'global_prompt_batch':4,
            'prompt_draws':cfg['train']['steps']*4,'same_data_and_lora':True,'compute_matched':False,
            'meaning':'SFT versus reward-based learning; not a pure formatting ablation',
            'exact_resume_supported':False})
        launch=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','spatial_intelligence']
        execute('train',launch+['train','--config',str(out/'train.json')])
        evaluation=json.loads((reference/'baseline-val.json').read_text())
        evaluation['output']=str(out/'post-val');evaluation['model']['adapter']=str(out/'training/final')
        write(out/'post-val.json',evaluation)
        execute('post-val',launch+['infer','--config',str(out/'post-val.json')])
        from spatial_intelligence.boolean_metrics import summarize
        from spatial_intelligence.evaluation import compare
        from spatial_intelligence.io import read_jsonl
        baseline=json.loads((reference/'baseline-val/metrics.json').read_text())
        after=json.loads((out/'post-val/metrics.json').read_text());compare([baseline,after])
        rows=read_jsonl(evaluation['data']['eval'])
        before=summarize(rows,read_jsonl(reference/'baseline-val/metrics.details.jsonl'))
        after=summarize(rows,read_jsonl(out/'post-val/metrics.details.jsonl'))
        write(out/'comparison.json',{'baseline':before,'sft':after,
            'accuracy_delta':after['accuracy']-before['accuracy'],
            'balanced_accuracy_delta':after['balanced_accuracy']-before['balanced_accuracy']})
        status('complete');write(out/'completion.json',{'status':'complete','steps':cfg['train']['steps'],'exact_resume_supported':False})
    except Exception as exc:status('failed',error=str(exc));raise


if __name__=='__main__':main()
