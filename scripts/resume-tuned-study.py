"""Wait for measured DDP selection, validate video evaluation, resume SFT, paired benchmarks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--detach',action='store_true')
    p.add_argument('--adopt-training-pid',type=int,help='Attach to verified existing torchrun without restarting training')
    a=p.parse_args();root=a.root
    if a.detach:
        with (root/'logs/tuned-supervisor.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print(json.dumps({'pid':child.pid}));return
    lock=(root/'state/study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(REPO))
    from spatial_intelligence.study import dump
    deadline=time.time()+7200
    while not a.selection.exists():
        if time.time()>deadline:raise TimeoutError('Batch selection did not complete')
        time.sleep(10)
    selection=json.loads(a.selection.read_text())['selected'];micro=selection['micro_batch'];ga=selection['ga']
    assert micro*ga*4==64
    adopted=False
    if a.adopt_training_pid:
        proc=Path(f'/proc/{a.adopt_training_pid}/cmdline')
        command=proc.read_bytes() if proc.exists() else b''
        if b'spatial_intelligence.study' not in command or b'sft-spar234k-hound64k' not in command:
            raise RuntimeError('Refuse to adopt unrelated/dead process')
        dump(root/'state/study.json',{'status':'adopted_training','pid':os.getpid(),'training_pid':a.adopt_training_pid})
        while proc.exists():time.sleep(10)
        receipt=root/'runs/sft-spar234k-hound64k/completion.json'
        if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='complete':
            raise RuntimeError('Adopted training exited without complete training receipt')
        dump(root/'state/sft-tuned.json',{'status':'complete','adopted_pid':a.adopt_training_pid,'evidence':str(receipt)})
        adopted=True
        deadline=time.time()+7200
    while True:
        mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if all(int(x)<500 for x in mem.splitlines()):break
        if time.time()>deadline:raise TimeoutError('GPUs not released')
        time.sleep(10)
    py=str(root/'envs/qwen35/bin/python')
    launch=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=4']
    env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1')
    def run(stage,cmd,required=True):
        receipt=root/'state'/f'{stage}.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('status')=='complete':return True
        with (root/'logs'/f'{stage}.log').open('ab') as log:
            child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            dump(receipt,{'status':'running','pid':child.pid,'command':cmd,'started':time.time()})
            code=child.wait()
        dump(receipt,{'status':'complete' if code==0 else 'failed','returncode':code,'finished':time.time(),'command':cmd})
        if code and required:raise RuntimeError(f'{stage} failed; inspect stage log')
        return code==0
    def evaluate(benchmark,name,model,smoke=0,answer_format='tagged',tokens=512):
        opts=['--root',str(root),'--name',name,'--benchmark',benchmark,'--model',str(model),
              '--answer-format',answer_format,'--max-new-tokens',str(tokens),'--smoke-per-type',str(smoke)]
        cmd=[str(REPO/'scripts/evaluate-spatial.py'),*opts]
        if not run(name,launch+cmd,False):return None
        if not run(name+'-score',[py,*cmd,'--merge'],False):return None
        return json.loads((root/'runs'/name/'metrics.json').read_text())
    base=root/'models/Qwen3.5-2B'
    dump(root/'state/study.json',{'status':'eval_smoke_then_resume','pid':os.getpid(),'selection':str(a.selection)})
    # These tiny, predeclared format tests do not choose a protocol by benchmark accuracy.
    evaluate('revsi','diagnostic-revsi-video-official16-v1',base,1,'official',16)
    smoke=evaluate('revsi','diagnostic-revsi-video-final-v1',base,1)
    gate=bool(smoke and smoke['parse_rate']>=.9 and smoke['truncation_rate']<=.05)
    dump(root/'receipts/spatial-eval-gate.json',{'status':'complete' if gate else 'failed',
        'parse_rate':smoke.get('parse_rate') if smoke else None,'accuracy_not_used_for_selection':True,
        'protocol':'native video 32-frame, tagged final, 512-token cap, greedy, ground-truth-blind extractor'})
    dump(root/'receipts/tuned-training-selection.json',{'micro_batch':micro,'ga':ga,'global_batch':64,
        'evidence':str(a.selection),'resume':'existing latest full checkpoint; not a benchmark model'})
    dump(root/'state/study.json',{'status':'running','pid':os.getpid(),'micro_batch':micro,'ga':ga})
    cmd=launch+['-m','spatial_intelligence.study','train','--root',str(root),'--name','sft-spar234k-hound64k',
                '--batch-size',str(micro),'--ga',str(ga)]
    if not adopted:run('sft-tuned',cmd)
    if not gate:
        dump(root/'state/study.json',{'status':'training_complete_eval_blocked','reason':'format smoke gate failed; no silent zero score'})
        return
    final=root/'runs/sft-spar234k-hound64k/final';results={}
    for benchmark in ('revsi','vsibench'):
        if benchmark=='vsibench' and not (root/'receipts/vsibench-prepared.json').exists():
            results[benchmark]={'status':'blocked_missing_prepared_data'};continue
        before=evaluate(benchmark,f'baseline-{benchmark}-video-final-v1',base)
        after=evaluate(benchmark,f'sft-{benchmark}-video-final-v1',final)
        results[benchmark]={'status':'complete' if before and after else 'failed','baseline':before,'sft':after}
        if before and after:
            # Model differs, all other inference fields and the exact ID list must match.
            contracts=[{k:v for k,v in x['contract'].items() if k not in ('model','model_files')} for x in (before,after)]
            if contracts[0]!=contracts[1]:raise ValueError('Unmatched before/after evaluation contracts')
            results[benchmark]['delta_extracted_points']=after['extracted']['overall_score']-before['extracted']['overall_score']
        dump(root/'runs/spatial-video-final-comparison.json',results)
    dump(root/'state/study.json',{'status':'complete' if all(x['status']=='complete' for x in results.values()) else 'evaluation_incomplete',
                                 'finished':time.time()})


if __name__=='__main__':main()
