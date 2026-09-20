"""Three-step continuous vs interrupted/resumed optimized Trainer acceptance.

Waits on the study lock, keeps max_steps=3 for both routes, and never resumes
the stopped research SFT. All artifacts have diagnostic names.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def dump(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--detach',action='store_true')
    p.add_argument('--branch-source',type=Path,help='Existing continuous diagnostic run: fork its exact checkpoint-2 to isolate restore from independent-run noise')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (a.output/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],cwd=REPO,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print(json.dumps({'pid':child.pid}));return
    own=(a.output/'lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(a.root/'state/study.lock').open('a')
    dump(a.output/'status.json',{'status':'waiting_for_throughput_suite','pid':os.getpid()})
    deadline=time.monotonic()+7200
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic()>deadline:raise TimeoutError('Throughput suite still owns GPUs after 2h')
            time.sleep(15)
    if json.loads((a.suite/'model-parity.json').read_text())['status']!='complete':
        raise RuntimeError('Numerical gate did not pass')
    rows=[json.loads(x) for x in (a.suite/'samples.jsonl').read_text().splitlines()][:192]
    manifest=a.output/'samples.jsonl'
    with manifest.open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    stem='diagnostic-'+a.output.name
    names=[stem+'-continuous',stem+'-resumed']
    if a.branch_source:
        source=a.branch_source.resolve()
        if source.parent!=(a.root/'runs').resolve() or not source.name.startswith('diagnostic-'):
            raise ValueError('Branch source must be a diagnostic run under this root')
        state=json.loads((source/'checkpoint-3/trainer_state.json').read_text())
        if state['global_step']!=3 or state['max_steps']!=3:raise ValueError('Branch source must have the same three-step budget')
        names[0]=source.name
    for name in names:
        if a.branch_source and name==names[0]:continue
        if (a.root/'runs'/name).exists():raise RuntimeError('Refuse to overwrite earlier resume evidence')
    if a.branch_source:
        target=a.root/'runs'/names[1]
        shutil.copytree(a.branch_source/'checkpoint-2',target/'checkpoint-2')
        shutil.copy2(a.branch_source/'throughput-contract.json',target/'throughput-contract.json')
    py=str(a.root/'envs/qwen35/bin/python')
    common=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','spatial_intelligence.study',
        'train','--root',str(a.root),'--model',str(a.checkpoint),'--processor',str(a.root/'models/Qwen3.5-2B'),
        '--diagnostic','--diagnostic-manifest',str(manifest),'--max-steps','3','--batch-size','1','--ga','16',
        '--throughput-policy','balanced','--delta-backend','auto','--loss-reduction','sample_mean']
    stages=[('continuous',names[0],[]),('pause',names[1],['--stop-after-steps','2']),('resume',names[1],[])]
    if a.branch_source:stages=stages[-1:]
    for stage,name,extra in stages:
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if any(int(x)>500 for x in memory.splitlines()):raise RuntimeError('GPU collision')
        dump(a.output/'status.json',{'status':'running','stage':stage,'pid':os.getpid()})
        with (a.output/f'{stage}.log').open('w') as log:
            child=subprocess.Popen(common+['--name',name]+extra,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),start_new_session=True)
            try:code=child.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        if code:raise RuntimeError(f'{stage} failed, see log')
        receipt=json.loads((a.root/'runs'/name/'completion.json').read_text())
        expected=2 if stage=='pause' else 3
        if receipt['steps']!=expected:raise RuntimeError('Unexpected completed step count')
        if stage=='resume' and not receipt['resumed_from'].endswith('checkpoint-2'):
            raise RuntimeError('Did not actually restore checkpoint-2')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    paths=[a.root/'runs'/name/'checkpoint-3' for name in names]
    numerator=0.;denominator=0.;maximum=0.;count=0
    with safe_open(str(paths[0]/'model.safetensors'),framework='pt') as x,safe_open(str(paths[1]/'model.safetensors'),framework='pt') as y:
        if set(x.keys())!=set(y.keys()):raise RuntimeError('Parameter coverage mismatch')
        for key in x.keys():
            left=x.get_tensor(key).float();right=y.get_tensor(key).float();diff=left-right
            numerator+=float(diff.square().sum());denominator+=float(left.square().sum());maximum=max(maximum,float(diff.abs().max()));count+=left.numel()
    schedulers=[torch.load(path/'scheduler.pt',map_location='cpu',weights_only=False) for path in paths]
    optimizers=[torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False) for path in paths]
    opt_num=0.;opt_den=0.;opt_max=0.;opt_equal=True
    if optimizers[0]['param_groups']!=optimizers[1]['param_groups']:raise RuntimeError('Optimizer groups changed')
    if optimizers[0]['state'].keys()!=optimizers[1]['state'].keys():raise RuntimeError('Optimizer state coverage changed')
    for index,left in optimizers[0]['state'].items():
        right=optimizers[1]['state'][index]
        if left.keys()!=right.keys():raise RuntimeError('Optimizer fields changed')
        for key,value in left.items():
            other=right[key]
            if torch.is_tensor(value):
                if key=='step':
                    if not torch.equal(value,other):raise RuntimeError('Optimizer step changed')
                    continue
                diff=value.float()-other.float();opt_num+=float(diff.square().sum());opt_den+=float(value.float().square().sum())
                opt_max=max(opt_max,float(diff.abs().max()));opt_equal=opt_equal and torch.equal(value,other)
            elif value!=other:raise RuntimeError('Non-tensor optimizer state differs')
    opt_relative=(opt_num/max(opt_den,1e-16))**.5
    states=[json.loads((path/'trainer_state.json').read_text()) for path in paths]
    relative=(numerator/max(denominator,1e-16))**.5
    sched_match=schedulers[0]==schedulers[1]
    step_match=all(s['global_step']==3 and s['max_steps']==3 for s in states)
    result={'status':'complete' if relative<1e-5 and sched_match and step_match else 'failed',
        'parameter_relative_l2':relative,'parameter_max_absolute_difference':maximum,'parameter_elements':count,
        'scheduler_equal':sched_match,'steps_and_budgets_equal':step_match,'budget_optimizer_steps':3,
        'optimizer_bitwise_equal':opt_equal,'optimizer_relative_l2':opt_relative,'optimizer_max_absolute_difference':opt_max,
        'interrupted_at':2,'manifest':str(manifest),'run_names':names,
        'shared_checkpoint_branch':bool(a.branch_source),
        'scope':'actual checkpoint restore and final model/scheduler comparison; three diagnostic updates, not task accuracy'}
    dump(a.output/'result.json',result);dump(a.output/'status.json',result)
    if result['status']!='complete':raise RuntimeError('Checkpoint restore comparison failed; see result.json')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1]);path.mkdir(parents=True,exist_ok=True)
            failure={'status':'failed','error':repr(exc),'time':time.time(),'pid':os.getpid()}
            dump(path/'failure.json',failure);dump(path/'status.json',failure)
        raise
