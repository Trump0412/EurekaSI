"""Persistent four-GPU Qwen3.5 reference GSPO pilot, NOT the VERL engine.

Train-only exact rewards; frozen reference, language LoRA; scene-held-out
pre/post validation. Small diagnostic gate must show real policy gradients.
The existing reference trainer saves adapters, NOT resumable optimizer state.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
# Conda clones initially retain the source editable finder. The controller can
# outlive pip reinstall; prioritize this checkout without changing shared envs.
sys.path.insert(0,str(REPO))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def cloning(prefix):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=proc.joinpath('cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if '--clone' in args and '--prefix' in args:
            if Path(args[args.index('--prefix')+1]).resolve()==prefix.resolve():return True
    return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--source-root',required=True)
    parser.add_argument('--name',default='qwen35-spatial-gspo-v1')
    parser.add_argument('--gpus',default='4,5,6,7');parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();source=Path(args.source_root).resolve()
    if not args.name or Path(args.name).name!=args.name:raise ValueError('name must be a single directory name')
    devices=args.gpus.split(',')
    if len(devices)!=4 or len(set(devices))!=4 or not all(x.isdigit() for x in devices):raise ValueError('Allocate exactly four distinct GPUs')
    for folder in ('logs','state','runs','configs'): (root/folder).mkdir(exist_ok=True)
    if args.detach:
        with (root/'logs'/f'{args.name}.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[s for s in sys.argv[1:] if s!='--detach']],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'log':str(root/'logs'/f'{args.name}.log')}));return
    lock=(root/'state/reference-rl.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=root/'runs'/args.name;run.mkdir(exist_ok=True)
    def state(phase,**values):
        save(root/'state/reference-rl.json',{'status':phase,'pid':os.getpid(),'updated':time.time(),
            'run':str(run),'devices':devices,'backbone':'Qwen3.5-2B','engine':'EurekaSI reference, not VERL',**values})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',
             TOKENIZERS_PARALLELISM='false',HF_ENDPOINT='https://hf-mirror.com')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    py=str(root/'envs/qwen35/bin/python')
    def execute(stage,command,stage_env=None):
        receipt=run/f'{stage}.receipt.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('status')=='complete':return
        state('running',stage=stage,command=command)
        started=time.monotonic()
        with (run/f'{stage}.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=REPO,env=stage_env or env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                metrics=run/stage/'training/metrics.jsonl'
                if metrics.exists():
                    lines=metrics.read_text().splitlines()
                    try:
                        last=json.loads(lines[-1]);done=last['step'];elapsed=time.monotonic()-started
                        target=8 if stage=='gate' else args.steps
                        save(run/'live-eta.json',{'stage':stage,'step':done,'max_steps':target,'elapsed_seconds':elapsed,
                            'remaining_hours':elapsed/max(done,1)*max(0,target-done)/3600,
                            'loss':last['loss'],'grad_norm':last['grad_norm'],
                            'status':'warmup' if done<10 else 'measured_with_startup_overhead'})
                    except (IndexError,ValueError,KeyError):pass
                time.sleep(5)
        save(receipt,{'status':'complete' if child.returncode==0 else 'failed','returncode':child.returncode,
                      'seconds':time.monotonic()-started,'command':command})
        if child.returncode:raise RuntimeError(f'{stage} failed; inspect {run/stage}.log')
    try:
        deadline=time.monotonic()+14400
        state('waiting_for_isolated_environment')
        while cloning(root/'envs/qwen35') or not Path(py).exists():
            if time.monotonic()>deadline:raise TimeoutError('Environment clone timeout')
            time.sleep(10)
        if Path(sys.executable).resolve()!=Path(py).resolve():
            # Re-enter using the isolated environment for YAML and project imports.
            lock.close()
            os.execve(py,[py,__file__,*[s for s in sys.argv[1:] if s!='--detach']],env)
        execute('install',[py,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(REPO)])
        execute('pipcheck',[py,'-m','pip','check'])
        execute('focused-tests',[py,'-m','pytest','-q','tests/test_spatial_rl_audit.py','tests/test_qwen35_posttraining.py'])
        dataset=root/'manifests-local/spatial-boolean-v1'
        if not (dataset/'selection.json').exists():
            execute('prepare',[py,str(REPO/'scripts/prepare-spatial-rl-probe.py'),
                          '--source-root',str(source),'--output',str(dataset),'--train-count','512','--val-count','128'])
        selection=json.loads((dataset/'selection.json').read_text())
        if (selection['answer_kind']!='boolean' or selection['seed']!=3407 or
                selection['train']['rows']!=512 or selection['val']['rows']!=128 or
                selection['scene_overlap']!=0 or selection['revsi_scene_overlap']!=0):
            raise ValueError('Existing selection does not match the locked boolean pilot')
        # Exactly the same validation rows and decoding before/after; no test tuning.
        import yaml
        cfg=yaml.safe_load((REPO/'configs/rl.yaml').read_text())
        cfg['model'].update(backend='spatial_intelligence.backends.qwen35:Qwen35Backend',
                            path=str(source/'models/Qwen3.5-2B'),adapter=None,max_context=8192)
        cfg['model']['lora'].update(enabled=True,rank=8,alpha=16,target_modules=['q_proj','v_proj'])
        cfg['teacher']=copy.deepcopy(cfg['model'])
        cfg['data']={'train':[{'path':str(dataset/'train.jsonl'),'weight':1.}],
                     'heldout':[str(dataset/'val.jsonl')],'eval':str(dataset/'val.jsonl')}
        cfg['protocol'].update(name='spatial-boolean-scene-heldout-v1',max_frames=3,image_max_side=448,
            instruction='Answer Yes or No. Put only the final answer in <answer>...</answer>.')
        cfg['protocol']['generation'].update(max_new_tokens=64,do_sample=True,temperature=1.,top_p=1.,top_k=0)
        cfg['train'].update(mode='rl',steps=args.steps,gradient_accumulation=1,batch_size=1,group_size=4,
            algorithm='gspo',clip=.0004,beta=.02,learning_rate=1e-5,save_every=25,
            reward_plugin='spatial_intelligence.spatial_rl_audit:audited_reward',seed=3410,
            numeric_atol=0.,numeric_rtol=0.)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        for relative in ('scripts/run-qwen35-spatial-rl.py','scripts/prepare-spatial-rl-probe.py',
                         'spatial_intelligence/training.py','spatial_intelligence/objectives.py',
                         'spatial_intelligence/backends/qwen35.py','spatial_intelligence/backends/hf.py',
                         'spatial_intelligence/distributed.py','spatial_intelligence/spatial_rl_audit.py',
                         'spatial_intelligence/boolean_metrics.py'):
            target=run/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(REPO/relative,target)
        save(run/'experiment.json',{'config':cfg,'world_size':4,'global_prompt_batch':4,
            'base_git_commit':commit,'code_snapshot':str(run/'code'),
            'rollouts_per_prompt':4,'claim':'small spatial QA RL exploration; not full benchmark or VERL reproduction',
            'checkpoint_limit':'reference trainer saves adapter weights; exact optimizer resume unsupported'})
        state('waiting_for_allocated_gpus')
        while True:
            memory=subprocess.check_output(['nvidia-smi','--id='+args.gpus,'--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'],text=True)
            if all(int(x.strip())<500 for x in memory.splitlines()):break
            if time.monotonic()>deadline:raise TimeoutError('Allocated GPUs stayed busy')
            time.sleep(10)
        launch=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','spatial_intelligence']
        for stage,steps in [('gate',8),('pilot',args.steps)]:
            stage_cfg=copy.deepcopy(cfg);stage_cfg['train']['steps']=steps
            stage_cfg['output']=str(run/stage/'training')
            path=run/f'{stage}.json';save(path,stage_cfg)
            execute(stage,launch+['train','--config',str(path)],
                    dict(env,EUREKASI_REWARD_TRACE=str(run/stage/'reward-trace')))
            from spatial_intelligence.spatial_rl_audit import audit_update
            evidence=audit_update(run/stage,4,expected_world=4,expected_steps=steps)
            save(run/f'{stage}.update-audit.json',evidence)
            # A full new pilot starts from the base checkpoint, not the gate adapter.
            if stage=='gate':
                eval_cfg=copy.deepcopy(cfg);eval_cfg['train']['mode']='sft'
                eval_cfg['protocol']['generation']={'do_sample':False,'max_new_tokens':64}
                eval_cfg['output']=str(run/'baseline-val');path=run/'baseline-val.json';save(path,eval_cfg)
                execute('baseline-val',launch+['infer','--config',str(path)])
        eval_cfg['model']['adapter']=str(run/'pilot/training/final')
        eval_cfg['output']=str(run/'post-val');path=run/'post-val.json';save(path,eval_cfg)
        execute('post-val',launch+['infer','--config',str(path)])
        from spatial_intelligence.boolean_metrics import summarize
        from spatial_intelligence.evaluation import compare
        from spatial_intelligence.io import read_jsonl
        rows=read_jsonl(dataset/'val.jsonl');reports=[];comparison={}
        for stage in ('baseline-val','post-val'):
            reports.append(json.loads((run/stage/'metrics.json').read_text()))
            comparison[stage]=summarize(rows,read_jsonl(run/stage/'metrics.details.jsonl'))
        compare(reports)  # Same protocol, data and scorer identities required.
        comparison['accuracy_delta']=comparison['post-val']['accuracy']-comparison['baseline-val']['accuracy']
        comparison['balanced_accuracy_delta']=comparison['post-val']['balanced_accuracy']-comparison['baseline-val']['balanced_accuracy']
        save(run/'comparison.json',comparison)
        state('complete',note='Checkpoint reloaded for identical heldout evaluation; compare saved metrics, not training reward')
        save(run/'completion.json',{'status':'complete','steps':args.steps,'gate':'nonconstant_rewards_and_nonzero_gradients',
                                  'evaluation':['baseline-val','post-val'],'exact_resume_supported':False})
    except Exception as exc:
        state('failed',error=str(exc));raise


if __name__=='__main__':main()
