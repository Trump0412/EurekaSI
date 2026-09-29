"""Explicit opt-in cold-start lifecycle; preparing/downloading never arms GPU work."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from spatial_intelligence.rft_coldstart import read, rows, write, write_rows, finalize


def run(plan, pilot=False):
    import fcntl
    root=Path(plan['root']); (root/'logs').mkdir(exist_ok=True)
    def state(status, **extra): write(root/'pipeline-state.json',dict(status=status,updated=time.time(),**extra))
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,plan['gpus'])),
            PYTHONPATH=str(REPO),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        def command(script, *args, teacher=False):
            py=plan['teacher_python'] if teacher else plan['python']
            return [py,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node='+str(len(plan['gpus'])),str(REPO/'scripts'/script),'--plan',str(root/'plan.json'),*args]
        def launch(cmd, name):
            state('running_'+name,command=cmd)
            with (root/'logs'/f'{name}.log').open('ab') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state('running_'+name,pid=child.pid,command=cmd)
                try: rc=child.wait(timeout=30*86400)
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
                    raise
                if rc: raise RuntimeError(name+' exited '+str(rc))
        try:
            if not pilot and not (root/'human-review.json').exists():
                state('waiting_for_pilot_visual_review'); return
            # Wait for the existing full allocation, not a transient idle gap.
            deadline=time.monotonic()+30*86400
            while True:
                if time.monotonic()>deadline: raise TimeoutError('GPU predecessor wait exceeded 30 days')
                dep=Path(plan['dependency'])
                ready=(dep.exists() and read(dep).get('gpu_work_finished') is True and
                    read(dep).get('status') in ('complete','complete_with_failures','failed'))
                if ready:
                    memory=subprocess.check_output(['nvidia-smi','--id='+env['CUDA_VISIBLE_DEVICES'],
                        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
                    if len(memory)==len(plan['gpus']) and all(int(v)<500 for v in memory): break
                state('waiting_for_existing_gpu_pipeline'); time.sleep(30)
            if pilot or not (root/'annotations.jsonl').exists():
                launch(command('annotate-rft-coldstart.py','--authorize-annotation',
                    *(['--pilot'] if pilot else []),teacher=True),'annotation-pilot' if pilot else 'annotation')
            if pilot:
                state('waiting_for_pilot_visual_review',required='Review all 200 pilot IDs, then provide human-review.json'); return
            if not (root/'annotations.jsonl').exists():
                annotations=sum((rows(root/f'annotations-all.rank{rank}.jsonl') for rank in range(len(plan['gpus']))),[])
                write_rows(root/'annotations.jsonl',annotations)
            if not (root/'acceptance.json').exists(): finalize(plan)
            if not (root/'coldstart/completion.json').exists():
                launch(command('train-rft-coldstart.py','--mode','train'),'coldstart-sft')
            if not (root/'coldstart/validation.json').exists():
                launch(command('train-rft-coldstart.py','--mode','validate'),'coldstart-validation')
            if read(root/'coldstart/validation.json').get('status')!='accepted':
                raise ValueError('Cold-start validation format acceptance failed')
            # Temporal-origin flags are not automatically repaired by a teacher.
            # Preserve the old RFT pool until a documented audit resolves them.
            audit=root/'rft-timebase-review.json'
            if not audit.exists() or read(audit).get('status')!='accepted' or not read(audit).get('evidence'):
                state('blocked_rft_timebase_audit',reason='Resolve source-time versus clip-time flags before full RFT'); return
            review=read(audit)
            data_receipt=review.get('verified_data_receipt',plan['data_receipt'])
            final=read(root/'coldstart/completion.json')
            rft=dict(root=str(root/'rft'),python=plan['python'],model_checkpoint=plan['model_checkpoint'],
                processor=plan['processor'],vggt_source=plan['vggt_source'],sft_receipt=plan['sft_receipt'],
                coldstart_policy=final['checkpoint'],coldstart_receipt=str(root/'coldstart/completion.json'),
                scientific_config=plan['scientific_config'],data_receipt=data_receipt,
                pair_id='coldstart-mixed-v1',gpus=plan['gpus'],group_size=8,prompts_per_update=16,
                seed=3407,rollout_micro_cap=2,allow_saturated_format_after_coldstart=True,
                dependencies=[dict(path=str(root/'coldstart/completion.json'),statuses=['complete'])],
                jobs=[dict(name='mixed')])
            spec=importlib.util.spec_from_file_location('rft_queue',REPO/'scripts/run-geometry-rft-queue.py')
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            module.validate_plan(rft); frozen=module.snapshot(rft,REPO)
            launch([plan['python'],str(frozen/'scripts/run-geometry-rft-queue.py'),
                '--plan',str(root/'rft/plan.json')],'rft-gate-train-evaluate')
            result=read(root/'rft/state/rft-queue.json')
            state(result['status'],rft=result,gpu_work_finished=True)
        except Exception as exc:
            state('failed',error=repr(exc),gpu_work_finished=True); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',required=True)
    p.add_argument('--execute-authorized',action='store_true'); p.add_argument('--pilot',action='store_true')
    p.add_argument('--detach',action='store_true'); a=p.parse_args()
    if not a.execute_authorized: p.error('Prepared only: explicit user instruction is required to execute annotation')
    plan=read(a.plan)
    if a.detach:
        root=Path(plan['root']); (root/'logs').mkdir(exist_ok=True)
        with (root/'logs/supervisor.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[v for v in sys.argv[1:] if v!='--detach']],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,status='launched_not_gpu_acceptance')))
    else:
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(InterruptedError('Supervisor stopped')))
        run(plan,a.pilot)
