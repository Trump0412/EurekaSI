"""Immutable post-RFT suite: independent benchmarks, resumable outputs, no training."""
import argparse,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from spatial_intelligence.dsr_sft_eval import read,write
from spatial_intelligence.rft_closeout import BENCHMARKS

def reasons(plan):
    errors=[]
    for dep in plan['dependencies']:
        value=read(dep['path']) if Path(dep['path']).exists() else {}
        if value.get('status') not in dep.get('statuses',['complete']):errors.append(str(dep['path'])+': '+str(value.get('status')))
        if dep.get('require_gpu_release') and value.get('gpu_work_finished') is not True:errors.append(str(dep['path'])+': GPU not released')
    return errors

def idle(gpus):
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
    values={int(a):int(b) for a,b in (line.split(',') for line in s.splitlines())}
    return all(g in values and values[g]<512 for g in gpus)

def launch(command,log,env,timeout):
    with log.open('ab') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
        try:
            result=child.wait(timeout=timeout)
            if result:raise RuntimeError('Command failed: '+str(result)+'; '+log.name)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def execute(plan):
    import fcntl
    root=Path(plan['root']);code=root/'code';states={}
    def status(name,**extra):write(root/'state.json',dict(status=name,pid=os.getpid(),time=time.time(),benchmarks=states,**extra))
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while reasons(plan):status('waiting_predecessors',waiting_for=reasons(plan));time.sleep(30)
        env=dict(os.environ,PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES=','.join(map(str,plan['gpus'])),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        # Preparation runs CPU-only in independent subprocesses. No slow download
        # blocks already-ready benchmarks. Terminal failures remain visible.
        pending=list(plan.get('benchmarks',BENCHMARKS));preps={}
        for name in pending:
            ready=Path(plan['prepared_root'])/(name+'.receipt.json')
            if not ready.exists():
                log=(root/'logs'/('prepare-'+name+'.log')).open('ab')
                preps[name]=subprocess.Popen([plan['python'],str(code/'scripts/prepare-rft-closeout.py'),'--plan',str(root/'plan.json'),'--benchmark',name],stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=''))
                log.close()
        deadline=time.monotonic()+plan.get('download_wait_seconds',172800)
        try:
            while pending:
                progressed=False
                for name in list(pending):
                    path=Path(plan['prepared_root'])/(name+'.receipt.json');data=read(path) if path.exists() else {}
                    if data.get('status') not in ('ready','failed'):
                        if time.monotonic()<deadline:continue
                        data=dict(status='failed',error='Preparation deadline expired')
                    if data['status']=='failed':states[name]=dict(status='failed_preparation',error=data.get('error'));pending.remove(name);progressed=True;continue
                    while reasons(plan) or not idle(plan['gpus']):status('waiting_gpus',benchmark=name);time.sleep(30)
                    try:
                        for smoke in (True,False):
                            phase='smoke' if smoke else 'full';receipt=root/'runs'/name/phase/'completion.json'
                            if receipt.exists() and read(receipt).get('accepted'):continue
                            args=[str(code/'scripts/evaluate-rft-closeout.py'),'--plan',str(root/'plan.json'),'--benchmark',name]+(['--smoke'] if smoke else [])
                            status('evaluating',benchmark=name,phase=phase)
                            launch([plan['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(plan['gpus']))]+args,root/'logs'/(name+'-'+phase+'.log'),env,plan.get('eval_timeout_seconds',259200))
                            launch([plan['python']]+args+['--merge'],root/'logs'/(name+'-'+phase+'-merge.log'),env,600)
                            if not read(receipt).get('accepted'):raise ValueError('Evaluation not accepted')
                        states[name]=dict(status='complete',metrics=read(receipt))
                    except Exception as e:states[name]=dict(status='failed_evaluation',error=repr(e))
                    pending.remove(name);progressed=True;status('between_benchmarks')
                if not progressed:status('waiting_data',pending=pending);time.sleep(30)
        finally:
            for child in preps.values():
                if child.poll() is None:child.terminate()
            for child in preps.values():
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        accepted=all(s['status']=='complete' for s in states.values())
        write(root/'completion.json',dict(status='complete' if accepted else 'complete_with_failures',
              accepted=accepted,queue_finished=True,gpu_work_finished=True,benchmarks=states,model_role=plan['model_role'],time=time.time()))
        status('complete' if accepted else 'complete_with_failures')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
    plan=read(a.plan);root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        if read(root/'plan.json')!=plan:raise ValueError('Armed plan changed; use new version')
    else:
        if (root/'code').exists():raise ValueError('Partial snapshot')
        for name in ('scripts','spatial_intelligence','configs','tests'):
            shutil.copytree(REPO/name,root/'code'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write(root/'plan.json',plan)
    (root/'logs').mkdir(exist_ok=True)
    if a.detach:
        with (root/'logs/supervisor.log').open('ab') as f:
            child=subprocess.Popen([plan['python'],str(root/'code/scripts/run-rft-closeout.py'),'--plan',str(root/'plan.json')],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,status='queued',root=str(root))))
    else:
        def stop(*unused):raise InterruptedError('Queue interrupted')
        signal.signal(signal.SIGTERM,stop)
        execute(plan)

if __name__=='__main__':main()
