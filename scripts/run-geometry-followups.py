"""Durable private-plan DAG for consecutive geometry studies.

Preparation may run now; paid GPU stages require successful predecessors,
immutable implementation/data receipts, an idle allocation and explicit commands.
An incomplete implementation remains visibly waiting, never a successful run.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def validate(plan):
    for key in ('root','python','dependencies','stages','gpus'):
        if not plan.get(key): raise ValueError('Missing '+key)
    if len(set(plan['gpus']))!=len(plan['gpus']): raise ValueError('Duplicate GPUs')
    if any(not isinstance(g,int) or isinstance(g,bool) or g<0 for g in plan['gpus']):
        raise ValueError('Explicit numeric GPU allocation required')
    names=[stage['name'] for stage in plan['stages']]
    if len(names)!=len(set(names)): raise ValueError('Duplicate stage IDs')
    fits=False
    for stage in plan['stages']:
        if stage['study'] not in ('georoute','geofits'): raise ValueError('Unknown study')
        if stage['study']=='geofits': fits=True
        elif fits: raise ValueError('GeoRoute must finish before GeoFits')
        if stage.get('use_lora') is not False: raise ValueError('All new studies explicitly disable LoRA')
        if not stage.get('requirements'): raise ValueError('Missing implementation/data gates')
        for dep in stage.get('after',[]):
            if dep not in names[:names.index(stage['name'])]: raise ValueError('Forward or unknown DAG dependency')
        for command in stage.get('commands',[]):
            if not isinstance(command,list) or not command or not all(isinstance(v,str) for v in command):
                raise ValueError('Commands must be explicit argv, never shell strings')
        if not stage.get('receipt'): raise ValueError('Each stage needs an acceptance receipt')
    return plan


def requirement_status(requirement):
    path=Path(requirement['path'])
    if not path.is_file(): return False,'missing '+str(path)
    try: value=read(path)
    except (ValueError,OSError) as error: return False,'unreadable '+str(error)
    for key,wanted in requirement.get('equals',{'status':'complete'}).items():
        if value.get(key)!=wanted: return False,f'{path.name}: {key}={value.get(key)!r}, expected {wanted!r}'
    return True,'accepted'


def accepted(path):
    if not Path(path).is_file(): return False
    value=read(path)
    return value.get('status')=='complete' and value.get('accepted') is True


def snapshot(plan):
    root=Path(plan['root']); root.mkdir(parents=True,exist_ok=True)
    config=root/'plan.json'; code=root/'code'
    if config.exists():
        if read(config)!=plan: raise ValueError('Changed armed plan requires a new version')
        if not code.is_dir(): raise ValueError('Incomplete prior snapshot; inspect manually')
        return code
    if code.exists(): raise ValueError('Unowned code snapshot exists')
    code.mkdir()
    for name in ('scripts','spatial_intelligence','configs','requirements','tests'):
        if (REPO/name).exists(): shutil.copytree(REPO/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name,value in plan.get('stage_plans',{}).items():
        if Path(name).name!=name or not name.endswith('.json'): raise ValueError('Unsafe stage plan filename')
        write(root/'inputs'/name,value)
    write(config,plan)
    return code


class Queue:
    def __init__(self,plan):
        self.plan=plan; self.root=Path(plan['root']); self.code=self.root/'code'
        self.states={}; self.child=None

    def status(self,status,**more):
        write(self.root/'state/followups.json',dict(status=status,pid=os.getpid(),time=time.time(),
              stages=self.states,gpu_work_finished=False,
              implementation_status=self.plan.get('implementation_status','not_reported'),
              unimplemented_stages=[stage['name'] for stage in self.plan['stages'] if not stage.get('commands')],**more))

    def idle(self):
        result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
            text=True,capture_output=True,check=True,timeout=20)
        memory={int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in result.stdout.splitlines())}
        return all(index in memory and memory[index]<512 for index in self.plan['gpus'])

    def wait(self,requirements,status,stage=None):
        while True:
            reasons=[reason for ok,reason in map(requirement_status,requirements) if not ok]
            if not reasons: return
            self.status(status,stage=stage,waiting_for=reasons)
            time.sleep(30)

    def launch(self,stage,command,index):
        expanded=[item.replace('{python}',self.plan['python']).replace('{code}',str(self.code))
                  .replace('{root}',str(self.root)) for item in command]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,self.plan['gpus'])),
                 PYTHONPATH=str(self.code),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        logs=self.root/'logs'; logs.mkdir(exist_ok=True)
        with (logs/f"{stage['name']}-{index}.log").open('ab') as log:
            self.child=subprocess.Popen(expanded,cwd=self.code,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            try:
                while self.child.poll() is None:
                    self.status('running_stage',stage=stage['name'],command_index=index,child_pid=self.child.pid)
                    if time.monotonic()-started>stage.get('timeout_seconds',2592000): raise TimeoutError(stage['name'])
                    time.sleep(15)
                if self.child.returncode: raise RuntimeError(f'{stage["name"]} command exited {self.child.returncode}')
            finally:
                if self.child.poll() is None:
                    os.killpg(self.child.pid,signal.SIGTERM)
                    try: self.child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid,signal.SIGKILL); self.child.wait()
                self.child=None

    def execute(self):
        import fcntl
        (self.root/'state').mkdir(exist_ok=True)
        with (self.root/'state/followups.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.wait(self.plan['dependencies'],'waiting_previous_paper')
            for stage in self.plan['stages']:
                if any(self.states.get(dep,{}).get('status')!='complete' for dep in stage.get('after',[])):
                    self.states[stage['name']]={'status':'blocked_dependency'}; continue
                if accepted(stage['receipt']):
                    self.states[stage['name']]={'status':'complete','reused':True}; continue
                self.wait(stage['requirements'],'waiting_preparation',stage['name'])
                if not stage.get('commands'):
                    self.states[stage['name']]={'status':'blocked_implementation','reason':'No accepted executable entrypoint'}
                    continue
                while not self.idle():
                    self.status('waiting_allocated_gpus',stage=stage['name']); time.sleep(30)
                # Cross-study/node conditions are checked again immediately before launch.
                self.wait(self.plan['dependencies']+stage['requirements'],'waiting_preparation',stage['name'])
                try:
                    for i,command in enumerate(stage['commands']): self.launch(stage,command,i)
                    if not accepted(stage['receipt']): raise ValueError('Command success without accepted scientific receipt')
                    self.states[stage['name']]={'status':'complete'}
                except Exception as error:
                    self.states[stage['name']]={'status':'failed','error':repr(error)}
                write(self.root/'state'/f"{stage['name']}.json",self.states[stage['name']])
            complete=all(value['status']=='complete' for value in self.states.values())
            write(self.root/'state/followups.json',dict(status='complete' if complete else 'complete_with_failures',
                accepted=complete,stages=self.states,gpu_work_finished=True,pid=os.getpid(),time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True); parser.add_argument('--detach',action='store_true')
    parser.add_argument('--authorize-run',action='store_true',help='Only supply after an explicit user launch instruction')
    args=parser.parse_args()
    if not args.authorize_run:
        parser.error('Manual hold: an explicit user launch instruction and --authorize-run are required')
    plan=validate(read(args.plan)); code=snapshot(plan)
    if args.detach:
        root=Path(plan['root']); (root/'logs').mkdir(exist_ok=True)
        with (root/'logs/supervisor.log').open('ab') as log:
            child=subprocess.Popen([plan['python'],str(code/'scripts'/Path(__file__).name),'--plan',str(root/'plan.json'),'--authorize-run'],
                cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':child.pid,'status':'waiting_queue_launched_not_training','root':str(root)}))
    else:
        def stop(signum,frame): raise InterruptedError('Supervisor interrupted')
        signal.signal(signal.SIGTERM,stop)
        Queue(plan).execute()


if __name__=='__main__': main()
