"""Own one allocation: opportunistic GeoRoute, priority RFT, then resume.

Private plans must transfer ownership of the waiting RFT/ablation controllers
before launch. Active SFT is never stopped. Formal workers pause cooperatively
at a complete checkpoint, not via signals or shortened training budgets.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def ready(requirements):
    for item in requirements:
        try:value=read(item['path'])
        except (OSError,ValueError):return False
        if any(value.get(k)!=v for k,v in item.get('equals',{}).items()):return False
    return True


def choose_action(priority_ready, preparation_ready):
    if priority_ready:return 'priority_rft'
    return 'georoute_gate' if preparation_ready else 'wait_data_or_priority'


class Queue:
    def __init__(self,plan):
        self.plan=plan;self.root=Path(plan['root']);self.child=None
        self.pause=self.root/'pause-request.json';self.priority_finished=False
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,plan['gpus'])),
                      OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
                      PYTHONPATH=plan['code'])

    def state(self,status,**extra):
        write(self.root/'state.json',dict(status=status,pid=os.getpid(),updated=time.time(),
            gpu_work_finished=False,**extra))

    def idle(self):
        p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                         text=True,capture_output=True,check=True,timeout=20)
        values={int(a):int(b) for a,b in (line.split(',') for line in p.stdout.splitlines())}
        return all(values.get(g,100000)<512 for g in self.plan['gpus'])

    def wait(self,requirements,status):
        while not ready(requirements):self.state(status);time.sleep(15)

    def run(self,name,command,preempt=False):
        logs=self.root/'logs';logs.mkdir(exist_ok=True)
        self.last_started=time.time()
        with (logs/(name+'.log')).open('ab') as log:
            self.child=subprocess.Popen(command,cwd=self.plan['code'],env=self.env,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            try:
                while self.child.poll() is None:
                    if preempt and not self.priority_finished and ready(self.plan['priority_requirements']):
                        if not self.pause.exists():write(self.pause,dict(reason='priority_rft_ready',requested=time.time()))
                    self.state('running_'+name,child_pid=self.child.pid,pause_requested=self.pause.exists())
                    time.sleep(10)
                return self.child.returncode
            finally:
                if self.child.poll() is None:
                    # Coordinator failure is not a claim of a successful pause.
                    os.killpg(self.child.pid,signal.SIGTERM)
                    try:self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
                self.child=None

    def paused(self,stage):
        if not self.pause.exists() or not ready(stage['pause_acceptance']):return False
        receipt=Path(stage['pause_acceptance'][0]['path'])
        if receipt.stat().st_mtime < self.last_started:return False
        value=read(receipt)
        if Path(value.get('request','')).resolve()!=self.pause.resolve():return False
        from spatial_intelligence.cooperative_pause import validate_full_checkpoint
        validate_full_checkpoint(value['checkpoint'])
        # torchrun translates rank exit75 into launcher exit1. The fresh verified
        # receipt, not that generic launcher code, identifies cooperative pause.
        return True

    def priority(self):
        if self.priority_finished:return
        if ready(self.plan['rft_completion']):
            self.priority_finished=True
            if self.pause.exists():self.pause.rename(self.root/('pause-consumed-'+str(time.time_ns())+'.json'))
            return
        self.wait(self.plan['priority_requirements'],'waiting_priority_inputs')
        while not self.idle():self.state('waiting_allocated_gpus');time.sleep(15)
        rc=self.run('priority_rft',self.plan['rft_command'])
        if rc or not ready(self.plan['rft_completion']):raise RuntimeError('Priority RFT did not complete; preserve route checkpoints')
        self.priority_finished=True
        if self.pause.exists():
            self.pause.rename(self.root/('pause-consumed-'+str(time.time_ns())+'.json'))

    def route(self):
        for stage in self.plan['route_stages']:
            if ready(stage['completion']):continue
            if not self.priority_finished and ready(self.plan['priority_requirements']):self.priority()
            while not self.idle():self.state('waiting_allocated_gpus');time.sleep(15)
            command=stage['train_command']+['--pause-request',str(self.pause)]
            rc=self.run(stage['name'],command,preempt=True)
            if rc and self.paused(stage):
                self.priority()
                rc=self.run(stage['name']+'-resumed',command)
            if rc:raise RuntimeError('GeoRoute training failed: '+stage['name'])
            if self.run(stage['name']+'-reload',stage['verify_command']):raise RuntimeError('GeoRoute reload failed')
            if not ready(stage['completion']):raise RuntimeError('GeoRoute completion not accepted')

    def execute(self):
        import fcntl
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/'allocation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not ready(self.plan['ownership_receipt']):raise RuntimeError('Waiting-controller ownership not transferred')
            self.wait(self.plan['local_predecessor'],'waiting_current_sft_and_evaluation')
            route_started=(self.root/'route-started.json').exists()
            while True:
                action=choose_action(ready(self.plan['priority_requirements']),ready(self.plan['preparation_requirements']))
                if action=='priority_rft':break
                if action=='georoute_gate':
                    while not self.idle():self.state('waiting_allocated_gpus');time.sleep(15)
                    if ready(self.plan['priority_requirements']):break
                    gate_command=list(self.plan['gate_command'])
                    if not ready(self.plan['runtime_acceptance']):
                        pos=gate_command.index('--output')+1
                        gate_command[pos]+='-'+str(time.time_ns())
                        gate_command+=['--yield-request',str(self.pause)]
                        gate_rc=self.run('georoute_runtime_gate',gate_command,preempt=True)
                    else:gate_rc=0
                    if gate_rc:
                        self.state('route_gate_failed_waiting_priority')
                        break
                    if not ready(self.plan['runtime_acceptance']):raise RuntimeError('Runtime gate missing actual acceptance')
                    # A gate may finish after priority work became ready.
                    if ready(self.plan['priority_requirements']):break
                    route_started=True
                    write(self.root/'route-started.json',dict(time=time.time()))
                    try:self.route()
                    except Exception as exc:
                        write(self.root/'route-error.json',dict(error=repr(exc),time=time.time()))
                    break
                self.state(action);time.sleep(15)
            self.priority()
            # Only resume the experiment that actually entered the gap.
            # Untouched route work remains in its original post-RFT queue.
            if route_started and not (self.root/'route-error.json').exists():self.route()
            rc=self.run('reward_ablation',self.plan['ablation_command'])
            if rc:raise RuntimeError('Reward ablation failed')
            write(self.root/'state.json',dict(status='complete',pid=os.getpid(),gpu_work_finished=True,
                route_started_in_gap=route_started,route_error=(self.root/'route-error.json').exists(),updated=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    args=p.parse_args();plan=read(args.plan);queue=Queue(plan)
    def stop(signum,frame):raise InterruptedError('Gap owner interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:queue.execute()
    except Exception as exc:
        queue.state('failed',error=repr(exc));raise


if __name__=='__main__':main()
