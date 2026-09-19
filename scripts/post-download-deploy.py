"""Persistent download barrier and failure-isolated, CPU-only environment setup."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def pending(root,names):
    result={}
    for name in names:
        file=root/'receipts'/f'{name}.json'
        status=json.loads(file.read_text()).get('status') if file.exists() else 'not_started'
        if status!='complete':result[name]=status
    return result


def stage(root,name,cmd,env,cwd=REPO):
    receipt=root/'state'/f'extension-{name}.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status')=='complete':return True
    log=root/'logs'/f'extension-{name}.log'
    write(receipt,{'status':'running','command':cmd,'started':time.time(),'log':str(log)})
    try:
        with log.open('ab') as output:
            child=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            write(receipt,{'status':'running','pid':child.pid,'command':cmd,'started':time.time(),'log':str(log)})
            try:code=child.wait(timeout=14400)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError('Stage exceeded 4-hour setup timeout; its process group was stopped')
        write(receipt,{'status':'complete' if code==0 else 'failed','returncode':code,
                       'finished':time.time(),'log':str(log),'command':cmd})
        return code==0
    except Exception as exc:
        write(receipt,{'status':'failed','error':str(exc),'log':str(log)});return False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--conda',required=True);p.add_argument('--detach',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--environments-now',action='store_true',help='Install independent environments without waiting for unrelated media')
    p.add_argument('--native-now',action='store_true',help='Wait only for legacy environment, then validate native sources')
    args=p.parse_args();root=Path(args.root).resolve()
    for name in ['logs','state']:(root/name).mkdir(parents=True,exist_ok=True)
    plan=json.loads((REPO/'configs/post-download.json').read_text())
    if args.check_only:print(json.dumps({'pending':pending(root,plan['wait_assets'])}));return
    if args.detach:
        with (root/'logs/post-download-deploy.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*[v for v in sys.argv[1:] if v!='--detach']],
                                   stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print('Post-download supervisor PID',child.pid);return
    lock_name='extension-native.lock' if args.native_now else ('extension-environments.lock' if args.environments_now else 'post-download.lock')
    lock=(root/'state'/lock_name).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,EUREKASI_ROOT=str(root),CONDA_BIN=str(Path(args.conda).resolve()),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    state=root/'state'/('extension-native.json' if args.native_now else 'post-download.json')
    if args.environments_now:
        def install_now(profile):
            return profile,stage(root,'env-'+profile,['bash',str(REPO/'scripts/bootstrap-extension.sh'),profile],env)
        with ThreadPoolExecutor(max_workers=2) as pool:
            print(json.dumps(dict(pool.map(install_now,plan['environment_profiles']))),flush=True)
        return
    while not args.native_now:
        waiting=pending(root,plan['wait_assets'])
        if not waiting:break
        write(state,{'status':'waiting_for_downloads','pending':waiting,'pid':os.getpid(),'updated':time.time(),
                     'note':'Failed downloads remain blocked until repaired; no false completion or host changes'})
        time.sleep(30)
    py=str(root/'envs/qwen35/bin/python')
    stage(root,'configs',[py,str(REPO/'scripts/prepare-extension-configs.py'),'--root',str(root)],env)
    def install(profile):
        return profile,stage(root,'env-'+profile,['bash',str(REPO/'scripts/bootstrap-extension.sh'),profile],env)
    if args.native_now:
        deadline=time.time()+14400
        while True:
            file=root/'state/extension-env-legacy.json'
            status=json.loads(file.read_text()).get('status') if file.exists() else 'not_started'
            if status in ('complete','failed'):break
            if time.time()>deadline:raise TimeoutError('Legacy environment not ready after four hours')
            write(state,{'status':'waiting_for_legacy_environment','pid':os.getpid(),'updated':time.time()})
            time.sleep(15)
        outcomes={'legacy':status=='complete'}
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:outcomes=dict(pool.map(install,plan['environment_profiles']))
    if outcomes.get('legacy'):
        native=str(root/'envs/legacy-geo/bin/python')
        stage(root,'opd-opsd',[native,'-m','pytest','tests/test_hf_training.py','tests/test_objectives.py','-q'],env)
        for name,sub in [('geowire','geowire'),('geopsro','.'),('geobridge','.')]:
            if not stage(root,'source-'+name,[py,str(REPO/'scripts/prepare-extension-sources.py'),'--root',str(root),name],env):continue
            receipt=json.loads((root/'receipts'/f'extension-source-{name}.json').read_text())
            cwd=Path(receipt['path'])/sub
            stage(root,name,[native,'-m','pytest','tests','-q'],dict(env,PYTHONPATH=str(cwd/'src') if name=='geobridge' else str(cwd)),cwd)
    records={p.stem:json.loads(p.read_text()).get('status') for p in (root/'state').glob('extension-*.json')}
    write(state,{'status':'finished_with_failures' if 'failed' in records.values() else 'environment_setup_complete',
                 'stages':records,'gpu_rl_validated':False,'extra_training_launched':False,'finished':time.time()})


if __name__=='__main__':main()
