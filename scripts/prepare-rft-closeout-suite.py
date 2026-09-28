"""Prepare suite CPU inputs now, independently of occupied training GPUs."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.dsr_sft_eval import read,write
from spatial_intelligence.rft_closeout import BENCHMARKS
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
    plan=read(a.plan);root=Path(plan['prepared_root']);root.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (root/'supervisor.log').open('ab') as f:
            child=subprocess.Popen([sys.executable,__file__,'--plan',a.plan],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'))
        print(json.dumps(dict(pid=child.pid,status='preparing')));return
    import fcntl
    with (root/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def one(name):
            with (root/(name+'.log')).open('ab') as f:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('prepare-rft-closeout.py')),'--plan',a.plan,'--benchmark',name],stdout=f,stderr=f)
            return name,result.returncode
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            result=dict(pool.map(one,BENCHMARKS))
        write(root/'suite.json',dict(status='complete' if not any(result.values()) else 'complete_with_failures',return_codes=result))
if __name__=='__main__':main()
