"""Publish dependency completion only after every named training/eval queue passes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--receipt',required=True,action='append')
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    if args.detach:
        with (args.root/'watch.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[v for v in sys.argv[1:] if v!='--detach']],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,status='watcher_launched')))
        return
    import fcntl
    with (args.root/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            states={}
            for path in args.receipt:
                try: states[path]=json.loads(Path(path).read_text())
                except (OSError,ValueError): states[path]={'status':'pending'}
            accepted=all(v.get('status')=='complete' and v.get('gpu_work_finished') is True
                         for v in states.values())
            write(args.root/'watch-state.json',dict(status='complete' if accepted else 'waiting',
                pid=os.getpid(),time=time.time(),sources={k:v.get('status') for k,v in states.items()}))
            if accepted:
                write(args.root/'completion.json',dict(status='complete',accepted=True,
                    gpu_work_finished=True,receipts=args.receipt,time=time.time()))
                return
            time.sleep(30)


if __name__=='__main__': main()
