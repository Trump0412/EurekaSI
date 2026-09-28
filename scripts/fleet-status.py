"""Read-only multi-host GPU/stage snapshots using existing local SSH aliases.

Example: python scripts/fleet-status.py --node train-host /persistent/node-a
         --node rl-host /persistent/node-b
No credentials, machine paths, or experiment data are stored in the repository.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import shlex
import subprocess


REMOTE = r'''
import json, os, pathlib, socket, subprocess, sys, time
root=pathlib.Path(sys.argv[1])
report={'hostname':socket.gethostname(),'unix_time':time.time(),'root':str(root),'states':{},'runs':{}}
def read(path):
    try:
        item=json.loads(path.read_text())
        item['_mtime']=path.stat().st_mtime
        pid=item.get('pid')
        if isinstance(pid,int):
            proc=pathlib.Path('/proc')/str(pid)/'cmdline'
            item['_pid_exists']=proc.exists()
            if proc.exists():item['_pid_command']=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')[:500]
        return item
    except (OSError,ValueError) as exc:return {'read_error':str(exc)}
for path in sorted((root/'state').glob('*.json')):report['states'][path.name]=read(path)
for pattern in ('*/live-eta.json','*/status.json','*/completion.json'):
    for path in sorted((root/'runs').glob(pattern)):report['runs'][str(path.relative_to(root/'runs'))]=read(path)
for name,query in [('gpus',['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw']),
                   ('compute_processes',['--query-compute-apps=gpu_uuid,pid,process_name,used_memory'])]:
    try:
        result=subprocess.run(['nvidia-smi',*query,'--format=csv'],capture_output=True,text=True,timeout=10)
        report[name]={'returncode':result.returncode,'csv':result.stdout.strip(),'error':result.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as exc:report[name]={'error':str(exc)}
print(json.dumps(report))
'''


def ssh_command(host, root):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',host):
        raise ValueError('Use an existing SSH alias, not options or a shell expression')
    if not root.startswith('/') or '\n' in root or '\r' in root:
        raise ValueError('A Linux absolute experiment root is required')
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','ForwardAgent=no','-o','ForwardX11=no',host,
            'python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(root)]


def inspect(node):
    host,root=node
    command=ssh_command(host,root)
    try:
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=45)
        if result.returncode:
            return {'alias':host,'root':root,'error':result.stderr.strip(),'returncode':result.returncode}
        return {'alias':host,**json.loads(result.stdout)}
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        return {'alias':host,'root':root,'error':str(exc)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node',nargs=2,action='append',required=True,metavar=('SSH_ALIAS','ROOT'))
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=min(8,len(args.node))) as pool:
        results=list(pool.map(inspect,args.node))
    print(json.dumps(results,ensure_ascii=False,indent=2))
    if any('error' in item for item in results):raise SystemExit(1)


if __name__=='__main__':main()
