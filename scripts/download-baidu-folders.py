"""Detached, bounded-concurrency downloads of explicit flat dataset folders.

Credentials stay in the client's private configuration. Raw client logs (which
may contain signed URLs) remain private and are never printed by this runner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def listing(text):
    summary=re.search(r'文件总数:\s*(\d+),\s*目录总数:\s*(\d+)',text)
    if not summary or int(summary[2]):raise ValueError('Expected an accessible flat folder; recursive inventory needed otherwise')
    rows=re.findall(r'^\s*\d+\s+([\d.]+(?:[KMGT]?B))\s+\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d\s+(.+?)\s*$',text,re.M)
    files={name:size for size,name in rows}
    if len(files)!=int(summary[1]) or len(files)!=len(rows) or not files:raise ValueError('Incomplete or duplicate listing')
    if any(Path(name).name!=name or name in ('.','..') for name in files):raise ValueError('Unsafe file name')
    total=re.search(r'总:\s*([\d.]+[KMGT]?B)',text)
    return {'files':files,'count':len(files),'display_size':total[1] if total else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--folders',nargs='+',required=True)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    os.umask(0o077);root=a.root.resolve(strict=True)
    if len(a.folders)>2 or any(not re.fullmatch(r'/[A-Za-z0-9_-]+',s) for s in a.folders):raise ValueError('Pass at most two explicit root folder names')
    private=root/'.private/baidu-downloads';private.mkdir(parents=True,exist_ok=True,mode=0o700);private.chmod(0o700)
    destination=root/'datasets/baidu';destination.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (private/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'destination':str(destination),'state':str(private)}));return
    lock=(private/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binary=root/'tools/baidupcs/v4.0.2/BaiduPCS-Go'
    env=dict(os.environ,BAIDUPCS_GO_CONFIG_DIR=str(root/'.private/baidu'),BAIDUPCS_GO_VERBOSE='0')
    def download(folder):
        name=folder[1:];state=private/f'{name}.json'
        report={'folder':folder,'status':'inventory','updated':time.time()};write(state,report)
        try:
            result=subprocess.run([str(binary),'ls',folder],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            inventory=listing(result.stdout.decode('utf-8',errors='replace'))
            write(private/f'{name}.inventory.json',inventory)
            expected=destination/name
            report.update(expected_files=inventory['count'],display_size=inventory['display_size'],destination=str(expected))
            for mode in ('locate','pcs'):
                command=[str(binary),'download','--saveto',str(destination),'--fullpath','--mode',mode,
                         '-p','2','-l','2','--retry','3',folder]
                with (private/f'{name}-{mode}.log').open('ab') as log:
                    proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    report.update(status='downloading',pid=proc.pid,mode=mode);write(state,report)
                    started=time.monotonic()
                    while proc.poll() is None:
                        files=[expected/f for f in inventory['files'] if (expected/f).is_file()]
                        report.update(observed_files=len(files),local_bytes=sum(f.stat().st_size for f in files),updated=time.time())
                        write(state,report)
                        if time.monotonic()-started>72*3600:
                            proc.terminate();proc.wait(timeout=30);raise TimeoutError('Download exceeded 72h')
                        time.sleep(15)
                files=[expected/f for f in inventory['files'] if (expected/f).is_file() and (expected/f).stat().st_size>0]
                partials=list(expected.rglob('*.downloading'))+list(expected.rglob('*.pcs*')) if expected.exists() else []
                report.update(observed_files=len(files),local_bytes=sum(f.stat().st_size for f in files),returncode=proc.returncode,updated=time.time())
                if proc.returncode==0 and len(files)==inventory['count'] and not partials:
                    report.update(status='download_exited_inventory_present',verification='Names/count/nonzero sizes; client checksum enabled where supported, not independent whole-dataset content audit')
                    write(state,report);return report
            report.update(status='incomplete',note='Inspect private logs locally; no raw URLs or credentials in public report')
        except Exception as exc:
            report.update(status='failed',error_type=type(exc).__name__)
        write(state,report);return report
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(download,a.folders))
    write(private/'summary.json',{'updated':time.time(),'datasets':results})


if __name__=='__main__':main()
