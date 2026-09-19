"""Pinned, resumable server-side mirror downloads, independent lanes and receipts.

No credentials in manifests. curl is used because the deployment mirror rejects
Python urllib user agents. A failed optional dataset never cancels another lane.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl
import re
import shutil


def ranged_download(url, partial, size, connections):
    """Keep the contiguous prefix; parallel ranges are validated before append."""
    start=partial.stat().st_size if partial.exists() else 0
    if start>size:raise ValueError('Partial exceeds expected asset size')
    if start==size:return
    span=max(16*2**20,(size-start+connections-1)//connections)
    ranges=[(i,min(size-1,i+span-1)) for i in range(start,size,span)]
    def fetch(bounds):
        a,b=bounds;dest=partial.with_name(partial.name+f'.range-{a}-{b}')
        header=dest.with_suffix(dest.suffix+'.headers')
        if dest.exists() and dest.stat().st_size==b-a+1 and header.exists():
            if re.search(rf'content-range:\s*bytes {a}-{b}/{size}',header.read_text().lower()):return dest
        subprocess.run(['curl','-fLsS','--retry','8','--retry-all-errors','--retry-delay','5','--connect-timeout','30','--speed-time','120','--speed-limit','1024',
                        '-r',f'{a}-{b}','-D',str(header),'-o',str(dest),url],check=True)
        if dest.stat().st_size!=b-a+1 or not re.search(rf'content-range:\s*bytes {a}-{b}/{size}',header.read_text().lower()):
            raise ValueError('Server did not honor exact byte range; refusing corrupt concatenation')
        return dest
    with ThreadPoolExecutor(max_workers=connections) as pool:pieces=list(pool.map(fetch,ranges))
    with partial.open('ab') as dst:
        for piece in pieces:
            with piece.open('rb') as src:shutil.copyfileobj(src,dst,8*2**20)
    # Keep validated parts until operator cleanup; no broad deletion in downloader.


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def download(root, name, spec, endpoint):
    folder = root / spec['destination']
    folder.mkdir(parents=True, exist_ok=True)
    receipt = root / 'receipts' / (name + '.json')
    state = dict(asset=name, repo=spec['repo'], revision=spec['revision'], status='running', files=[])
    write_json(receipt, state)
    kind = 'datasets' if spec['kind'] == 'dataset' else 'models'
    url = f"{endpoint}/api/{kind}/{spec['repo']}/revision/{spec['revision']}?blobs=true"
    data = json.loads(subprocess.check_output(['curl','-fLsS','--retry','4','--max-time','120',url], text=True))
    if data['sha'] != spec['revision']:
        raise ValueError('Revision mismatch')
    prefix = 'datasets/' if kind == 'datasets' else ''
    chosen = [x for x in data['siblings'] if any(fnmatch.fnmatch(x['rfilename'], p) for p in spec['patterns'])]
    if not chosen:
        raise ValueError('No files matched')
    # Small annotations/configs first, before multi-GB payloads.
    chosen.sort(key=lambda x: x.get('size', 0))
    state['planned_bytes'] = sum(x.get('size',0) for x in chosen)
    write_json(receipt,state)
    for item in chosen:
        rel = item['rfilename']
        target = folder / rel
        if not target.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Unsafe remote path')
        target.parent.mkdir(parents=True, exist_ok=True)
        size = item.get('size', item.get('lfs',{}).get('size'))
        if not (target.is_file() and size is not None and target.stat().st_size == size):
            if target.exists():
                raise FileExistsError(f'Unexpected existing final file: {target}')
            partial = target.with_name(target.name+'.part')
            u = f"{endpoint}/{prefix}{spec['repo']}/resolve/{spec['revision']}/{rel}"
            print(f'DOWNLOAD {name} {rel} expected_bytes={size}',flush=True)
            connections=int(os.environ.get('EUREKASI_DOWNLOAD_CONNECTIONS','1'))
            if connections>1 and size and size>128*2**20:
                ranged_download(u,partial,size,connections)
            else:
                subprocess.run(['curl','-fL','--retry','8','--retry-delay','5','--connect-timeout','30',
                                '--speed-time','120','--speed-limit','1024','-C','-','-o',str(partial),u],check=True)
            if size is not None and partial.stat().st_size != size:
                raise ValueError(f'Size mismatch: {partial}')
            partial.replace(target)
        state['files'].append({'path':rel,'bytes':target.stat().st_size})
        write_json(receipt,state)
    state.update(status='complete',finished=time.time())
    write_json(receipt,state)
    return name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--assets',nargs='+')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--endpoint',default='https://hf-mirror.com')
    p.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1]/'configs/qwen35-assets.json')
    p.add_argument('--detach',action='store_true')
    p.add_argument('--wait-lock',action='store_true',help='Wait for an existing owner, then resume; never bypass its lock')
    args=p.parse_args()
    root=Path(args.root).resolve(); (root/'receipts').mkdir(parents=True,exist_ok=True)
    if args.detach:
        import sys
        (root/'logs').mkdir(parents=True,exist_ok=True)
        log=root/'logs'/('download-'+('-'.join(args.assets) if args.assets else args.catalog.stem)+'.log')
        with log.open('ab') as stream:
            child=subprocess.Popen([sys.executable,__file__,*[v for v in sys.argv[1:] if v!='--detach']],
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'log':str(log)}));return
    specs=json.loads(args.catalog.read_text())
    names=args.assets or sorted(specs,key=lambda n:specs[n]['priority'])
    # Same asset cannot be run twice concurrently; different lanes can proceed.
    locks={}
    for name in sorted(names):
        f=(root/'receipts'/f'{name}.lock').open('a')
        fcntl.flock(f,fcntl.LOCK_EX if args.wait_lock else fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks[name]=f
    errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(download,root,n,specs[n],args.endpoint):n for n in names}
        for job in as_completed(jobs):
            name=jobs[job]
            try:print('COMPLETE',job.result(),flush=True)
            except Exception as e:
                errors.append(name)
                path=root/'receipts'/f'{name}.json'
                state=json.loads(path.read_text()) if path.exists() else {}
                state.update(status='failed',error=str(e));write_json(path,state)
                print('FAILED',name,str(e),flush=True)
            finally:
                # A completed/failed lane must not block recovery until other assets finish.
                locks[name].close()
    if errors:raise SystemExit('Failed assets: '+','.join(errors))


if __name__=='__main__':main()
