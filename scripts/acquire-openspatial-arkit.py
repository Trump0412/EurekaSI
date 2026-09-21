"""Bounded mirror acquisition of an ARKitScenes candidate pool, not training readiness."""
import argparse,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path

REPO='jdopensource/JoyAI-Image-OpenSpatial'
REVISION='9abfd33a80fa87230bbade5ae811616d980014c2'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True);p.add_argument('--create-inventory',action='store_true')
    p.add_argument('--root');p.add_argument('--reuse-root');p.add_argument('--count',type=int,default=100000)
    p.add_argument('--max-gib',type=float,default=20);p.add_argument('--detach',action='store_true')
    a=p.parse_args()
    if a.create_inventory:
        url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data?limit=1000'
        rows=json.load(urllib.request.urlopen(url,timeout=30))
        files=[dict(path=x['path'],size=x['size']) for x in rows if x['type']=='file' and x['path'].endswith('.parquet')]
        write(a.inventory,dict(repo=REPO,revision=REVISION,files=sorted(files,key=lambda x:x['path']),
            scope='First API page sorted by file path; deterministic bounded candidate pool, not uniform sample of the complete release'))
        print(json.dumps(dict(status='inventory_only',files=len(files))));return
    if not a.root or a.count<1 or a.max_gib<=0:p.error('Positive count/budget and root required')
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (root/'download.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'))
        print(json.dumps(dict(pid=child.pid,status='acquisition_started_not_training',root=str(root))));return
    import fcntl,pyarrow.parquet as pq
    lock=(root/'acquisition.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=json.loads(Path(a.inventory).read_text())
    if inventory['repo']!=REPO or inventory['revision']!=REVISION:raise ValueError('Unexpected source')
    snapshot=root/'inventory.json'
    if snapshot.exists() and json.loads(snapshot.read_text())!=inventory:raise ValueError('Changed inventory')
    write(snapshot,inventory)
    state=dict(status='downloading',pid=os.getpid(),requested_source_rows=a.count,source_rows=0,scanned_rows=0,
        bytes=0,files=[],training_ready=False,revision=REVISION,started=time.time())
    try:
        for item in inventory['files']:
            relative=Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe path')
            if state['bytes']+item['size']>a.max_gib*2**30:
                state['status']='blocked_byte_budget';break
            if shutil.disk_usage(root).free<30*2**30:
                state['status']='blocked_disk_reserve';break
            reuse=Path(a.reuse_root)/relative if a.reuse_root else None
            path=root/relative
            if reuse is not None and reuse.is_file() and reuse.stat().st_size==item['size']:
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists() and path.resolve()!=reuse.resolve():raise ValueError('Preserve existing non-reuse target')
                if not path.exists():path.symlink_to(reuse.resolve())
            elif path.exists():
                if path.stat().st_size!=item['size']:raise ValueError('Preserve mismatched existing file')
            else:
                path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix('.parquet.part')
                command=['curl','-fL','--retry','5','--retry-delay','5','--connect-timeout','30','--max-time','3600',
                    '--limit-rate','10M','-C','-','-o',str(part),f'https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/{item["path"]}']
                done=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                if done.returncode:raise RuntimeError('Download failed '+item['path'])
                if part.stat().st_size!=item['size']:raise ValueError('Size mismatch')
                part.replace(path)
            table=pq.read_table(path,columns=['data_source'])
            count=sum(x=='arkitscenes' for x in table.column('data_source').to_pylist())
            state['source_rows']+=count;state['scanned_rows']+=len(table);state['bytes']+=item['size']
            state['files'].append(dict(path=str(path.resolve()),source_path=item['path'],source_rows=count,bytes=item['size']))
            state['updated']=time.time();write(root/'acquisition.json',state)
            if state['source_rows']>=a.count:
                state['status']='candidate_pool_downloaded_not_prepared';break
        else:state['status']='blocked_inventory_exhausted'
    except Exception as exc:
        state.update(status='blocked_download_or_schema',error=repr(exc))
    state['updated']=time.time();write(root/'acquisition.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
