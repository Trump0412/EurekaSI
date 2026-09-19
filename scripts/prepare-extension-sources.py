"""Fetch locked legacy sources into separate directories and apply shipped patches."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import yaml
import fcntl

REPO=Path(__file__).resolve().parents[1]


def fetch(root,name):
    (root/'receipts').mkdir(parents=True,exist_ok=True)
    with (root/'receipts'/f'extension-source-{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _fetch(root,name)


def _fetch(root,name):
    item=yaml.safe_load((REPO/'catalog/sources.yaml').read_text())[name]
    dest=root/'external/extension-sources'/f'{name}-{item["commit"][:12]}'
    receipt=root/'receipts'/f'extension-source-{name}.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value.get('status')=='complete' and value.get('commit')==item['commit'] and dest.is_dir():
            for patch in item.get('patches',[]):
                if patch not in value.get('patches',[]):
                    check=subprocess.run(['git','apply','--check',str(REPO/patch)],cwd=dest,capture_output=True)
                    if check.returncode==0:
                        subprocess.run(['git','apply',str(REPO/patch)],cwd=dest,check=True)
                    else:
                        # Recover an interrupted installer only if the exact patch is already applied.
                        subprocess.run(['git','apply','--reverse','--check',str(REPO/patch)],cwd=dest,check=True)
            if (value.get('restored_files')!=item.get('restored_files',[]) or
                    not all((dest/r['path']).is_file() for r in item.get('restored_files',[]))):
                restore_files(dest,item)
            value['patches']=item.get('patches',[])
            value['restored_files']=item.get('restored_files',[])
            receipt.write_text(json.dumps(value,indent=2));return
        raise ValueError('Existing source receipt does not match selected commit')
    source=root/'sources'/f'{name}-{item["commit"]}.tar.gz';source.parent.mkdir(parents=True,exist_ok=True)
    if not source.exists():
        partial=source.with_suffix('.part')
        url=item['url'].replace('https://github.com/','https://codeload.github.com/')+'/tar.gz/'+item['commit']
        subprocess.run(['curl','-fLsS','--retry','3','--connect-timeout','30','--max-time','900','-o',str(partial),url],check=True)
        partial.replace(source)
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(source,'r:gz') as archive:
        for entry in archive:
            parts=Path(entry.name).parts
            if len(parts)<2:continue
            target=dest.joinpath(*parts[1:]).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError('Unsafe source archive')
            if entry.isdir():target.mkdir(parents=True,exist_ok=True)
            elif entry.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(entry) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
            else:raise ValueError('Unexpected link/special source entry')
    for patch in item.get('patches',[]):
        subprocess.run(['git','apply',str(REPO/patch)],cwd=dest,check=True)
    restore_files(dest,item)
    value={'status':'complete','commit':item['commit'],'path':str(dest),'patches':item.get('patches',[]),
           'restored_files':item.get('restored_files',[])}
    receipt.write_text(json.dumps(value,indent=2))


def restore_files(dest,item):
    for recovery in item.get('restored_files',[]):
        target=(dest/recovery['path']).resolve()
        if not target.is_relative_to(dest.resolve()):raise ValueError('Unsafe recovery path')
        url=item['url'].replace('https://github.com/','https://raw.githubusercontent.com/')+'/'+recovery['commit']+'/'+recovery['path']
        partial=target.with_suffix(target.suffix+'.recovery')
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['curl','-fLsS','--retry','3','--max-time','120','-o',str(partial),url],check=True)
        if target.exists():
            if target.read_bytes()!=partial.read_bytes():raise ValueError('Refusing to overwrite modified recovery source')
            partial.unlink()
        else:partial.replace(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('name',choices=['geowire','geopsro','geobridge','vggt'])
    a=p.parse_args();fetch(Path(a.root).resolve(),a.name)
