"""Pinned public asset acquisition, no GPU, no installation, no implied readiness."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def write(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reuse-root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=2: p.error('workers must be 1..2')
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (root/'download.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'))
        print(json.dumps({'pid':child.pid,'receipt':str(root/'assets-receipt.json')}));return
    import fcntl
    lock=(root/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(a.config.read_text());write(root/'config.snapshot.json',config)
    inventory=json.loads((a.config.parent/config['inventory_file']).read_text())
    write(root/'assets-receipt.json',{'status':'downloading','pid':os.getpid(),'started':time.time(),
          'training_ready':False,'assets':list(config['assets'])})
    state=root/'state';state.mkdir(exist_ok=True)
    reused={k:[{'path':str((a.reuse_root/path).resolve()),'exists':(a.reuse_root/path).exists()} for path in paths] for k,paths in config['reuse'].items()}
    write(root/'reuse-audit.json',reused)
    # Readiness remains closed until converters, provenance, media, and all
    # benchmark source-group exclusions have independent acceptance receipts.
    write(root/'data-readiness.json',{'status':'blocked_preparation','train_manifest':None,
          'eval_manifests':{},'media_verified':False,'leakage_checked':False,
          'geometry_source_capability':{'vggt':'existing external source','pi3':'acquiring original Pi3; not Pi3X'},
          'gaps':['VLM3R corrected label mapping and source-scene conversion',
                  'OpenSpatial verified five-stratum selection and media export',
                  'all-benchmark source lineage exclusion and1%source-scene split',
                  'benchmark-specific extraction and official scoring adapters'],
          'selection':config['selection']})
    def fetch(name, spec):
        receipt=state/(name+'.json');dest=root/spec['destination'];dest.mkdir(parents=True,exist_ok=True)
        result={'name':name,'repo':spec['repo'],'revision':spec['revision'],'license':spec['license'],
                'status':'downloading','started':time.time(),'destination':str(dest),'files':[]}
        write(receipt,result)
        try:
            if spec['kind']=='github':
                files=[{'rfilename':'source.tar.gz','size':None,'url':f"https://codeload.github.com/{spec['repo']}/tar.gz/{spec['revision']}"}]
            else:
                # Reviewed public/non-gated inventory avoids requiring direct HF
                # API connectivity from a mirror-only compute node. Download
                # URLs still include the immutable revision; 401/403 fail closed.
                prefix='datasets/' if spec['kind']=='dataset' else ''
                files=[dict(rfilename=item['file'],size=item['size'],url=f"{config['endpoint']}/{prefix}{spec['repo']}/resolve/{spec['revision']}/{item['file']}")
                       for item in inventory[name] if any(fnmatch.fnmatchcase(item['file'],pattern) for pattern in spec['patterns'])]
                if not files: raise ValueError('No files match pinned selection')
            for item in files:
                relative=Path(item['rfilename'])
                if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe asset path')
                target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
                expected=item.get('size') or (item.get('lfs') or {}).get('size')
                if target.exists() and expected and target.stat().st_size==expected:
                    mode='reused_size_verified'
                else:
                    if target.exists(): raise ValueError('Existing target lacks matching size; preserve and inspect')
                    part=target.with_suffix(target.suffix+'.part')
                    if spec['kind']=='github' and part.exists():
                        part.rename(part.with_suffix(part.suffix+f'.interrupted-{time.time_ns()}'))
                    command=['curl','-fL','--retry','5','--retry-delay','5','--connect-timeout','30',
                             '--max-time','21600','--limit-rate','20M','-C','-','-o',str(part),item['url']]
                    run=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    if run.returncode: raise RuntimeError(f'Download exit {run.returncode}: {relative}')
                    if expected and part.stat().st_size!=expected: raise ValueError('Downloaded size mismatch')
                    part.replace(target);mode='downloaded'
                result['files'].append({'file':str(relative),'bytes':target.stat().st_size,'mode':mode})
                result['updated']=time.time();write(receipt,result)
            result.update(status='downloaded_not_prepared',completed=time.time())
        except Exception as error: result.update(status='failed',error_type=type(error).__name__,error=str(error))
        write(receipt,result);return result
    ordered=sorted(config['assets'].items(),key=lambda pair: 0 if pair[0] in ('pi3_weights','pi3_source','openspatial_probe') else 1)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        results=list(executor.map(lambda pair:fetch(*pair),ordered))
    write(root/'assets-receipt.json',{'status':'downloaded_not_prepared' if all(x['status']=='downloaded_not_prepared' for x in results) else 'incomplete',
          'updated':time.time(),'assets':results,'reused':reused,'training_ready':False})


if __name__=='__main__':main()
