"""Read-only status, unique downloaded bytes and recent evaluation throughput."""
import argparse
from collections import defaultdict
import fnmatch
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


def unique_bytes(folder, patterns):
    intervals=defaultdict(list)
    for path in folder.rglob('*'):
        if not path.is_file() or path.name.endswith('.headers'):continue
        try:size=path.stat().st_size
        except FileNotFoundError:continue # Downloader can atomically rename .part.
        rel=str(path.relative_to(folder))
        match=re.fullmatch(r'(.*)\.part\.range-(\d+)-(\d+)',rel)
        if match:
            rel,a,b=match.groups();a=int(a);b=int(b)
            end=min(b+1,a+size)
        else:
            rel=rel.removesuffix('.part');a=0;end=size
        if any(fnmatch.fnmatch(rel,p) for p in patterns):intervals[rel].append((a,end))
    total=0
    for spans in intervals.values():
        end=0
        for a,b in sorted(spans):total+=max(0,b-max(a,end));end=max(end,b)
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    root=Path(parser.parse_args().root)
    catalog=json.loads((Path(__file__).resolve().parents[1]/'configs/qwen35-assets.json').read_text())
    report={'unix_time':time.time(),'assets':{},'stages':{},'evaluations':{}}
    for name,spec in catalog.items():
        p=root/'receipts'/f'{name}.json';r=json.loads(p.read_text()) if p.exists() else {}
        report['assets'][name]={'status':r.get('status','not_started'),'planned_gb':r.get('planned_bytes',0)/1e9,
                               'unique_downloaded_gb':round(unique_bytes(root/spec['destination'],spec['patterns'])/1e9,3),
                               'error':r.get('error')}
    for p in sorted((root/'state').glob('*.json')):
        r=json.loads(p.read_text());report['stages'][p.stem]={'status':r.get('status'),'error':r.get('error'),'log':r.get('log')}
    for folder in (root/'runs').glob('*'):
        counts=[];eta=[]
        for pred in folder.glob('predictions.rank*.jsonl'):
            contract=folder/pred.name.replace('predictions.','contract.').replace('.jsonl','.json')
            if not contract.exists():continue
            cfg=json.loads(contract.read_text());rows=[]
            for line in pred.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass # Display only; resume/scoring remains strict.
            counts.append(len(rows))
            if rows:
                sec=statistics.median(x['seconds_per_batch'] for x in rows[-100:])
                eta.append(max(0,cfg['rows']/cfg['world']-len(rows))*sec/cfg['batch_size'])
        if counts:report['evaluations'][folder.name]={'predictions':sum(counts),'remaining_minutes_estimate':round(max(eta,default=0)/60,1)}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw','--format=csv'],capture_output=True,text=True)
    report['gpu']=gpu.stdout.strip();report['gpu_query_error']=gpu.stderr.strip()
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
