"""Read-only ETA: preparation, measured SFT projection/live steps, baseline timing."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
root=Path(p.parse_args().root);report={}
progress=root/'receipts/train-render-progress.json'
if progress.exists():report['preparation']=json.loads(progress.read_text())
estimates=list((root/'runs').glob('diagnostic-sft-throughput-*/eta.json'))
if estimates:
    value=json.loads(max(estimates,key=lambda f:f.stat().st_mtime).read_text())
    report['training_projection']={k:v for k,v in value.items() if k!='strata'}
live=root/'runs/sft-spar234k-hound64k/live-eta.json'
report['formal_sft_started']=live.exists()
if live.exists():report['live_training']=json.loads(live.read_text())
times=[]
for file in (root/'runs/baseline-revsi32').glob('predictions.rank*.jsonl'):
    contract=json.loads(file.with_name(file.name.replace('predictions.','contract.').replace('.jsonl','.json')).read_text())
    seconds=sum(json.loads(s)['seconds_per_batch'] for s in file.read_text().splitlines())/contract['batch_size']
    times.append(seconds)
if times:report['repeat_eval_compute_hours_from_baseline']=max(times)/3600
print(json.dumps(report,indent=2))
