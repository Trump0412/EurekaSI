"""Aggregate required accepted receipts; cannot fabricate pending benchmarks."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--stage',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());stage=next(s for s in plan['stages'] if s['name']==a.stage)
    evidence=[]
    for req in stage['requirements']:
        value=json.loads(Path(req['path']).read_text())
        if any(value.get(k)!=v for k,v in req['equals'].items()):raise ValueError('Incomplete prerequisite: '+req['path'])
        evidence.append(req['path'])
    path=Path(stage['receipt']);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(status='complete',accepted=True,evidence=evidence),indent=2))

if __name__=='__main__':main()
