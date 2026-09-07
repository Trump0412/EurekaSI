"""Paired scene-cluster bootstrap for diagnostic per-example correctness."""
import json
from pathlib import Path
import numpy as np
from .data import key,load_samples,content_fingerprint
from .evaluation import compare
from .io import read_jsonl,write_json


def cluster_bootstrap(differences,groups,replicates=5000,seed=3407):
    if len(differences)!=len(groups) or len(differences)<2 or replicates<100:raise ValueError('Need matched samples and >=100 replicates')
    if any(g is None for g in groups):raise ValueError('Scene IDs required; examples within a scene are correlated')
    keys=sorted(set(groups));sums=np.array([sum(d for d,g in zip(differences,groups) if g==k) for k in keys]);counts=np.array([groups.count(k) for k in keys])
    if len(keys)<2:raise ValueError('Need at least two independent scene clusters')
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(replicates):
        sampled=rng.integers(0,len(keys),len(keys));draws.append(sums[sampled].sum()/counts[sampled].sum())
    lo,hi=np.quantile(draws,[.025,.975])
    return {'delta_b_minus_a':float(np.mean(differences)),'ci95':[float(lo),float(hi)],'n':len(groups),'scenes':len(keys),'replicates':replicates,'seed':seed}


def paired(a,b,output,replicates=5000,seed=3407):
    a,b=Path(a),Path(b);reports=[json.loads((p/'metrics.json').read_text(encoding="utf-8")) for p in [a,b]]
    compare(reports)
    if any('mock_not_model_result' in r.get('backend_status',[]) for r in reports):raise ValueError('Mock results cannot support research inference')
    run=json.loads((a/'run.json').read_text(encoding="utf-8"));rows=load_samples(run['config']['data']['eval'])
    if content_fingerprint(rows)!=reports[0]['dataset_hash']:
        raise ValueError('Evaluation manifest/media changed since inference; cannot reuse altered scene clusters')
    values=[{r['id']:r for r in read_jsonl(p/'metrics.details.jsonl')} for p in [a,b]]
    if any(set(v)!={key(r) for r in rows} for v in values):raise ValueError('Mismatched coverage')
    delta=[float(values[1][key(r)]['correct'])-float(values[0][key(r)]['correct']) for r in rows]
    result=cluster_bootstrap(delta,[r['scene_id'] for r in rows],replicates,seed)
    result.update(metric='generic_per_example_correctness_not_official_aggregate',protocol_hash=reports[0]['protocol_hash'],
        limitation='Bootstrap within this test set, not training-seed uncertainty; use multiple independent training seeds.')
    write_json(output,result);return result
