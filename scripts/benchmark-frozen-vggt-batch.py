"""Bounded teacher-only batch diagnostic; never alters a training checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','weights','manifest','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    import torch
    from spatial_intelligence.geometry_backbone import RegisteredVGGT
    torch.set_num_threads(4);torch.cuda.set_device(a.gpu)
    free,total=torch.cuda.mem_get_info()
    if free<18*2**30: raise RuntimeError('Need 18GiB free; do not compete with active allocations')
    torch.cuda.set_per_process_memory_fraction(.30,a.gpu)
    rows=[]
    with open(a.manifest,encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if len(row['media'])==2: rows.append(row)
            if len(rows)==4: break
    if len(rows)!=4: raise ValueError('Need four real two-frame training samples')
    model=RegisteredVGGT(a.source,a.weights,False).to(device=f'cuda:{a.gpu}',dtype=torch.bfloat16).eval()
    sequences=[model.preprocess(row['media'])[0].cuda() for row in rows]
    results=[];reference=None
    for size in (1,2,4):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            outputs=model.forward_batch(sequences,size)
            if reference is None: reference=[x.float().cpu() for x in outputs]
            diffs=[(x.float().cpu()-y) for x,y in zip(outputs,reference)]
            relative=(sum(float(d.double().square().sum()) for d in diffs)/max(1e-30,sum(float(x.double().square().sum()) for x in reference)))**.5
            maximum=max(float(d.abs().max()) for d in diffs)
            del outputs,diffs
            times=[]
            for _ in range(3):
                torch.cuda.synchronize();start=time.perf_counter()
                outputs=model.forward_batch(sequences,size)
                torch.cuda.synchronize();times.append(time.perf_counter()-start);del outputs
        results.append(dict(batch=size,seconds=times,sequences_per_second=12/sum(times),
            relative_feature_l2=relative,max_abs_feature_delta=maximum,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    report=dict(status='diagnostic_complete',ids=[r['id'] for r in rows],results=results,
        scope='Frozen teacher only, four two-frame real samples, shared GPU timing; not end-to-end training acceptance or dedicated-GPU throughput')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
