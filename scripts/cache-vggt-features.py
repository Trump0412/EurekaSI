"""Resumable rank-sharded frozen VGGT cache. Never silently drop frames on OOM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--source-revision', default='a288dd0f14786c93483e45524328726ab7b1b4ce')
    p.add_argument('--weight-revision', default='860abec7937da0a4c03c41d3c269c366e82abdf9')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='Diagnostic only; not a complete cache')
    a = p.parse_args()
    from spatial_intelligence.vggt_features import FrozenVGGT, SCHEMA
    from spatial_intelligence.study import dump, read_rows
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.getenv('RANK', '0')); world = int(os.getenv('WORLD_SIZE', '1'))
    rows = read_rows(Path(a.manifest)); unique = {}
    for row in rows:
        key = tuple(row['media'])
        if key not in unique:
            unique[key] = {'media': list(key), 'representative_id': row['id']}
    groups = list(unique.values())
    source_commit = a.source_revision
    if (Path(a.source)/'.git').exists():
        actual = subprocess.check_output(['git', '-C', a.source, 'rev-parse', 'HEAD'], text=True).strip()
        if actual != source_commit: raise ValueError('VGGT checkout revision mismatch')
    weight = Path(a.weights) / 'model.pt'
    contract = {'schema': SCHEMA, 'source_commit': source_commit, 'weight_revision': a.weight_revision,
                'weight_bytes': weight.stat().st_size, 'weight_mtime_ns': weight.stat().st_mtime_ns,
                'source_provenance': 'git checkout' if (Path(a.source)/'.git').exists() else 'pinned source archive; revision declared',
                'manifest': str(Path(a.manifest).resolve()), 'rows': len(rows), 'unique_sequences': len(groups),
                'frames': sum(len(x['media']) for x in groups), 'world': world, 'diagnostic_limit': a.limit,
                'payload_bytes_estimate': sum(len(x['media']) for x in groups)*1024*2048*2,
                'input': 'exact marked RGB frame order; upstream square pad/resize448; no ground truth geometry'}
    if a.plan_only:
        if rank == 0: dump(out/'plan.json', contract); print(json.dumps(contract), flush=True)
        return
    receipt = out / f'contract.rank{rank}.json'
    if receipt.exists() and json.loads(receipt.read_text()) != contract:
        raise ValueError('Cache extraction contract changed; use a new directory')
    dump(receipt, contract)
    import torch
    torch.set_num_threads(4); torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    extractor = FrozenVGGT(a.source, a.weights, f"cuda:{os.getenv('LOCAL_RANK', '0')}")
    assigned = groups[rank::world]
    if a.limit: assigned = assigned[:a.limit]
    started = time.monotonic()
    for i, item in enumerate(assigned):
        path = extractor.save(out, item['media'])
        elapsed = time.monotonic()-started
        if i % 10 == 0 or i+1 == len(assigned):
            dump(out/f'progress.rank{rank}.json', {'done': i+1, 'total': len(assigned),
                 'elapsed': elapsed, 'remaining_hours': elapsed/(i+1)*(len(assigned)-i-1)/3600,
                 'last_id': item['representative_id'], 'last_cache': str(path)})
    dump(out/f'complete.rank{rank}.json', {'status': 'diagnostic' if a.limit else 'complete',
         'sequences': len(assigned), 'seconds': time.monotonic()-started, 'contract': contract})


if __name__ == '__main__':
    main()
