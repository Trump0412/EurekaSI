"""GeoFits inference with real teachers and immutable, per-example evidence.

ReVSI/VSI reuse the existing audited arithmetic. Other benchmark adapters must
be explicitly implemented; a generic accuracy must not impersonate them.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_rows(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique benchmark IDs required')
    for row in rows:
        if not 1 <= len(row.get('media', [])) <= 32:
            raise ValueError('Actual 1..32 images required; no fabricated frames')


def lock(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Changed evaluation identity: ' + str(path))
    if not path.exists():
        path.write_text(text, encoding='utf-8')


def merge(source, out, world, checkpoint, benchmark):
    from spatial_intelligence.followup_benchmarks import score_prediction, summarize
    records = []
    shared_contract = None
    for rank in range(world):
        contract = json.loads((out / f'contract.rank{rank}.json').read_text(encoding='utf-8'))
        if contract['checkpoint'] != str(Path(checkpoint).resolve()) or contract['benchmark'] != benchmark or contract['world'] != world:
            raise ValueError('Shard contract mismatch')
        if contract['manifest'] != source or (shared_contract is not None and contract != shared_contract):
            raise ValueError('Changed source manifest or different shard model/teacher identity')
        shared_contract = contract
        shard = read_rows(out / f'predictions.rank{rank}.jsonl')
        expected = [r['id'] for r in source[rank::world]]
        if [r['id'] for r in shard] != expected:
            raise ValueError('Incomplete/duplicate/out-of-order benchmark shard')
        records.extend(shard)
    by_id = {r['id']: r for r in records}
    # Recompute scores from locked source labels, never trust stale stored scores.
    scores = [score_prediction(dict(row, ground_truth=row.get('ground_truth', row.get('answer'))),
              by_id[row['id']]['response'], benchmark, by_id[row['id']]['truncated']) for row in source]
    report = summarize(scores, benchmark)
    if benchmark not in ('revsi','vsibench') and not report.get('complete_benchmark'):
        raise ValueError('Incomplete official benchmark coverage; do not publish accepted full score')
    report.update(status='complete', accepted=True, checkpoint=str(Path(checkpoint).resolve()),
        protocol='ordered-independent-images448; real frozen teachers392; tagged greedy512',
        claim='Local adapted protocol, not native-video leaderboard equivalence')
    lock(out / 'completion.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'checkpoint', 'manifest', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--benchmark', choices=['revsi','vsibench','mmsi','mindcube_tiny','viewspatial','cvbench','site'], required=True)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    if args.world < 1:
        raise ValueError('Positive world size required')
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    source = read_rows(args.manifest); check_rows(source)
    from spatial_intelligence.followup_benchmarks import validate_rows
    validate_rows(source,args.benchmark)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(source, out, args.world, args.checkpoint, args.benchmark)
        return
    import torch
    from transformers import AutoProcessor
    from spatial_intelligence.geofits_model import load_geofits_model
    from spatial_intelligence.georoute_inputs import RouteCollator
    spec = importlib.util.spec_from_file_location('fits_worker', REPO / 'scripts/train-geofits-stage.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    rank = int(os.getenv('RANK', 0)); world = int(os.getenv('WORLD_SIZE', 1))
    local = int(os.getenv('LOCAL_RANK', 0))
    if world != args.world or not 0 <= rank < world:
        raise ValueError('Allocation does not match shard contract')
    torch.cuda.set_device(local); torch.set_num_threads(4)
    device = torch.device('cuda', local)
    identity = [dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                for p in sorted(Path(args.checkpoint).glob('*')) if p.is_file()]
    if not identity:
        raise ValueError('Checkpoint is empty')
    contract = dict(checkpoint=str(Path(args.checkpoint).resolve()), identity=identity,
        architecture=plan['architecture'], variant=plan['variant'], world=world,
        benchmark=args.benchmark, manifest=source, greedy=True, max_new_tokens=512,
        teachers={k: plan.get(k) for k in ('vggt_source', 'vggt_weights', 'pi3_source', 'pi3_weights')})
    lock(out / f'contract.rank{rank}.json', contract)
    model = load_geofits_model(args.checkpoint, architecture=plan['architecture'],
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    model.config.use_cache = True
    teachers = worker.load_teachers(plan, device)
    processor = AutoProcessor.from_pretrained(plan['processor'])
    collator = RouteCollator(processor, training=False)
    selected = source[rank::world]
    path = out / f'predictions.rank{rank}.jsonl'
    previous = read_rows(path) if path.exists() else []
    if [r['id'] for r in previous] != [r['id'] for r in selected[:len(previous)]]:
        raise ValueError('Resume shard is not an exact prefix')
    with path.open('a', encoding='utf-8') as stream:
        for row in selected[len(previous):]:
            request = dict(row, instruction=row.get('instruction', '') + '\nReturn the final answer inside <answer>...</answer>.')
            batch = collator([request]); batch.pop('_route_rows')
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                features = worker.feature_entries([request], batch, model, teachers, device)
                with model.fusion_diagnostics() as diagnostics, model.feature_context(features):
                    generated = model.generate(**batch, do_sample=False, max_new_tokens=512, use_cache=True)
            completion = generated[0, batch['input_ids'].shape[1]:]
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos or []
            record = dict(id=row['id'], response=processor.decode(completion, skip_special_tokens=True),
                tokens=completion.tolist(), frames=len(row['media']),
                fusion_diagnostics=diagnostics,
                truncated=len(completion) >= 512 and int(completion[-1]) not in eos)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush()


if __name__ == '__main__':
    main()
