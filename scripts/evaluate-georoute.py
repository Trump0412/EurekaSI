"""Dedicated GeoRoute/RGB evaluation; never drop routing via a vanilla loader.

Currently implements ReVSI and VSI arithmetic on the declared image-input
adaptation. Other benchmarks must receive their own verified adapters.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--benchmark',choices=['revsi','vsibench','mmsi','mindcube_tiny','viewspatial','cvbench'],required=True)
    parser.add_argument('--world',type=int,default=1);parser.add_argument('--merge',action='store_true')
    parser.add_argument('--rgb-only',action='store_true')
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    source=rows(args.manifest);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not source or len({row['id'] for row in source})!=len(source): raise ValueError('Empty/duplicate evaluation manifest')
    from spatial_intelligence.followup_benchmarks import score_prediction,summarize,validate_rows
    validate_rows(source,args.benchmark)
    if args.merge:
        records=[];shared=None
        for rank in range(args.world):
            contract=json.loads((out/f'contract.rank{rank}.json').read_text(encoding='utf-8'))
            if (contract['checkpoint']!=str(Path(args.checkpoint).resolve()) or contract['benchmark']!=args.benchmark
                    or contract['world']!=args.world or contract['ids']!=[r['id'] for r in source]
                    or (shared is not None and shared!=contract)):
                raise ValueError('Changed model or shard contract')
            shared=contract
            if (out/f'manifest.rank{rank}.jsonl').read_bytes()!=Path(args.manifest).read_bytes():
                raise ValueError('Changed benchmark source content')
            shard=rows(out/f'predictions.rank{rank}.jsonl')
            if [r['id'] for r in shard]!=[r['id'] for r in source[rank::args.world]]:
                raise ValueError('Incomplete/duplicate/out-of-order benchmark shard')
            records.extend(shard)
        if len(records)!=len(source) or {row['id'] for row in records}!={row['id'] for row in source}:
            raise ValueError('Incomplete/duplicate benchmark coverage')
        by_id={r['id']:r for r in records}
        scored=[]
        for row in source:
            prediction=by_id[row['id']]
            truncated=prediction.get('truncated',prediction.get('score',{}).get('extraction',{}).get('truncated',False))
            scored.append(score_prediction(dict(row,ground_truth=row.get('ground_truth',row.get('answer'))),
                prediction['response'],args.benchmark,truncated))
        report=summarize(scored,args.benchmark)
        if args.benchmark not in ('revsi','vsibench') and not report.get('complete_benchmark'):
            raise ValueError('Incomplete official benchmark coverage')
        report.update(status='complete',accepted=True,checkpoint=args.checkpoint,
            protocol='Independent-image letterbox448/tagged512; not native-video leaderboard parity')
        (out/'completion.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from spatial_intelligence.georoute import load_georoute_model
    from spatial_intelligence.georoute_inputs import RouteCollator,GraphCache,load_teacher
    rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0))
    if world!=args.world: raise ValueError('Declared world must match process launcher')
    torch.cuda.set_device(local);torch.set_num_threads(4)
    identity=[dict(file=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
              for p in sorted(Path(args.checkpoint).glob('*')) if p.is_file()]
    contract=dict(model_identity=identity,checkpoint=str(Path(args.checkpoint).resolve()),world=world,
        benchmark=args.benchmark,rgb_only=args.rgb_only,graph=plan.get('graph'),ids=[row['id'] for row in source],
        prompt='official row instruction + explicit final answer tag',max_new_tokens=512,
        geometry_source_revision=plan.get('vggt_revision'))
    contract_path=out/f'contract.rank{rank}.json'
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract: raise ValueError('Changed inference contract')
    contract_path.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    manifest_snapshot=out/f'manifest.rank{rank}.jsonl'
    original=Path(args.manifest).read_bytes()
    if manifest_snapshot.exists() and manifest_snapshot.read_bytes()!=original: raise ValueError('Changed benchmark contents')
    if not manifest_snapshot.exists(): manifest_snapshot.write_bytes(original)
    loader=Qwen3VLForConditionalGeneration.from_pretrained if args.rgb_only else load_georoute_model
    model=loader(args.checkpoint,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    processor=AutoProcessor.from_pretrained(plan['processor'])
    collator=RouteCollator(processor,training=False)
    if not args.rgb_only:
        weights=Path(plan['vggt_weights'])
        teacher=load_teacher(plan['vggt_source'],weights,torch.device('cuda',local))
        cache=GraphCache(plan['graph_cache'],teacher,dict(path=str(weights.resolve()),size=weights.stat().st_size,
            mtime_ns=weights.stat().st_mtime_ns,source_revision=plan['vggt_revision']),plan['graph'],torch.device('cuda',local))
    path=out/f'predictions.rank{rank}.jsonl'
    previous=rows(path) if path.exists() else [];done={row['id'] for row in previous}
    selected=source[rank::world]
    if len(done)!=len(previous) or not done.issubset({row['id'] for row in selected}): raise ValueError('Foreign/duplicate prior predictions')
    with path.open('a',encoding='utf-8') as stream:
        for row in selected:
            if row['id'] in done: continue
            normalized=dict(row,ground_truth=row.get('ground_truth',row.get('answer')))
            request=dict(row,instruction=row.get('instruction','')+'\nReturn the final answer inside <answer>...</answer>.')
            batch=collator([request]);batch.pop('_route_rows');batch={key:value.cuda() for key,value in batch.items()}
            graph=None if args.rgb_only else cache.get(row).to(torch.device('cuda',local))
            context=nullcontext() if args.rgb_only else model.georoute.graph_context(graph)
            with context,torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated=model.generate(**batch,do_sample=False,max_new_tokens=512,use_cache=True)
            completion=generated[0,batch['input_ids'].shape[1]:]
            text=processor.decode(completion,skip_special_tokens=True)
            eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
            truncated=len(completion)>=512 and int(completion[-1]) not in eos
            record=dict(id=row['id'],response=text,tokens=completion.tolist(),frames=len(row['media']),truncated=truncated,
                graph_edges=None if graph is None else graph.src.numel(),
                score=score_prediction(normalized,text,args.benchmark,truncated))
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()


if __name__=='__main__': main()
