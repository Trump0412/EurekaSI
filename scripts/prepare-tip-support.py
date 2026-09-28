"""Audit actual graph support before TIP; never filter instruction SFT rows."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def owns_row(index,rank,world):
 if not 0<=rank<world:raise ValueError('Invalid shard rank')
 return index%world==rank

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--merge',action='store_true');p.add_argument('--yield-request');p.add_argument('--cache-tracking-positions',action='store_true');a=p.parse_args()
 plan=json.loads(Path(a.plan).read_text());data=json.loads(Path(plan['data_receipt']).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 from spatial_intelligence.study import dump
 manifest=Path(data['train_manifest']);stat=manifest.stat()
 contract=dict(manifest=str(manifest.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,graph=plan['graph'],teacher=plan['vggt_weights'],placement=plan['architecture'].get('placement','pre'),policy='actual_support_and_non_neighbor_substitution_v1')
 if a.merge:
  contracts=[json.loads(x.read_text()) for x in out.glob('contract.rank*.json')]
  if not contracts or any(x!=contracts[0] for x in contracts):raise ValueError('Shard contract mismatch')
  world=contracts[0]['world'];records={}
  for rank in range(world):
   done=json.loads((out/f'complete.rank{rank}.json').read_text())
   rows=[json.loads(l) for l in (out/f'rows.rank{rank}.jsonl').read_text().splitlines()]
   if len(rows)!=done['count']:raise ValueError('Incomplete support audit')
   for row in rows:
    if row['id'] in records:raise ValueError('Duplicate audit ID')
    records[row['id']]=row
  expected={json.loads(l)['id'] for l in manifest.read_text().splitlines() if l}
  if set(records)!=expected:raise ValueError('Audit coverage mismatch')
  ids=[r['id'] for r in records.values() if r['eligible']]
  if not ids:raise ValueError('No TIP supported samples')
  dump(out/'completion.json',dict(status='complete',contract=contract,eligible_ids=ids,total=len(records),eligible=len(ids),excluded=len(records)-len(ids),instruction_rows_unchanged=True));return
 import torch
 from spatial_intelligence.georoute import make_tip_intervention,build_graph
 from spatial_intelligence.georoute_inputs import GraphCache,load_teacher
 rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0))
 torch.cuda.set_device(local);torch.set_num_threads(2)
 cp=out/f'contract.rank{rank}.json';c=dict(contract,world=world)
 if cp.exists() and json.loads(cp.read_text())!=c:raise ValueError('Changed audit contract')
 dump(cp,c)
 teacher=load_teacher(plan['vggt_source'],plan['vggt_weights'],torch.device('cuda',local));weight=Path(plan['vggt_weights']);st=weight.stat()
 cache=GraphCache(plan['graph_cache'],teacher,dict(path=str(weight.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,source_revision=plan['vggt_revision']),plan['graph'],torch.device('cuda',local),cache_tracking_positions=a.cache_tracking_positions)
 dump(out/f'execution.rank{rank}.json',dict(cache_tracking_positions=a.cache_tracking_positions,semantic_contract_unchanged=True))
 path=out/f'rows.rank{rank}.jsonl';previous=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else [];done={r['id'] for r in previous};n=len(previous);start=time.monotonic();new=0;memo={}
 with manifest.open() as src,path.open('a') as dst:
  for index,line in enumerate(src):
   if not owns_row(index,rank,world):continue
   if a.yield_request and Path(a.yield_request).exists():
    dump(out/f'yield.rank{rank}.json',dict(status='yielded',processed=n,time=time.time()));return
   row=json.loads(line)
   if row['id'] in done:continue
   key=tuple(row['media'])
   if len(key)==1:eligible,reason,edges=False,'single_frame',0
   elif key in memo:eligible,reason,edges=memo[key]
   else:
    graph=cache.get(row)
    if contract['placement']=='post':
     graph=build_graph(graph.frame_ids[::4],graph.src//4,graph.dst//4,graph.weight,sample_ids=graph.sample_ids[::4],topk=8)
    edges=len(graph.src)
    try:
     make_tip_intervention(graph,generator=torch.Generator().manual_seed(3407));eligible,reason=True,'supported'
    except ValueError as exc:
     if str(exc) not in ('No supported TIP destinations','No valid non-neighbor TIP substitution'):raise
     eligible,reason=False,str(exc)
    memo[key]=(eligible,reason,edges)
   dst.write(json.dumps(dict(id=row['id'],eligible=eligible,reason=reason,edges=edges))+'\n');dst.flush();n+=1;new+=1
   elapsed=time.monotonic()-start
   progress=dict(status='auditing',rank=rank,processed=n,new_rows=new,elapsed_seconds=elapsed,rows_per_second=new/max(elapsed,.001),last_id=row['id'],last_eligible=eligible,last_edges=edges,time=time.time())
   dump(out/f'progress.rank{rank}.json',progress)
   if new<=3 or new%100==0:print(json.dumps(progress),flush=True)
 dump(out/f'complete.rank{rank}.json',dict(count=n))
if __name__=='__main__':main()
