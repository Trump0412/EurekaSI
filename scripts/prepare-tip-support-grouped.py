"""Dynamically schedule unique ordered image groups across multiple workers/GPU."""
import argparse,json,os,sys,time,queue,multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.geometry_group_audit import open_index,accept_result,summary,finish
from spatial_intelligence.study import dump


def worker(plan,device,index,tasks,results,output):
    import torch
    from spatial_intelligence.georoute import make_tip_intervention,build_graph
    from spatial_intelligence.georoute_inputs import GraphCache,load_teacher
    torch.set_num_threads(2);torch.cuda.set_device(device)
    state=Path(output)/f'worker{index}.json'
    dump(state,dict(status='loading',pid=os.getpid(),gpu=device,time=time.time()))
    teacher=load_teacher(plan['vggt_source'],plan['vggt_weights'],torch.device('cuda',device))
    weight=Path(plan['vggt_weights']);st=weight.stat()
    cache=GraphCache(plan['graph_cache'],teacher,dict(path=str(weight.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,source_revision=plan['vggt_revision']),plan['graph'],torch.device('cuda',device),cache_tracking_positions=True)
    results.put(dict(status='ready',worker=index,gpu=device))
    while True:
        task=tasks.get()
        if task is None:return
        gid,media=task;start=time.monotonic()
        dump(state,dict(status='building',pid=os.getpid(),gpu=device,gid=gid,frames=len(media),time=time.time()))
        try:
            graph=cache.get(dict(media=media))
            if plan['architecture'].get('placement','pre')=='post':
                graph=build_graph(graph.frame_ids[::4],graph.src//4,graph.dst//4,graph.weight,sample_ids=graph.sample_ids[::4],topk=8)
            edges=len(graph.src)
            try:
                make_tip_intervention(graph,generator=torch.Generator().manual_seed(3407));eligible,reason=True,'supported'
            except ValueError as exc:
                if str(exc) not in ('No supported TIP destinations','No valid non-neighbor TIP substitution'):raise
                eligible,reason=False,str(exc)
            peak=torch.cuda.max_memory_reserved()/1024**3
            results.put(dict(status='result',worker=index,gpu=device,gid=gid,result=dict(eligible=eligible,reason=reason,edges=edges),seconds=time.monotonic()-start,peak_gib=peak))
            dump(state,dict(status='ready',pid=os.getpid(),gpu=device,last_gid=gid,peak_gib=peak,time=time.time()))
        except torch.cuda.OutOfMemoryError as exc:
            results.put(dict(status='oom',worker=index,gpu=device,gid=gid,error=str(exc)));return
        except Exception as exc:
            results.put(dict(status='error',worker=index,gpu=device,gid=gid,error=repr(exc)));return


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers-per-gpu',type=int,default=3);p.add_argument('--gpus',type=int,default=8);p.add_argument('--previous-audit')
    a=p.parse_args()
    if not 1<=a.workers_per_gpu<=3 or a.gpus<1:raise ValueError('Unsupported allocation')
    plan=json.loads(Path(a.plan).read_text());data=json.loads(Path(plan['data_receipt']).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=Path(data['train_manifest']);st=manifest.stat()
    contract=dict(manifest=str(manifest.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,graph=plan['graph'],teacher=plan['vggt_weights'],placement=plan['architecture'].get('placement','pre'),policy='actual_support_and_non_neighbor_substitution_v1')
    if a.previous_audit:
        for f in Path(a.previous_audit).glob('contract.rank*.json'):
            c=json.loads(f.read_text());c.pop('world',None)
            if c!=contract:raise ValueError('Prior audit semantic contract mismatch')
    import fcntl
    with (out/'grouped.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dump(out/'grouped-progress.json',dict(status='indexing',time=time.time()))
        weight=Path(plan['vggt_weights']);ws=weight.stat()
        index_contract=dict(contract,teacher_identity=dict(path=str(weight.resolve()),bytes=ws.st_size,mtime_ns=ws.st_mtime_ns,source_revision=plan['vggt_revision']))
        db=open_index(out/'groups.sqlite3',manifest,index_contract,a.previous_audit)
        dump(out/'grouped-contract.json',contract)
        pending=iter(db.execute('SELECT gid,media FROM groups WHERE result IS NULL ORDER BY gid').fetchall())
        ctx=mp.get_context('spawn');tasks=ctx.Queue();results=ctx.Queue();children={};ready=set();retired=set();inflight=set();retries={};done=0;start=time.monotonic();next_index=0
        def spawn(gpu):
            nonlocal next_index
            idx=next_index;next_index+=1
            c=ctx.Process(target=worker,args=(plan,gpu,idx,tasks,results,str(out)));c.start();children[idx]=(gpu,c)
        def fill():
            while len(inflight)<max(1,len(ready))*2:
                try:gid,media=next(pending)
                except StopIteration:return
                assert gid not in inflight;inflight.add(gid);tasks.put((gid,json.loads(media)))
        try:
            for gpu in range(a.gpus):
                for _ in range(a.workers_per_gpu):spawn(gpu)
            while True:
                try:event=results.get(timeout=10)
                except queue.Empty:
                    if any(c.exitcode is not None and idx not in retired for idx,(_,c) in children.items()):raise RuntimeError('Worker exited without completion; pending groups retained')
                    dump(out/'grouped-progress.json',dict(status='working_or_loading',**summary(db),new_groups=done,elapsed_seconds=time.monotonic()-start,ready_workers=len(ready),time=time.time()));continue
                status=event['status'];idx=event['worker']
                if status=='ready':ready.add(idx)
                elif status=='result':
                    accept_result(db,event['gid'],event['result']);db.commit();inflight.remove(event['gid']);done+=1
                    with (out/'grouped-metrics.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                elif status=='oom':
                    ready.discard(idx);retired.add(idx);gpu,child=children[idx];child.join(timeout=30)
                    if child.is_alive():child.terminate();child.join()
                    gid=event['gid'];retries[gid]=retries.get(gid,0)+1
                    if retries[gid]>3:raise RuntimeError('Repeated group OOM; no data/frame reduction allowed')
                    media=json.loads(db.execute('SELECT media FROM groups WHERE gid=?',(gid,)).fetchone()[0]);tasks.put((gid,media))
                    if not any(g==gpu and j not in retired and c.is_alive() for j,(g,c) in children.items()):spawn(gpu)
                    with (out/'oom-recovery.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                else:raise RuntimeError(str(event))
                fill()
                counts=summary(db);elapsed=time.monotonic()-start
                dump(out/'grouped-progress.json',dict(status='running',**counts,new_groups=done,elapsed_seconds=elapsed,groups_per_second=done/max(elapsed,1),ready_workers=len(ready),retired_workers=len(retired),time=time.time()))
                if done<=3 or done%100==0:print(json.dumps(dict(event=status,**counts,new_groups=done,ready_workers=len(ready))),flush=True)
                if counts['pending_groups']==0:break
            # Some workers may still be loading when a small/resumed queue ends.
            # Every nonretired child needs a sentinel, not only ready reporters.
            for idx in children:
                if idx not in retired:tasks.put(None)
            for _,c in children.values():
                c.join(timeout=60)
                if c.is_alive():raise RuntimeError('Workers did not release GPUs')
            dump(out/'completion.json',finish(db,contract))
        finally:
            for _,c in children.values():
                if c.is_alive():c.terminate()
            for _,c in children.values():
                c.join(timeout=10)
                if c.is_alive():c.kill();c.join()
            db.close()

if __name__=='__main__':main()
