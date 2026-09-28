"""Refresh evidence tables and read-only checkpoint inventories; never upload."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.dsr_sft_eval import read,write

def summarize(plan):
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    results=[];inventory=[];audit=[]
    for item in plan['artifacts']:
        path=Path(item['receipt']);receipt=read(path) if path.exists() else {}
        if item.get('receipt_glob'):
            matches=sorted(Path(item['discovery_root']).glob(item['receipt_glob']))
            if len(matches)>1:raise ValueError('Ambiguous formal artifact receipt')
            if matches:path=matches[0];receipt=read(path)
        checkpoint=receipt.get('checkpoint',item.get('checkpoint'))
        record=dict(role=item['role'],status=receipt.get('status','pending'),receipt=str(path),
                    checkpoint=checkpoint,dependencies=item.get('dependencies',[]),files=[],uploaded=False)
        if checkpoint and Path(checkpoint).is_dir():
            root=Path(checkpoint)
            for f in sorted(root.rglob('*')):
                if f.is_file():record['files'].append(dict(relative=str(f.relative_to(root)),bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns))
            record['bytes']=sum(f['bytes'] for f in record['files'])
        inventory.append(record)
        audit.append(dict(role=item['role'],accepted=receipt.get('status')=='complete' and bool(receipt.get('reload_verified')),
                          reload_verified=receipt.get('reload_verified',False),file_count=len(record['files']),
                          policy_scope=receipt.get('policy_scope'),diagnostic=receipt.get('diagnostic',receipt.get('diagnostic_only'))))
    for item in plan['metrics']:
        path=Path(item['path']);data=read(path) if path.exists() else None
        if item.get('receipt_glob'):
            matches=sorted(Path(item['discovery_root']).glob(item['receipt_glob']))
            if len(matches)>1:raise ValueError('Ambiguous metric receipt')
            if matches:path=matches[0];data=read(path)
        if data is None:
            results.append(dict(label=item['label'],status='pending',protocol=item['protocol'],evidence=str(path)));continue
        if 'paired' in data:
            for benchmark,variants in data['paired'].items():
                for model,r in variants.items():
                    value=r.get('extracted',{}).get('overall_score')
                    if value is None and 'mean_answer_reward' in r:value=100*r['mean_answer_reward']
                    results.append(dict(label=item['label']+'/'+model,benchmark=benchmark,status=data['status'],
                      score=value,count=r.get('count'),protocol=item['protocol'],evidence=str(path)))
        elif 'benchmarks' in data:
            for benchmark,r in data['benchmarks'].items():
                m=r.get('metrics',{});results.append(dict(label=item['label'],benchmark=benchmark,status=r['status'],
                  score=m.get('overall_score'),count=m.get('count'),protocol=m.get('protocol',item['protocol']),evidence=str(path)))
        else:
            results.append(dict(label=item['label'],status='complete',benchmark=item.get('benchmark'),
              score=data.get('extracted',{}).get('overall_score',data.get('accuracy')),count=data.get('count'),protocol=item['protocol'],evidence=str(path)))
    write(out/'results.json',results);write(out/'checkpoint-inventory.json',inventory)
    finished=all(Path(p).exists() and read(p).get('queue_finished') for p in plan['terminal_receipts'])
    if finished and not (out/'final-code-audit.json').exists():
        code=Path(__file__).resolve().parents[1]
        tests=['tests/test_rft_closeout.py','tests/test_geometry_rft_full_policy.py',
               'tests/test_geometry_rft_reward.py','tests/test_dsr_sft_eval.py']
        with (out/'final-code-audit.log').open('ab') as f:
            try:
                result=subprocess.run([sys.executable,'-m','pytest',*tests,'-q'],cwd=code,
                    stdout=f,stderr=f,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONPATH=str(code)),timeout=1800)
                write(out/'final-code-audit.json',dict(status='passed' if result.returncode==0 else 'failed',returncode=result.returncode,tests=tests,
                      limitation='CPU regression only; independent restore/upload and official scoring parity are separate'))
            except subprocess.TimeoutExpired:write(out/'final-code-audit.json',dict(status='timeout',tests=tests))
    write(out/'audit.json',dict(status='pending_final_evaluation' if not finished else 'review_required',
          artifacts=audit,queue_finished=finished,upload_target_confirmed=False,
          warning='Receipt/file audit only; not standalone restore or archival verification'))
    lines=['# RFT 项目结果汇总（自动更新）','',
           '更新时间：'+datetime.datetime.now(datetime.timezone.utc).isoformat(),'',
           '同协议才可比较。缺失结果为 pending，失败不是 0 分。分数为 0–100；不同 benchmark 不合成总分。','',
           '| 实验 | 测试集 | 状态 | 分数 | 样本数 | 协议 |','|---|---|---|---:|---:|---|']
    for r in results:
        number='—' if r.get('score') is None else f"{r['score']:.2f}"
        lines.append(f"| {r['label']} | {r.get('benchmark','待汇总')} | {r['status']} | {number} | {r.get('count','—')} | {r['protocol']} |")
    lines+=['','## 权重保存','', '| 权重 | 状态 | 文件数 | GiB | 云端 |','|---|---|---:|---:|---|']
    for r in inventory:lines.append(f"| {r['role']} | {r['status']} | {len(r['files'])} | {r.get('bytes',0)/2**30:.2f} | 未上传 |")
    lines+=['','RFT policy 依赖初始化 SFT 中冻结的视觉/VGGT；不能只备份 policy。另需 processor、代码、科学配置及运行配置。',
            '当前表是过程汇总，不代表项目已结项。上传目标尚待确认；没有删除任何原权重。',
            '旧的 native32-tagged512 与 RFT structured512 提示不同，不能把跨表差异归因于训练。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return finished

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--watch',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    plan=read(a.plan);out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (out/'supervisor.log').open('ab') as f:
            child=subprocess.Popen([sys.executable,__file__,'--plan',a.plan,'--watch'],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,status='report_watcher')));return
    import fcntl
    with (out/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            finished=summarize(plan)
            if not a.watch or finished:break
            time.sleep(300)
if __name__=='__main__':main()
