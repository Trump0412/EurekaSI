"""Portable Qwen3.5 RGB study: locked ReVSI evaluation and one-epoch SFT.

Invoked by scripts/run-qwen35-study.py; roots are runtime arguments, not source paths.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import re
import tarfile
import time
import zipfile
import io


def training_identity(dataset, source_index, source_id):
    """Released source IDs are not unique; index is in the pinned source BEFORE filtering."""
    return {'id':f'{dataset}::row::{source_index}',
            'source_id':str(source_id),'source_index':source_index}


def dump(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    temp.replace(path)


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]


def safe_destination(root, name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Archive path escapes root')
    return p


class JoinedReader(io.RawIOBase):
    """Read byte-split tar.gz volumes as one stream (not independent archives)."""
    def __init__(self, paths):
        self.paths=iter(paths);self.current=None
    def readable(self):return True
    def read(self,n=-1):
        if n<0:raise ValueError('Bounded streaming reads required')
        result=bytearray()
        while len(result)<n:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            part=self.current.read(n-len(result))
            if part:result.extend(part)
            else:self.current.close();self.current=None
        return bytes(result)
    def close(self):
        if self.current:self.current.close()
        super().close()


def decode_video(video, folder, n, exact=False):
    import cv2
    cv2.setNumThreads(1)
    marker=folder/'complete.json'
    if marker.exists():
        data=json.loads(marker.read_text())
        if all(Path(x).is_file() for x in data['media']):return data
        raise ValueError(f'Incomplete cached frames: {folder}')
    cap=cv2.VideoCapture(str(video))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or total<1:raise ValueError(f'Cannot decode {video}')
    if exact and total!=n:raise ValueError(f'Official video has {total} frames, expected {n}')
    n=min(n,total)
    indices=[round(i*(total-1)/(n-1)) for i in range(n)] if n>1 else [0]
    folder.mkdir(parents=True,exist_ok=True);paths=[]
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
            if not ok:raise ValueError(f'Failed frame {video}:{idx}')
            p=folder/f'{idx:06d}.png'
            if not cv2.imwrite(str(p),frame):raise IOError(p)
            paths.append(str(p))
    finally:cap.release()
    value={'media':paths,'frame_indices':indices,'source':str(video),'total_frames':total}
    dump(marker,value);return value


def prepare_revsi(root):
    import pyarrow.parquet as pq
    from .build_data import choices_from
    raw=pq.read_table(root/'datasets/revsi/32_frame/test-00000-of-00001.parquet').to_pylist()
    videos=root/'media/revsi'
    with zipfile.ZipFile(root/'datasets/revsi/video.zip') as archive:
        for info in archive.infolist():
            if '/32_frame/' not in '/'+info.filename or not info.filename.endswith('.mp4'):continue
            rel='32_frame/'+info.filename.split('/32_frame/')[-1] if '/32_frame/' in info.filename else info.filename
            dest=safe_destination(videos,rel)
            if dest.is_file() and dest.stat().st_size==info.file_size:continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            import shutil
            with archive.open(info) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
    scenes=sorted({str(x['scene_id']) for x in raw})
    def build(scene):return scene,decode_video(videos/'32_frame'/f'{scene}.mp4',root/'frames/revsi32'/scene,32,True)
    with ThreadPoolExecutor(max_workers=16) as pool:frames=dict(pool.map(build,scenes))
    rows=[]
    for r in raw:
        opts=choices_from(r['options']) if r.get('options') else None
        rows.append({'id':str(r['id']),'dataset':'revsi','scene_id':str(r['scene_id']),
          'question':'These are frames of a video.\n'+r['question'],'answer':r['ground_truth'],'ground_truth':r['ground_truth'],
          'choices':opts,'question_type':r['question_type'],'split':'test',**frames[str(r['scene_id'])],
          'instruction':"Answer with the option's letter from the given choices directly." if opts else 'Answer the question using a single integer or decimal number.'})
    jsonl(root/'manifests/revsi32.test.jsonl',rows)
    dump(root/'receipts/revsi-prepared.json',{'status':'complete','samples':len(rows),'scenes':len(scenes),'frames':32,
          'input_protocol':'all prescribed frames as ordered RGB images; no video temporal tokens',
          'dataset_revision':'e80b1cde454ab7e2783eff96277038e29e2b2dca'})


def evaluate(root, model_path, name, batch_size=2, limit=None):
    import torch
    from transformers import AutoProcessor
    from .qwen35 import Collator,load_model
    from .revsi_scoring import score_row,aggregate
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);torch.set_num_threads(4)
    rows=read_rows(root/'manifests/revsi32.test.jsonl')
    if limit:rows=rows[:limit]
    out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    contract={'model':str(model_path),'rows':len(rows),'world':world,'max_side':448,'frames':32,
              'thinking':False,'generation':{'max_new_tokens':16,'do_sample':False},'batch_size':batch_size,
              'manifest':str(root/'manifests/revsi32.test.jsonl'),'status':'running'}
    contract_path=out/f'contract.rank{rank}.json'
    if contract_path.exists():
        previous=json.loads(contract_path.read_text())
        if previous!=contract:raise ValueError('Resume evaluation contract changed')
    else:dump(contract_path,contract)
    path=out/f'predictions.rank{rank}.jsonl'
    old=read_rows(path) if path.exists() else []
    done={x['id'] for x in old}
    assigned=rows[rank::world]
    if len(done)!=len(old) or not done.issubset({x['id'] for x in assigned}):raise ValueError('Duplicate/foreign predictions')
    todo=[x for x in assigned if x['id'] not in done]
    if todo:
        proc=AutoProcessor.from_pretrained(model_path)
        collator=Collator(proc,training=False)
        model=load_model(model_path).to(f'cuda:{local}')
        with path.open('a',encoding='utf-8') as f:
            for start in range(0,len(todo),batch_size):
                part=todo[start:start+batch_size];t=time.monotonic()
                batch={k:v.to(model.device) for k,v in collator(part).items()}
                n=batch['input_ids'].shape[1]
                with torch.inference_mode():
                    outputs=model.generate(**batch,max_new_tokens=16,do_sample=False,use_cache=True,
                                           pad_token_id=proc.tokenizer.pad_token_id)
                decoded=proc.batch_decode(outputs[:,n:],skip_special_tokens=True)
                for row,text in zip(part,decoded):
                    pred={'id':row['id'],'question_type':row['question_type'],'response':text,
                          'acc':score_row(row,text),'seconds_per_batch':time.monotonic()-t,
                          'input_tokens_padded':n,'generated_tokens_padded':outputs.shape[1]-n}
                    f.write(json.dumps(pred,ensure_ascii=False)+'\n');f.flush()
                print(json.dumps({'rank':rank,'completed':len(done)+min(start+batch_size,len(todo)),
                                  'total':len(assigned),'seconds':time.monotonic()-t}),flush=True)
    dump(out/f'complete.rank{rank}.json',{'count':len(assigned)})
    # Aggregation is a separate parent stage, so no GPU waits for another rank.


def merge_eval(root,name):
    from .revsi_scoring import aggregate
    out=root/'runs'/name;contracts=sorted(out.glob('contract.rank*.json'))
    if not contracts:raise ValueError('No predictions')
    cfg=json.loads(contracts[0].read_text());rows=[]
    for rank in range(cfg['world']):
        if not (out/f'complete.rank{rank}.json').exists():raise ValueError('Incomplete rank')
        rows.extend(read_rows(out/f'predictions.rank{rank}.jsonl'))
    if len(rows)!=cfg['rows'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Coverage mismatch')
    report=aggregate(rows);report.update(protocol=cfg,status='complete',missing=0)
    jsonl(out/'predictions.jsonl',sorted(rows,key=lambda x:int(x['id'])))
    dump(out/'metrics.json',report);print(json.dumps(report),flush=True)


def verify_model(root,model_path):
    """Real-image interface gate only; no optimizer update or benchmark claim."""
    import torch
    from transformers import AutoProcessor
    from .qwen35 import Collator,load_model
    torch.set_num_threads(4);torch.cuda.set_device(0)
    source=read_rows(root/'manifests/revsi32.test.jsonl')[0]
    row={'media':source['media'][:1],'question':'Reply with the word ready.','answer':'ready','instruction':''}
    proc=AutoProcessor.from_pretrained(model_path)
    batch={k:v.cuda() for k,v in Collator(proc)([row]).items()}
    labels=batch.pop('labels');pos=(labels[:,1:]!=-100).any(0).nonzero().flatten()
    model=load_model(model_path,training=True).cuda();model.eval()
    with torch.no_grad():
        full=model(**batch,use_cache=False).logits
        gold=torch.nn.functional.cross_entropy(full[:,:-1].float().reshape(-1,full.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
        reference=full[:,pos].clone();del full
        selected=model(**batch,use_cache=False,logits_to_keep=pos).logits
        test=torch.nn.functional.cross_entropy(selected.float().reshape(-1,selected.shape[-1]),labels[:,pos+1].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(reference,selected,rtol=.01,atol=.02)
        torch.testing.assert_close(gold,test,rtol=.002,atol=.002)
    model.train();outputs=model(**batch,use_cache=False,logits_to_keep=pos)
    loss=torch.nn.functional.cross_entropy(outputs.logits.float().reshape(-1,outputs.logits.shape[-1]),labels[:,pos+1].reshape(-1),ignore_index=-100)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
    if not torch.isfinite(norm) or not torch.isfinite(loss):raise ValueError('Nonfinite gradient gate')
    report={'status':'complete','diagnostic_only':True,'optimizer_updates':0,'images':1,
            'loss':loss.item(),'full_loss':gold.item(),'selected_loss':test.item(),'grad_norm':norm.item(),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'supervised_tokens':int((labels!=-100).sum())}
    dump(root/'receipts/model-gate.json',report);print(json.dumps(report),flush=True)


def prepare_train(root):
    """Preserve released sample selection; explicitly exclude ReVSI scene overlap."""
    import shutil
    from PIL import Image
    from .benchmarks import conversation_qa
    renderer_path=root/'sources/draw_marker.py'
    spec=importlib.util.spec_from_file_location('geothinker_marker',renderer_path)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    sources={key:json.loads((root/f'datasets/vg-llm/train/{filename}').read_text())
             for key,filename in [('spar','spar_234k.json'),('hound','llava_hound_64k.json')]}
    if len(sources['spar'])!=234277 or len(sources['hound'])!=63750:
        raise ValueError('Released annotation counts changed')
    selected_hound=json.loads((root/'datasets/geothinker/train/llava_hound_64k.json').read_text())
    if {str(x['id']) for x in selected_hound}!={str(x['id']) for x in sources['hound']}:
        raise ValueError('GeoThinker / VG-LLM Hound sample selections differ')
    sources['hound']=selected_hound
    wanted_hound={str(Path(p).relative_to('llava_hound/frames')) for x in selected_hound for p in x['images']}
    wanted_spar={str(Path(p)) for x in sources['spar'] for p in x['images']}
    media=root/'media'
    def unpack(archive, dest, selected=None):
        receipt=root/'receipts'/('extracted-'+archive.parent.name+'-'+archive.name+'.json')
        if receipt.exists():return
        with tarfile.open(archive,'r|gz') as tar:
            for entry in tar:
                if not entry.isfile():continue
                if selected is not None and str(Path(entry.name)) not in selected:continue
                target=safe_destination(dest,entry.name)
                if target.exists() and target.stat().st_size==entry.size:continue
                target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_name(target.name+'.part')
                with tar.extractfile(entry) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
                tmp.replace(target)
        dump(receipt,{'status':'complete','archive':str(archive),'size':archive.stat().st_size})
    archives=[(p,media/'llava_hound/frames',wanted_hound) for p in sorted((root/'datasets/hound/train_300k').glob('*.tar.gz'))]
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda args:unpack(*args),archives))
    print('Checking prescribed Hound frames with 32 metadata workers',flush=True)
    def missing_hound(p):
        return p if not (media/'llava_hound/frames'/p).is_file() else None
    with ThreadPoolExecutor(max_workers=32) as pool:
        missing={p for p in pool.map(missing_hound,wanted_hound) if p is not None}
    print(f'Hound frame validation complete: {len(wanted_hound)} unique paths',flush=True)
    if missing:
        dump(root/'receipts/hound-missing.json',{'count':len(missing),'ids':sorted(missing)})
        raise ValueError(f'{len(missing)} prescribed Hound frames missing; refusing reduced training set')
    rgbd_receipt=root/'receipts/spar-rgbd-extracted.json'
    if not rgbd_receipt.exists():
        parts=sorted((root/'datasets/spar-rgbd').glob('spar-rgbd-*.tar.gz'))
        if len(parts)!=18:raise ValueError('SPAR RGBD needs all 18 byte-split volumes')
        with JoinedReader(parts) as reader,tarfile.open(fileobj=reader,mode='r|gz') as archive:
            for entry in archive:
                if not entry.isfile():continue
                rel=str(Path(entry.name))
                if not rel.startswith('spar/'):rel='spar/'+rel
                # Preserve the requested RGBD archive; materialize referenced RGB now.
                # Depth/pose stay available in archives for a separately defined modality task.
                if rel not in wanted_spar:continue
                dest=safe_destination(media,rel);dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.is_file() and dest.stat().st_size==entry.size:continue
                tmp=dest.with_name(dest.name+'.part')
                with archive.extractfile(entry) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
                tmp.replace(dest)
        missing_rgb=[p for p in wanted_spar if not (media/p).is_file()]
        if missing_rgb:
            dump(root/'receipts/spar-missing.json',{'count':len(missing_rgb),'paths':missing_rgb[:1000]})
            raise ValueError(f'{len(missing_rgb)} SPAR RGB frames missing')
        dump(rgbd_receipt,{'status':'complete','rgb_files':len(wanted_spar),'depth_and_pose':'retained in original RGBD archives'})
    heldout={r['scene_id'] for r in read_rows(root/'manifests/revsi32.test.jsonl')}
    # Scene names, not dataset-prefixed aliases; also compares ScanNet++ short IDs.
    heldout |= {x.split('/')[-1] for x in heldout}
    # Render only accessed frames, concurrently; no change to decoded RGB pixels.
    # Do not materialize all 32 frames for a marker touching just one of them.
    from .marker_cache import materialize
    import cv2
    cv2.setNumThreads(1)
    start=time.monotonic()
    def render_one(pair):
        i,raw=pair
        if not raw.get('spar_info'):return
        parts=Path(raw['images'][0]).parts
        if parts[parts.index('images')+1] in heldout:return
        info=json.loads(raw['spar_info']) if isinstance(raw['spar_info'],str) else raw['spar_info']
        materialize([str(media/p) for p in raw['images']],info,renderer.DRAW_FUNCTIONS[info['type']],
                    root/'frames/spar-marked'/str(i))
    with ThreadPoolExecutor(max_workers=int(os.environ.get('EUREKASI_RENDER_WORKERS','16'))) as pool:
        for n,_ in enumerate(pool.map(render_one,enumerate(sources['spar'])),1):
            if n%1000==0:
                elapsed=time.monotonic()-start
                progress={'status':'rendering','rows_done':n,'rows_total':len(sources['spar']),
                          'elapsed_seconds':elapsed,'eta_seconds':elapsed/n*(len(sources['spar'])-n)}
                dump(root/'receipts/train-render-progress.json',progress);print(json.dumps(progress),flush=True)
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def resolve_spar_image(rel):
        candidates=[media/rel,media/'spar'/rel,media/'spar'/str(rel).removeprefix('spar/')]
        found=next((p for p in candidates if p.is_file()),None)
        if found is None:raise FileNotFoundError(f'SPAR image missing: {rel}')
        return str(found)
    excluded=[];prepared=[]
    def prepare_one(item):
            ds,i,raw=item
            qa=conversation_qa(raw)
            if ds=='hound':
                paths=[str(media/p) for p in raw['images']]
                scene=Path(raw['images'][0]).parent.name
                clean=paths
            else:
                clean=[]
                for rel in raw.get('images',[]):
                    clean.append(resolve_spar_image(rel))
                if not clean:raise ValueError('SPAR sample has no explicit images')
                parts=Path(raw['images'][0]).parts
                scene=parts[parts.index('images')+1]
                paths=clean
            if scene in heldout:
                return None,{**training_identity(ds,i,raw['id']),'dataset':ds,'scene_id':scene,'reason':'ReVSI scene overlap'}
            if ds=='spar' and raw.get('spar_info'):
                info=json.loads(raw['spar_info']) if isinstance(raw['spar_info'],str) else raw['spar_info']
                folder=root/'frames/spar-marked'/str(i);marker=folder/'complete.json'
                paths=materialize(clean,info,renderer.DRAW_FUNCTIONS[info['type']],folder)
            if len(paths)>32:raise ValueError('Unexpected >32 views; no silent subsampling')
            return {**training_identity(ds,i,raw['id']),'dataset':ds,'scene_id':scene,'split':'train',
                             'question':qa['question'],'answer':str(qa['answer']),'media':paths,'geometry_media':clean,
                             'instruction':'','source_row':i},None
    for ds,raw_rows in sources.items():
        with ThreadPoolExecutor(max_workers=16) as pool:
            # map preserves source order; never append in task-completion order.
            for i,(row,exclusion) in enumerate(pool.map(prepare_one,((ds,i,r) for i,r in enumerate(raw_rows)))):
                if row is not None:prepared.append(row)
                if exclusion is not None:excluded.append(exclusion)
                if i%1000==0:print('PREPARED',ds,i,'/',len(raw_rows),flush=True)
    ids=[x['id'] for x in prepared]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate training IDs')
    jsonl(root/'manifests/sft.train.jsonl',prepared);jsonl(root/'manifests/train-exclusions.jsonl',excluded)
    dump(root/'receipts/train-prepared.json',{'status':'complete','rows':len(prepared),
          'source_counts':{k:len(v) for k,v in sources.items()},'excluded':len(excluded),'scene_overlap':0,
          'counts':{k:sum(x['dataset']==k for x in prepared) for k in sources},
          'hound_frames':'exact ordered paths in GeoThinker train/llava_hound_64k.json; no resampling',
          'limitations':'scene-ID audit; near-duplicate images and foundation model pretraining exposure not excluded'})


def prepare_probe(root):
    """Small released Hound subset for disposable DDP/save/reload validation only."""
    import shutil
    from .benchmarks import conversation_qa
    selected=json.loads((root/'datasets/geothinker/train/llava_hound_64k.json').read_text())
    candidates={str(Path(p).relative_to('llava_hound/frames')) for r in selected for p in r['images']}
    media=root/'media/llava_hound/frames'
    archive=root/'datasets/hound/train_300k/chunk_15.tar.gz'
    # This final file exists only after the resumable downloader verifies its size.
    if not archive.is_file():raise FileNotFoundError('Wait for the small final Hound chunk_15 archive')
    with tarfile.open(archive,'r|gz') as tar:
        for entry in tar:
            if not entry.isfile() or str(Path(entry.name)) not in candidates:continue
            target=safe_destination(media,entry.name)
            if target.is_file() and target.stat().st_size==entry.size:continue
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.name+'.part')
            with tar.extractfile(entry) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
            tmp.replace(target)
    rows=[]
    for raw in selected:
        paths=[str(root/'media'/p) for p in raw['images']]
        if not all(Path(p).is_file() for p in paths):continue
        qa=conversation_qa(raw)
        rows.append({'id':'hound::'+str(raw['id']),'dataset':'hound','question':qa['question'],
                     'answer':str(qa['answer']),'media':paths,'instruction':'','diagnostic_only':True})
        if len(rows)==32:break
    if len(rows)<8:raise ValueError('Too few complete Hound samples in downloaded diagnostic shard')
    jsonl(root/'manifests/sft-probe.train.jsonl',rows)
    dump(root/'receipts/train-probe-prepared.json',{'status':'complete','diagnostic_only':True,
         'rows':len(rows),'source':'GeoThinker prescribed Hound frames, first available 32 annotation IDs',
         'sample_ids':[r['id'] for r in rows]})


def train(root,model_path,name,batch_size=1,ga=16,max_steps=-1,diagnostic=False):
    import torch
    from transformers import AutoProcessor,Trainer,TrainingArguments,TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from .qwen35 import Collator,load_model,completion_loss
    if diagnostic and (not name.startswith('diagnostic-') or not 1<=max_steps<=5):
        raise ValueError('Probe is limited to 1..5 steps in a diagnostic-* run, never a research checkpoint')
    gate=json.loads((root/('receipts/train-probe-prepared.json' if diagnostic else 'receipts/train-prepared.json')).read_text())
    if gate['status']!='complete' or (not diagnostic and gate['scene_overlap']!=0):raise ValueError('Data gate failed')
    rows=read_rows(root/('manifests/sft-probe.train.jsonl' if diagnostic else 'manifests/sft.train.jsonl'))
    class Rows(torch.utils.data.Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):return rows[i]
    class Telemetry(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            self.started=time.monotonic();self.start_step=state.global_step
        def on_log(self,args,state,control,logs=None,**kwargs):
            if state.is_world_process_zero:
                elapsed=time.monotonic()-self.started;done=state.global_step-self.start_step
                report={'step':state.global_step,'max_steps':state.max_steps,**(logs or {}),
                        'allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                        'elapsed_seconds_this_session':elapsed,
                        'remaining_hours':elapsed/max(done,1)*max(0,state.max_steps-state.global_step)/3600,
                        'eta_status':'warming_up' if done<20 else 'measured_session_average'}
                dump(Path(args.output_dir)/'live-eta.json',report)
                print(json.dumps(report),flush=True)
    out=root/'runs'/name
    proc=AutoProcessor.from_pretrained(model_path)
    model=load_model(model_path,training=True)
    args=TrainingArguments(output_dir=str(out),num_train_epochs=1,max_steps=max_steps,
        per_device_train_batch_size=batch_size,gradient_accumulation_steps=ga,learning_rate=1e-5,
        warmup_ratio=.03,lr_scheduler_type='cosine',weight_decay=.01,bf16=True,tf32=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        dataloader_num_workers=8,dataloader_pin_memory=True,dataloader_persistent_workers=True,
        logging_steps=1,save_steps=1 if diagnostic else 100,save_total_limit=3,report_to=[],remove_unused_columns=False,
        ddp_find_unused_parameters=False,seed=3407,data_seed=3407,optim='adamw_torch_fused')
    class CompletionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss,outputs=completion_loss(model,inputs)
            return (loss,outputs) if return_outputs else loss
    trainer=CompletionTrainer(model=model,args=args,train_dataset=Rows(),data_collator=Collator(proc),callbacks=[Telemetry()])
    trainer.model_accepts_loss_kwargs=False
    last=get_last_checkpoint(str(out)) if out.exists() else None
    if out.exists() and any(out.iterdir()) and not last:raise ValueError('Nonempty training output without resumable checkpoint')
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(out/'final'))
    if trainer.is_world_process_zero():
        proc.save_pretrained(out/'final')
        dump(out/'completion.json',{'status':'complete','steps':trainer.state.global_step,'rows':len(rows),
                                   'checkpoint':str(out/'final'),'freeze_vision':True,'epochs':trainer.state.epoch,
                                   'diagnostic_only':diagnostic,'resumed_from':last})


def profile(root,model_path,batch_size):
    """Disposable real-data optimizer benchmark. Never overwrites research weights."""
    import torch
    from transformers import AutoProcessor
    from .qwen35 import Collator,load_model,completion_loss
    torch.cuda.set_device(0);torch.set_num_threads(4)
    target=root/'receipts'/f'profile-b{batch_size}.json'
    rows=read_rows(root/'manifests/sft.train.jsonl')
    rows.sort(key=lambda x:(len(x['media']),len(x['question'])+len(x['answer'])),reverse=True)
    probe=rows[:batch_size]
    try:
        proc=AutoProcessor.from_pretrained(model_path)
        batch={k:v.cuda() for k,v in Collator(proc)(probe).items()}
        model=load_model(model_path,training=True).cuda()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,fused=True)
        timings=[];losses=[]
        for i in range(5):
            torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
            loss,outputs=completion_loss(model,batch);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm) or not torch.isfinite(loss):raise ValueError('Nonfinite profiling step')
            opt.step();torch.cuda.synchronize()
            if i>=2:timings.append(time.monotonic()-start)
            losses.append(loss.item());del outputs,loss
        dump(target,{'status':'complete','batch_size':batch_size,'samples_per_second':batch_size/(sum(timings)/len(timings)),
                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                     'gpu_total_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
                     'input_shape':list(batch['input_ids'].shape),'timed_step_seconds':timings,
                     'diagnostic_losses':losses,'sample_ids':[x['id'] for x in probe],
                     'scope':'disposable single-rank worst-length proxy; formal DDP throughput measured separately'})
    except torch.OutOfMemoryError:
        dump(target,{'status':'oom','batch_size':batch_size})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare-revsi','prepare-train','prepare-probe','verify-model','profile','eval','merge-eval','train'])
    p.add_argument('--root',required=True);p.add_argument('--model');p.add_argument('--name',default='baseline-revsi32')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--ga',type=int,default=16)
    p.add_argument('--limit',type=int);p.add_argument('--max-steps',type=int,default=-1)
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();model=a.model or str(root/'models/Qwen3.5-2B')
    if a.mode=='prepare-revsi':prepare_revsi(root)
    elif a.mode=='prepare-train':prepare_train(root)
    elif a.mode=='prepare-probe':prepare_probe(root)
    elif a.mode=='verify-model':verify_model(root,model)
    elif a.mode=='profile':profile(root,model,a.batch_size)
    elif a.mode=='eval':evaluate(root,model,a.name,a.batch_size,a.limit)
    elif a.mode=='merge-eval':merge_eval(root,a.name)
    elif a.mode=='train':train(root,model,a.name,a.batch_size,a.ga,a.max_steps,a.diagnostic)


if __name__=='__main__':main()
