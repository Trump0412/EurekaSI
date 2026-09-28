"""CPU-only, independently failing canonical benchmark preparation."""
import argparse,ast,importlib.util,json,math,re,shutil,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.dsr_sft_eval import read,write,rows
from spatial_intelligence.followup_benchmarks import parse_choices
from spatial_intelligence.rft_closeout import EXISTING,validate_manifest

def native_frame_count(total, cap=32):
    """Use real distinct frames; short clips respect Qwen's temporal pair size."""
    count=min(total,cap)
    count-=count%2
    if count<2:raise ValueError('Native video requires at least two real frames')
    return count

def extract_zip(path,target):
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            dest=(target/info.filename).resolve()
            if not dest.is_relative_to(target.resolve()) or '\\' in info.filename:raise ValueError('Unsafe archive member')
            if info.is_dir():continue
            if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlink member')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and dest.stat().st_size==info.file_size:continue
            temp=dest.with_suffix(dest.suffix+'.part')
            with z.open(info) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
            temp.replace(dest)

def build(plan,name,out):
    import pyarrow.parquet as pq
    from PIL import Image
    from spatial_intelligence.rft_data import extract_video
    spec=importlib.util.spec_from_file_location('oldprep',Path(__file__).with_name('prepare-followup-benchmarks.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    if name in EXISTING:
        result=old.normalize_prepared(Path(plan['reuse_prepared']),out,name)
        data=rows(result['manifest'])
        for r in data:r['answer_type']='mcq'
    else:
        asset={'spbench_si':'spbench','spbench_mv':'spbench'}.get(name,name)
        folder=Path(plan['assets_root'])/'datasets'/asset
        for archive in folder.glob('*.zip'):extract_zip(archive,out/'unpacked'/asset)
        def tables(pattern):
            files=sorted(folder.glob(pattern))
            if not files:raise ValueError('Missing parquet')
            return [r for p in files for r in pq.read_table(p).to_pylist()]
        def image(img,key):
            return old.save_image(img['bytes'] if isinstance(img,dict) else img,out/'media'/name/(str(key)+'.png'))
        def record(key,q,a,media,category,choices=None,typ=None,**extra):
            return dict(id=name+'::'+str(key),dataset=name,source_id=str(key),question=q,answer=a,
                        media=media,question_type=category,choices=choices,answer_type=typ or ('mcq' if choices else 'numeric'),**extra)
        data=[]
        if name.startswith('spbench_'):
            raw=tables('SPBench-'+('SI' if name.endswith('_si') else 'MV')+'.parquet')
            for r in raw:
                media=[]
                for f in r['images']:
                    direct=out/'unpacked/spbench'/r['scene_name']/f
                    candidates=[direct] if direct.is_file() else []
                    if len(candidates)!=1:raise ValueError('Missing/ambiguous SPBench image')
                    media.append(str(candidates[0]))
                ch=parse_choices(r['options']) if r.get('options') else None
                if ch:ch={k:re.sub(r'^'+k+r'[.)]\s*','',v) for k,v in ch.items()}
                data.append(record(r['id'],r['question'],str(r['ground_truth']),media,r['question_type'],ch,scene_id=r['scene_name']))
        elif name=='sparbench':
            raw=tables('data/*.parquet')
            for r in raw:
                typ='vci' if r['task']=='view_change_infer' else ('numeric' if r['format_type']=='fill' else 'mcq')
                ch=parse_choices(r['question']) if typ=='mcq' else None
                data.append(record(r['id'],r['question'],r['answer'],[image(im,str(r['id'])+'-'+str(i)) for i,im in enumerate(r['image'])],r['task'],ch,typ,choices_in_question=True))
        elif name=='mmbench_en_dev':
            for r in tables('en/*.parquet'):
                ch={k:r[k] for k in 'ABCD' if r.get(k) is not None and str(r[k]).lower()!='nan'}
                hint='' if str(r.get('hint')).lower() in ('nan','none','') else r['hint']+'\n'
                data.append(record(r['index'],hint+r['question'],r['answer'],[image(r['image'],r['index'])],r['category'],ch,split='dev'))
        elif name=='mmmu_validation':
            for r in tables('*/validation*.parquet'):
                ch=parse_choices(ast.literal_eval(r['options'])) if r['question_type']=='multiple-choice' else None
                images={str(i):image(r['image_'+str(i)],r['id']+'-'+str(i)) for i in range(1,8) if r.get('image_'+str(i))}
                question=r['question']
                if ch:question+='\nOptions:\n'+'\n'.join(k+'. '+v for k,v in ch.items())
                refs=re.findall(r'<image (\d+)>',question)
                if not refs:
                    question=''.join('<image '+k+'>\n' for k in images)+question;refs=list(images)
                if not refs or any(i not in images for i in refs):raise ValueError('MMMU image reference mismatch')
                data.append(record(r['id'],question,r['answer'],[images[i] for i in refs],r['subfield'],ch,'mcq' if ch else 'open',interleaved_image_refs=True,choices_in_question=True,split='validation'))
        else:
            if name=='vlm4d':
                raw=[]
                for file in sorted((folder/'QA').glob('*.json')):
                    part=read(file)
                    if not isinstance(part,list):raise ValueError('VLM4D schema')
                    raw.extend(dict(r,_part=file.stem) for r in part)
            elif name=='stibench':raw=tables('data/*.parquet')
            elif name=='videomme':raw=tables('videomme/*.parquet')
            else:raise ValueError('Unsupported benchmark')
            for index,r in enumerate(raw):
                if name=='vlm4d':
                    ref=r['video'];relative=ref.split('/resolve/main/')[-1]
                    video=folder/relative;ch=r['choices'];q=r['question'];category=r['_part']
                    answers=[k for k,v in ch.items() if v==r['answer']]
                    a=r['answer'] if r['answer'] in ch else (answers[0] if answers else None)
                    key=r['_part']+'-'+str(r['id'])
                elif name=='stibench':
                    candidates=list((out/'unpacked/stibench').glob('**/'+r['Source']+'/'+r['Video']))
                    if not candidates:candidates=list((out/'unpacked/stibench').glob('**/'+r['Video']))
                    if len(candidates)!=1:raise ValueError('STI media ambiguity')
                    video=candidates[0];ch=r['Candidates'];a=r['Answer'];q=(r.get('Prompt') or '')+'\n'+r['Question'];category=r['Task'];key=index
                else:
                    candidates=list((out/'unpacked/videomme').glob('**/'+r['videoID']+'.mp4'))
                    if len(candidates)!=1:raise ValueError('Video-MME media missing/ambiguous')
                    video=candidates[0];ch=parse_choices(r['options']);a=r['answer'];q=r['question'];category=r['duration'];key=r['question_id']
                if a not in ch:raise ValueError('Video gold mapping failed')
                # Cache unique videos by their canonical position in the source tree.
                rel=video.relative_to(folder if name=='vlm4d' else out/'unpacked'/name)
                import cv2
                capture=cv2.VideoCapture(str(video))
                try:
                    if not capture.isOpened():raise ValueError('Cannot open video')
                    selected=native_frame_count(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                finally:capture.release()
                frames=extract_video(video,out/'frames'/name/rel,selected)
                data.append(record(key,q,a,frames['media'],category,ch,input_mode='video',
                            acceptable_answers=([r['answer']] if r['answer'] in ch else answers) if name=='vlm4d' else [a],
                            source_answer=r.get('answer',a),
                            sampling_protocol='uniform-full-clip-max32-even-real-v1',selected_frame_count=selected,
                            frame_indices=frames['frame_indices'],fps=frames['fps'],total_num_frames=frames['total_num_frames'],source_video=str(video)))
    validate_manifest(data)
    for path in {p for r in data for p in r['media']}:
        with Image.open(path) as im:im.verify()
    target=out/(name+'.test.jsonl')
    with target.open('w',encoding='utf-8') as f:
        for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return dict(status='ready',benchmark=name,manifest=str(target),rows=len(data),media_verified=True,
                protocol='adapted paired input; all rows in pinned source split; video max32 even real frames',full_model_verified=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--benchmark',required=True);a=p.parse_args()
    plan=read(a.plan);out=Path(plan['prepared_root']);out.mkdir(parents=True,exist_ok=True);name=a.benchmark
    import fcntl
    with (out/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=out/(name+'.receipt.json')
        if receipt.exists() and read(receipt).get('status')=='ready':return
        try:
            if name not in EXISTING:
                asset={'spbench_si':'spbench','spbench_mv':'spbench'}.get(name,name)
                deadline=time.monotonic()+plan.get('download_wait_seconds',172800)
                while True:
                    path=Path(plan['assets_root'])/'receipts'/(asset+'.json');state=read(path) if path.exists() else {}
                    if state.get('status')=='complete':break
                    if state.get('status')=='failed':raise RuntimeError('Asset download failed; see receipt')
                    if time.monotonic()>deadline:raise TimeoutError('Asset wait timed out')
                    time.sleep(30)
            write(receipt,dict(status='preparing',benchmark=name))
            target=out/name;target.mkdir(exist_ok=True)
            result=build(plan,name,target);write(receipt,result);print(json.dumps(result),flush=True)
        except Exception as e:
            write(receipt,dict(status='failed',benchmark=name,error=repr(e)));raise

if __name__=='__main__':main()
