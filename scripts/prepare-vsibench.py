"""Full 5130-question VSI-Bench + separate debiased v1 labels, explicit frame protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from spatial_intelligence.study import decode_video,dump,jsonl,read_rows,safe_destination
from spatial_intelligence.build_data import choices_from


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();root=a.root
    if a.detach:
        with (root/'logs/prepare-vsibench.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,'--root',str(root)],stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print(json.dumps({'pid':child.pid}));return
    while True:
        receipt=root/'receipts/vsibench.json'
        state=json.loads(receipt.read_text()) if receipt.exists() else {}
        if state.get('status')=='failed':raise RuntimeError(state.get('error'))
        if state.get('status')=='complete':break
        time.sleep(15)
    import pyarrow.parquet as pq
    folder=root/'datasets/vsibench'
    raw=[]
    for name in ['test_debiased.parquet','test_pruned.parquet']:
        raw.extend(pq.read_table(folder/name).to_pylist())
    if len(raw)!=5130 or len({str(x['id']) for x in raw})!=5130:raise ValueError('Full VSI coverage/uniqueness mismatch')
    raw.sort(key=lambda x:int(x['id']))
    dest=root/'media/vsibench'
    for archive_path in sorted(folder.glob('*.zip')):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith('.mp4'):continue
                target=safe_destination(dest,info.filename)
                if target.is_file() and target.stat().st_size==info.file_size:continue
                target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_suffix('.part')
                with archive.open(info) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
                tmp.replace(target)
    scenes=sorted({(str(x['dataset']),str(x['scene_name'])) for x in raw})
    def frames(pair):
        ds,scene=pair
        return pair,decode_video(dest/ds/f'{scene}.mp4',root/'frames/vsibench32'/ds/scene,32)
    with ThreadPoolExecutor(max_workers=8) as pool:cache=dict(pool.map(frames,scenes))
    train_scenes={x['scene_id'] for x in read_rows(root/'manifests/sft.train.jsonl')}
    rows=[]
    for r in raw:
        choices=choices_from(r['options']) if r.get('options') else None
        scene=str(r['scene_name'])
        rows.append({'id':str(r['id']),'dataset':'vsibench','scene_id':scene,
            'question':'These are frames of a video.\n'+r['question'],'choices':choices,
            'answer':str(r['ground_truth']),'ground_truth':str(r['ground_truth']),
            'question_type':r['question_type'],'split':'test','pruned':bool(r['pruned']),
            'training_scene_overlap':scene in train_scenes,
            'instruction':"Answer with the option's letter from the given choices directly." if choices else 'Please answer the question using a single word or phrase.',
            **cache[(str(r['dataset']),scene)]})
    jsonl(root/'manifests/vsibench32.test.jsonl',rows)
    dump(root/'receipts/vsibench-prepared.json',{'status':'complete','rows':len(rows),'scenes':len(scenes),
        'debiased_v1_rows':sum(not x['pruned'] for x in rows),'training_scene_overlap_rows':sum(x['training_scene_overlap'] for x in rows),
        'frames':'32 uniform rounded indices of original full videos; no claim of all-frame equivalence',
        'revision':'bdcadb3fea447621a828a24911801faba3587c12'})


if __name__=='__main__':main()
