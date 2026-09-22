"""Convert already downloaded complementary benchmarks; no GPU/network needed."""
import argparse
from collections import Counter
import io
import json
import re
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.followup_benchmarks import parse_choices, answer_letter, validate_rows, PROTOCOL


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def save_image(data,target):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:image.load()
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=data:raise ValueError('Conflicting media path')
    else:target.write_bytes(data)
    return str(target)


def zip_image(archive,member,out):
    if Path(member).is_absolute() or '..' in Path(member).parts:raise ValueError('Unsafe archive path')
    return save_image(archive.read(member),out/member)


def make_row(benchmark,identity,question,answer,choices,media,category,**metadata):
    choices=parse_choices(choices)
    original_question=question
    option_start=re.search(r'(?<!\w)A[.:)]\s+',question)
    if option_start:
        question=question[:option_start.start()].rstrip()
        question=re.sub(r'(?:Options:|Select from the following choices\.)\s*$','',question,flags=re.I).strip()
    return dict(id=f'{benchmark}::{identity}',dataset=benchmark,source_id=str(identity),split='test',
                question=question.strip(),original_question=original_question,
                answer=answer_letter(answer),ground_truth=answer_letter(answer),choices=choices,media=media,
                instruction='Answer with the option letter only.',
                question_type=category,scoring_protocol=PROTOCOL,**metadata)


def convert(root,out,name):
    rows=[]
    if name in {'mmsi','cvbench'}:
        import pyarrow.parquet as pq
        files=['MMSI_Bench.parquet'] if name=='mmsi' else ['test_2d.parquet','test_3d.parquet']
        for filename in files:
            for raw in pq.read_table(root/name/filename).to_pylist():
                identity=raw['id'] if name=='mmsi' else raw['idx']
                images=raw['images'] if name=='mmsi' else [raw['image']]
                media=[save_image(img['bytes'] if isinstance(img,dict) else img,out/'media'/name/f'{identity}-{i}.jpg') for i,img in enumerate(images)]
                if name=='mmsi':
                    rows.append(make_row(name,identity,raw['question'],raw['answer'],raw['question'],media,raw['question_type'],difficulty=raw.get('difficulty')))
                else:
                    rows.append(make_row(name,identity,raw['prompt'],raw['answer'],raw['choices'],media,raw['task'],source=raw['source'],source_dataset=raw['source_dataset'],source_filename=raw['source_filename'],task_dimension=raw['type']))
    elif name=='mindcube_tiny':
        with zipfile.ZipFile(root/'mindcube/data.zip') as z:
            raw_rows=[json.loads(l) for l in z.read('data/raw/MindCube_tinybench.jsonl').splitlines() if l.strip()]
            train=[json.loads(l) for l in z.read('data/raw/MindCube_train.jsonl').splitlines() if l.strip()]
            train_ids={r['id'] for r in train};train_media={p for r in train for p in r['images']}
            train_groups={str(Path(p).parent) for p in train_media}
            for raw in raw_rows:
                if raw['id'] in train_ids or set(raw['images'])&train_media or any(str(Path(p).parent) in train_groups for p in raw['images']):raise ValueError('MindCube train/tiny leakage')
                media=[zip_image(z,'data/'+p,out/'media/mindcube') for p in raw['images']]
                rows.append(make_row(name,raw['id'],raw['question'],raw['gt_answer'],raw['question'],media,'|'.join(raw['category']),source_type=raw['type']))
    elif name=='viewspatial':
        archives={n:zipfile.ZipFile(root/name/(n+'.zip')) for n in ['scannetv2_val','val2017']}
        try:
            for i,raw in enumerate(json.loads((root/name/'ViewSpatial-Bench.json').read_text())):
                media=[]
                for path in raw['image_path']:
                    rel=path.removeprefix('ViewSpatial-Bench/')
                    media.append(zip_image(archives[rel.split('/')[0]],rel,out/'media/viewspatial'))
                question=raw['question']+'\n'+raw['choices']
                rows.append(make_row(name,i,question,raw['answer'],raw['choices'],media,raw['question_type']))
        finally:
            for archive in archives.values():archive.close()
    else:raise ValueError('Unverified converter: '+name)
    validation=validate_rows(rows,name)
    if not validation['complete_benchmark']:raise ValueError('Unexpected official denominator')
    path=out/(name+'.test.jsonl')
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return dict(status='ready',benchmark=name,manifest=str(path),media_verified=True,
                rows=len(rows),unique_ids=len({r['id'] for r in rows}),category_counts=dict(Counter(r['question_type'] for r in rows)),
                protocol=PROTOCOL,full_model_verified=False,
                adaptation='Strict gold-blind final-letter parser; ambiguous/truncated invalid responses count wrong; no LLM judge',
                aggregation='official source-weighted CV-Bench; micro+category breakdown otherwise (ViewSpatial macro included as adaptation)',
                training_overlap='MindCube own train/tiny IDs/media/groups checked; cross-dataset training exposure not certified')


def normalize_prepared(previous,out,name):
    receipt=json.loads((previous/(name+'.receipt.json')).read_text())
    if receipt['status']!='ready':raise ValueError('Previous benchmark not prepared')
    rows=[]
    with Path(receipt['manifest']).open() as f:
        for line in f:
            old=json.loads(line)
            normalized=make_row(name,old['source_id'],old.get('original_question',old['question']),old['answer'],old['choices'],old['media'],old['question_type'])
            rows.append(dict(old,**normalized))
    validate_rows(rows,name)
    if len(rows)!=receipt['rows']:raise ValueError('Changed denominator')
    from PIL import Image
    for path in {p for row in rows for p in row['media']}:
        with Image.open(path) as image:image.load()
    target=out/(name+'.test.jsonl')
    with target.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return dict(receipt,manifest=str(target),prepared_from=str(previous),prompt_policy='choices supplied once separately; no ground truth in prompt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reuse-prepared',help='Normalize previously verified manifests; reuse media read-only')
    p.add_argument('--benchmarks',nargs='+',default=['mmsi','mindcube_tiny','viewspatial','cvbench','site']);a=p.parse_args()
    root=Path(a.assets_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    statuses={}
    for name in a.benchmarks:
        try:
            if name=='site':
                statuses[name]=dict(status='blocked_input_adapter',benchmark=name,full_model_verified=False,scoring_verified=True,
                                    reason='Official SITE requires image-option interleave; current all-images-first collator cannot preserve it. Video input protocol also needs acceptance. No full manifest claimed.',
                                    metric='CAA=sum(correct-1/K)/sum(1-1/K)',
                                    official_source='https://github.com/wenqi-wang20/SITE-Bench/blob/main/eval_scripts/aggregate.py',
                                    released_image_rows=len(json.loads((root/name/'image_test.json').read_text())),released_video_rows=len(json.loads((root/name/'video_test.json').read_text())))
            elif a.reuse_prepared:statuses[name]=normalize_prepared(Path(a.reuse_prepared),out,name)
            else:statuses[name]=convert(root,out,name)
        except Exception as exc:statuses[name]=dict(status='failed',benchmark=name,error=type(exc).__name__+': '+str(exc))
        dump(out/(name+'.receipt.json'),statuses[name]);dump(out/'receipt.json',statuses)
        print(json.dumps(statuses[name]),flush=True)
