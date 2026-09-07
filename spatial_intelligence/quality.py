"""Auditable structural filtering; no claim that a heuristic certifies spatial intelligence."""
from pathlib import Path
from collections import Counter
from PIL import Image
from .data import load_samples,key
from .io import digest,file_digest,write_json,write_jsonl


def clean(manifest,output):
    rows=load_samples(manifest,training=True);out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Cleaning output is nonempty')
    groups={};bad={};hashes={}
    for r in rows:
        reasons=[]
        for p in r['media']+r.get('geometry_media',[]):
            try:
                with Image.open(p) as im:im.verify()
                if p not in hashes:hashes[p]=file_digest(p)
            except (OSError,ValueError):reasons.append('unreadable_media')
        if not r['media']:reasons.append('no_visual_evidence')
        if reasons:bad[key(r)]=reasons;continue
        # Markers change the referred objects: compare the actual VLM evidence.
        signature=digest([r['question'].strip().lower(),r['choices'],[hashes[p] for p in r['media']]])
        groups.setdefault(signature,[]).append(r)
    for group in groups.values():
        if len({r['answer'] for r in group})>1:
            for r in group:bad[key(r)]=['conflicting_labels_same_evidence_and_question']
        else:
            for r in group[1:]:bad[key(r)]=['exact_duplicate_qa']
    kept=[];excluded=[];review=[]
    for r in rows:
        if key(r) in bad:excluded.append({'id':key(r),'reasons':bad[key(r)]})
        else:
            kept.append(r)
            if r['scene_id'] is None or r['task']=='unspecified':review.append({'id':key(r),'reason':'needs_scene_or_capability_annotation'})
    write_jsonl(out/'clean.train.jsonl',kept);write_jsonl(out/'excluded.jsonl',excluded);write_jsonl(out/'review.jsonl',review)
    report={'input_sha256':file_digest(manifest),'input_rows':len(rows),'kept':len(kept),'excluded':len(excluded),
       'reasons':dict(Counter(reason for r in excluded for reason in r['reasons'])),'review':len(review),
       'claim':'Structural filtering only. Human spatial-evidence annotation and cross-split scene audit still required.'}
    write_json(out/'report.json',report);return report
