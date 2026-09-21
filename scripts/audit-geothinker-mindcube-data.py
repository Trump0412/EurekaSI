"""Prepare uncapped GeoThinker VLM3R and official MindCube train, not a full mixture."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import zipfile


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def norm_question(row):
    return re.sub(r'\s+', ' ', row['conversations'][0]['value'].replace('<image>', '')).strip()


def mind_groups(rows):
    return {str(Path(image).parent) for row in rows for image in row['images']}


def mind_overlap(train, test):
    ids = {r['id'] for r in train}
    media = {p for r in train for p in r['images']}
    groups = mind_groups(train)
    return dict(ids=sum(r['id'] in ids for r in test),
                media_rows=sum(bool(set(r['images']) & media) for r in test),
                group_rows=sum(bool(mind_groups([r]) & groups) for r in test))


def run(args):
    root, out = Path(args.source_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(selection_name='shared-six-source-geothinker-vlm3r-mindcube-v3',
                   status='preparing_vlm3r_mindcube', ready_for_training=False,
                   scope='two-source component; full six-source mixture requires independent merge gate',
                   counts={}, source_manifests={}, exclusions={}, blockers=[])
    def save():
        temp = out / 'data-readiness.tmp'
        temp.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        temp.replace(out / 'data-readiness.json')
    save()
    heldout = set()
    for name in ['revsi32.test.jsonl', 'vsibench32.test.jsonl']:
        with (root / 'manifests' / name).open() as f:
            heldout.update(str(json.loads(line)['scene_id']) for line in f if line.strip())
    corrected, dropped = {}, set()
    for path in sorted((root / 'datasets/vlm3r-original/vstibench_train').glob('*.json')):
        rows = load(path)
        if path.name == 'erratum_dropped_items.json':
            dropped.update(r['id'] for r in rows)
        else:
            for row in rows:
                if row['id'] in corrected:
                    raise ValueError('Duplicate corrected source ID')
                corrected[row['id']] = row
    seen, unique_media, accepted_groups = set(), set(), set()
    counts, exclusions = Counter(), Counter()
    with (out / 'vlm3r.jsonl').open('w', encoding='utf-8') as dest, (out / 'source-ledger.jsonl').open('w', encoding='utf-8') as ledger:
        for kind, filename in [('vsi', 'vlm3r_vsi_205k_32frames.json'), ('vst', 'vlm3r_vst_132k_32frames.json')]:
            path = root / 'datasets/geothinker/train' / filename
            rows = load(path)
            receipt['source_manifests'][kind] = dict(filename=filename, rows=len(rows), bytes=path.stat().st_size)
            for index, row in enumerate(rows):
                sid, scene = row['id'], row['scene_name']
                counts[kind + '_input'] += 1
                if sid in seen:
                    raise ValueError('Duplicate GeoThinker source ID: ' + sid)
                seen.add(sid)
                reason = 'benchmark_scene' if scene in heldout else ('official_erratum_drop' if sid in dropped else None)
                if reason:
                    exclusions[reason] += 1
                    ledger.write(json.dumps(dict(id=sid, reason=reason, source_index=index, source_file=filename))+'\n')
                    continue
                answer = row['conversations'][1]['value']
                question = norm_question(row)
                if kind == 'vst':
                    if sid not in corrected:
                        raise ValueError('Missing corrected VST ID: ' + sid)
                    cr = corrected[sid]
                    if cr['scene_name'] != scene or cr['data_source'] != row['data_source']:
                        raise ValueError('Correction scene/source mismatch: ' + sid)
                    if row.get('question_type') and cr.get('question_type') != row['question_type']:
                        raise ValueError('Correction task mismatch: ' + sid)
                    if question != norm_question(cr):
                        counts['corrected_questions'] += 1
                        ledger.write(json.dumps(dict(id=sid, old_question=question, question=norm_question(cr), reason='official_vst_question_correction'))+'\n')
                        question = norm_question(cr)
                    new_answer = cr['conversations'][1]['value']
                    if new_answer != answer:
                        counts['corrected_answers'] += 1
                        ledger.write(json.dumps(dict(id=sid, old_answer=answer, answer=new_answer, reason='official_vst_correction'))+'\n')
                    answer = new_answer
                images = [str(root / 'media' / p) for p in row['images']]
                if len(images) != 32:
                    raise ValueError('Expected exactly 32 released frames: ' + sid)
                unique_media.update(images)
                accepted_groups.add(scene)
                result = dict(id='vlm3r::'+sid, source_id=sid, source_index=index, source_file=filename,
                              dataset='vlm3r', scene_id=scene, question=question, answer=answer,
                              question_type=row.get('question_type', 'unspecified_in_release'), media=images, split='candidate_train')
                dest.write(json.dumps(result, ensure_ascii=False)+'\n')
                counts[kind + '_accepted'] += 1
            receipt['counts'] = dict(counts); receipt['exclusions'] = dict(exclusions); save()
    receipt['status'] = 'verifying_media'
    save()
    from PIL import Image
    failures = []
    for index, path in enumerate(sorted(unique_media)):
        try:
            with Image.open(path) as im:
                im.load()
        except Exception as exc:
            failures.append(dict(path=path, error=type(exc).__name__))
        if index % 1000 == 0:
            receipt['media_checked'] = index + 1; save()
    receipt['vlm3r_media'] = dict(unique=len(unique_media), failed=len(failures), scenes=len(accepted_groups))
    (out / 'media-failures.json').write_text(json.dumps(failures, indent=2))
    with zipfile.ZipFile(args.mindcube_zip) as archive:
        def raw(name):
            return [json.loads(line) for line in archive.read('data/raw/'+name+'.jsonl').splitlines() if line.strip()]
        train, tiny, full = raw('MindCube_train'), raw('MindCube_tinybench'), raw('MindCube')
        overlap = mind_overlap(train, tiny)
        if any(overlap.values()):
            raise ValueError('MindCube train/tiny leakage: '+str(overlap))
        receipt['mindcube'] = dict(train=len(train), tiny=len(tiny), full=len(full), train_tiny_overlap=overlap,
                                  train_full_overlap=mind_overlap(train, full), full_is_not_heldout=True)
        for split, rows in [('train', train), ('tiny.test', tiny)]:
            ids = set()
            with (out / ('mindcube.'+split+'.jsonl')).open('w', encoding='utf-8') as dest:
                for index, row in enumerate(rows):
                    if row['id'] in ids:
                        raise ValueError('Duplicate MindCube ID')
                    ids.add(row['id'])
                    paths = []
                    for rel in row['images']:
                        target = out / 'media/mindcube' / rel
                        if not target.resolve().is_relative_to((out / 'media/mindcube').resolve()):
                            raise ValueError('Unsafe media path')
                        if not target.exists():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(archive.read('data/'+rel))
                            with Image.open(target) as im:
                                im.load()
                        paths.append(str(target))
                    result = dict(id='mindcube::'+row['id'], source_id=row['id'], source_index=index,
                                  dataset='mindcube', scene_id='|'.join(sorted(mind_groups([row]))),
                                  question=row['question'], answer=row['gt_answer'], media=paths, split=split)
                    dest.write(json.dumps(result, ensure_ascii=False)+'\n')
    receipt['counts'] = dict(counts)
    receipt['component_ready'] = not failures
    receipt['status'] = 'component_prepared' if not failures else 'media_failed'
    receipt['blockers'] = ['six-source merge, dedup and global scene split not yet performed']
    if failures:
        receipt['blockers'].append('VLM3R media missing or unreadable; no silent removal')
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mindcube-zip', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        path = Path(args.output) / 'data-readiness.json'
        if path.exists() and not isinstance(exc, FileExistsError):
            receipt = load(path)
            receipt.update(status='failed', ready_for_training=False, error=type(exc).__name__+': '+str(exc))
            path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        raise
