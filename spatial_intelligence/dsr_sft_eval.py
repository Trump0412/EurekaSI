"""Auditable four-arm DSR MCQ evaluation; no GPU imports in contract helpers."""
import json
from pathlib import Path
from collections import Counter


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def validate_rows(selected):
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Empty or duplicate DSR IDs')
    for row in selected:
        if row['answer_type'] != 'mcq' or not isinstance(row['choices'], dict):
            raise ValueError('Only canonical MCQ DSR rows accepted')
        if row['answer'] not in row['choices'] or not row['media']:
            raise ValueError('Invalid answer or visual evidence')
        if row.get('input_mode') != 'video':
            raise ValueError('This locked DSR protocol requires native video')
        indices = row['frame_indices']
        if (len(indices) != len(row['media']) or indices != sorted(set(indices))
                or float(row['fps']) <= 0 or min(indices) < 0
                or max(indices) >= row['total_num_frames']):
            raise ValueError('Invalid frame metadata')
    return selected


def select_rows(manifest, smoke=False):
    selected = validate_rows(rows(manifest))
    if smoke:
        # Fixed first row per type, plus longest sequence. Never select on accuracy.
        seen = set()
        result = []
        for row in selected:
            kind = row.get('question_type', 'unknown')
            if kind not in seen:
                result.append(row)
                seen.add(kind)
        longest = max(selected, key=lambda r: len(r['media']))
        if longest['id'] not in {r['id'] for r in result}:
            result.append(longest)
        selected = result
    return selected


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def model_identity(path):
    root = Path(path)
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in ('.safetensors', '.json', '.bin') or p.name == 'model.pt'))
    if not (root/'config.json').is_file() or not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Missing full SFT model/config')
    return [file_identity(p) for p in files]


def validate_records(records, selected, complete=False):
    expected = {r['id'] for r in selected}
    actual = [r['id'] for r in records]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected):
        raise ValueError('Duplicate/foreign prediction IDs')
    if complete and set(actual) != expected:
        raise ValueError('Incomplete benchmark coverage')
    return {r['id']: r for r in records}


def summarize(records):
    if not records:
        raise ValueError('No DSR predictions')
    n = len(records)
    counts = Counter(r.get('question_type', 'unknown') for r in records)
    return dict(count=n, accuracy=100*sum(r['score']['answer'] for r in records)/n,
                parsed_rate=sum(r['score']['parsed_answer'] is not None for r in records)/n,
                truncation_rate=sum(r['truncated'] for r in records)/n,
                metric='MCQ exact accuracy, 0-100; not composite RL reward',
                protocol='adapted structured prompt; not official leaderboard equivalence',
                per_task={kind: dict(count=count, accuracy=100*sum(
                    r['score']['answer'] for r in records if r.get('question_type', 'unknown') == kind)/count)
                    for kind, count in sorted(counts.items())})


def paired_difference(left, right):
    a = {r['id']: bool(r['score']['answer']) for r in left}
    b = {r['id']: bool(r['score']['answer']) for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys() or not a:
        raise ValueError('Pair requires identical unique IDs')
    return dict(count=len(a), right_minus_left_pp=100*(sum(b.values())-sum(a.values()))/len(a),
                right_only_correct=[k for k in a if b[k] and not a[k]],
                left_only_correct=[k for k in a if a[k] and not b[k]],
                both_correct=sum(a[k] and b[k] for k in a),
                both_wrong=sum(not a[k] and not b[k] for k in a))
MCQ_INSTRUCTION = 'This is a multiple-choice question. Choose exactly one of the listed options. Output only <answer>OPTION_LETTER</answer>, replacing OPTION_LETTER with the selected letter.'

def prompt_protocol(plan):
    mode=plan.get('prompt_mode','structured-v1')
    if mode=='structured-v1':return None
    if mode=='mcq-tagged-v2':return MCQ_INSTRUCTION
    raise ValueError('Unknown DSR prompt protocol: '+str(mode))
