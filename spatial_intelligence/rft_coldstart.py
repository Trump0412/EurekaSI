"""Auditable cold-start selection/acceptance; no GPU work or gold-conditioned teacher."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

from .geometry_rft_reward import parse_response, score_response, RewardConfig

SOURCES = ('4drl', 'spatialladder')
RATIO = {'4drl': .7, 'spatialladder': .3}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_coldstart_initialization(plan):
    if not plan.get('coldstart_policy'): return None
    receipt = read(plan['coldstart_receipt'])
    if receipt.get('status') != 'complete' or receipt.get('diagnostic_only') is not False:
        raise ValueError('Cold-start must be formal, not diagnostic')
    if not all(receipt.get(k) is True for k in ('finite_loss', 'nonzero_update', 'reload_verified')):
        raise ValueError('Cold-start update/reload evidence missing')
    if Path(receipt['base_checkpoint']).resolve() != Path(plan['model_checkpoint']).resolve():
        raise ValueError('Cold-start base lineage mismatch')
    if Path(receipt['checkpoint']).resolve() != Path(plan['coldstart_policy']).resolve():
        raise ValueError('Cold-start policy identity mismatch')
    policy = Path(plan['coldstart_policy'])
    if not (policy / 'full_trainable.pt').is_file(): raise ValueError('Cold-start policy missing')
    return {name: dict(bytes=(policy / name).stat().st_size, mtime_ns=(policy / name).stat().st_mtime_ns)
        for name in ('full_trainable.pt', 'policy_scope.json')}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'); tmp.replace(path)


def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Immutable manifest already exists: ' + str(path))
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')


def stratum(row):
    return (row['source'], row.get('question_type', 'unknown'), row['answer_type'], len(row['media']))


def temporal_issue(row):
    if row.get('input_mode') != 'video':
        return None
    if not row.get('fps') or not row.get('total_num_frames'):
        return 'unknown_video_timebase'
    duration = row['total_num_frames'] / row['fps']
    times = [float(x) for x in re.findall(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(?:s\b|seconds?\b)', row['question'])]
    if times and max(times) > duration + max(.5, 2 / row['fps']):
        return 'question_time_exceeds_clip_duration_unresolved_origin'
    return None


def allocate(counts, total):
    """Largest remainder: proportional quotas, stable ties; no invented balanced prior."""
    if total < 0 or total > sum(counts.values()):
        raise ValueError('Insufficient eligible samples for declared quota')
    denominator = sum(counts.values())
    if not denominator: return {}
    exact = {k: total * n / denominator for k, n in counts.items()}
    result = {k: math.floor(n) for k, n in exact.items()}
    order = sorted(counts, key=lambda k: (-(exact[k] - result[k]), str(k)))
    for key in order[:total - sum(result.values())]: result[key] += 1
    return result


def select(pool, total, seed=3407, reference=None):
    """70:30 source mix; proportional task/type/frame quotas, group-round-robin."""
    output = []
    for source in SOURCES:
        buckets = defaultdict(list)
        for row in pool:
            if row['source'] == source: buckets[stratum(row)].append(row)
        wanted = round(total * RATIO[source])
        counts = ({k: len(v) for k, v in buckets.items()} if reference is None else
            dict(Counter(stratum(r) for r in reference if r['source'] == source)))
        quotas = allocate(counts, wanted)
        if any(len(buckets.get(k, [])) < n for k,n in quotas.items()):
            raise ValueError('Insufficient accepted samples in a predeclared stratum; replenish, do not bias toward easy tasks')
        rng = random.Random(seed)
        for key in sorted(quotas):
            groups = defaultdict(list)
            for row in buckets[key]: groups[row['source_group']].append(row)
            keys = sorted(groups); rng.shuffle(keys)
            for group in keys: rng.shuffle(groups[group])
            ordered = []
            while len(ordered) < quotas[key]:
                for group in keys:
                    if groups[group] and len(ordered) < quotas[key]: ordered.append(groups[group].pop())
            output.extend(ordered)
    random.Random(seed).shuffle(output)
    if len(output) != total or len({x['id'] for x in output}) != total:
        raise ValueError('Selection count/identity mismatch')
    return output


def summary(values):
    return dict(rows=len(values), sources=dict(Counter(x['source'] for x in values)),
        groups=len({x['source_group'] for x in values}),
        strata=dict(Counter('|'.join(map(str, stratum(x))) for x in values)))


def teacher_row(row):
    """Positive allowlist: do not pass answer, raw annotations, Bbox_cot or Answer_cot."""
    fields = ('id', 'source', 'question', 'choices', 'answer_type', 'question_type', 'media',
              'input_mode', 'frame_indices', 'fps', 'total_num_frames')
    return {key: row[key] for key in fields if key in row}


def annotation_check(row, annotation, reward):
    if annotation['id'] != row['id']:
        raise ValueError('Annotation ID mismatch')
    response = annotation['response']
    score = score_response(response, row['answer'], task_type=row['answer_type'],
        choices=row.get('choices'), truncated=annotation.get('truncated', True), config=RewardConfig(**reward))
    reasons = []
    if score['structure'] != 1: reasons.append('invalid_three_stage_format')
    if score['answer'] != 1: reasons.append('incorrect_or_insufficiently_precise_answer')
    # Numeric reward=1 still permits 5% error: cold-start targets require exact
    # scalar agreement. No replacing a teacher's answer with the gold value.
    if row['answer_type'] == 'numeric' and score['parsed_answer'] != float(row['answer']):
        reasons.append('numeric_not_exact')
    if annotation.get('student_token_count', 10**9) > 511:
        reasons.append('exceeds_student_512_token_budget_including_eos')
    if temporal_issue(row): reasons.append(temporal_issue(row))
    return dict(accepted=not reasons, reasons=reasons, score=score)


def prepare(plan):
    root = Path(plan['root'])
    if (root / 'selection.json').exists(): raise FileExistsError('Use an independent selection version')
    data = read(plan['data_receipt'])
    if not all(data.get(k) is True for k in ('ready_for_training', 'media_verified', 'leakage_checked')):
        raise ValueError('Source data not accepted')
    train = sum((rows(p) for p in data['train_manifests'].values()), [])
    valid = sum((rows(p) for p in data['validation_manifests'].values()), [])
    train_groups = {x['source_group'] for x in train}
    if train_groups & {x['source_group'] for x in valid}: raise ValueError('Train/validation group overlap')
    if len({x['id'] for x in train + valid}) != len(train + valid): raise ValueError('Duplicate source IDs')
    rejected = []
    def eligible(pool):
        good = []
        for row in pool:
            reason = temporal_issue(row)
            if reason: rejected.append(dict(id=row['id'], split=row['split'], reason=reason))
            else: good.append(row)
        return good
    train_ok, val_ok = eligible(train), eligible(valid)
    candidates = select(train_ok, plan.get('candidate_count', 6000))
    validation = select(val_ok, plan.get('validation_count', 400))
    pilot = select(candidates, 200)
    # Pilot is a prefix of the same candidate order, not extra training samples.
    pilot_ids = {r['id'] for r in pilot}
    candidates = pilot + [r for r in candidates if r['id'] not in pilot_ids]
    for name, values in [('candidates', candidates), ('validation', validation), ('pilot', pilot),
                         ('temporal-exclusions', rejected)]:
        write_rows(root / 'inputs' / (name + '.jsonl'), values)
    write(root / 'selection.json', dict(status='prepared_annotation_not_authorized', seed=3407,
        actual_rft_pool=summary(train), eligible_coldstart_pool=summary(train_ok),
        candidates=summary(candidates), validation=summary(validation), pilot=summary(pilot),
        target_train_count=plan.get('target_count', 4000), temporal_exclusions=len(rejected),
        limitations=['Timestamp origin discrepancies quarantined, not asserted to be annotation errors',
            'Existing RFT manifests unchanged; upstream CoT never supplied to teacher',
            'Source-group disjoint validation, not a proof against unknown aliases or pretraining exposure']))


def finalize(plan):
    root = Path(plan['root']); recipe = read(plan['scientific_config'])
    reviews = read(root / 'human-review.json')
    if reviews.get('approved') is not True or not reviews.get('reviewer'):
        raise ValueError('Independent evidence review required')
    candidates = rows(root / 'inputs/candidates.jsonl'); validation = rows(root / 'inputs/validation.jsonl')
    annotations = rows(root / 'annotations.jsonl')
    by_id = {r['id']: r for r in annotations}
    if len(by_id) != len(annotations): raise ValueError('Duplicate annotations')
    pilot_ids = {r['id'] for r in rows(root / 'inputs/pilot.jsonl')}
    reviewed = set(reviews.get('reviewed_ids', []))
    if not pilot_ids <= reviewed: raise ValueError('Review every pilot sample, including rejected ones')
    rejected_human = set(reviews.get('rejected_ids', []))
    if not rejected_human <= reviewed: raise ValueError('Unreviewed manual rejection')
    if not all(r['id'] in by_id for r in candidates + validation):
        raise ValueError('Annotation coverage incomplete; no silent shrinking')
    accepted, accepted_val, audit = [], [], []
    for pool, output in [(candidates, accepted), (validation, accepted_val)]:
        for row in pool:
            ann = by_id[row['id']]
            if (ann.get('teacher_saw_gold') is not False or ann.get('teacher_input') != teacher_row(row)
                    or ann.get('teacher_revision') != plan['teacher_revision']):
                raise ValueError('Teacher input/revision provenance mismatch')
            result = annotation_check(row, ann, recipe['training']['reward'])
            audit.append(dict(id=row['id'], **result))
            if result['accepted'] and row['id'] not in rejected_human:
                output.append(dict(row, coldstart_response=ann['response'], teacher=ann['teacher'],
                    teacher_revision=ann['teacher_revision']))
    selected = select(accepted, plan.get('target_count', 4000), reference=candidates)
    # Validation includes ALL questions with original gold labels, not just
    # teacher-correct ones. Teacher acceptance on validation is diagnostic only.
    write_rows(root / 'manifests/coldstart.train.jsonl', selected)
    write_rows(root / 'manifests/coldstart.validation.jsonl', validation)
    write_rows(root / 'annotation-audit.jsonl', audit)
    write(root / 'acceptance.json', dict(status='accepted', train=summary(selected),
        validation=summary(validation), teacher_accepted_validation=len(accepted_val),
        human_review=str(root / 'human-review.json'), student=plan['model_checkpoint'],
        mixture='70:30', guarantee='No downstream improvement guarantee; measured after RFT'))
