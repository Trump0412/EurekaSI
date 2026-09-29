"""Audit realized shaping rewards, not merely configured reward weights."""
import math


def standardized(values):
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return [(x - mean) / (math.sqrt(var) + 1e-6) for x in values]


def group_activity(scores, config):
    if not scores:
        raise ValueError('Empty reward group')
    fw = float(config.get('structure_weight', .5))
    ww = float(config.get('words_weight', .05))
    answer = [float(x['answer']) for x in scores]
    fmt = [fw * float(x['structure']) for x in scores]
    words = [ww * float(x['words']) for x in scores]
    # Version v1 gates words by correctness outside the lexical function.
    if config.get('version', 'structured-spatial-qa-v1') == 'structured-spatial-qa-v1':
        words = [x * a for x, a in zip(words, answer)]
    total = [float(x['total']) for x in scores]
    if any(not math.isfinite(v) for seq in (answer, fmt, words, total) for v in seq):
        raise ValueError('Non-finite reward')
    if any(abs(t-a-f-w) > 1e-7 for t,a,f,w in zip(total,answer,fmt,words)):
        raise ValueError('Reward decomposition mismatch')
    base = standardized(total)
    def affects(component):
        other = standardized([t-c for t,c in zip(total,component)])
        return int(any(abs(a-b) > 1e-6 for a,b in zip(base,other)))
    return dict(groups=1, responses=len(scores), answer_sum=sum(answer),
                format_sum=sum(fmt), words_sum=sum(words), total_sum=sum(total),
                format_positive=sum(x['structure'] > 0 for x in scores),
                words_positive=sum(x['words'] > 0 for x in scores),
                format_advantage_groups=affects(fmt), words_advantage_groups=affects(words))


def merge_activity(items):
    result = {}
    for item in items:
        for key, value in item.items():
            result[key] = result.get(key, 0) + value
    return result


def activity_gate(activity, config, *, allow_saturated_format=False):
    missing = []
    saturated = []
    if not activity.get('responses'):
        missing.append('no_rollouts')
    for name, key, default in [('format', 'structure_weight', .5), ('words', 'words_weight', .05)]:
        if float(config.get(key, default)) > 0:
            if activity.get(name + '_sum', 0) <= 0:
                missing.append(name + '_never_rewarded')
            elif activity.get(name + '_advantage_groups', 0) == 0:
                if (name == 'format' and allow_saturated_format and
                        activity.get('format_positive') == activity.get('responses')):
                    saturated.append('format_already_satisfied_no_advantage_effect')
                else:
                    missing.append(name + '_no_group_advantage_effect')
    return dict(accepted=not missing, failures=missing,
                saturated_terms=saturated,
                limitation='Necessary activity check, not semantic reasoning quality or efficacy evidence')
