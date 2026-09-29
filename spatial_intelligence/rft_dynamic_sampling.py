"""Bounded DAPO-style group selection; no persistent prompt blacklist."""
import math
import random


def informative_group(scores, epsilon=1e-8):
    values = [float(s['total']) for s in scores]
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('Expected a complete finite reward group')
    return max(values) - min(values) > epsilon


def replacement_row(datasets, source, seed, slot, attempt):
    # Keep the original slot's source, so accepted data do not drift toward
    # whichever source happens to have more reward variation. With replacement:
    # rejected questions remain eligible in every future optimizer update.
    pool = datasets[source]
    return pool[random.Random(f'{seed}:{slot}:{attempt}').randrange(len(pool))]


def select_group(first, generate, score, replace, record, max_attempts=64, epsilon=1e-8):
    if max_attempts < 1 or epsilon < 0:
        raise ValueError('Invalid dynamic sampling budget')
    counts = {'candidate_groups': 0, 'discarded_groups': 0, 'generated_tokens': 0}
    for attempt in range(max_attempts):
        row = first if attempt == 0 else replace(attempt)
        prompt, responses = generate(row)
        scores = [score(response, row) for response in responses]
        accepted = informative_group(scores, epsilon)
        counts['candidate_groups'] += 1
        counts['generated_tokens'] += sum(len(r['tokens']) for r in responses)
        counts['discarded_groups'] += int(not accepted)
        record(row, responses, scores, attempt, accepted)
        if accepted:
            return (row, prompt, responses, scores), counts
        del prompt, responses, scores
    return None, counts
