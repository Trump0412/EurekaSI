"""ReVSI scoring arithmetic, preserving upstream relative-error thresholds.

Compatibility target: lmms-eval/tasks/revsi/utils.py. This is scoring parity,
not a claim that a frame-explicit RGB frontend equals every official video model.
"""
import numpy as np

NUMERIC = {'object_counting_single','object_counting_multiple','object_abs_distance',
           'object_size_estimation','room_size_estimation_single','room_size_estimation_multiple'}
GROUPS={
    'object_rel_direction':['object_rel_direction_forward_easy','object_rel_direction_backward_easy',
                            'object_rel_direction_forward_hard','object_rel_direction_backward_hard'],
    'object_rel_distance':['object_rel_distance_closest','object_rel_distance_farthest'],
    'object_counting':['object_counting_single','object_counting_multiple'],
    'room_size_estimation':['room_size_estimation_single','room_size_estimation_multiple']}


def score_row(raw, response):
    pred=str(response).strip().split(' ')[0].rstrip('.').strip()
    if raw['question_type'] in NUMERIC:
        try:
            # Preserve upstream float-to-int arithmetic instead of rounding it.
            with np.errstate(divide='ignore',invalid='ignore'):
                return float((np.divide(abs(float(pred)-float(raw['ground_truth'])),float(raw['ground_truth'])) <=
                              1-np.linspace(.5,.95,int((.95-.5)/.05+2))).mean())
        except (ValueError,TypeError,OverflowError):return 0.0
    return float(pred.lower()==str(raw['ground_truth']).lower())


def aggregate(rows):
    by_type={}
    for row in rows:by_type.setdefault(row['question_type'],[]).append(row['acc'])
    scores={k:float(np.mean(v)) for k,v in by_type.items()}
    for group,types in GROUPS.items():
        values=[scores.pop(t) for t in types if t in scores]
        if values:scores[group]=float(np.mean(values))
    return {'overall_acc':float(np.mean(list(scores.values()))) if scores else 0.0,
            'subscores':scores,'question_types':{k:{'n':len(v),'acc':float(np.mean(v))} for k,v in by_type.items()},
            'count':len(rows),'scale':'0..1; macro over collapsed task groups'}
