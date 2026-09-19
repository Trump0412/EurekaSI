"""Diagnostic lexical F1 against Hound TRAIN annotations, not a benchmark metric."""
from collections import Counter
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source != 'hound_diagnostic_train':
        raise ValueError('This diagnostic reward is restricted to Hound training samples')
    words=lambda s: re.findall(r'\w+',s.lower())
    predicted=Counter(words(solution_str));gold=Counter(words(ground_truth))
    overlap=sum((predicted & gold).values())
    denom=sum(predicted.values())+sum(gold.values())
    return 2.0*overlap/denom if denom else 0.0
