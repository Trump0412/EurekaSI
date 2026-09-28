"""Strict SPAR Yes/No reward; no lexical-overlap or test-label reward shaping."""
import re


def answer(text):
    text = str(text).strip()
    tagged = re.fullmatch(r'<answer>\s*(yes|no)\s*</answer>', text, re.I)
    if tagged:
        return tagged.group(1).lower()
    bare = re.fullmatch(r'(yes|no)[.!。]?', text, re.I)
    return bare.group(1).lower() if bare else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source != 'spar_spatial_yesno':
        raise ValueError('Restricted to audited SPAR Yes/No experiment')
    gold = answer(ground_truth)
    if gold is None:
        raise ValueError('Gold label must be Yes or No')
    parsed = answer(solution_str)
    return {'score': float(parsed == gold), 'parsed': parsed or '',
            'parse_success': float(parsed is not None),
            'sample_id': (extra_info or {}).get('index', '')}
