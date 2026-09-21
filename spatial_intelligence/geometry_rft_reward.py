"""Versioned, gold-blind structured QA reward; no learned judge or text F1.

Numeric scoring adapts SpatialLadder commit 7a0d2ee85c28728835300310a349a53a15967f2e,
VLM-R1/.../open_r1/grpo_spld_stage3.py (accuracy_reward): its TEN thresholds
are preserved for positive targets. Zero targets use exact equality; negative
targets use absolute denominator. These two safety rules and the strict parser
are explicit adaptations. This is not RoboRefer's pixel-point reward.
"""
from dataclasses import dataclass
import math
import re

FIELDS = ("Spatial Observation", "Spatial Transition", "Answer Derivation")
# Engineering choices, NOT an officially published vocabulary or tau.
VOCABULARY_V1 = (
    ("left", "right", "above", "below", "front", "behind", "near", "far", "between", "inside", "outside", "distance", "depth", "viewpoint", "occluded"),
    ("left", "right", "forward", "backward", "toward", "away", "rotate", "rotation", "move", "motion", "closer", "farther", "clockwise", "counterclockwise", "approach", "approaching", "across", "frames", "rotating"),
    ("distance", "direction", "position", "relative", "closer", "farther", "nearest", "farthest", "count", "size", "angle", "rotation", "compare", "comparing", "therefore", "conclude", "infer"),
)


@dataclass(frozen=True)
class RewardConfig:
    version: str = "structured-spatial-qa-v1"
    tau: int = 3
    vocabulary: tuple = VOCABULARY_V1
    structure_weight: float = 0.5
    words_weight: float = 0.05


def _scalar(text, task_type, choices=None):
    text = str(text).strip()
    if task_type == "mcq":
        # Normalize only an unambiguous whole-payload letter, balanced brackets
        # and an optional final full stop; never extract a letter from prose.
        match = re.fullmatch(r"(?:([A-Za-z])|\(\s*([A-Za-z])\s*\)|\[\s*([A-Za-z])\s*\])\s*\.?", text)
        if not match:
            return None
        value = next(x for x in match.groups() if x is not None).upper()
        if choices:
            allowed = {str(k).upper() for k in choices} if isinstance(choices, dict) else {chr(65 + i) for i in range(len(choices))}
            if value not in allowed:
                return None
        return value
    if task_type != "numeric":
        raise ValueError("only mcq and numeric tasks are supported")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def parse_response(response, *, task_type="mcq", choices=None, truncated=False):
    """Never receives the gold answer. Reject duplicate/missing/nonfinal answers.

    An answer-only wrapper can earn correctness but cannot earn structure/words.
    Content outside wrappers, nested tags, and multiple answer claims abstain.
    """
    result = {"parsed_answer": None, "structure": 0.0, "fields": {}, "reason": "invalid_structure"}
    if task_type not in ("mcq", "numeric"):
        raise ValueError("unsupported task type")
    if truncated:
        return dict(result, reason="truncated")
    text = str(response).strip()
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return dict(result, reason="absent_or_duplicate_answer")
    match = re.fullmatch(r"(?:<think>(.*?)</think>\s*)?<answer>([^<>]*)</answer>", text, re.S)
    if not match or (match[1] is not None and re.search(r"</?(?:think|answer)\b", match[1], re.I)):
        return result
    answer = _scalar(match[2], task_type, choices)
    result.update(parsed_answer=answer, reason="ok" if answer is not None else "invalid_answer")
    if match[1] is None:
        return result
    think = match[1]
    headings = list(re.finditer(r"(Spatial Observation|Spatial Transition|Answer Derivation):", think))
    if [m[1] for m in headings] != list(FIELDS) or think[:headings[0].start()].strip():
        return result
    fields = {m[1]: think[m.end():headings[i + 1].start() if i + 1 < 3 else len(think)].strip()
              for i, m in enumerate(headings)}
    if all(fields.values()) and answer is not None:
        result.update(structure=1.0, fields=fields)
    return result


def spatialladder_numeric_reward(prediction, target):
    if not (math.isfinite(prediction) and math.isfinite(target)):
        return 0.0
    if target == 0:
        return float(prediction == 0)
    error = abs(prediction - target) / abs(target)
    # Same Python float count and linspace endpoints as the pinned source.
    n = int((0.95 - 0.5) / 0.05 + 2)
    step = (0.95 - 0.5) / (n - 1)
    thresholds = [0.5 + step * i for i in range(n)]
    thresholds[-1] = 0.95  # np.linspace explicitly sets the endpoint.
    return sum(error <= 1 - threshold for threshold in thresholds) / n


def score_response(response, ground_truth, task_type="mcq", choices=None, truncated=False, config=None):
    config = config or RewardConfig()
    if config.version not in ('structured-spatial-qa-v1', 'geopsro-lexicon-strict-v2'):
        raise ValueError('Unknown reward version')
    if config.tau <= 0 or len(config.vocabulary) != 3:
        raise ValueError("positive tau and three field vocabularies required")
    parsed = parse_response(response, task_type=task_type, choices=choices, truncated=truncated)
    gold = _scalar(ground_truth, task_type, choices)
    if gold is None:
        raise ValueError("invalid ground truth for declared task type")
    pred = parsed["parsed_answer"]
    answer = 0.0 if pred is None else (float(pred == gold) if task_type == "mcq" else spatialladder_numeric_reward(pred, gold))
    matches = {}
    for field, vocabulary in zip(FIELDS, config.vocabulary):
        tokens = set(re.findall(r"[a-z0-9]+", parsed["fields"].get(field, "").casefold()))
        matches[field] = sorted(tokens.intersection(word.casefold() for word in vocabulary))
    words = sum(min(1.0, len(value) / config.tau) for value in matches.values()) / 3
    structure = parsed["structure"]
    if config.version == 'geopsro-lexicon-strict-v2':
        from .geometry_rft_legacy_words import legacy_words_score
        # Restore the full original lexical reward, not its unsafe answer parser.
        # Numeric partial credit remains in answer; legacy words require answer=1.
        words, breakdown = legacy_words_score(parsed['fields'], answer) if structure else (0.0, {})
        return {**parsed, 'answer': answer, 'words': words, 'matches': {},
                'word_breakdown': breakdown,
                'total': answer + config.structure_weight * structure + config.words_weight * words,
                'reward_version': config.version}
    return {**parsed, "answer": answer, "words": words, "matches": matches,
            "total": answer + config.structure_weight * structure + config.words_weight * answer * words,
            "reward_version": config.version}
