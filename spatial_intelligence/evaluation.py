"""Strict generic research scores, never presented as official benchmark scores."""
import math
import re
import statistics
import inspect
from pathlib import Path

from .data import key
from .io import digest, symbol, file_digest


def scorer_fingerprint(cfg):
    files = {"generic":file_digest(__file__)}
    if cfg.get("official_scorer"):
        fn = symbol(cfg["official_scorer"])
        path = inspect.getsourcefile(fn)
        if path is None:
            raise ValueError("Official scorer must expose auditable Python source")
        files["plugin"] = file_digest(path)
        if cfg["official_scorer"] == "spatial_intelligence.benchmarks:vsi_official":
            files["vsi_upstream"] = file_digest(Path(__file__).parent/"vendor/vsi/utils.py")
    return digest({"config":cfg,"implementation":files})


def extract(text):
    tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", str(text), flags=re.S | re.I)
    if len(tags) > 1:
        return None
    if tags:
        return tags[0].strip()
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I).strip()
    if re.search(r"</?(?:think|answer)>", clean, re.I):
        return None
    found = re.findall(r"(?:^|\n)(?:final\s+)?answer\s*[:：]\s*(.+)", clean, re.I)
    return found[-1].strip() if found else clean


def parse(text, row):
    target = extract(text)
    if not target:
        return None
    if row["metric"] == "choice":
        # Do not scan the reasoning for arbitrary A/B/C/D tokens.
        m = re.fullmatch(r"\(?([A-Z])\)?[.。]?", target.upper())
        if m and m.group(1) in row["choices"]:
            return m.group(1)
        matches = [k for k, v in row["choices"].items() if target.casefold() == v.strip().casefold()]
        return matches[0] if len(matches) == 1 else None
    if row["metric"] == "numeric":
        try:
            n = float(target)
            return n if math.isfinite(n) else None
        except ValueError:
            return None
    return " ".join(target.casefold().split()).strip(".。")


def score_one(text, row, numeric_atol=0.0, numeric_rtol=0.0):
    if row["answer"] is None or not str(row["answer"]).strip():
        raise ValueError(f"Missing gold: {key(row)}")
    pred = parse(text, row)
    gold = parse(row["answer"], row)
    if gold is None:
        raise ValueError(f"Unparseable gold: {key(row)}")
    if row["metric"] == "numeric":
        ok = pred is not None and abs(pred-gold) <= numeric_atol + numeric_rtol * abs(gold)
    else:
        ok = pred is not None and pred == gold
    return {"id": key(row), "parsed": pred, "correct": bool(ok), "task": row["task"]}


def score(rows, predictions, cfg):
    by_id = {}
    for pred in predictions:
        if pred["id"] in by_id:
            raise ValueError("Duplicate prediction ID")
        by_id[pred["id"]] = pred
    expected = {key(r) for r in rows}
    extra = set(by_id) - expected
    missing = expected - set(by_id)
    if extra or (missing and not cfg.get("allow_missing", False)):
        raise ValueError(f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    details = [score_one(by_id.get(key(r), {}).get("response", ""), r,
                         cfg.get("numeric_atol", 0.0), cfg.get("numeric_rtol", 0.0)) for r in rows]
    tasks = sorted({d["task"] for d in details})
    per_task = {t: {"accuracy": statistics.mean(d["correct"] for d in details if d["task"] == t),
                    "n": sum(d["task"] == t for d in details)} for t in tasks}
    report = {"metric_status": "generic_research_not_official", "n": len(rows), "missing": len(missing),
              "accuracy": statistics.mean(d["correct"] for d in details),
              "parse_rate": statistics.mean(d["parsed"] is not None for d in details),
              "macro_task_accuracy": statistics.mean(v["accuracy"] for v in per_task.values()),
              "by_task": per_task, "scorer_hash": scorer_fingerprint(cfg)}
    if cfg.get("official_scorer"):
        # Official adapter receives complete labels and predictions; it owns aggregation.
        report["official"] = symbol(cfg["official_scorer"])(rows, predictions, cfg)
    return report, details


def compare(reports):
    for field in ("protocol_hash", "dataset_hash", "scorer_hash"):
        if any(field not in r for r in reports) or len({r[field] for r in reports}) != 1:
            raise ValueError(f"Incompatible or missing {field}; do not merge into one fair-comparison table")
    return [{"model": r["model"], "accuracy": r["accuracy"], "n": r["n"],
             "official":r.get("official"), "model_identity":r.get("model_identity"),
             "parse_rate": r["parse_rate"], "protocol_hash": r["protocol_hash"]} for r in reports]
