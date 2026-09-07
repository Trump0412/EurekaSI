import copy
import json
from pathlib import Path

import pytest

from spatial_intelligence.config import validate
from spatial_intelligence.data import normalize, model_input, load_samples, leakage
from spatial_intelligence.evaluation import parse, score, compare
from spatial_intelligence.io import load_config, write_jsonl
from spatial_intelligence.runner import infer, evaluate


def row(**updates):
    return normalize({"id":0, "dataset":"test", "question":"Which?", "answer":"B",
                      "choices":{"A":"left", "B":"right"}, "media":[], "split":"test", **updates})


def test_zero_id_and_answer_preserved():
    r = normalize({"id":0,"dataset":"d","question":"How many?","answer":0,"media":[],"split":"train","metric":"numeric"})
    assert r["id"] == "0" and r["answer"] == "0"


def test_inference_label_allowlist():
    inp = model_input(row(metadata={"answer":"SECRET"}))
    assert "answer" not in inp and "metadata" not in inp


@pytest.mark.parametrize("text", ["<think>A is wrong</think><answer>B</answer>", "B", "right"])
def test_final_answer_only(text):
    assert parse(text, row()) == "B"


@pytest.mark.parametrize("text", ["A or B", "<answer>A</answer><answer>B</answer>", "<think>B", "Banana", ""])
def test_ambiguous_rejected(text):
    assert parse(text, row()) is None


def test_missing_prediction_counts_as_wrong():
    report, _ = score([row()], [], {"allow_missing":True})
    assert report["n"] == 1 and report["accuracy"] == 0
    with pytest.raises(ValueError):
        score([row()], [], {})


def test_missing_gold_not_correct():
    with pytest.raises(ValueError):
        score([row(answer=None)], [{"id":"test::0","response":""}], {})


def test_duplicate_prediction_rejected():
    p = {"id":"test::0","response":"B"}
    with pytest.raises(ValueError):
        score([row()], [p,p], {})


def test_split_guard(tmp_path):
    path = tmp_path/"data.jsonl"
    write_jsonl(path, [row()])
    with pytest.raises(ValueError):
        load_samples(path, training=True)


def test_cross_alias_media_leakage(tmp_path):
    a,b = tmp_path/"a.bin", tmp_path/"b.bin"
    a.write_bytes(b"same image bytes")
    b.write_bytes(a.read_bytes())
    overlaps = leakage([row(dataset="alias1", media=[str(a)])], [row(dataset="alias2", media=[str(b)])])
    assert overlaps["media"] and overlaps["question_media"]


def test_coverage_protocol_compare():
    a = {"protocol_hash":"a", "dataset_hash":"d", "scorer_hash":"s", "model":{},"accuracy":1,"parse_rate":1,"n":1}
    with pytest.raises(ValueError):
        compare([a,{**a,"protocol_hash":"b"}])


def test_unknown_knob_rejected():
    cfg = load_config("configs/smoke.yaml")
    cfg["train"]["learning_reate"] = 1
    with pytest.raises(ValueError):
        validate(cfg)


def test_prediction_tampering_rejected(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg["output"] = str(tmp_path/"run")
    result = infer(cfg)
    assert result["accuracy"] == .5
    path = Path(cfg["output"])/"predictions.jsonl"
    values = [json.loads(s) for s in path.read_text().splitlines()]
    values[0]["response"] = "B"
    write_jsonl(path, values)
    with pytest.raises(ValueError, match="changed"):
        evaluate(cfg, path, tmp_path/"report.json")
