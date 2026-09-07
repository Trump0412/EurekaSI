import sys
from pathlib import Path
import pytest

source=Path(__file__).resolve().parents[1]/"legacy_sources/GeoPSRO"
if not source.exists():
    from spatial_intelligence.workspace import settings
    try:source=Path(settings()["external"])/"geopsro"
    except FileNotFoundError:pytest.skip("Historical source absent: spatial init, then spatial source geopsro",allow_module_level=True)
if not source.exists():pytest.skip("Fetch pinned source: spatial source geopsro",allow_module_level=True)
sys.path.insert(0,str(source/''))
from geopsro4d.reward.answer_parser import parse_answer
from geopsro4d.data.schema import normalize_sample
from geopsro4d.eval.eval_dsr import evaluate_predictions


def test_xml_reasoning_does_not_override_final_answer():
    assert parse_answer("<think>A is wrong</think><answer>B</answer>",["left","right"]) == "B"


def test_zero_survives_schema():
    row=normalize_sample({"id":0,"images":["a.png"],"question":"Count?","answer":0})
    assert row.sample_id=="0" and row.answer=="0"


def test_missing_gold_fails(tmp_path):
    p=tmp_path/"p.jsonl";p.write_text('{"id":"x","response":""}\n')
    with pytest.raises(ValueError,match="Missing gold"):
        evaluate_predictions(p,tmp_path/"out",prompt_mode="direct_answer",geometry_mode="normal")
