import pytest

from spatial_intelligence.benchmarks import vsi_official
from spatial_intelligence.data import normalize,key
from spatial_intelligence.vendor.vsi.utils import MCA_QUESTION_TYPES,NA_QUESTION_TYPES


def test_official_perfect_score_and_task_aggregation():
    rows=[];predictions=[]
    for i,task in enumerate(MCA_QUESTION_TYPES+NA_QUESTION_TYPES):
        choices={"A":"one","B":"two"} if task in MCA_QUESTION_TYPES else None
        row=normalize({"id":i,"dataset":"VSI-Bench","split":"test","media":[],"task":task,
                       "question":"Q","choices":choices,"answer":"A" if choices else "10"})
        rows.append(row);predictions.append({"id":key(row),"response":row["answer"]})
    result=vsi_official(rows,predictions,{})["metrics"]
    assert result["overall"]==100.
    assert result["object_rel_direction_accuracy"]==100.
    assert len([k for k in result if k not in {"overall","tabulated_keys","tabulated_results"}])==8


def test_official_subset_not_claimed_as_full():
    with pytest.raises(ValueError,match="all 10"):
        vsi_official([],[],{})
