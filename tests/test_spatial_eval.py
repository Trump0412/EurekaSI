import ast
from collections import OrderedDict
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from spatial_intelligence.spatial_eval import metric_row,metric_aggregate,score_prediction,summarize


def test_vsi_arithmetic_matches_pinned_upstream():
    path=Path(__file__).resolve().parents[1]/'spatial_intelligence/vendor/vsi/utils.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions={'fuzzy_matching','exact_match','abs_dist_norm','mean_relative_accuracy','to_float',
               'vsibench_process_results','vsibench_aggregate_results'}
    keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions or
          isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id.isupper() for t in n.targets)]
    class Log:
        def info(self,*args):pass
    scope={'np':np,'pd':pd,'partial':partial,'OrderedDict':OrderedDict,'eval_logger':Log()}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),scope)
    scored=[];official=[]
    for kind in scope['MCA_QUESTION_TYPES']+scope['NA_QUESTION_TYPES']:
        numeric=kind in scope['NA_QUESTION_TYPES']
        for prediction in ['A','B.','10','15','nan','The answer is A','<answer>10</answer>']:
            raw={'question_type':kind,'ground_truth':'10' if numeric else 'A'}
            result=scope['vsibench_process_results'](dict(raw),[prediction])['vsibench_score']
            score=metric_row(raw,prediction,'vsibench')
            assert score==pytest.approx(result['MRA:.5:.95:.05' if numeric else 'accuracy'])
            official.append(result);scored.append({'question_type':kind,'acc':score})
    assert 100*metric_aggregate(scored,'vsibench')['overall_acc']==pytest.approx(scope['vsibench_aggregate_results'](official)['overall'])


def test_extractor_cannot_use_ground_truth():
    row={'id':'1','question_type':'object_rel_distance','choices':{'A':'chair','B':'table'},'ground_truth':'A'}
    first=score_prediction(row,'<answer>B</answer>','vsibench')
    second=score_prediction(dict(row,ground_truth='B'),'<answer>B</answer>','vsibench')
    assert first['extraction']==second['extraction']
    assert first['extracted_acc']==0 and second['extracted_acc']==1
    report=summarize([second],'vsibench')
    assert report['strict']['overall_score']==0 and report['extracted']['overall_score']==100
