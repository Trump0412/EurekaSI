import ast
from pathlib import Path
import numpy as np
import pytest
from spatial_intelligence.revsi_scoring import score_row,aggregate
from spatial_intelligence.study import safe_destination,JoinedReader


def test_revsi_numeric_and_parser():
    doc={'question_type':'object_abs_distance','ground_truth':'10'}
    assert score_row(doc,'10')==1
    assert score_row(doc,'The answer is 10')==0
    assert score_row(doc,'<answer>10</answer>')==0
    assert score_row(doc,'15')==pytest.approx(.1)
    assert score_row(doc,'nan')==0
    assert score_row({'question_type':'object_counting_single','ground_truth':'0'},'0')==0


def test_revsi_group_macro_not_micro():
    rows=[{'question_type':'object_counting_single','acc':1}]*10
    rows += [{'question_type':'object_counting_multiple','acc':0},{'question_type':'object_abs_distance','acc':0}]
    assert aggregate(rows)['overall_acc']==.25


def test_safe_archive(tmp_path):
    assert safe_destination(tmp_path,'frames/a.png').is_relative_to(tmp_path)
    with pytest.raises(ValueError):safe_destination(tmp_path,'../../outside')


def test_joined_gzip_volumes(tmp_path):
    import io
    import tarfile
    raw=io.BytesIO()
    payload=b'ordered RGB media'
    with tarfile.open(fileobj=raw,mode='w:gz') as archive:
        item=tarfile.TarInfo('spar/example.txt');item.size=len(payload)
        archive.addfile(item,io.BytesIO(payload))
    data=raw.getvalue();parts=[]
    for i,start in enumerate(range(0,len(data),11)):
        path=tmp_path/f'part{i:03d}';path.write_bytes(data[start:start+11]);parts.append(path)
    with JoinedReader(parts) as joined,tarfile.open(fileobj=joined,mode='r|gz') as archive:
        item=next(iter(archive))
        assert archive.extractfile(item).read()==payload


def test_download_progress_does_not_double_count(tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/study-status.py'
    spec=importlib.util.spec_from_file_location('study_status',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name,size in [('a.tar.gz',10),('a.tar.gz.part.range-0-9',10),
                      ('b.tar.gz.part',3),('b.tar.gz.part.range-2-6',5)]:
        (tmp_path/name).write_bytes(b'x'*size)
    assert module.unique_bytes(tmp_path,['*.tar.gz'])==17


def test_selected_logits_loss_and_gradient_equivalence():
    import torch
    torch.manual_seed(3407)
    hidden=torch.randn(2,13,7,dtype=torch.float64,requires_grad=True)
    head=torch.randn(11,7,dtype=torch.float64,requires_grad=True)
    labels=torch.randint(0,11,(2,13));labels[:,:8]=-100;labels[1,11:]=-100
    full=hidden@head.T
    expected=torch.nn.functional.cross_entropy(full[:,:-1].reshape(-1,11),labels[:,1:].reshape(-1),ignore_index=-100)
    g0=torch.autograd.grad(expected,(hidden,head),retain_graph=True)
    positions=(labels[:,1:]!=-100).any(0).nonzero().flatten()
    selected=hidden[:,positions]@head.T
    actual=torch.nn.functional.cross_entropy(selected.reshape(-1,11),labels[:,positions+1].reshape(-1),ignore_index=-100)
    g1=torch.autograd.grad(actual,(hidden,head))
    torch.testing.assert_close(expected,actual)
    for a,b in zip(g0,g1):torch.testing.assert_close(a,b)


def test_against_downloaded_official_scorer():
    # Run when pinned source is present in runtime sources directory.
    import os
    import pandas as pd
    path=Path(os.environ.get('EUREKASI_ROOT','.'))/'sources/revsi_utils.py'
    if not path.exists():pytest.skip('Pinned upstream scorer not downloaded')
    tree=ast.parse(path.read_text())
    keep=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id.isupper() for t in node.targets):keep.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in {'_mean_relative_accuracy','revsi_process_results','_collapse_question_types','_compute_all_subscores'}:keep.append(node)
    scope={'np':np,'pd':pd}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),scope)
    rows=[]
    for kind in scope['MCQ_QUESTION_TYPES']+scope['NQ_QUESTION_TYPES']:
        numeric=kind in scope['NQ_QUESTION_TYPES']
        for response in ['A','A.','B','10','12.5','15','nan','<answer>10</answer>','10 meters']:
            doc={'question_type':kind,'ground_truth':'10' if numeric else 'A'}
            ref=scope['revsi_process_results'](dict(doc),[response])['overall_acc']['acc']
            assert score_row(doc,response)==pytest.approx(ref)
            rows.append(dict(doc,acc=ref))
    assert aggregate(rows)['overall_acc']==pytest.approx(scope['_compute_all_subscores'](rows)['overall_acc'])
