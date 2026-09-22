import pytest
from spatial_intelligence.followup_benchmarks import extract_choice,score_prediction,summarize,validate_rows,parse_choices


def row(i=0,answer='A',source='ADE20K'):
    return dict(id=str(i),question='Which?',choices={'A':'left','B':'right'},answer=answer,media=['image.png'],source=source,question_type='direction')


def test_gold_blind_parser_and_truncation():
    assert extract_choice('The answer is (B).',{'A':'left','B':'right'})=='B'
    assert extract_choice('<think>A maybe</think><final>B</final>',{'A':'left','B':'right'})=='B'
    assert extract_choice('<answer>B</answer>',{'A':'left','B':'right'})=='B'
    assert extract_choice('A or B',{'A':'left','B':'right'}) is None
    a=score_prediction(row(answer='A'),'B','mmsi',False)
    b=score_prediction(row(answer='B'),'B','mmsi',False)
    assert a['prediction']==b['prediction']=='B' and not a['correct'] and b['correct']
    assert not score_prediction(row(),'A','mmsi',True)['correct']


def test_cvbench_uses_source_weighting_not_micro():
    rows=[row(0,source='ADE20K'),row(1,source='COCO')]+[row(i+2,source='Omni3D') for i in range(4)]
    scores=[score_prediction(r,'A' if i<2 else 'B','cvbench') for i,r in enumerate(rows)]
    result=summarize(scores,'cvbench')
    assert result['accuracy']==.5 and result['micro_accuracy']==pytest.approx(1/3)
    assert result['total']==6 and not result['complete_benchmark']
    assert summarize(scores[:1],'cvbench')['accuracy'] is None


def test_full_denominator_keeps_invalid_and_rejects_duplicates():
    scores=[score_prediction(row(i),text,'mindcube_tiny') for i,text in enumerate(['A','unsure','A or B'])]
    result=summarize(scores,'mindcube_tiny')
    assert result['total']==3 and result['accuracy']==pytest.approx(1/3) and result['unparseable']==2
    with pytest.raises(ValueError):summarize(scores+scores[:1],'mindcube_tiny')
    with pytest.raises(ValueError):score_prediction(row(),'A','unknown')


def test_schema_and_choices():
    assert parse_choices('Options: A: left, B: right')=={'A':'left','B':'right'}
    assert parse_choices('A. right\nB. front')=={'A':'right','B':'front'}
    assert validate_rows([row()],'cvbench')['rows']==1
    with pytest.raises(ValueError):validate_rows([row(),row()],'cvbench')
    with pytest.raises(ValueError):validate_rows([row(answer='C')],'cvbench')


def test_ground_truth_consistency_and_spatial_passthrough(monkeypatch):
    import sys,types
    r=row();r['ground_truth']='B'
    with pytest.raises(ValueError):score_prediction(r,'A','mmsi')
    with pytest.raises(ValueError):validate_rows([r],'mmsi')
    fake=types.SimpleNamespace(score_prediction=lambda *a:('score',a),summarize=lambda *a:('summary',a))
    monkeypatch.setitem(sys.modules,'spatial_intelligence.spatial_eval',fake)
    assert score_prediction(row(),'A','revsi',True)[1][-1] is True
    assert summarize([],'vsibench')[0]=='summary'


def test_normalized_prompt_does_not_duplicate_options():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('prepare_benchmark',Path(__file__).parents[1]/'scripts/prepare-followup-benchmarks.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    r=module.make_row('mmsi','id','Where?\nOptions: A: left, B: right','B',{'A':'left','B':'right'},['image'],'type')
    assert r['question']=='Where?'
    assert r['answer']==r['ground_truth']=='B' and 'A: left' in r['original_question']


def test_site_caa_and_failclosed_input_adapter():
    first=row(0);second=row(1);second['choices']={'A':'a','B':'b','C':'c','D':'d'}
    scores=[score_prediction(first,'A','site'),score_prediction(second,'B','site')]
    result=summarize(scores,'site')
    assert result['accuracy']==pytest.approx((.5-.25)/(.5+.75))
    assert result['micro_accuracy']==.5 and result['total']==2
    with pytest.raises(ValueError,match='interleave'):validate_rows([first],'site')
    first['input_adapter_verified']=True
    assert validate_rows([first],'site')['rows']==1
