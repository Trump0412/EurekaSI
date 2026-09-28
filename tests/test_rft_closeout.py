import importlib.util
from pathlib import Path
import pytest
from spatial_intelligence.rft_closeout import score,summary,validate_manifest,final_text,relative_accuracy

def row():return dict(id='x',question='Which?',media=['a.png'],choices={'A':'left','B':'right'},answer='A',answer_type='mcq',question_type='direction')

def test_gold_blind_and_truncated():
    r=row();s=score(r,'<answer>B</answer>','stibench')
    assert s['prediction']=='B' and s['score']==0
    r['answer']='B';assert score(r,'<answer>B</answer>','stibench')['score']==1
    assert score(r,'<answer>B</answer>','stibench',True)['score']==0

def test_numeric_and_zero():
    assert relative_accuracy(2,2)==1
    assert relative_accuracy(float('nan'),2)==0
    assert relative_accuracy(0,0)==1

def test_validate_duplicates():
    with pytest.raises(ValueError):validate_manifest([row(),row()])

def test_invalid_not_dropped():
    a=score(row(),'I cannot answer','stibench');b=score(dict(row(),id='b'),'<answer>A</answer>','stibench')
    r=summary([a,b],'stibench');assert r['count']==2 and r['overall_score']==50

def test_vci_requires_named_final():
    r=dict(row(),choices=None,answer_type='vci',answer='move_right:1')
    assert score(r,'<answer>move_right:1</answer>','sparbench')['score']==1
    assert score(r,'move_right:1, bad:2','sparbench')['score']==0

def test_dependency_failure_not_success(tmp_path):
    spec=importlib.util.spec_from_file_location('queue',Path(__file__).parents[1]/'scripts/run-rft-closeout.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    f=tmp_path/'state.json';f.write_text('{"status":"failed","gpu_work_finished":true}')
    assert m.reasons({'dependencies':[{'path':str(f),'statuses':['complete'],'require_gpu_release':True}]})

def test_short_video_sampling():
    spec=importlib.util.spec_from_file_location('prep',Path(__file__).parents[1]/'scripts/prepare-rft-closeout.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert [m.native_frame_count(n) for n in (100,32,31,30,3,2)]==[32,32,30,30,2,2]
    with pytest.raises(ValueError):m.native_frame_count(1)

def test_duplicate_option_gold_equivalence():
    r=dict(row(),choices={'A':'near','C':'away','D':'away'},answer='C',acceptable_answers=['C','D'])
    for a in ['C','D']:assert score(r,'<answer>'+a+'</answer>','vlm4d')['score']==1
    assert score(r,'<answer>A</answer>','vlm4d')['score']==0
    assert score(r,'<answer>D</answer>','vlm4d',truncated=True)['score']==0
    from spatial_intelligence.rft_closeout import validate_manifest
    validate_manifest([dict(r,media=['test.png'],question='direction?')])
    with pytest.raises(ValueError):validate_manifest([dict(r,media=['test.png'],question='direction?',acceptable_answers=['A','C'])])

def test_prepare_path_traversal(tmp_path):
    import zipfile
    spec=importlib.util.spec_from_file_location('prep',Path(__file__).parents[1]/'scripts/prepare-rft-closeout.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../outside','bad')
    with pytest.raises(ValueError):m.extract_zip(p,tmp_path/'out')

def test_failed_benchmark_continues_to_next(tmp_path,monkeypatch):
    import json
    spec=importlib.util.spec_from_file_location('queue',Path(__file__).parents[1]/'scripts/run-rft-closeout.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir();ready=tmp_path/'data';ready.mkdir()
    for name in ('first','second'):(ready/(name+'.receipt.json')).write_text('{"status":"ready"}')
    calls=[]
    def fake(command,log,env,timeout):
        name=command[command.index('--benchmark')+1];calls.append(name)
        if name=='first':raise RuntimeError('Independent benchmark failed')
        phase='smoke' if '--smoke' in command else 'full'
        target=root/'runs'/name/phase/'completion.json';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('{"accepted":true,"status":"complete"}')
    monkeypatch.setattr(m,'launch',fake);monkeypatch.setattr(m,'idle',lambda _:True)
    m.execute(dict(root=str(root),prepared_root=str(ready),dependencies=[],gpus=[0],python='python',benchmarks=['first','second'],model_role='sft'))
    value=json.loads((root/'completion.json').read_text())
    assert value['queue_finished'] and value['gpu_work_finished'] and not value['accepted']
    assert value['benchmarks']['first']['status']=='failed_evaluation'
    assert value['benchmarks']['second']['status']=='complete' and 'second' in calls
def test_typed_instruction_is_gold_blind():
    from spatial_intelligence.rft_closeout import inference_instruction
    from spatial_intelligence.dsr_sft_eval import MCQ_INSTRUCTION
    assert inference_instruction({'answer_type':'mcq'})==MCQ_INSTRUCTION
    assert inference_instruction({'answer_type':'numeric'}) is None
    assert inference_instruction({'answer_type':'mcq','answer':'SECRET'})==MCQ_INSTRUCTION
