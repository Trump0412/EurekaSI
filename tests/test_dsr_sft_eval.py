import importlib.util
import json
from pathlib import Path
import pytest
from spatial_intelligence.dsr_sft_eval import validate_rows, select_rows, validate_records, summarize, paired_difference


def row(i, kind='a', frames=2):
    return dict(id=str(i), answer_type='mcq', choices={'A':'one','B':'two'}, answer='A',
                media=['frame']*frames, input_mode='video', frame_indices=list(range(frames)),
                total_num_frames=frames, fps=2, question_type=kind)


def record(i, correct, parsed='A'):
    return dict(id=str(i), question_type='a', score=dict(answer=int(correct), parsed_answer=parsed), truncated=False)


def test_duplicate_and_frame_validation():
    with pytest.raises(ValueError): validate_rows([row(0), row(0)])
    bad=row(1); bad['frame_indices']=[0,0]
    with pytest.raises(ValueError): validate_rows([bad])


def test_smoke_fixed_types_and_longest(tmp_path):
    path=tmp_path/'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in [row(0), row(1,'b'), row(2,frames=8)]))
    assert [r['id'] for r in select_rows(path,True)] == ['0','1','2']


def test_exact_prediction_coverage():
    with pytest.raises(ValueError): validate_records([record(0,True)], [row(0),row(1)],True)
    with pytest.raises(ValueError): validate_records([record(2,True)], [row(0)])
    assert len(validate_records([record(0,True)], [row(0)],True))==1


def test_accuracy_not_composite_reward():
    report=summarize([record(0,True),record(1,False,None)])
    assert report['accuracy']==50
    assert report['parsed_rate']==.5
    assert report['per_task']['a']['accuracy']==50


def test_paired_correctness():
    result=paired_difference([record(0,True),record(1,False)], [record(0,False),record(1,True)])
    assert result['right_minus_left_pp']==0
    assert result['right_only_correct']==['1']
    assert result['left_only_correct']==['0']
    with pytest.raises(ValueError): paired_difference([record(0,True)],[record(1,True)])


def test_tail_waits_gpu_release(tmp_path):
    spec=importlib.util.spec_from_file_location('dsr_queue',Path(__file__).resolve().parents[1]/'scripts/run-dsr-sft-queue.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    state=tmp_path/'state.json'
    plan={'dependencies':[dict(path=str(state),statuses=['complete','complete_with_failures'],require_gpu_release=True)]}
    state.write_text(json.dumps(dict(status='complete',gpu_work_finished=False)))
    assert mod.dependency_reasons(plan)
    state.write_text(json.dumps(dict(status='complete_with_failures',gpu_work_finished=True)))
    assert not mod.dependency_reasons(plan)
def test_explicit_mcq_prompt_keeps_default_unchanged():
    from spatial_intelligence.dsr_sft_eval import prompt_protocol,MCQ_INSTRUCTION
    assert prompt_protocol({}) is None
    assert prompt_protocol({'prompt_mode':'mcq-tagged-v2'})==MCQ_INSTRUCTION
    assert 'numeric' not in MCQ_INSTRUCTION and 'OPTION_LETTER' in MCQ_INSTRUCTION
    import pytest
    with pytest.raises(ValueError):prompt_protocol({'prompt_mode':'unknown'})
