"""No GPU imports: paired sampling and worker entrypoint contracts."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('geometry_rft_worker',
    Path(__file__).resolve().parents[1]/'scripts/train-geometry-rft.py')
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixture():
    return {'4drl':[{'id':f'd-{i}','source':'4drl'} for i in range(19)],
            'spatialladder':[{'id':f's-{i}','source':'spatialladder'} for i in range(11)]}


def test_legacy_eval_preserves_evidence_and_does_not_put_gold_in_question():
    row={'id':'e1','question':'Which object?', 'choices':{'A':'chair','B':'table'},
         'ground_truth':'B','source':'clip.mp4','media':['f0','f5'],'frame_indices':[0,5]}
    result=worker.evaluation_row(row,lambda path:(29.97,100))
    assert result['question']==row['question']
    assert result['media']==row['media'] and result['frame_indices']==[0,5]
    assert result['fps']==29.97 and result['total_num_frames']==100
    assert result['answer_type']=='mcq' and result['answer']=='B'
    assert 'input_mode' not in row


def test_canonical_eval_is_not_reinterpreted_as_legacy_video():
    row={'question':'count','answer':'2','answer_type':'numeric','input_mode':'images'}
    assert worker.evaluation_row(row)==row


def test_mixture_exact_seven_three_each_block_and_control_only_four_d():
    pools=fixture(); mixed={'four_d_rl_fraction':.7,'spatial_fraction':.3}
    control={'four_d_rl_fraction':1.,'spatial_fraction':0.}
    for start in range(0,100,10):
        selected=[worker.sampled_row(pools,mixed,3407,i) for i in range(start,start+10)]
        assert sum(r['source']=='4drl' for r in selected)==7
        assert sum(r['source']=='spatialladder' for r in selected)==3
    assert all(worker.sampled_row(pools,control,3407,i)['source']=='4drl' for i in range(100))


def test_resume_and_world_size_do_not_change_global_sample_stream():
    pools=fixture(); arm={'four_d_rl_fraction':.7,'spatial_fraction':.3}
    expected=[worker.sampled_row(pools,arm,3407,i)['id'] for i in range(128)]
    for world in (4,8):
        by_index={}
        for step in range(8):
            for rank in range(world):
                for local in range(16//world):
                    index=step*16+local*world+rank
                    by_index[index]=worker.sampled_row(pools,arm,3407,index)['id']
        assert [by_index[i] for i in range(128)]==expected
    resumed=[worker.sampled_row(pools,arm,3407,i)['id'] for i in range(48,128)]
    assert resumed==expected[48:]
