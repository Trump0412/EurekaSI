import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('route_stage',Path(__file__).resolve().parents[1]/'scripts/train-georoute-stage.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


def test_optimizer_batch_replay_and_full_instruction_coverage():
    rows=[{'id':str(i),'media':['a','b']} for i in range(1001)]
    arranged,counts=stage.schedule(rows,'sft')
    assert counts==dict(original_rows=1001,tail_padding=23,tip_replay_rows=64,
        total_scheduled_rows=1088,unit='15 instruction optimizer updates then 1 TIP update')
    assert {r['id'] for r in arranged if r['_task']=='sft'}=={r['id'] for r in rows}
    groups=[{r['_task'] for r in arranged[i:i+64]} for i in range(0,len(arranged),64)]
    assert groups[:15]==[{'sft'}]*15 and groups[15]=={'tip'} and groups[16]=={'sft'}
    assert stage.schedule(rows,'sft')==(arranged,counts)


def test_tip_excludes_single_image_not_fabricating_graph_edges():
    rows=[{'id':'a','media':['a']},{'id':'b','media':['a','b']}]
    arranged,count=stage.schedule(rows,'tip')
    assert {r['id'] for r in arranged}=={'b'} and count['tail_padding']==63
