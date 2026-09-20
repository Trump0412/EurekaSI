import importlib.util
import json
from pathlib import Path


def test_public_summary_does_not_copy_paths_or_raw_logs(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/summarize-throughput.py'
    spec=importlib.util.spec_from_file_location('throughput_summary',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    private='/private/operator/secret-machine'
    result={'backend':'auto','layout':'legacy','micro_batch':1,'ga':16,'samples_per_second':10.,
        'effective_tokens_per_second':1000.,'padding_fraction':0.,'peak_reserved_gib':20.,'gpu_total_gib':40.,
        'timed_steps':9,'warmup_steps':3,'checkpoint':private,'raw_log':private}
    recommendation={'status':'complete','results':{'auto-legacy-b1':[result,result]},'checkpoint':private+'/checkpoint-600',
        'training_options':{'batch_size':1},'speed_ratio':1.,'accepted':False}
    parity={'cases':[{'backend':'auto','micro_batch':1,'loss':1.,'kernels':[private]}],
        'comparison_reference':'auto micro1','eligible_variants':[{'backend':'auto','micro_batch':1}]}
    (tmp_path/'recommendation.json').write_text(json.dumps(recommendation))
    (tmp_path/'model-parity.json').write_text(json.dumps(parity))
    (tmp_path/'samples.jsonl').write_text(json.dumps({'id':'public-dataset-row-1','media':[private]})+'\n')
    report=module.summarize(tmp_path)
    assert private not in json.dumps(report)
    assert report['source_checkpoint_step']==600
    assert report['table'][0]['numerically_eligible']
