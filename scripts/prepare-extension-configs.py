"""Materialize explicit, disabled-until-validated benchmark/training recipes."""
import argparse
import json
from pathlib import Path
import yaml

REPO=Path(__file__).resolve().parents[1]


def prepare(root):
    plan=json.loads((REPO/'configs/post-download.json').read_text())
    assets=yaml.safe_load((REPO/'catalog/assets.yaml').read_text())
    benchmarks=yaml.safe_load((REPO/'catalog/benchmarks.yaml').read_text())
    dest=root/'extension-configs';dest.mkdir(parents=True,exist_ok=True)
    jobs={}; downloads={}
    for name in plan['benchmarks']:
        item=benchmarks[name]
        jobs[name]={'kind':'benchmark','enabled':False,'catalog':item,
                    'model':str(root/'models/Qwen3.5-2B'),'status':'needs_model_engine_media_preflight',
                    'required_checks':['model registered in engine','official prompt/frame/scorer','QA and media coverage','no train-scene overlap'],
                    'warnings':[item.get('notes','')],'evaluation_only':True}
        asset=item.get('asset')
        if asset and name!='revsi':
            raw=assets[asset]
            downloads['extra-'+asset]={'repo':raw['repo_id'],'kind':raw['repo_type'],'revision':raw['revision'],
                                      'patterns':['*'],'priority':3,'destination':'datasets/extensions/'+asset}
    for name in plan['training_assets']:
        raw=assets[name]
        jobs[name]={'kind':'training_dataset','asset':raw,'enabled':False,'mixture_weight':0,
                    'status':'needs_conversion_media_and_split_audit','heldout_benchmarks':plan['benchmarks'],
                    'checks':['official train split only','QA-media resolution','scene dedup','marker/coordinate convention'],
                    'notes':'VLM-3R VST must use corrected labels; existing VSI590K/VLM3R downloads are reused'}
        if name not in ['vlm3r-data','vsi590k']:
            downloads['extra-'+name]={'repo':raw['repo_id'],'kind':raw['repo_type'],'revision':raw['revision'],
                                     'patterns':['*'],'priority':3,'destination':'datasets/extensions/'+name}
    for name in ['opd','opsd']:
        jobs[name]={'kind':'posttraining','enabled':False,'student':str(root/'models/Qwen3.5-2B'),
                    'teacher':None,'status':'requires_qwen35_multimodal_backend_gate',
                    'checks':['response alignment','teacher frozen','nonzero learning signal','save/reload'],
                    'notes':'OPD also needs an explicitly selected same-tokenizer teacher; OPSD uses answer-conditioned teacher'}
    jobs['gspo']={'kind':'rl','enabled':False,'framework':'verl==0.7.0','environment':'verl-legacy',
                  'status':'framework_only_qwen35_not_supported_by_pinned_rollout',
                  'required_checks':['compatible model/engine','multimodal reward adapter','actual rollout','nonzero optimizer update','resume'],
                  'notes':'Do not silently switch the research backbone. Newer Qwen3.5 stack requires separate driver/engine validation.'}
    (dest/'resources.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'downloads.json').write_text(json.dumps(downloads,indent=2),encoding='utf-8')
    return {'status':'configured_not_executed','resources':len(jobs),'path':str(dest),'auto_extra_download':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    print(json.dumps(prepare(Path(p.parse_args().root).resolve()),indent=2))
