import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import pytest
pytestmark = pytest.mark.distributed
from transformers import GPT2LMHeadModel
from test_hf_training import tiny
from spatial_intelligence.training import train
from spatial_intelligence.io import load_config,write_json,write_jsonl,read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def launch(argv):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT),'PYTHONNOUSERSITE':'1'}
    # Let Gloo select its interface unless the test environment explicitly pins one.
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','2',*argv],cwd=ROOT,env=env,timeout=90,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr


def test_unused_gradients_collective_order():
    launch(['tests/distributed_worker.py'])


def test_two_rank_training_matches_single_global_batch(tiny):
    cfg,root=tiny
    # Test collective semantics rather than FP32 reduction-order roundoff in
    # mathematically zero GPT2 key-bias gradients amplified by Adam's epsilon.
    # Keep the existing tight weight tolerance; do not loosen it to hide drift.
    cfg['model']['dtype']='float64'
    rows=read_jsonl(cfg['data']['train'][0]['path'])
    rows.append({**rows[0],'id':'train2','question':'Which answer?','answer':'B'})
    write_jsonl(cfg['data']['train'][0]['path'],rows)
    cfg['train'].update(batch_size=2,gradient_accumulation=2)
    cfg['output']=str(root/'single');baseline=train(cfg)
    cfg['train']['batch_size']=1;cfg['output']=str(root/'parallel')
    plan=root/'plan.json';write_json(plan,cfg)
    launch(['-m','spatial_intelligence','train','--config',str(plan)])
    a=GPT2LMHeadModel.from_pretrained(baseline).state_dict();b=GPT2LMHeadModel.from_pretrained(root/'parallel/final').state_dict()
    assert a.keys()==b.keys()
    for k in a:assert torch.allclose(a[k],b[k],atol=3e-6,rtol=3e-5),k
    info=json.loads((root/'parallel/completion.json').read_text());assert info['world_size']==2 and info['global_prompt_batch']==4
    assert all(len(r['sample_ids'])==4 for r in read_jsonl(root/'parallel/metrics.jsonl'))


def test_two_rank_inference_exact_coverage(tmp_path):
    cfg=load_config('configs/smoke.yaml');cfg['model']['device']='cpu';cfg['output']=str(tmp_path/'inference')
    plan=tmp_path/'plan.json';write_json(plan,cfg)
    launch(['-m','spatial_intelligence','infer','--config',str(plan)])
    report=json.loads((tmp_path/'inference/metrics.json').read_text());assert report['n']==2 and report['accuracy']==.5
    preds=read_jsonl(tmp_path/'inference/predictions.jsonl');assert len(preds)==len({p['id'] for p in preds})==2
