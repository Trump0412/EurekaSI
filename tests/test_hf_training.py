"""Real random-weight HF model: no downloads, no GPU, no benchmark claims."""
import copy
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from spatial_intelligence.backends.hf import HFBackend
from spatial_intelligence.data import model_input, normalize
from spatial_intelligence.io import load_config, write_jsonl
from spatial_intelligence.training import train


@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(1)
    vocab = {v:i for i,v in enumerate(["[UNK]","[EOS]","[PAD]","user","assistant","A","B","Which","left","right","answer","<",">","/","Training","only","verified",":","."])}
    raw=Tokenizer(WordLevel(vocab,unk_token="[UNK]"));raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token="[UNK]",eos_token="[EOS]",pad_token="[PAD]")
    tok.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    path=tmp_path/"tiny"
    torch.manual_seed(42)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab),n_positions=512,n_embd=16,n_layer=1,n_head=2,
                                     bos_token_id=1,eos_token_id=1,pad_token_id=2))
    model.save_pretrained(path);tok.save_pretrained(path)
    cfg=load_config("configs/sft.yaml")
    cfg["model"].update(path=str(path),device="cpu",dtype="float32",text_only=True,max_context=512)
    cfg["model"]["lora"]["enabled"]=False
    cfg["teacher"]=copy.deepcopy(cfg["model"])
    cfg["protocol"]["generation"]={"max_new_tokens":4,"do_sample":True,"temperature":1.0,"top_p":1.0,"top_k":0}
    cfg["train"].update(steps=2,gradient_accumulation=1,save_every=0,group_size=3,learning_rate=.01)
    training=tmp_path/"train.jsonl"; heldout=tmp_path/"test.jsonl"
    write_jsonl(training,[{"dataset":"toy","id":"train","question":"Which left?","answer":"A","split":"train","media":[],"metadata":{"teacher_response":"<answer>A</answer>"}}])
    write_jsonl(heldout,[{"dataset":"toy","id":"test","question":"Which right?","answer":"B","split":"test","media":[]}])
    cfg["data"]={"train":[{"path":str(training),"weight":1.}],"heldout":[str(heldout)],"eval":str(heldout)}
    return cfg,tmp_path


@pytest.mark.parametrize("mode", ["sft","rl","opd","opsd","offline_kd"])
def test_actual_training_and_reload(tiny,mode):
    cfg,root=tiny
    cfg["train"]["mode"]=mode
    cfg["output"]=str(root/mode)
    path=train(cfg)
    done=json.loads((Path(cfg["output"])/"completion.json").read_text())
    assert done["steps"]==2 and done["status"]=="optimizer_training_completed"
    backend=HFBackend({**cfg["model"],"path":path})
    sample=normalize({"id":"x","dataset":"toy","question":"Which?","media":[],"split":"test"})
    result=backend.generate(model_input(sample),cfg["protocol"],123)
    assert result["completion_tokens"]>0


def test_response_position_alignment_matches_hf_labels(tiny):
    cfg,_=tiny
    backend=HFBackend(cfg["model"],training=True)
    sample=normalize({"id":"x","dataset":"toy","question":"Which?","media":[],"split":"test"})
    batch=backend.encode(model_input(sample),cfg["protocol"])
    ids=backend.response_ids("A")
    from spatial_intelligence.objectives import sft_loss
    actual=sft_loss(backend.response_logits(batch,ids),ids)
    inp=torch.cat([batch["input_ids"],ids],dim=1)
    labels=inp.clone();labels[:,:batch["input_ids"].shape[1]]=-100
    full={**batch,"input_ids":inp,"attention_mask":torch.ones_like(inp),"labels":labels}
    if "token_type_ids" in full:
        full["token_type_ids"]=torch.cat([full["token_type_ids"],torch.zeros_like(ids)],dim=1)
    expected=backend.model(**full).loss
    assert torch.allclose(actual,expected,atol=1e-6)


def test_media_cannot_be_silently_dropped(tiny):
    cfg,_=tiny
    backend=HFBackend(cfg["model"])
    sample=normalize({"id":"x","dataset":"toy","question":"Which?","media":["fake.png"],"split":"test"})
    with pytest.raises(ValueError,match="discard"):
        backend.encode(model_input(sample),cfg["protocol"])


def test_lora_training_checkpoint_reload(tiny):
    cfg,root=tiny
    cfg["model"]["lora"].update(enabled=True,rank=2,alpha=4,target_modules=["c_attn"])
    cfg["output"]=str(root/"lora")
    path=train(cfg)
    backend=HFBackend({**cfg["model"],"adapter":path})
    assert all(not p.requires_grad for p in backend.model.parameters())
    assert (Path(path)/"adapter_config.json").exists()


def test_fusion_train_save_reload_and_hook_cleanup(tiny):
    import numpy as np
    from spatial_intelligence.io import digest,file_digest,write_json
    from spatial_intelligence.backends.fusion import FusionBackend
    cfg,root=tiny
    blob=root/'geometry.npz';np.savez(blob,tokens=np.arange(56,dtype=np.float32).reshape(8,7)/56)
    index=root/'index.json';write_json(index,{'entries':{digest([]):{'path':str(blob),'sha256':file_digest(blob)}}})
    cfg['model'].update(backend='spatial_intelligence.backends.fusion:FusionBackend',options={'cache_index':str(index),'checkpoint':None,'heads':4})
    cfg['output']=str(root/'fusion')
    final=Path(train(cfg))
    backend=FusionBackend({**cfg['model'],'path':str(final),'options':{**cfg['model']['options'],'checkpoint':str(final/'spatial_fusion.pt')}})
    sample=normalize({'id':'x','dataset':'toy','question':'Which?','media':[],'split':'test'})
    result=backend.generate(model_input(sample),cfg['protocol'],7)
    assert result['completion_tokens']>0
    assert len(backend.model.get_input_embeddings()._forward_hooks)==0
    batch=backend.encode(model_input(sample),cfg['protocol'])
    with pytest.raises(ValueError):
        backend.response_logits(batch,torch.zeros((1,600),dtype=torch.long))
    assert len(backend.model.get_input_embeddings()._forward_hooks)==0
    assert backend.model.spatial_fusion.gate.abs()>0
