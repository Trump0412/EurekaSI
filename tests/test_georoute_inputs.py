"""CPU preprocessing/cache contracts; real processor test is opt-in via env paths."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from spatial_intelligence.georoute import build_graph
from spatial_intelligence.georoute_inputs import canvas,RouteCollator,GraphCache


def test_letterbox_canvas_exact_coordinates_and_padding(tmp_path):
    path=tmp_path/'landscape.png';Image.new('RGB',(100,50),(255,0,0)).save(path)
    image,transform=canvas(path)
    assert image.size==(448,448)
    assert transform['source_size']==[100,50]
    assert transform['resized_size']==[448,224]
    assert transform['offset']==[0,112]
    assert transform['valid_box']==[0,112,448,336]
    assert image.getpixel((224,111))==(0,0,0)
    assert image.getpixel((224,112))==(255,0,0)
    assert image.getpixel((224,335))==(255,0,0)
    assert image.getpixel((224,336))==(0,0,0)


@pytest.mark.parametrize('count',[0,33])
def test_frame_count_rejected_before_processor_or_media_access(count):
    collator=RouteCollator(SimpleNamespace(tokenizer=SimpleNamespace()))
    with pytest.raises(ValueError,match='1..32'):
        collator([{'media':['unused']*count}])


def test_geometry_cache_reuses_pixels_not_answers_and_invalidates_media(tmp_path,monkeypatch):
    first=tmp_path/'first.png';second=tmp_path/'second.png'
    Image.new('RGB',(8,8),'red').save(first);Image.new('RGB',(8,8),'blue').save(second)
    cache=GraphCache(tmp_path/'cache',None,{'revision':'test'},dict(side=448),'cpu')
    calls=[]
    def build(row):
        calls.append(list(row['media']))
        return build_graph([0,1],[],[],[])
    monkeypatch.setattr(cache,'build',build)
    row={'media':[str(first),str(second)],'question':'one','answer':'A'}
    cache.get(row);cache.get(dict(row,question='different',answer='B'))
    assert len(calls)==1, 'A geometry-only cache must not depend on QA labels'
    old=first.stat().st_mtime_ns
    Image.new('RGB',(9,8),'green').save(first)
    os.utime(first,ns=(old+1000000,old+1000000))
    cache.get(row)
    assert len(calls)==2
    cache.get(dict(row,media=list(reversed(row['media']))))
    assert len(calls)==3, 'Frame order is part of geometry identity'
    assert len(list((tmp_path/'cache').glob('*.pt')))==3


def test_real_qwen_processor_two_training_samples_batch_parity_and_labels():
    processor_path=os.getenv('GEOROUTE_PROCESSOR')
    manifest=os.getenv('GEOROUTE_MANIFEST')
    if not processor_path or not manifest:
        pytest.skip('Set GEOROUTE_PROCESSOR and GEOROUTE_MANIFEST for real CPU input acceptance')
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    selected=[]
    with Path(manifest).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if not selected and len(row['media'])==1: selected.append(row)
            elif selected and 2<=len(row['media'])<=8:
                selected.append(row);break
    assert len(selected)==2, 'Need one real single-image and one real multi-image row'
    processor=AutoProcessor.from_pretrained(processor_path,local_files_only=True)
    training=RouteCollator(processor,training=True)
    combined=training(selected)
    total_frames=sum(len(row['media']) for row in selected)
    assert combined['image_grid_thw'].tolist()==[[1,28,28]]*total_frames
    assert combined['pixel_values'].shape[0]==total_frames*28*28
    offset=0
    for index,row in enumerate(selected):
        single=training([row]);length=int(single['attention_mask'].sum())
        assert torch.equal(combined['input_ids'][index,:length],single['input_ids'][0])
        assert torch.equal(combined['labels'][index,:length],single['labels'][0])
        patches=len(row['media'])*28*28
        assert torch.equal(combined['pixel_values'][offset:offset+patches],single['pixel_values'])
        offset+=patches
        labels=single['labels'][0];supervised=labels!=-100
        assert supervised.any()
        assert str(row['answer']) in processor.tokenizer.decode(labels[supervised],skip_special_tokens=True)
        # Changing ONLY the gold answer cannot alter the generation prompt.
        inference=RouteCollator(processor,training=False)
        original=inference([row]);changed=inference([dict(row,answer='GOLD_SENTINEL_NOT_IN_PROMPT')])
        assert torch.equal(original['input_ids'],changed['input_ids'])
        prompt_length=int(original['attention_mask'].sum())
        assert (labels[:prompt_length]==-100).all()
        assert torch.equal(single['input_ids'][0,:prompt_length],original['input_ids'][0])
    assert (combined['labels'][combined['attention_mask']==0]==-100).all()
    print(json.dumps({'accepted_real_samples':len(selected),'frame_counts':[len(r['media']) for r in selected],
        'image_grid':[1,28,28],'batch1_batch2_identical':True,'labels_prefix_masked':True,
        'source_ids':[r['id'] for r in selected]}))
