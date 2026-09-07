import json
import tarfile
import io
from pathlib import Path
import pytest
import yaml
from PIL import Image
import numpy as np
from spatial_intelligence.workspace import initialize
from spatial_intelligence.assets import safe_extract
from spatial_intelligence.build_data import build_annotations,partition
from spatial_intelligence.data import normalize,model_input,leakage
from spatial_intelligence.io import read_jsonl,write_jsonl
from spatial_intelligence.quality import clean
from spatial_intelligence.geometry.cache import point_tokens,unproject,cache_key


def test_archive_traversal_rejected(tmp_path):
    archive=tmp_path/'evil.tar'
    with tarfile.open(archive,'w') as t:
        info=tarfile.TarInfo('../escape');info.size=1;t.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError):safe_extract(archive,tmp_path/'out')
    assert not (tmp_path/'escape').exists()


def test_build_spatial_markers_and_clean_geometry(tmp_path):
    im=tmp_path/'x.png';Image.new('RGB',(16,16),'white').save(im)
    src=tmp_path/'ann.json';src.write_text(json.dumps([{'id':0,'image':'x.png','conversations':[{'from':'human','value':'<image> Which point?\nOptions: A. left\nB. right'},{'from':'gpt','value':'B'}],'spar_info':{'type':'point'}}]))
    with pytest.raises(ValueError,match='markers required'):build_annotations(src,tmp_path/'bad.jsonl','spar','train',tmp_path,tmp_path/'frames')
    renderer=tmp_path/'renderer.py';renderer.write_text("def draw(im, info):\n    im.putpixel((0,0),(255,0,0))\nDRAW_FUNCTIONS={'point':draw}\n")
    dst=tmp_path/'good.jsonl';build_annotations(src,dst,'spar','train',tmp_path,tmp_path/'frames',marker_source=renderer)
    r=read_jsonl(dst)[0]
    assert r['id']=='0' and r['answer']=='B' and r['choices']=={'A':'left','B':'right'}
    assert Image.open(r['media'][0]).getpixel((0,0))==(255,0,0)
    assert Image.open(r['geometry_media'][0]).getpixel((0,0))==(255,255,255)
    assert 'answer' not in model_input(r)
    assert leakage([r],[normalize({**r,'dataset':'other','media':[str(im)]})])['media']


def test_point_geometry_contract():
    d=np.ones((1,2,2));k=np.eye(3)[None];e=np.eye(4)[None];e[0,0,3]=2
    p=unproject(d,k,e)
    np.testing.assert_allclose(p[0,0,0],[-2,0,1])
    t=point_tokens(p,d,2);assert t.shape==(4,7)
    with pytest.raises(ValueError):point_tokens(p*np.nan,d)


def test_conflicting_labels_both_excluded(tmp_path):
    im=tmp_path/'x.png';Image.new('RGB',(2,2)).save(im)
    base={'dataset':'d','question':'q','media':[str(im)],'split':'train'}
    rows=[normalize({**base,'id':1,'answer':'yes'}),normalize({**base,'id':2,'answer':'no'})]
    manifest=tmp_path/'d.jsonl';write_jsonl(manifest,rows)
    report=clean(manifest,tmp_path/'clean');assert report['kept']==0 and report['excluded']==2


def test_split_groups_and_no_repartition_test(tmp_path):
    rows=[normalize({'id':i,'dataset':'d','question':'q','answer':'a','media':[],'scene_id':'s'+str(i//2),'split':'train'}) for i in range(100)]
    path=tmp_path/'train.jsonl';write_jsonl(path,rows);partition(path,tmp_path/'split')
    scenes=[]
    for split in ['train','val','test']:scenes.append({r['scene_id'] for r in read_jsonl(tmp_path/'split'/(split+'.jsonl'))})
    assert not scenes[0]&scenes[1] and not scenes[0]&scenes[2]
    rows[0]['split']='test';write_jsonl(path,rows)
    with pytest.raises(ValueError):partition(path,tmp_path/'bad')


def test_workspace_does_not_overwrite(tmp_path):
    cfg=tmp_path/'local.yaml';initialize(tmp_path/'data',cfg)
    with pytest.raises(FileExistsError):initialize(tmp_path/'other',cfg)


def test_cluster_bootstrap_preserves_pairing():
    from spatial_intelligence.statistics import cluster_bootstrap
    result=cluster_bootstrap([1,1,1,1],['a','a','b','b'],100)
    assert result['ci95']==[1.,1.] and result['delta_b_minus_a']==1
    with pytest.raises(ValueError):cluster_bootstrap([0,1],['same','same'],100)
