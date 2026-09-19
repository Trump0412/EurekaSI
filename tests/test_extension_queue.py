import importlib.util
import json
from pathlib import Path
import sys
import pytest

REPO=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_extension_configs_are_not_training_authorization(tmp_path):
    result=module('prepare-extension-configs').prepare(tmp_path)
    configs=json.loads((tmp_path/'extension-configs/resources.json').read_text())
    assert result['status']=='configured_not_executed'
    assert {'vsibench','revsi','mmsi','viewspatial','mindcube','dsr','opd','opsd','gspo'}<=set(configs)
    assert all(item['enabled'] is False for item in configs.values())
    assert all(item.get('mixture_weight',0)==0 for item in configs.values())


@pytest.mark.skipif(sys.platform!='linux',reason='Linux supervisor uses flock')
def test_download_failure_does_not_release_deployment_barrier(tmp_path):
    receipts=tmp_path/'receipts';receipts.mkdir()
    (receipts/'a.json').write_text('{"status":"complete"}')
    (receipts/'b.json').write_text('{"status":"failed"}')
    assert module('post-download-deploy').pending(tmp_path,['a','b','c'])=={'b':'failed','c':'not_started'}


@pytest.mark.skipif(sys.platform!='linux',reason='Source installer uses flock')
def test_recovery_refuses_modified_local_source(tmp_path,monkeypatch):
    installer=module('prepare-extension-sources')
    def download(cmd,**kwargs):
        Path(cmd[cmd.index('-o')+1]).write_text('upstream')
    monkeypatch.setattr(installer.subprocess,'run',download)
    item={'url':'https://github.com/example/repo','restored_files':[{'path':'helper.py','commit':'pinned'}]}
    installer.restore_files(tmp_path,item)
    assert (tmp_path/'helper.py').read_text()=='upstream'
    installer.restore_files(tmp_path,item)
    (tmp_path/'helper.py').write_text('user edit')
    with pytest.raises(ValueError,match='overwrite'):
        installer.restore_files(tmp_path,item)
    assert (tmp_path/'helper.py').read_text()=='user edit'


def test_diagnostic_reward_is_scoped_and_nonconstant():
    reward=module('verl-hound-reward').compute_score
    assert reward('hound_diagnostic_train','red ball','red ball')==1
    assert reward('hound_diagnostic_train','blue cube','red ball')==0
    with pytest.raises(ValueError):reward('revsi','red ball','red ball')
