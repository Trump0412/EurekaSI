import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('cookie_login',Path(__file__).resolve().parents[1]/'scripts/baidu-cookie-login.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_normalization():
    assert m.normalize_cookie('Cookie: BDUSS=dummy; STOKEN=test')==('BDUSS=dummy; STOKEN=test',True)
    assert m.normalize_cookie('BDUSS=dummy')[1] is False


@pytest.mark.parametrize('value',['BDUSS=x\nhelp','BDUSS="x"','BDUSS=x\\y','BDUSS=x; BDUSS=y','OTHER=x','BDUSS='])
def test_reject_bad_cookie(value):
    with pytest.raises(ValueError):m.normalize_cookie(value)


def test_secret_only_sent_via_stdin(monkeypatch,tmp_path):
    secret=b'login -cookies="BDUSS=dummy-test-secret"\n'
    def fake(argv,**kwargs):
        assert argv==['/test/client']
        assert kwargs['input']==secret
        assert 'dummy-test-secret' not in repr(argv)+repr(kwargs['env'])
        assert kwargs['stdout']==m.subprocess.PIPE
        assert not kwargs.get('shell',False)
        return SimpleNamespace(returncode=0,stdout=b'')
    monkeypatch.setattr(m.subprocess,'run',fake)
    m.execute('/test/client',tmp_path,secret)
