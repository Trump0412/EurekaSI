import importlib.util
from pathlib import Path
import shlex

import pytest


spec=importlib.util.spec_from_file_location('fleet_status',Path(__file__).resolve().parents[1]/'scripts/fleet-status.py')
fleet=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet)


def test_remote_command_quotes_root_and_disables_forwarding():
    root="/persistent/node with space/'quoted'"
    command=fleet.ssh_command('train-host',root)
    assert shlex.split(command[-1])==['python3','-c',fleet.REMOTE,root]
    assert 'ForwardAgent=no' in command and 'BatchMode=yes' in command
    compile(fleet.REMOTE,'<remote-status>','exec')


@pytest.mark.parametrize('host,root',[('-Fbad','/data'),('host;bad','/data'),('host','relative'),('host','/a\nb')])
def test_invalid_target_rejected(host,root):
    with pytest.raises(ValueError):fleet.ssh_command(host,root)
