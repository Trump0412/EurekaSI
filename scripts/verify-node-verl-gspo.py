"""Node diagnostic wrapper preserving the shared gate and adding rollout evidence.

Loads the repo's existing verifier rather than maintaining a divergent trainer.
Only native VERL invocations receive additional artifact logging overrides.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


class RecordedProcesses:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, args, *positional, **kwargs):
        if isinstance(args, list) and 'verl.trainer.main_ppo' in args:
            out = Path(sys.argv[sys.argv.index('--output') + 1])
            args = [*args, f'trainer.rollout_data_dir={out}/rollouts',
                    f'trainer.validation_data_dir={out}/validation']
            (out / 'command.json').write_text(json.dumps(args, indent=2))
        return subprocess.Popen(args, *positional, **kwargs)


if __name__ == '__main__':
    source = Path(__file__).with_name('verify-verl-gspo.py')
    spec = importlib.util.spec_from_file_location('shared_verl_gate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.subprocess = RecordedProcesses()
    module.main()
