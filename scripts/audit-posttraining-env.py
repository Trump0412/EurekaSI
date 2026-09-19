"""Read-only environment inventory; does not claim distributed RL acceptance."""
import importlib.metadata as metadata
import importlib.util
import json
import sys

result={'python':sys.executable,'version':sys.version,'packages':{}}
for package,module in [('torch','torch'),('transformers','transformers'),('verl','verl'),
                       ('vllm','vllm'),('ray','ray'),('deepspeed','deepspeed'),
                       ('sglang','sglang'),('flash-attn','flash_attn')]:
    try:version=metadata.version(package)
    except metadata.PackageNotFoundError:version=None
    try:available=importlib.util.find_spec(module) is not None
    except (ValueError,ImportError):available=False
    result['packages'][package]={'installed_version':version,'discoverable':available}
print(json.dumps(result,indent=2))
