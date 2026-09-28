"""Version-locked fix for VERL 0.7/vLLM 0.11.2 non-blocking executor contract."""
from concurrent.futures import Future
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess

if importlib.metadata.version('verl') != '0.7.0' or importlib.metadata.version('vllm') != '0.11.2':
    raise RuntimeError('Executor compatibility patch only validated for VERL0.7/vLLM0.11.2')
root = Path(importlib.util.find_spec('verl').origin).parent
patch = Path(__file__).resolve().parents[1] / 'patches/verl-vllm0112-executor.patch'
check = subprocess.run(['git', 'apply', '--check', str(patch)], cwd=root, capture_output=True)
if check.returncode == 0:
    subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
else:
    subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=root, check=True)

from verl.workers.rollout.vllm_rollout.vllm_async_server import ExternalZeroMQDistributedExecutor
executor = object.__new__(ExternalZeroMQDistributedExecutor)
executor.collective_rpc = lambda *args, **kwargs: [None]
result = executor.execute_model(object(), non_block=True)
assert isinstance(result, Future) and result.result() is None
executor.collective_rpc = lambda *args, **kwargs: ['sampled-token-result']
result = executor.sample_tokens(None, non_block=True)
assert isinstance(result, Future) and result.result() == 'sampled-token-result'
print('Non-blocking execute_model/sample_tokens Future contract passed; real GPU gate remains required')
