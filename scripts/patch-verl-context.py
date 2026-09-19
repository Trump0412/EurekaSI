"""Apply the pinned VERL 0.7 nested-Qwen-config compatibility patch idempotently."""
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess

if importlib.metadata.version('verl')!='0.7.0':raise RuntimeError('Patch validated only for VERL 0.7.0')
root=Path(importlib.util.find_spec('verl').origin).parent
patch=Path(__file__).resolve().parents[1]/'patches/verl-qwen3vl-context.patch'
check=subprocess.run(['git','apply','--unidiff-zero','--check',str(patch)],cwd=root,capture_output=True)
if check.returncode==0:subprocess.run(['git','apply','--unidiff-zero',str(patch)],cwd=root,check=True)
else:subprocess.run(['git','apply','--unidiff-zero','--reverse','--check',str(patch)],cwd=root,check=True)
print('VERL nested context-limit patch applied; full GPU acceptance is separate')
