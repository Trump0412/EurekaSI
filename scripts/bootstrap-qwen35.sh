#!/usr/bin/env bash
# Dedicated Conda profile; never mutates the legacy native environment.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${EUREKASI_ROOT:?Set EUREKASI_ROOT to your persistent experiment directory}
CONDA_BIN=${CONDA_BIN:-conda}
ENV_DIR="$ROOT/envs/qwen35"
mkdir -p "$ROOT/logs" "$ROOT/envs" "$ROOT/cache/pip"
mkdir -p "$ROOT/receipts"
mkdir -p "$ROOT/state"
exec 9>"$ROOT/state/bootstrap.lock"
flock -n 9 || { echo 'Another bootstrap is running'; exit 2; }
record_environment() {
  python3 - "$1" <<'PY'
import json,os,pathlib,sys
p=pathlib.Path(os.environ['EUREKASI_ROOT'])/'receipts/environment.json'
t=p.with_suffix('.tmp')
t.write_text(json.dumps({'status':sys.argv[1],'environment':os.environ['EUREKASI_ROOT']+'/envs/qwen35','log':'logs/bootstrap-qwen35.log'}))
t.replace(p)
PY
}
trap 'record_environment failed' ERR
record_environment running
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export PIP_CACHE_DIR="$ROOT/cache/pip"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_DIR" --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main python=3.12 pip
fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "${TORCH_INDEX_URL:-$PIP_INDEX_URL}"
"$PY" -m pip install -r "$PROJECT_DIR/requirements/qwen35.txt"
"$PY" -m pip install -e "$PROJECT_DIR" --no-deps
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/logs/qwen35-freeze.txt"
"$PY" -c 'import torch; from transformers import Qwen3_5ForConditionalGeneration; print(torch.__version__, torch.cuda.device_count()); assert torch.cuda.is_available()'
record_environment complete
echo "Environment ready: $ENV_DIR"
