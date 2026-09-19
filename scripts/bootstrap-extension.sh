#!/usr/bin/env bash
# Invoked only after the download barrier by the persistent extension supervisor.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${EUREKASI_ROOT:?Set EUREKASI_ROOT}
PROFILE=${1:?legacy or verl}
CONDA_BIN=${CONDA_BIN:?Set CONDA_BIN}
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export PIP_CACHE_DIR="$ROOT/cache/pip" PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
case "$PROFILE" in
  legacy) ENV_DIR="$ROOT/envs/legacy-geo"; VERSION=3.10 ;;
  verl) ENV_DIR="$ROOT/envs/verl-legacy"; VERSION=3.12 ;;
  *) exit 2 ;;
esac
exec 9>"$ROOT/state/extension-$PROFILE-install.lock"
flock -n 9 || { echo 'Another installer owns this environment'; exit 2; }
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_DIR" --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main "python=$VERSION" pip
fi
PY="$ENV_DIR/bin/python"
if [[ "$PROFILE" == legacy ]]; then
  "$PY" -m pip install torch==2.5.1 torchvision==0.20.1
  "$PY" -m pip install -r "$REPO/requirements/native.txt"
  "$PY" -m pip install -e "$REPO" --no-deps
else
  # Install one coherent engine/torch stack, never upgrade the active qwen35 env.
  "$PY" -m pip install -r "$REPO/requirements/verl-legacy.txt"
  "$PY" "$REPO/scripts/patch-verl-context.py"
fi
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/logs/extension-$PROFILE-freeze.txt"
"$PY" -c 'import torch,transformers; print(torch.__version__,transformers.__version__)'
if [[ "$PROFILE" == verl ]]; then
  "$PY" -c 'import verl,ray,vllm; from verl.trainer.ppo import core_algos; print("VERL imports available; GPU rollout/GSPO updates NOT validated")'
fi
