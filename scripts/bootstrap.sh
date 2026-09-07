#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$PROJECT_DIR/scripts/bootstrap-common.sh"
PROFILE=${1:-native}
case "$PROFILE" in native|cpu|core) ;; *) echo "Usage: bash scripts/bootstrap.sh [native|cpu|core]" >&2; exit 2;; esac
PYTHON_BIN=${PYTHON_BIN:-python3}
ENV_DIR=${SPATIAL_VENV:-"$PROJECT_DIR/.venv"}
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10-3.12 (recommended: 3.12)"'
TORCH_FLAVOR=${TORCH_FLAVOR:-cu124}
if [[ "$PROFILE" == cpu ]]; then TORCH_FLAVOR=cpu; fi
case "$TORCH_FLAVOR" in cpu|cu124|cu121) ;; *) echo "Unsupported TORCH_FLAVOR; use cpu/cu121/cu124" >&2; exit 2;; esac
spatial_prepare_venv
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements/core.txt"
if [[ "$PROFILE" != core ]]; then
  "$ENV_DIR/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 -c "$PROJECT_DIR/requirements/core.txt" --index-url "https://download.pytorch.org/whl/$TORCH_FLAVOR"
  "$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements/native.txt"
fi
"$ENV_DIR/bin/python" -m pip install -e "$PROJECT_DIR" --no-deps
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements/core.txt"
"$ENV_DIR/bin/python" -m pip check
rm -- "$INCOMPLETE_MARKER"
echo "Ready. Activate with: source $ENV_DIR/bin/activate"
echo "Next: spatial init --root /data/spatial"
