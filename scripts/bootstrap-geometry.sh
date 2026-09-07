#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$PROJECT_DIR/scripts/bootstrap-common.sh"
ENCODER=${1:?Usage: bootstrap-geometry.sh vggt|da3|pi3}
case "$ENCODER" in vggt|da3|pi3) ;; *) exit 2;; esac
PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10-3.12"'
TORCH_FLAVOR=${TORCH_FLAVOR:-cu124}
case "$TORCH_FLAVOR" in cpu|cu124|cu121) ;; *) echo "Unsupported TORCH_FLAVOR" >&2; exit 2;; esac
CONSTRAINTS="$PROJECT_DIR/requirements/geometry-constraints.txt"
# The activated core environment owns workspace paths and pinned source fetching.
spatial source "$ENCODER"
SOURCE_DIR=$(python -c 'from spatial_intelligence.workspace import settings; import sys; from pathlib import Path; print(Path(settings()["external"])/sys.argv[1])' "$ENCODER")
ENV_DIR=${SPATIAL_GEOMETRY_VENV:-"$PROJECT_DIR/.venv-$ENCODER"}
spatial_prepare_venv
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements/core.txt"
"$ENV_DIR/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 -c "$PROJECT_DIR/requirements/core.txt" --index-url "https://download.pytorch.org/whl/${TORCH_FLAVOR:-cu124}"
if [[ "$ENCODER" == pi3 ]]; then
  "$ENV_DIR/bin/python" -m pip install -r "$SOURCE_DIR/requirements.txt" -c "$CONSTRAINTS"
fi
"$ENV_DIR/bin/python" -m pip install -e "$SOURCE_DIR" -c "$CONSTRAINTS"
"$ENV_DIR/bin/python" -m pip install -e "$PROJECT_DIR" -r "$PROJECT_DIR/requirements/core.txt" -c "$CONSTRAINTS"
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze > "$ENV_DIR/resolved-requirements.txt"
rm -- "$INCOMPLETE_MARKER"
echo "Geometry environment created. Verify CUDA and upstream optional extensions before cache extraction."
echo "Use $ENV_DIR/bin/spatial cache-geometry --help"
