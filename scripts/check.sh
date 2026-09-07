#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n "${SPATIAL_PYTHON:-}" ]]; then
  PYTHON_CMD=$SPATIAL_PYTHON
else
  ENV_DIR=${SPATIAL_VENV:-${VIRTUAL_ENV:-"$PROJECT_DIR/.venv"}}
  PYTHON_CMD="$ENV_DIR/bin/python"
fi
"$PYTHON_CMD" - <<'PY'
from pathlib import Path
import sys
if (Path(sys.prefix) / '.spatial-bootstrap-incomplete').exists():
    raise SystemExit('Selected environment has an incomplete bootstrap; rerun its installation script')
PY
"$PYTHON_CMD" -m pip check
"$PYTHON_CMD" scripts/static_check.py
"$PYTHON_CMD" scripts/release_check.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_CMD" -m pytest tests -q
