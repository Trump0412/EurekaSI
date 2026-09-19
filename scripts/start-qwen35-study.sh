#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export EUREKASI_ROOT=${1:?Usage: bash scripts/start-qwen35-study.sh /persistent/experiment-root}
export CONDA_BIN=${CONDA_BIN:-$(command -v conda || true)}
if [[ ! -x "$CONDA_BIN" ]]; then echo "Set CONDA_BIN to an existing Conda executable" >&2; exit 2; fi
mkdir -p "$EUREKASI_ROOT/logs" "$EUREKASI_ROOT/state"
export EUREKASI_DOWNLOAD_CONNECTIONS=${EUREKASI_DOWNLOAD_CONNECTIONS:-4}
if ! python3 -c 'import json,os,pathlib,sys; p=pathlib.Path(os.environ["EUREKASI_ROOT"])/"receipts/environment.json"; sys.exit(0 if p.exists() and json.loads(p.read_text()).get("status")=="complete" else 1)'; then
  nohup bash "$REPO/scripts/bootstrap-qwen35.sh" >> "$EUREKASI_ROOT/logs/bootstrap-qwen35.log" 2>&1 < /dev/null &
fi
python3 "$REPO/scripts/run-qwen35-study.py" --root "$EUREKASI_ROOT" --detach
