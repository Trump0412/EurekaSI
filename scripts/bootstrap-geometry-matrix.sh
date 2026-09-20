#!/usr/bin/env bash
# Clone a prepared Qwen environment; never modify the active source prefix.
set -euo pipefail

: "${NODE_ROOT:?Set NODE_ROOT to an independent persistent root for this node}"
: "${SOURCE_ENV:?Set SOURCE_ENV to a prepared compatible Qwen training Conda prefix}"
: "${CONDA_BIN:?Set CONDA_BIN to the Conda executable}"

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_prefix="$(readlink -f -- "$SOURCE_ENV")"
node_prefix="$(readlink -m -- "$NODE_ROOT")"
target_prefix="$(readlink -m -- "$node_prefix/envs/geometry-matrix")"

if [[ ! -x "$CONDA_BIN" || ! -x "$source_prefix/bin/python" || ! -f "$source_prefix/conda-meta/history" ]]; then
    printf '%s\n' 'Conda executable and a complete prepared source environment are required.' >&2
    exit 2
fi
if [[ "$target_prefix" == "$source_prefix" || "$target_prefix" == "$source_prefix/"* ]]; then
    printf '%s\n' 'Refusing to clone/install into the source environment or its descendants.' >&2
    exit 2
fi
if [[ -e "$target_prefix" && ( ! -x "$target_prefix/bin/python" || ! -f "$target_prefix/conda-meta/history" ) ]]; then
    printf '%s\n' 'Target prefix exists but is incomplete. Inspect it manually; this script will not overwrite or delete it.' >&2
    exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export DS_BUILD_OPS=0
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1

if [[ ! -e "$target_prefix" ]]; then
    # Explicit mirror avoids silently accepting unrelated channel terms.
    "$CONDA_BIN" create -y --copy --override-channels \
        -c "${GEOMETRY_CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}" \
        --prefix "$target_prefix" --clone "$source_prefix"
fi

if [[ "$(readlink -f -- "$target_prefix")" == "$(readlink -f -- "$source_prefix")" ]]; then
    printf '%s\n' 'Resolved target unexpectedly aliases the source; refusing installation.' >&2
    exit 2
fi

# Existing complete targets are reused, never re-cloned. Pins prevent a silent
# Torch/Transformers upgrade while adding DeepSpeed to the independent copy.
"$target_prefix/bin/python" -m pip install --no-build-isolation \
    --index-url "${GEOMETRY_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    -r "$repo_dir/requirements/geometry-matrix.txt"
"$target_prefix/bin/python" -m pip check
"$target_prefix/bin/python" - <<'PY'
import json
import importlib.metadata as metadata
import torch
import transformers
import deepspeed

expected = {
    "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "5.3.0",
    "numpy": "1.26.4", "deepspeed": "0.18.8",
}
actual = {name: metadata.version(name) for name in expected}
for name, version in expected.items():
    if actual[name].split("+")[0] != version:
        raise RuntimeError(f"Unexpected {name}: {actual[name]} (expected {version})")
print(json.dumps({"status": "environment_imports_verified", "versions": actual,
    "scope": "Not a GPU optimizer/checkpoint acceptance; run the matrix gates next"}, indent=2))
PY

printf 'Isolated environment ready: %s\n' "$target_prefix"
