#!/usr/bin/env bash
# Run manually in your own SSH terminal. Do not give credentials as arguments.
set -euo pipefail
umask 077
NODE_ROOT=${NODE_ROOT:?Export NODE_ROOT to your dedicated node root}
export BAIDUPCS_GO_CONFIG_DIR="$NODE_ROOT/.private/baidu"
export BAIDUPCS_GO_VERBOSE=0
[[ -d "$BAIDUPCS_GO_CONFIG_DIR" ]] || { echo 'Run setup-baidu-cli.sh first'; exit 2; }
chmod 700 "$BAIDUPCS_GO_CONFIG_DIR"
exec "$NODE_ROOT/tools/baidupcs/v4.0.2/BaiduPCS-Go" "$@"
