#!/usr/bin/env bash
# Third-party CLI only. Never logs in or reads account configuration.
set -euo pipefail
umask 077
NODE_ROOT=${1:?Usage: bash scripts/setup-baidu-cli.sh /absolute/node/root}
[[ "$NODE_ROOT" = /* && "$NODE_ROOT" != / ]] || { echo 'Use a dedicated absolute node root'; exit 2; }
version=v4.0.2
case "$(uname -m)" in x86_64) arch=amd64;; aarch64) arch=arm64;; *) echo 'Unsupported architecture'; exit 2;; esac
for utility in curl unzip install; do command -v "$utility" >/dev/null; done
install -d -m 700 "$NODE_ROOT/.private/baidu" "$NODE_ROOT/tools/baidupcs/$version"
archive="$NODE_ROOT/tools/baidupcs/BaiduPCS-Go-$version-linux-$arch.zip"
url="https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/$version/BaiduPCS-Go-$version-linux-$arch.zip"
if ! unzip -t "$archive" >/dev/null 2>&1; then
  curl -4 -fsSL --connect-timeout 10 --max-time 900 --retry 2 -C - "$url" -o "$archive.part"
  unzip -t "$archive.part" >/dev/null
  mv "$archive.part" "$archive"
fi
if [[ ! -x "$NODE_ROOT/tools/baidupcs/$version/BaiduPCS-Go" ]]; then
  member=$(unzip -Z1 "$archive" | grep -E '(^|/)BaiduPCS-Go$')
  [[ $(printf '%s\n' "$member" | wc -l) = 1 ]] || { echo 'Ambiguous executable in release'; exit 2; }
  unzip -p "$archive" "$member" > "$NODE_ROOT/tools/baidupcs/$version/BaiduPCS-Go"
  chmod 700 "$NODE_ROOT/tools/baidupcs/$version/BaiduPCS-Go"
fi
export BAIDUPCS_GO_CONFIG_DIR="$NODE_ROOT/.private/baidu"
export BAIDUPCS_GO_VERBOSE=0
"$NODE_ROOT/tools/baidupcs/$version/BaiduPCS-Go" --version
printf '%s\n' 'Installed third-party CLI; authentication and downloading are NOT verified.'
