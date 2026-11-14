#!/usr/bin/env bash
# Run on the user's local PC (same Wi-Fi as the board). Relays a bundle from the GPU server to the
# board, runs the benchmark there, and brings the results back to the server:
#
#   server:<repo>/artifacts/bundles/<name>.tar.gz -> local -> board:<workdir>/<name>/
#   board: cd <board project> && uv run python <workdir>/<name>/bench/run_all.py
#   board:<workdir>/<name>/out/<stamp> -> local -> server:<repo>/incoming/<name>_results.tar.gz
#
# Usage:
#   SERVER=user@gpu-server SERVER_REPO=/path/to/Recurrent_JSCC_NPU \
#   BOARD_PROJECT=/home/ubuntu/recurrent-npu-jscc tools/relay.sh <bundle name> [run_all.py args...]
#
# Optional: BOARD (default ubuntu@192.168.1.9), BOARD_WORKDIR (default jscc_bench, relative to the
# board home directory). Afterwards, on the server: uv run python tools/collect_npu.py <bundle name>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,14p' "$0"
    exit 2
fi
name=$1
shift
: "${SERVER:?set SERVER to the ssh destination of the GPU server}"
: "${SERVER_REPO:?set SERVER_REPO to the repository path on the GPU server}"
: "${BOARD_PROJECT:?set BOARD_PROJECT to the recurrent-npu-jscc uv project path on the board}"
board=${BOARD:-ubuntu@192.168.1.9}
workdir=${BOARD_WORKDIR:-jscc_bench}
stamp=$(date +%Y%m%d_%H%M%S)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "[relay] server -> local: $name.tar.gz"
scp "$SERVER:$SERVER_REPO/artifacts/bundles/$name.tar.gz" "$tmp/$name.tar.gz"

echo "[relay] local -> board: $workdir/$name"
ssh "$board" "mkdir -p '$workdir' && rm -rf '$workdir/$name.new' && mkdir '$workdir/$name.new'"
ssh "$board" "tar xzf - -C '$workdir/$name.new'" <"$tmp/$name.tar.gz"
# Keep earlier results (out/) when the same bundle name is sent again.
ssh "$board" "cd '$workdir' && if [ -d '$name/out' ]; then mv '$name/out' '$name.new/$name/out'; fi \
    && rm -rf '$name' && mv '$name.new/$name' '$name' && rmdir '$name.new'"

echo "[relay] board: running bench/run_all.py (out/$stamp)"
status=0
ssh "$board" "bundle=\$(cd '$workdir/$name' && pwd) && cd '$BOARD_PROJECT' \
    && uv run python \"\$bundle/bench/run_all.py\" --out \"\$bundle/out/$stamp\" $*" || status=$?
echo "[relay] run_all.py exit status: $status"

echo "[relay] board -> local -> server: results (context binaries stay on the board)"
ssh "$board" "tar czf - --exclude='*.bin' -C '$workdir/$name/out' '$stamp'" >"$tmp/${name}_results.tar.gz"
ssh "$SERVER" "mkdir -p '$SERVER_REPO/incoming'"
scp "$tmp/${name}_results.tar.gz" "$SERVER:$SERVER_REPO/incoming/${name}_results.tar.gz"
echo "[relay] done. On the server: uv run python tools/collect_npu.py $name"
exit "$status"
