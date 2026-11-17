#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES=0
cd "$RLT_ROOT/src/AlphaBrain"
exec "$RLT_ROOT/envs/alphabrain/bin/python" -u "$RLT_ROOT/scripts/compare_live.py" "$@"
