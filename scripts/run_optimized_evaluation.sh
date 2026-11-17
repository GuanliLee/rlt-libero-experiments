#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export RLT_SOURCE_ROOT="$RLT_ROOT/src/AlphaBrain-opt"
export PYTHONPATH="$RLT_SOURCE_ROOT:$RLT_ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$RLT_SOURCE_ROOT"
exec "$RLT_ROOT/envs/alphabrain/bin/python" -u "$RLT_ROOT/scripts/evaluate_optimized_policy.py" "$@"
