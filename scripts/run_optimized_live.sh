#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$RLT_ROOT/envs/alphabrain/bin/python" "$RLT_ROOT/scripts/compare_optimized_live.py" "$@"
