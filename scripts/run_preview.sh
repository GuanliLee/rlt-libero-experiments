#!/usr/bin/env bash
# Usage: bash scripts/run_preview.sh [task_ids=0] [episodes_per_task=1] [gpu=0]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
task_ids="${1:-0}"
episodes="${2:-1}"
gpu_id="${3:-0}"
run_dir="$RLT_ROOT/results/preview_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run_dir/videos"
cd "$RLT_ROOT/src/AlphaBrain"
printf 'Results and videos: %s\n' "$run_dir"
"$RLT_ROOT/envs/alphabrain/bin/python" -u "$RLT_ROOT/scripts/eval_rlt_a_sdpa.py" \
  --vla_ckpt "$RLT_ROOT/checkpoints/qwenoft_5traj" \
  --action_token_ckpt "$RLT_ROOT/checkpoints/rlt_a/rl_offpolicy_iter_00400" \
  --suite libero_goal --task_ids "$task_ids" --n_eps_per_task "$episodes" \
  --gpu "$gpu_id" --num_workers 1 \
  --bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4 \
  --actor_hidden_dim 512 --ref_dropout 0.5 --fixed_std 0.1 --prop_dim 8 \
  --video_dir "$run_dir/videos" --results_json "$run_dir/summary.json" \
  2>&1 | tee "$run_dir/run.log"

