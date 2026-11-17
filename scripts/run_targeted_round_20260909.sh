#!/usr/bin/env bash
set -euo pipefail
source /opt/shared_assets/8t/lgl/RLT/scripts/env.sh
export RLT_SOURCE_ROOT="$RLT_ROOT/src/AlphaBrain-opt"
export PYTHONPATH="$RLT_SOURCE_ROOT:$RLT_ROOT/src/LIBERO"
trap 'echo "Targeted experiment stopped; inspect logs/targeted_rlt_20260909_a*.log" >&2' ERR
bash "$RLT_ROOT/scripts/run_optimization_v2.sh" \
  --output "$RLT_ROOT/results/targeted_rlt_20260909_a" \
  --tasks 0,5,6 --train-initials 15-29 \
  --warmup-episodes 5 --train-episodes 30 --batch-size 120 \
  --val-initials 31-33 --save-videos \
  > "$RLT_ROOT/logs/targeted_rlt_20260909_a_training.log" 2>&1
echo 'Targeted training completed; starting all-task frozen validation.'
"$RLT_ROOT/envs/alphabrain/bin/python" -u "$RLT_ROOT/scripts/evaluate_rlt_blends.py" \
  --run "$RLT_ROOT/results/targeted_rlt_20260909_a" \
  --output "$RLT_ROOT/results/targeted_rlt_20260909_a_validation" \
  --split validation --tasks 0-9 --initials 31-33 --alphas 1 \
  --policies baseline,release,bc_only,candidate --save-videos \
  > "$RLT_ROOT/logs/targeted_rlt_20260909_a_validation.log" 2>&1
"$RLT_ROOT/envs/libero/bin/python" "$RLT_ROOT/scripts/build_optimization_viewer.py" \
  --evaluation "$RLT_ROOT/results/targeted_rlt_20260909_a_validation" --candidate-policy candidate_alpha1
echo 'Targeted experiment and all-task validation completed.'
