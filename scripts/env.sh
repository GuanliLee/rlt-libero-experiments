#!/usr/bin/env bash
# Source this file to use the isolated RLT inference installation.
RLT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RLT_ROOT
export HF_HOME="$RLT_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$RLT_ROOT/cache"
export UV_CACHE_DIR="$RLT_ROOT/cache/uv"
export TORCH_HOME="$RLT_ROOT/cache/torch"
export NUMBA_CACHE_DIR="$RLT_ROOT/cache/numba"
export MPLCONFIGDIR="$RLT_ROOT/cache/matplotlib"
export LIBERO_HOME="$RLT_ROOT/src/LIBERO"
export LIBERO_PYTHON="$RLT_ROOT/envs/libero/bin/python"
export LIBERO_CONFIG_PATH="$RLT_ROOT/config/libero"
export PYTHONPATH="$RLT_ROOT/src/AlphaBrain-opt:$RLT_ROOT/src/LIBERO"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
