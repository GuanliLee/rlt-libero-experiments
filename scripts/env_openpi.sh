#!/usr/bin/env bash
export RLT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENPI_DATA_HOME="$RLT_ROOT/cache/openpi"
export HF_HOME="$RLT_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$RLT_ROOT/cache"
export JAX_COMPILATION_CACHE_DIR="$RLT_ROOT/cache/jax"
export UV_CACHE_DIR="$RLT_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$RLT_ROOT/envs/openpi"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
if [ -x "$RLT_ROOT/envs/openpi/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas" ]; then
    export PATH="$RLT_ROOT/envs/openpi/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH"
fi
