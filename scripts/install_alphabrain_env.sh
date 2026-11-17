#!/usr/bin/env bash
set -euo pipefail
RLT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UV_CACHE_DIR="$RLT_ROOT/cache/uv"
if [[ ! -x "$RLT_ROOT/envs/alphabrain/bin/python" ]]; then
    uv venv --python "$RLT_ROOT/envs/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11" \
        "$RLT_ROOT/envs/alphabrain"
fi
RLT_TORCH_WHEEL="$RLT_ROOT/cache/wheels/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
if [[ ! -f "$RLT_TORCH_WHEEL" || -f "$RLT_TORCH_WHEEL.aria2" ]]; then
    RLT_TORCH_WHEEL='https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
fi
# The session's HTTP proxy stalled wheel requests. Direct official indexes
# worked; this affects only these installation commands, not shell settings.
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    uv pip install --python "$RLT_ROOT/envs/alphabrain/bin/python" \
    "$RLT_TORCH_WHEEL" \
    'https://download.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl' \
    'numpy==1.26.4' \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    uv pip install --python "$RLT_ROOT/envs/alphabrain/bin/python" \
    -r "$RLT_ROOT/manifests/alphabrain-inference-requirements.in" \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install --python "$RLT_ROOT/envs/alphabrain/bin/python" \
    --no-deps --no-build-isolation -e "$RLT_ROOT/src/AlphaBrain"
uv pip freeze --python "$RLT_ROOT/envs/alphabrain/bin/python" \
    > "$RLT_ROOT/manifests/alphabrain-env-freeze.txt"
