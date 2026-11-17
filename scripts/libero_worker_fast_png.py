#!/usr/bin/env python3
"""Run the pinned LIBERO worker with faster, lossless PNG transport."""

import importlib.util
import io
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_WORKER = (
    ROOT
    / "src/AlphaBrain/AlphaBrain/training/reinforcement_learning/envs/libero_env_worker.py"
)


def encode_png_fast(array):
    """Keep the original RGB pixels and PNG wire format; reduce compression work."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def main():
    # The simulator remains in its separate Python environment. These paths
    # expose the same pinned source packages used by the original worker.
    for source in (ROOT / "src/AlphaBrain", ROOT / "src/LIBERO"):
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("rlt_pinned_libero_worker", UPSTREAM_WORKER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the upstream LIBERO worker: {UPSTREAM_WORKER}")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker._img_to_bytes = encode_png_fast
    worker.main()


if __name__ == "__main__":
    main()
