#!/usr/bin/env python3
"""Pinned simulator with OpenPI's quaternion convention and lossless transport."""
import importlib.util
import math
import sys
import numpy as np
from libero_worker_fast_png import ROOT, UPSTREAM_WORKER, encode_png_fast


def quat2axisangle(quat):
    # Match openpi/examples/libero/main.py, including rotations with negative w.
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return np.asarray(quat[:3]) * (2.0 * math.acos(w) / den)


def main():
    for source in (ROOT / 'src/AlphaBrain', ROOT / 'src/LIBERO'):
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('pi_pinned_libero_worker', UPSTREAM_WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker._img_to_bytes = encode_png_fast
    worker._quat2axisangle = quat2axisangle
    worker.main()


if __name__ == '__main__':
    main()
