#!/usr/bin/env python3
"""Fresh environment per episode with one model build, guarded by equivalence tests."""
import sys
import types
from libero_worker_fast_png import ROOT, UPSTREAM_WORKER, encode_png_fast


def main():
    for source in (ROOT / 'src/AlphaBrain', ROOT / 'src/LIBERO'):
        sys.path.insert(0, str(source))
    source = UPSTREAM_WORKER.read_text()
    old = '                    "camera_widths":  LIBERO_ENV_RESOLUTION,\n'
    if source.count(old) != 1:
        raise RuntimeError('Pinned environment kwargs changed')
    source = source.replace(old, old+'                    "hard_reset": False,\n')
    worker = types.ModuleType('rlt_single_build_worker')
    worker.__file__ = str(UPSTREAM_WORKER)
    exec(compile(source, str(UPSTREAM_WORKER), 'exec'), worker.__dict__)
    worker._img_to_bytes = encode_png_fast
    worker.main()


if __name__ == '__main__':
    main()
