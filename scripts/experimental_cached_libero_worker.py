#!/usr/bin/env python3
"""Experimental soft-reset worker; NOT used by optimization_v2 training."""
import hashlib
import sys
import types
from pathlib import Path
from libero_worker_fast_png import ROOT, UPSTREAM_WORKER, encode_png_fast


def main():
    for source in (ROOT / 'src/AlphaBrain', ROOT / 'src/LIBERO'):
        sys.path.insert(0, str(source))
    source = UPSTREAM_WORKER.read_text()
    old = '''                if env is not None:
                    env.close()

                env_args = {
                    "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
                    "camera_heights": LIBERO_ENV_RESOLUTION,
                    "camera_widths":  LIBERO_ENV_RESOLUTION,
                }
                env = OffScreenRenderEnv(**env_args)
'''
    new = '''                key = (suite_name, task_id)
                if env is None or getattr(env, "_rlt_task_key", None) != key:
                    if env is not None:
                        env.close()
                    env_args = {
                        "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
                        "camera_heights": LIBERO_ENV_RESOLUTION,
                        "camera_widths": LIBERO_ENV_RESOLUTION,
                        "hard_reset": False,
                    }
                    env = OffScreenRenderEnv(**env_args)
                    env._rlt_task_key = key
'''
    if source.count(old) != 1:
        raise RuntimeError('Pinned reset source did not match experiment patch')
    source = source.replace(old, new)
    worker = types.ModuleType('rlt_experimental_cached_worker')
    worker.__file__ = str(UPSTREAM_WORKER)
    exec(compile(source, str(UPSTREAM_WORKER), 'exec'), worker.__dict__)
    worker._img_to_bytes = encode_png_fast
    worker.main()


if __name__ == '__main__':
    main()
