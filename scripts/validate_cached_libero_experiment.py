#!/usr/bin/env python3
"""Compare experimental cached reset with original physics and rendered pixels."""
import hashlib
import json
from types import SimpleNamespace
from validate_persistent_libero import ROOT, libero_env, trajectory
from persistent_libero import install_persistent_env


def main():
    output = ROOT / 'manifests/experimental-cached-libero-validation.json'
    tasks = [0, 1, 3, 6]
    libero_env._WORKER_SCRIPT = str(ROOT / 'scripts/libero_worker_fast_png.py')
    fixtures = {}
    for task in tasks:
        env = libero_env.LiberoEnv()
        try:
            fixtures[task] = trajectory(env, task)
        finally:
            env.close()
    libero_env._WORKER_SCRIPT = str(ROOT / 'scripts/experimental_cached_libero_worker.py')
    exp = SimpleNamespace(Env=libero_env.LiberoEnv, output=output.parent)
    close = install_persistent_env(exp, output.with_suffix('.stderr.log'))
    records = []
    try:
        for task in [0,0,1,1,3,3,6,6,0]:
            record = trajectory(exp.Env(), task)
            record['pixel_state_reward_done_equal'] = record['frames'] == fixtures[task]['frames']
            records.append(record)
    finally:
        close()
    report = dict(status='passed' if all(r['pixel_state_reward_done_equal'] for r in records) else 'failed',
                  production_enabled=False, no_vla_loaded=True, tasks=tasks, fixtures=fixtures, runs=records,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/experimental_cached_libero_worker.py',ROOT/'scripts/validate_cached_libero_experiment.py']})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'], 'runs':[{k:r[k] for k in ['task_id','reset_seconds','pixel_state_reward_done_equal']} for r in records]}))


if __name__ == '__main__':
    main()
