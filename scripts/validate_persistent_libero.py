#!/usr/bin/env python3
"""Fixed-action CPU-policy validation of persistent original LIBERO workers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/AlphaBrain-opt'))
from AlphaBrain.training.reinforcement_learning.envs import libero_env
from persistent_libero import install_persistent_env


def digest(obs):
    return {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for name, value in obs.items()}


def trajectory(env, task):
    started = time.monotonic()
    obs = env.reset('libero_goal', task, initial_state_idx=0, seed=42)
    reset_seconds = time.monotonic() - started
    records = [dict(obs=digest(obs))]
    actions = [np.array([0., 0., 0., 0., 0., 0., -1.])] * 10
    actions += [np.array([.02 if i % 2 else -.02, .01, -.01, 0., 0., 0., -1.]) for i in range(12)]
    for action in actions:
        obs, reward, done = env.step(action)
        records.append(dict(obs=digest(obs), reward=reward, done=done))
    return dict(task_id=task, initial_state=0, seed=42, reset_seconds=reset_seconds,
                wall_seconds=time.monotonic()-started, frames=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'manifests/persistent-libero-validation.json')
    args = parser.parse_args()
    libero_env._WORKER_SCRIPT = str(ROOT / 'scripts/libero_worker_fast_png.py')
    fixtures = {}
    for task in [0, 1]:
        env = libero_env.LiberoEnv()
        try:
            fixtures[task] = trajectory(env, task)
        finally:
            env.close()
    exp = SimpleNamespace(Env=libero_env.LiberoEnv, output=args.output.parent)
    close_final = install_persistent_env(exp, args.output.with_suffix('.stderr.log'))
    runs = []
    try:
        for label, task in [('first', 0), ('repeat', 0), ('switch', 1), ('switch_back', 0)]:
            env = exp.Env()
            result = trajectory(env, task)
            result.update(label=label, worker_pid=env._proc.pid,
                          pixel_state_reward_done_equal=result['frames'] == fixtures[task]['frames'])
            runs.append(result)
            env.close()
    finally:
        close_final()
    output = dict(status='passed' if all(r['pixel_state_reward_done_equal'] for r in runs) else 'failed',
                  no_vla_loaded=True, fixed_actions_only=True, no_training_or_model_selection=True,
                  same_worker_reused=len({r['worker_pid'] for r in runs}) == 1,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), ROOT/'scripts/persistent_libero.py', ROOT/'scripts/libero_worker_fast_png.py']},
                  fresh_process_fixtures=fixtures, persistent_runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(status=output['status'], same_worker_reused=output['same_worker_reused'],
                         reset_seconds={r['label']: r['reset_seconds'] for r in runs}, output=str(args.output))))
    if output['status'] != 'passed' or not output['same_worker_reused']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
