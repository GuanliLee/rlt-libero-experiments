#!/usr/bin/env python3
"""Compare full frozen VLA episodes under original vs single-build reset."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from evaluate_optimized_policy import FrozenEvaluation, sha256
from optimize_rlt import ROOT, write_json


def main():
    run = ROOT/'results/optimization_pilot_20260908_beta3'
    output = ROOT/'results/single_build_policy_equivalence_20260908'
    output.mkdir()
    config = json.loads((run/'config.json').read_text())
    os.environ['RLT_SOURCE_ROOT'] = config['source_root']
    for name,digest in json.loads((run/'code_sha256.json').read_text()).items():
        if sha256(Path(name)) != digest: raise RuntimeError('Parent source changed')
    config.update(output=run,resume=True)
    exp = FrozenEvaluation(argparse.Namespace(**config), output)
    import numpy as np
    import torch
    from AlphaBrain.training.reinforcement_learning.envs import libero_env
    records = []
    for task,initial in [(0,31),(3,31),(6,30)]:
        for variant,worker in [('original','libero_worker_fast_png.py'),('single_build','libero_worker_single_build.py')]:
            libero_env._WORKER_SCRIPT = str(ROOT/'scripts'/worker)
            action_hash, obs_hash = hashlib.sha256(), hashlib.sha256()
            class TraceEnv:
                def __init__(self): self.env = libero_env.LiberoEnv()
                def record(self,obs):
                    for key in sorted(obs): obs_hash.update(np.asarray(obs[key]).tobytes())
                    return obs
                def reset(self,*args,**kwargs): return self.record(self.env.reset(*args,**kwargs))
                def step(self,action):
                    action_hash.update(np.asarray(action,dtype=np.float64).tobytes())
                    obs,reward,done=self.env.step(action)
                    return self.record(obs),reward,done
                def close(self): self.env.close()
                def __getattr__(self,key): return getattr(self.env,key)
            exp.Env=TraceEnv
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            record=exp.episode(task,initial,'baseline')
            record.update(variant=variant,action_sha256=action_hash.hexdigest(),observation_sha256=obs_hash.hexdigest())
            records.append(record)
            write_json(output/'episodes.json',records)
    keys=['success','action_steps','chunks','reward','action_sha256','observation_sha256']
    matches=[all(a[k]==b[k] for k in keys) for a,b in zip(records[::2],records[1::2])]
    report=dict(status='passed' if all(matches) else 'failed',paired_equal=matches,episodes=records,
                source_sha256={str(p):sha256(p) for p in [Path(__file__),ROOT/'scripts/libero_worker_single_build.py']},
                no_training=True,all_observations_and_executed_actions_hashed=True)
    write_json(output/'run_summary.json',report)
    write_json(ROOT/'manifests/single-build-policy-equivalence.json',report)
    if not all(matches):raise SystemExit(1)


if __name__=='__main__':main()
