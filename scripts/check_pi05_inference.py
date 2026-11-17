#!/usr/bin/env python3
"""Smoke-check frozen pi05_libero with synthetic observations, not task evaluation."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest_path = ROOT / 'manifests/pi05_libero_download_verified.json'
    if not manifest_path.exists():
        raise RuntimeError('pi05_libero download and CRC32C verification have not completed')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete':
        raise RuntimeError('Checkpoint is not verified complete')
    for f in manifest['files']:
        if not Path(f['path']).is_file() or Path(f['path']).stat().st_size != f['size']:
            raise RuntimeError('Verified checkpoint file is missing or size changed: '+f['path'])
    import jax
    import numpy as np
    from openpi.policies import libero_policy, policy_config
    from openpi.training import config
    devices = jax.devices()
    if not any(d.platform=='gpu' for d in devices):
        raise RuntimeError('No JAX GPU device; refusing a misleading CPU timing result')
    started = time.perf_counter()
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'),ROOT/'checkpoints/pi05_libero')
    load_seconds = time.perf_counter()-started
    np.random.seed(42)
    observation = libero_policy.make_libero_example()
    observation['prompt'] = 'put the bowl on the plate'
    timings=[]
    for _ in range(3):
        start=time.perf_counter()
        result=policy.infer(observation)
        actions=np.asarray(result['actions'])
        timings.append(time.perf_counter()-start)
        if actions.shape != (10,7) or not np.isfinite(actions).all():
            raise ValueError(f'Unexpected action output: {actions.shape}, finite={np.isfinite(actions).all()}')
    report=dict(status='complete',kind='synthetic_observation_inference_smoke_only',training_updates=0,
                task_success_rate=None,checkpoint='pi05_libero',devices=[str(d) for d in devices],
                load_seconds=load_seconds,inference_seconds=timings,
                first_call_includes_compilation=True,action_shape=list(actions.shape),
                openpi_commit=subprocess.check_output(['git','-C',str(ROOT/'src/openpi'),'rev-parse','HEAD'],text=True).strip())
    path=ROOT/'results/pi05_inference_smoke_20260909'
    path.mkdir(exist_ok=True)
    (path/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
