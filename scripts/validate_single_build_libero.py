#!/usr/bin/env python3
"""Validate fresh single-build reset for all 10 tasks with nonzero actions."""
import hashlib
import json
from types import SimpleNamespace
from validate_persistent_libero import ROOT, libero_env, trajectory
from persistent_libero import install_persistent_env


def main():
    output = ROOT / 'manifests/single-build-libero-validation.json'
    trajectories = {}
    for name, script in [('original','libero_worker_fast_png.py'),('single_build','libero_worker_single_build.py')]:
        libero_env._WORKER_SCRIPT = str(ROOT / 'scripts' / script)
        exp = SimpleNamespace(Env=libero_env.LiberoEnv, output=output.parent)
        close = install_persistent_env(exp, output.with_suffix('.'+name+'.stderr.log'))
        try:
            trajectories[name] = [trajectory(exp.Env(), task) for task in range(10)]
        finally:
            close()
    matches = [a['frames']==b['frames'] for a,b in zip(trajectories['original'],trajectories['single_build'])]
    report = dict(status='passed' if all(matches) else 'failed',no_vla_loaded=True,production_enabled=False,
                  equality_by_task=matches, trajectories=trajectories,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/libero_worker_single_build.py',ROOT/'scripts/validate_single_build_libero.py']})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'equality_by_task':matches,'reset_seconds':{k:[r['reset_seconds'] for r in v] for k,v in trajectories.items()}}))
    if not all(matches): raise SystemExit(1)


if __name__ == '__main__':
    main()
