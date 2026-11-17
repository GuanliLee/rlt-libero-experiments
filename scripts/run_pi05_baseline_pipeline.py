#!/usr/bin/env python3
"""Wait for the current downloads, then run GPU, policy and simulation checks."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'manifests/pi05_baseline_pipeline.json'


def status(stage, **extra):
    row = dict(stage=stage, updated=datetime.datetime.now().astimezone().isoformat(), **extra)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
    temp.replace(STATUS)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def process_running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installer-pid', type=int, required=True)
    parser.add_argument('--downloader-pid', type=int, required=True)
    args = parser.parse_args()
    lock = (ROOT / 'manifests/pi05_baseline_pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Load exactly the same local environment used for manual runs.
    raw = subprocess.check_output(['bash', '-c', 'source "$1"; env -0', 'bash', str(ROOT / 'scripts/env_openpi.sh')])
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    python = str(ROOT / 'envs/openpi/bin/python')
    marker = ROOT / 'manifests/pi05_libero_download_verified.json'
    def run(stage, command, timeout):
        log = ROOT / 'logs' / f'{stage}.log'
        status(stage, status='running', log=str(log), training_updates=0)
        with log.open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    try:
        deadline = time.monotonic() + 4 * 3600
        while process_running(args.installer_pid):
            if time.monotonic() > deadline:
                raise TimeoutError('Environment installation exceeded pipeline wait budget')
            status('waiting_for_environment', status='running', training_updates=0)
            time.sleep(30)
        run('pi05_gpu_check', [python, '-c',
            'import json,jax,numpy,torch; from openpi.policies import policy_config; '
            'd=jax.devices(); assert any(x.platform=="gpu" for x in d),d; '
            'x=jax.numpy.ones((64,64)); y=(x@x).block_until_ready(); '
            'assert float(y[0,0])==64.; '
            'print(json.dumps(dict(jax=jax.__version__,torch=torch.__version__,devices=[str(x) for x in d],gpu_matmul="passed")))'], 600)
        while not marker.exists():
            if not process_running(args.downloader_pid):
                raise RuntimeError('Checkpoint downloader exited without verified completion marker')
            if time.monotonic() > deadline:
                raise TimeoutError('Checkpoint download exceeded pipeline wait budget')
            present = sum(p.stat().st_size for p in (ROOT / 'checkpoints/pi05_libero').rglob('*') if p.is_file())
            status('waiting_for_checkpoint', status='running', downloaded_bytes=present, training_updates=0)
            time.sleep(30)
        if json.loads(marker.read_text()).get('status') != 'complete':
            raise RuntimeError('Invalid checkpoint completion marker')
        run('pi05_inference_smoke', [python, str(ROOT / 'scripts/check_pi05_inference.py')], 1800)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        evaluator = str(ROOT / 'scripts/evaluate_pi05_baseline.py')
        probe = ROOT / 'results' / f'pi05_libero_probe_{stamp}'
        run('pi05_simulation_probe', [python, evaluator, '--output', str(probe), '--split', 'training_probe',
                                   '--tasks', '7', '--initials', '0', '--save-videos'], 1800)
        validation = ROOT / 'results' / f'pi05_libero_validation_{stamp}'
        run('pi05_baseline_validation', [python, evaluator, '--output', str(validation), '--split', 'validation',
                                      '--tasks', '0-9', '--initials', '31-33', '--save-videos'], 7200)
        summary = json.loads((validation / 'summary.json').read_text())
        status('complete', status='complete', validation=str(validation), probe=str(probe),
               counts=summary['counts'], training_updates=0,
               note='Frozen pi05 baseline only. Pi RLT training has not started. Test initials 40-49 untouched.')
    except BaseException as exc:
        status('failed', status='failed', error=repr(exc), training_updates=0)
        raise


if __name__ == '__main__':
    main()
