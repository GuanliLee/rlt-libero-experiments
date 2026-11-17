#!/usr/bin/env python3
"""Confirm the frozen selected model on predeclared seed7 after training ends."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--trainer-pid',type=int,required=True);a=p.parse_args()
    train=ROOT/'results/pi05_moka_rlt_20260909_a';out=ROOT/'results/pi05_moka_confirmation_20260909_a'
    state=ROOT/'manifests/pi05_moka_confirmation_status.json'
    def status(stage,**kw):
        row=dict(stage=stage,updated=datetime.datetime.now().astimezone().isoformat(),**kw)
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');tmp.replace(state)
    # Declare this before seeing any seed7 outcomes; do not tune on confirmation.
    status('waiting_for_training',seed=7,initials=list(range(30,40)),scales=[0,1],test_initials_used=False)
    try:
        deadline=time.monotonic()+7200
        while Path(f'/proc/{a.trainer_pid}').exists():
            if time.monotonic()>deadline:raise TimeoutError('Training wait expired')
            time.sleep(10)
        report=json.loads((train/'summary.json').read_text())
        if report['status']!='complete':raise RuntimeError('Training did not finish')
        if report['best_validation_successes']<=report['baseline_successes']:
            status('no_improved_candidate',report=report);return
        raw=subprocess.check_output(['bash','-c','source "$1"; env -0','bash',str(ROOT/'scripts/env_openpi.sh')])
        env=dict(x.decode().split('=',1) for x in raw.split(b'\0') if x)
        status('confirming_frozen_model',seed=7,best_episode=report['best_episode'],test_initials_used=False)
        cmd=[str(ROOT/'envs/openpi/bin/python'),str(ROOT/'scripts/evaluate_pi05_moka_variant.py'),
            '--checkpoint',str(train/'best.pt'),'--output',str(out),'--scales','0,1','--seeds','7',
            '--initials','30-39','--split','validation','--purpose','confirmation']
        with (ROOT/'logs/pi05_moka_confirmation.log').open('w') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
        confirmation=json.loads((out/'summary.json').read_text());status('complete',result=str(out),comparisons=confirmation['comparisons'])
    except BaseException as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
