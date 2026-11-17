#!/usr/bin/env python3
"""Finish data collection, fit encoder, then run the bounded RLT experiment."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'manifests/pi05_moka_optimization_status.json'


def status(stage,**kwargs):
    row=dict(stage=stage,updated=datetime.datetime.now().astimezone().isoformat(),**kwargs)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');tmp.replace(STATUS)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--collector-pid',type=int,required=True);a=p.parse_args()
    lock=(ROOT/'manifests/pi05_moka_optimization.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=ROOT/'results/pi05_moka_training_data_20260909';enc=ROOT/'results/pi05_moka_encoder_20260909';out=ROOT/'results/pi05_moka_rlt_20260909_a'
    raw=subprocess.check_output(['bash','-c','source "$1"; env -0','bash',str(ROOT/'scripts/env_openpi.sh')])
    env=dict(x.decode().split('=',1) for x in raw.split(b'\0') if x)
    py=str(ROOT/'envs/openpi/bin/python')
    def run(stage,command,timeout):
        log=ROOT/'logs'/f'{stage}.log';status(stage,status='running',log=str(log))
        with log.open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    try:
        deadline=time.monotonic()+3600
        while not (data/'summary.json').exists():
            if not Path(f'/proc/{a.collector_pid}').exists():raise RuntimeError('Collector exited without complete summary')
            if time.monotonic()>deadline:raise TimeoutError('Collection timeout')
            status('collect_training_baseline',status='running');time.sleep(15)
        if json.loads((data/'summary.json').read_text())['status']!='complete':raise RuntimeError('Collection incomplete')
        run('pi05_moka_network_tests',[py,str(ROOT/'scripts/test_pi05_moka_rlt.py')],300)
        run('pi05_moka_encoder_fit',[py,str(ROOT/'scripts/fit_pi05_moka_encoder.py'),'--data',str(data),'--output',str(enc)],3600)
        run('pi05_moka_rlt_train',[py,str(ROOT/'scripts/train_pi05_moka_rlt.py'),'--data',str(data),'--encoder',str(enc/'encoder.pt'),'--output',str(out),'--episodes','60'],7200)
        report=json.loads((out/'summary.json').read_text());status('complete',status='complete',result=str(out),report=report)
    except BaseException as exc:
        status('failed',status='failed',error=repr(exc));raise


if __name__=='__main__':main()
