#!/usr/bin/env python3
"""Finish B, evaluate fixed C, then freeze a qualifying model before final test."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'results/pi05_moka_direct_20260909_b';C=ROOT/'results/pi05_moka_direct_20260909_c'
CE=ROOT/'results/pi05_moka_direct_20260909_c_validation'
FINAL=ROOT/'results/pi05_moka_direct_20260909_final_test'
STATUS=ROOT/'manifests/pi05_direct_pipeline_status.json'
def write(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):
    obj=dict(stage=stage,updated=datetime.datetime.now().astimezone().isoformat(),**kw);write(STATUS,obj);print(json.dumps(obj,ensure_ascii=False),flush=True)
def main():
    raw=subprocess.check_output(['bash','-c','source "$1"; env -0','bash',str(ROOT/'scripts/env_openpi.sh')]);env=dict(x.decode().split('=',1) for x in raw.split(b'\0') if x)
    py=str(ROOT/'envs/openpi/bin/python')
    def run(cmd,log):
        with (ROOT/'logs'/log).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
    status('waiting_for_b',final_test_rule='Select only a trained candidate with >15/20 validation successes and no seed below baseline (42:6,7:9); otherwise do not use final-test data.')
    deadline=time.monotonic()+5400
    while not (B/'summary.json').exists():
        if (B/'run_error.json').exists():raise RuntimeError('B training failed')
        if time.monotonic()>deadline:raise TimeoutError('B training timeout')
        time.sleep(10)
    report=json.loads((B/'summary.json').read_text())
    if report['status']!='complete':raise RuntimeError(f'Unexpected B status {report}')
    run([py,str(ROOT/'scripts/report_pi05_direct.py'),str(B)],'pi05_moka_direct_b_report.log')
    status('evaluating_c',b_result=report)
    run([py,str(ROOT/'scripts/evaluate_pi05_direct.py'),'--checkpoint',str(C/'policy.pt'),'--output',str(CE),'--split','validation','--initials','30-39','--seeds','42,7','--policies','candidate'],'pi05_moka_direct_c_validation.log')
    b_rows=json.loads((B/'episodes.json').read_text());c_rows=json.loads((CE/'episodes.json').read_text())
    base={(r['seed'],r['initial_state']):r for r in b_rows if r['policy']=='baseline'}
    b_selected=[r for r in b_rows if r['policy']=='candidate' and r['training_episode']==report['best_episode']]
    comparisons=[]
    for name,rows,checkpoint in [('B',b_selected,B/'best.pt'),('C',c_rows,C/'policy.pt')]:
        counts={str(seed):sum(r['success'] for r in rows if r['seed']==seed) for seed in (42,7)}
        comparisons.append(dict(name=name,checkpoint=str(checkpoint),successes=sum(counts.values()),episodes=len(rows),by_seed=counts,
            wins=[(r['seed'],r['initial_state']) for r in rows if r['success'] and not base[(r['seed'],r['initial_state'])]['success']],
            losses=[(r['seed'],r['initial_state']) for r in rows if not r['success'] and base[(r['seed'],r['initial_state'])]['success']]))
    write(ROOT/'results/pi05_direct_comparison_20260909.json',dict(baseline={'42':6,'7':9,'total':15,'episodes':20},candidates=comparisons,note='Validation/model selection only. C baseline reused from B/A, not rerun.'))
    qualifying=[x for x in comparisons if x['successes']>15 and x['by_seed']['42']>=6 and x['by_seed']['7']>=9]
    if not qualifying:
        status('no_validation_gain',comparisons=comparisons,final_test_used=False);return
    selected=max(qualifying,key=lambda x:x['successes']);checkpoint=Path(selected['checkpoint'])
    selection=dict(selected=selected,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),frozen_before_final_test=True,final_initials=list(range(40,50)),final_seeds=[123,456],selection_time=datetime.datetime.now().astimezone().isoformat())
    write(ROOT/'manifests/pi05_direct_frozen_selection.json',selection)
    status('final_test',selection=selection)
    run([py,str(ROOT/'scripts/evaluate_pi05_direct.py'),'--checkpoint',str(checkpoint),'--output',str(FINAL),'--split','final_test','--initials','40-49','--seeds','123,456','--policies','baseline,candidate'],'pi05_moka_direct_final_test.log')
    final=json.loads((FINAL/'summary.json').read_text());status('complete',comparisons=comparisons,selection=selection,final_test=final)
if __name__=='__main__':
    try:main()
    except BaseException as exc:status('failed',error=repr(exc));raise
