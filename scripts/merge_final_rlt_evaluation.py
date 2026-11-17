#!/usr/bin/env python3
"""Merge two predeclared test shards; reject duplicates and incomplete cohorts."""
import argparse
import json
from pathlib import Path
from optimize_rlt import write_json


def merge(output):
    cohort=json.loads((output/'cohort.json').read_text())
    if cohort['tasks']!=list(range(10)) or cohort['initial_states']!=list(range(40,50)) or cohort['policies']!=['baseline','release','candidate']:
        raise RuntimeError('Unexpected global test cohort')
    episodes=[];protocols=[];summaries=[]
    for name,tasks in [('first_half',list(range(5))),('second_half',list(range(5,10)))]:
        directory=output/name
        summary=json.loads((directory/'run_summary.json').read_text())
        if summary.get('status')!='complete' or not summary.get('replay_unchanged') or not summary.get('candidate_parameters_unchanged') or summary.get('training_updates')!=0:
            raise RuntimeError('A test shard is incomplete or changed training state')
        protocol=json.loads((directory/'protocol.json').read_text())
        if protocol['tasks']!=tasks or protocol['initial_states']!=cohort['initial_states'] or protocol['policies']!=cohort['policies'] or protocol['split']!='test':
            raise RuntimeError('Shard does not match declared task/state/policy partition')
        if protocol['run']!=cohort['run'] or protocol['alpha']!=cohort['alpha'] or protocol['candidate_weights_sha256']['actor.pt']!=cohort['actor_sha256']:
            raise RuntimeError('Shard candidate differs from predeclared selection')
        local=json.loads((directory/'episodes.json').read_text())
        for e in local:
            if e['task_id'] not in tasks or e['initial_state'] not in cohort['initial_states'] or e['seed']!=42:
                raise RuntimeError('Episode outside shard cohort')
            if 'video' in e:
                p=directory/e['video']
                if not p.is_file():raise RuntimeError('Missing evaluation video')
                e['video']=str(p.relative_to(output))
            episodes.append(e)
        protocols.append(protocol);summaries.append(summary)
    for key in ['candidate_weights_sha256','training_source_sha256','evaluation_source_sha256','environment','alpha','delta_max','published_actor_sha256','original_model_manifests']:
        if protocols[0][key]!=protocols[1][key]:raise RuntimeError('Shard identity/config differs: '+key)
    expected={(t,i,p) for t in cohort['tasks'] for i in cohort['initial_states'] for p in cohort['policies']}
    observed=[(e['task_id'],e['initial_state'],e['policy']) for e in episodes]
    if len(observed)!=300 or len(set(observed))!=300 or set(observed)!=expected:
        raise RuntimeError('Final cohort has duplicates or missing episodes')
    episodes.sort(key=lambda e:(e['task_id'],e['initial_state'],cohort['policies'].index(e['policy'])))
    counts={p:dict(successes=sum(e['success'] for e in episodes if e['policy']==p),episodes=100) for p in cohort['policies']}
    per_task={t:{p:dict(successes=sum(e['success'] for e in episodes if e['policy']==p and e['task_id']==t),episodes=10) for p in cohort['policies']} for t in cohort['tasks']}
    pairs=[]
    for task in cohort['tasks']:
        for initial in cohort['initial_states']:
            pair={e['policy']:e for e in episodes if e['task_id']==task and e['initial_state']==initial};b,c=pair['baseline'],pair['candidate']
            pairs.append(dict(task_id=task,initial_state=initial,baseline=b['success'],candidate=c['success'],candidate_only=c['success'] and not b['success'],baseline_only=b['success'] and not c['success'],step_delta_if_both_success=c['action_steps']-b['action_steps'] if b['success'] and c['success'] else None))
    protocol=protocols[0].copy();protocol.update(tasks=cohort['tasks'],task_shard=None,shards=['first_half','second_half'],primary_policy=cohort['primary_policy'],selection_manifest=cohort['selection_manifest'])
    write_json(output/'protocol.json',protocol)
    write_json(output/'episodes.json',episodes)
    write_json(output/'summary.json',dict(split='test',tasks=cohort['tasks'],initial_states=cohort['initial_states'],counts=counts,per_task=per_task,macro_success_rate={p:counts[p]['successes']/100 for p in counts},paired=pairs,episodes=episodes,protocol='protocol.json'))
    write_json(output/'run_summary.json',dict(status='complete',mode='fixed_100_pair_test_three_policies',training_updates=0,replay_unchanged=True,candidate_parameters_unchanged=True,shards=2,episodes=300,counts=counts,primary_policy=cohort['primary_policy'],peak_gpu_allocated_gib_per_process=max(s['peak_gpu_allocated_gib'] for s in summaries)))
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    merge(parser.parse_args().output.resolve())
