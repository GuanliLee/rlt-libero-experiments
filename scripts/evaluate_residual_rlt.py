#!/usr/bin/env python3
"""Frozen evaluation of the custom bounded-residual RLT architecture."""
import argparse
import json
import os
from pathlib import Path
import random
import time
from evaluate_optimized_policy import FrozenEvaluation, sha256
from optimize_rlt import ROOT, indices, write_json
from persistent_libero import install_persistent_env


def wrap_residual(raw_actor, delta_max):
    import torch
    from residual_actor import residual_action
    class FrozenResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();self.raw_actor=raw_actor
        def forward(self,token,reference,prop=None,deterministic=True):
            if not deterministic:raise RuntimeError('Frozen residual evaluation is deterministic')
            return residual_action(self.raw_actor,token,reference,prop,delta_max=delta_max),None
    return FrozenResidual().eval().requires_grad_(False)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--initials',type=indices)
    p.add_argument('--tasks',type=indices)
    p.add_argument('--alpha',type=float,default=1.)
    p.add_argument('--save-videos',action='store_true')
    args=p.parse_args();args.run=args.run.resolve();args.output=args.output.resolve()
    if args.output.exists() or args.run in args.output.parents:p.error('Use a new sibling output directory')
    if not 0<args.alpha<=1:p.error('alpha must be in (0,1]')
    config=json.loads((args.run/'config.json').read_text())
    summary=json.loads((args.run/'run_summary.json').read_text())
    if summary.get('status')!='complete' or config.get('runner_version')!=4 or config.get('architecture')!='bounded_residual':
        p.error('A completed, explicitly labeled bounded-residual v4 run is required')
    args.tasks=args.tasks or config['tasks']
    args.initials=args.initials or list(range(31,34) if args.split=='validation' else range(40,50))
    if not set(args.tasks)<=set(range(10)):p.error('Tasks must be in 0-9')
    allowed=range(30,40) if args.split=='validation' else range(40,50)
    if not set(args.initials)<=set(allowed):p.error('Initial states outside the declared split')
    if args.split=='test' and (args.tasks!=list(range(10)) or args.initials!=list(range(40,50))):
        p.error('Final test is fixed to all 10 tasks and all 10 reserved initial states')
    return args,config


def main():
    args,config=arguments()
    os.environ['RLT_SOURCE_ROOT']=config['source_root']
    codes=json.loads((args.run/'code_sha256.json').read_text())
    for name,digest in codes.items():
        if sha256(Path(name))!=digest:raise RuntimeError('Training source changed: '+name)
    identities={name:sha256(args.run/name) for name in ['checkpoint.pt','actor.pt','bc_only_actor.pt','identity_control_audit.json']}
    args.output.mkdir()
    protocol=dict(mode='frozen_custom_residual_RLT',architecture='bounded_residual',not_original_RLT_a=True,
        run=str(args.run),split=args.split,tasks=args.tasks,initial_states=args.initials,
        policies=['baseline','candidate'],seed=42,action_horizon=320,chunk_length=8,warmup_steps=10,
        delta_max=config['delta_max'],alpha=args.alpha,effective_delta_max=config['delta_max']*args.alpha,
        action='clamp(VLA_reference + effective_delta_max*tanh(raw_residual_network),-1,1)',
        environment='Original fresh double-build environment, seed/state loading and lossless PNG; persistent Python worker only',
        single_build_environment=False,training_updates=0,collect_into_replay=False,
        bc_control='Structural identity BC verified to equal the baseline for every finite input; no duplicate rollout needed',
        split_limitation='Held out only from this local continuation; published weights may have seen these states',
        candidate_weights_sha256=identities,training_source_sha256=codes,
        evaluation_source_sha256={str(p):sha256(p) for p in [Path(__file__),ROOT/'scripts/residual_actor.py',ROOT/'scripts/persistent_libero.py',ROOT/'scripts/evaluate_optimized_policy.py']})
    write_json(args.output/'protocol.json',protocol)
    close=None;started=time.monotonic()
    try:
        import numpy as np
        import torch
        payload=torch.load(args.run/'checkpoint.pt',map_location='cpu',weights_only=False)
        parent=json.loads((args.run/'parent_identity.json').read_text())
        if payload.get('v4_code_sha256')!=codes or payload.get('architecture')!='bounded_residual' or payload.get('delta_max')!=config['delta_max']:
            raise RuntimeError('Residual checkpoint architecture/source metadata mismatch')
        if not payload.get('parent_checkpoint_sha256')==config['parent_checkpoint_sha256']==parent['files']['checkpoint.pt']:
            raise RuntimeError('Residual parent identity mismatch')
        if payload['progress']['actor_steps']!=payload['progress']['bc_control_steps']:
            raise RuntimeError('Residual and identity control update counts differ')
        supplied=torch.load(args.run/'actor.pt',map_location='cpu',weights_only=True)
        supplied_bc=torch.load(args.run/'bc_only_actor.pt',map_location='cpu',weights_only=True)
        def equal(a,b):return set(a)==set(b) and all(torch.equal(v.cpu(),b[k].cpu()) for k,v in a.items())
        if not equal(payload['actor'],supplied) or not equal(payload['bc_actor'],supplied_bc):raise RuntimeError('Actor/control file differs from checkpoint')
        if torch.count_nonzero(supplied_bc['net.4.weight']) or torch.count_nonzero(supplied_bc['net.4.bias']):raise RuntimeError('Identity BC head is not zero')
        del payload
        load_config=config.copy();load_config.update(output=args.run,resume=True)
        exp=FrozenEvaluation(argparse.Namespace(**load_config),args.output)
        raw=exp.actor
        if not equal(raw.state_dict(),supplied):raise RuntimeError('Loaded raw residual parameters differ')
        if not equal(exp.bc_actor.state_dict(),supplied_bc):raise RuntimeError('Loaded identity control differs')
        exp.actor=wrap_residual(raw,config['delta_max']*args.alpha)
        exp.bc_actor=wrap_residual(exp.bc_actor,config['delta_max']*args.alpha)
        if any(p.requires_grad for model in [exp.vla,exp.encoder,exp.actor,exp.bc_actor] for p in model.parameters()):raise RuntimeError('Evaluation parameters are not frozen')
        before=dict(progress=exp.progress.copy(),replay_size=len(exp.buffer),position=exp.buffer.pos)
        close=install_persistent_env(exp)
        episodes=[]
        for task in args.tasks:
            for initial in args.initials:
                for mode in ['baseline','candidate']:
                    random.seed(42);np.random.seed(42);torch.manual_seed(42)
                    episode=exp.episode(task,initial,mode,video_dir=args.output if args.save_videos else None)
                    episode.update(architecture='bounded_residual' if mode=='candidate' else 'VLA',alpha=args.alpha if mode=='candidate' else None)
                    episodes.append(episode);write_json(args.output/'episodes.json',episodes)
        if before!=dict(progress=exp.progress.copy(),replay_size=len(exp.buffer),position=exp.buffer.pos) or not equal(raw.state_dict(),supplied):raise RuntimeError('Frozen model or replay changed')
        if any(sha256(args.run/name)!=digest for name,digest in identities.items()):raise RuntimeError('Frozen checkpoint files changed')
        close();close=None
        counts={m:dict(successes=sum(e['success'] for e in episodes if e['policy']==m),episodes=sum(e['policy']==m for e in episodes)) for m in ['baseline','candidate']}
        per_task={t:{m:dict(successes=sum(e['success'] for e in episodes if e['policy']==m and e['task_id']==t),episodes=len(args.initials)) for m in counts} for t in args.tasks}
        paired=[]
        for task in args.tasks:
            for initial in args.initials:
                pair={e['policy']:e for e in episodes if e['task_id']==task and e['initial_state']==initial};b,c=pair['baseline'],pair['candidate']
                paired.append(dict(task_id=task,initial_state=initial,baseline=b['success'],candidate=c['success'],candidate_only=c['success'] and not b['success'],baseline_only=b['success'] and not c['success'],step_delta_if_both_success=c['action_steps']-b['action_steps'] if b['success'] and c['success'] else None))
        write_json(args.output/'summary.json',dict(split=args.split,tasks=args.tasks,initial_states=args.initials,counts=counts,per_task=per_task,paired=paired,episodes=episodes,macro_success_rate={m:float(np.mean([per_task[t][m]['successes']/len(args.initials) for t in args.tasks])) for m in counts},protocol='protocol.json'))
        report=dict(status='complete',mode=protocol['mode'],split=args.split,counts=counts,training_updates=0,replay_unchanged=True,candidate_parameters_unchanged=True,identity_BC_equals_baseline=True,wall_seconds=time.monotonic()-started,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        write_json(args.output/'run_summary.json',report);exp.log('evaluation_complete',**report)
    except BaseException as exc:
        write_json(args.output/'run_summary.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if close:close()


if __name__=='__main__':main()
