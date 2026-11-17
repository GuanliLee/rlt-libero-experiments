#!/usr/bin/env python3
"""Offline custom bounded-residual RLT variant; no VLA or simulation loading.

Fresh last layer starts at zero, giving exactly the VLA reference action.
Frozen v2 replay supplies verified behavior MC returns for fresh Critic warmup.
The zero-Q BC control must remain exactly identical to its zero-head start.
This is a new architecture, not the original RL Token or released RLT_a.
"""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import time

import offline_rlt_mc_candidate as mc
from optimize_rlt import ROOT,write_json


def arguments():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--run',type=Path,required=True,help='Completed trusted local v2 run')
    p.add_argument('--output',type=Path,required=True,help='New directory outside the parent')
    p.add_argument('--mc-steps',type=int,default=1024)
    p.add_argument('--actor-updates',type=int,default=512)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--delta-max',type=float,default=.1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args()
    if min(args.mc_steps,args.actor_updates,args.batch_size,args.cpu_threads)<1:
        p.error('Step, batch and thread counts must be positive')
    if not math.isfinite(args.delta_max) or args.delta_max<=0:
        p.error('delta-max must be finite and positive')
    args.run,args.output=args.run.resolve(),args.output.resolve()
    if args.output.exists():p.error('Output must be a new directory')
    try:
        args.output.relative_to(args.run)
        p.error('Output must be outside the parent run')
    except ValueError:pass
    return args


def paired_residual_losses(actor,control,critic,replay,*,batch_size,beta,device,task_ids):
    """One batch, deterministic actions; BC regularizes the actual correction."""
    z,ref,_,_,_,_,_,prop,_=replay.sample_balanced(batch_size,len(task_ids),device,task_ids=task_ids)
    delta=actor.correction(z,ref,prop)
    control_delta=control.correction(z,ref,prop)
    action=(ref+delta).clamp(-1,1)
    q=critic.q1_forward(z,action,prop)
    bc=delta.square().sum((-1,-2)).mean()
    control_bc=control_delta.square().sum((-1,-2)).mean()
    loss=-q.mean()+beta*bc
    control_loss=beta*control_bc
    return loss,control_loss,dict(actor_loss=float(loss.detach()),q_actor_mean=float(q.detach().mean()),
        residual_bc_penalty=float(bc.detach()),control_bc_penalty=float(control_bc.detach()),
        max_abs_correction=float(delta.detach().abs().max()),mean_abs_correction=float(delta.detach().abs().mean()),
        control_max_abs_correction=float(control_delta.detach().abs().max()),
        actor_q_action_max_abs=float(action.detach().abs().max()),matched_batch=True,
        deterministic_actor_update=True,reference_dropout=0.)


def run(args):
    # Completion guard precedes opening the source checkpoint.
    report=json.loads((args.run/'run_summary.json').read_text())
    if report.get('status')!='complete' or report.get('runner_version')!=2:
        raise ValueError('Only fully completed v2 parents are accepted')
    events=[json.loads(s) for s in (args.run/'events.jsonl').read_text().splitlines() if s]
    if not events or events[-1].get('event')!='v2_complete':
        raise ValueError('Parent has not written its final v2_complete event')
    config=json.loads((args.run/'config.json').read_text())
    source_code=json.loads((args.run/'code_sha256.json').read_text())
    for name,expected in source_code.items():
        if mc.sha256(name)!=expected:raise ValueError(f'Parent source changed: {name}')
    identities={n:mc.sha256(args.run/n)for n in ['checkpoint.pt','config.json','events.jsonl','actor.pt','run_summary.json']}
    import numpy as np
    import torch
    import residual_actor as residual
    torch.set_num_threads(args.cpu_threads)
    payload=torch.load(args.run/'checkpoint.pt',map_location='cpu',weights_only=False)
    if config.get('runner_version')!=2 or payload.get('v2_code_sha256')!=source_code:
        raise ValueError('Parent v2 source identity mismatch')
    supplied=torch.load(args.run/'actor.pt',map_location='cpu',weights_only=True)
    if not mc.same_state(supplied,payload['actor']):raise ValueError('Parent Actor checkpoint/file mismatch')
    for name,expected in identities.items():
        if mc.sha256(args.run/name)!=expected:raise ValueError('Parent changed while loading')
    returns,audit=mc.reconstruct_behavior_returns(payload['replay'],events,config,payload['progress'])
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda'):raise ValueError('Only CPU/CUDA small-network training is supported')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    source=Path(config['source_root'])
    ac=mc.load_module(source/'AlphaBrain/training/reinforcement_learning/algos/RLT_a/action_token_actor_critic.py','residual_original_ac')
    rb=mc.load_module(source/'AlphaBrain/training/reinforcement_learning/common/replay_buffer.py','residual_replay')
    actor=residual.BoundedResidualActor(delta_max=args.delta_max).to(device)
    actor.load_state_dict(payload['actor'],strict=True)
    actor.zero_head()
    initial_state={k:v.detach().cpu().clone()for k,v in actor.state_dict().items()}
    control=copy.deepcopy(actor)
    actor_optimizer=torch.optim.Adam(actor.parameters(),lr=config['actor_lr'])
    control_optimizer=torch.optim.Adam(control.parameters(),lr=config['actor_lr'])
    critic=ac.ActionTokenQCritic(bottleneck_dim=256,action_dim=7,chunk_len=8,hidden_dim=512,prop_dim=8).to(device)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=config['critic_lr'])
    target_actor=copy.deepcopy(actor).eval().requires_grad_(False)
    target_critic=copy.deepcopy(critic).eval().requires_grad_(False)
    replay=rb.ReplayBuffer(payload['replay']['capacity'])
    replay.buffer,replay.pos=payload['replay']['buffer'],payload['replay']['pos']
    replay._task_index=defaultdict(list,payload['replay']['task_index'])
    progress=copy.deepcopy(payload['progress'])
    progress.update(parent_actor_steps=progress['actor_steps'],parent_critic_steps=progress['critic_steps'],
                    actor_steps=0,bc_control_steps=0,bc_steps=0,bc_control_initial_steps=0,
                    critic_steps=0,critic_pretrain_steps=0,offline_mc_steps=0,offline_critic_steps=0)
    new_config=copy.deepcopy(config)
    new_config.update(output=str(args.output),resume=False,runner_version=4,architecture='bounded_residual',
        parent_run=str(args.run),parent_checkpoint_sha256=identities['checkpoint.pt'],delta_max=args.delta_max,
        offline_mc_steps=args.mc_steps,offline_actor_updates=args.actor_updates,offline_batch_size=args.batch_size,
        offline_seed=args.seed,offline_device=str(device),critic_initialization='fresh',
        offline_critic_initialization='fresh_behavior_MC_then_TD3',ref_dropout=0.,exploration_std=0.,
        actor_update_mode='deterministic',bc_target='squared_bounded_correction',
        actor_initialization='v2_hidden_layers_zero_last_layer_fresh_Adam',
        bc_control='exact_identity_to_VLA;zero_Q_zero_head_fresh_Adam',
        inference_formula='reference + delta_max * tanh(raw_actor._get_mean(..., apply_dropout=False)); then existing action clamp/postprocess',
        not_original_RLT_a=True)
    new_code=dict(source_code)
    for p in [Path(__file__),Path(residual.__file__),Path(mc.__file__)]:new_code[str(p.resolve())]=mc.sha256(p)
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',new_config)
    write_json(args.output/'code_sha256.json',new_code)
    write_json(args.output/'parent_identity.json',dict(run=str(args.run),files=identities,source=source_code))
    write_json(args.output/'replay_return_audit.json',audit)
    torch.save(returns,args.output/'behavior_returns.pt')
    torch.save(initial_state,args.output/'initial_actor.pt')
    started=time.monotonic()
    def log(event,**fields):
        row=dict(event=event,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**fields)
        line=json.dumps(row,ensure_ascii=False,allow_nan=False)
        with(args.output/'events.jsonl').open('a')as stream:stream.write(line+'\n')
        print(line,flush=True)
    def step(loss,model,optimizer):
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all()for p in model.parameters()):raise FloatingPointError('Nonfinite parameters')
        return float(norm)
    try:
        actor.train();control.train();critic.train()
        batch_size=min(args.batch_size,len(replay));tasks=config['tasks'];per_task=max(1,batch_size//len(tasks))
        log('residual_started',parent_run=str(args.run),new_environment_steps=0,replay_chunks=len(replay),
            delta_max=args.delta_max,mc_steps=args.mc_steps,actor_updates=args.actor_updates,
            architecture='custom_bounded_residual',vla_loaded=False)
        for update in range(args.mc_steps):
            ids=np.concatenate([np.random.choice(replay._task_index[t],min(per_task,len(replay._task_index[t])),replace=False)for t in tasks])
            z,_,action,_,_,_,_,prop,_=replay._collect(ids,device)
            target=returns[torch.as_tensor(ids,dtype=torch.long)].to(device)
            q1,q2=critic(z,action,prop)
            loss=(q1-target).square().mean()+(q2-target).square().mean()
            norm=step(loss,critic,critic_optimizer)
            progress['offline_mc_steps']+=1;progress['critic_steps']+=1
            if update==0 or(update+1)%64==0 or update+1==args.mc_steps:
                log('mc_update',step=update+1,loss=float(loss.detach()),q_mean=float(q1.detach().mean()),
                    behavior_return_mean=float(target.mean()),grad_norm=norm)
        target_critic.load_state_dict(critic.state_dict())
        for update in range(args.actor_updates):
            for _ in range(2):
                loss,metrics=mc.td_critic_loss(actor,critic,target_critic,replay,batch_size=batch_size,
                    gamma=config['gamma']**8,device=device,target_actor=target_actor,tasks=tasks,
                    noise_std=config['target_noise_std'],noise_clip=config['target_noise_clip'])
                metrics['critic_grad_norm']=step(loss,critic,critic_optimizer)
                progress['critic_steps']+=1;progress['offline_critic_steps']+=1
            critic.requires_grad_(False)
            try:
                aloss,bloss,ametrics=paired_residual_losses(actor,control,critic,replay,batch_size=batch_size,
                    beta=config['beta'],device=device,task_ids=tasks)
                ametrics['actor_grad_norm']=step(aloss,actor,actor_optimizer)
                ametrics['bc_control_grad_norm']=step(bloss,control,control_optimizer)
            finally:critic.requires_grad_(True)
            progress['actor_steps']+=1;progress['bc_control_steps']+=1
            if not mc.same_state(control.state_dict(),initial_state):
                raise RuntimeError('Identity BC control changed parameters despite its zero correction')
            if ametrics['control_max_abs_correction']!=0. or ametrics['control_bc_penalty']!=0.:
                raise RuntimeError('Zero-Q control is not an exact identity')
            ac.soft_update_target(actor,target_actor,config['tau']);ac.soft_update_target(critic,target_critic,config['tau'])
            if update==0 or(update+1)%32==0 or update+1==args.actor_updates:
                log('residual_td3_update',step=update+1,**metrics,**ametrics,identity_control_parameters_unchanged=True)
        for name,expected in identities.items():
            if mc.sha256(args.run/name)!=expected:raise RuntimeError('Parent changed during offline training')
        changed=not mc.same_state(actor.state_dict(),initial_state)
        if not changed:raise RuntimeError('Residual Actor did not update')
        final=dict(config=new_config,progress=progress,rng=mc.rng_state(),actor=actor.state_dict(),critic=critic.state_dict(),
            target_actor=target_actor.state_dict(),target_critic=target_critic.state_dict(),
            actor_optimizer=actor_optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
            bc_actor=control.state_dict(),bc_optimizer=control_optimizer.state_dict(),replay=payload['replay'],
            v4_code_sha256=new_code,parent_checkpoint_sha256=identities['checkpoint.pt'],
            architecture='bounded_residual',delta_max=args.delta_max)
        for name,obj in [('checkpoint.pt',final),('actor.pt',actor.state_dict()),('bc_only_actor.pt',control.state_dict())]:
            tmp=args.output/(name+'.tmp');torch.save(obj,tmp);tmp.replace(args.output/name)
        write_json(args.output/'progress.json',progress)
        identity=dict(exact_parameter_identity_to_zero_head_initialization=True,
            every_control_update_verified=True,control_updates=progress['bc_control_steps'],
            candidate_updates=progress['actor_steps'],head_weight_max_abs=float(control.net[-1].weight.detach().abs().max()),
            head_bias_max_abs=float(control.net[-1].bias.detach().abs().max()),
            formula='reference + delta_max*tanh(0) == reference',independent_simulation_required=False)
        write_json(args.output/'identity_control_audit.json',identity)
        summary=dict(status='complete',runner_version=4,architecture='bounded_residual',not_original_RLT_a=True,
            parent_run=str(args.run),progress=progress,new_environment_steps=0,mc_critic_updates=args.mc_steps,
            additional_td3_critic_updates=2*args.actor_updates,actor_updates=args.actor_updates,
            identity_bc_updates=args.actor_updates,identity_bc_parameters_unchanged=True,actor_parameters_changed=True,
            parent_files_unchanged=True,vla_loaded=False,encoder_loaded=False,simulation_loaded=False,
            wall_seconds=time.monotonic()-started,efficacy_claim=False,
            inference_requires_residual_wrapper=True,delta_max=args.delta_max,
            limitation='Custom architecture plus MC initialization and offline training; evaluate before claiming improvement.')
        write_json(args.output/'run_summary.json',summary)
        log('residual_complete',**summary)
    except BaseException as exc:
        write_json(args.output/'run_summary.json',dict(status='failed',runner_version=4,error=repr(exc),progress=progress))
        log('failed',error=repr(exc));raise


if __name__=='__main__':run(arguments())
