#!/usr/bin/env python3
"""Bounded offline continuation of a completed v2 experiment, without VLA/env.

Fresh twin Critic receives behavior Monte Carlo targets before TD3 updates.
Actor and matched BC control continue their respective v2 weights and Adam
states. This changes initialization AND adds training; it is not an isolated
causal comparison of MC versus TD initialization.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

from optimize_rlt import ROOT, write_json
from optimize_rlt_v2 import paired_actor_losses


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='Completed trusted local v2 run')
    p.add_argument('--output', type=Path, required=True, help='New sibling experiment directory')
    p.add_argument('--mc-steps', type=int, default=1024, help='Fresh Critic regression updates on actual behavior returns')
    p.add_argument('--actor-updates', type=int, default=512, help='Additional TD3 Actor and matched BC updates each')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cpu', help='cpu or explicit cuda device; no VLA is loaded')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    args = p.parse_args()
    if min(args.mc_steps, args.actor_updates, args.batch_size, args.cpu_threads) < 1:
        p.error('MC steps, Actor updates, batch size and CPU threads must be positive')
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.output.exists():
        p.error('Output must be a new directory')
    try:
        args.output.relative_to(args.run)
        p.error('Output must be outside the parent run')
    except ValueError:
        pass
    return args


def reconstruct_behavior_returns(replay, events, config, progress):
    """Audit chronological chunk/episode correspondence; reject incomplete data.

    This uses this runner's success-only reward=5 and first-success termination.
    It does not infer return labels from model predictions or validation.
    """
    import numpy as np
    import torch
    buffer = replay['buffer']
    episodes = [e for e in events if e.get('event')=='episode' and e.get('collect') is True]
    tasks = config['tasks']
    train_initials = config['train_initials']
    horizon, chunk_len, scale = config['action_horizon'], config['chunk_length'], config['reward_scale']
    gamma = config['gamma']
    expected_episodes = progress['warmup_done'] + progress['episodes_done']
    if not tasks or len(set(tasks)) != len(tasks) or not set(train_initials) <= set(range(30)):
        raise ValueError('Invalid task coverage or training-only initial-state definition')
    if not 0 < gamma <= 1 or chunk_len != 8 or horizon != 320 or scale != 5:
        raise ValueError('Unsupported source reward/chunk/horizon semantics')
    if len(episodes) != expected_episodes:
        raise ValueError('Collected episode count does not match completed progress')
    if progress['warmup_done'] != len(tasks)*config['warmup_episodes'] or progress['episodes_done'] != config['train_episodes']:
        raise ValueError('Only fully completed warmup and training budgets are accepted')
    total_chunks = sum(e['chunks'] for e in episodes)
    if total_chunks != len(buffer) or total_chunks > replay['capacity']:
        raise ValueError('Replay capacity overflow, missing chunks, or event/replay length mismatch')
    if not buffer or replay['pos'] != total_chunks % replay['capacity']:
        raise ValueError('Replay cursor inconsistent with a complete, non-overwritten chronological buffer')
    if sum(e['action_steps'] for e in episodes) != progress['env_steps']:
        raise ValueError('Environment action-step total does not match progress')
    if set(e['task_id'] for e in episodes) != set(tasks):
        raise ValueError('Replay episodes do not cover exactly the configured tasks')
    actual_task_index = {t: [] for t in tasks}
    returns = torch.zeros(total_chunks, dtype=torch.float32)
    records, cursor = [], 0
    for episode_index, e in enumerate(episodes):
        if episode_index < progress['warmup_done']:
            expected_policy = 'baseline'
            expected_task = tasks[episode_index % len(tasks)]
            expected_initial = train_initials[(episode_index // len(tasks)) % len(train_initials)]
        else:
            n = episode_index-progress['warmup_done']
            expected_policy = 'candidate'
            expected_task = tasks[n % len(tasks)]
            expected_initial = train_initials[(n // len(tasks)+config['warmup_episodes']) % len(train_initials)]
        if (e['policy'], e['task_id'], e['initial_state']) != (expected_policy, expected_task, expected_initial):
            raise ValueError(f'Episode {episode_index} task/policy/initial order differs from the declared schedule')
        steps, chunks = e['action_steps'], e['chunks']
        if not 1 <= steps <= horizon or chunks != (steps+chunk_len-1)//chunk_len:
            raise ValueError(f'Episode {episode_index} action-step/chunk count mismatch')
        expected_reward = scale if e['success'] else 0.
        if not np.isclose(e['reward'], expected_reward, atol=1e-6, rtol=0):
            raise ValueError(f'Episode {episode_index} violates success-only reward semantics')
        expected_last_chunk_reward = gamma**((steps-1)%chunk_len)*expected_reward
        segment = buffer[cursor:cursor+chunks]
        for j, transition in enumerate(segment):
            if len(transition) != 10 or transition[9] != e['task_id']:
                raise ValueError(f'Chunk {cursor+j} task/field mismatch')
            if any(not torch.isfinite(x).all() for x in transition[:9]):
                raise ValueError(f'Chunk {cursor+j} contains nonfinite data')
            if tuple(transition[2].shape) != (chunk_len, 7) or float(transition[2].abs().max()) > 1.000001:
                raise ValueError(f'Chunk {cursor+j} action representation differs from v2')
            final = j == chunks-1
            if float(transition[6]) != float(final):
                raise ValueError(f'Chunk {cursor+j} terminal flag is not aligned to the episode boundary')
            expected_chunk_reward = expected_last_chunk_reward if final else 0.
            if not np.isclose(float(transition[3]), expected_chunk_reward, atol=2e-6, rtol=1e-6):
                raise ValueError(f'Chunk {cursor+j} reward does not match logged success and action timing')
            if final:
                if any(torch.count_nonzero(transition[k]).item() for k in [4, 5, 8]):
                    raise ValueError(f'Terminal chunk {cursor+j} must use zero next observations')
            else:
                following = segment[j+1]
                if not all(torch.equal(transition[a], following[b]) for a, b in [(4,0),(5,1),(8,7)]):
                    raise ValueError(f'Chunk {cursor+j} next observation does not equal the next chunk input')
            actual_task_index[e['task_id']].append(cursor+j)
        value = 0.
        for j in range(chunks-1, -1, -1):
            transition = segment[j]
            value = float(transition[3]) + gamma**chunk_len*(1-float(transition[6]))*value
            returns[cursor+j] = value
        records.append(dict(episode_index=episode_index, task_id=e['task_id'], initial_state=e['initial_state'],
                            policy=e['policy'], first_chunk=cursor, chunk_count=chunks, action_steps=steps,
                            success=bool(e['success']), raw_reward=e['reward'], start_behavior_return=float(returns[cursor])))
        cursor += chunks
    supplied_task_index = {int(k): sorted(v) for k,v in replay['task_index'].items() if v}
    if supplied_task_index != actual_task_index:
        raise ValueError('Replay task index differs from audited chronological task mapping')
    if float(returns.min()) < 0 or float(returns.max()) > scale+1e-5:
        raise ValueError('Reconstructed behavior return outside valid reward bounds')
    audit = dict(episodes=len(episodes), chunks=total_chunks, action_steps=progress['env_steps'],
                 capacity_overflow=False, all_terminal_reward_task_mappings_verified=True,
                 task_episode_counts={t:sum(e['task_id']==t for e in episodes) for t in tasks},
                 task_successes={t:sum(e['task_id']==t and e['success'] for e in episodes) for t in tasks},
                 behavior_return_mean=float(returns.mean()), behavior_return_max=float(returns.max()),
                 note='Realized behavior-policy returns, not ground-truth current-policy Q values.', trajectories=records)
    return returns, audit


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def td_critic_loss(actor, critic, target_critic, buffer, *, batch_size, gamma,
                   device, target_actor, tasks, noise_std, noise_clip):
    """Same v2 TD3 target and twin-Q loss, with no model/env framework import."""
    import torch
    import torch.nn.functional as F
    z, ref, action, reward, nz, nref, done, prop, nprop = buffer.sample_balanced(
        batch_size, len(tasks), device, task_ids=tasks)
    with torch.no_grad():
        na, _ = target_actor(nz, nref, nprop, deterministic=True)
        noise = (torch.randn_like(na)*noise_std).clamp(-noise_clip, noise_clip)
        na = (na+noise).clamp(-1,1)
        nq1, nq2 = target_critic(nz, na, nprop)
        target = reward + gamma*(1-done)*torch.minimum(nq1,nq2)
    q1,q2 = critic(z, action, prop)
    loss = F.mse_loss(q1,target)+F.mse_loss(q2,target)
    return loss, dict(critic_loss=float(loss.detach()), q1_mean=float(q1.detach().mean()),
                      q2_mean=float(q2.detach().mean()), target_mean=float(target.mean()))


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def same_state(left, right):
    import torch
    return left.keys()==right.keys() and all(torch.equal(left[k].cpu(),right[k].cpu()) for k in left)


def run(args):
    # Check completion before opening the potentially large training checkpoint.
    parent_summary = json.loads((args.run/'run_summary.json').read_text())
    if parent_summary.get('status') != 'complete' or parent_summary.get('runner_version') != 2:
        raise ValueError('Parent must be a fully completed v2 run; refusing an active checkpoint')
    events = [json.loads(line) for line in (args.run/'events.jsonl').read_text().splitlines() if line]
    if not events or events[-1].get('event') != 'v2_complete':
        raise ValueError('Parent lacks its final v2_complete event; wait for the process to finish')
    config = json.loads((args.run/'config.json').read_text())
    parent_code = json.loads((args.run/'code_sha256.json').read_text())
    if config.get('runner_version') != 2 or not parent_code:
        raise ValueError('Source configuration is not v2')
    for filename, expected in parent_code.items():
        if sha256(filename) != expected:
            raise ValueError(f'Parent source changed: {filename}')
    identities = {name:sha256(args.run/name) for name in ['checkpoint.pt','config.json','events.jsonl','actor.pt','bc_only_actor.pt','run_summary.json']}
    import numpy as np
    import torch
    torch.set_num_threads(args.cpu_threads)
    payload = torch.load(args.run/'checkpoint.pt', map_location='cpu', weights_only=False)
    if payload.get('v2_code_sha256') != parent_code:
        raise ValueError('Parent checkpoint source identity differs from its manifest')
    if payload['progress']['actor_steps'] != payload['progress']['bc_control_steps']:
        raise ValueError('Parent TD3 and BC update counts do not match')
    for key, filename in [('actor','actor.pt'),('bc_actor','bc_only_actor.pt')]:
        state = torch.load(args.run/filename, map_location='cpu', weights_only=True)
        if payload.get(key) is None or not same_state(payload[key],state):
            raise ValueError(f'Parent {key} checkpoint/file identity differs')
    for name, expected in identities.items():
        if sha256(args.run/name) != expected:
            raise ValueError(f'Parent changed during loading: {name}')
    returns, audit = reconstruct_behavior_returns(payload['replay'], events, config, payload['progress'])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type not in ['cpu','cuda']:
        raise ValueError('Only CPU or CUDA small-network training is supported')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable')
    source = Path(config['source_root'])
    ac = load_module(source/'AlphaBrain/training/reinforcement_learning/algos/RLT_a/action_token_actor_critic.py','offline_rlt_ac')
    rb = load_module(source/'AlphaBrain/training/reinforcement_learning/common/replay_buffer.py','offline_rlt_replay')
    actor_args = dict(bottleneck_dim=256,action_dim=7,chunk_len=8,hidden_dim=512,prop_dim=8,
                      ref_dropout=config['ref_dropout'],fixed_std=config['exploration_std'])
    actor, control = ac.ActionTokenActor(**actor_args).to(device), ac.ActionTokenActor(**actor_args).to(device)
    actor.load_state_dict(payload['actor'],strict=True); control.load_state_dict(payload['bc_actor'],strict=True)
    actor_optimizer = torch.optim.Adam(actor.parameters(),lr=config['actor_lr'])
    control_optimizer = torch.optim.Adam(control.parameters(),lr=config['actor_lr'])
    actor_optimizer.load_state_dict(payload['actor_optimizer'])
    control_optimizer.load_state_dict(payload['bc_optimizer'])
    critic = ac.ActionTokenQCritic(bottleneck_dim=256,action_dim=7,chunk_len=8,hidden_dim=512,prop_dim=8).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(),lr=config['critic_lr'])
    target_actor = copy.deepcopy(actor).eval().requires_grad_(False)
    target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
    replay = rb.ReplayBuffer(capacity=payload['replay']['capacity'])
    replay.buffer, replay.pos = payload['replay']['buffer'], payload['replay']['pos']
    from collections import defaultdict
    replay._task_index = defaultdict(list,payload['replay']['task_index'])
    progress = copy.deepcopy(payload['progress'])
    progress.update(offline_mc_steps=0,offline_critic_steps=0,offline_actor_steps=0)
    new_config = copy.deepcopy(config)
    new_config.update(output=str(args.output),resume=False,runner_version=3,parent_run=str(args.run),
                      parent_checkpoint_sha256=identities['checkpoint.pt'],offline_mc_steps=args.mc_steps,
                      offline_actor_updates=args.actor_updates,offline_batch_size=args.batch_size,
                      offline_device=str(device),offline_seed=args.seed,
                      # Legacy frozen loader expects this original key to stay
                      # 'fresh'; the extra key records the actual v3 treatment.
                      critic_initialization='fresh',
                      offline_critic_initialization='fresh_behavior_MC_then_TD3',
                      offline_causal_scope='additional_offline_continuation; MC_initialization_and_training_budget_both_change')
    new_code = dict(parent_code)
    new_code[str(Path(__file__).resolve())] = sha256(__file__)
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',new_config)
    write_json(args.output/'code_sha256.json',new_code)
    write_json(args.output/'parent_identity.json',dict(run=str(args.run),files=identities,source=parent_code))
    write_json(args.output/'replay_return_audit.json',audit)
    torch.save(returns,args.output/'behavior_returns.pt')
    started = time.monotonic()
    def log(event, **fields):
        row = dict(event=event,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**fields)
        line = json.dumps(row,ensure_ascii=False,allow_nan=False)
        with (args.output/'events.jsonl').open('a') as stream: stream.write(line+'\n')
        print(line,flush=True)
    def step(loss, model, optimizer):
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite parameters after update')
        return float(norm)
    try:
        critic.train(); actor.train(); control.train()
        batch_size = min(args.batch_size,len(replay))
        tasks = config['tasks']
        per_task = max(1,batch_size//len(tasks))
        log('offline_started',parent_run=str(args.run),replay_chunks=len(replay),new_environment_steps=0,
            mc_steps=args.mc_steps,actor_updates=args.actor_updates,device=str(device),vla_loaded=False)
        for update in range(args.mc_steps):
            ids = np.concatenate([np.random.choice(replay._task_index[t],min(per_task,len(replay._task_index[t])),replace=False) for t in tasks])
            z,_,action,_,_,_,_,prop,_ = replay._collect(ids,device)
            target = returns[torch.as_tensor(ids,dtype=torch.long)].to(device)
            q1,q2 = critic(z,action,prop)
            loss = (q1-target).square().mean()+(q2-target).square().mean()
            norm = step(loss,critic,critic_optimizer)
            progress['offline_mc_steps']+=1; progress['critic_steps']+=1
            if (update+1)%64==0 or update==0 or update+1==args.mc_steps:
                log('mc_update',step=update+1,loss=float(loss.detach()),q_mean=float(q1.detach().mean()),
                    behavior_return_mean=float(target.mean()),grad_norm=norm)
        target_critic.load_state_dict(critic.state_dict())
        for update in range(args.actor_updates):
            for _ in range(2):
                loss, metrics = td_critic_loss(actor,critic,target_critic,replay,batch_size=batch_size,
                    gamma=config['gamma']**8,device=device,target_actor=target_actor,tasks=tasks,
                    noise_std=config['target_noise_std'],noise_clip=config['target_noise_clip'])
                metrics['critic_grad_norm']=step(loss,critic,critic_optimizer)
                progress['critic_steps']+=1; progress['offline_critic_steps']+=1
            critic.requires_grad_(False)
            try:
                aloss,bloss,ametrics = paired_actor_losses(actor,control,critic,replay,batch_size=batch_size,
                    beta=config['beta'],device=device,task_ids=tasks)
                ametrics['actor_grad_norm']=step(aloss,actor,actor_optimizer)
                ametrics['bc_control_grad_norm']=step(bloss,control,control_optimizer)
            finally:
                critic.requires_grad_(True)
            progress['actor_steps']+=1; progress['bc_control_steps']+=1; progress['offline_actor_steps']+=1
            ac.soft_update_target(actor,target_actor,config['tau'])
            ac.soft_update_target(critic,target_critic,config['tau'])
            if (update+1)%32==0 or update==0 or update+1==args.actor_updates:
                log('offline_td3_update',step=update+1,**metrics,**ametrics,
                    cumulative_actor_steps=progress['actor_steps'],cumulative_bc_steps=progress['bc_control_steps'])
        if progress['actor_steps'] != progress['bc_control_steps']:
            raise RuntimeError('Matched BC update audit failed')
        for name,expected in identities.items():
            if sha256(args.run/name)!=expected:
                raise RuntimeError('Parent changed during offline continuation')
        final = dict(config=new_config,progress=progress,rng=rng_state(),actor=actor.state_dict(),critic=critic.state_dict(),
                     target_actor=target_actor.state_dict(),target_critic=target_critic.state_dict(),
                     actor_optimizer=actor_optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
                     bc_actor=control.state_dict(),bc_optimizer=control_optimizer.state_dict(),replay=payload['replay'],
                     v3_code_sha256=new_code,parent_checkpoint_sha256=identities['checkpoint.pt'])
        for name,obj in [('checkpoint.pt',final),('actor.pt',actor.state_dict()),('bc_only_actor.pt',control.state_dict())]:
            tmp=args.output/(name+'.tmp');torch.save(obj,tmp);tmp.replace(args.output/name)
        write_json(args.output/'progress.json',progress)
        summary=dict(status='complete',runner_version=3,mode='offline_MC_then_TD3_continuation',parent_run=str(args.run),
                     progress=progress,new_environment_steps=0,mc_critic_updates=args.mc_steps,
                     additional_td3_critic_updates=2*args.actor_updates,additional_actor_updates=args.actor_updates,
                     additional_bc_updates=args.actor_updates,bc_control_update_count_matches=True,
                     parent_files_unchanged=True,vla_loaded=False,encoder_loaded=False,simulation_loaded=False,
                     wall_seconds=time.monotonic()-started,efficacy_claim=False,
                     limitation='MC behavior-return initialization plus additional offline updates; not an isolated single-factor ablation.')
        write_json(args.output/'run_summary.json',summary)
        log('offline_complete',**summary)
    except BaseException as exc:
        write_json(args.output/'run_summary.json',dict(status='failed',runner_version=3,error=repr(exc),progress=progress))
        log('failed',error=repr(exc));raise


if __name__ == '__main__':
    run(arguments())
