#!/usr/bin/env python3
"""Serial, single-GPU RLT_a optimization with paired held-out validation.

Uses the audited AlphaBrain-opt checkout. Original checkpoints stay unchanged.
This is an experiment runner, not a claim that a candidate improves the VLA.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def indices(value):
    result = []
    try:
        for item in value.split(','):
            if '-' in item:
                a, b = map(int, item.split('-'))
                if a > b:
                    raise ValueError('reversed range')
                result.extend(range(a, b + 1))
            else:
                result.append(int(item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated integers or ranges, e.g. 0-29') from exc
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError('Indices must be nonempty and unique')
    return result


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--output', type=Path, required=True, help='New experiment directory')
    p.add_argument('--resume', action='store_true', help='Resume this output directory from checkpoint.pt (trusted local files only)')
    p.add_argument('--tasks', type=indices, default=indices('0-9'))
    p.add_argument('--train-episodes', type=int, default=100, help='Total RL collection episodes, excluding warmup; total target on resume')
    p.add_argument('--warmup-episodes', type=int, default=1, help='Baseline collection episodes per task')
    p.add_argument('--train-initials', type=indices, default=indices('0-29'))
    p.add_argument('--val-initials', type=indices, default=indices('30-39'))
    p.add_argument('--val-limit', type=int, default=0, help='First N validation initial states per task; 0 means all')
    p.add_argument('--val-every', type=int, default=0, help='Validation interval in training episodes; 0 means final only')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--updates-per-episode', type=int, default=32)
    p.add_argument('--bc-steps', type=int, default=0, help='Optional initial imitation-only updates; saves a BC-only ablation')
    p.add_argument('--critic-pretrain-steps', type=int, default=0, help='Optional Critic-only warmup updates after BC; Polyak-update target Critic each step')
    p.add_argument('--actor-lr', type=float, default=1e-5)
    p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--beta', type=float, default=3.0)
    p.add_argument('--exploration-std', type=float, default=0.05)
    p.add_argument('--ref-dropout', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--target-noise-std', type=float, default=0.2)
    p.add_argument('--target-noise-clip', type=float, default=0.5)
    p.add_argument('--capacity', type=int, default=100000)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-videos', action='store_true', help='Save held-out validation videos')
    p.add_argument('--smoke', action='store_true', help='First requested task, 1 warmup + 1 RL episode, 4 updates, 1 validation initial; plumbing only')
    args = p.parse_args()
    if args.smoke:
        args.tasks = args.tasks[:1]
        args.train_episodes = args.warmup_episodes = args.val_limit = 1
        args.batch_size, args.updates_per_episode = 8, 4
        args.bc_steps = min(args.bc_steps, 2)
    if not set(args.tasks) <= set(range(10)):
        p.error('LIBERO-Goal task IDs must be in 0-9')
    if not set(args.train_initials) <= set(range(30)):
        p.error('Training initial states must be within 0-29')
    if not set(args.val_initials) <= set(range(30, 40)):
        p.error('Validation initial states must be within 30-39; 40-49 are reserved for untouched testing')
    if set(args.train_initials) & set(args.val_initials):
        p.error('Training and validation initial states must not overlap')
    if args.train_episodes < 0 or args.warmup_episodes < 0 or args.bc_steps < 0 or args.critic_pretrain_steps < 0 or args.val_limit < 0 or args.val_every < 0:
        p.error('Episode counts, BC steps, and validation limits must be nonnegative')
    if min(args.batch_size, args.updates_per_episode, args.capacity, args.save_every) < 1:
        p.error('Batch size, update count, capacity and save interval must be positive')
    if not 0 < args.gamma <= 1 or not 0 < args.tau <= 1 or not 0 <= args.ref_dropout < 1:
        p.error('Invalid gamma, tau or reference dropout')
    if min(args.actor_lr, args.critic_lr, args.max_grad_norm) <= 0 or min(args.beta, args.exploration_std, args.target_noise_std, args.target_noise_clip) < 0:
        p.error('Learning rates must be positive and penalty/noise parameters nonnegative')
    args.output = args.output.resolve()
    if args.resume:
        if not (args.output / 'checkpoint.pt').is_file():
            p.error('--resume requires output/checkpoint.pt')
    elif args.output.exists():
        p.error('Output already exists; choose a new directory or explicitly --resume')
    return args


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


class Experiment:
    def __init__(self, args):
        # Imports are delayed so --help works without importing a GPU framework.
        global torch, np
        import torch
        import numpy as np
        self.args, self.output = args, args.output
        self.source = Path(os.environ.get('RLT_SOURCE_ROOT', ROOT / 'src/AlphaBrain-opt')).resolve()
        os.environ['RLT_SOURCE_ROOT'] = str(self.source)
        sys.path.insert(0, str(self.source))
        import eval_rlt_a_sdpa as checked
        if Path(checked.REPO).resolve() != self.source:
            raise RuntimeError('Strict-load helper must honor RLT_SOURCE_ROOT; refusing unaudited imports')
        from AlphaBrain.training.reinforcement_learning.envs import libero_env
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor, ActionTokenQCritic, soft_update_target
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import action_token_td_actor_update, action_token_td_critic_update
        from AlphaBrain.training.reinforcement_learning.common.replay_buffer import ReplayBuffer
        from AlphaBrain.training.reinforcement_learning.common.rollout import DUMMY_ACTION, _unnormalize, _postprocess_action, _save_video
        libero_env._WORKER_SCRIPT = str(ROOT / 'scripts/libero_worker_fast_png.py')
        self.Env = libero_env.LiberoEnv
        self.dummy, self.unnormalize, self.postprocess, self.save_video = DUMMY_ACTION, _unnormalize, _postprocess_action, _save_video
        self.actor_update, self.critic_update, self.soft_update = action_token_td_actor_update, action_token_td_critic_update, soft_update_target
        self.device = 'cuda:0'
        if not torch.cuda.is_available():
            raise RuntimeError('This runner needs one CUDA GPU')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        self.output.mkdir(parents=True, exist_ok=args.resume)
        self.config = vars(args).copy()
        self.config['output'] = str(args.output)
        self.config.update(source_root=str(self.source), chunk_length=8, action_horizon=320,
                           reward_scale=5, evaluation_seed=42, test_initials_reserved=list(range(40, 50)),
                           critic_initialization='fresh', terminal_time_limit=True,
                           torch_version=torch.__version__, device_name=torch.cuda.get_device_name(0))
        if args.resume:
            original = json.loads((self.output / 'config.json').read_text())
            for key, value in self.config.items():
                if key not in {'resume', 'train_episodes', 'save_videos'} and original.get(key) != value:
                    raise ValueError(f'Resume configuration mismatch: {key}')
        else:
            write_json(self.output / 'config.json', self.config)
            source_files = [Path(__file__), self.source / 'AlphaBrain/training/reinforcement_learning/algos/RLT_a/action_token_trainer.py',
                            self.source / 'AlphaBrain/training/reinforcement_learning/common/replay_buffer.py']
            write_json(self.output / 'code_sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
        self.log('loading', source_root=str(self.source))
        self.vla = checked.base_framework.BaseFramework.from_pretrained(ROOT / 'checkpoints/qwenoft_5traj')
        self.vla = self.vla.to(torch.bfloat16).to(self.device).eval().requires_grad_(False)
        self.encoder = ActionTokenEncoderDecoder(input_dim=2048, bottleneck_dim=256, chunk_len=8,
                                                 num_heads=4, encoder_layers=2, decoder_layers=2).to(self.device)
        release_dir = ROOT / 'checkpoints/rlt_a/rl_offpolicy_iter_00400'
        self.encoder.load_state_dict(torch.load(release_dir / 'encoder.pt', map_location='cpu', weights_only=True), strict=True)
        self.encoder.eval().requires_grad_(False)
        actor_args = dict(bottleneck_dim=256, action_dim=7, chunk_len=8, hidden_dim=512, ref_dropout=args.ref_dropout, fixed_std=args.exploration_std, prop_dim=8)
        self.actor = ActionTokenActor(**actor_args).to(self.device)
        self.actor.load_state_dict(torch.load(release_dir / 'actor.pt', map_location='cpu', weights_only=True), strict=True)
        if sum(p.numel() for p in self.actor.parameters()) != 455736:
            raise RuntimeError('Unexpected actor parameter count')
        self.release_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        # Checkpoint fixed_std is a buffer: overwrite only the experimental actor.
        self.actor.fixed_std.fill_(args.exploration_std)
        self.critic = ActionTokenQCritic(bottleneck_dim=256, action_dim=7, chunk_len=8, hidden_dim=512, prop_dim=8).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.buffer = ReplayBuffer(capacity=args.capacity)
        self.stats = next(iter(self.vla.norm_stats.values()))['action']
        self.progress = dict(warmup_done=0, episodes_done=0, critic_steps=0, actor_steps=0, bc_steps=0, critic_pretrain_steps=0, env_steps=0, validation_runs=0)
        self.bc_actor = None
        self.started = time.monotonic()
        if args.resume:
            self.restore()
        self.log('ready', actor_parameters=455736, vla_frozen=True, encoder_frozen=True, critic_initialization='fresh' if not args.resume else 'resumed')

    def log(self, event, **fields):
        item = dict(event=event, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **fields)
        line = json.dumps(item, ensure_ascii=False, allow_nan=False)
        with (self.output / 'events.jsonl').open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)

    @staticmethod
    def rng_state():
        return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

    @staticmethod
    def set_rng(state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])

    def save(self):
        payload = dict(config=self.config, progress=self.progress, rng=self.rng_state(),
                       actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                       target_actor=self.target_actor.state_dict(), target_critic=self.target_critic.state_dict(),
                       actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                       replay=dict(capacity=self.buffer.capacity, buffer=self.buffer.buffer, pos=self.buffer.pos, task_index=dict(self.buffer._task_index)))
        temp = self.output / 'checkpoint.pt.tmp'
        torch.save(payload, temp)
        temp.replace(self.output / 'checkpoint.pt')
        torch.save(self.actor.state_dict(), self.output / 'actor.pt.tmp')
        (self.output / 'actor.pt.tmp').replace(self.output / 'actor.pt')
        write_json(self.output / 'progress.json', self.progress)

    def restore(self):
        from collections import defaultdict
        # Only an explicitly requested resume of this runner's trusted local output.
        payload = torch.load(self.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        for name in ['actor', 'critic', 'target_actor', 'target_critic']:
            getattr(self, name).load_state_dict(payload[name], strict=True)
        self.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        self.progress = payload['progress']
        state = payload['replay']
        self.buffer.capacity, self.buffer.buffer, self.buffer.pos = state['capacity'], state['buffer'], state['pos']
        self.buffer._task_index = defaultdict(list, state['task_index'])
        if (self.output / 'bc_only_actor.pt').exists():
            self.bc_actor = copy.deepcopy(self.release_actor)
            self.bc_actor.load_state_dict(torch.load(self.output / 'bc_only_actor.pt', map_location='cpu', weights_only=True))
        self.set_rng(payload['rng'])

    def encode_observation(self, obs, instruction):
        # no_grad creates ordinary tensors, safe to cache as later training inputs.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            features, reference = self.vla.get_vla_action(batch_images=[[obs['primary_image'], obs['wrist_image']]], instructions=[instruction])
        with torch.no_grad():
            token = self.encoder.encode(features).float()
            reference = reference[:, :8, :].float()
            prop = torch.as_tensor(obs['state'], dtype=torch.float32, device=self.device).unsqueeze(0)
        for tensor in [token, reference, prop]:
            if not torch.isfinite(tensor).all():
                raise FloatingPointError('Nonfinite VLA/encoder observation')
        return token, reference, prop

    def episode(self, task, initial, mode, *, collect=False, video_dir=None):
        start = time.monotonic()
        env = self.Env()
        steps, reward_sum, success, chunks = 0, 0., False, 0
        action_deltas, frames = [], []
        actors = {'release': self.release_actor, 'candidate': self.actor, 'bc_only': self.bc_actor}
        actor = actors.get(mode)
        if mode != 'baseline':
            actor.eval()
        try:
            obs = env.reset('libero_goal', task, initial_state_idx=initial, seed=42)
            for _ in range(10):
                obs, _, _ = env.step(self.dummy)
            token, reference, prop = self.encode_observation(obs, env.task_description)
            while steps < 320:
                with torch.no_grad():
                    action = reference.clone() if mode == 'baseline' else actor(token, reference, prop, deterministic=True)[0]
                    if collect and mode == 'candidate':
                        action = action + torch.randn_like(action) * self.args.exploration_std
                    action = action.clamp(-1, 1)
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Nonfinite actor action')
                action_deltas.append(float((action-reference).abs().mean()))
                physical = self.unnormalize(action[0].cpu().numpy(), self.stats)
                chunk_reward, terminal = 0., False
                for k in range(8):
                    obs, reward, done = env.step(self.postprocess(physical[k]))
                    steps += 1
                    reward_sum += reward * 5.
                    chunk_reward += self.args.gamma ** k * reward * 5.
                    success = success or reward > .5
                    if video_dir:
                        frames.append(obs['primary_image'].copy())
                    terminal = bool(done or steps >= 320)
                    if terminal:
                        break
                chunks += 1
                if terminal:
                    next_token, next_reference, next_prop = torch.zeros_like(token), torch.zeros_like(reference), torch.zeros_like(prop)
                else:
                    next_token, next_reference, next_prop = self.encode_observation(obs, env.task_description)
                if collect:
                    self.buffer.push(token[0], reference[0], action[0], chunk_reward, next_token[0], next_reference[0], terminal,
                                     task_id=task, prop_state=prop[0], next_prop_state=next_prop[0])
                if terminal:
                    break
                token, reference, prop = next_token, next_reference, next_prop
            result = dict(policy=mode, task_id=task, task=env.task_description, initial_state=initial, seed=42,
                          success=bool(success), action_steps=steps, chunks=chunks, reward=reward_sum,
                          mean_abs_action_difference=float(np.mean(action_deltas)), wall_seconds=time.monotonic()-start)
            if collect:
                self.progress['env_steps'] += steps
            if video_dir:
                filename = f'{mode}_task{task:02d}_init{initial:02d}_{"success" if success else "fail"}.mp4'
                self.save_video(frames, str(video_dir / filename))
                result['video'] = filename
            self.log('episode', collect=collect, **result)
            return result
        finally:
            env.close()

    def backward_step(self, loss, model, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite model parameters after update')
        return float(norm)

    def learn(self):
        if not len(self.buffer):
            raise RuntimeError('No training transitions')
        self.actor.train()
        self.critic.train()
        effective_batch = min(self.args.batch_size, len(self.buffer))
        if effective_batch < 2:
            raise RuntimeError('At least two transitions are required for a meaningful update')
        for _ in range(self.args.updates_per_episode):
            loss, metrics = self.critic_update(self.actor, self.critic, self.target_critic, self.buffer,
                batch_size=effective_batch, gamma=self.args.gamma**8, device=self.device,
                target_noise_std=self.args.target_noise_std, target_noise_clip=self.args.target_noise_clip,
                target_actor=self.target_actor, task_ids=self.args.tasks)
            metrics['critic_grad_norm'] = self.backward_step(loss, self.critic, self.critic_optimizer)
            self.progress['critic_steps'] += 1
            if self.progress['critic_steps'] % 2 == 0:
                self.critic.requires_grad_(False)
                try:
                    actor_loss, actor_metrics = self.actor_update(self.actor, self.critic, self.buffer,
                        batch_size=effective_batch, beta=self.args.beta, device=self.device, task_ids=self.args.tasks)
                    actor_metrics['actor_grad_norm'] = self.backward_step(actor_loss, self.actor, self.actor_optimizer)
                finally:
                    self.critic.requires_grad_(True)
                metrics.update(actor_metrics)
                self.progress['actor_steps'] += 1
                self.soft_update(self.actor, self.target_actor, self.args.tau)
                self.soft_update(self.critic, self.target_critic, self.args.tau)
            self.log('update', **self.progress, **metrics, effective_batch=effective_batch, replay_size=len(self.buffer), task_counts=self.buffer.task_counts())

    def calibrate_bc(self):
        if not self.args.bc_steps:
            return
        if not len(self.buffer):
            raise RuntimeError('BC calibration needs warmup data')
        self.actor.train()
        while self.progress['bc_steps'] < self.args.bc_steps:
            token, reference, _, _, _, _, _, prop, _ = self.buffer.sample_balanced(self.args.batch_size, len(self.args.tasks), self.device, task_ids=self.args.tasks)
            # Pure imitation ablation, without a Q term and without actor noise.
            mean = self.actor._get_mean(token, reference, prop, apply_dropout=False)
            loss = (mean-reference).square().sum(dim=(-2, -1)).mean()
            grad = self.backward_step(loss, self.actor, self.actor_optimizer)
            self.progress['bc_steps'] += 1
            self.log('bc_update', step=self.progress['bc_steps'], bc_loss=float(loss), grad_norm=grad)
        self.bc_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        torch.save(self.bc_actor.state_dict(), self.output / 'bc_only_actor.pt')
        self.target_actor.load_state_dict(self.actor.state_dict())

    def calibrate_critic(self):
        if len(self.buffer) < 2:
            raise RuntimeError('Critic calibration needs at least two warmup transitions')
        self.actor.eval()
        self.critic.train()
        while self.progress['critic_pretrain_steps'] < self.args.critic_pretrain_steps:
            loss, metrics = self.critic_update(self.actor, self.critic, self.target_critic, self.buffer,
                batch_size=min(self.args.batch_size, len(self.buffer)), gamma=self.args.gamma**8, device=self.device,
                target_noise_std=self.args.target_noise_std, target_noise_clip=self.args.target_noise_clip,
                target_actor=self.target_actor, task_ids=self.args.tasks)
            metrics['critic_grad_norm'] = self.backward_step(loss, self.critic, self.critic_optimizer)
            self.progress['critic_steps'] += 1
            self.progress['critic_pretrain_steps'] += 1
            # No Actor update during calibration; only the target Critic follows.
            self.soft_update(self.critic, self.target_critic, self.args.tau)
            self.log('critic_pretrain_update', **self.progress, **metrics)

    def verify_checkpoint(self):
        """Smoke verification: reload the saved Actor and complete local state."""
        saved = torch.load(self.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        reloaded_actor = copy.deepcopy(self.actor).cpu()
        reloaded_actor.load_state_dict(saved['actor'], strict=True)

        def equal(left, right):
            if isinstance(left, torch.Tensor):
                return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
            if isinstance(left, np.ndarray):
                return isinstance(right, np.ndarray) and np.array_equal(left, right)
            if isinstance(left, dict):
                return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
            if isinstance(left, (tuple, list)):
                return isinstance(right, type(left)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
            return left == right

        checks = dict(actor_reload_equal=equal(reloaded_actor.state_dict(), self.actor.state_dict()),
                      critic_equal=equal(saved['critic'], self.critic.state_dict()),
                      target_actor_equal=equal(saved['target_actor'], self.target_actor.state_dict()),
                      target_critic_equal=equal(saved['target_critic'], self.target_critic.state_dict()),
                      actor_optimizer_equal=equal(saved['actor_optimizer'], self.actor_optimizer.state_dict()),
                      critic_optimizer_equal=equal(saved['critic_optimizer'], self.critic_optimizer.state_dict()),
                      rng_equal=equal(saved['rng'], self.rng_state()),
                      progress_equal=equal(saved['progress'], self.progress),
                      replay_equal=equal(saved['replay']['buffer'], self.buffer.buffer),
                      replay_index_equal=equal(saved['replay']['task_index'], dict(self.buffer._task_index)),
                      replay_position_equal=saved['replay']['pos'] == self.buffer.pos)
        if not all(checks.values()):
            raise RuntimeError(f'Checkpoint roundtrip verification failed: {checks}')
        report = dict(checks=checks, replay_size=len(self.buffer), progress=self.progress.copy(), success=True)
        write_json(self.output / 'checkpoint_verification.json', report)
        self.log('checkpoint_verified', **report)
        del reloaded_actor, saved
        return True

    def validate(self):
        rng = self.rng_state()
        self.progress['validation_runs'] += 1
        folder = self.output / f'validation_{self.progress["validation_runs"]:03d}_ep{self.progress["episodes_done"]:05d}'
        if folder.exists():
            # A previous process may have crashed after creating this validation.
            # Keep those artifacts and allocate a new attempt on explicit resume.
            base = folder
            retry = 1
            while folder.exists():
                folder = base.with_name(base.name + f'_retry{retry:02d}')
                retry += 1
        folder.mkdir()
        initial_states = self.args.val_initials[:self.args.val_limit or None]
        episodes = []
        policies = ['baseline', 'release'] + (['bc_only'] if self.bc_actor is not None else []) + ['candidate']
        try:
            for task in self.args.tasks:
                for initial in initial_states:
                    for mode in policies:
                        random.seed(42)
                        np.random.seed(42)
                        torch.manual_seed(42)
                        episodes.append(self.episode(task, initial, mode, video_dir=folder if self.args.save_videos else None))
                        write_json(folder / 'episodes.json', episodes)
            counts = {mode: dict(successes=sum(e['success'] for e in episodes if e['policy']==mode),
                                 episodes=sum(e['policy']==mode for e in episodes)) for mode in policies}
            per_task = {task: {mode: dict(successes=sum(e['success'] for e in episodes if e['policy']==mode and e['task_id']==task),
                                         episodes=sum(e['policy']==mode and e['task_id']==task for e in episodes))
                               for mode in policies} for task in self.args.tasks}
            macro_rates = {mode: float(np.mean([per_task[task][mode]['successes']/per_task[task][mode]['episodes'] for task in self.args.tasks]))
                           for mode in policies}
            paired = []
            for task in self.args.tasks:
                for initial in initial_states:
                    items = {e['policy']:e for e in episodes if e['task_id']==task and e['initial_state']==initial}
                    base, candidate = items['baseline'], items['candidate']
                    paired.append(dict(task_id=task, initial_state=initial, baseline=base['success'], candidate=candidate['success'],
                                       candidate_only=bool(candidate['success'] and not base['success']),
                                       baseline_only=bool(base['success'] and not candidate['success']),
                                       step_delta_if_both_success=candidate['action_steps']-base['action_steps'] if candidate['success'] and base['success'] else None))
            report = dict(split='validation_only', initial_states=initial_states, tasks=self.args.tasks, counts=counts,
                          per_task=per_task, macro_success_rate=macro_rates,
                          paired=paired, progress=self.progress.copy(), episodes=episodes,
                          note='Exploratory validation for model selection; reserved test states 40-49 remain untouched. Smoke results are not efficacy evidence.')
            write_json(folder / 'summary.json', report)
            write_json(self.output / 'latest_validation.json', report)
            torch.save(self.actor.state_dict(), folder / 'actor.pt')
            self.log('validation_complete', directory=str(folder), counts=counts)
        finally:
            self.set_rng(rng)

    def run(self):
        warmup_pairs = [(task, self.args.train_initials[i % len(self.args.train_initials)])
                        for i in range(self.args.warmup_episodes) for task in self.args.tasks]
        for task, initial in warmup_pairs[self.progress['warmup_done']:]:
            self.episode(task, initial, 'baseline', collect=True)
            self.progress['warmup_done'] += 1
            self.save()
        if self.progress['bc_steps'] < self.args.bc_steps:
            self.calibrate_bc()
            self.save()
        if self.progress['critic_pretrain_steps'] < self.args.critic_pretrain_steps:
            self.calibrate_critic()
            self.save()
        while self.progress['episodes_done'] < self.args.train_episodes:
            ep = self.progress['episodes_done']
            task = self.args.tasks[ep % len(self.args.tasks)]
            initial = self.args.train_initials[(ep // len(self.args.tasks) + self.args.warmup_episodes) % len(self.args.train_initials)]
            self.episode(task, initial, 'candidate', collect=True)
            self.learn()
            self.progress['episodes_done'] += 1
            if self.args.val_every and self.progress['episodes_done'] % self.args.val_every == 0:
                self.validate()
            if self.progress['episodes_done'] % self.args.save_every == 0:
                self.save()
        self.save()
        if not self.args.val_every or self.progress['episodes_done'] % self.args.val_every:
            self.validate()
        self.save()
        checkpoint_verified = self.verify_checkpoint() if self.args.smoke else None
        delta = sum((a.detach()-b.detach()).double().square().sum().item() for a, b in zip(self.actor.parameters(), self.release_actor.parameters())) ** .5
        frozen = all(not p.requires_grad and p.grad is None for model in [self.vla, self.encoder] for p in model.parameters())
        report = dict(status='complete', progress=self.progress, actor_parameter_l2_change_from_release=delta,
                      frozen_vla_encoder_verified=frozen, wall_seconds=time.monotonic()-self.started,
                      peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                      smoke=self.args.smoke, checkpoint_roundtrip_verified=checkpoint_verified, efficacy_claim=False)
        if not frozen or (self.progress['actor_steps'] and delta == 0):
            raise RuntimeError('Frozen-model or actor-update audit failed')
        write_json(self.output / 'run_summary.json', report)
        self.log('complete', **report)


def main():
    args = arguments()
    experiment = Experiment(args)
    try:
        experiment.run()
    except BaseException as exc:
        experiment.log('failed', error=repr(exc), progress=experiment.progress)
        # Last atomic checkpoint is kept; do not overwrite it with a partial update.
        raise


if __name__ == '__main__':
    main()
