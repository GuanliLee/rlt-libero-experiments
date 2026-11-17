#!/usr/bin/env python3
"""RLT_a v2: bounded Critic actions and a matched-data/matched-update BC control.

The original runner and AlphaBrain-opt source remain unchanged. Normalized
motion actions are clamped for Q; LIBERO's separate gripper binarization remains
an approximation in the continuous Critic, exactly as documented in v1.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import optimize_rlt as legacy
from optimize_rlt import ROOT, indices, write_json


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--tasks', type=indices, default=indices('0-9'))
    p.add_argument('--train-episodes', type=int, default=100)
    p.add_argument('--warmup-episodes', type=int, default=5, help='Baseline episodes per task')
    p.add_argument('--train-initials', type=indices, default=indices('0-29'))
    p.add_argument('--val-initials', type=indices, default=indices('31-33'))
    p.add_argument('--val-limit', type=int, default=0)
    p.add_argument('--val-every', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--updates-per-episode', type=int, default=32)
    p.add_argument('--bc-steps', type=int, default=128)
    p.add_argument('--critic-pretrain-steps', type=int, default=128)
    p.add_argument('--actor-lr', type=float, default=1e-5)
    p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--beta', type=float, default=3.)
    p.add_argument('--exploration-std', type=float, default=.05)
    p.add_argument('--ref-dropout', type=float, default=.1)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--tau', type=float, default=.005)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--target-noise-std', type=float, default=.2)
    p.add_argument('--target-noise-clip', type=float, default=.5)
    p.add_argument('--capacity', type=int, default=100000)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-videos', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.tasks = args.tasks[:1]
        args.train_episodes = args.warmup_episodes = args.val_limit = 1
        args.batch_size, args.updates_per_episode = 8, 4
        args.bc_steps, args.critic_pretrain_steps = min(args.bc_steps, 2), min(args.critic_pretrain_steps, 2)
    if not set(args.tasks) <= set(range(10)):
        p.error('Task IDs must be within 0-9')
    if not set(args.train_initials) <= set(range(30)):
        p.error('Training initial states must be within 0-29')
    if not set(args.val_initials) <= set(range(30, 40)):
        p.error('Validation must stay within 30-39; 40-49 remain reserved')
    if min(args.train_episodes, args.warmup_episodes, args.bc_steps, args.critic_pretrain_steps, args.val_limit, args.val_every) < 0:
        p.error('Episode and calibration counts must be nonnegative')
    if min(args.batch_size, args.updates_per_episode, args.capacity, args.save_every) < 1:
        p.error('Batch size, updates, capacity and save interval must be positive')
    if not 0 < args.gamma <= 1 or not 0 < args.tau <= 1 or not 0 <= args.ref_dropout < 1:
        p.error('Invalid gamma, tau or reference dropout')
    if min(args.actor_lr, args.critic_lr, args.max_grad_norm) <= 0 or min(args.exploration_std, args.target_noise_std, args.target_noise_clip) < 0 or args.beta <= 0:
        p.error('Learning rates, beta and gradient bound must be positive; noise must be nonnegative')
    args.output = args.output.resolve()
    if args.resume:
        if not (args.output / 'checkpoint.pt').is_file():
            p.error('--resume requires output/checkpoint.pt')
    elif args.output.exists():
        p.error('Output exists; choose a new path or explicitly --resume')
    return args


def code_fingerprint(source):
    paths = [Path(__file__), Path(legacy.__file__), ROOT / 'scripts/run_optimization_v2.sh',
             ROOT / 'scripts/persistent_libero.py',
             ROOT / 'scripts/eval_rlt_a_sdpa.py', ROOT / 'scripts/libero_worker_fast_png.py',
             source / 'AlphaBrain/training/reinforcement_learning/algos/RLT_a/action_token_trainer.py',
             source / 'AlphaBrain/training/reinforcement_learning/algos/RLT_a/action_token_actor_critic.py',
             source / 'AlphaBrain/training/reinforcement_learning/common/replay_buffer.py',
             source / 'AlphaBrain/training/reinforcement_learning/common/rollout.py']
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def paired_actor_losses(actor, bc_actor, critic, replay, *, batch_size, beta, device, task_ids):
    """One replay sample, coupled policy noise, same BC term, Q only in candidate.

    BC keeps the raw action in its penalty to retain a restoring gradient for
    outputs beyond the action bounds. Both BC reference targets are clamped.
    Forking RNG for the control prevents its update changing the candidate's
    subsequent exploration/random minibatch sequence.
    """
    import torch
    batch = replay.sample_balanced(batch_size, len(task_ids), device, task_ids=task_ids)
    token, reference, _, _, _, _, _, prop, _ = batch
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(token.device) if token.is_cuda else None
    action, _ = actor(token, reference, prop, deterministic=False)
    devices = [token.device.index] if token.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, token.device)
        control_action, _ = bc_actor(token, reference, prop, deterministic=False)
    reference_target = reference.clamp(-1., 1.)
    bc = (action-reference_target).square().sum(dim=(-2, -1)).mean()
    control_bc = (control_action-reference_target).square().sum(dim=(-2, -1)).mean()
    executed_action = action.clamp(-1., 1.)
    q = critic.q1_forward(token, executed_action, prop)
    loss = -q.mean() + beta * bc
    control_loss = beta * control_bc
    metrics = dict(actor_loss=float(loss.detach()), actor_batch_size=int(action.shape[0]),
                   q_actor_mean=float(q.detach().mean()), bc_penalty=float(bc.detach()),
                   bc_control_loss=float(control_loss.detach()), bc_control_penalty=float(control_bc.detach()),
                   actor_raw_outside_fraction=float((action.detach().abs()>1).float().mean()),
                   control_raw_outside_fraction=float((control_action.detach().abs()>1).float().mean()),
                   actor_q_action_max_abs=float(executed_action.detach().abs().max()),
                   bc_reference_max_abs=float(reference_target.detach().abs().max()),
                   matched_replay_batch=True, matched_policy_noise=True)
    return loss, control_loss, metrics


class ExperimentV2(legacy.Experiment):
    def __init__(self, args):
        source = Path(os.environ.get('RLT_SOURCE_ROOT', ROOT / 'src/AlphaBrain-opt')).resolve()
        self.v2_fingerprint = code_fingerprint(source)
        if args.resume:
            manifest = json.loads((args.output / 'code_sha256.json').read_text())
            if manifest != self.v2_fingerprint:
                raise ValueError('Resume refused: v2 source/dependency SHA-256 differs from the saved run')
            cfg = json.loads((args.output / 'config.json').read_text())
            if cfg.get('runner_version') != 2:
                raise ValueError('Resume requires an authentic v2 run')
        self.bc_optimizer = None
        super().__init__(args)
        import torch
        self.config.update(runner_version=2, actor_q_action='clamp_normalized_to_minus1_plus1',
                           bc_target='clamped_reference; raw_actor_output_retains_restoring_gradient',
                           bc_control='same_initial_weights_optimizer_data_noise_and_update_count; Q_term_removed',
                           gripper_approximation='continuous_normalized_Q_action; unchanged_environment_gripper_binarization')
        if not args.resume:
            self.progress.update(bc_control_steps=0, bc_control_initial_steps=0)
            write_json(self.output / 'code_sha256.json', self.v2_fingerprint)
            write_json(self.output / 'config.json', self.config)
        # With zero calibration steps, both policies start from the release.
        if self.bc_actor is None and not args.bc_steps:
            self.initialize_control()

    def initialize_control(self):
        import torch
        self.bc_actor = copy.deepcopy(self.actor).train().requires_grad_(True)
        self.bc_optimizer = torch.optim.Adam(self.bc_actor.parameters(), lr=self.args.actor_lr)
        self.bc_optimizer.load_state_dict(copy.deepcopy(self.actor_optimizer.state_dict()))
        self.progress['bc_control_initial_steps'] = self.progress['bc_steps']
        self.log('bc_control_initialized', initial_steps=self.progress['bc_steps'],
                 weights_and_optimizer_identical=True)

    def calibrate_bc(self):
        import torch
        if not len(self.buffer):
            raise RuntimeError('BC calibration needs training warmup transitions')
        self.actor.train()
        while self.progress['bc_steps'] < self.args.bc_steps:
            token, reference, _, _, _, _, _, prop, _ = self.buffer.sample_balanced(
                self.args.batch_size, len(self.args.tasks), self.device, task_ids=self.args.tasks)
            mean = self.actor._get_mean(token, reference, prop, apply_dropout=False)
            loss = (mean-reference.clamp(-1, 1)).square().sum(dim=(-2, -1)).mean()
            grad = self.backward_step(loss, self.actor, self.actor_optimizer)
            self.progress['bc_steps'] += 1
            self.log('bc_update', step=self.progress['bc_steps'], bc_loss=float(loss.detach()), grad_norm=grad)
        self.initialize_control()
        torch.save(self.actor.state_dict(), self.output / 'bc_initial_actor.pt')
        self.target_actor.load_state_dict(self.actor.state_dict())

    def learn(self):
        if len(self.buffer) < 2 or self.bc_actor is None or self.bc_optimizer is None:
            raise RuntimeError('TD3 learning needs replay and an initialized matched BC control')
        self.actor.train()
        self.bc_actor.train().requires_grad_(True)
        self.critic.train()
        effective_batch = min(self.args.batch_size, len(self.buffer))
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
                    actor_loss, control_loss, actor_metrics = paired_actor_losses(
                        self.actor, self.bc_actor, self.critic, self.buffer, batch_size=effective_batch,
                        beta=self.args.beta, device=self.device, task_ids=self.args.tasks)
                    actor_metrics['actor_grad_norm'] = self.backward_step(actor_loss, self.actor, self.actor_optimizer)
                    actor_metrics['bc_control_grad_norm'] = self.backward_step(control_loss, self.bc_actor, self.bc_optimizer)
                finally:
                    self.critic.requires_grad_(True)
                metrics.update(actor_metrics)
                self.progress['actor_steps'] += 1
                self.progress['bc_control_steps'] += 1
                self.soft_update(self.actor, self.target_actor, self.args.tau)
                self.soft_update(self.critic, self.target_critic, self.args.tau)
            self.log('update', **self.progress, **metrics, effective_batch=effective_batch,
                     replay_size=len(self.buffer), task_counts=self.buffer.task_counts())

    def save(self):
        import torch
        if self.progress['actor_steps'] != self.progress['bc_control_steps']:
            raise RuntimeError('Refusing checkpoint: TD3 and BC update counts differ')
        payload = dict(config=self.config, progress=self.progress, rng=self.rng_state(),
                       actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                       target_actor=self.target_actor.state_dict(), target_critic=self.target_critic.state_dict(),
                       actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                       bc_actor=self.bc_actor.state_dict() if self.bc_actor is not None else None,
                       bc_optimizer=self.bc_optimizer.state_dict() if self.bc_optimizer is not None else None,
                       v2_code_sha256=self.v2_fingerprint,
                       replay=dict(capacity=self.buffer.capacity, buffer=self.buffer.buffer,
                                   pos=self.buffer.pos, task_index=dict(self.buffer._task_index)))
        path = self.output / 'checkpoint.pt.tmp'
        torch.save(payload, path)
        path.replace(self.output / 'checkpoint.pt')
        for name, model in [('actor.pt', self.actor), ('bc_only_actor.pt', self.bc_actor)]:
            if model is not None:
                path = self.output / (name+'.tmp')
                torch.save(model.state_dict(), path)
                path.replace(self.output / name)
        write_json(self.output / 'progress.json', self.progress)

    def restore(self):
        import torch
        payload = torch.load(self.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if payload.get('v2_code_sha256') != self.v2_fingerprint:
            raise ValueError('Resume refused: checkpoint source SHA-256 differs')
        if payload['progress'].get('actor_steps') != payload['progress'].get('bc_control_steps'):
            raise ValueError('Checkpoint has unmatched TD3 and BC update counts')
        super().restore()
        if payload['bc_actor'] is not None:
            self.bc_actor = copy.deepcopy(self.actor).requires_grad_(True)
            self.bc_actor.load_state_dict(payload['bc_actor'], strict=True)
            self.bc_optimizer = torch.optim.Adam(self.bc_actor.parameters(), lr=self.args.actor_lr)
            self.bc_optimizer.load_state_dict(payload['bc_optimizer'])
        else:
            self.bc_actor = None
            self.bc_optimizer = None
        self.set_rng(payload['rng'])

    def run(self):
        super().run()
        report = json.loads((self.output / 'run_summary.json').read_text())
        report.update(runner_version=2, bc_control_update_count_matches=self.progress['actor_steps']==self.progress['bc_control_steps'],
                      bc_control_initial_count_matches=self.progress['bc_steps']==self.progress['bc_control_initial_steps'],
                      bc_control_additional_updates=self.progress['bc_control_steps'],
                      algorithm_changes='bounded_Q_actions_and_clamped_BC_targets',
                      bc_control='matched_data_noise_initial_optimizer_and_update_count')
        if not report['bc_control_update_count_matches'] or not report['bc_control_initial_count_matches']:
            raise RuntimeError('Matched BC control audit failed')
        write_json(self.output / 'run_summary.json', report)
        self.log('v2_complete', **report)


def main():
    args = arguments()
    experiment = ExperimentV2(args)
    from persistent_libero import install_persistent_env
    close_final = install_persistent_env(experiment)
    try:
        experiment.run()
    except BaseException as exc:
        experiment.log('failed', error=repr(exc), progress=experiment.progress)
        raise
    finally:
        close_final()


if __name__ == '__main__':
    main()
