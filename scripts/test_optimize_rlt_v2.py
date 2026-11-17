#!/usr/bin/env python3
"""CPU regression checks for the v2 action boundary and paired BC experiment."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

import optimize_rlt as legacy
import optimize_rlt_v2 as v2

torch.set_num_threads(2)
legacy.torch, legacy.np = torch, np


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 14)
        self.ref_dropout = .5
        self.register_buffer('fixed_std', torch.tensor(.05))

    def forward(self, token, reference, prop, deterministic=False):
        mask = (torch.rand(token.shape[0], 1) > self.ref_dropout).float()
        action = self.net(token*mask).reshape(-1, 2, 7)
        if not deterministic:
            action = action + torch.randn_like(action)*self.fixed_std
        return action, None


class TinyCritic(nn.Module):
    def __init__(self, flat=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0. if flat else 1.))
        self.last_action = None

    def q1_forward(self, token, action, prop):
        self.last_action = action.detach().clone()
        return action.sum((-1, -2))*self.scale


class Replay:
    def __init__(self):
        token = torch.tensor([[.3, -.2, .5], [.8, .4, -.9]])
        ref = torch.tensor([2., -2., 0., .5, 0., .8, 1.2]).repeat(2, 2, 1)
        prop = torch.zeros(2, 8)
        self.batch = (token, ref, ref, torch.zeros(2), token, ref, torch.zeros(2), prop, prop)
        self.calls = 0
        self.capacity, self.pos = 10, 2
        self.buffer = [tuple(x[i].clone() for x in self.batch)+(i,) for i in range(2)]
        self._task_index = {0: [0], 1: [1]}

    def sample_balanced(self, *args, **kwargs):
        self.calls += 1
        return self.batch

    def __len__(self):
        return len(self.buffer)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a)==len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a==b


class V2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)

    def test_q_sees_only_executable_range_and_bc_clamps_reference(self):
        a = TinyActor()
        with torch.no_grad():
            a.net.weight.zero_()
            a.net.bias.fill_(1.5)
        b, q, replay = copy.deepcopy(a), TinyCritic(), Replay()
        loss, control, metrics = v2.paired_actor_losses(a, b, q, replay, batch_size=2,
            beta=3., device='cpu', task_ids=[0, 1])
        self.assertEqual(replay.calls, 1)
        self.assertLessEqual(float(q.last_action.abs().max()), 1.)
        self.assertEqual(metrics['bc_reference_max_abs'], 1.)
        self.assertEqual(metrics['bc_penalty'], metrics['bc_control_penalty'])
        # Saturated Q has zero action gradient; raw-action BC still pulls back.
        loss.backward()
        self.assertGreater(float(a.net.bias.grad.mean()), 0.)
        self.assertAlmostEqual(float(control), 3*metrics['bc_penalty'], places=5)

    def test_matched_data_noise_updates_identical_when_q_is_constant(self):
        a, q, replay = TinyActor(), TinyCritic(flat=True), Replay()
        b = copy.deepcopy(a)
        ao, bo = torch.optim.Adam(a.parameters(), lr=1e-3), torch.optim.Adam(b.parameters(), lr=1e-3)
        for _ in range(4):
            loss, control, metrics = v2.paired_actor_losses(a, b, q, replay,
                batch_size=2, beta=3., device='cpu', task_ids=[0, 1])
            self.assertAlmostEqual(float(loss), float(control), places=6)
            ao.zero_grad(); loss.backward(); ao.step()
            bo.zero_grad(); control.backward(); bo.step()
        self.assertTrue(equal(a.state_dict(), b.state_dict()))
        self.assertTrue(equal(ao.state_dict(), bo.state_dict()))
        self.assertEqual(replay.calls, 4)

    def test_control_does_not_advance_policy_rng(self):
        a, q, replay = TinyActor(), TinyCritic(), Replay()
        b = copy.deepcopy(a)
        start = torch.get_rng_state()
        a(replay.batch[0], replay.batch[1], replay.batch[7], deterministic=False)
        expected_after = torch.get_rng_state()
        torch.set_rng_state(start)
        v2.paired_actor_losses(a, b, q, replay, batch_size=2, beta=3., device='cpu', task_ids=[0, 1])
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_after))

    def fixture(self, folder):
        e = v2.ExperimentV2.__new__(v2.ExperimentV2)
        e.output = Path(folder)
        e.args = argparse.Namespace(actor_lr=1e-3)
        e.actor, e.critic = TinyActor(), TinyCritic()
        e.target_actor, e.target_critic = copy.deepcopy(e.actor), copy.deepcopy(e.critic)
        e.actor_optimizer = torch.optim.Adam(e.actor.parameters(), lr=1e-3)
        e.critic_optimizer = torch.optim.Adam(e.critic.parameters(), lr=1e-3)
        e.release_actor = copy.deepcopy(e.actor)
        e.bc_actor = copy.deepcopy(e.actor)
        e.bc_optimizer = torch.optim.Adam(e.bc_actor.parameters(), lr=1e-3)
        e.config, e.v2_fingerprint = {'runner_version': 2}, {'fixture': 'fixedhash'}
        e.progress = dict(actor_steps=0, bc_control_steps=0, bc_steps=0, bc_control_initial_steps=0)
        e.buffer = Replay()
        e.log = lambda *a, **kw: None
        return e

    def test_checkpoint_roundtrip_preserves_control_and_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            e = self.fixture(folder)
            loss, control, _ = v2.paired_actor_losses(e.actor, e.bc_actor, e.critic, e.buffer,
                batch_size=2, beta=3., device='cpu', task_ids=[0, 1])
            e.actor_optimizer.zero_grad(); loss.backward(); e.actor_optimizer.step()
            e.bc_optimizer.zero_grad(); control.backward(); e.bc_optimizer.step()
            e.progress['actor_steps'] = e.progress['bc_control_steps'] = 1
            e.save()
            expected_actor = copy.deepcopy(e.bc_actor.state_dict())
            expected_optimizer = copy.deepcopy(e.bc_optimizer.state_dict())
            with torch.no_grad():
                for p in e.bc_actor.parameters(): p.zero_()
            e.restore()
            self.assertTrue(equal(e.bc_actor.state_dict(), expected_actor))
            self.assertTrue(equal(e.bc_optimizer.state_dict(), expected_optimizer))
            self.assertEqual(e.progress['actor_steps'], e.progress['bc_control_steps'])
            e.v2_fingerprint = {'fixture': 'changed'}
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                e.restore()

    def test_checkpoint_rejects_unmatched_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            e = self.fixture(folder)
            e.progress['actor_steps'] = 1
            with self.assertRaisesRegex(RuntimeError, 'counts differ'):
                e.save()


if __name__ == '__main__':
    unittest.main(verbosity=2)
