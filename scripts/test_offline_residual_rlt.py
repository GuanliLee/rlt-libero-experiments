#!/usr/bin/env python3
"""CPU checks for exact identity, correction bounds, learning and raw loading."""
import copy
import unittest

import torch
from torch import nn

import offline_rlt_mc_candidate as mc
import offline_residual_rlt as training
from optimize_rlt import ROOT
from residual_actor import BoundedResidualActor,residual_action

torch.set_num_threads(2)


class Replay:
    def __init__(self,batch):self.batch=batch;self.calls=0
    def sample_balanced(self,*args,**kwargs):self.calls+=1;return self.batch


class LinearQ(nn.Module):
    def q1_forward(self,z,action,prop):return action.sum((-1,-2))


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.actor=BoundedResidualActor()
        self.z=torch.randn(4,1,256)
        self.ref=torch.randn(4,8,7)*1.5
        self.prop=torch.randn(4,8)

    def test_zero_head_is_exact_reference_for_varied_inputs(self):
        for scale in [.001,1.,1000.]:
            z=self.z*scale;ref=self.ref*scale;prop=self.prop*scale
            for train in [True,False]:
                self.actor.train(train)
                actual,_=self.actor(z,ref,prop,deterministic=not train)
                self.assertTrue(torch.equal(actual,ref))
                self.assertTrue(torch.equal(actual.clamp(-1,1),ref.clamp(-1,1)))

    def test_correction_and_executed_difference_are_bounded(self):
        with torch.no_grad():self.actor.net[-1].weight.normal_(0,3);self.actor.net[-1].bias.fill_(3.)
        correction=self.actor.correction(self.z,self.ref,self.prop)
        action,_=self.actor(self.z,self.ref,self.prop)
        self.assertLessEqual(float(correction.abs().max()),.10000001)
        self.assertLessEqual(float((action.clamp(-1,1)-self.ref.clamp(-1,1)).abs().max()),.100001)

    def test_td3_actor_changes_and_zero_q_control_stays_exactly_fixed(self):
        control=copy.deepcopy(self.actor)
        initial={k:v.clone()for k,v in control.state_dict().items()}
        ao=torch.optim.Adam(self.actor.parameters(),lr=1e-3)
        bo=torch.optim.Adam(control.parameters(),lr=1e-3)
        q=LinearQ()
        batch=(self.z,self.ref,self.ref,torch.zeros(4),self.z,self.ref,torch.zeros(4),self.prop,self.prop)
        replay=Replay(batch)
        for _ in range(4):
            loss,bcloss,metrics=training.paired_residual_losses(self.actor,control,q,replay,
                batch_size=4,beta=3.,device='cpu',task_ids=[0])
            ao.zero_grad();loss.backward();ao.step()
            bo.zero_grad();bcloss.backward();bo.step()
            self.assertEqual(float(bcloss),0.)
            self.assertEqual(metrics['control_max_abs_correction'],0.)
            self.assertTrue(mc.same_state(control.state_dict(),initial))
            actual,_=control(self.z,self.ref,self.prop)
            self.assertTrue(torch.equal(actual,self.ref))
        self.assertFalse(mc.same_state(self.actor.state_dict(),initial))
        self.assertGreater(float(self.actor.net[-1].weight.abs().sum()),0.)
        self.assertEqual(replay.calls,4)

    def test_original_actor_load_is_strict_and_wrapper_matches(self):
        ac=mc.load_module(ROOT/'src/AlphaBrain-opt/AlphaBrain/training/reinforcement_learning/algos/RLT_a/action_token_actor_critic.py','residual_test_original_ac')
        raw=ac.ActionTokenActor(bottleneck_dim=256,action_dim=7,chunk_len=8,hidden_dim=512,prop_dim=8)
        with torch.no_grad():self.actor.net[-1].weight.normal_(0,.01)
        raw.load_state_dict(self.actor.state_dict(),strict=True)
        expected,_=self.actor(self.z,self.ref,self.prop)
        actual=residual_action(raw,self.z,self.ref,self.prop,delta_max=.1)
        self.assertTrue(torch.equal(actual,expected))
        # Loading succeeds, but using the raw full-action forward is incorrect.
        incorrect,_=raw(self.z,self.ref,self.prop,deterministic=True)
        self.assertFalse(torch.equal(incorrect,expected))

    def test_dropout_and_invalid_bounds_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'dropout'):
            self.actor._get_mean(self.z,self.ref,self.prop,apply_dropout=True)
        for delta in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):BoundedResidualActor(delta_max=delta)


if __name__=='__main__':unittest.main(verbosity=2)
