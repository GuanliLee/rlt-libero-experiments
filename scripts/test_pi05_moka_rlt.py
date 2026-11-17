import unittest
import numpy as np
import torch
from pi05_rlt_networks import TokenAutoencoder,ResidualActor,TwinCritic,TD3,td_target
from train_pi05_moka_rlt import Replay


class RLTChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)

    def test_zero_correction_preserves_float64_wire_actions(self):
        a=ResidualActor();z=torch.randn(4,256);s=torch.randn(4,8)
        raw=np.random.default_rng(0).normal(size=(4,5,7))
        ref=torch.from_numpy(raw.astype(np.float32))
        delta=a.correction(z,s,ref).detach().numpy()
        np.testing.assert_array_equal(raw+delta,raw)
        with torch.no_grad():a.net[-1].bias.fill_(100.)
        self.assertLessEqual(float(a.correction(z,s,ref).abs().max().detach()),.100001)

    def test_terminal_target_and_real_block_duration(self):
        y=td_target(torch.tensor([5.,0.]),torch.tensor([3,5]),torch.tensor([True,False]),torch.tensor([100.,2.]),torch.tensor([90.,3.]))
        torch.testing.assert_close(y,torch.tensor([5.,2*.99**5]))

    def test_actor_delay_bc_control_and_target_gradients(self):
        a=ResidualActor();td=TD3(a,TwinCritic());before={k:v.clone() for k,v in a.state_dict().items()}
        z=torch.randn(8,256);s=torch.randn(8,8);ref=torch.randn(8,5,7)
        b=dict(z=z,state=s,ref=ref,action=ref,next_z=z,next_state=s,next_ref=ref,reward=torch.ones(8),duration=torch.full((8,),5),terminal=torch.zeros(8,dtype=torch.bool))
        td.update(b);self.assertEqual(td.actor_updates,0)
        td.update(b);self.assertEqual(td.actor_updates,1)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in a.state_dict().items()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in td.bc.state_dict().items()))
        self.assertFalse(any(p.grad is not None for p in td.target_actor.parameters()))

    def test_replay_episode_boundaries_returns_and_dtypes(self):
        r=Replay();z=np.ones((2,256));s=np.ones((2,8));ref=np.ones((2,5,7),dtype=np.float64)
        r.add_episode(z,s,ref,ref,[0.,5.],[5,3]);r.add_episode(z*2,s*2,ref*2,ref*2,[0.,0.],[5,5])
        self.assertAlmostEqual(float(r.rows[0]['mc']),5*.99**5,places=5)
        self.assertTrue(r.rows[1]['terminal']);self.assertTrue(np.all(r.rows[1]['next_z']==0))
        self.assertFalse(r.rows[2]['terminal']);self.assertEqual(r.batch()['ref'].dtype,torch.float32)

    def test_encoder_reconstruction_gradient_and_frozen_encoding(self):
        m=TokenAutoencoder();x=torch.randn(2,33,2048)
        loss=(m(x)-x).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(m.project.weight.grad).all())
        self.assertGreater(float(m.project.weight.grad.abs().sum()),0.)
        m.eval().requires_grad_(False)
        with torch.no_grad():z=m.encode(x)
        self.assertEqual(tuple(z.shape),(2,256));self.assertFalse(z.requires_grad)


if __name__=='__main__':unittest.main()
