import unittest
import torch
from pi05_direct_networks import DirectActor,DirectCritic,DirectTD3,td_target,module

class DirectTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.z=torch.randn(8,256);self.s=torch.randn(8,8);self.ref=torch.randn(8,5,7)
    def test_published_actor_and_no_residual_path(self):
        a=DirectActor();self.assertIsInstance(a.model,module.ActionTokenActor)
        with torch.no_grad():
            for p in a.parameters():p.zero_()
        out=a(self.z,self.s,self.ref)
        self.assertEqual(tuple(out.shape),(8,5,7));self.assertEqual(float(out.abs().sum()),0.)
        self.assertFalse(torch.equal(out,self.ref))
    def test_direct_output_can_differ_without_residual_bound(self):
        a=DirectActor()
        with torch.no_grad():
            for p in a.parameters():p.zero_()
            a.model.net[-1].bias.fill_(.7)
        torch.testing.assert_close(a(self.z,self.s,torch.zeros_like(self.ref)),torch.full_like(self.ref,.7))
    def test_td_terminal_duration_and_minimum(self):
        y=td_target(torch.tensor([5.,0.]),torch.tensor([2,5]),torch.tensor([True,False]),torch.tensor([10.,2.]),torch.tensor([9.,3.]))
        torch.testing.assert_close(y,torch.tensor([5.,2*.99**5]))
    def test_delayed_updates_and_actual_bc_learning(self):
        a=DirectActor();td=DirectTD3(a,DirectCritic(a));before={k:v.clone() for k,v in a.state_dict().items()}
        b=dict(z=self.z,state=self.s,ref=self.ref,action=self.ref,next_z=self.z,next_state=self.s,next_ref=self.ref,reward=torch.ones(8),duration=torch.full((8,),5),terminal=torch.zeros(8,dtype=torch.bool))
        td.update(b);self.assertEqual(td.actor_updates,0)
        td.update(b);self.assertEqual(td.actor_updates,1)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in a.state_dict().items()))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in td.bc.state_dict().items()))
        self.assertFalse(any(p.grad is not None for p in td.target_actor.parameters()))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(td.critic.model.q1.parameters(),td.critic.model.q2.parameters())))
    def test_exact_published_forward_after_preprocessing(self):
        a=DirectActor();a.eval()
        expected,_=a.model(self.z,self.ref,prop_state=self.s,deterministic=True)
        torch.testing.assert_close(a(self.z,self.s,self.ref),expected)
if __name__=='__main__':unittest.main()
