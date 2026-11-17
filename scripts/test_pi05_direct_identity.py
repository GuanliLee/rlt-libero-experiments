import unittest
import numpy as np
import torch
from pi05_direct_identity import IdentityDirectActor
from pi05_direct_networks import DirectActor,DirectCritic,DirectTD3

class IdentityChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.z=torch.randn(16,256);self.s=torch.randn(16,8)
        self.ref=torch.from_numpy(np.random.default_rng(42).normal(size=(16,5,7)))
    def test_exact_float64_imitation_and_no_residual_connection(self):
        a=IdentityDirectActor()
        with torch.no_grad():self.assertTrue(torch.equal(a(self.z,self.s,self.ref),self.ref))
        self.assertTrue(all(p.requires_grad for p in a.parameters()))
        with torch.no_grad():
            for p in a.parameters():p.zero_()
            self.assertTrue(torch.equal(a(self.z,self.s,self.ref),torch.zeros_like(self.ref)))
    def test_q_changes_actor_while_bc_remains_identity(self):
        a=IdentityDirectActor();td=DirectTD3(a,DirectCritic(a).double());before={k:v.clone() for k,v in a.state_dict().items()}
        b=dict(z=self.z,state=self.s,ref=self.ref.float(),action=self.ref.float(),next_z=self.z,next_state=self.s,next_ref=self.ref.float(),reward=torch.ones(16),duration=torch.full((16,),5),terminal=torch.zeros(16,dtype=torch.bool))
        td.update(b);td.update(b)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in a.state_dict().items()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in td.bc.state_dict().items()))
        self.assertFalse(any(p.grad is not None for p in td.target_actor.parameters()))
    def test_generic_frozen_loader_matches_identity_checkpoint(self):
        trained=IdentityDirectActor();loaded=DirectActor().double();loaded.load_state_dict(trained.state_dict())
        with torch.no_grad():self.assertTrue(torch.equal(loaded(self.z,self.s,self.ref),self.ref))
if __name__=='__main__':unittest.main()
