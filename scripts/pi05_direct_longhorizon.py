"""TD3 with an explicit per-control-step discount; direct action output."""
import copy
import torch
from torch import nn
from pi05_rlt_networks import soft_update,td_target

class LongHorizonTD3:
    def __init__(self,actor,critic,seed=42,gamma=.999):
        self.gamma=gamma
        self.actor,self.critic=actor,critic
        self.bc=copy.deepcopy(actor)
        self.target_actor,self.target_critic=copy.deepcopy(actor),copy.deepcopy(critic)
        self.actor_opt=torch.optim.Adam(actor.parameters(),lr=1e-5)
        self.critic_opt=torch.optim.Adam(critic.parameters(),lr=3e-4)
        self.bc_opt=torch.optim.Adam(self.bc.parameters(),lr=1e-5)
        self.generator=torch.Generator().manual_seed(seed)
        self.updates=self.actor_updates=0

    def update(self,b):
        self.updates+=1
        with torch.no_grad():
            a=self.target_actor(b['next_z'],b['next_state'],b['next_ref'])
            # TD3 smoothing around the predicted FULL action, no VLA-relative clamp.
            a=a+(torch.randn(a.shape,generator=self.generator)*.02).clamp(-.05,.05)
            q1,q2=self.target_critic(b['next_z'],b['next_state'],a)
            y=td_target(b['reward'],b['duration'],b['terminal'],q1,q2,gamma=self.gamma)
        q1,q2=self.critic(b['z'],b['state'],b['action'])
        loss=(q1-y).square().mean()+(q2-y).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite critic loss')
        self.critic_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.critic.parameters(),1.);self.critic_opt.step()
        result=dict(update=self.updates,critic_loss=float(loss.detach()),target_mean=float(y.mean()))
        if self.updates%2==0:
            # Give RL and BC-only the same reference-dropout mask.
            rng=torch.get_rng_state()
            action=self.actor(b['z'],b['state'],b['ref'],dropout=True)
            self.critic.requires_grad_(False)
            try:
                q,_=self.critic(b['z'],b['state'],action)
                penalty=(action-b['ref']).square().flatten(1).sum(1).mean()
                actor_loss=-q.mean()+3.*penalty
                if not torch.isfinite(actor_loss):raise FloatingPointError('Nonfinite actor loss')
                self.actor_opt.zero_grad();actor_loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1.);self.actor_opt.step()
            finally:self.critic.requires_grad_(True)
            torch.set_rng_state(rng)
            bc_action=self.bc(b['z'],b['state'],b['ref'],dropout=True)
            bc_loss=3.*(bc_action-b['ref']).square().flatten(1).sum(1).mean()
            self.bc_opt.zero_grad();bc_loss.backward();nn.utils.clip_grad_norm_(self.bc.parameters(),1.);self.bc_opt.step()
            self.actor_updates+=1
            soft_update(self.target_actor,self.actor);soft_update(self.target_critic,self.critic)
            result.update(actor_loss=float(actor_loss.detach()),q_actor=float(q.mean().detach()),bc_penalty=float(penalty.detach()),bc_only_loss=float(bc_loss.detach()),actor_update=self.actor_updates)
        return result
