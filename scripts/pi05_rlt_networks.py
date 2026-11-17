"""Compact experimental pi-prefix RLT and bounded residual TD3 networks (CPU)."""
import copy
import torch
from torch import nn


class TokenAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(2048))
        self.register_buffer('std', torch.ones(2048))
        self.project = nn.Linear(2048, 128)
        self.cls = nn.Parameter(torch.zeros(1, 1, 128))
        self.position = nn.Parameter(torch.randn(1, 34, 128) * .02)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0., batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.to_token = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256))
        self.expand = nn.Sequential(nn.Linear(256, 33 * 128), nn.GELU())
        self.reconstruct = nn.Linear(128, 2048)

    def normalize(self, x):
        return (x.float() - self.mean) / self.std

    def encode(self, x):
        h = self.project(self.normalize(x))
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1) + self.position
        return self.to_token(self.transformer(h)[:, 0])

    def forward(self, x):
        z = self.encode(x)
        return self.reconstruct(self.expand(z).reshape(-1, 33, 128))


class StateModule(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer('state_mean', torch.zeros(8) if mean is None else mean.clone())
        self.register_buffer('state_std', torch.ones(8) if std is None else std.clone())

    def state_input(self, state):
        return ((state - self.state_mean) / self.state_std).clamp(-10., 10.)


class ResidualActor(StateModule):
    def __init__(self, mean=None, std=None, bound=.1):
        super().__init__(mean, std)
        self.bound = float(bound)
        self.net = nn.Sequential(nn.Linear(299, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 35))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correction(self, z, state, reference):
        ref = reference.reshape(-1, 35)
        residual = self.bound * torch.tanh(self.net(torch.cat([z, self.state_input(state), ref], dim=-1)))
        return residual.reshape(-1, 5, 7)

    def forward(self, z, state, reference):
        return reference.reshape(-1, 5, 7) + self.correction(z, state, reference)

class DirectActor(StateModule):
    """RLT_a-style actor: directly predicts the complete action chunk."""
    is_residual = False
    def __init__(self, mean=None, std=None):
        super().__init__(mean, std)
        self.bound = 1.0
        self.net = nn.Sequential(nn.Linear(299,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,35))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, z, state, reference):
        h=torch.cat([z,self.state_input(state),reference.reshape(-1,35)],dim=-1)
        return torch.tanh(self.net(h)).reshape(-1,5,7)


class TwinCritic(StateModule):
    def __init__(self, mean=None, std=None):
        super().__init__(mean, std)
        self.q1 = nn.Sequential(nn.Linear(299, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = copy.deepcopy(self.q1)
        # Independent initialization; identical critics would defeat clipped double Q.
        for m in self.q2.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, z, state, action):
        h = torch.cat([z, self.state_input(state), action.reshape(-1, 35)], dim=-1)
        return self.q1(h).squeeze(-1), self.q2(h).squeeze(-1)


def td_target(reward, duration, terminal, next_q1, next_q2, gamma=.99):
    return reward + (~terminal).float() * gamma ** duration * torch.minimum(next_q1, next_q2)


@torch.no_grad()
def soft_update(target, source, tau=.005):
    for a, b in zip(target.parameters(), source.parameters(), strict=True):
        a.lerp_(b, tau)


class TD3:
    def __init__(self, actor, critic, seed=42):
        self.actor, self.critic = actor, critic
        self.bc = copy.deepcopy(actor)
        self.target_actor, self.target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
        self.bc_opt = torch.optim.Adam(self.bc.parameters(), lr=1e-4)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.updates = self.actor_updates = 0

    def update(self, b):
        self.updates += 1
        with torch.no_grad():
            a = self.target_actor(b['next_z'], b['next_state'], b['next_ref'])
            noise = torch.randn(a.shape, generator=self.generator) * .02
            if getattr(self.actor, 'is_residual', True):
                a = b['next_ref'] + (a + noise.clamp(-.05, .05) - b['next_ref']).clamp(-self.actor.bound, self.actor.bound)
            else:
                a = (a + noise.clamp(-.05, .05)).clamp(-1., 1.)
            q1, q2 = self.target_critic(b['next_z'], b['next_state'], a)
            y = td_target(b['reward'], b['duration'], b['terminal'], q1, q2)
        q1, q2 = self.critic(b['z'], b['state'], b['action'])
        loss = (q1 - y).square().mean() + (q2 - y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite critic loss')
        self.critic_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.critic.parameters(), 1.);self.critic_opt.step()
        result = dict(critic_loss=float(loss.detach()), target_mean=float(y.mean()), update=self.updates)
        if self.updates % 2 == 0:
            self.critic.requires_grad_(False)
            try:
                action = self.actor(b['z'], b['state'], b['ref'])
                q, _ = self.critic(b['z'], b['state'], action)
                penalty = (action - b['ref']).square().flatten(1).sum(1).mean()
                actor_loss = -q.mean() + 3. * penalty
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError('Nonfinite actor loss')
                self.actor_opt.zero_grad();actor_loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(), 1.);self.actor_opt.step()
            finally:
                self.critic.requires_grad_(True)
            bc_action = self.bc(b['z'], b['state'], b['ref'])
            bc_loss = (bc_action - b['ref']).square().flatten(1).sum(1).mean()
            self.bc_opt.zero_grad();bc_loss.backward();self.bc_opt.step()
            self.actor_updates += 1
            soft_update(self.target_actor, self.actor);soft_update(self.target_critic, self.critic)
            result.update(actor_loss=float(actor_loss.detach()), q_actor=float(q.mean().detach()),
                          bc_penalty=float(penalty.detach()), bc_only_loss=float(bc_loss.detach()), actor_update=self.actor_updates)
        return result
