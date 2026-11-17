"""Custom bounded-residual RLT actor, distinct from released RLT_a.

Tensor keys/shapes match ActionTokenActor so its raw network can be loaded by
existing frozen loaders. Applying the residual formula is mandatory at inference.
"""
import math

import torch
from torch import nn


def residual_action(raw_actor, token, reference, prop=None, delta_max=.1):
    """Wrap a standard ActionTokenActor loaded with RAW residual weights."""
    if not math.isfinite(delta_max) or delta_max <= 0:
        raise ValueError('delta_max must be finite and positive')
    raw = raw_actor._get_mean(token, reference, prop, apply_dropout=False)
    return reference + delta_max*torch.tanh(raw)


class BoundedResidualActor(nn.Module):
    def __init__(self, bottleneck_dim=256, action_dim=7, chunk_len=8, hidden_dim=512,
                 prop_dim=8, delta_max=.1):
        super().__init__()
        if not math.isfinite(delta_max) or delta_max <= 0:
            raise ValueError('delta_max must be finite and positive')
        self.action_dim, self.chunk_len, self.prop_dim = action_dim, chunk_len, prop_dim
        self.delta_max, self.ref_dropout = float(delta_max), 0.
        self.net = nn.Sequential(nn.Linear(bottleneck_dim+prop_dim+chunk_len*action_dim,hidden_dim),
                                 nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                 nn.Linear(hidden_dim,chunk_len*action_dim))
        # Compatibility buffer; this policy does not sample action noise.
        self.register_buffer('fixed_std',torch.tensor(0.))
        self.zero_head()

    def zero_head(self):
        with torch.no_grad():
            self.net[-1].weight.zero_();self.net[-1].bias.zero_();self.fixed_std.zero_()

    def _get_mean(self,rl_token,vla_action,prop_state=None,apply_dropout=False):
        """Raw residual preactivation; signature matches the original actor."""
        if apply_dropout:
            raise ValueError('Reference dropout is disabled for bounded residual training')
        batch=rl_token.shape[0]
        z=rl_token.squeeze(1) if rl_token.ndim==3 else rl_token
        pieces=[z]
        if self.prop_dim:
            if prop_state is None:
                prop_state=torch.zeros(batch,self.prop_dim,dtype=z.dtype,device=z.device)
            pieces.append(prop_state)
        pieces.append(vla_action.reshape(batch,-1))
        return self.net(torch.cat(pieces,dim=-1)).reshape(batch,self.chunk_len,self.action_dim)

    def correction(self,token,reference,prop=None):
        return self.delta_max*torch.tanh(self._get_mean(token,reference,prop))

    def forward(self,rl_token,vla_action,prop_state=None,deterministic=True):
        # Both training and inference use the deterministic mean. TD3 target
        # smoothing, if desired, is added externally after this function.
        return vla_action+self.correction(rl_token,vla_action,prop_state),None
