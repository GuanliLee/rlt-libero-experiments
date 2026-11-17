#!/usr/bin/env python3
"""Fixed-budget direct TD3 on existing TRAINING replay, with longer credit horizon."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from pi05_direct_identity import IdentityDirectActor
from pi05_direct_networks import DirectCritic
from pi05_direct_longhorizon import LongHorizonTD3
from train_pi05_moka_direct import Replay
from evaluate_pi05_baseline import write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1280);p.add_argument('--gamma',type=float,default=.999);a=p.parse_args()
    if a.output.exists() or not 0<a.gamma<1:p.error('New output and valid gamma required')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(42);start=time.monotonic()
    initial=torch.load(a.initial,weights_only=True,map_location='cpu')
    actor=IdentityDirectActor();actor.load_state_dict(initial['actor']);critic=DirectCritic(actor).double();critic.load_state_dict(initial['critic'])
    td=LongHorizonTD3(actor,critic,gamma=a.gamma)
    with np.load(a.replay) as f:data={k:f[k].copy() for k in f.files}
    positive=data['reward']>0
    assert np.all(data['terminal'][positive])
    np.testing.assert_allclose(data['reward'][positive],5*.99**(data['duration'][positive]-1),atol=1e-5)
    data['reward'][positive]=(5*a.gamma**(data['duration'][positive]-1)).astype(np.float32)
    ret=0.
    for i in reversed(range(len(data['reward']))):
        if data['terminal'][i]:ret=0.
        ret=float(data['reward'][i])+a.gamma**int(data['duration'][i])*ret;data['mc'][i]=ret
    replay=Replay();replay.rows=[{k:v[i] for k,v in data.items()} for i in range(len(data['reward']))]
    protocol=dict(gamma=a.gamma,updates=a.updates,critic_mc_updates=500,online_collection_episodes=0,reference_dropout=0.,initial=str(a.initial),replay=str(a.replay),
        initial_sha256=hashlib.sha256(a.initial.read_bytes()).hexdigest(),replay_sha256=hashlib.sha256(a.replay.read_bytes()).hexdigest(),replay_rows=len(replay.rows),
        action_form='Published direct MLP, exact imitation initialization, no residual',note='Training replay only; no validation/test transitions. Credit-horizon optimization trial, not an isolated gamma ablation due replay availability differences.')
    write(a.output/'protocol.json',protocol)
    snapshot=a.output/'source_snapshot';snapshot.mkdir()
    for filename in ['fit_pi05_direct_offline.py','pi05_direct_longhorizon.py','pi05_direct_identity.py','pi05_direct_networks.py']:
        source=Path(__file__).parent/filename;(snapshot/filename).write_bytes(source.read_bytes())
    for step in range(500):
        b=replay.batch();q1,q2=td.critic(b['z'],b['state'],b['action']);loss=(q1-b['mc']).square().mean()+(q2-b['mc']).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MC loss')
        td.critic_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(td.critic.parameters(),1.);td.critic_opt.step()
    td.target_critic.load_state_dict(td.critic.state_dict())
    with (a.output/'training_losses.jsonl').open('w') as f:
        for step in range(a.updates):
            stats=td.update(replay.batch());f.write(json.dumps(stats)+'\n')
            if (step+1)%256==0:print(json.dumps(stats),flush=True)
    assert all(torch.equal(v,initial['actor'][k]) for k,v in td.bc.state_dict().items())
    state=dict(actor=td.actor.state_dict(),critic=td.critic.state_dict(),bc=td.bc.state_dict(),encoder=initial['encoder'],target_actor=td.target_actor.state_dict(),target_critic=td.target_critic.state_dict(),
        actor_optimizer=td.actor_opt.state_dict(),critic_optimizer=td.critic_opt.state_dict(),episode=0,offline_updates=td.updates,actor_updates=td.actor_updates,
        architecture='published direct identity-initialized ActionTokenActor; local pooled pi encoder; offline long-horizon TD3',actor_dtype='float64',reference_dropout=0.,gamma=a.gamma)
    torch.save(state,a.output/'policy.pt')
    write(a.output/'summary.json',dict(status='trained_not_evaluated',actor_updates=td.actor_updates,critic_td_updates=td.updates,critic_mc_updates=500,gamma=a.gamma,wall_seconds=time.monotonic()-start))
if __name__=='__main__':main()
