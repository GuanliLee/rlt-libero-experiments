#!/usr/bin/env python3
"""Direct published action-chunk Actor with local frozen pi features; TD3+BC."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from pi05_direct_networks import TokenAutoencoder,DirectCritic,DirectTD3,SOURCE
from pi05_direct_identity import IdentityDirectActor
from evaluate_pi05_baseline import ROOT,write,save_video


class Replay:
    def __init__(self,seed=42):
        self.rows=[];self.rng=np.random.default_rng(seed)

    def add_episode(self,z,states,refs,actions,rewards,durations):
        z=np.asarray(z,dtype=np.float32);states=np.asarray(states,dtype=np.float32)
        refs=np.asarray(refs,dtype=np.float32);actions=np.asarray(actions,dtype=np.float32)
        n=len(z);ret=0.;returns=np.zeros(n,dtype=np.float32)
        for i in reversed(range(n)):
            ret=float(rewards[i])+.99**int(durations[i])*ret;returns[i]=ret
        for i in range(n):
            terminal=i==n-1;next_i=i+1 if not terminal else i
            self.rows.append(dict(z=z[i],state=states[i],ref=refs[i],action=actions[i],
                next_z=z[next_i] if not terminal else np.zeros_like(z[i]),
                next_state=states[next_i] if not terminal else np.zeros_like(states[i]),
                next_ref=refs[next_i] if not terminal else np.zeros_like(refs[i]),
                reward=np.float32(rewards[i]),duration=np.int64(durations[i]),terminal=np.bool_(terminal),mc=returns[i]))

    def batch(self,size=128):
        indices=self.rng.integers(len(self.rows),size=size)
        return {key:torch.from_numpy(np.stack([self.rows[i][key] for i in indices])) for key in self.rows[0]}

    def save(self,path):
        np.savez_compressed(path,**{key:np.stack([row[key] for row in self.rows]) for key in self.rows[0]})


class Run:
    def __init__(self,a):
        self.a=a;self.out=a.output;self.out.mkdir(parents=True)
        torch.set_num_threads(4);torch.manual_seed(42)
        self.encoder=TokenAutoencoder()
        self.encoder.load_state_dict(torch.load(a.encoder,weights_only=True,map_location='cpu')['model'])
        self.encoder.eval().requires_grad_(False)
        self.replay=Replay();self.episodes=[];self.baseline=[];self.best_success=-1;self.best_episode=0
        self.started=time.monotonic();self.close=lambda:None
        self.log('loading_training_features')
        for f in sorted(a.data.glob('training_task08_init*.npz')):
            with np.load(f) as d:
                assert int(d['task'])==8 and 0<=int(d['initial'])<30
                feats=d['features'];z=[]
                with torch.no_grad():
                    for start in range(0,len(feats),32):z.append(self.encoder.encode(torch.from_numpy(feats[start:start+32]).float()).numpy())
                self.replay.add_episode(np.concatenate(z),d['states'],d['reference'],d['reference'],d['rewards'],d['durations'])
        if not self.replay.rows or not any(row['reward']>0 for row in self.replay.rows):raise ValueError('No positive training experience')
        states=torch.from_numpy(np.stack([r['state'] for r in self.replay.rows]))
        mean=states.mean(0);std=states.std(0).clamp_min(.05)
        zs=torch.from_numpy(np.stack([r['z'] for r in self.replay.rows]))
        actor=IdentityDirectActor(zs.mean(0),zs.std(0).clamp_min(.05),mean,std)
        self.td=DirectTD3(actor,DirectCritic(actor).double())
        self.initial_actor=copy.deepcopy(self.td.actor.state_dict())
        self.rng=np.random.default_rng(12345)
        self.warm_count=len(self.replay.rows)
        self.log('warm_replay_ready',transitions=self.warm_count)

    def log(self,event,**kw):
        row=dict(event=event,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)

    def save(self,name,episode):
        state=dict(actor=self.td.actor.state_dict(),critic=self.td.critic.state_dict(),bc=self.td.bc.state_dict(),
            target_actor=self.td.target_actor.state_dict(),target_critic=self.td.target_critic.state_dict(),
            actor_optimizer=self.td.actor_opt.state_dict(),critic_optimizer=self.td.critic_opt.state_dict(),bc_optimizer=self.td.bc_opt.state_dict(),
            encoder=self.encoder.state_dict(),episode=episode,updates=self.td.updates,actor_updates=self.td.actor_updates,
            torch_rng=torch.get_rng_state(),td_noise_rng=self.td.generator.get_state(),numpy_rng=self.rng.bit_generator.state,
            replay_rng=self.replay.rng.bit_generator.state,best_success=self.best_success,best_episode=self.best_episode,
            architecture='published direct identity-initialized ActionTokenActor; local pooled pi encoder; task8 only',actor_dtype='float64',reference_dropout=0.,initialization='trainable MLP identity weights, no residual path')
        tmp=self.out/(name+'.tmp');torch.save(state,tmp);tmp.replace(self.out/name)

    def warm_critic(self):
        for step in range(500):
            b=self.replay.batch();q1,q2=self.td.critic(b['z'],b['state'],b['action'])
            loss=(q1-b['mc']).square().mean()+(q2-b['mc']).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MC loss')
            self.td.critic_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.td.critic.parameters(),1.);self.td.critic_opt.step()
            if step%100==0:self.log('critic_mc_warmup',step=step,loss=float(loss.detach()))
        self.td.target_critic.load_state_dict(self.td.critic.state_dict())

    def load_policy(self):
        from openpi.policies import policy_config
        from openpi.training import config
        from openpi_client import image_tools
        from pi05_features import PrefixFeatures
        from persistent_libero import install_persistent_env
        self.image_tools=image_tools
        self.log('loading_frozen_pi')
        self.policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),ROOT/'checkpoints/pi05_libero')
        self.features=PrefixFeatures(self.policy)
        os.environ.update(LIBERO_HOME=str(ROOT/'src/LIBERO'),LIBERO_PYTHON=str(ROOT/'envs/libero/bin/python'),
            LIBERO_CONFIG_PATH=str(ROOT/'config/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
        source=ROOT/'src/AlphaBrain-opt/AlphaBrain/training/reinforcement_learning/envs/libero_env.py'
        spec=importlib.util.spec_from_file_location('moka_proxy',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        m._WORKER_SCRIPT=str(ROOT/'scripts/libero_worker_pi05.py');self.holder=SimpleNamespace(Env=m.LiberoEnv,output=self.out)
        self.close=install_persistent_env(self.holder)
        self.log('policy_ready')

    def episode(self,initial,mode,episode=0,save=False,task=8,seed=42):
        if mode in ('train','bc_probe') and not 0<=initial<30:raise ValueError('Training split violation')
        if mode not in ('train','bc_probe') and not 30<=initial<40:raise ValueError('Validation split violation')
        env=self.holder.Env();obs=env.reset('libero_10',task,initial_state_idx=initial,seed=seed)
        for _ in range(10):obs,_,_=env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
        noise_rng=np.random.default_rng(np.random.SeedSequence([seed,task,initial]))
        frames=[];records=[];steps=0;success=False;t0=time.monotonic();delta_sums=[]
        actor=self.td.actor if mode!='bc_only' else self.td.bc
        while steps<520:
            element={'observation/image':self.image_tools.convert_to_uint8(self.image_tools.resize_with_pad(np.asarray(obs['primary_image']),224,224)),
                'observation/wrist_image':self.image_tools.convert_to_uint8(self.image_tools.resize_with_pad(np.asarray(obs['wrist_image']),224,224)),
                'observation/state':np.asarray(obs['state'],dtype=np.float32),'prompt':env.task_description}
            state=element['observation/state'].copy()
            noise=noise_rng.standard_normal((10,32)).astype(np.float32)
            ref_env=np.asarray(self.policy.infer(element,noise=noise)['actions'])[:5].copy()
            ref=ref_env.astype(np.float32)
            if mode!='baseline' and task==8:
                feat=self.features(element)
                with torch.no_grad():
                    z=self.encoder.encode(torch.from_numpy(feat).float()[None])[0]
                    action=actor(z[None],torch.from_numpy(state)[None],torch.from_numpy(ref_env)[None])[0].numpy().astype(np.float64)
                latent=z.numpy().copy()
                if mode=='train':
                    action=action+self.rng.normal(0,.02,ref.shape)
            else:action=ref_env.copy();latent=None
            if not np.isfinite(action).all():raise FloatingPointError('Nonfinite action')
            delta_sums.append(float(np.square(action-ref_env).sum()))
            reward_sum=0.;duration=0
            for command in action:
                obs,reward,done=env.step(command);steps+=1
                reward_sum+=.99**duration*reward*5.;duration+=1
                if save:frames.append(np.asarray(obs['primary_image']).copy())
                success=bool(done and reward>.5)
                if done or steps>=520:break
            if mode=='train':records.append((latent,state,ref,action.copy(),reward_sum,duration))
            if done:break
        if records:
            arrays=[np.stack([x[k] for x in records]) for k in range(6)]
            self.replay.add_episode(*arrays)
        row=dict(policy=mode,suite='libero_10',task_id=task,initial_state=initial,seed=seed,success=success,action_steps=steps,
                 training_episode=episode,rollout_seconds=time.monotonic()-t0,mean_squared_deviation=float(np.mean(delta_sums)))
        if save:
            name=f'{mode}_ep{episode:03d}_seed{seed}_task{task:02d}_init{initial:02d}_{"success" if success else "fail"}.mp4'
            save_video(self.out/name,frames);row['video']=name
        self.episodes.append(row);write(self.out/'episodes.json',self.episodes);self.log('episode',**row);env.close()
        return row

    def fit_bc(self):
        # Exact imitation at initialization through ordinary trainable MLP weights.
        b=self.replay.batch(128)
        with torch.no_grad():
            predicted=self.td.actor(b['z'],b['state'],b['ref'].double())
        assert torch.equal(predicted,b['ref'].double()),'Identity initialization failed'
        write(self.out/'bc_initialization.json',dict(steps=0,kind='exact imitation via trainable direct MLP identity initialization',reference_dropout=0.,actor_dtype='float64',no_residual_connection=True,all_weights_trainable=all(p.requires_grad for p in self.td.actor.parameters())))
        self.save('bc_initial.pt',0);self.log('exact_imitation_initialized',mse=0.,reference_dropout=0.)

    def validate(self,episode):
        rows=[self.episode(i,'candidate',episode,save=True,seed=seed) for seed in (42,7) for i in range(30,40)]
        successes=sum(r['success'] for r in rows)
        wins=[(r['seed'],r['initial_state']) for r,b in zip(rows,self.baseline) if r['success'] and not b['success']]
        losses=[(r['seed'],r['initial_state']) for r,b in zip(rows,self.baseline) if b['success'] and not r['success']]
        improved=successes>self.best_success
        if improved:self.best_success=successes;self.best_episode=episode;self.save('best.pt',episode)
        self.save(f'validation_ep{episode:03d}.pt',episode)
        self.log('validation',episode=episode,successes=successes,episodes=len(rows),wins=wins,losses=losses,save_best=improved)
        return rows

    def run(self):
        self.fit_bc()
        if self.a.bc_only_fit:
            write(self.out/'summary.json',dict(status='bc_fit_complete',wall_seconds=time.monotonic()-self.started));return
        self.warm_critic();self.load_policy()
        # BC initialization gets a real closed-loop check before RL.
        probes=[self.episode(i,'bc_probe',0,save=True,seed=42) for i in range(24,30)]
        probe_success=sum(r['success'] for r in probes)
        self.log('bc_probe_complete',successes=probe_success,episodes=6)
        if probe_success<2:
            write(self.out/'summary.json',dict(status='bc_initialization_failed',probe_successes=probe_success,episodes=6,note='RL not started; fix closed-loop imitation first'))
            return
        source=self.a.baseline_from
        self.baseline=json.loads((source/'baseline.json').read_text())
        assert {(r['seed'],r['initial_state']) for r in self.baseline}=={(seed,i) for seed in (42,7) for i in range(30,40)}
        for row in self.baseline:
            original=source/row['video'];dest=self.out/row['video'];dest.hardlink_to(original)
            row['reused_from']=str(source)
        self.episodes.extend(self.baseline);write(self.out/'episodes.json',self.episodes)
        write(self.out/'baseline_reuse.json',dict(source=str(source),sha256=hashlib.sha256((source/'baseline.json').read_bytes()).hexdigest(),reason='Same frozen pi, environment, seeds, initial states and action protocol; all20 matched previous baseline success and step counts',new_baseline_rollouts=0))
        write(self.out/'baseline.json',self.baseline)
        # Score the initialized Actor separately; best candidate need not beat pi.
        self.validate(0)
        loss_path=self.out/'training_losses.jsonl'
        for episode in range(1,self.a.episodes+1):
            seed=42 if episode<=30 else 7
            self.episode((episode-1)%30,'train',episode,seed=seed)
            with loss_path.open('a') as f:
                for _ in range(64):
                    row=self.td.update(self.replay.batch());row['episode']=episode;f.write(json.dumps(row,allow_nan=False)+'\n')
            if episode%10==0:
                self.save('latest.pt',episode);self.replay.save(self.out/'replay_latest.npz')
                self.log('training_checkpoint',episode=episode,transitions=len(self.replay.rows),critic_updates=self.td.updates,actor_updates=self.td.actor_updates)
            if episode%20==0 or episode==self.a.episodes:self.validate(episode)
        self.save('last.pt',self.a.episodes);self.replay.save(self.out/'replay_final.npz')
        # Matched BC checkpoint from the selected RL iteration, evaluated for real.
        selected=torch.load(self.out/'best.pt',weights_only=True,map_location='cpu')
        self.td.bc.load_state_dict(selected['bc'])
        bc=[self.episode(i,'bc_only',self.best_episode,save=True,seed=seed) for seed in (42,7) for i in range(30,40)]
        report=dict(status='complete',kind='published_direct_action_actor_identity_init_local_pi_encoder',baseline_successes=sum(r['success'] for r in self.baseline),
            best_validation_successes=self.best_success,bc_only_successes=sum(r['success'] for r in bc),validation_episodes=20,best_episode=self.best_episode,
            collected_training_episodes=self.a.episodes,warmup_episodes=30,replay_transitions=len(self.replay.rows),critic_mc_updates=500,
            critic_td_updates=self.td.updates,actor_updates=self.td.actor_updates,wall_seconds=time.monotonic()-self.started,test_initials_used=False,
            note='Two-seed validation model selection, not independent final-test evidence; original action form but local encoder.')
        write(self.out/'summary.json',report);self.log('complete',**report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--encoder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-from',type=Path,required=True);p.add_argument('--episodes',type=int,default=40);p.add_argument('--bc-steps',type=int,default=10000);p.add_argument('--bc-only-fit',action='store_true');a=p.parse_args()
    if a.output.exists() or a.episodes<1 or a.bc_steps<1:p.error('New output and positive steps required')
    files=[Path(__file__),ROOT/'scripts/pi05_direct_networks.py',ROOT/'scripts/pi05_direct_identity.py',ROOT/'scripts/pi05_features.py',ROOT/'scripts/libero_worker_pi05.py',SOURCE]
    protocol=dict(model='pi05_libero',suite='libero_10',task=8,training_initials=list(range(30)),validation_initials=list(range(30,40)),validation_seeds=[42,7],
        reserved_test_initials=list(range(40,50)),reserved_test_seeds=[123,456],episodes=a.episodes,replan_steps=5,max_control_steps=520,
        action_form='Actor directly outputs 5x7 complete action chunk. No residual addition or reference-relative clamp.',
        encoder='Frozen local compact pooled-prefix encoder, not the published encoder',actor='Imported published ActionTokenActor, hidden512, chunk5, prop8, token256',
        reward_success=5,gamma=.99,actor_lr=1e-5,critic_lr=3e-4,beta=3.,reference_dropout=0.,bc_steps=0,initialization='Exact imitation in trainable MLP weights; float64 for exact reference preservation',
        batch_size=128,updates_per_episode=64,actor_delay=2,exploration_std=.02,encoder_sha256=hashlib.sha256(a.encoder.read_bytes()).hexdigest(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        bc_control='Same identity initialization, batches, optimizer and updates; reference dropout disabled; BC-only removes Q term; actual paired rollouts')
    exp=None
    try:
        exp=Run(a);write(a.output/'protocol.json',protocol)
        snapshot=a.output/'source_snapshot';snapshot.mkdir()
        for f in files:(snapshot/f.name).write_bytes(f.read_bytes())
        exp.run()
    except BaseException as exc:
        if a.output.exists():write(a.output/'run_error.json',dict(status='failed',error=repr(exc)))
        raise
    finally:
        if exp is not None:exp.close()

if __name__=='__main__':main()
