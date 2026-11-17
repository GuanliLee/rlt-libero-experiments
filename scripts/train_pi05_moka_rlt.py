#!/usr/bin/env python3
"""Train a local compact-prefix, residual RLT adaptation on train initial states.

This is not the released Qwen RLT_a or an official pi RLT reproduction.
Frozen pi+encoder, CPU TD3, reward5, gamma.99, bound.1, matched BC identity.
"""
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
from pi05_rlt_networks import TokenAutoencoder,ResidualActor,DirectActor,TwinCritic,TD3
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
        self.td=TD3((DirectActor(mean,std) if a.actor_mode=='direct' else ResidualActor(mean,std)),TwinCritic(mean,std))
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
            encoder=self.encoder.state_dict(),bound=self.td.actor.bound,episode=episode,updates=self.td.updates,actor_updates=self.td.actor_updates,
            torch_rng=torch.get_rng_state(),td_noise_rng=self.td.generator.get_state(),numpy_rng=self.rng.bit_generator.state,
            replay_rng=self.replay.rng.bit_generator.state,best_success=self.best_success,best_episode=self.best_episode,
            architecture=f'local compact pooled-prefix {self.a.actor_mode} RLT; task8 libero_10 only')
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

    def pretrain_direct(self):
        if self.a.actor_mode != 'direct': return
        for step in range(500):
            b=self.replay.batch(); pred=self.td.actor(b['z'],b['state'],b['ref'])
            loss=(pred-b['ref']).square().flatten(1).sum(1).mean()
            self.td.actor_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.td.actor.parameters(),1.); self.td.actor_opt.step()
            if step % 100 == 0: self.log('direct_bc_warmup',step=step,loss=float(loss.detach()))
        self.td.target_actor.load_state_dict(self.td.actor.state_dict())

    def episode(self,initial,mode,episode=0,save=False,task=8):
        if mode=='train' and not 0<=initial<30:raise ValueError('Training split violation')
        if mode!='train' and not 30<=initial<40:raise ValueError('Validation split violation')
        env=self.holder.Env();obs=env.reset('libero_10',task,initial_state_idx=initial,seed=42)
        for _ in range(10):obs,_,_=env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
        noise_rng=np.random.default_rng(np.random.SeedSequence([42,task,initial]))
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
                    if self.a.actor_mode == 'direct':
                        predicted=actor(z[None],torch.from_numpy(state)[None],torch.from_numpy(ref)[None])[0].numpy()
                        delta=predicted-ref_env
                    else:
                        delta=actor.correction(z[None],torch.from_numpy(state)[None],torch.from_numpy(ref)[None])[0].numpy()
                latent=z.numpy().copy()
                if mode=='train' and self.a.actor_mode != 'direct':
                    delta=(delta+self.rng.normal(0,.02,ref.shape)).clip(-actor.bound,actor.bound)
                if mode=='train' and self.a.actor_mode == 'direct':
                    action=np.clip(ref_env+delta+self.rng.normal(0,.02,ref.shape),-1.,1.)
                else: action=ref_env+delta
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
        row=dict(policy=mode,suite='libero_10',task_id=task,initial_state=initial,seed=42,success=success,action_steps=steps,
                 training_episode=episode,rollout_seconds=time.monotonic()-t0,mean_squared_residual=float(np.mean(delta_sums)))
        if save:
            name=f'{mode}_ep{episode:03d}_task{task:02d}_init{initial:02d}_{"success" if success else "fail"}.mp4'
            save_video(self.out/name,frames);row['video']=name
        self.episodes.append(row);write(self.out/'episodes.json',self.episodes);self.log('episode',**row);env.close()
        return row

    def validate(self,episode):
        rows=[self.episode(i,'candidate',episode,save=True) for i in range(30,40)]
        wins=[r['initial_state'] for r,b in zip(rows,self.baseline) if r['success'] and not b['success']]
        losses=[r['initial_state'] for r,b in zip(rows,self.baseline) if b['success'] and not r['success']]
        successes=sum(r['success'] for r in rows)
        improved=successes>self.best_success
        if improved:
            self.best_success=successes;self.best_episode=episode;self.save('best.pt',episode)
        self.log('validation',episode=episode,successes=successes,episodes=10,wins=wins,losses=losses,save_best=improved,
                 bc_only_is_exact_baseline=all(torch.equal(v,self.initial_actor[k]) for k,v in self.td.bc.state_dict().items()))
        return rows

    def run(self):
        self.save('initial.pt',0);self.warm_critic();self.pretrain_direct();self.load_policy()
        self.baseline=[self.episode(i,'baseline',0,save=True) for i in range(30,40)]
        self.best_success=sum(r['success'] for r in self.baseline);self.save('best.pt',0)
        write(self.out/'baseline.json',self.baseline)
        loss_path=self.out/'training_losses.jsonl'
        for episode in range(1,self.a.episodes+1):
            self.episode((episode-1)%30,'train',episode)
            with loss_path.open('a') as f:
                for _ in range(64):
                    row=self.td.update(self.replay.batch());row['episode']=episode;f.write(json.dumps(row,allow_nan=False)+'\n')
            if episode%10==0:
                self.save('latest.pt',episode);self.replay.save(self.out/'replay_latest.npz')
                self.log('training_checkpoint',episode=episode,transitions=len(self.replay.rows),critic_updates=self.td.updates,actor_updates=self.td.actor_updates)
            if episode%20==0 or episode==self.a.episodes:self.validate(episode)
        self.save('last.pt',self.a.episodes)
        self.replay.save(self.out/'replay_final.npz')
        assert all(torch.equal(v,self.initial_actor[k]) for k,v in self.td.bc.state_dict().items()),'BC identity changed'
        report=dict(status='complete',kind='experimental_compact_pi_residual_rlt',baseline_successes=sum(r['success'] for r in self.baseline),
            best_validation_successes=self.best_success,validation_episodes=10,best_episode=self.best_episode,
            collected_training_episodes=self.a.episodes,warmup_episodes=30,replay_transitions=len(self.replay.rows),
            critic_mc_updates=500,critic_td_updates=self.td.updates,actor_updates=self.td.actor_updates,
            bc_updates=self.td.actor_updates,bc_only_exact_baseline=True,wall_seconds=time.monotonic()-self.started,
            test_initials_used=False,full_suite_non_target_policy='Original pi unchanged; task8-specific routing',
            note='Validation-selected candidate only; direct mode follows RLT_a action-chunk output, not an official checkpoint.')
        write(self.out/'summary.json',report);self.log('complete',**report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--encoder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=60);p.add_argument('--actor-mode',choices=['residual','direct'],default='residual');a=p.parse_args()
    if a.output.exists() or a.episodes<1:p.error('New output and positive episodes required')
    source_files=[Path(__file__),ROOT/'scripts/pi05_rlt_networks.py',ROOT/'scripts/pi05_features.py',ROOT/'scripts/libero_worker_pi05.py']
    protocol=dict(model='pi05_libero',suite='libero_10',task=8,training_initials=list(range(30)),validation_initials=list(range(30,40)),
        reserved_test=list(range(40,50)),episodes=a.episodes,replan_steps=5,max_control_steps=520,reward_success=5,gamma=.99,
        actor_lr=1e-4,critic_lr=3e-4,batch_size=128,updates_per_episode=64,actor_delay=2,beta=3.,residual_bound=.1,
        exploration_std=.02,encoder_sha256=hashlib.sha256(a.encoder.read_bytes()).hexdigest(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        actor_mode=a.actor_mode,bc_control='Direct mode uses BC warmup then direct action chunks; residual mode is zero-initialized residual')
    exp=None
    try:
        exp=Run(a);write(a.output/'protocol.json',protocol);exp.run()
    except BaseException as exc:
        if a.output.exists():write(a.output/'run_error.json',dict(status='failed',error=repr(exc)))
        raise
    finally:
        if exp is not None:exp.close()


if __name__=='__main__':main()
