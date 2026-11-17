from pathlib import Path
p=Path(__file__).parent
s=(p/'train_pi05_moka_rlt.py').read_text()
s=s.replace('from pi05_rlt_networks import TokenAutoencoder,ResidualActor,TwinCritic,TD3','from pi05_direct_networks import TokenAutoencoder,DirectActor,DirectCritic,DirectTD3,SOURCE')
s=s.replace('self.td=TD3(ResidualActor(mean,std),TwinCritic(mean,std))', '''zs=torch.from_numpy(np.stack([r['z'] for r in self.replay.rows]))
        actor=DirectActor(zs.mean(0),zs.std(0).clamp_min(.05),mean,std)
        self.td=DirectTD3(actor,DirectCritic(actor))''')
s=s.replace('encoder=self.encoder.state_dict(),bound=self.td.actor.bound,episode=episode','encoder=self.encoder.state_dict(),episode=episode')
s=s.replace("architecture='local compact pooled-prefix residual RLT; task8 libero_10 only'","architecture='published direct ActionTokenActor; local pooled pi encoder; task8 libero_10 only'")
s=s.replace('def episode(self,initial,mode,episode=0,save=False,task=8):','def episode(self,initial,mode,episode=0,save=False,task=8,seed=42):')
s=s.replace("if mode=='train' and not 0<=initial<30:","if mode in ('train','bc_probe') and not 0<=initial<30:")
s=s.replace("if mode!='train' and not 30<=initial<40:","if mode not in ('train','bc_probe') and not 30<=initial<40:")
s=s.replace('initial_state_idx=initial,seed=42','initial_state_idx=initial,seed=seed').replace('[42,task,initial]','[seed,task,initial]')
s=s.replace("delta=actor.correction(z[None],torch.from_numpy(state)[None],torch.from_numpy(ref)[None])[0].numpy()","action=actor(z[None],torch.from_numpy(state)[None],torch.from_numpy(ref)[None])[0].numpy().astype(np.float64)")
s=s.replace("delta=(delta+self.rng.normal(0,.02,ref.shape)).clip(-actor.bound,actor.bound)\n                action=ref_env+delta","action=action+self.rng.normal(0,.02,ref.shape)")
s=s.replace('initial_state=initial,seed=42','initial_state=initial,seed=seed').replace('mean_squared_residual=','mean_squared_deviation=')
s=s.replace("f'{mode}_ep{episode:03d}_task{task:02d}_init{initial:02d}_", "f'{mode}_ep{episode:03d}_seed{seed}_task{task:02d}_init{initial:02d}_")
start=s.index('    def validate(self,episode):')
s=s[:start]+'''    def fit_bc(self):
        # Whole-trajectory split within training data only. Never use validation actions.
        episode_ids=np.cumsum([False]+[bool(r['terminal']) for r in self.replay.rows[:-1]])
        train_idx=np.where(episode_ids<24)[0];held_idx=np.where(episode_ids>=24)[0]
        data={k:torch.from_numpy(np.stack([r[k] for r in self.replay.rows])) for k in ('z','state','ref')}
        # Recompute preprocessing using only fit trajectories.
        with torch.no_grad():
            self.td.actor.zmean.copy_(data['z'][train_idx].mean(0));self.td.actor.zstd.copy_(data['z'][train_idx].std(0).clamp_min(.05))
            self.td.actor.smean.copy_(data['state'][train_idx].mean(0));self.td.actor.sstd.copy_(data['state'][train_idx].std(0).clamp_min(.05))
        opt=torch.optim.Adam(self.td.actor.parameters(),lr=3e-4)
        best=float('inf');best_state=None;best_step=0
        for step in range(1,self.a.bc_steps+1):
            idx=self.rng.choice(train_idx,256)
            pred=self.td.actor(data['z'][idx],data['state'][idx],data['ref'][idx],dropout=True)
            loss=(pred-data['ref'][idx]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite BC initialization')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.td.actor.parameters(),1.);opt.step()
            if step%500==0 or step==self.a.bc_steps:
                with torch.no_grad():
                    errors=[]
                    for offset in range(0,len(held_idx),256):
                        j=held_idx[offset:offset+256]
                        err=self.td.actor(data['z'][j],data['state'][j],data['ref'][j])-data['ref'][j]
                        errors.append(err.square().flatten(1).mean(1))
                    mse=float(torch.cat(errors).mean())
                if mse<best:best=mse;best_state=copy.deepcopy(self.td.actor.state_dict());best_step=step
                self.log('bc_initialization',step=step,loss=float(loss.detach()),heldout_mse=mse,best_mse=best)
        self.td.actor.load_state_dict(best_state)
        actor=self.td.actor
        self.td=DirectTD3(actor,DirectCritic(actor))
        write(self.out/'bc_initialization.json',dict(steps=self.a.bc_steps,best_step=best_step,heldout_mse=best,fit_initials=list(range(24)),heldout_training_initials=list(range(24,30)),reference_dropout=.5))
        self.save('bc_initial.pt',0)

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
        self.baseline=[self.episode(i,'baseline',0,save=True,seed=seed) for seed in (42,7) for i in range(30,40)]
        write(self.out/'baseline.json',self.baseline)
        # Score the initialized Actor separately; best candidate need not beat pi.
        self.validate(0)
        loss_path=self.out/'training_losses.jsonl'
        for episode in range(1,self.a.episodes+1):
            seed=42 if episode<=30 else 7
            self.episode((episode-1)%30,'train',episode,seed=seed)
            with loss_path.open('a') as f:
                for _ in range(64):
                    row=self.td.update(self.replay.batch());row['episode']=episode;f.write(json.dumps(row,allow_nan=False)+'\\n')
            if episode%10==0:
                self.save('latest.pt',episode);self.replay.save(self.out/'replay_latest.npz')
                self.log('training_checkpoint',episode=episode,transitions=len(self.replay.rows),critic_updates=self.td.updates,actor_updates=self.td.actor_updates)
            if episode%20==0 or episode==self.a.episodes:self.validate(episode)
        self.save('last.pt',self.a.episodes);self.replay.save(self.out/'replay_final.npz')
        # Matched BC checkpoint from the selected RL iteration, evaluated for real.
        selected=torch.load(self.out/'best.pt',weights_only=True,map_location='cpu')
        self.td.bc.load_state_dict(selected['bc'])
        bc=[self.episode(i,'bc_only',self.best_episode,save=True,seed=seed) for seed in (42,7) for i in range(30,40)]
        report=dict(status='complete',kind='published_direct_action_actor_with_local_pi_encoder',baseline_successes=sum(r['success'] for r in self.baseline),
            best_validation_successes=self.best_success,bc_only_successes=sum(r['success'] for r in bc),validation_episodes=20,best_episode=self.best_episode,
            collected_training_episodes=self.a.episodes,warmup_episodes=30,replay_transitions=len(self.replay.rows),critic_mc_updates=500,
            critic_td_updates=self.td.updates,actor_updates=self.td.actor_updates,wall_seconds=time.monotonic()-self.started,test_initials_used=False,
            note='Two-seed validation model selection, not independent final-test evidence; original action form but local encoder.')
        write(self.out/'summary.json',report);self.log('complete',**report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--encoder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=40);p.add_argument('--bc-steps',type=int,default=10000);p.add_argument('--bc-only-fit',action='store_true');a=p.parse_args()
    if a.output.exists() or a.episodes<1 or a.bc_steps<1:p.error('New output and positive steps required')
    files=[Path(__file__),ROOT/'scripts/pi05_direct_networks.py',ROOT/'scripts/pi05_features.py',ROOT/'scripts/libero_worker_pi05.py',SOURCE]
    protocol=dict(model='pi05_libero',suite='libero_10',task=8,training_initials=list(range(30)),validation_initials=list(range(30,40)),validation_seeds=[42,7],
        reserved_test_initials=list(range(40,50)),reserved_test_seeds=[123,456],episodes=a.episodes,replan_steps=5,max_control_steps=520,
        action_form='Actor directly outputs 5x7 complete action chunk. No residual addition or reference-relative clamp.',
        encoder='Frozen local compact pooled-prefix encoder, not the published encoder',actor='Imported published ActionTokenActor, hidden512, chunk5, prop8, token256',
        reward_success=5,gamma=.99,actor_lr=1e-5,critic_lr=3e-4,beta=3.,reference_dropout=.5,bc_steps=a.bc_steps,
        batch_size=128,updates_per_episode=64,actor_delay=2,exploration_std=.02,encoder_sha256=hashlib.sha256(a.encoder.read_bytes()).hexdigest(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        bc_control='Same initialization, batches, dropout masks, optimizer and updates; BC-only removes Q term; actual paired rollouts')
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
'''
s=s.replace('"""Train a local compact-prefix, residual RLT adaptation on train initial states.\n\nThis is not the released Qwen RLT_a or an official pi RLT reproduction.\nFrozen pi+encoder, CPU TD3, reward5, gamma.99, bound.1, matched BC identity.\n"""','"""Direct published action-chunk Actor with local frozen pi features; TD3+BC."""')
(p/'train_pi05_moka_direct.py').write_text(s)
