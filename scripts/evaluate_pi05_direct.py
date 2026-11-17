#!/usr/bin/env python3
"""Frozen paired pi/BC/direct-Actor evaluation; no residual and no training."""
import argparse,hashlib,html,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from evaluate_pi05_baseline import ROOT,write,save_video,indices
from train_pi05_moka_direct import Run
from pi05_direct_networks import TokenAutoencoder,DirectActor,DirectCritic,SOURCE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['training_probe','validation','final_test'],required=True)
    p.add_argument('--initials',type=indices,required=True);p.add_argument('--seeds',type=indices,required=True)
    p.add_argument('--policies',default='baseline,bc_only,candidate');p.add_argument('--q-threshold',type=float,default=0.0);a=p.parse_args();modes=a.policies.split(',')
    allowed={'training_probe':set(range(30)),'validation':set(range(30,40)),'final_test':set(range(40,50))}[a.split]
    if a.output.exists() or not set(a.initials)<=allowed or set(modes)-{'baseline','bc_only','candidate','candidate_gated'} or len(set(modes))!=len(modes):p.error('Invalid output/split/policy')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    ckpt=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    if not ckpt['architecture'].startswith('published direct'):raise ValueError('Not a direct-action checkpoint')
    encoder=TokenAutoencoder();encoder.load_state_dict(ckpt['encoder']);encoder.eval().requires_grad_(False)
    actors={}
    for name,key in [('candidate','actor'),('bc_only','bc')]:
        actor=DirectActor()
        if ckpt.get('actor_dtype')=='float64':actor.double()
        actor.model.ref_dropout=ckpt.get('reference_dropout',.5)
        actor.load_state_dict(ckpt[key]);actor.eval().requires_grad_(False);actors[name]=actor
    if 'candidate_gated' in modes:
        actors['candidate_gated']=actors['candidate']
        critic=DirectCritic(actors['candidate']).to(next(actors['candidate'].parameters()).dtype)
        critic.load_state_dict(ckpt['critic']);critic.eval().requires_grad_(False)
    protocol=dict(split=a.split,initials=a.initials,seeds=a.seeds,policies=modes,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        training_episode=ckpt['episode'],checkpoint_offline_updates=ckpt.get('offline_updates',0),training_updates=0,max_steps=520,replan_steps=5,action_form='Direct complete 5x7 action chunk; no residual addition',q_threshold=a.q_threshold,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'scripts/pi05_direct_networks.py',ROOT/'scripts/pi05_features.py',ROOT/'scripts/libero_worker_pi05.py',SOURCE]})
    write(a.output/'protocol.json',protocol)
    # Reuse ONLY policy/environment loading, not any training initialization.
    runner=Run.__new__(Run);runner.out=a.output;runner.close=lambda:None
    rows=[];started=time.monotonic()
    try:
        runner.load_policy()
        for seed in a.seeds:
            for mode in modes:
                for initial in a.initials:
                    env=runner.holder.Env();obs=env.reset('libero_10',8,initial_state_idx=initial,seed=seed)
                    for _ in range(10):obs,_,_=env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
                    rng=np.random.default_rng(np.random.SeedSequence([seed,8,initial]));steps=0;success=False;frames=[];t0=time.monotonic()
                    gate_actor=gate_pi=0;q_margins=[]
                    while steps<520:
                        element={'observation/image':runner.image_tools.convert_to_uint8(runner.image_tools.resize_with_pad(np.asarray(obs['primary_image']),224,224)),
                            'observation/wrist_image':runner.image_tools.convert_to_uint8(runner.image_tools.resize_with_pad(np.asarray(obs['wrist_image']),224,224)),
                            'observation/state':np.asarray(obs['state'],dtype=np.float32),'prompt':env.task_description}
                        ref=np.asarray(runner.policy.infer(element,noise=rng.standard_normal((10,32)).astype(np.float32))['actions'])[:5].copy()
                        action=ref
                        if mode!='baseline':
                            with torch.no_grad():
                                z=encoder.encode(torch.from_numpy(runner.features(element)).float()[None])
                                action=actors[mode](z,torch.from_numpy(element['observation/state'])[None],torch.from_numpy(ref).to(next(actors[mode].parameters()).dtype)[None])[0].numpy().astype(np.float64)
                                if mode=='candidate_gated':
                                    dtype=next(actors[mode].parameters()).dtype
                                    st=torch.from_numpy(element['observation/state'])[None]
                                    qa=torch.minimum(*critic(z,st,torch.from_numpy(action).to(dtype)[None])).item()
                                    qr=torch.minimum(*critic(z,st,torch.from_numpy(ref).to(dtype)[None])).item()
                                    q_margins.append(qa-qr)
                                    if qa < qr+a.q_threshold: action=ref;gate_pi+=1
                                    else: gate_actor+=1
                        if not np.isfinite(action).all():raise FloatingPointError('Nonfinite action')
                        for command in action:
                            obs,reward,done=env.step(command);steps+=1;success=bool(done and reward>.5);frames.append(np.asarray(obs['primary_image']).copy())
                            if done or steps>=520:break
                        if done:break
                    video=f'{mode}_seed{seed}_init{initial}_{"success" if success else "fail"}.mp4'
                    elapsed=time.monotonic()-t0;save_video(a.output/video,frames)
                    row=dict(policy=mode,seed=seed,initial_state=initial,success=success,action_steps=steps,video=video,rollout_seconds=elapsed,gate_actor=gate_actor,gate_pi=gate_pi,mean_q_margin=float(np.mean(q_margins)) if q_margins else None)
                    rows.append(row);write(a.output/'episodes.json',rows);runner.log('episode',**row);env.close()
        metrics=[]
        for seed in a.seeds:
            base={r['initial_state']:r for r in rows if r['seed']==seed and r['policy']=='baseline'}
            for mode in modes:
                subset=[r for r in rows if r['seed']==seed and r['policy']==mode]
                metrics.append(dict(seed=seed,policy=mode,successes=sum(r['success'] for r in subset),episodes=len(subset),
                    wins=[r['initial_state'] for r in subset if r['success'] and r['initial_state'] in base and not base[r['initial_state']]['success']],
                    losses=[r['initial_state'] for r in subset if not r['success'] and r['initial_state'] in base and base[r['initial_state']]['success']]))
        report=dict(status='complete',comparisons=metrics,wall_seconds=time.monotonic()-started,protocol=protocol)
        write(a.output/'summary.json',report);runner.log('complete',comparisons=metrics)
        checkpoint_episode=ckpt['episode'];offline_updates=ckpt.get('offline_updates',0);policy_names=', '.join(modes)
        parts=['<!doctype html><meta charset="utf-8"><title>完整动作块：冻结评估</title><style>body{font:16px/1.7 system-ui;margin:28px}video{width:256px}section{margin:22px 0}</style>',f'<h1>完整动作块：{a.split}</h1><p>冻结模型（在线回合{checkpoint_episode}，离线更新{offline_updates}）；没有训练更新。本次执行策略：{policy_names}。</p>', '<pre>'+html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))+'</pre>']
        for seed in a.seeds:
            for initial in a.initials:
                parts.append(f'<section><b>种子{seed}，初态{initial}</b><br>')
                for mode in modes:
                    row=next(r for r in rows if r['seed']==seed and r['initial_state']==initial and r['policy']==mode)
                    parts.append(f'<span>{mode}：{"成功" if row["success"] else "失败"}</span><video controls preload="none" src="{row["video"]}"></video> ')
                parts.append('</section>')
        (a.output/'index.html').write_text(''.join(parts),encoding='utf-8')
    except BaseException as exc:
        write(a.output/'run_error.json',dict(status='failed',error=repr(exc)));raise
    finally:runner.close()
if __name__=='__main__':main()
