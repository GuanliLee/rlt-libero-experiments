#!/usr/bin/env python3
"""Frozen paired evaluation of trained RLT residual strengths; no training."""
import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from evaluate_pi05_baseline import ROOT,indices,write,save_video
from pi05_rlt_networks import TokenAutoencoder,ResidualActor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scales',default='0,0.25,0.5,1');p.add_argument('--seeds',type=indices,default=[42])
    p.add_argument('--initials',type=indices,default=list(range(30,40)))
    p.add_argument('--split',choices=['training_probe','validation'],default='validation')
    p.add_argument('--purpose',choices=['calibration','validation','confirmation'],default='validation')
    a=p.parse_args();a.scales=[float(x) for x in a.scales.split(',')]
    allowed=set(range(30)) if a.split=='training_probe' else set(range(30,40))
    if not set(a.initials)<=allowed or not a.scales or len(set(a.scales))!=len(a.scales) or any(not 0<=x<=1 for x in a.scales):p.error('Invalid split or residual scale')
    if 0 not in a.scales:p.error('Include scale0 for the paired pi baseline')
    if a.output.exists():p.error('Choose new output')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    checkpoint=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    encoder=TokenAutoencoder();encoder.load_state_dict(checkpoint['encoder']);encoder.eval().requires_grad_(False)
    actor=ResidualActor(bound=checkpoint['bound']);actor.load_state_dict(checkpoint['actor']);actor.eval().requires_grad_(False)
    protocol=dict(status='running',suite='libero_10',task=8,split=a.split,purpose=a.purpose,initials=a.initials,seeds=a.seeds,scales=a.scales,
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        training_episode=checkpoint['episode'],training_updates=0,max_steps=520,replan_steps=5,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'scripts/pi05_rlt_networks.py',ROOT/'scripts/pi05_features.py']},
        action='Original float64 pi action plus scale times trained residual; scale0 bypasses residual exactly',
        note='Additional validation/confirmation, not untouched final-test evidence.')
    write(a.output/'protocol.json',protocol);rows=[];close=lambda:None;started=time.monotonic()
    def log(event,**kw):
        row=dict(event=event,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        from openpi.policies import policy_config
        from openpi.training import config
        from openpi_client import image_tools
        from pi05_features import PrefixFeatures
        from persistent_libero import install_persistent_env
        log('loading')
        policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),ROOT/'checkpoints/pi05_libero');features=PrefixFeatures(policy)
        os.environ.update(LIBERO_HOME=str(ROOT/'src/LIBERO'),LIBERO_PYTHON=str(ROOT/'envs/libero/bin/python'),
            LIBERO_CONFIG_PATH=str(ROOT/'config/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
        spec=importlib.util.spec_from_file_location('moka_variant_proxy',ROOT/'src/AlphaBrain-opt/AlphaBrain/training/reinforcement_learning/envs/libero_env.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module._WORKER_SCRIPT=str(ROOT/'scripts/libero_worker_pi05.py')
        holder=SimpleNamespace(Env=module.LiberoEnv,output=a.output);close=install_persistent_env(holder)
        for seed in a.seeds:
            for scale in a.scales:
                for initial in a.initials:
                    env=holder.Env();obs=env.reset('libero_10',8,initial_state_idx=initial,seed=seed)
                    for _ in range(10):obs,_,_=env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
                    rng=np.random.default_rng(np.random.SeedSequence([seed,8,initial]));steps=0;success=False;frames=[];t0=time.monotonic()
                    while steps<520:
                        element={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(obs['primary_image']),224,224)),
                            'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(obs['wrist_image']),224,224)),
                            'observation/state':np.asarray(obs['state'],dtype=np.float32),'prompt':env.task_description}
                        ref=np.asarray(policy.infer(element,noise=rng.standard_normal((10,32)).astype(np.float32))['actions'])[:5].copy()
                        action=ref
                        if scale:
                            with torch.no_grad():
                                z=encoder.encode(torch.from_numpy(features(element)).float()[None])
                                delta=actor.correction(z,torch.from_numpy(element['observation/state'])[None],torch.from_numpy(ref.astype(np.float32))[None])[0].numpy()
                            action=ref+scale*delta
                        if not np.isfinite(action).all():raise FloatingPointError('Invalid action')
                        for command in action:
                            obs,reward,done=env.step(command);steps+=1;success=bool(done and reward>.5);frames.append(np.asarray(obs['primary_image']).copy())
                            if done or steps>=520:break
                        if done:break
                    mode='baseline' if scale==0 else f'rlt_scale_{scale:g}'
                    name=f'{mode}_seed{seed}_init{initial}_{"success" if success else "fail"}.mp4'
                    rollout=time.monotonic()-t0;save_video(a.output/name,frames)
                    row=dict(policy=mode,scale=scale,task_id=8,initial_state=initial,seed=seed,success=success,action_steps=steps,video=name,rollout_seconds=rollout)
                    rows.append(row);write(a.output/'episodes.json',rows);log('episode',**row);env.close()
        comparisons=[]
        for seed in a.seeds:
            base={r['initial_state']:r for r in rows if r['seed']==seed and r['scale']==0}
            for scale in a.scales:
                subset=[r for r in rows if r['seed']==seed and r['scale']==scale]
                comparisons.append(dict(seed=seed,scale=scale,successes=sum(r['success'] for r in subset),episodes=len(subset),
                    wins=[r['initial_state'] for r in subset if r['success'] and not base[r['initial_state']]['success']],
                    losses=[r['initial_state'] for r in subset if not r['success'] and base[r['initial_state']]['success']]))
        report=dict(status='complete',comparisons=comparisons,wall_seconds=time.monotonic()-started,training_updates=0,protocol=protocol)
        write(a.output/'summary.json',report)
        parts=['<!doctype html><meta charset="utf-8"><title>双摩卡壶：冻结策略对照</title>',
            '<style>body{font:16px/1.7 system-ui;margin:28px}video{width:256px}section{margin:24px 0}td,th{padding:8px;border-bottom:1px solid #ddd}</style>',
            f'<h1>双摩卡壶：{html.escape(a.purpose)}</h1><p>冻结第{checkpoint["episode"]}回合模型；没有训练更新。左侧原始π，右侧RLT。当前为额外验证，最终初态40–49未使用。</p>',
            '<table><tr><th>种子</th><th>修正比例</th><th>成功数</th><th>救回初态</th><th>退步初态</th></tr>']
        for c in comparisons:parts.append(f'<tr><td>{c["seed"]}</td><td>{c["scale"]}</td><td>{c["successes"]}/{c["episodes"]}</td><td>{c["wins"]}</td><td>{c["losses"]}</td></tr>')
        parts.append('</table>')
        for row in rows:
            if row['scale']==0:continue
            base=next(x for x in rows if x['scale']==0 and x['seed']==row['seed'] and x['initial_state']==row['initial_state'])
            parts.append(f'<section><b>种子{row["seed"]} / 初态{row["initial_state"]} / 修正比例{row["scale"]}：π {"成功" if base["success"] else "失败"}，RLT {"成功" if row["success"] else "失败"}</b><br>')
            for x in (base,row):parts.append(f'<video controls preload="none" src="{html.escape(x["video"],quote=True)}"></video> ')
            parts.append('</section>')
        (a.output/'index.html').write_text(''.join(parts),encoding='utf-8');log('complete',comparisons=comparisons)
    except BaseException as exc:
        write(a.output/'run_error.json',dict(status='failed',error=repr(exc)));raise
    finally:close()


if __name__=='__main__':main()
