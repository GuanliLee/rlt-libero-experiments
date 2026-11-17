#!/usr/bin/env python3
"""Frozen pi05-LIBERO evaluation with explicit local initial-state splits.

Uses official OpenPI preprocessing, 5 executed actions per 10-action prediction,
and the existing pinned LIBERO subprocess. No training, no Qwen action transform.
"""
import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
SUITE_STEPS={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}

def indices(value):
    out=[]
    for part in value.split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));out.extend(range(lo,hi+1))
        else:out.append(int(part))
    if not out or len(out)!=len(set(out)):raise argparse.ArgumentTypeError('Use unique, nonempty indices')
    return out

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=tuple(SUITE_STEPS),default='libero_goal')
    p.add_argument('--tasks',type=indices,default=indices('0-9'))
    p.add_argument('--initials',type=indices,default=indices('31-33'))
    p.add_argument('--split',choices=['training_probe','validation','test'],default='validation')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-videos',action='store_true')
    p.add_argument('--record-features',action='store_true',help='Record frozen prefix features on training initial states only')
    a=p.parse_args();a.output=a.output.resolve();a.max_steps=SUITE_STEPS[a.suite]
    if a.output.exists():p.error('Choose a new output directory')
    allowed=set(range(30)) if a.split=='training_probe' else set(range(30,40)) if a.split=='validation' else set(range(40,50))
    if not set(a.tasks)<=set(range(10)) or not set(a.initials)<=allowed:p.error('Task/initial indices violate the declared split')
    if a.split=='test' and (set(a.tasks)!=set(range(10)) or set(a.initials)!=set(range(40,50))):p.error('Final test must cover all100 pairs')
    if a.record_features and a.split!='training_probe':p.error('Feature collection is restricted to training initial states')
    return a

def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)

def save_video(path,frames):
    import numpy as np
    if not frames:raise ValueError('No video frames')
    h,w=frames[0].shape[:2]
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{w}x{h}','-r','10','-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(path)],input=np.stack(frames).tobytes(),check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

def main():
    a=arguments()
    from check_pi05_inference import ROOT as CHECK_ROOT
    assert CHECK_ROOT==ROOT
    marker=ROOT/'manifests/pi05_libero_download_verified.json'
    if not marker.exists() or json.loads(marker.read_text()).get('status')!='complete':
        raise RuntimeError('Download verification is incomplete')
    inventory=json.loads(marker.read_text())
    for f in inventory['files']:
        if not Path(f['path']).is_file() or Path(f['path']).stat().st_size!=f['size']:raise RuntimeError('Checkpoint size/availability changed')
    if a.save_videos and not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg is needed for videos')
    os.environ.update(LIBERO_HOME=str(ROOT/'src/LIBERO'),LIBERO_PYTHON=str(ROOT/'envs/libero/bin/python'),
        LIBERO_CONFIG_PATH=str(ROOT/'config/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
    a.output.mkdir(parents=True)
    code_paths=[Path(__file__),ROOT/'scripts/persistent_libero.py',ROOT/'scripts/libero_worker_fast_png.py',ROOT/'scripts/libero_worker_pi05.py',
                ROOT/'src/AlphaBrain/AlphaBrain/training/reinforcement_learning/envs/libero_env_worker.py',
                ROOT/'src/openpi/src/openpi/policies/policy.py',ROOT/'src/openpi/src/openpi/policies/libero_policy.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    if a.record_features:
        feature_source=ROOT/'scripts/pi05_features.py'
        hashes[str(feature_source)]=hashlib.sha256(feature_source.read_bytes()).hexdigest()
    protocol=dict(model='pi05_libero',policy='baseline',suite=a.suite,split=a.split,tasks=a.tasks,initial_states=a.initials,
        seed=a.seed,action_horizon=a.max_steps,max_control_steps=a.max_steps,replan_steps=5,predicted_actions=10,warmup_steps=10,
        image_size=224,render_size=256,training_updates=0,collect_into_replay=a.record_features,
        feature_format='33x2048 frozen prefix features; 4x4 pooled tokens per active camera plus masked text mean' if a.record_features else None,
        noise='Explicit PCG64 normal(10,32), SeedSequence([seed,task,initial]); public Policy.infer noise argument',
        action_postprocess='Official LiberoOutputs and normalization; directly passed to controller, no Qwen clamp or gripper inversion',
        reset='original pinned fresh simulator per episode with persistent Python worker; OpenPI acos quaternion state convention',
        checkpoint_inventory_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),source_sha256=hashes,
        openpi_commit=subprocess.check_output(['git','-C',str(ROOT/'src/openpi'),'rev-parse','HEAD'],text=True).strip(),
        note='Local controlled baseline; different horizon/replan/noise protocol from historical Qwen evaluation and published benchmark.')
    write(a.output/'protocol.json',protocol)
    episodes=[];close_final=lambda:None;started=time.monotonic()
    def log(event,**kw):
        row=dict(event=event,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
        with(a.output/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
    try:
        import jax
        import numpy as np
        from openpi.policies import policy_config
        from openpi.training import config
        from openpi_client import image_tools
        from persistent_libero import install_persistent_env
        if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('No GPU device')
        cfg=config.get_config('pi05_libero')
        log('loading')
        t=time.monotonic();policy=policy_config.create_trained_policy(cfg,ROOT/'checkpoints/pi05_libero')
        extractor=None
        if a.record_features:
            from pi05_features import PrefixFeatures
            extractor=PrefixFeatures(policy)
        log('ready',load_seconds=time.monotonic()-t)
        source=ROOT/'src/AlphaBrain-opt/AlphaBrain/training/reinforcement_learning/envs/libero_env.py'
        spec=importlib.util.spec_from_file_location('pi_libero_proxy',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module._WORKER_SCRIPT=str(ROOT/'scripts/libero_worker_pi05.py')
        holder=SimpleNamespace(Env=module.LiberoEnv,output=a.output)
        close_final=install_persistent_env(holder)
        for task in a.tasks:
            for initial in a.initials:
                episode_start=time.monotonic();env=holder.Env();frames=[];records=[]
                obs=env.reset(a.suite,task,initial_state_idx=initial,seed=a.seed)
                for _ in range(10):obs,_,_=env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
                rng=np.random.default_rng(np.random.SeedSequence([a.seed,task,initial]))
                steps=0;calls=0;success=False;reward_sum=0.;infer_times=[];env_seconds=0.
                while steps<a.max_steps and not success:
                    element={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(obs['primary_image']),224,224)),
                             'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(obs['wrist_image']),224,224)),
                             'observation/state':np.asarray(obs['state'],dtype=np.float32),'prompt':env.task_description}
                    noise=rng.standard_normal((cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32)
                    t=time.monotonic();result=policy.infer(element,noise=noise);actions=np.asarray(result['actions'])
                    infer_times.append(time.monotonic()-t);calls+=1
                    if actions.shape!=(10,7) or not np.isfinite(actions).all():raise ValueError('Invalid policy action output')
                    if extractor is not None:
                        feature=extractor(element)
                        state=np.asarray(obs['state'],dtype=np.float32).copy()
                    chunk_reward=0.;duration=0
                    for action in actions[:5]:
                        t=time.monotonic();obs,reward,done=env.step(action);env_seconds+=time.monotonic()-t
                        steps+=1;reward_sum+=reward;success=bool(done and reward>.5)
                        chunk_reward+=(.99**duration)*reward*5.;duration+=1
                        if a.save_videos:frames.append(np.asarray(obs['primary_image']).copy())
                        if done or steps>=a.max_steps:break
                    if extractor is not None:
                        records.append((feature,state,actions[:5].copy(),chunk_reward,duration,bool(done or steps>=a.max_steps)))
                    if done:break
                if records:
                    arrays=dict(features=np.stack([x[0] for x in records]),states=np.stack([x[1] for x in records]),
                                reference=np.stack([x[2] for x in records]),rewards=np.asarray([x[3] for x in records],dtype=np.float32),
                                durations=np.asarray([x[4] for x in records],dtype=np.int32),terminal=np.asarray([x[5] for x in records],dtype=bool),
                                task=np.asarray(task),initial=np.asarray(initial),seed=np.asarray(a.seed))
                    np.savez_compressed(a.output/f'training_task{task:02d}_init{initial:02d}.npz',**arrays)
                row=dict(policy='baseline',suite=a.suite,task_id=task,task=env.task_description,initial_state=initial,seed=a.seed,
                         success=success,action_steps=steps,chunks=calls,reward=reward_sum,
                         inference_seconds=sum(infer_times),first_inference_seconds=infer_times[0],
                         median_inference_seconds=float(np.median(infer_times)),environment_step_seconds=env_seconds,
                         rollout_wall_seconds=time.monotonic()-episode_start)
                if a.save_videos:
                    name=f'baseline_task{task:02d}_init{initial:02d}_{"success" if success else "fail"}.mp4'
                    save_video(a.output/name,frames);row['video']=name
                episodes.append(row);write(a.output/'episodes.json',episodes);log('episode',**row);env.close()
        counts=dict(baseline=dict(successes=sum(e['success'] for e in episodes),episodes=len(episodes)))
        report=dict(status='complete',suite=a.suite,split=a.split,model='pi05_libero',counts=counts,episodes=episodes,
                    training_updates=0,wall_seconds=time.monotonic()-started,protocol=protocol)
        write(a.output/'summary.json',report);write(a.output/'run_summary.json',{k:v for k,v in report.items() if k not in ['episodes','protocol']})
        cells=[]
        for e in episodes:
            video=f'<video controls preload="metadata" width="256" src="{html.escape(e["video"],quote=True)}"></video>' if e.get('video') else ''
            cells.append(f'<section><b>任务{e["task_id"]} / 初态{e["initial_state"]}</b><p>{html.escape(e["task"])}</p><p>{"成功" if e["success"] else "失败"} · {e["action_steps"]}步</p>{video}</section>')
        page='<!doctype html><meta charset="utf-8"><title>π₀.₅-LIBERO 基线</title><style>body{font:16px/1.7 system-ui;margin:24px}main{display:flex;flex-wrap:wrap;gap:24px}section{width:280px}</style><h1>π₀.₅-LIBERO 基线</h1>'
        page+=f'<p>{html.escape(a.suite)}：成功 {counts["baseline"]["successes"]}/{len(episodes)}；仅基线、无RLT训练。{html.escape(a.split)}，{a.max_steps}步上限，每5步重新规划。尚不能与Qwen旧结果直接归因比较。</p><main>'+''.join(cells)+'</main>'
        (a.output/'index.html').write_text(page,encoding='utf-8');log('complete',counts=counts)
    except BaseException as exc:
        write(a.output/'run_summary.json',dict(status='failed',error=repr(exc),completed_episodes=len(episodes),training_updates=0))
        log('failed',error=repr(exc));raise
    finally:close_final()

if __name__=='__main__':main()
