#!/usr/bin/env python3
"""Align official POST-ACTION observations with future expert actions for TD3."""
import argparse,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from evaluate_pi05_baseline import ROOT,write
from pi05_direct_networks import TokenAutoencoder
from train_pi05_moka_direct import Replay

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('New output required')
    download=json.loads((ROOT/'manifests/moka_expert_download.json').read_text());assert download['status']=='verified'
    checks=json.loads((ROOT/'manifests/moka_expert_success_audit.json').read_text())
    demos=[x['demo'] for x in checks if x['success_after_last_action'] and x['steps']<=520]
    assert set(demos)<=set(range(30)) and len(demos)>=20,'Too few verified training demonstrations'
    alignment=json.loads((ROOT/'manifests/moka_expert_alignment_audit.json').read_text())
    assert all(x['state_max_error']<.001 for x in alignment if x['state_index']==x['observation_index']+1)
    a.output.mkdir(parents=True);torch.set_num_threads(4);start_time=time.monotonic()
    ckpt=torch.load(ROOT/'results/pi05_moka_direct_20260909_b/bc_initial.pt',weights_only=True,map_location='cpu')
    encoder=TokenAutoencoder();encoder.load_state_dict(ckpt['encoder']);encoder.eval().requires_grad_(False)
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_client import image_tools
    from pi05_features import PrefixFeatures
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),ROOT/'checkpoints/pi05_libero');extract=PrefixFeatures(policy)
    protocol=dict(source=download,expert_demo_ids=demos,expert_demo_ids_are_not_benchmark_initial_ids=True,benchmark_validation_initials=list(range(30,40)),benchmark_test_initials=list(range(40,50)),
        training_coverage='Independent official training demonstrations0-29, in addition to simulator training initial0-29; no validation/test rollout observations.',
        alignment='obs[t] was recorded AFTER action[t]; pair obs[t] with actions[t+1:t+6] and next_obs[t+5]',
        images='Official128x128 RGB, rotate180 degrees then resize/pad224 with same pi preprocessing',reward='Stored final success verified in simulator; reward5 and canonical per-step gamma.99 before later relabeling',
        actor_bc_target='Frozen pi reference, NOT expert action; expert action is the actual action for Critic replay only',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(a.output/'protocol.json',protocol);replay=Replay();episodes=[]
    with h5py.File(download['local_path'],'r') as f:
        prompt=json.loads(f['data'].attrs['problem_info'])['language_instruction']
        for demo in demos:
            g=f['data'][f'demo_{demo}'];n=len(g['actions']);rng=np.random.default_rng(np.random.SeedSequence([1701,8,demo]))
            front=g['obs/agentview_rgb'][:];wrist=g['obs/eye_in_hand_rgb'][:]
            states=np.concatenate([g['obs/ee_pos'][:],g['obs/ee_ori'][:],g['obs/gripper_states'][:]],axis=1).astype(np.float32)
            recorded=g['actions'][:];rewards=g['rewards'][:];records=[];raw=[]
            for t in range(0,n-1,5):
                element={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(front[t,::-1,::-1].copy(),224,224)),
                    'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist[t,::-1,::-1].copy(),224,224)),
                    'observation/state':states[t],'prompt':prompt}
                ref=np.asarray(policy.infer(element,noise=rng.standard_normal((10,32)).astype(np.float32))['actions'])[:5].copy()
                features=extract(element)
                with torch.no_grad():z=encoder.encode(torch.from_numpy(features).float()[None])[0].numpy()
                duration=min(5,n-1-t);action=ref.copy();action[:duration]=recorded[t+1:t+1+duration]
                reward=float(sum(.99**k*float(rewards[t+1+k])*5 for k in range(duration)))
                records.append((z,states[t],ref,action,reward,duration));raw.append(features)
            arrays=[np.stack([x[k] for x in records]) for k in range(6)];replay.add_episode(*arrays)
            np.savez_compressed(a.output/f'expert_demo{demo:02d}.npz',features=np.stack(raw),z=arrays[0],states=arrays[1],reference=arrays[2],actions=arrays[3],rewards=arrays[4],durations=arrays[5],demo=demo)
            row=dict(demo=demo,original_steps=n,aligned_steps=n-1,chunks=len(records));episodes.append(row);write(a.output/'episodes.json',episodes);print(json.dumps(row),flush=True)
    replay.save(a.output/'replay.npz')
    write(a.output/'summary.json',dict(status='complete',expert_episodes=len(episodes),transitions=len(replay.rows),wall_seconds=time.monotonic()-start_time,validation_or_test_rollouts_used=0))
if __name__=='__main__':main()
