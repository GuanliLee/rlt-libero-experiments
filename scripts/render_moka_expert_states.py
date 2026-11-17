#!/usr/bin/env python3
"""Re-render only training expert states at the benchmark's 256px resolution."""
import argparse,json,sys,time
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/LIBERO'))
from libero.libero.envs import OffScreenRenderEnv
from libero_worker_pi05 import quat2axisangle

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('New output required')
    a.output.mkdir(parents=True);start=time.monotonic()
    manifest=json.loads((ROOT/'manifests/moka_expert_download.json').read_text());assert manifest['status']=='verified'
    success=json.loads((ROOT/'manifests/moka_expert_success_audit.json').read_text());assert len(success)==30 and all(x['success_after_last_action'] for x in success)
    env=OffScreenRenderEnv(bddl_file_name=str(ROOT/'src/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl'),camera_heights=256,camera_widths=256)
    rows=[]
    try:
        with h5py.File(manifest['local_path'],'r') as f:
            for demo in range(30):
                env.reset();g=f['data'][f'demo_{demo}'];n=len(g['actions']);assert n<=520
                frames=[];wrists=[];states=[];sim_indices=[];durations=[];rewards=[];actions=[]
                for t in range(0,n-1,5):
                    # states[j] is BEFORE action[j]; render states[t+1] and execute actions[t+1:...].
                    obs=env.set_init_state(g['states'][t+1]);duration=min(5,n-1-t)
                    frames.append(np.asarray(obs['agentview_image'])[::-1,::-1].copy());wrists.append(np.asarray(obs['robot0_eye_in_hand_image'])[::-1,::-1].copy())
                    states.append(np.concatenate([obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']]).astype(np.float32))
                    action=np.zeros((5,7),dtype=np.float64);action[:duration]=g['actions'][t+1:t+1+duration];actions.append(action)
                    sim_indices.append(t+1);durations.append(duration);rewards.append(sum(.99**k*float(g['rewards'][t+1+k])*5 for k in range(duration)))
                np.savez_compressed(a.output/f'demo{demo:02d}.npz',images=np.stack(frames),wrist=np.stack(wrists),states=np.stack(states),actions=np.stack(actions),durations=np.asarray(durations),rewards=np.asarray(rewards),sim_indices=np.asarray(sim_indices),demo=demo)
                row=dict(demo=demo,frames=len(frames),steps=n-1);rows.append(row);print(json.dumps(row),flush=True)
    finally:env.close()
    (a.output/'summary.json').write_text(json.dumps(dict(status='complete',demos=rows,wall_seconds=time.monotonic()-start,source=manifest,render_resolution=256,orientation='rotated180 matching benchmark',alignment='demo.states[t+1] paired with demo.actions[t+1:t+6]',training_demonstrations_only=True,benchmark_initial_ids_are_separate=True),indent=2)+'\n')
if __name__=='__main__':main()
