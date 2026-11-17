#!/usr/bin/env python3
"""Fit a compact RLT autoencoder using training trajectories only."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from pi05_rlt_networks import TokenAutoencoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=800);a=p.parse_args()
    summary=json.loads((a.data/'summary.json').read_text())
    if summary['status']!='complete' or summary['split']!='training_probe':raise ValueError('Incomplete or nontraining data')
    if a.output.exists():raise ValueError('Choose new output')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42)
    train=[];valid=[]
    for f in sorted(a.data.glob('training_task08_init*.npz')):
        with np.load(f) as d:
            initial=int(d['initial']);assert 0<=initial<30
            (train if initial<24 else valid).append(d['features'])
    if len(train)!=24 or len(valid)!=6:raise ValueError('Expected24 train and6 trajectory-heldout initial states')
    x=np.concatenate(train);v=np.concatenate(valid);del train,valid
    model=TokenAutoencoder();mean=x.astype(np.float32).mean(axis=(0,1));std=x.astype(np.float32).std(axis=(0,1)).clip(.1)
    model.mean.copy_(torch.from_numpy(mean));model.std.copy_(torch.from_numpy(std))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);best=float('inf');best_state=None;start=time.monotonic();history=[]
    # Fixed heldout batches make reconstruction checkpoints comparable.
    vi=rng.choice(len(v),min(256,len(v)),replace=False)
    @torch.no_grad()
    def evaluate():
        model.eval();losses=[];zeros=[]
        for start in range(0,len(vi),16):
            data=torch.from_numpy(v[vi[start:start+16]]).float();target=model.normalize(data)
            losses.append(float((model(data)-target).square().mean()));zeros.append(float(target.square().mean()))
        return float(np.mean(losses)),float(np.mean(zeros))
    initial_loss,zero_loss=evaluate()
    for step in range(1,a.steps+1):
        model.train();data=torch.from_numpy(x[rng.integers(len(x),size=32)]).float()
        loss=(model(data)-model.normalize(data)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reconstruction loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0 or step==a.steps:
            val,zero=evaluate();row=dict(step=step,train_loss=float(loss.detach()),heldout_loss=val,zero_predictor_loss=zero,wall_seconds=time.monotonic()-start)
            history.append(row);print(json.dumps(row),flush=True)
            if val<best:best=val;best_state=copy.deepcopy(model.state_dict());best_step=step
            (a.output/'losses.json').write_text(json.dumps(history,indent=2)+'\n')
    if not best<zero_loss*.9:raise RuntimeError('Compression did not beat zero predictor by10%; do not start RL')
    torch.save(dict(model=best_state,architecture='pooled33x2048 -> projection128 +2 transformer layers/4heads -> RLT256; decoder33x128 ->2048',
                    best_step=best_step,initial_loss=initial_loss,heldout_loss=best,zero_loss=zero_loss),a.output/'encoder.pt')
    report=dict(status='complete',training_updates=a.steps,best_step=best_step,heldout_loss=best,zero_predictor_loss=zero_loss,
                training_initials=list(range(24)),representation_validation_initials=list(range(24,30)),
                task_validation_initials_used=False,wall_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
