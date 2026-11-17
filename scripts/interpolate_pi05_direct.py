#!/usr/bin/env python3
"""Conservatively interpolate a trained direct-action Actor toward exact imitation."""
import argparse, hashlib, json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--trained',type=Path,required=True)
    p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 0.<a.alpha<1. or a.output.exists():p.error('alpha must be in (0,1) and output must be new')
    base=torch.load(a.identity,weights_only=True,map_location='cpu')
    trained=torch.load(a.trained,weights_only=True,map_location='cpu')
    out=dict(trained)
    actor={}
    for key,value in base['actor'].items():
        other=trained['actor'][key]
        actor[key]=value+(other-value)*a.alpha if value.is_floating_point() else other
    out['actor']=actor
    out['bc']=base['actor']
    out['episode']=trained['episode']
    out['interpolation_alpha']=a.alpha
    out['architecture']='published direct interpolated ActionTokenActor; local pooled pi encoder; task8 only'
    out['interpolation']=dict(
        identity=str(a.identity),trained=str(a.trained),alpha=a.alpha,
        identity_sha256=hashlib.sha256(a.identity.read_bytes()).hexdigest(),
        trained_sha256=hashlib.sha256(a.trained.read_bytes()).hexdigest(),
        meaning='Actor still directly emits the full 5x7 chunk; only trainable weights are interpolated.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,a.output)
    a.output.with_suffix('.json').write_text(json.dumps(out['interpolation'],indent=2)+'\n')

if __name__=='__main__':main()
