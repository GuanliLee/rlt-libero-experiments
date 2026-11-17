#!/usr/bin/env python3
"""Live paired playback using the exact frozen optimization episode path."""
import argparse
import html
import json
import os
import queue
import random
import traceback
from datetime import datetime
from pathlib import Path
import compare_live as viewer
from evaluate_optimized_policy import FrozenEvaluation, sha256
from evaluate_rlt_blends import make_blend_actor
from optimize_rlt import ROOT, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--alpha',type=float,default=1.)
    parser.add_argument('--port',type=int,default=7861)
    parser.add_argument('--autostart',action='store_true')
    args=parser.parse_args()
    args.run=args.run.resolve()
    if not 0 < args.alpha <= 1:parser.error('alpha must be in (0,1]')
    if json.loads((args.run/'run_summary.json').read_text()).get('status')!='complete':
        parser.error('A completed frozen training run is required')
    config=json.loads((args.run/'config.json').read_text())
    for filename,digest in json.loads((args.run/'code_sha256.json').read_text()).items():
        if sha256(Path(filename))!=digest:raise RuntimeError('Training source changed: '+filename)
    os.environ['RLT_SOURCE_ROOT']=config['source_root']
    actor_identity=sha256(args.run/'actor.pt')
    viewer.HTML=viewer.HTML.replace('VLA + RLT_a',f'优化 RLT · α={args.alpha:g}')
    viewer.HTML=viewer.HTML.replace('<h1>基础 VLA / RLT_a · 实时仿真对照</h1>',
        '<h1>基础 VLA / 优化 RLT · 实时仿真对照</h1><p>'+html.escape(args.run.name)+
        f' · 固定混合系数 α={args.alpha:g} · 使用冻结评估的同一动作执行流程</p>')

    class OptimizedApp(viewer.App):
        def worker(self):
            try:
                import numpy as np
                import torch
                output=ROOT/'results'/('optimized_live_session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                output.mkdir()
                cfg=config.copy();cfg.update(output=args.run,resume=True)
                self.exp=FrozenEvaluation(argparse.Namespace(**cfg),output)
                supplied=torch.load(args.run/'actor.pt',map_location='cpu',weights_only=True)
                if not all(torch.equal(v.cpu(),supplied[k]) for k,v in self.exp.actor.state_dict().items()):
                    raise RuntimeError('Actor file and training checkpoint disagree')
                if config.get('architecture')=='bounded_residual':
                    from evaluate_residual_rlt import wrap_residual
                    self.exp.actor=wrap_residual(self.exp.actor,config['delta_max']*args.alpha)
                else:
                    self.exp.actor=make_blend_actor(self.exp.actor,args.alpha)
                write_json(output/'deployment.json',dict(run=str(args.run),alpha=args.alpha,
                    actor_sha256=actor_identity,architecture=config.get('architecture','released_full_action_RLT_a'),
                    delta_max=config.get('delta_max'),training_updates=0,episode_path='optimize_rlt.Experiment.episode'))
                self.update(ready=True,busy=False,message=f'优化模型已就绪，α={args.alpha:g}。点击运行对照。')
                if self.autostart:self.submit(dict(task=0,initial=0,episodes=1))
                while not self.stop.is_set():
                    try:job=self.jobs.get(timeout=.5)
                    except queue.Empty:continue
                    try:self.compare(job)
                    except Exception as exc:
                        traceback.print_exc();self.update(busy=False,message='执行失败：'+str(exc))
            except Exception as exc:
                traceback.print_exc();self.update(ready=False,busy=False,message='模型加载失败：'+str(exc))

        def episode(self,mode,task,initial,episode,output):
            import numpy as np
            import torch
            original_env=self.exp.Env
            app=self
            class StreamingEnv:
                def __init__(self):self.env=original_env();self.count=0
                def reset(self,*a,**kw):
                    obs=self.env.reset(*a,**kw)
                    app.publish(mode,obs,status='执行中',phase='初始化',success=None,initial=initial,episode=episode+1,steps=0)
                    return obs
                def step(self,action):
                    if app.stop.is_set():raise RuntimeError('Live session stopped')
                    obs,reward,done=self.env.step(action);self.count+=1
                    app.publish(mode,obs,phase='执行中',steps=max(0,self.count-10))
                    return obs,reward,done
                def close(self):self.env.close()
                def __getattr__(self,key):return getattr(self.env,key)
            self.exp.Env=StreamingEnv
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            try:
                result=self.exp.episode(task,initial,'baseline' if mode=='baseline' else 'candidate',video_dir=output)
            finally:self.exp.Env=original_env
            result.update(policy=mode,episode=episode,run=str(args.run),alpha=args.alpha,actor_sha256=actor_identity)
            self.publish(mode,success=result['success'],status='完成',delta_mean_abs=result['mean_abs_action_difference'])
            return result

    viewer.App=OptimizedApp
    viewer.serve(args)


if __name__=='__main__':main()
