#!/usr/bin/env python3
"""Read-only progress and paired video page for the local pi RLT experiment."""
import argparse
import html
import json
from pathlib import Path
import time


def refresh(out,pid):
    episodes=json.loads((out/'episodes.json').read_text()) if (out/'episodes.json').exists() else []
    baseline=[e for e in episodes if e['policy']=='baseline']
    train=[e for e in episodes if e['policy']=='train']
    epochs=sorted({e['training_episode'] for e in episodes if e['policy']=='candidate'})
    counts=[]
    for ep in epochs:
        rows=[e for e in episodes if e['policy']=='candidate' and e['training_episode']==ep]
        counts.append(dict(episode=ep,completed=len(rows),successes=sum(e['success'] for e in rows)))
    summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
    status='complete' if summary else 'failed' if (out/'run_error.json').exists() else 'running' if Path(f'/proc/{pid}').exists() else 'interrupted'
    parts=['<!doctype html><meta charset="utf-8"><title>双摩卡壶 RLT 优化</title>',
        '<meta http-equiv="refresh" content="15">' if status=='running' else '',
        '<style>body{font:16px/1.7 system-ui;margin:28px;max-width:1100px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}video{width:256px}section{margin:24px 0}</style>',
        f'<h1>双摩卡壶 RLT 优化 · {status}</h1>',
        f'<p>已采集30场基础训练数据；在线训练已完成 {len(train)}/60 回合。训练回合带探索噪声，其成功率不用于宣称提升。</p>',
        f'<p>冻结π基线：{sum(e["success"] for e in baseline)}/{len(baseline)}。BC-only使用零初始化残差，检查其参数保持不变，因此与π基线等价，没有重复计入评估回合。</p>',
        '<p>本地轻量RLT适配：冻结π与编码器，TD3训练有界残差Actor。并非官方已发布的π-RLT权重。</p>',
        '<table><tr><th>训练回合</th><th>验证已完成</th><th>成功</th></tr>']
    for c in counts:parts.append(f'<tr><td>{c["episode"]}</td><td>{c["completed"]}/10</td><td>{c["successes"]}</td></tr>')
    parts.append('</table>')
    if summary:
        parts.append(f'<p>最佳检查点：第{summary["best_episode"]}回合；验证成功{summary["best_validation_successes"]}/10。第0回合表示保留原始基线。尚未进行独立最终测试。</p>')
    complete=[c for c in counts if c['completed']==10]
    if complete:
        latest=summary['best_episode'] if summary and summary['best_episode']>0 else complete[-1]['episode']
        label='已选最佳验证' if summary and summary['best_episode']>0 else '最近完整验证'
        parts.append(f'<h2>{label}：第{latest}回合</h2><p>左：原始π；右：RLT候选。视频按控制帧播放，不代表真实推理速度。</p>')
        rows=[e for e in episodes if e['policy']=='candidate' and e['training_episode']==latest]
        for e in rows:
            b=next(x for x in baseline if x['initial_state']==e['initial_state'])
            parts.append(f'<section><b>初态{e["initial_state"]} · π {"成功" if b["success"] else "失败"} / RLT {"成功" if e["success"] else "失败"}</b><br>')
            for x in (b,e):parts.append(f'<video controls preload="none" src="{html.escape(x["video"],quote=True)}"></video> ')
            parts.append('</section>')
    parts.append('<p>训练初态0–29，验证30–39；最终测试40–49保留。</p>')
    temp=out/'index.html.tmp';temp.write_text(''.join(parts),encoding='utf-8');temp.replace(out/'index.html')
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args()
    while refresh(a.output,a.pid)=='running':time.sleep(10)
