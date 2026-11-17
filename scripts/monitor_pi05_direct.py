"""Read-only progress and paired videos for direct action-block training."""
import argparse,html,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
while True:
    events=[];rows=[];summary={}
    try:
        for line in (out/'events.jsonl').read_text().splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
        if (out/'episodes.json').exists():rows=json.loads((out/'episodes.json').read_text())
        if (out/'summary.json').exists():summary=json.loads((out/'summary.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):pass
    parts=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>双摩卡壶：完整动作块训练</title><style>body{font:16px/1.65 system-ui;margin:26px;max-width:1150px}td,th{padding:7px;border-bottom:1px solid #ddd}video{width:256px}section{margin:20px 0}</style>',
        '<h1>双摩卡壶：Actor 直接输出完整动作块</h1><p>公开 ActionTokenActor + 本地 π 特征编码器。没有残差相加。训练初态0–29；验证初态30–39、种子42和7。视频播放速度不代表推理耗时。</p>']
    parts.append('<p>状态：'+html.escape(summary.get('status','运行中'))+'</p>')
    if events:parts.append('<p>最新事件：'+html.escape(json.dumps(events[-1],ensure_ascii=False))+'</p>')
    parts.append('<table><tr><th>策略</th><th>训练回合</th><th>种子</th><th>成功/已完成</th></tr>')
    keys=sorted(set((r['policy'],r['training_episode'] if r['policy']!='train' else -1,r['seed']) for r in rows))
    for mode,ep,seed in keys:
        group=[r for r in rows if r['policy']==mode and (ep==-1 or r['training_episode']==ep) and r['seed']==seed]
        parts.append(f'<tr><td>{mode}</td><td>{ep}</td><td>{seed}</td><td>{sum(r["success"] for r in group)}/{len(group)}</td></tr>')
    parts.append('</table><p>探索训练成功率不等于冻结策略验证成功率；最优验证检查点也不是最终测试结果。</p>')
    candidates=[r for r in rows if r['policy']=='candidate']
    if candidates:
        selected=summary.get('best_episode',max(r['training_episode'] for r in candidates))
        for r in candidates:
            if r['training_episode']!=selected:continue
            base=next((b for b in rows if b['policy']=='baseline' and b['seed']==r['seed'] and b['initial_state']==r['initial_state']),None)
            if base is None:continue
            parts.append(f'<section><b>第{selected}回合 / 种子{r["seed"]} / 初态{r["initial_state"]}：π {"成功" if base["success"] else "失败"}，完整动作块 {"成功" if r["success"] else "失败"}</b><br>')
            for x in (base,r):parts.append(f'<video controls preload="none" src="{html.escape(x["video"],quote=True)}"></video>')
            parts.append('</section>')
    tmp=out/'index.html.tmp';tmp.write_text(''.join(parts),encoding='utf-8');tmp.replace(out/'index.html')
    if summary or (out/'run_error.json').exists():break
    time.sleep(10)
