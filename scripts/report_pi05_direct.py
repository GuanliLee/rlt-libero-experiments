"""Summarize all direct-action validation checkpoints, without cherry-picking seeds."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
summary=json.loads((out/'summary.json').read_text());rows=json.loads((out/'episodes.json').read_text())
assert summary['status']=='complete'
metrics=[]
for mode,ep in sorted(set((r['policy'],r['training_episode']) for r in rows if r['policy'] in ('baseline','candidate','bc_only'))):
    subset=[r for r in rows if r['policy']==mode and r['training_episode']==ep]
    metrics.append(dict(policy=mode,episode=ep,successes=sum(r['success'] for r in subset),episodes=len(subset),
        by_seed={str(seed):dict(successes=sum(r['success'] for r in subset if r['seed']==seed),episodes=sum(r['seed']==seed for r in subset)) for seed in (42,7)}))
selected=[r for r in rows if r['policy']=='candidate' and r['training_episode']==summary['best_episode']]
comparisons={}
for mode in ['baseline','bc_only']:
    base={(r['seed'],r['initial_state']):r for r in rows if r['policy']==mode}
    comparisons[mode]=dict(wins=[(r['seed'],r['initial_state']) for r in selected if r['success'] and not base[(r['seed'],r['initial_state'])]['success']],
        losses=[(r['seed'],r['initial_state']) for r in selected if not r['success'] and base[(r['seed'],r['initial_state'])]['success']])
losses=[json.loads(x) for x in (out/'training_losses.jsonl').read_text().splitlines()]
lossstats={}
for key in ['critic_loss','actor_loss','bc_penalty','q_actor','target_mean']:
    vals=[r[key] for r in losses if key in r]
    lossstats[key]=dict(first100_mean=statistics.mean(vals[:100]),last100_mean=statistics.mean(vals[-100:]),count=len(vals))
result=dict(summary=summary,all_checkpoints=metrics,selected_paired_comparisons=comparisons,loss_statistics=lossstats)
(out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
parts=['# 双摩卡壶：完整动作块 Actor 结果','',
    'Actor直接输出完整5×7动作块，没有残差相加。复用公开ActionTokenActor，但π特征编码器仍为本地适配；不是完整论文复现。','',
    '| 策略 | 训练回合 | 种子42 | 种子7 | 合计 |','|---|---:|---:|---:|---:|']
for m in metrics:
    s=m['by_seed'];parts.append(f'| {m["policy"]} | {m["episode"]} | {s["42"]["successes"]}/{s["42"]["episodes"]} | {s["7"]["successes"]}/{s["7"]["episodes"]} | {m["successes"]}/{m["episodes"]} |')
parts+=['',f'选择第{summary["best_episode"]}回合的模型；BC-only使用该回合配套权重，并实际执行20个对照回合。若第0回合最佳，意味着RL没有超过模仿初始化。',
    '',f'相对π救回：{comparisons["baseline"]["wins"]}；退步：{comparisons["baseline"]["losses"]}。',
    f'相对BC-only救回：{comparisons["bc_only"]["wins"]}；退步：{comparisons["bc_only"]["losses"]}。','',
    '这些场景全部属于验证/模型选择，不能当作独立最终测试。不能将多个随机种子下的相同初态简单当作完全独立的新场景。',
    '',f'在线训练{summary["collected_training_episodes"]}回合；Actor更新{summary["actor_updates"]}次，Critic TD更新{summary["critic_td_updates"]}次，另有500次MC预热。总流程耗时{summary["wall_seconds"]/60:.1f}分钟（含初始化及仿真验证）。',
    '', '## 损失的意义','',
    '- Critic loss：预测的未来折扣奖励与训练目标之间的平方误差。目标也在变，不能仅凭下降判定策略提升。',
    '- Actor loss：负Q值加偏离π参考动作的BC惩罚；可以是负值。',
    '- BC penalty：完整动作块与π参考动作的平方差之和，再取批次均值。',
    '- Q值：Critic的预测，不是真实成功率。真实效果以冻结策略执行的成功数为准。','',
    '| 指标 | 最初100次均值 | 最后100次均值 |','|---|---:|---:|']
for k,v in lossstats.items():parts.append(f'| {k} | {v["first100_mean"]:.6g} | {v["last100_mean"]:.6g} |')
parts+=['','视频入口：同目录 `index.html`。源码：`source_snapshot/`。权重：`best.pt`；训练日志：`training_losses.jsonl`。']
(out/'report.zh-CN.md').write_text('\n'.join(parts)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
