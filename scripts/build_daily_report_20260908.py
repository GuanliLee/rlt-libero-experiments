#!/usr/bin/env python3
"""Read completed experiment logs and export a reproducible daily report."""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/daily_report_20260908'
OUT.mkdir(exist_ok=True)
RUN = ROOT / 'results/optimization_v2_20260908_beta3'
events = [json.loads(line) for line in (RUN / 'events.jsonl').read_text().splitlines()]
updates = [e for e in events if e['event'] == 'update']
actors = [e for e in updates if 'actor_loss' in e]
validation = json.loads((RUN / 'latest_validation.json').read_text())
run_summary = json.loads((RUN / 'run_summary.json').read_text())
sources = [RUN / 'events.jsonl', RUN / 'latest_validation.json', RUN / 'run_summary.json', RUN / 'config.json']

def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

metrics = []
for event, keys, window in [
    ('bc_update', ['bc_loss'], 20),
    ('critic_pretrain_update', ['critic_loss'], 20),
    ('update', ['critic_loss', 'actor_loss', 'q_actor_mean', 'bc_penalty', 'bc_control_penalty'], 100),
]:
    for key in keys:
        values = [e[key] for e in events if e['event'] == event and key in e]
        assert values and all(math.isfinite(v) for v in values)
        metrics.append(dict(phase=event, metric=key, count=len(values), window=window,
                            first=values[0], last=values[-1],
                            early_mean=st.mean(values[:window]), late_mean=st.mean(values[-window:])))
write_csv('loss_summary.csv', metrics)
write_csv('training_losses.csv', [{k: v for k, v in e.items() if not isinstance(v, (dict, list))}
                                  for e in events if e['event'] in ['bc_update', 'critic_pretrain_update', 'update']])

cohorts = [
    ('v2 主实验', validation, 'original', RUN / 'latest_validation.json'),
]
for label, name, reset in [
    ('v2 动作混合', 'optimization_v2_blend_validation_20260908', 'single_build'),
    ('MC v3 离线继续训练', 'offline_mc_v3_validation_20260908', 'single_build'),
    ('有界残差 v4', 'offline_residual_v4_validation_20260908', 'original'),
    ('有界残差 v4 幅度减半', 'offline_residual_v4_half_validation_20260908', 'original'),
]:
    path = ROOT / 'results' / name / 'summary.json'
    cohorts.append((label, json.loads(path.read_text()), reset, path))
    sources.append(path)

success_rows = []
for label, cohort, reset, path in cohorts:
    eps = cohort['episodes']
    base = {(e['task_id'], e['initial_state']): e for e in eps if e['policy'] == 'baseline'}
    assert len(base) == 30
    for policy, count in cohort['counts'].items():
        current = {(e['task_id'], e['initial_state']): e for e in eps if e['policy'] == policy}
        assert current.keys() == base.keys()
        successes = sum(e['success'] for e in current.values())
        assert successes == count['successes'] and len(current) == count['episodes']
        wins = sum(current[k]['success'] and not base[k]['success'] for k in base)
        losses = sum(base[k]['success'] and not current[k]['success'] for k in base)
        both = [k for k in base if base[k]['success'] and current[k]['success']]
        bsteps = st.mean(base[k]['action_steps'] for k in both) if both else None
        psteps = st.mean(current[k]['action_steps'] for k in both) if both else None
        success_rows.append(dict(cohort=label, reset=reset, policy=policy, successes=successes,
            episodes=len(current), success_rate=successes / len(current),
            delta_percentage_points=100 * (wins-losses) / len(current), paired_wins=wins,
            paired_losses=losses, joint_success_count=len(both),
            joint_success_baseline_mean_steps=bsteps, joint_success_policy_mean_steps=psteps,
            source=str(path)))
write_csv('validation_success_rates.csv', success_rows)

names = ['打开柜子中间抽屉', '把碗放到炉灶上', '把酒瓶放到柜子顶部', '打开上层抽屉并把碗放进去',
         '把碗放到柜子顶部', '把盘子推到炉灶前方', '把奶油奶酪放进碗里', '打开炉灶', '把碗放到盘子上', '把酒瓶放到架子上']
task_rows = []
for task in range(10):
    eps = [e for e in validation['episodes'] if e['task_id'] == task]
    row = dict(task_id=task, task=names[task], original_task=eps[0]['task'])
    row.update({p: sum(e['success'] for e in eps if e['policy'] == p)
                for p in ['baseline', 'release', 'bc_only', 'candidate']})
    row['episodes_per_policy'] = 3
    task_rows.append(row)
write_csv('per_task_validation.csv', task_rows)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
def series(ax, records, key, label, color, xkey, raw=False):
    x = np.asarray([e[xkey] for e in records])
    y = np.asarray([e[key] for e in records])
    if raw:
        ax.plot(x, y, color=color, alpha=.16, linewidth=.5)
    ax.plot(x[99:], np.convolve(y, np.ones(100)/100, mode='valid'), color=color, label=label)
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=8)
series(axs[0,0], updates, 'critic_loss', 'Twin-Q TD loss', '#2864b0', 'critic_steps', True)
axs[0,0].set(title='Critic: fitting moving TD targets', xlabel='Cumulative critic updates', ylabel='MSE(Q1, target) + MSE(Q2, target)')
series(axs[0,1], actors, 'actor_loss', 'Actor objective', '#b75b14', 'actor_steps', True)
axs[0,1].axhline(0, color='gray', linewidth=.6)
axs[0,1].set(title='Actor: negative values are valid', xlabel='RL actor updates', ylabel='- mean Q1 + 3 * BC penalty')
for key, label, color in [('q1_mean','Q1','#2864b0'), ('q2_mean','Q2','#b75b14'), ('target_mean','TD target','#25835d')]:
    series(axs[1,0], updates, key, label, color, 'critic_steps')
axs[1,0].set(title='Predicted returns, not measured success rates', xlabel='Cumulative critic updates', ylabel='Batch mean discounted return')
for key, label, color in [('bc_penalty','Candidate BC penalty','#2864b0'), ('bc_control_penalty','BC-only penalty','#b75b14')]:
    series(axs[1,1], actors, key, label, color, 'actor_steps')
axs[1,1].set(title='Action deviation from VLA reference', xlabel='RL actor / matched BC updates', ylabel='Mean over batch of 56-D squared-error sum')
fig.suptitle('RLT v2 | 2026-09-08 | Online phase | Trailing mean over 100 updates', fontsize=13)
fig.text(.5, .015, 'Training objectives improved; validation success remained 21/30 for both baseline and candidate.', ha='center', fontsize=10)
fig.tight_layout(rect=[0,.045,1,.95])
fig.savefig(OUT / 'training_losses.png', dpi=180)
fig.savefig(OUT / 'training_losses.pdf')
plt.close(fig)

selected = [r for r in success_rows if r['cohort'] == 'v2 主实验']
fig, ax = plt.subplots(figsize=(8,4.6))
labels = ['Base VLA', 'Published RLT', 'BC-only', 'Trained RLT v2']
rates = [r['success_rate']*100 for r in selected]
bars = ax.bar(labels, rates, color=['#6b7280','#2864b0','#25835d','#b75b14'], width=.6)
for bar, row in zip(bars, selected):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f"{row['successes']}/30 ({row['success_rate']:.1%})", ha='center')
ax.set(ylim=(0,100), ylabel='Validation success rate (%)', title='Same 30 paired scenes | 10 tasks x 3 initial states')
ax.grid(axis='y',alpha=.2)
ax.set_axisbelow(True)
fig.text(.5,.025,'Exploratory validation only. Published-weight gains are not gains from local training.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1])
fig.savefig(OUT / 'validation_success.png', dpi=180)
fig.savefig(OUT / 'validation_success.pdf')
plt.close(fig)

labelmap = {'baseline':'基础 VLA', 'release':'公开 RLT', 'bc_only':'BC-only 对照', 'candidate':'本地训练 RLT v2'}
main_table = table(['策略','成功数','成功率','相比基础 VLA','救回 / 退步场景'],
    [[labelmap[r['policy']],f"{r['successes']}/30",f"{r['success_rate']:.1%}",f"{r['delta_percentage_points']:+.2f} 个百分点",
      f"{r['paired_wins']} / {r['paired_losses']}"] for r in selected])
loss_labels = ['初始 BC 校准损失','初始 Critic 校准损失','在线 Critic 损失','在线 Actor 总损失','Actor 动作的 Q1 预测均值','候选 Actor 的 BC 惩罚','BC-only 的 BC 惩罚']
loss_table = table(['指标','比较窗口','前段均值','后段均值'],
    [[label,f"各 {m['window']} 次更新",f"{m['early_mean']:.6f}",f"{m['late_mean']:.6f}"] for label,m in zip(loss_labels,metrics)])
speed_table = table(['策略','共同成功场景数','基础 VLA 平均步数','该策略平均步数','步数变化'],
    [[labelmap[r['policy']],r['joint_success_count'],f"{r['joint_success_baseline_mean_steps']:.2f}",
      f"{r['joint_success_policy_mean_steps']:.2f}",
      f"{(r['joint_success_policy_mean_steps']/r['joint_success_baseline_mean_steps']-1)*100:+.2f}%"]
     for r in selected if r['policy']!='baseline'])
variant_table = table(['实验','策略','成功数','成功率','相对本组 baseline','重置方式'],
    [[r['cohort'],r['policy'],f"{r['successes']}/{r['episodes']}",f"{r['success_rate']:.1%}",
      f"{r['delta_percentage_points']:+.2f} 个百分点",r['reset']] for r in success_rows if r['cohort']!='v2 主实验'])
task_table = table(['任务编号','任务','基础 VLA','公开 RLT','BC-only','候选 v2'],
    [[r['task_id'],r['task']]+[f"{r[p]}/3" for p in ['baseline','release','bc_only','candidate']] for r in task_rows])

report = f'''**RLT 实验日报：2026-09-08；整理日期：2026-09-09**

本报告根据已完成的本地日志重新统计。实验采用 AlphaBrain 的 QwenOFT + RLT_a、LIBERO-Goal，不是原论文 π 模型的严格复现。今天整理报告没有启动新训练。全部文件位于 `{ROOT}`。

**可直接用于汇报的结论：**完成 RLT 单卡训练流程修复、150 回合仿真数据采集和多组受控验证。主实验在线 Critic 损失的前后各 100 次更新均值由 0.086733 降至 0.057591，Actor 总目标由 0.170892 降至 -0.818181，说明优化器在更新并改变训练目标；但本地候选策略在 30 个配对验证场景中的成功率为 70.0%，与基础 VLA 持平，尚未证实强化学习带来任务效果提升。公开 RLT 为 73.3%，BC-only 为 76.7%，两者均不能归因于本轮 RL 优化。

**完成工作及其意义。**修复部分更新配置下 Actor 不更新、TD 目标受同批其他奖励不合理裁剪、多任务采样遗漏等问题；对空采集与工作进程缺失增加失败检查。6 项相关回归测试通过，记录见 `manifests/optimization_cpu_regressions.txt`。这些测试验证代码行为，不是任务成功率测试；尚无修复前后同条件性能实验，不能给这些修复宣称成功率增益。

建立匹配初始参数、优化器状态、训练样本、噪声和更新次数的 BC-only 对照，用来检查 Q 项是否提供额外收益。保存配置、权重、完整经验池与日志，并建立同任务同初态的视频对比报告。

**主实验训练配置与规模。**

{table(['项目','实际值'],[
['设备','RTX 5090 32GB'],['环境','LIBERO-Goal，10 个任务'],['冻结模块','基础 VLA 与已发布 RLT 编码器'],
['训练模块','Actor 约 45.6 万参数；双 Q Critic 约 85.5 万参数；另有 BC-only 对照'],
['初始化','Actor 加载公开权重；Critic 重新初始化'],['采集','基础 VLA 50 回合 + 候选策略 100 回合'],
['有效控制步骤 / 经验条数','24,808 / 3,147 个动作块（不含每回合 10 个初始静置步）'],
['Actor 更新','初始 BC 128 次 + 带 Q 的更新 1,600 次'],['Critic 更新','初始校准 128 次 + 在线 3,200 次 = 3,328 次'],
['BC-only 更新','与候选匹配：初始 128 次 + 后续 1,600 次'],['每回合更新','Critic 32 次、Actor 16 次'],
['批量大小','配置 128；日志实际为 120，每个任务 12 条'],['学习率','Actor 1e-5；Critic 3e-4，Adam'],
['奖励 / 折扣','成功单步奖励 5，其他 0；gamma=0.99，动作块 bootstrap 使用 0.99^8'],
['BC 权重 / 探索噪声 / 参考丢弃','3 / 标准差 0.05 / 10%（训练时）'],
['动作块 / 回合上限','8 步 / 320 控制步骤'],['总运行耗时','约 27.83 分钟，包含加载、采集、训练和 120 回合验证，不是纯训练耗时'],
['显存统计','PyTorch 峰值 allocated 8.61 GiB、reserved 8.88 GiB；不是整卡总占用']])}

训练采集阶段基础 VLA 成功 38/50（76%），带探索噪声的动态候选策略成功 68/100（68%）。两组初态不同，后者策略持续变化且加噪声，因此这两个比例不能作为前后性能比较。训练允许初态 0–29，本轮实际使用 0–14：热身 0–4，在线采集 5–14。

**成功率如何统计。**主验证固定 10 个任务、每任务初态 31–33，共 30 个场景；4 个策略各跑一次，共 120 回合。成功率 = 环境判定成功的回合数 / 总回合数。验证关闭探索、不更新权重、不写入训练经验池。一个回合即一个任务从初态开始的一次完整尝试，不是一个动作块。每任务样本数相同，总成功率与任务成功率宏平均相同。

{main_table}

![主实验成功率](validation_success.png)

候选 v2 与基础 VLA 不仅总成功率相同，30 个场景的成功/失败结果逐项一致：没有救回失败，也没有新增失败。公开 RLT 多救回 1 个场景；BC-only 多救回 2 个。30 场景中单个回合就相当于 3.33 个百分点，不能把这些小差异写成已证实的稳定提升。公开 RLT 的表现来自下载的权重；BC-only 不使用 Q 项。

**执行效率。**仅在每一对策略都成功的场景中比较控制步数，避免失败提前结束被误当成更快。各对这里恰好都有 21 个共同成功场景。表内不是视频播放速度，也不是计算机推理耗时。

{speed_table}

候选 v2 平均仅少约 0.67 步（0.63%），不足以称为明显加速。公开 RLT 在这批共同成功场景中少约 5.34% 步数；BC-only 多约 10.96% 步数。均为当前验证集的描述性结果，没有独立复测的稳定性结论。

**训练损失是什么。**Critic loss = Q1 与 TD 目标的均方误差 + Q2 与该目标的均方误差，衡量“预测器是否拟合当前训练目标”；目标也随训练变化，因此下降不能证明估值真实准确或已收敛。Actor loss = -平均 Q1 + 3 × BC 惩罚，是一个复合优化目标，可以为负数；不是错误率。BC 惩罚衡量 Actor 动作与 VLA 参考动作的平方差，在 56 个动作分量上求和，再对批量求平均。Q 值是预测折扣回报，不是实际成功率。

为减少单个随机 batch 的偶然性，初始校准比较最前/最后各 20 次更新均值，在线阶段比较最前/最后各 100 次有效更新均值。Critic 与 Actor 更新频率不同，各自按自己的记录计数；这些是训练批次统计，不是固定验证损失。

{loss_table}

![在线训练损失与预测回报](training_losses.png)

图中实线为 100 次更新的后向移动平均；部分面板浅色线为原始批次值。仅展示 v2 在线阶段，未把初始校准混入。Critic 曲线先明显下降，后半段又有回升和波动，并非持续单调下降；不能只依据首尾窗口称其已稳定收敛。Q 与 TD 目标是在同一批 replay 动作上统计；Actor 的 q_actor_mean 使用 Actor 生成动作，因此不是完全相同的统计对象。

Actor 后段均值可以直接拆开：-1.421613 + 3 × 0.201144 = -0.818181。损失降低主要伴随预测 Q 增长，BC 惩罚变化不大。这只能说明训练目标变化，不能排除 Q 估值偏差。0.0576 的 Critic loss 不是“错误率 5.76%”；0.2011 的 BC 惩罚也不是“20.11% 动作出错”。

初始 BC 校准与在线 BC 惩罚不能直接相减判断变差：初始校准使用确定性输出、没有参考丢弃；在线 Actor 训练使用标准差 0.05 的动作采样和参考丢弃。仅 56 维高斯采样噪声的期望平方和就约为 56×0.05²=0.14。尚未计算独立固定数据集上的 Critic/Actor 验证损失，不能声称训练损失与验证损失同时下降。

**其他优化方向及结果。**动作混合以 alpha 控制候选动作占比，alpha=0 为基础 VLA、alpha=1 为完整候选。MC v3 利用已有 150 回合的实际行为折扣回报先训练新 Critic，再进行离线 TD3 更新；没有新采集数据，且同时改变初始化与更新量，不是单因素消融。v4 改为自定义有界残差结构：VLA参考动作 + 0.1×tanh(网络输出)，幅度减半版本将修正量再乘 0.5；它不是原版 RLT_a 架构。

{variant_table}

表中 single_build 为尝试加速的重置方式。扩大检查后发现任务9初态31的基础 VLA 轨迹步数存在 131 与 127 的差异，虽然成功结果相同，因此不能把这些组与 original 组视作完全一致的仿真条件。每组内部策略使用同一重置方式并有 baseline 配对；跨组不混合步数统计。后续 v4 两组采用 original，最终测试也应采用 original。MC v3 及动作混合结果仅作各自组内的探索性证据。

MC v3 的 MC 回归损失首条/末条日志分别为 14.180850 / 0.186842；v4 为 13.726853 / 0.175611。它们是不同随机批次的端点，不等价于固定数据集误差下降比例，也不能与 v2 的 TD loss 混为一种指标；这两个分支的日志经过间隔采样，未伪造完整更新曲线。

**各任务结果与瓶颈。**每个数字分母均为 3。

{task_table}

任务5（推盘子）和任务6（奶油奶酪入碗）四种策略都为 0/3，值得优先分析失败轨迹；样本只有 3，不能断言这两个任务在所有初态下必然失败。任务0基础 VLA 与候选仅 1/3。多数其他任务已经 3/3，因此增加这些容易场景的训练不一定能暴露关键短板。

**下一步计划（未完成事项）。**优先回看任务0、5、6失败视频，区分定位、抓取、推动或放置阶段的问题；用固定数据检查 Critic 对成功与失败动作的预测，确定是否提供可靠指导；在训练初态内补充针对性的交互数据，并保留匹配 BC 对照。冻结候选和评估规则后，再在预留初态40–49进行 10任务×10初态=100个场景的配对测试。这批最终测试尚未运行。不能根据最终测试结果继续调参后仍将它称为独立测试。

本轮已完成的是工程修复、可追溯训练与验证；尚未完成的是“让本地训练 RLT 明显优于基础 VLA”的效果目标。工程修复没有单独测得端到端提速比例，仿真加速也未通过全面等价性验证，不写成已验证性能提升。

**交付与溯源。**`training_losses.csv` 为三段主训练日志中的标量记录，`loss_summary.csv` 为统计窗口结果，`validation_success_rates.csv` 为全部上述验证组的配对统计，`per_task_validation.csv` 为主实验逐任务成功数。两张图片同时提供 PNG 与 PDF。`source_manifest.json` 保存原始输入文件 SHA-256，方便复核。主训练代码为 `scripts/optimize_rlt_v2.py`，采集逻辑为 `scripts/optimize_rlt.py`；原始日志和权重在 `results/optimization_v2_20260908_beta3/`。主实验视频报告在 `results/optimization_v2_20260908_beta3/validation_001_ep00100/index.html`。
'''
(OUT / '日报_RLT_20260908.md').write_text(report, encoding='utf-8')
summary = dict(experiment_date='2026-09-08', report_date='2026-09-09', main_success=selected,
               loss_metrics=metrics, all_validation=success_rows, per_task=task_rows,
               training=run_summary, independent_final_test_completed=False,
               conclusion='No demonstrated local RLT success-rate improvement over baseline on current validation.')
(OUT / 'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
sources += [ROOT / 'manifests/optimization_cpu_regressions.txt', ROOT / 'manifests/single-build-expanded-check.json',
            ROOT / 'results/offline_mc_v3_20260908/events.jsonl', ROOT / 'results/offline_residual_v4_20260908/events.jsonl', Path(__file__)]
(OUT / 'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
print(json.dumps({'output':str(OUT),'files':[p.name for p in sorted(OUT.iterdir())],
                  'online_updates':len(updates),'actor_updates':len(actors)},ensure_ascii=False))
