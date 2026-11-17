#!/usr/bin/env python3
"""CPU-only audit/report for a completed optimize_rlt.py validation run."""
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

POLICIES = ('baseline', 'release', 'bc_only', 'candidate')
LABELS = {'baseline': '基础 VLA', 'release': '发布版 RLT_a',
          'bc_only': '仅 BC 校准', 'candidate': 'BC + TD3 候选'}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def exact_mcnemar(candidate_only, comparator_only):
    """Two-sided exact conditional binomial test, with p=1 for no discordance."""
    n = candidate_only + comparator_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(candidate_only, comparator_only) + 1))
    return min(1.0, float(Fraction(2 * tail, 1 << n)))


def bootstrap_macro(rows, tasks, repetitions=10000):
    """Resample matched initials within each fixed task; retain equal task weights."""
    grouped = {task: [r['success_delta'] for r in rows if r['task_id'] == task] for task in tasks}
    sizes = {str(task): len(values) for task, values in grouped.items()}
    result = dict(ci_95=None, method='paired_initial_resampling_within_fixed_tasks',
                  repetitions=repetitions, seed=20260908, pairs_per_task=sizes)
    if not grouped or any(len(values) < 2 for values in grouped.values()):
        result['reason'] = '至少一个任务不足 2 个配对初始状态，不能估计任务内初始状态变异；不报告置信区间。'
        return result
    try:
        import numpy as np
    except ImportError:
        result['reason'] = '当前 Python 未安装 numpy，未执行可选 bootstrap。'
        return result
    rng = np.random.default_rng(result['seed'])
    draws = np.zeros(repetitions)
    for values in grouped.values():
        values = np.asarray(values, dtype=float)
        indices = rng.integers(len(values), size=(repetitions, len(values)))
        draws += values[indices].mean(axis=1) / len(grouped)
    result['ci_95'] = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
    result['reason'] = ('仅描述固定任务集合、观测初始状态的重采样差异；不包括新任务、训练种子或选模偏差。'
                        '少量样本或结果无变异时区间可能退化，不能据此认定没有不确定性。')
    return result


def key(episode):
    return (int(episode['task_id']), int(episode['initial_state']), episode.get('seed'))


def summarize_episodes(episodes):
    successful = [e for e in episodes if e['success']]
    return dict(successes=len(successful), episodes=len(episodes),
                success_rate=len(successful) / len(episodes) if episodes else None,
                successful_action_steps_mean=mean([e['action_steps'] for e in successful]),
                all_action_steps_mean=mean([e['action_steps'] for e in episodes]),
                action_steps_sum=sum(e['action_steps'] for e in episodes),
                successful_action_steps=[e['action_steps'] for e in successful])


def paired_comparison(index, comparator, tasks):
    candidate = index.get('candidate', {})
    reference = index.get(comparator, {})
    shared = candidate.keys() & reference.keys()
    rows = []
    for k in sorted(shared, key=lambda k: (k[0], k[1], str(k[2]))):
        a, b = candidate[k], reference[k]
        rows.append(dict(task_id=k[0], initial_state=k[1], seed=k[2],
                         candidate_success=bool(a['success']), comparator_success=bool(b['success']),
                         success_delta=int(a['success']) - int(b['success']),
                         candidate_action_steps=a['action_steps'], comparator_action_steps=b['action_steps'],
                         step_delta_if_both_success=a['action_steps'] - b['action_steps']
                         if a['success'] and b['success'] else None))
    wins = sum(r['success_delta'] == 1 for r in rows)
    losses = sum(r['success_delta'] == -1 for r in rows)
    both = [r['step_delta_if_both_success'] for r in rows if r['step_delta_if_both_success'] is not None]
    per_task = {str(task): summarize_pairs([r for r in rows if r['task_id'] == task]) for task in tasks}
    macro = mean([p['success_rate_delta'] for p in per_task.values()]) if all(p['pairs'] for p in per_task.values()) else None
    return dict(comparator=comparator, pairs=len(rows), candidate_only=wins, comparator_only=losses,
                both_success=sum(r['candidate_success'] and r['comparator_success'] for r in rows),
                both_fail=sum(not r['candidate_success'] and not r['comparator_success'] for r in rows),
                unpaired_candidate=len(candidate) - len(rows), unpaired_comparator=len(reference) - len(rows),
                net_success_count_delta=wins - losses, macro_success_rate_delta=macro,
                exact_mcnemar_two_sided_p=exact_mcnemar(wins, losses),
                both_success_step_delta_mean=mean(both), both_success_pairs=len(both),
                per_task=per_task, bootstrap=bootstrap_macro(rows, tasks), episodes=rows)


def summarize_pairs(rows):
    return dict(pairs=len(rows), candidate_only=sum(r['success_delta'] == 1 for r in rows),
                comparator_only=sum(r['success_delta'] == -1 for r in rows),
                success_rate_delta=mean([r['success_delta'] for r in rows]),
                both_success_step_delta_mean=mean([r['step_delta_if_both_success'] for r in rows
                                                   if r['step_delta_if_both_success'] is not None]))


def interpretation(comparisons, smoke):
    lines = []
    if smoke:
        lines.append('这是流程 smoke 验证；成功率和步数只用于检查运行，不构成有效性证据。')
    base = comparisons.get('baseline', {}).get('macro_success_rate_delta')
    bc = comparisons.get('bc_only', {}).get('macro_success_rate_delta')
    release = comparisons.get('release', {}).get('macro_success_rate_delta')
    if base is None:
        lines.append('缺少完整的基础 VLA 配对，无法判断候选相对基础模型的变化。')
    elif base > 0:
        lines.append('候选在本次验证集上的成功率高于基础 VLA；这只是选模阶段观察，尚未在锁定测试集上确认。')
    elif base < 0:
        lines.append('候选在本次验证集上的成功率低于基础 VLA，尚未达到提升基础模型的目标。')
    else:
        lines.append('候选与基础 VLA 在本次验证集上成功率相同；不能据此宣称成功率提升。')
    if bc is not None and bc > 0:
        lines.append('候选的成功率高于仅 BC 校准快照，观察到了后续联合训练的额外增益；'
                     '由于后续还包含 BC 损失、额外更新和新采集数据，不能将其完全归因于 TD3。')
    elif bc is not None and bc == 0:
        lines.append('候选与仅 BC 校准快照成功率相同，没有观察到后续 TD3 + BC 训练的净成功率增益。')
        if release is not None and release > 0:
            lines.append('相对发布版的净改善已由仅 BC 校准快照达到；目前不能把该改善算作强化学习收益。')
    elif bc is not None:
        lines.append('候选成功率低于仅 BC 校准快照，后续联合训练在这批验证样本上出现净退步。')
    else:
        lines.append('没有仅 BC 校准对照，无法区分恢复基础 VLA 行为与后续强化学习的收益。')
    lines.append('配对步数只比较双方都成功的回合，负值代表候选步数更少；失败回合提前结束不算速度提升。')
    return lines


def build_report(run):
    inputs = {name: run / name for name in ('latest_validation.json', 'run_summary.json', 'events.jsonl',
                                          'config.json', 'code_sha256.json')}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise ValueError('缺少完整运行报告，暂不推测最终结果：' + ', '.join(missing))
    validation, summary, config, code = [read_json(inputs[name]) for name in
                                       ('latest_validation.json', 'run_summary.json', 'config.json', 'code_sha256.json')]
    if summary.get('status') != 'complete':
        raise ValueError('运行尚未标记 complete，不能生成最终报告')
    if validation.get('split') != 'validation_only':
        raise ValueError('本工具只汇总 validation_only；锁定测试需要独立的测试报告')
    events = [json.loads(line) for line in inputs['events.jsonl'].read_text().splitlines() if line.strip()]
    episodes = validation['episodes']
    reserved = set(config.get('test_initials_reserved', []))
    used_initials = {e['initial_state'] for e in episodes}
    if reserved & used_initials:
        raise ValueError('验证回合使用了声明保留的测试初始状态，不能标记为未触碰测试集')
    if set(config.get('train_initials', [])) & used_initials:
        raise ValueError('验证初始状态与训练初始状态重叠，不能作为此协议下的验证报告')
    tasks = sorted(int(task) for task in validation['tasks'])
    index = defaultdict(dict)
    for episode in episodes:
        policy, k = episode['policy'], key(episode)
        if k in index[policy]:
            raise ValueError(f'重复的配对键：{policy} {k}')
        index[policy][k] = episode
    policies = [p for p in POLICIES if p in index]
    counts = {p: summarize_episodes(list(index[p].values())) for p in policies}
    for p in policies:
        expected = validation.get('counts', {}).get(p, {})
        for field in ('successes', 'episodes'):
            if field in expected and counts[p][field] != expected[field]:
                raise ValueError(f'validation counts 与逐回合记录不一致：{p} {field}')
    per_task = {str(task): {p: summarize_episodes([e for e in index[p].values() if e['task_id'] == task])
                            for p in policies} for task in tasks}
    macro = {p: mean([per_task[str(task)][p]['success_rate'] for task in tasks])
             if all(per_task[str(task)][p]['episodes'] for task in tasks) else None for p in policies}
    comparisons = {p: paired_comparison(index, p, tasks) for p in ('baseline', 'release', 'bc_only') if p in index}
    collected = [e for e in events if e.get('event') == 'episode' and e.get('collect')]
    observed = dict(collected_episodes=len(collected), environment_action_steps=sum(e['action_steps'] for e in collected),
                    collected_by_policy=dict(Counter(e['policy'] for e in collected)),
                    critic_updates=sum(e.get('event') in ('update', 'critic_pretrain_update') for e in events),
                    actor_updates=sum(e.get('event') == 'update' and 'actor_loss' in e for e in events),
                    bc_updates=sum(e.get('event') == 'bc_update' for e in events),
                    all_validation_episode_events=sum(e.get('event') == 'episode' and not e.get('collect') for e in events))
    progress = summary['progress']
    checks = dict(collected_episodes=observed['collected_episodes'] == progress['warmup_done'] + progress['episodes_done'],
                  environment_action_steps=observed['environment_action_steps'] == progress['env_steps'],
                  critic_updates=observed['critic_updates'] == progress['critic_steps'],
                  actor_updates=observed['actor_updates'] == progress['actor_steps'],
                  bc_updates=observed['bc_updates'] == progress['bc_steps'])
    warnings = []
    if not all(checks.values()):
        warnings.append('部分事件计数与 checkpoint 进度不一致；恢复运行可能留下未提交事件，请分别审计，不采用配置中的计划数量代替实际数量。')
    if any(c['unpaired_candidate'] or c['unpaired_comparator'] for c in comparisons.values()):
        warnings.append('存在未配对回合；配对统计仅使用相同任务、初始状态和 seed 的交集。')
    singleton = any(per_task[str(t)][p]['episodes'] < 2 for t in tasks for p in policies)
    limitations = ['所有结果来自 validation_only 选模集合，没有锁定测试集结果，不能报告最终泛化增益。',
                   '单一训练随机种子，尚未测量不同训练种子的稳定性。',
                   'McNemar p 值以配对初始状态独立为前提，不修正多重比较，也不覆盖选模偏差；仅用于探索。',
                   'BC-only 是初始校准快照，不是与候选等更新次数、等数据的纯 BC 控制组。']
    if singleton:
        limitations.insert(1, '至少一个任务每种策略仅 1 个初始状态；N=1/任务不足以作为性能 benchmark，也不能估计任务内变异。')
    validation_dirs = [e.get('directory') for e in events if e.get('event') == 'validation_complete']
    source = {path: dict(recorded_sha256=digest, current_sha256=sha256(Path(path)) if Path(path).is_file() else None)
              for path, digest in code.items()}
    for audit in source.values():
        audit['current_matches_run'] = audit['recorded_sha256'] == audit['current_sha256']
    return dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(), run=str(run),
                status=summary['status'], split=validation.get('split'), locked_test_evaluated=False,
                tasks=tasks, initial_states=validation['initial_states'], policies=policies,
                counts=counts, macro_success_rate=macro, per_task=per_task, comparisons=comparisons,
                interpretation=interpretation(comparisons, summary.get('smoke', False)),
                limitations=limitations, warnings=warnings,
                training=dict(actual_progress=progress, event_observations=observed, event_progress_agreement=checks,
                              wall_seconds=summary.get('wall_seconds'),
                              peak_gpu_allocated_gib=summary.get('peak_gpu_allocated_gib'),
                              peak_gpu_reserved_gib=summary.get('peak_gpu_reserved_gib'),
                              frozen_vla_encoder_verified=summary.get('frozen_vla_encoder_verified'),
                              actor_parameter_l2_change_from_release=summary.get('actor_parameter_l2_change_from_release')),
                provenance=dict(input_sha256={name: sha256(path) for name, path in inputs.items()},
                                config=config, source_code=source,
                                latest_validation_directory=validation_dirs[-1] if validation_dirs else None),
                episodes=episodes)


def number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def render_markdown(report):
    p, training = report['policies'], report['training']
    progress = training['actual_progress']
    lines = ['# RLT_a 优化验证报告', '', f"运行目录：`{report['run']}`", '',
             '本报告汇总实际执行结果；验证集用于选模，保留测试初始状态尚未在本报告中评估。', '']
    lines.extend('- ' + item for item in report['interpretation'])
    lines += ['', '## 成功率与完成步数', '',
              '| 策略 | 成功 / 回合 | 任务等权成功率 | 成功回合平均动作步数 |',
              '|---|---:|---:|---:|']
    for policy in p:
        item = report['counts'][policy]
        lines.append(f"| {LABELS[policy]} | {item['successes']}/{item['episodes']} | "
                     f"{number(report['macro_success_rate'][policy] * 100 if report['macro_success_rate'][policy] is not None else None)}% | "
                     f"{number(item['successful_action_steps_mean'])} |")
    lines += ['', '各任务单元格为“成功 / 回合；成功回合平均步数”。不同策略的成功回合集合可能不同，'
              '速度结论应以随后双方均成功的配对差为准。', '',
              '| 任务 | ' + ' | '.join(LABELS[x] for x in p) + ' |',
              '|---|' + '---:|' * len(p)]
    for task in report['tasks']:
        task_name = next((e.get('task', '') for e in report['episodes'] if e['task_id'] == task), '')
        cells = []
        for policy in p:
            item = report['per_task'][str(task)][policy]
            cells.append(f"{item['successes']}/{item['episodes']}；{number(item['successful_action_steps_mean'])}")
        lines.append(f'| {task}: {task_name} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 候选策略的配对对照', '',
              '| 对照策略 | 配对数 | 候选独成功 / 对照独成功 | 成功率宏平均差（百分点） | 双方成功步数差 | exact McNemar p |',
              '|---|---:|---:|---:|---:|---:|']
    for comparator, item in report['comparisons'].items():
        delta = item['macro_success_rate_delta']
        lines.append(f"| {LABELS[comparator]} | {item['pairs']} | {item['candidate_only']} / {item['comparator_only']} | "
                     f"{number(delta * 100 if delta is not None else None)} | "
                     f"{number(item['both_success_step_delta_mean'])}（{item['both_success_pairs']} 对） | "
                     f"{number(item['exact_mcnemar_two_sided_p'], 6)} |")
    lines += ['', '成功率差和步数差均为“候选 − 对照”；步数差为负才表示成功回合中步数更少。']
    for comparator, item in report['comparisons'].items():
        boot = item['bootstrap']
        ci = boot['ci_95']
        interval = f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}] 个百分点" if ci else 'null'
        lines += ['', f"- 对 {LABELS[comparator]} 的任务内配对 bootstrap 95% 区间：{interval}。{boot['reason']}"]
    lines += ['', '## 实际训练资源与更新', '',
              f"- 基础 VLA 采集预热：{progress['warmup_done']} 回合；候选在线采集：{progress['episodes_done']} 回合。",
              f"- 训练采集动作步数：{progress['env_steps']}；不含验证回合及重置时的预热动作。",
              f"- Critic 更新：{progress['critic_steps']}（其中预校准 {progress.get('critic_pretrain_steps', 0)}）；"
              f"Actor TD3 + BC 更新：{progress['actor_steps']}；初始纯 BC 校准：{progress['bc_steps']}。",
              f"- 已完成验证轮次：{progress['validation_runs']}；本报告只统计最新一轮，避免把重复选模评估当作独立样本。",
              f"- 运行计时：{number(training['wall_seconds'])} 秒；"
              f"PyTorch 峰值 allocated / reserved：{number(training['peak_gpu_allocated_gib'])} / "
              f"{number(training['peak_gpu_reserved_gib'])} GiB（该进程分配器统计，不是整卡总占用）。",
              f"- VLA 和 encoder 冻结审计：{training['frozen_vla_encoder_verified']}；"
              f"事件计数与提交进度一致：{all(training['event_progress_agreement'].values())}。",
              '', '## 证据边界', '']
    lines.extend('- ' + item for item in report['limitations'] + report['warnings'])
    lines += ['', '## 可追溯文件', '',
              '- [逐项统计与配对回合](analysis.json)',
              '- [最新验证原始记录](latest_validation.json)',
              '- [实际运行摘要](run_summary.json)',
              '- [训练事件](events.jsonl)',
              '- [运行配置](config.json)',
              '- [运行时源码 SHA-256](code_sha256.json)', '',
              'analysis.json 记录输入文件哈希、原始配置、运行时源码哈希与当前源码匹配情况。'
              '当前源码后来发生变化时，以运行时记录为准。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        report = build_report(run)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    (run / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (run / 'REPORT.zh-CN.md').write_text(render_markdown(report))
    print(json.dumps(dict(report=str(run / 'REPORT.zh-CN.md'), counts=report['counts'],
                          macro_success_rate=report['macro_success_rate'],
                          warnings=report['warnings']), ensure_ascii=False))


if __name__ == '__main__':
    main()
