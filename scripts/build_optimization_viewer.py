#!/usr/bin/env python3
"""Build a Chinese, offline, paired LIBERO optimization viewer using only the stdlib.

Examples (run from RLT):
  python scripts/build_optimization_viewer.py --evaluation results/pilot_expanded_validation_20260908
  python scripts/build_optimization_viewer.py --evaluation results/optimization_pilot_20260908_beta3
  python scripts/build_optimization_viewer.py --evaluation results/blend_evaluation --candidate-policy candidate_alpha_025
  python scripts/serve_optimization_results.py

An in-progress evaluation produces an explicitly incomplete snapshot. Re-run this
command after more episodes finish. No source evaluation files are modified.
"""
import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


TASK_NAMES = [
    '打开中间抽屉', '把碗放在炉灶上', '把酒瓶放在柜子上',
    '打开上层抽屉并放入碗', '把碗放在柜子上', '把盘子推到炉灶前',
    '把奶油奶酪放进碗', '打开炉灶', '把碗放到盘子上', '把酒瓶放在架子上',
]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relative_url(path, output):
    return quote(Path(os.path.relpath(path, output)).as_posix(), safe='/')


def identity(ep):
    return (int(ep['task_id']), int(ep['initial_state']), ep.get('seed'))


def episode_fingerprint(episodes):
    return sorted(json.dumps(ep, sort_keys=True, ensure_ascii=False) for ep in episodes)


def load_evaluation(evaluation):
    """Select one evaluation, never combine distinct validation rounds."""
    source_digests = {}

    def read_snapshot(path):
        content = path.read_bytes()
        source_digests[path] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    summary_path = evaluation / 'summary.json'
    latest_path = evaluation / 'latest_validation.json'
    episodes_path = evaluation / 'episodes.json'
    protocol_path = evaluation / 'protocol.json'
    protocol = read_snapshot(protocol_path) if protocol_path.is_file() else {}
    sources = [protocol_path] if protocol_path.is_file() else []
    media_root = evaluation
    if episodes_path.is_file():
        raw = read_snapshot(episodes_path)
        episodes = raw['episodes'] if isinstance(raw, dict) else raw
        summary = read_snapshot(summary_path) if summary_path.is_file() else {}
        sources += [episodes_path] + ([summary_path] if summary_path.is_file() else [])
    elif summary_path.is_file():
        summary = read_snapshot(summary_path)
        if 'episodes' not in summary:
            raise ValueError('summary.json 没有 episodes，不能从聚合数字重建逐回合结果')
        episodes = summary['episodes']
        sources.append(summary_path)
    elif latest_path.is_file():
        summary = read_snapshot(latest_path)
        episodes = summary['episodes']
        sources.append(latest_path)
        expected = episode_fingerprint(episodes)
        matches = []
        for directory in sorted(evaluation.glob('validation_*')):
            child_episodes = directory / 'episodes.json'
            child_summary = directory / 'summary.json'
            if child_episodes.is_file():
                child = read_json(child_episodes)
                child = child['episodes'] if isinstance(child, dict) else child
            elif child_summary.is_file():
                child = read_json(child_summary).get('episodes', [])
            else:
                continue
            if episode_fingerprint(child) == expected:
                matches.append(directory)
        if len(matches) == 1:
            media_root = matches[0]
        elif len(matches) > 1:
            raise ValueError('多个 validation_* 目录匹配 latest_validation；请将 --evaluation 指向具体轮次，避免误选视频')
        elif any(ep.get('video') and not (evaluation / ep['video']).is_file() for ep in episodes):
            raise ValueError('无法把 latest_validation 回合匹配到 validation_* 视频目录；请指定具体轮次')
    else:
        raise FileNotFoundError(f'{evaluation} 没有 episodes.json、summary.json 或 latest_validation.json')
    if not isinstance(episodes, list):
        raise ValueError('episodes 必须为数组')
    return episodes, summary, protocol, media_root, [(p, source_digests[p]) for p in sources]


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(episodes):
    successes = sum(ep['success'] for ep in episodes)
    return dict(episodes=len(episodes), successes=successes,
                success_rate=successes / len(episodes) if episodes else None,
                successful_steps_mean=mean([ep['action_steps'] for ep in episodes if ep['success']]))


def exact_mcnemar(wins, losses):
    discordant = wins + losses
    if not discordant:
        return 1.0
    return min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)


def quantile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def bootstrap_macro(rows, tasks, repetitions=10000, seed=42):
    # Resample initial states (clusters of evaluation seeds), retaining pairs.
    strata = {task: defaultdict(list) for task in tasks}
    for row in rows:
        strata[row['task_id']][row['initial_state']].append(row['success_delta'])
    result = dict(method='paired_initial_state_resampling_within_fixed_tasks',
                  estimand='macro_success_rate_delta_percentage_points', repetitions=repetitions,
                  seed=seed, ci_95_percentage_points=None,
                  initial_states_per_task={str(t): len(strata[t]) for t in tasks})
    if not tasks or any(len(strata[t]) < 2 for t in tasks):
        result['reason'] = '至少一个纳入任务不足 2 个完整配对初态；不估计任务内 bootstrap 区间。'
        return result
    grouped = [[values for values in strata[t].values()] for t in tasks]
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        task_means = []
        for groups in grouped:
            selected = [groups[rng.randrange(len(groups))] for _ in groups]
            task_means.append(sum(sum(g) for g in selected) / sum(len(g) for g in selected))
        samples.append(100 * mean(task_means))
    samples.sort()
    result['ci_95_percentage_points'] = [quantile(samples, 0.025), quantile(samples, 0.975)]
    result['reason'] = '固定当前任务，在各任务内有放回重采样配对初态；不覆盖训练种子或模型选择的不确定性。'
    return result


def compare(index, policy, baseline, tasks):
    candidate, reference = index.get(policy, {}), index.get(baseline, {})
    shared = candidate.keys() & reference.keys()
    rows = []
    for key in sorted(shared, key=lambda k: (k[0], k[1], str(k[2]))):
        a, b = candidate[key], reference[key]
        rows.append(dict(task_id=key[0], initial_state=key[1], seed=key[2],
                         candidate_success=a['success'], baseline_success=b['success'],
                         success_delta=int(a['success']) - int(b['success']),
                         candidate_only=bool(a['success'] and not b['success']),
                         baseline_only=bool(b['success'] and not a['success']),
                         step_delta_if_both_success=a['action_steps'] - b['action_steps']
                         if a['success'] and b['success'] else None))
    wins = sum(r['candidate_only'] for r in rows)
    losses = sum(r['baseline_only'] for r in rows)
    task_means = [mean([r['success_delta'] for r in rows if r['task_id'] == task]) for task in tasks]
    steps = [r['step_delta_if_both_success'] for r in rows if r['step_delta_if_both_success'] is not None]
    return dict(policy=policy, baseline=baseline, pairs=len(rows), candidate_only=wins,
                baseline_only=losses, both_success=len(steps),
                both_fail=sum(not r['candidate_success'] and not r['baseline_success'] for r in rows),
                unpaired_candidate=len(candidate) - len(rows), unpaired_baseline=len(reference) - len(rows),
                micro_success_rate_delta_percentage_points=100 * (wins - losses) / len(rows) if rows else None,
                macro_success_rate_delta_percentage_points=100 * mean(task_means)
                if task_means and all(x is not None for x in task_means) else None,
                exact_mcnemar_two_sided_p=exact_mcnemar(wins, losses) if rows else None,
                joint_success_mean_steps_delta=mean(steps), joint_success_pairs=len(steps),
                bootstrap=bootstrap_macro(rows, tasks), episodes=rows)


def build(evaluation, output=None, candidate_policy='candidate', baseline_policy='baseline'):
    evaluation = evaluation.resolve()
    output = (output or evaluation).resolve()
    raw, summary, protocol, media_root, source_paths = load_evaluation(evaluation)
    warnings, episodes, index = [], [], defaultdict(dict)
    for ep in raw:
        item = dict(ep)
        if not isinstance(item.get('policy'), str) or not item['policy']:
            raise ValueError('每个回合必须有非空策略名')
        if not isinstance(item.get('success'), bool):
            raise ValueError(f"success 必须为 JSON boolean：{item}")
        item['task_id'], item['initial_state'], item['action_steps'] = map(int,
            (item['task_id'], item['initial_state'], item['action_steps']))
        if item['action_steps'] < 0:
            raise ValueError('action_steps 不能小于零')
        item.setdefault('seed', None)
        key = identity(item)
        if key in index[item['policy']]:
            raise ValueError(f"重复回合：{item['policy']} task/initial/seed={key}；不会静默丢弃重跑结果")
        item.update(video_url=None, video_status='not_recorded')
        if item.get('video'):
            path = Path(item['video'])
            options = [path] if path.is_absolute() else [media_root / path, evaluation / path]
            paths = [p.resolve() for p in options]
            if any(not p.is_relative_to(evaluation) for p in paths):
                raise ValueError(f"视频路径越出评估目录：{item['video']}")
            found = next((p for p in paths if p.is_file()), None)
            if found:
                item.update(video_url=relative_url(found, output), video_status='available')
            else:
                item['video_status'] = 'missing'
        index[item['policy']][key] = item
        episodes.append(item)
    declared_policies = (protocol.get('policies') or
                         [item['label'] for item in protocol.get('policy_plan', [])] or
                         list(summary.get('counts', {})))
    policies = list(dict.fromkeys(list(declared_policies) + [ep['policy'] for ep in episodes]))
    observed_tasks = {ep['task_id'] for ep in episodes}
    tasks = sorted(set(map(int, protocol.get('tasks', summary.get('tasks', [])))) | observed_tasks)
    initials = sorted(set(map(int, protocol.get('initial_states', summary.get('initial_states', [])))) |
                      {ep['initial_state'] for ep in episodes})
    counts = {p: summarize(list(index[p].values())) for p in policies}
    per_task = {str(t): {p: summarize([ep for ep in episodes if ep['policy'] == p and ep['task_id'] == t])
                        for p in policies} for t in sorted(set(range(10)) | set(tasks))}
    macro = {p: mean([per_task[str(t)][p]['success_rate'] for t in tasks])
             if tasks and all(per_task[str(t)][p]['episodes'] for t in tasks) else None for p in policies}
    expected_per_policy = len(tasks) * len(initials) if tasks and initials else None
    # Exact key coverage also catches missing cells concealed by extra seeds.
    expected_cells = {(t, i) for t in tasks for i in initials}
    complete = bool(summary) and all(expected_cells <= {(k[0], k[1]) for k in index[p]} for p in policies)
    for p in policies:
        declared = summary.get('counts', {}).get(p, {})
        if any(declared[k] != counts[p][k] for k in ('episodes', 'successes') if k in declared):
            warnings.append(f'{p} 的 summary 聚合值与当前逐回合记录不同；本页使用全部逐回合记录重新统计。')
    if candidate_policy not in policies:
        warnings.append(f'指定候选 {candidate_policy} 尚无记录或未纳入本次评估。')
    if baseline_policy not in policies:
        warnings.append(f'对照 {baseline_policy} 尚无记录或未纳入本次评估。')
    split = str(protocol.get('split', summary.get('split', 'unknown')))
    if split in ('validation', 'validation_only'):
        split_label = '验证集 · validation（用于选型）'
        split_note = '这批 validation 可用于选择策略、混合系数和检查点；不能当作最终测试集上的提升证据。'
    elif split in ('test', 'reserved_test', 'reserved_local_test', 'locked_test'):
        split_label = '本地保留测试集 · reserved local test'
        split_note = '这些初态仅对本次本地继续训练保留；不能称为发布模型历史上从未见过的数据。若据此继续选型，应重新标为开发数据。'
    else:
        split_label, split_note = f'集合未确认 · {split}', '来源未明确声明 validation 或本地保留 test；不推测其独立性。'
    missing = sum(ep['video_status'] != 'available' for ep in episodes)
    if missing:
        warnings.append(f'{missing} 个回合未录制或视频文件缺失；这些回合仍全部计入统计。')
    if not complete:
        warnings.append('这是尚未确认全部完成的静态快照；剩余回合不计为失败。评估完成后需重新生成。')
    comparisons = {p: compare(index, p, baseline_policy, tasks) for p in policies if p != baseline_policy}
    # A selected baseline is a valid deployment choice; retain a self-comparison.
    if candidate_policy == baseline_policy and baseline_policy in policies:
        comparisons[candidate_policy] = compare(index, candidate_policy, baseline_policy, tasks)
    report = dict(
        schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
        evaluation=str(evaluation), media_directory=relative_url(media_root, output),
        model='AlphaBrain 第三方 QwenOFT + RLT_a / LIBERO Goal；不是 π 原论文复现',
        split=split, split_label=split_label, split_note=split_note,
        status='complete' if complete else 'incomplete_snapshot',
        candidate_policy=candidate_policy, baseline_policy=baseline_policy,
        policies=policies, evaluation_tasks=tasks, initial_states=initials,
        expected_episodes_per_policy=expected_per_policy, total_recorded_episodes=len(episodes),
        counts=counts, macro_success_rate=macro, per_task=per_task,
        tasks=[dict(task_id=t, name=TASK_NAMES[t] if 0 <= t < len(TASK_NAMES) else f'任务 {t}',
                    included=t in tasks) for t in sorted(set(range(10)) | set(tasks))],
        comparisons=comparisons, episodes=episodes, protocol=protocol, warnings=warnings,
        source_files=[dict(name=p.name, url=relative_url(p, output), sha256=digest)
                      for p, digest in source_paths],
        limitations=[
            '完整展示成功与失败；总成功率按已记录回合计算，宏平均按本次纳入任务等权计算。未完成任务不作为失败填入。',
            '配对使用相同 task、initial state 和 seed；轨迹分叉后，相同视频时间不表示相同观测。',
            '仅双方都成功时比较完成步数；失败步数短不代表效率更高。视频播放时间不代表真实推理耗时。',
            '第三方 QwenOFT / RLT_a 的本地结果不等于 π 原论文复现，也不自动证明 TD3 的因果收益。',
            'BC-only 的含义以本次协议为准：v1 为初始校准快照，v2 可为等数据等更新控制；不能仅按策略名称推断公平消融。混合策略也有独立的选择效应。',
            'validation 用于选型；reserved local test 只对本次继续训练保留，发布权重可能已见过相同初态。',
            'McNemar 为双侧 exact 检验；假设配对样本独立，未校正多重比较或选型偏差。bootstrap 固定任务、每任务至少 2 个配对初态、10000 次、seed 42。',
        ],
    )
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    embedded = json.dumps(report, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    # Replace atomically so a concurrent static reader sees a complete artifact.
    for name, body in [('report.json', serialized), ('index.html', HTML.replace('__REPORT_DATA__', embedded))]:
        temporary = output / (name + '.tmp')
        temporary.write_text(body, encoding='utf-8')
        temporary.replace(output / name)
    return report


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLT 优化评估 · 全部回合对照</title>
<style>
:root{color-scheme:dark;--bg:#0f1621;--panel:#182333;--line:#33455c;--muted:#b0bfd0;--blue:#9abbff;--green:#8bdfb2;--red:#ffa3a6}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef4ff;font:15px/1.65 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:28px 22px 65px}h1{font-size:29px;margin:10px 0}h2{font-size:21px}p{color:var(--muted)}a{color:var(--blue)}button,select{font:inherit;background:#26384f;border:1px solid #516b8d;border-radius:7px;color:#f1f5ff;padding:8px 12px;max-width:100%}button{cursor:pointer}button:hover{background:#354b66}button:disabled{opacity:.45;cursor:default}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:15px 0}.toolbar label{display:flex;gap:8px;align-items:center}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:22px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}.number{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}.small{font-size:13px;color:var(--muted)}.notice{background:#1c2c41;border-left:3px solid var(--blue);padding:13px 18px;margin:16px 0}.viewer{display:grid;grid-template-columns:1fr 1fr;gap:14px}.video-wrap{aspect-ratio:1;position:relative;background:#080d13;display:grid;place-items:center;border-radius:8px;overflow:hidden}video{width:100%;height:100%;object-fit:contain}.empty-video{position:absolute;padding:20px;text-align:center;color:var(--muted)}.badge{display:inline-block;padding:2px 10px;border-radius:15px;background:#2c3d54;font-size:13px}.success{color:var(--green)}.failure{color:var(--red)}.video-label{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}.video-label strong{overflow-wrap:anywhere}.seek{display:flex;gap:12px;align-items:center}.seek input{flex:1;accent-color:var(--blue);min-width:90px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px;margin:16px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-size:13px;font-weight:500}tbody tr[data-task]{cursor:pointer}tbody tr[data-task]:hover,tbody tr.selected{background:#263b55}.quick{display:flex;gap:8px;flex-wrap:wrap}.quick button{font-size:13px}details{margin-top:20px}summary{cursor:pointer}li{margin:6px 0;color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere}#media-error{color:var(--red)}[hidden]{display:none!important}@media(max-width:700px){main{padding:18px 12px}.viewer{gap:8px}.panel{padding:9px}.video-label{display:block}.video-label strong{display:block;font-size:13px}.number{font-size:24px}h1{font-size:25px}}
</style></head><body><main>
<div class="small">LIBERO GOAL · 配对评估 · 完整成功与失败记录</div>
<h1>候选策略与基础策略，表现有什么变化？</h1>
<p id="model"></p><div id="split" class="notice"></div><p id="coverage"></p>
<p><strong>策略说明：</strong>baseline 是基础 VLA；release 是发布的 RLT；bc_only 是只模仿 VLA 动作、训练时不使用 Q 值的 Actor；candidate 是经过强化学习的候选。带 alpha 的策略会按该固定比例混合 Actor 与 VLA 动作。</p>
<div id="cards" class="cards"></div><div id="warnings" role="status"></div>
<h2>全部任务成功率</h2><p>每格显示成功回合 / 已记录回合及成功率；N=0 表示尚无记录。点击任务查看所有初态。</p>
<div class="table-wrap"><table><thead id="task-head"></thead><tbody id="task-body"></tbody></table></div>
<h2>所有策略相对基础策略的配对统计</h2><p id="paired-note"></p>
<div class="table-wrap"><table><thead><tr><th>策略</th><th>完整配对 N</th><th>仅该策略成功</th><th>仅基础成功</th><th>宏平均差（百分点）</th><th>bootstrap 95% 区间</th><th>exact McNemar p</th><th>共同成功：步数差 / N</th></tr></thead><tbody id="comparison-body"></tbody></table></div>
<h2>同一初态，双视频同步回放</h2>
<div class="toolbar"><label>任务<select id="task" aria-label="选择任务"></select></label><label>初态<select id="initial" aria-label="选择初始状态"></select></label><label>seed<select id="seed" aria-label="选择随机种子"></select></label><label>左侧策略<select id="left-policy"></select></label><label>右侧策略<select id="right-policy"></select></label><button id="swap">交换左右</button></div>
<p id="pair-summary"></p><div class="viewer">
<div class="panel"><div class="video-label"><strong id="left-title"></strong><span id="left-result" class="badge"></span></div><div class="video-wrap"><video id="left-video" muted playsinline preload="metadata" aria-label="左侧策略回放"></video><div id="left-empty" class="empty-video"></div></div><p id="left-steps" class="small"></p></div>
<div class="panel"><div class="video-label"><strong id="right-title"></strong><span id="right-result" class="badge"></span></div><div class="video-wrap"><video id="right-video" muted playsinline preload="metadata" aria-label="右侧策略回放"></video><div id="right-empty" class="empty-video"></div></div><p id="right-steps" class="small"></p></div>
</div>
<div class="toolbar"><button id="play">同步播放</button><button id="restart">回到开头</button><label>倍速<select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label><span class="small">较短视频结束后保留最后一帧</span></div>
<div class="seek"><input id="seek" type="range" min="0" max="0" step="0.05" value="0" aria-label="同步跳转视频时间"><span id="position">0.0 / 0.0 秒</span></div><p id="media-error" role="alert"></p>
<p class="small">按视频时间同步，视频总步数保留原始记录。回放速度不代表实际推理耗时；轨迹分叉后，同一时刻的观测可能不同。缺少视频的回合仍计入完整统计。</p>
<h2 id="quick-title">候选胜负快速入口</h2><p>这些入口仅用于定位差异；上方统计始终包含全部成功与失败。入口会恢复指定候选与基础策略。</p><p id="wins-label"></p><div id="wins" class="quick"></div><p id="losses-label"></p><div id="losses" class="quick"></div>
<h2>当前任务的全部回合</h2><div class="table-wrap"><table><thead><tr><th>初态</th><th>seed</th><th>策略</th><th>结果</th><th>动作步数</th><th>视频</th></tr></thead><tbody id="episode-body"></tbody></table></div>
<details><summary>统计范围与结论边界</summary><ul id="limitations"></ul><p class="small">所有配对步数差均为候选（或表中策略）减去基础；负值表示双方成功时该策略使用更少步。小样本点估计不保证稳定提升。</p></details>
<details><summary>原始数据与协议</summary><p><a href="report.json" download>下载 report.json</a></p><div id="sources"></div><pre id="protocol" class="small"></pre></details><p id="generated" class="small"></p>
</main><script id="report-data" type="application/json">__REPORT_DATA__</script><script>
'use strict';
const report=JSON.parse(document.querySelector('#report-data').textContent),$=s=>document.querySelector(s);
const number=(v,d=1)=>v===null||v===undefined?'—':Number(v).toFixed(d),percent=v=>v===null?'—':number(100*v)+'%';
function el(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e}
function option(select,value,text){const o=el('option',text);o.value=String(value);select.append(o)}
function key(ep){return JSON.stringify([ep.task_id,ep.initial_state,ep.seed])}
$('#model').textContent=report.model;$('#split').textContent=report.split_label+'。'+report.split_note;
$('#coverage').textContent=`状态：${report.status==='complete'?'已完成':'进行中 / 未确认完成的快照'} · 记录 ${report.total_recorded_episodes} 回合 · 纳入 ${report.evaluation_tasks.length} 个任务、${report.initial_states.length} 个初态 · 计划每策略 ${report.expected_episodes_per_policy??'未知'} 回合。每策略 N 见下方；这不是全部 LIBERO 基准。`;
$('#paired-note').textContent=`基础策略：${report.baseline_policy}。只配对 task + initial + seed 相同的回合；宏平均按本次纳入任务等权。至少一个任务不足 2 个配对初态时不显示 bootstrap 区间。`;
for(const p of report.policies){const c=report.counts[p],card=el('div',undefined,'card');card.append(el('div',p,'small'),el('div',`${c.successes} / ${c.episodes} · ${percent(c.success_rate)}`,'number'),el('div',`N=${c.episodes}；任务宏平均 ${percent(report.macro_success_rate[p])}`,'small'));$('#cards').append(card)}
for(const warning of report.warnings)$('#warnings').append(el('p',warning,'notice'));
for(const note of report.limitations)$('#limitations').append(el('li',note));
for(const source of report.source_files){const a=el('a',source.name);a.href=source.url;const p=el('p');p.append(a,el('span',' · SHA256 '+source.sha256,'small'));$('#sources').append(p)}
$('#protocol').textContent=JSON.stringify(report.protocol,null,2);$('#generated').textContent='静态快照生成时间：'+new Date(report.generated_at).toLocaleString('zh-CN')+'；后续回合完成后重新运行生成器。';
const th=el('tr');th.append(el('th','任务'));for(const p of report.policies)th.append(el('th',p));$('#task-head').append(th);
for(const task of report.tasks){const row=el('tr');row.append(el('td',`${task.task_id} · ${task.name}${task.included?'':'（未纳入）'}`));for(const p of report.policies){const c=report.per_task[String(task.task_id)][p];row.append(el('td',c.episodes?`${c.successes} / ${c.episodes} · ${percent(c.success_rate)}`:'— · N=0'))}if(task.included){row.dataset.task=task.task_id;row.tabIndex=0;row.onclick=()=>chooseTask(task.task_id);row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseTask(task.task_id)}};option($('#task'),task.task_id,`${task.task_id} · ${task.name}`)}$('#task-body').append(row)}
for(const [p,c] of Object.entries(report.comparisons)){const row=el('tr'),ci=c.bootstrap.ci_95_percentage_points;for(const value of [p,c.pairs,c.candidate_only,c.baseline_only,number(c.macro_success_rate_delta_percentage_points),ci?`[${number(ci[0])}, ${number(ci[1])}]`:'样本不足',number(c.exact_mcnemar_two_sided_p,5),`${number(c.joint_success_mean_steps_delta)} / ${c.joint_success_pairs}`])row.append(el('td',String(value)));if(c.unpaired_candidate||c.unpaired_baseline)row.title=`未配对：该策略 ${c.unpaired_candidate}，基础 ${c.unpaired_baseline}`;$('#comparison-body').append(row)}
for(const p of report.policies){option($('#left-policy'),p,p);option($('#right-policy'),p,p)}
const requested=new URLSearchParams(location.hash.slice(1));
$('#left-policy').value=report.policies.includes(requested.get('left'))?requested.get('left'):report.policies.includes(report.baseline_policy)?report.baseline_policy:report.policies[0]??'';
$('#right-policy').value=report.policies.includes(requested.get('right'))?requested.get('right'):report.policies.includes(report.candidate_policy)?report.candidate_policy:report.policies.find(p=>p!==$('#left-policy').value)??report.policies[0]??'';
const videos={left:$('#left-video'),right:$('#right-video')};let active={left:null,right:null},playing=false,master=null,frame=null,generation=0;
function playable(){return Object.keys(videos).filter(s=>active[s]?.video_url&&videos[s].readyState>=1&&!videos[s].error&&Number.isFinite(videos[s].duration)&&videos[s].duration>0)}
function mediaReady(){return Object.keys(videos).filter(s=>active[s]?.video_url&&!videos[s].error).every(s=>playable().includes(s))}
function duration(){return Math.max(0,...playable().map(s=>videos[s].duration))}
function setTime(v,t){if(v.readyState>=1&&Number.isFinite(v.duration))v.currentTime=Math.max(0,Math.min(t,Math.max(0,v.duration-.001)))}
function position(t){const max=duration();$('#seek').max=max;$('#seek').value=Math.min(t,max);$('#position').textContent=`${number(Math.min(t,max))} / ${number(max)} 秒`;$('#play').disabled=playable().length===0||!mediaReady();$('#seek').disabled=max===0}
function pause(){playing=false;generation++;for(const v of Object.values(videos))v.pause();$('#play').textContent='同步播放';if(frame!==null)cancelAnimationFrame(frame);frame=null}
function seekTo(t){pause();for(const side of playable())setTime(videos[side],Number(t));position(Number(t))}
function tick(){if(!playing||!master)return;const t=master.currentTime;for(const s of playable()){const v=videos[s];if(v!==master){if(t>=v.duration-.015){v.pause();if(!v.ended)setTime(v,v.duration)}else if(Math.abs(v.currentTime-t)>.12)setTime(v,t)}}position(t);if(master.ended){const end=duration();pause();position(end);return}frame=requestAnimationFrame(tick)}
async function play(){if(playing){pause();return}if(!mediaReady())return;const sides=playable();if(!sides.length)return;master=videos[sides.reduce((a,b)=>videos[a].duration>=videos[b].duration?a:b)];let time=Number($('#seek').value);if(time>=duration()-.03)time=0;for(const s of sides){setTime(videos[s],time);videos[s].playbackRate=Number($('#speed').value)}playing=true;const ticket=++generation;$('#play').textContent='暂停';const outcomes=await Promise.allSettled(sides.filter(s=>time<videos[s].duration-.015).map(s=>videos[s].play()));if(ticket!==generation)return;if(outcomes.some(x=>x.status==='rejected')){pause();$('#media-error').textContent='视频尚未就绪或浏览器拒绝播放；请等待后重试。';return}frame=requestAnimationFrame(tick)}
function currentKey(){return JSON.stringify([Number($('#task').value),Number($('#initial').value),JSON.parse($('#seed').value||'null')])}
function showPair(){pause();const k=currentKey();$('#media-error').textContent='';for(const side of ['left','right']){const policy=$('#'+side+'-policy').value,ep=report.episodes.find(e=>e.policy===policy&&key(e)===k)??null;active[side]=ep;const v=videos[side];v.removeAttribute('src');$('#'+side+'-title').textContent=policy||'没有策略';$('#'+side+'-result').textContent=ep?(ep.success?'成功':'失败'):'未完成 / 未记录';$('#'+side+'-result').className='badge '+(ep?(ep.success?'success':'failure'):'');$('#'+side+'-steps').textContent=ep?`原始动作步数：${ep.action_steps} 步`:'无回合记录，不作为失败计数';$('#'+side+'-empty').hidden=!!ep?.video_url;$('#'+side+'-empty').textContent=!ep?'该配对尚无回合记录':ep.video_status==='not_recorded'?'本回合未录制视频':'视频文件缺失';if(ep?.video_url)v.src=ep.video_url;v.load()}
const a=active.left,b=active.right;let verdict='该初态缺少一侧记录，暂不能配对';if(a&&b){verdict=a.success&&b.success?`双方成功；右侧减左侧 ${b.action_steps-a.action_steps} 步`:a.success?'仅左侧成功':b.success?'仅右侧成功':'双方失败'}$('#pair-summary').textContent=`task ${$('#task').value} · initial ${$('#initial').value} · seed ${JSON.parse($('#seed').value||'null')??'未记录'}：${verdict}`;position(0);const hash=new URLSearchParams({task:$('#task').value,initial:$('#initial').value,seed:$('#seed').value,left:$('#left-policy').value,right:$('#right-policy').value});history.replaceState(null,'','#'+hash);for(const row of document.querySelectorAll('tr[data-task]'))row.classList.toggle('selected',row.dataset.task===$('#task').value)}
function chooseSeed(seed){const eps=report.episodes.filter(e=>e.task_id===Number($('#task').value)&&e.initial_state===Number($('#initial').value)),seeds=[...new Set(eps.map(e=>JSON.stringify(e.seed)))];if(!seeds.length)seeds.push('null');$('#seed').replaceChildren();for(const value of seeds)option($('#seed'),value,JSON.parse(value)??'未记录');$('#seed').value=seeds.includes(String(seed))?String(seed):seeds[0];showPair()}
function chooseTask(taskId,initial,seed){if(!report.evaluation_tasks.includes(Number(taskId)))return;$('#task').value=taskId;const eps=report.episodes.filter(e=>e.task_id===Number(taskId)),initials=[...new Set([...report.initial_states,...eps.map(e=>e.initial_state)])].sort((a,b)=>a-b);$('#initial').replaceChildren();for(const i of initials)option($('#initial'),i,i);$('#initial').value=initial!==null&&initial!==undefined&&initials.includes(Number(initial))?Number(initial):initials[0]??'';$('#episode-body').replaceChildren();for(const ep of eps.slice().sort((a,b)=>a.initial_state-b.initial_state||String(a.seed).localeCompare(String(b.seed))||report.policies.indexOf(a.policy)-report.policies.indexOf(b.policy))){const row=el('tr');for(const value of [ep.initial_state,ep.seed??'未记录',ep.policy,ep.success?'成功':'失败',ep.action_steps])row.append(el('td',String(value)));const td=el('td');if(ep.video_url){const a=el('a','原始视频');a.href=ep.video_url;td.append(a)}else td.textContent=ep.video_status==='not_recorded'?'未录制':'文件缺失';row.append(td);$('#episode-body').append(row)}chooseSeed(seed)}
$('#quick-title').textContent=`指定候选 ${report.candidate_policy} 相对 ${report.baseline_policy}：胜负快速入口`;
const selected=report.comparisons[report.candidate_policy];for(const [id,field,label] of [['wins','candidate_only','仅候选成功 · candidate-only win'],['losses','baseline_only','仅基础成功 / 候选退步 · baseline-only loss']]){const rows=selected?.episodes.filter(e=>e[field])??[];$('#'+id+'-label').textContent=`${label}：${rows.length} / ${selected?.pairs??0} 完整配对`;if(!rows.length)$('#'+id).append(el('span','本次没有此类已记录配对','small'));for(const row of rows){const button=el('button',`任务 ${row.task_id} · 初态 ${row.initial_state} · seed ${row.seed??'未记录'}`);button.onclick=()=>{$('#left-policy').value=report.baseline_policy;$('#right-policy').value=report.candidate_policy;chooseTask(row.task_id,row.initial_state,JSON.stringify(row.seed));$('#pair-summary').scrollIntoView({behavior:'smooth',block:'center'})};$('#'+id).append(button)}}
for(const [side,v] of Object.entries(videos)){v.onloadedmetadata=()=>position(Number($('#seek').value));v.onerror=()=>{if(active[side]?.video_url){pause();$('#'+side+'-empty').hidden=false;$('#'+side+'-empty').textContent='视频加载失败';$('#media-error').textContent='视频无法加载。请检查文件与静态服务地址；回合统计已保留。';position(Number($('#seek').value))}}}
$('#task').onchange=()=>chooseTask($('#task').value);$('#initial').onchange=()=>chooseSeed();$('#seed').onchange=showPair;$('#left-policy').onchange=showPair;$('#right-policy').onchange=showPair;$('#swap').onclick=()=>{const p=$('#left-policy').value;$('#left-policy').value=$('#right-policy').value;$('#right-policy').value=p;showPair()};$('#play').onclick=play;$('#restart').onclick=()=>seekTo(0);$('#seek').oninput=()=>seekTo($('#seek').value);$('#speed').onchange=()=>{for(const v of Object.values(videos))v.playbackRate=Number($('#speed').value)};
if(report.evaluation_tasks.length){const task=requested.has('task')&&report.evaluation_tasks.includes(Number(requested.get('task')))?Number(requested.get('task')):report.evaluation_tasks[0];chooseTask(task,requested.get('initial'),requested.get('seed'))}else{position(0);$('#pair-summary').textContent='尚无任务或回合记录。'}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evaluation', type=Path, required=True, help='独立评估目录或含 latest_validation.json 的训练目录')
    parser.add_argument('--output', type=Path, help='默认写入评估目录；视频 URL 相对此目录生成')
    parser.add_argument('--candidate-policy', default='candidate', help='默认右侧及快速入口的候选策略名，支持任意混合策略名')
    parser.add_argument('--baseline-policy', default='baseline', help='所有配对统计使用的基础策略名')
    args = parser.parse_args()
    try:
        report = build(args.evaluation, args.output, args.candidate_policy, args.baseline_policy)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'无法生成报告：{exc}\n')
    print(json.dumps(dict(output=str((args.output or args.evaluation).resolve()), status=report['status'],
                          split=report['split'], counts=report['counts']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
