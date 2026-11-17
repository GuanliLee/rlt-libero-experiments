#!/usr/bin/env python3
"""Publish progress from saved episodes; never modifies evaluation inputs."""
import argparse
import datetime
import html
import json
from pathlib import Path
import statistics
import time


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value, encoding='utf-8')
    tmp.replace(path)


def snapshot(out, pid):
    protocol = json.loads((out / 'protocol.json').read_text())
    episodes = json.loads((out / 'episodes.json').read_text()) if (out / 'episodes.json').exists() else []
    expected = {(t, i) for t in protocol['tasks'] for i in protocol['initial_states']}
    actual = {(e['task_id'], e['initial_state']) for e in episodes}
    assert actual <= expected and len(actual) == len(episodes)
    summary_file = out / 'run_summary.json'
    status = json.loads(summary_file.read_text())['status'] if summary_file.exists() else 'running'
    if status == 'running' and not Path(f'/proc/{pid}/cmdline').exists():
        status = 'interrupted'
    if status == 'complete':
        assert actual == expected
    successes = sum(e['success'] for e in episodes)
    by_task = []
    for task in protocol['tasks']:
        rows = [e for e in episodes if e['task_id'] == task]
        by_task.append(dict(task_id=task, task=rows[0]['task'] if rows else '', completed=len(rows),
                            successes=sum(e['success'] for e in rows),
                            failed_initials=[e['initial_state'] for e in rows if not e['success']]))
    report = dict(status=status, updated=datetime.datetime.now().astimezone().isoformat(),
                  suite=protocol['suite'], completed=len(episodes), planned=len(expected), successes=successes,
                  success_rate=successes/len(episodes) if episodes else None, tasks=by_task,
                  mean_steps=statistics.mean(e['action_steps'] for e in episodes) if episodes else None,
                  training_updates=0, reserved_test_untouched=True)
    atomic(out / 'progress.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    parts = ['<!doctype html><meta charset="utf-8">',
             '<meta http-equiv="refresh" content="15">' if status == 'running' else '',
             '<title>π₀.₅ 长程任务评估进度</title>',
             '<style>body{font:16px/1.7 system-ui;margin:28px;max-width:1100px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}video{width:256px}section{display:inline-block;vertical-align:top;width:280px;margin:12px}</style>',
             f'<h1>π₀.₅ 基线 · {html.escape(protocol["suite"])}</h1>',
             f'<p>状态：{status}；已完成 {len(episodes)}/{len(expected)} 场，其中成功 {successes} 场。运行中数字不是最终成功率。</p>',
             '<p>基础模型、无 RLT 训练。验证初态30–39；最终测试初态40–49未使用。</p>',
             '<table><tr><th>任务</th><th>完成</th><th>成功</th><th>失败初态</th></tr>']
    for t in by_task:
        parts.append(f'<tr><td>{t["task_id"]}: {html.escape(t["task"])}</td><td>{t["completed"]}</td><td>{t["successes"]}</td><td>{t["failed_initials"]}</td></tr>')
    parts.append('</table><h2>已记录的失败视频</h2>')
    failed = [e for e in episodes if not e['success']]
    if not failed:
        parts.append('<p>暂未记录到失败。</p>')
    for e in failed:
        parts.append(f'<section><p>任务{e["task_id"]} / 初态{e["initial_state"]} / {e["action_steps"]}步</p><video controls preload="none" src="{html.escape(e["video"],quote=True)}"></video></section>')
    if status == 'complete':
        parts.append('<p><a href="index.html">全部视频</a> · <a href="report.zh-CN.md">评估报告</a></p>')
        lines = ['# π₀.₅-LIBERO 长程任务基线', '',
                 f'任务集：{protocol["suite"]}；10任务 × 10验证初态，成功 {successes}/{len(episodes)}（{successes/len(episodes):.1%}）。',
                 f'每回合最多{protocol["max_control_steps"]}控制步，静置10步，每5步重新规划；种子{protocol["seed"]}。',
                 '冻结公开微调模型；没有训练π的RLT。初态0–29留作训练，30–39为验证，40–49为最终测试。', '',
                 '| 任务 | 成功/完成 | 失败初态 |', '|---|---:|---|']
        for t in by_task:
            lines.append(f'| {t["task_id"]}: {t["task"]} | {t["successes"]}/{t["completed"]} | {t["failed_initials"]} |')
        lines += ['', '以上失败只能用于诊断和模型选择，不能再称为独立最终测试。失败视频不单独证明失败原因。',
                  '后续训练应在训练初态采集经验；保持完整任务集对照，避免只汇报改进的任务。比较原始π、匹配BC-only和π+RLT，报告救回和退步数量。',
                  '该模型已用LIBERO示范微调，以上结果不能称为对未见任务的零样本泛化。']
        atomic(out / 'report.zh-CN.md', '\n'.join(lines) + '\n')
    atomic(out / 'progress.html', ''.join(parts))
    return status


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    a = p.parse_args()
    while snapshot(a.output, a.pid) == 'running':
        time.sleep(10)
