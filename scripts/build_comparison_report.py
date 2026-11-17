#!/usr/bin/env python3
"""Build an offline, inspectable paired LIBERO report; no model dependencies."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


TASK_NAMES = [
    '打开中间抽屉', '把碗放在炉灶上', '把酒瓶放在柜子上',
    '打开上层抽屉并放入碗', '把碗放在柜子上', '把盘子推到炉灶前',
    '把奶油奶酪放进碗', '打开炉灶', '把碗放到盘子上', '把酒瓶放在架子上',
]


def read_episode(ep, run):
    item = dict(ep)
    video = Path(ep['video'])
    if video.name != str(video) or video.suffix != '.mp4':
        raise ValueError(f'Unexpected video filename: {video}')
    if not (run / video).is_file():
        raise FileNotFoundError(run / video)
    item['video_url'] = f'/results/{quote(run.name)}/{quote(video.name)}'
    trace_file = run / (video.stem + '_actions.json')
    item['actions_url'] = f'/results/{quote(run.name)}/{quote(trace_file.name)}'
    trace = json.loads(trace_file.read_text()) if trace_file.exists() else []
    item['action_difference_by_chunk'] = [
        {'step': int(t['step']), 'mean_absolute_difference': float(t['mean_absolute_difference'])}
        for t in trace
    ]
    return item


def build(cohort_path, output):
    cohort = json.loads(cohort_path.read_text())
    by_task = {}
    for entry in cohort.get('runs', []):
        task_id = int(entry['task_id'])
        if task_id not in range(10):
            raise ValueError(f'Unknown LIBERO-Goal task: {task_id}')
        if task_id in by_task:
            raise ValueError(f'Duplicate task {task_id}; avoid silently selecting a repeat.')
        run = Path(entry['output']).resolve()
        summary_path = run / 'summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if int(summary['task_id']) != task_id:
            raise ValueError(f'Task mismatch in {summary_path}')
        if int(summary['seed']) != 42 or int(summary['initial_state_start']) != 0:
            raise ValueError(f'Expected prospective initial 0 / seed 42 cohort: {summary_path}')
        if int(summary['episodes_per_policy']) != 1 or len(summary['episodes']) != 2:
            raise ValueError(f'Expected one episode per policy: {summary_path}')
        policies = {ep['policy']: read_episode(ep, run) for ep in summary['episodes']}
        if set(policies) != {'baseline', 'rlt'}:
            raise ValueError(f'Missing or repeated policy in {summary_path}')
        if any(int(ep['initial_state']) != 0 for ep in policies.values()):
            raise ValueError(f'Initial state mismatch in {summary_path}')
        baseline, rlt = policies['baseline'], policies['rlt']
        bs, rs = bool(baseline['success']), bool(rlt['success'])
        outcome = ('both_success' if bs else 'rlt_only_success') if rs else ('baseline_only_success' if bs else 'both_fail')
        by_task[task_id] = dict(
            task_id=task_id, name=TASK_NAMES[task_id], status='complete',
            run_directory=str(run), summary_url=f'/results/{quote(run.name)}/summary.json',
            seed=summary['seed'], initial_state=0, action_chunk_length=summary['action_chunk_length'],
            baseline=baseline, rlt=rlt, outcome=outcome,
            success_step_difference=int(rlt['action_steps'])-int(baseline['action_steps']) if bs and rs else None,
        )
    tasks = [by_task.get(i, dict(task_id=i, name=TASK_NAMES[i], status='pending')) for i in range(10)]
    complete = [t for t in tasks if t['status'] == 'complete']
    outcomes = {name: sum(t['outcome'] == name for t in complete)
                for name in ('both_success', 'rlt_only_success', 'baseline_only_success', 'both_fail')}
    both_success = [t for t in complete if t['outcome'] == 'both_success']
    step_totals = {mode: sum(t[mode]['action_steps'] for t in both_success) for mode in ('baseline', 'rlt')}
    report = dict(
        schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
        cohort_file=str(cohort_path.resolve()), protocol=cohort.get('protocol', {}),
        model='AlphaBrain QwenOFT + RLT_a (third-party implementation)',
        replay_fps=10, tasks=tasks,
        aggregate=dict(
            completed_tasks=len(complete), planned_tasks=10, episodes_per_task_per_policy=1,
            baseline_successes=sum(bool(t['baseline']['success']) for t in complete),
            rlt_successes=sum(bool(t['rlt']['success']) for t in complete),
            paired_outcomes=outcomes,
            both_success_steps=dict(
                task_count=len(both_success), baseline_total=step_totals['baseline'], rlt_total=step_totals['rlt'],
                rlt_minus_baseline=step_totals['rlt']-step_totals['baseline'],
                rlt_shorter=sum(t['success_step_difference'] < 0 for t in both_success),
                rlt_longer=sum(t['success_step_difference'] > 0 for t in both_success),
                equal=sum(t['success_step_difference'] == 0 for t in both_success),
            ),
        ),
        limitations=[
            '每任务只有初始状态 0 的一个配对回合；不能据此声称总体成功率提升或复现论文基准。',
            '两种策略使用相同初始状态、任务和 seed 42；之后轨迹会分叉，同一步数不代表相同观测。',
            '步数只在两者都成功时用于比较完成效率；失败回合到达上限不等于成功更慢。',
            '环境控制频率为 20 Hz，视频以 10 fps 编码；1× 回放是仿真时间的半速，回放时间不等于实际推理耗时。',
            '动作差是 RLT 轨迹同一次观测下 Actor 输出与 VLA 参考动作的归一化差值；有差值不等于更好。',
            '此处是 AlphaBrain 的 RLT_a 预训练权重，结构和基座与原博客 / 论文实现不完全相同。',
        ],
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    embedded = json.dumps(report, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (output / 'index.html').write_text(HTML.replace('__REPORT_DATA__', embedded))
    return report


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>看见 RLT_a 的作用 · 相同初始状态对照</title>
<style>
:root{color-scheme:dark;--bg:#0e1420;--panel:#172131;--line:#2e3c50;--muted:#a6b5ca;--blue:#8ab2ff;--green:#80dab0;--red:#ff9da0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f8;font:15px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1200px;margin:0 auto;padding:30px 24px 70px}h1{font-size:30px;line-height:1.3;margin:10px 0 14px}h2{font-size:20px;margin:0 0 10px}p{margin:8px 0;color:var(--muted)}a{color:var(--blue)}.eyebrow{letter-spacing:.06em;color:var(--blue);font-size:13px}.intro{max-width:950px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px}.number{font-size:29px;font-weight:650;font-variant-numeric:tabular-nums}.label{font-size:13px;color:var(--muted)}.notice{padding:14px 18px;background:#1b293d;border-left:3px solid var(--blue);border-radius:4px;margin:18px 0;color:#c1cee0}.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:16px 0}button,select{font:inherit;color:#e9eff8;border:1px solid #48607e;border-radius:7px;background:#23334c;padding:9px 14px}button{cursor:pointer}button:hover{background:#304562}button:disabled{opacity:.4;cursor:default}select{max-width:100%}.viewer{display:grid;grid-template-columns:1fr 1fr;gap:18px}.video-panel{padding:16px}.video-label{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:10px}.badge{font-size:13px;border-radius:30px;padding:2px 10px;background:#27384b;white-space:nowrap}.success{color:var(--green);background:#163c34}.fail{color:var(--red);background:#412733}.video-wrap{position:relative;background:#080d14;border-radius:8px;overflow:hidden;aspect-ratio:1}video{width:100%;height:100%;object-fit:contain;display:block}.ended{position:absolute;bottom:12px;left:12px;right:12px;background:#101720de;padding:7px 12px;border-radius:6px;font-size:13px;display:none}.metric{margin-top:9px;color:#cad5e6;min-height:25px}.timeline{display:flex;align-items:center;gap:14px;width:100%;font-variant-numeric:tabular-nums}.timeline input{flex:1;min-width:80px;accent-color:#8ab2ff}.small{font-size:13px;color:var(--muted)}#task-summary{font-size:17px;color:#e9eef5}table{width:100%;border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums}th,td{padding:11px 12px;border-bottom:1px solid var(--line)}th{font-size:13px;color:var(--muted);font-weight:500}tbody tr[data-task]{cursor:pointer}tbody tr[data-task]:hover,tbody tr.selected{background:#22344d}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:11px;margin:16px 0 24px}.good{color:var(--green)}.bad{color:var(--red)}.pending{color:#73849e}details{margin-top:22px}summary{cursor:pointer;color:#d1dced}li{margin:7px 0;color:var(--muted)}#chart{width:100%;height:auto;max-height:175px;display:block}#media-error{color:var(--red)}.explain{margin-top:20px}.legend{display:flex;gap:16px;flex-wrap:wrap}section{margin-top:24px}footer{margin-top:28px;color:var(--muted);font-size:13px}.chart-wrap{background:#111b29;border-radius:8px;padding:10px 8px}.empty{padding:40px;text-align:center;color:var(--muted)}@media(max-width:750px){main{padding:20px 12px}.cards{grid-template-columns:1fr 1fr}.viewer{gap:8px}.video-panel{padding:9px}.video-label{display:block}.video-label strong{display:block}.metric{font-size:12px}h1{font-size:25px}th,td{padding:9px 8px}.number{font-size:26px}}
</style></head><body><main>
<div class="eyebrow">LIBERO-GOAL · FRANKA PANDA · RTX 5090</div>
<h1>加入 RLT_a，机械臂的行为有什么变化？</h1>
<p class="intro">左边直接执行基础 QwenOFT 的动作；右边用 RLT_a 的 Actor 重新产生动作。下面把相同任务、初始状态和随机种子的两次执行并排回放，分别观察<strong>任务是否完成</strong>、<strong>成功用了多少步</strong>以及<strong>Actor 如何改变动作</strong>。</p>
<div class="cards"><div class="card"><div class="label">基础 VLA 成功</div><div class="number" id="baseline-total">—</div></div><div class="card"><div class="label">VLA + RLT_a 成功</div><div class="number" id="rlt-total">—</div></div><div class="card"><div class="label">仅 RLT_a 成功 / 仅基础成功</div><div class="number" id="paired-total">—</div></div><div class="card"><div class="label">两者都成功时：RLT_a 更少步 / 更多步</div><div class="number" id="steps-total">—</div></div></div>
<div class="notice" id="cohort-note"></div>
<div class="toolbar"><label for="task">选择任务</label><select id="task"></select><a href="/">返回实时仿真</a><a href="report.json" download>下载统计 JSON</a></div>
<p id="task-summary"></p><p class="small" id="pair-protocol"></p>
<div id="empty" class="empty" hidden>对照还在运行，完成后刷新此页即可查看。</div>
<div id="replay">
<div class="viewer"><div class="panel video-panel"><div class="video-label"><strong>基础 VLA</strong><span class="badge" id="baseline-result"></span></div><div class="video-wrap"><video id="baseline-video" playsinline muted preload="metadata" aria-label="基础 VLA 对照视频"></video><div class="ended" id="baseline-ended"></div></div><div class="metric" id="baseline-metric"></div></div><div class="panel video-panel"><div class="video-label"><strong>VLA + RLT_a</strong><span class="badge" id="rlt-result"></span></div><div class="video-wrap"><video id="rlt-video" playsinline muted preload="metadata" aria-label="RLT_a 对照视频"></video><div class="ended" id="rlt-ended"></div></div><div class="metric" id="rlt-metric"></div></div></div>
<div class="toolbar"><button id="play">同步播放</button><button id="restart">回到开头</button><button id="previous" title="往前一个动作步">−1 步</button><button id="next" title="往后一个动作步">+1 步</button><label for="speed">速度</label><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><span class="small">短视频结束后停留在最后一帧</span></div>
<div class="timeline"><input id="seek" type="range" min="1" max="320" value="1" step="1" aria-label="同时跳转到动作步"><span id="position">第 1 步</span></div>
<p class="small">视频仅显示外部相机，以 10 fps 保存；环境控制频率为 20 Hz，所以 1× 回放是仿真时间的半速，2× 对应仿真时间。回放速度不表示实际推理速度。两条轨迹从相同状态出发，之后同一步数的观测可能不同。</p><p id="media-error" role="alert"></p>
<section class="panel explain"><h2>RLT_a 有没有实际介入？</h2><p>曲线表示右侧轨迹每次重新规划时，Actor 输出与<strong>同一次观测</strong>下基础 VLA 参考动作的平均绝对差，单位是归一化动作值。每次产生 8 步 × 7 维动作；最后一组可能只执行一部分。</p><div class="chart-wrap"><svg id="chart" viewBox="0 0 1000 180" role="img" aria-label="RLT 轨迹内 Actor 与参考动作差异"></svg></div><p class="small" id="delta-label"></p><p class="small">非零差值证明 Actor 改变了动作，不能单独证明改得更好；效果要结合上面的成功与步数判断。</p></section>
</div>
<section><h2>全部任务对照</h2><p>点击任意已完成行查看对应回放。完整列出全部 10 个任务，成功和退步都保留。</p><div class="table-wrap"><table><thead><tr><th>任务</th><th>基础 VLA</th><th>VLA + RLT_a</th><th>这一次的差异</th></tr></thead><tbody id="tasks"></tbody></table></div></section>
<details><summary>对照条件与结果的边界</summary><ul id="limitations"></ul><p class="small">基础模型：AlphaBrainGroup/qwenoft-5traj-libero-goal；RLT 权重：AlphaBrainGroup/alphabrain-rlt-5traj-alltasks-libero-goal，rl_offpolicy_iter_00400。演示期间所有模型冻结，Actor 使用确定性输出。</p><p class="small">RLT_a 是 AlphaBrain 的实现：将 VLA 动作位置特征压缩为 256 维 token，Actor 结合 token、机器人状态和 VLA 参考动作，直接输出新的动作组。它与原博客 / 论文使用的模型及实现有差别。</p><pre id="protocol" style="white-space:pre-wrap;font-size:12px;color:#a6b5ca"></pre></details>
<footer id="generated"></footer>
</main><script id="report-data" type="application/json">__REPORT_DATA__</script><script>
'use strict';
const report=JSON.parse(document.querySelector('#report-data').textContent), $=s=>document.querySelector(s);
const aggregate=report.aggregate, completed=report.tasks.filter(t=>t.status==='complete');
$('#baseline-total').textContent=aggregate.baseline_successes+' / '+aggregate.completed_tasks;
$('#rlt-total').textContent=aggregate.rlt_successes+' / '+aggregate.completed_tasks;
$('#paired-total').textContent=aggregate.paired_outcomes.rlt_only_success+' / '+aggregate.paired_outcomes.baseline_only_success;
$('#steps-total').textContent=aggregate.both_success_steps.rlt_shorter+' / '+aggregate.both_success_steps.rlt_longer;
$('#cohort-note').textContent=`已完成 ${aggregate.completed_tasks}/10 个任务，每个任务每种策略仅跑 1 回合，统一初始状态 0、seed 42。这是一组固定条件下的效果预览，不能据此认定总体成功率提高。${aggregate.completed_tasks<10?' 剩余任务仍在运行，稍后刷新此页。':''}`;
$('#generated').textContent='生成时间：'+new Date(report.generated_at).toLocaleString('zh-CN')+' · 来源：每回合的 summary.json、原始 MP4 与动作记录。';
$('#protocol').textContent=JSON.stringify(report.protocol,null,2);
for(const note of report.limitations){const li=document.createElement('li');li.textContent=note;$('#limitations').append(li)}
function verdict(t){switch(t.outcome){case 'rlt_only_success':return 'RLT_a 成功，基础未成功';case 'baseline_only_success':return '基础成功，RLT_a 未成功';case 'both_fail':return '两者均未成功';default:{const d=t.success_step_difference;return d<0?`两者成功，RLT_a 少 ${-d} 步`:d>0?`两者成功，RLT_a 多 ${d} 步`:'两者成功，步数相同'}}}
function resultText(ep){return (ep.success?'成功':'未成功')+' · '+ep.action_steps+' 步'}
for(const task of report.tasks){const row=document.createElement('tr');const name=document.createElement('td');name.textContent=task.task_id+' · '+task.name;row.append(name);if(task.status==='complete'){row.dataset.task=task.task_id;row.tabIndex=0;row.onclick=()=>choose(task.task_id);row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(task.task_id)}};for(const mode of ['baseline','rlt']){const cell=document.createElement('td');cell.textContent=resultText(task[mode]);cell.className=task[mode].success?'good':'bad';row.append(cell)}const cell=document.createElement('td');cell.textContent=verdict(task);row.append(cell);const option=document.createElement('option');option.value=task.task_id;option.textContent=task.task_id+' · '+task.name;$('#task').append(option)}else{for(let i=0;i<3;i++){const cell=document.createElement('td');cell.textContent=i===2?'等待完成':'—';cell.className='pending';row.append(cell)}}$('#tasks').append(row)}
const videos={baseline:$('#baseline-video'),rlt:$('#rlt-video')};let active=null,playing=false,master=null,frameRequest=null,maxSteps=0;
function pause(){playing=false;for(const v of Object.values(videos))v.pause();$('#play').textContent='同步播放';if(frameRequest!==null)cancelAnimationFrame(frameRequest);frameRequest=null}
function seekTo(step){if(!active)return;pause();const s=Math.min(maxSteps,Math.max(1,Number(step)));for(const mode of ['baseline','rlt']){const v=videos[mode],end=(active[mode].action_steps-1)/report.replay_fps;v.currentTime=Math.min((s-1)/report.replay_fps,end)}updatePosition(s)}
function updatePosition(step){const s=Math.min(maxSteps,Math.max(1,Math.floor(step)));$('#seek').value=s;$('#position').textContent=`第 ${s} / ${maxSteps} 步`;for(const mode of ['baseline','rlt']){const ep=active[mode],done=s>=ep.action_steps;$('#'+mode+'-ended').style.display=done?'block':'none';$('#'+mode+'-ended').textContent=`第 ${ep.action_steps} 步结束 · ${ep.success?'任务成功':'未成功，达到回合上限'}`;$('#'+mode+'-metric').textContent=`当前第 ${Math.min(s,ep.action_steps)} 步 / 本回合 ${ep.action_steps} 步`};const marker=$('#cursor');if(marker){const x=60+900*Math.min(s-1,active.rlt.action_steps)/Math.max(active.rlt.action_steps,1);marker.setAttribute('x1',x);marker.setAttribute('x2',x)}}
function tick(){if(!playing)return;const time=master.currentTime;for(const v of Object.values(videos)){if(v!==master&&!v.ended&&time<v.duration-.03&&Math.abs(v.currentTime-time)>.12)v.currentTime=time}updatePosition(Math.floor(time*report.replay_fps)+1);if(master.ended){pause();updatePosition(maxSteps);return}frameRequest=requestAnimationFrame(tick)}
async function play(){if(!active)return;if(playing){pause();return}if(master.ended||Number($('#seek').value)>=maxSteps)seekTo(1);playing=true;$('#play').textContent='暂停';const time=master.currentTime;for(const mode of ['baseline','rlt']){const v=videos[mode];v.playbackRate=Number($('#speed').value);if(time<active[mode].action_steps/report.replay_fps-.02){if(Math.abs(v.currentTime-time)>.08)v.currentTime=time}}const outcomes=await Promise.allSettled(Object.entries(videos).filter(([mode])=>time<active[mode].action_steps/report.replay_fps-.02).map(([,v])=>v.play()));if(outcomes.some(x=>x.status==='rejected')){$('#media-error').textContent='浏览器暂时无法播放视频，请稍等加载后重试。';pause();return}frameRequest=requestAnimationFrame(tick)}
function chart(task){const values=task.rlt.action_difference_by_chunk,svg=$('#chart');svg.replaceChildren();if(!values.length){$('#delta-label').textContent='该回合没有动作记录。';return}const peak=Math.max(...values.map(x=>x.mean_absolute_difference),.001),top=peak*1.1;const points=values.map(v=>[60+900*v.step/task.rlt.action_steps,140-110*v.mean_absolute_difference/top]);const ns='http://www.w3.org/2000/svg';function node(tag,attrs,text){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;svg.append(el);return el}for(let i=0;i<=2;i++){const y=140-i*55;node('line',{x1:60,y1:y,x2:960,y2:y,stroke:'#304052','stroke-width':1});node('text',{x:50,y:y+4,fill:'#a6b5ca','font-size':13,'text-anchor':'end'},(top*i/2).toFixed(3))}node('path',{d:points.map((p,i)=>(i?'L':'M')+p.join(',')).join(' '),fill:'none',stroke:'#8ab2ff','stroke-width':2.5});for(const p of points)node('circle',{cx:p[0],cy:p[1],r:2.4,fill:'#8ab2ff'});node('line',{id:'cursor',x1:60,y1:20,x2:60,y2:140,stroke:'#80dab0','stroke-width':1.8,'stroke-dasharray':'4 3'});node('text',{x:60,y:165,fill:'#a6b5ca','font-size':13},'0 步');node('text',{x:960,y:165,fill:'#a6b5ca','font-size':13,'text-anchor':'end'},task.rlt.action_steps+' 步（RLT_a 轨迹）');$('#delta-label').textContent=`本回合 ${values.length} 次规划，平均动作差 ${task.rlt.mean_abs_action_difference.toFixed(4)}；单次规划最大差 ${peak.toFixed(4)}。绿色虚线标示当前回放位置。`}
function choose(id){const task=completed.find(t=>t.task_id===Number(id));if(!task)return;pause();active=task;maxSteps=Math.max(task.baseline.action_steps,task.rlt.action_steps);$('#task').value=task.task_id;$('#seek').max=maxSteps;$('#task-summary').textContent=verdict(task);$('#pair-protocol').textContent=`任务 ${task.task_id} · ${task.name} · 初始状态 ${task.initial_state} · seed ${task.seed} · 相同基础权重、每次规划 ${task.action_chunk_length} 步 · 上限 320 步`;$('#media-error').textContent='';for(const mode of ['baseline','rlt']){const ep=task[mode],v=videos[mode];$('#'+mode+'-result').textContent=resultText(ep);$('#'+mode+'-result').className='badge '+(ep.success?'success':'fail');v.src=ep.video_url;v.load();v.onerror=()=>{$('#media-error').textContent='视频加载失败，请通过本机 7860 端口的对照报告地址打开，并检查服务是否运行。'}}master=videos[task.baseline.action_steps>=task.rlt.action_steps?'baseline':'rlt'];chart(task);updatePosition(1);for(const row of document.querySelectorAll('tr[data-task]'))row.classList.toggle('selected',Number(row.dataset.task)===task.task_id);history.replaceState(null,'','#task='+task.task_id)}
$('#task').onchange=()=>choose($('#task').value);$('#play').onclick=play;$('#restart').onclick=()=>seekTo(1);$('#previous').onclick=()=>seekTo(Number($('#seek').value)-1);$('#next').onclick=()=>seekTo(Number($('#seek').value)+1);$('#seek').oninput=()=>seekTo($('#seek').value);$('#speed').onchange=()=>{for(const v of Object.values(videos))v.playbackRate=Number($('#speed').value)};
if(completed.length){const requested=Number(new URLSearchParams(location.hash.slice(1)).get('task'));choose(completed.some(t=>t.task_id===requested)?requested:completed[0].task_id)}else{$('#empty').hidden=false;$('#replay').hidden=true;$('#task').disabled=true}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.cohort, args.output)
    print(json.dumps(report['aggregate'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
