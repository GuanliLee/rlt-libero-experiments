#!/usr/bin/env python3
"""Local live viewer and paired VLA/RLT_a evaluation; original weights stay frozen."""
import argparse
import io
import json
import mimetypes
import os
import queue
import re
import threading
import time
import traceback
import statistics
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VLA / RLT_a 实时对照</title>
<style>
body{margin:0;background:#10151e;color:#e5ebf4;font:16px system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:28px}
h1{font-size:27px;margin:0 0 10px}p{color:#abb9ce;line-height:1.65}button,select,input{font:inherit;border-radius:7px;padding:10px;border:1px solid #45536a;background:#202b3e;color:white}button{cursor:pointer;background:#3157b8}button:disabled{opacity:.45;cursor:default}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:24px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{background:#1b2432;padding:18px;border-radius:14px}.images{display:grid;grid-template-columns:2fr 1fr;gap:8px;align-items:start}.images img{width:100%;background:#111923;min-height:150px;border-radius:8px;image-rendering:auto}.small{font-size:13px;color:#9cadc5}h2{font-size:20px;margin:0 0 12px}.metric{font-variant-numeric:tabular-nums;line-height:1.8;min-height:80px}.note{padding:15px;border-left:3px solid #6d8ff0;margin-top:20px}#message{padding:12px;background:#202c41;border-radius:8px;overflow-wrap:anywhere}input{width:60px}@media(max-width:760px){.grid{grid-template-columns:1fr}.images img{min-height:100px}}
</style><main><h1>基础 VLA / RLT_a · 实时仿真对照</h1>
<p>LIBERO-Goal · Franka Panda 机械臂 · 相同任务、初始状态与随机种子。两种策略依次执行，画面来自正在运行的仿真。</p>
<p><a href="/results/paired_goal_initial0_20260908/index.html" style="display:inline-block;padding:12px 18px;border-radius:8px;background:#3157b8;color:white;text-decoration:none">查看 10 个任务的并排回放 →</a></p>
<div class="controls"><label>任务 <select id="task"></select></label><label>初始状态 <input id="initial" type="number" min="0" max="49" value="0"></label><label>每种策略回合数 <input id="episodes" type="number" min="1" max="10" value="1"></label><button id="run" disabled>运行对照</button><button id="stop">停止服务并释放显存</button></div>
<div id="message">正在启动模型…</div><div class="grid" style="margin-top:20px">
<section class="panel"><h2>基础 VLA</h2><p class="small">相机 + 指令 → QwenOFT → 直接执行参考动作</p><img id="baseline-combined" style="width:100%;min-height:180px;border-radius:8px;background:#111923" alt="等待仿真相机画面"><div class="small">外部相机（左） · 手腕相机（右） · 同步视频流</div><div class="metric" id="baseline-info">等待执行</div></section>
<section class="panel"><h2>VLA + RLT_a</h2><p class="small">VLA 特征 → 256 维 token；token + 状态 + 参考动作 → Actor → 执行动作</p><img id="rlt-combined" style="width:100%;min-height:180px;border-radius:8px;background:#111923" alt="等待仿真相机画面"><div class="small">外部相机（左） · 手腕相机（右） · 同步视频流</div><div class="metric" id="rlt-info">等待执行</div></section></div>
<p class="note">这里运行已训练好的策略，不会边演示边训练。动作差值是在同一次观测下，将 Actor 输出与 VLA 参考动作作比较，能说明 RLT_a 是否介入；判断成功率是否提升，需要更多相同初始状态的配对回合。推理期间画面暂停属正常，并不保证固定帧率。</p><p id="results" class="small"></p></main>
<script>
const names=['打开中间抽屉','把碗放在炉灶上','把酒瓶放在柜子上','打开上层抽屉并放入碗','把碗放在柜子上','把盘子推到炉灶前','把奶油奶酪放进碗','打开炉灶','把碗放到盘子上','把酒瓶放在架子上'];
names.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=i+' · '+x;document.querySelector('#task').append(o)});
document.querySelector('#run').onclick=async()=>{let r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:Number(document.querySelector('#task').value),initial:Number(document.querySelector('#initial').value),episodes:Number(document.querySelector('#episodes').value)})});if(!r.ok)alert((await r.json()).error)};
let ended=false;document.querySelector('#stop').onclick=async()=>{await fetch('/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});ended=true;document.querySelector('#message').textContent='服务已停止，模型显存即将释放。';document.querySelector('#run').disabled=true};
for(let mode of ['baseline','rlt']){const img=document.querySelector('#'+mode+'-combined');const connect=()=>{if(!ended)img.src=`/stream?mode=${mode}&t=${Date.now()}`};img.onerror=()=>{if(!ended)setTimeout(connect,1500)};connect()}
async function poll(){try{let s=await(await fetch('/status',{cache:'no-store'})).json();document.querySelector('#message').textContent=s.message;document.querySelector('#run').disabled=!s.ready||s.busy;document.querySelector('#results').textContent=s.output?'视频和动作记录：'+s.output:'';for(let mode of ['baseline','rlt']){let p=s[mode]||{};let status=p.success===true?'成功':p.success===false?'未成功':p.phase||p.status||'等待';document.querySelector('#'+mode+'-info').textContent=`${status} · 初始状态 ${p.initial??'—'} · 回合 ${p.episode||0} · 动作步数 ${p.steps||0}`+(p.vla_ms!=null?` · 本次VLA ${p.vla_ms.toFixed(0)}ms`:'')+(p.rlt_ms!=null?` · 本次RLT ${p.rlt_ms.toFixed(1)}ms`:'')+(p.delta_mean_abs!=null?` · 动作差 ${p.delta_mean_abs.toFixed(4)}（归一化）`:'')+(p.token_shape?' · Token '+p.token_shape.join(' × '):'')+(p.completed!=null?` · 已完成 ${p.successes}/${p.completed} 成功`:'')}}catch(e){if(!ended)document.querySelector('#message').textContent='连接中断；请检查服务或 SSH 端口转发。'}if(!ended)setTimeout(poll,500)}poll();
</script></html>'''


def _result_file(url_path, results_root):
    """Only expose supported report files contained in the results directory."""
    if not url_path.startswith('/results/'):
        return None
    try:
        relative = unquote(url_path[len('/results/'):], errors='strict')
        root = results_root.resolve()
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
        if candidate.suffix.lower() not in {'.html', '.json', '.mp4', '.png', '.jpg', '.jpeg'}:
            return None
        return candidate if candidate.is_file() else None
    except (OSError, ValueError, UnicodeError):
        return None


def _byte_range(header, size):
    """Resolve one HTTP byte range; reject malformed or unsatisfiable ranges."""
    if header is None:
        return 0, size - 1, 200
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or not any(match.groups()) or size == 0:
        raise ValueError('invalid byte range')
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
        if start >= size or end < start:
            raise ValueError('unsatisfiable byte range')
    else:
        suffix = int(last)
        if suffix == 0:
            raise ValueError('invalid suffix byte range')
        start, end = max(0, size - suffix), size - 1
    return start, end, 206


def _serve_result_file(handler, url_path, head=False):
    path = _result_file(url_path, ROOT / 'results')
    if path is None:
        handler.send_error(404)
        return
    try:
        file = path.open('rb')
    except OSError:
        handler.send_error(404)
        return
    with file:
        size = os.fstat(file.fileno()).st_size
        try:
            start, end, status = _byte_range(handler.headers.get('Range'), size)
        except ValueError:
            handler.send_response(416)
            handler.send_header('Content-Range', f'bytes */{size}')
            handler.send_header('Content-Length', '0')
            handler.end_headers()
            return
        content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix.lower() in {'.html', '.json'}:
            content_type += '; charset=utf-8'
        handler.send_response(status)
        handler.send_header('Content-Type', content_type)
        handler.send_header('Content-Length', str(end - start + 1))
        handler.send_header('Accept-Ranges', 'bytes')
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('X-Content-Type-Options', 'nosniff')
        if status == 206:
            handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        handler.end_headers()
        if head:
            return
        file.seek(start)
        remaining = end - start + 1
        try:
            while remaining:
                data = file.read(min(1024 * 1024, remaining))
                if not data:
                    break
                handler.wfile.write(data)
                remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


class App:
    def __init__(self, autostart=False):
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.jobs = queue.Queue()
        self.stop = threading.Event()
        self.frames = {}
        self.frame_versions = {}
        self.state = dict(ready=False, busy=True, message='正在加载本地模型…', baseline={}, rlt={})
        self.autostart = autostart

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def publish(self, mode, obs=None, **values):
        pictures = {}
        if obs is not None:
            from PIL import Image
            for camera, key in [('primary', 'primary_image'), ('wrist', 'wrist_image')]:
                buf = io.BytesIO()
                obs[key].save(buf, format='JPEG', quality=85)
                pictures[(mode, camera)] = buf.getvalue()
            combined = Image.new('RGB', (512, 256))
            combined.paste(obs['primary_image'], (0, 0))
            combined.paste(obs['wrist_image'], (256, 0))
            buf = io.BytesIO()
            combined.save(buf, format='JPEG', quality=85)
            pictures[(mode, 'combined')] = buf.getvalue()
        with self.changed:
            panel = self.state[mode]
            panel.update(values)
            if pictures:
                panel['seq'] = panel.get('seq', 0) + 1
                self.frames.update(pictures)
                for key in pictures:
                    self.frame_versions[key] = self.frame_versions.get(key, 0) + 1
                self.changed.notify_all()

    def submit(self, config):
        with self.lock:
            if not self.state['ready'] or self.state['busy']:
                raise ValueError('模型正在加载或运行，请稍后再试。')
            self.state.update(busy=True, message='准备同初始状态对照…', baseline={}, rlt={})
            self.frames.clear()
        self.jobs.put(config)

    def worker(self):
        try:
            import torch
            import eval_rlt_a_sdpa as checked
            from AlphaBrain.training.reinforcement_learning.envs import libero_env
            libero_env._WORKER_SCRIPT = str(ROOT/'scripts/libero_worker_fast_png.py')
            from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder
            from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor
            torch.manual_seed(42)
            self.device = 'cuda:0'
            self.vla = checked.base_framework.BaseFramework.from_pretrained(ROOT/'checkpoints/qwenoft_5traj')
            self.vla = self.vla.to(torch.bfloat16).to(self.device).eval()
            self.encoder = ActionTokenEncoderDecoder(input_dim=2048, bottleneck_dim=256, chunk_len=8,
                                                     num_heads=4, encoder_layers=2, decoder_layers=2).to(self.device)
            self.actor = ActionTokenActor(bottleneck_dim=256, action_dim=7, chunk_len=8,
                                          hidden_dim=512, ref_dropout=0.5, fixed_std=0.1, prop_dim=8).to(self.device)
            ckpt = ROOT/'checkpoints/rlt_a/rl_offpolicy_iter_00400'
            self.encoder.load_state_dict(torch.load(ckpt/'encoder.pt', map_location='cpu', weights_only=True))
            self.actor.load_state_dict(torch.load(ckpt/'actor.pt', map_location='cpu', weights_only=True))
            for model in [self.vla, self.encoder, self.actor]:
                model.eval()
                model.requires_grad_(False)
            self.stats = next(iter(self.vla.norm_stats.values()))['action']
            self.update(ready=True, busy=False, message='模型已就绪。点击“运行对照”即可实时观看。')
            if self.autostart:
                self.submit(dict(task=0, initial=0, episodes=1))
            while not self.stop.is_set():
                try:
                    config = self.jobs.get(timeout=.5)
                except queue.Empty:
                    continue
                try:
                    self.compare(config)
                except Exception as exc:
                    traceback.print_exc()
                    self.update(busy=False, message='执行失败：'+str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.update(ready=False, busy=False, message='模型加载失败：'+str(exc))

    def compare(self, config):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = ROOT/'results'/('comparison_'+stamp)
        output.mkdir(parents=True)
        self.update(output=str(output))
        results = []
        for mode in ['baseline', 'rlt']:
            successes = 0
            for ep in range(config['episodes']):
                if self.stop.is_set():
                    return
                result = self.episode(mode, config['task'], (config['initial']+ep)%50, ep, output)
                results.append(result)
                successes += int(result['success'])
                self.publish(mode, completed=ep+1, successes=successes)
        summary = dict(task_id=config['task'], initial_state_start=config['initial'], seed=42,
                       episodes_per_policy=config['episodes'], action_chunk_length=8,
                       note='Paired initial states; small-sample comparison, not a benchmark claim.', episodes=results)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
        counts = {mode:sum(x['success'] for x in results if x['policy']==mode) for mode in ['baseline','rlt']}
        self.update(busy=False, message=f"对照完成：基础 VLA {counts['baseline']}/{config['episodes']}，RLT_a {counts['rlt']}/{config['episodes']}。可以更换任务或初始状态再次运行。")

    def episode(self, mode, task, initial, episode, output):
        import numpy as np
        import torch
        from AlphaBrain.training.reinforcement_learning.envs.libero_env import LiberoEnv
        from AlphaBrain.training.reinforcement_learning.common.rollout import DUMMY_ACTION, _unnormalize, _postprocess_action, _save_video
        env = LiberoEnv()
        frames, trace = [], []
        timings = {'vla_ms': [], 'rlt_ms': [], 'env_step_ms': [], 'display_encode_ms': []}
        success = False
        started = time.monotonic()
        torch.manual_seed(42)
        np.random.seed(42)
        try:
            obs = env.reset('libero_goal', task, initial_state_idx=initial, seed=42)
            self.update(message=f"正在执行 {'基础 VLA' if mode=='baseline' else 'RLT_a'}：{env.task_description}，初始状态 {initial}")
            self.publish(mode, obs, status='执行中', phase='初始化', success=None, initial=initial, episode=episode+1, steps=0)
            for _ in range(10):
                obs, _, _ = env.step(DUMMY_ACTION)
            cache, cache_index, steps = None, 0, 0
            loop_started = time.monotonic()
            with torch.inference_mode():
                while steps < 320 and not self.stop.is_set():
                    if cache is None or cache_index >= 8:
                        self.publish(mode, phase='计算下一组动作')
                        tick = time.perf_counter()
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            features, reference = self.vla.get_vla_action(
                                batch_images=[[obs['primary_image'], obs['wrist_image']]],
                                instructions=[env.task_description])
                        torch.cuda.synchronize()
                        vla_ms = (time.perf_counter()-tick)*1000
                        timings['vla_ms'].append(vla_ms)
                        reference = reference[:, :8, :]
                        tick = time.perf_counter()
                        if mode == 'rlt':
                            token = self.encoder.encode(features)
                            state = torch.as_tensor(obs['state'], dtype=torch.float32, device=self.device).unsqueeze(0)
                            action, _ = self.actor(token, reference, state, deterministic=True)
                        else:
                            action = reference
                        torch.cuda.synchronize()
                        rlt_ms = (time.perf_counter()-tick)*1000
                        if mode == 'rlt':
                            timings['rlt_ms'].append(rlt_ms)
                        action_np = action[0].float().cpu().numpy()
                        reference_np = reference[0].float().cpu().numpy()
                        if not np.isfinite(action_np).all():
                            raise RuntimeError('模型生成了非有限动作。')
                        delta = float(np.abs(action_np-reference_np).mean())
                        self.publish(mode, phase='执行动作', vla_ms=vla_ms, rlt_ms=rlt_ms if mode=='rlt' else None)
                        if mode == 'rlt':
                            self.publish(mode, delta_mean_abs=delta, token_shape=list(token.shape[1:]))
                        trace.append(dict(step=steps, reference_normalized=reference_np.tolist(),
                                          executed_normalized=action_np.tolist(), mean_absolute_difference=delta))
                        cache = _unnormalize(action_np, self.stats)
                        cache_index = 0
                    tick = time.perf_counter()
                    obs, reward, done = env.step(_postprocess_action(cache[cache_index]))
                    timings['env_step_ms'].append((time.perf_counter()-tick)*1000)
                    cache_index += 1
                    steps += 1
                    frames.append(obs['primary_image'].copy())
                    tick = time.perf_counter()
                    self.publish(mode, obs, steps=steps)
                    timings['display_encode_ms'].append((time.perf_counter()-tick)*1000)
                    if done:
                        success = bool(reward > .5)
                        break
            loop_seconds = time.monotonic()-loop_started
            profile = {key:dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values), p95=float(np.percentile(values,95)))
                       for key,values in timings.items() if values}
            filename = f'{mode}_init{initial:02d}_ep{episode:02d}_{"success" if success else "fail"}'
            _save_video(frames, str(output/(filename+'.mp4')))
            (output/(filename+'_actions.json')).write_text(json.dumps(trace, indent=2)+'\n')
            result = dict(policy=mode, initial_state=initial, episode=episode, success=success, action_steps=steps,
                          wall_seconds=time.monotonic()-started, video=filename+'.mp4',
                          action_loop_seconds=loop_seconds, action_steps_per_wall_second=steps/max(loop_seconds,1e-9), timing=profile,
                          mean_abs_action_difference=float(np.mean([x['mean_absolute_difference'] for x in trace])))
            self.publish(mode, success=success, status='完成')
            return result
        finally:
            env.close()


def serve(args):
    app = App(args.autostart)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, content_type='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/':
                self.send(HTML.encode(), 'text/html; charset=utf-8')
            elif url.path.startswith('/results/'):
                _serve_result_file(self, url.path)
            elif url.path == '/status':
                with app.lock:
                    data = json.dumps(app.state, ensure_ascii=False).encode()
                self.send(data)
            elif url.path == '/stream':
                mode = parse_qs(url.query).get('mode',[''])[0]
                if mode not in ['baseline','rlt']:
                    self.send(b'{}', status=404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                self.connection.settimeout(5)
                key, previous, next_send = (mode,'combined'), -1, 0.0
                try:
                    while not app.stop.is_set():
                        with app.changed:
                            app.changed.wait_for(lambda:app.stop.is_set() or (key in app.frames and app.frame_versions.get(key,0)!=previous), timeout=2)
                            if app.stop.is_set():
                                break
                            if key not in app.frames or app.frame_versions.get(key,0)==previous:
                                continue
                        if app.stop.wait(max(0, next_send-time.monotonic())):
                            break
                        with app.lock:
                            data = app.frames.get(key)
                            previous = app.frame_versions.get(key,0)
                        if data is None:
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n')
                        self.wfile.flush()
                        next_send = time.monotonic()+.05
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
            elif url.path == '/frame':
                query = parse_qs(url.query)
                with app.lock:
                    data = app.frames.get((query.get('mode', [''])[0], query.get('camera', [''])[0]))
                self.send(data or b'', 'image/jpeg', 200 if data else 404)
            else:
                self.send(b'{}', status=404)

        def do_HEAD(self):
            url = urlsplit(self.path)
            if url.path.startswith('/results/'):
                _serve_result_file(self, url.path, head=True)
            else:
                self.send_error(404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if self.headers.get('Content-Type') != 'application/json' or (origin and origin != 'http://'+self.headers.get('Host','')):
                self.send(b'{"error":"invalid request origin or content type"}', status=403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('invalid request length')
                data = json.loads(self.rfile.read(length))
                if self.path == '/stop':
                    app.stop.set()
                    self.send(b'{}')
                    threading.Thread(target=server.shutdown, daemon=True).start()
                    return
                if self.path != '/run':
                    self.send(b'{}', status=404)
                    return
                config = {key:int(data[key]) for key in ['task', 'initial', 'episodes']}
                if not (0 <= config['task'] < 10 and 0 <= config['initial'] < 50 and 1 <= config['episodes'] <= 10):
                    raise ValueError('任务需为 0–9，初始状态 0–49，回合数 1–10。')
                app.submit(config)
                self.send(b'{}')
            except (ValueError, TypeError, KeyError) as exc:
                self.send(json.dumps({'error':str(exc)}, ensure_ascii=False).encode(), status=409)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    worker = threading.Thread(target=app.worker, daemon=True)
    worker.start()
    print(f'Live comparison: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        server.server_close()
        worker.join(timeout=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--autostart', action='store_true')
    serve(parser.parse_args())
