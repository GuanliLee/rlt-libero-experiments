#!/usr/bin/env python3
"""Run a predeclared paired task cohort through the already loaded local viewer."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'cohort.json'
    opener = build_opener(ProxyHandler({}))

    def request(path, payload=None):
        req = Request(args.url+path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={} if payload is None else {'Content-Type': 'application/json'})
        with opener.open(req, timeout=20) as response:
            return json.load(response)

    if manifest_path.exists():
        cohort = json.loads(manifest_path.read_text())
    else:
        cohort = dict(protocol=dict(task_suite='libero_goal', task_ids=list(range(10)), initial_state=0,
                      seed=42, episodes_per_policy_per_task=1, max_action_steps=320, action_chunk_length=8,
                      policies=['baseline', 'rlt'], inference='frozen deterministic policies',
                      selection='All ten task ids declared before this cohort was run; no outcome-based selection.',
                      limitation='One initial state per task; exploratory paired comparison, not a benchmark estimate.',
                      created=datetime.now().isoformat()), runs=[])

    def save():
        temp = manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(cohort, ensure_ascii=False, indent=2)+'\n')
        temp.replace(manifest_path)

    save()
    for task in cohort['protocol']['task_ids']:
        if any(run['task_id'] == task for run in cohort['runs']):
            continue
        deadline = time.monotonic()+1200
        while True:
            state = request('/status')
            if state['ready'] and not state['busy']:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Local viewer did not become idle within 20 minutes.')
            time.sleep(1)
        previous = state.get('output')
        request('/run', {'task': task, 'initial': 0, 'episodes': 1})
        print(f'Task {task}: paired evaluation started', flush=True)
        while True:
            time.sleep(1)
            state = request('/status')
            if not state['busy']:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f'Task {task} did not finish within 20 minutes.')
        output = state.get('output')
        if not output or output == previous:
            raise RuntimeError(f'No new output for task {task}: {state["message"]}')
        summary = json.loads((Path(output)/'summary.json').read_text())
        if (summary['task_id'], summary['initial_state_start'], summary['seed'],
                summary['episodes_per_policy']) != (task, 0, 42, 1):
            raise RuntimeError('Another run replaced this request; refusing to misattribute results.')
        cohort['runs'].append(dict(task_id=task, output=output))
        save()
        print(json.dumps({'task': task, 'results': [{k: ep[k] for k in ['policy', 'success', 'action_steps']}
                         for ep in summary['episodes']], 'output': output}), flush=True)
    cohort['completed'] = datetime.now().isoformat()
    save()


if __name__ == '__main__':
    main()
