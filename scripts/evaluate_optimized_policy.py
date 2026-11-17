#!/usr/bin/env python3
"""Evaluate one completed, frozen RLT experiment without collecting training data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

from optimize_rlt import Experiment, ROOT, indices, write_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed trusted local optimization directory')
    parser.add_argument('--output', type=Path, required=True, help='New evaluation directory')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--initials', type=indices, help='Default: validation 31-39; explicit test split 40-49')
    parser.add_argument('--tasks', type=indices, help='Default: all tasks in the training configuration')
    parser.add_argument('--policies', help='Comma-separated baseline,release,bc_only,candidate; default includes BC when available')
    parser.add_argument('--save-videos', action='store_true')
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error('Output must be a new directory')
    try:
        args.output.relative_to(args.run)
        parser.error('Evaluation output must be outside the original run directory')
    except ValueError:
        pass
    for filename in ['config.json', 'checkpoint.pt', 'actor.pt', 'run_summary.json', 'code_sha256.json']:
        if not (args.run / filename).is_file():
            parser.error(f'Completed optimization run is missing {filename}')
    if json.loads((args.run / 'run_summary.json').read_text()).get('status') != 'complete':
        parser.error('Freeze a completed optimization run before evaluation')
    allowed = set(range(30, 40)) if args.split == 'validation' else set(range(40, 50))
    args.initials = args.initials or list(range(31, 40) if args.split == 'validation' else range(40, 50))
    if not set(args.initials) <= allowed:
        parser.error(f'{args.split} initial states must be within {min(allowed)}-{max(allowed)}')
    config = json.loads((args.run / 'config.json').read_text())
    args.tasks = args.tasks or config['tasks']
    if not set(args.tasks) <= set(range(10)):
        parser.error('LIBERO-Goal task IDs must be within 0-9')
    args.policies = args.policies.split(',') if args.policies else ['baseline', 'release'] + (['bc_only'] if (args.run / 'bc_only_actor.pt').is_file() else []) + ['candidate']
    if len(args.policies) != len(set(args.policies)) or not {'baseline', 'candidate'} <= set(args.policies) or not set(args.policies) <= {'baseline', 'release', 'bc_only', 'candidate'}:
        parser.error('Policies must be unique known names and include baseline and candidate')
    if 'bc_only' in args.policies and not (args.run / 'bc_only_actor.pt').is_file():
        parser.error('This run has no BC-only checkpoint')
    return args, config


class FrozenEvaluation(Experiment):
    """Reuse strict loading and execution while isolating all evaluation writes."""
    def __init__(self, args, output):
        self.evaluation_output = output
        super().__init__(args)
        self.output = output
        for name in ['vla', 'encoder', 'actor', 'release_actor', 'critic', 'target_actor', 'target_critic', 'bc_actor']:
            model = getattr(self, name, None)
            if model is not None:
                model.eval().requires_grad_(False)

    def log(self, event, **fields):
        item = dict(event=event, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **fields)
        line = json.dumps(item, ensure_ascii=False, allow_nan=False)
        with (self.evaluation_output / 'events.jsonl').open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)

    def save(self):
        raise RuntimeError('Frozen evaluation cannot save training checkpoints')

    def backward_step(self, *args, **kwargs):
        raise RuntimeError('Frozen evaluation cannot update model parameters')

    def episode(self, *args, **kwargs):
        if kwargs.get('collect', False):
            raise RuntimeError('Validation/test experience cannot enter the replay buffer')
        return super().episode(*args, **kwargs)


def main():
    args, config = arguments()
    source = Path(config['source_root']).resolve()
    if source != (ROOT / 'src/AlphaBrain-opt').resolve():
        raise ValueError('Unexpected source checkout in the saved configuration')
    os.environ['RLT_SOURCE_ROOT'] = str(source)
    code_identity = json.loads((args.run / 'code_sha256.json').read_text())
    for filename, expected in code_identity.items():
        if sha256(Path(filename)) != expected:
            raise RuntimeError(f'Training/evaluation code changed since this run: {filename}')
    weights = {name: dict(path=str(args.run / name), sha256=sha256(args.run / name))
               for name in ['checkpoint.pt', 'actor.pt'] + (['bc_only_actor.pt'] if 'bc_only' in args.policies else [])}
    manifests = {name: json.loads((ROOT / 'manifests' / name).read_text()) for name in ['qwenoft_5traj.json', 'rlt_a.json']}
    protocol = dict(mode='frozen_evaluation', run=str(args.run), split=args.split, tasks=args.tasks,
                    initial_states=args.initials, policies=args.policies, seed=42, action_horizon=320,
                    candidate_frozen_before_evaluation=True, training_updates=0, collect_into_replay=False,
                    candidate_weights=weights, source_code_sha256=code_identity,
                    evaluator_sha256=sha256(Path(__file__)), original_model_manifests=manifests,
                    model_selection_allowed=args.split == 'validation',
                    note='Test is a final fixed-candidate assessment; do not tune on these outcomes.' if args.split == 'test' else 'Validation supports selection; reserved test states are not executed.')
    args.output.mkdir(parents=True)
    write_json(args.output / 'protocol.json', protocol)
    load_config = config.copy()
    load_config.update(output=args.run, resume=True)
    start = time.monotonic()
    try:
        exp = FrozenEvaluation(argparse.Namespace(**load_config), args.output)
        import numpy as np
        import torch
        supplied_actor = torch.load(args.run / 'actor.pt', map_location='cpu', weights_only=True)
        if not all(torch.equal(value.cpu(), supplied_actor[key]) for key, value in exp.actor.state_dict().items()) or set(supplied_actor) != set(exp.actor.state_dict()):
            raise RuntimeError('actor.pt differs from checkpoint.pt; candidate identity is ambiguous')
        for filename, identity in weights.items():
            if sha256(args.run / filename) != identity['sha256']:
                raise RuntimeError('Source checkpoint changed while loading; freeze the run first')
        before = dict(progress=exp.progress.copy(), replay_size=len(exp.buffer), replay_position=exp.buffer.pos)
        episodes = []
        for task in args.tasks:
            for initial in args.initials:
                for policy in args.policies:
                    random.seed(42)
                    np.random.seed(42)
                    torch.manual_seed(42)
                    episodes.append(exp.episode(task, initial, policy, collect=False, video_dir=args.output if args.save_videos else None))
                    write_json(args.output / 'episodes.json', episodes)
        if before != dict(progress=exp.progress.copy(), replay_size=len(exp.buffer), replay_position=exp.buffer.pos):
            raise RuntimeError('Evaluation unexpectedly modified training progress or replay')
        if not all(torch.equal(value.cpu(), supplied_actor[key]) for key, value in exp.actor.state_dict().items()):
            raise RuntimeError('Candidate parameters changed during evaluation')
        counts = {p: dict(successes=sum(e['success'] for e in episodes if e['policy']==p), episodes=sum(e['policy']==p for e in episodes)) for p in args.policies}
        per_task = {t: {p: dict(successes=sum(e['success'] for e in episodes if e['policy']==p and e['task_id']==t), episodes=len(args.initials)) for p in args.policies} for t in args.tasks}
        paired = []
        for task in args.tasks:
            for initial in args.initials:
                pair = {e['policy']:e for e in episodes if e['task_id']==task and e['initial_state']==initial}
                b, c = pair['baseline'], pair['candidate']
                paired.append(dict(task_id=task, initial_state=initial, baseline=b['success'], candidate=c['success'],
                                   candidate_only=bool(c['success'] and not b['success']), baseline_only=bool(b['success'] and not c['success']),
                                   step_delta_if_both_success=c['action_steps']-b['action_steps'] if c['success'] and b['success'] else None))
        report = dict(split=args.split, initial_states=args.initials, tasks=args.tasks, counts=counts, per_task=per_task,
                      macro_success_rate={p: float(np.mean([per_task[t][p]['successes']/len(args.initials) for t in args.tasks])) for p in args.policies},
                      paired=paired, episodes=episodes, protocol='protocol.json')
        write_json(args.output / 'summary.json', report)
        write_json(args.output / 'run_summary.json', dict(status='complete', mode='frozen_evaluation', split=args.split,
                   training_updates=0, replay_unchanged=True, candidate_parameters_unchanged=True,
                   wall_seconds=time.monotonic()-start, peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                   peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, counts=counts))
        exp.log('evaluation_complete', split=args.split, counts=counts, output=str(args.output))
    except BaseException as exc:
        write_json(args.output / 'run_summary.json', dict(status='failed', mode='frozen_evaluation', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
