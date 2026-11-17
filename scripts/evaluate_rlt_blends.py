#!/usr/bin/env python3
"""Frozen paired evaluation of globally fixed RLT/VLA action blends.

Uses the exact legacy episode implementation and a persistent original LIBERO
worker. No state-dependent gating, policy updates, or replay collection occurs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from evaluate_optimized_policy import FrozenEvaluation, sha256
from optimize_rlt import ROOT, indices, write_json
from persistent_libero import install_persistent_env


def alphas(value):
    try:
        result = [float(v) for v in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated alpha values, e.g. 0.25,0.5,1') from exc
    if not result or any(not math.isfinite(a) or not 0 < a <= 1 for a in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('Alpha values must be unique finite numbers in (0,1]')
    return result


def label_for_alpha(alpha, policy='candidate'):
    return policy if alpha == 1. else f'{policy}_alpha{alpha:g}'


def make_blend_actor(actor, alpha):
    import torch

    class FrozenBlend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = actor

        def forward(self, token, reference, prop, deterministic=False):
            if not deterministic:
                raise RuntimeError('Evaluation blend requires deterministic actor execution')
            mean, _ = self.actor(token, reference, prop, deterministic=True)
            # Preserve exact alpha=1 floating-point behavior of the original.
            action = mean if alpha == 1. else (1.-alpha) * reference + alpha * mean
            return action, None

    return FrozenBlend().eval().requires_grad_(False)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--initials', type=indices, help='Default validation 31-39; test 40-49')
    parser.add_argument('--tasks', type=indices)
    parser.add_argument('--alphas', type=alphas, default=alphas('0.25,0.5,1'))
    parser.add_argument('--policies', help='Comma-separated baseline,release,bc_only,candidate; use --blend-bc to blend BC too')
    parser.add_argument('--blend-bc', action='store_true', help='Also evaluate BC at the same alpha values; required for final test with alpha below 1')
    parser.add_argument('--save-videos', action='store_true')
    parser.add_argument('--single-build-env', action='store_true', help='Use fresh per-episode environments with one model build; requires passing pixel/action equivalence manifests')
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error('Output must be a new directory')
    if args.output == args.run or args.run in args.output.parents:
        parser.error('Output must be outside the frozen training run')
    for name in ['config.json', 'checkpoint.pt', 'actor.pt', 'run_summary.json', 'code_sha256.json']:
        if not (args.run / name).is_file():
            parser.error(f'Completed training run is missing {name}')
    if json.loads((args.run / 'run_summary.json').read_text()).get('status') != 'complete':
        parser.error('Training run must be complete before evaluation')
    config = json.loads((args.run / 'config.json').read_text())
    args.tasks = args.tasks or config['tasks']
    if not set(args.tasks) <= set(range(10)):
        parser.error('LIBERO-Goal task IDs must be within 0-9')
    args.initials = args.initials or list(range(31, 40) if args.split == 'validation' else range(40, 50))
    allowed = set(range(30, 40) if args.split == 'validation' else range(40, 50))
    if not set(args.initials) <= allowed:
        parser.error(f'Initial states are outside the local {args.split} split')
    if args.split == 'test' and len(args.alphas) != 1:
        parser.error('Final test requires one alpha selected beforehand on validation: --alphas <value>')
    args.policies = args.policies.split(',') if args.policies else ['baseline', 'release'] + (['bc_only'] if (args.run/'bc_only_actor.pt').is_file() else []) + ['candidate']
    if (len(set(args.policies)) != len(args.policies) or not {'baseline', 'candidate'} <= set(args.policies)
            or not set(args.policies) <= {'baseline', 'release', 'bc_only', 'candidate'}):
        parser.error('Policies must be unique known names and include baseline and candidate')
    if 'bc_only' in args.policies and not (args.run/'bc_only_actor.pt').is_file():
        parser.error('BC-only actor checkpoint is missing')
    if args.blend_bc and 'bc_only' not in args.policies:
        parser.error('--blend-bc requires bc_only in --policies')
    if args.split == 'test' and args.alphas[0] < 1. and not args.blend_bc:
        parser.error('Final test with alpha below 1 requires --blend-bc and a BC control at the same alpha')
    return args, config


class BlendEvaluation(FrozenEvaluation):
    def log(self, event, **fields):
        if event == 'episode' and hasattr(self, '_public_policy'):
            fields.update(policy=self._public_policy, alpha=self._public_alpha)
        super().log(event, **fields)


def verify_v2_checkpoint(payload, code_identity, supplied_bc, version=2):
    import torch
    if version not in (2,3) or payload.get(f'v{version}_code_sha256') != code_identity:
        raise RuntimeError('Checkpoint embedded source identity differs from code_sha256.json')
    if payload['progress'].get('actor_steps') != payload['progress'].get('bc_control_steps'):
        raise RuntimeError('V2 candidate and BC have unequal update counts')
    embedded_bc = payload.get('bc_actor')
    if supplied_bc is None or embedded_bc is None or set(embedded_bc) != set(supplied_bc) or not all(torch.equal(value.cpu(), supplied_bc[key]) for key,value in embedded_bc.items()):
        raise RuntimeError('V2 BC control file differs from the paired checkpoint')


def main():
    args, config = arguments()
    environment_files = []
    if args.single_build_env:
        worker = ROOT/'scripts/libero_worker_single_build.py'
        environment_files.append(worker)
        for manifest_name in ['single-build-libero-validation.json','single-build-policy-equivalence.json']:
            manifest = json.loads((ROOT/'manifests'/manifest_name).read_text())
            if manifest.get('status') != 'passed' or manifest.get('source_sha256',{}).get(str(worker)) != sha256(worker):
                raise RuntimeError('Single-build environment equivalence validation is missing or stale')
    source = Path(config['source_root']).resolve()
    if source != (ROOT / 'src/AlphaBrain-opt').resolve():
        raise ValueError('Unexpected source checkout in frozen training configuration')
    os.environ['RLT_SOURCE_ROOT'] = str(source)
    code_identity = json.loads((args.run/'code_sha256.json').read_text())
    for filename, expected in code_identity.items():
        if sha256(Path(filename)) != expected:
            raise RuntimeError(f'Training source changed since run: {filename}')
    weight_names = ['checkpoint.pt', 'actor.pt'] + (['bc_only_actor.pt'] if 'bc_only' in args.policies else [])
    weights = {name: dict(path=str(args.run/name), sha256=sha256(args.run/name)) for name in weight_names}
    plan = []
    for policy in args.policies:
        if policy == 'candidate' or (policy == 'bc_only' and args.blend_bc):
            plan.extend(dict(label=label_for_alpha(a, policy), mode=policy, alpha=a) for a in args.alphas)
        else:
            plan.append(dict(label=policy, mode=policy, alpha=None))
    labels = [p['label'] for p in plan]
    protocol = dict(mode='frozen_global_action_blend_evaluation', run=str(args.run), split=args.split,
                    tasks=args.tasks, initial_states=args.initials, policy_plan=plan, seed=42,
                    action_horizon=320, chunk_length=8, warmup_steps=10, model_batch_size=1,
                    blend='clamp((1-alpha)*VLA_reference + alpha*deterministic_candidate_mean, -1, 1)',
                    alpha_fixed_for_all_tasks_and_states=True, candidate_frozen_before_evaluation=True,
                    training_updates=0, collect_into_replay=False, model_selection_allowed=args.split=='validation',
                    split_limitation='States are held out only from local continued training; original released weights may have seen all 50 states.',
                    bc_control=('matched_update_BC_offline_continuation' if config.get('runner_version')==3 else 'matched_update_BC') if config.get('runner_version') in (2,3) else 'initial_BC_snapshot_not_matched_update',
                    blend_bc=args.blend_bc, bc_alpha_matched=args.blend_bc,
                    candidate_weights=weights, training_source_sha256=code_identity,
                    evaluation_source_sha256={str(p): sha256(p) for p in [Path(__file__), ROOT/'scripts/evaluate_optimized_policy.py', ROOT/'scripts/persistent_libero.py']+environment_files},
                    environment='Fresh environment each episode, original seed/state/actions, lossless PNG level 1; '+('single model build validated by complete pixel/action traces' if args.single_build_env else 'original double build'),
                    single_build_environment=args.single_build_env,
                    original_model_manifests={name: json.loads((ROOT/'manifests'/name).read_text()) for name in ['qwenoft_5traj.json','rlt_a.json']})
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', protocol)
    load_config = config.copy()
    load_config.update(output=args.run, resume=True)
    started = time.monotonic()
    close_final = None
    try:
        exp = BlendEvaluation(argparse.Namespace(**load_config), args.output)
        import numpy as np
        import torch
        candidate = exp.actor
        bc_actor = exp.bc_actor
        supplied_actor = torch.load(args.run/'actor.pt', map_location='cpu', weights_only=True)
        supplied_bc = torch.load(args.run/'bc_only_actor.pt', map_location='cpu', weights_only=True) if bc_actor is not None else None
        def candidate_matches():
            current = candidate.state_dict()
            return set(current) == set(supplied_actor) and all(torch.equal(value.cpu(), supplied_actor[key]) for key,value in current.items())
        def bc_matches():
            if bc_actor is None:
                return True
            current = bc_actor.state_dict()
            return set(current) == set(supplied_bc) and all(torch.equal(value.cpu(), supplied_bc[key]) for key,value in current.items())
        if not candidate_matches():
            raise RuntimeError('actor.pt does not match checkpoint.pt')
        if not bc_matches():
            raise RuntimeError('Loaded BC actor does not match bc_only_actor.pt')
        if config.get('runner_version') in (2,3):
            # Trusted local runner checkpoint: verify the embedded provenance
            # and paired control in addition to the external file manifest.
            payload = torch.load(args.run/'checkpoint.pt', map_location='cpu', weights_only=False)
            verify_v2_checkpoint(payload, code_identity, supplied_bc, version=config['runner_version'])
            if config['runner_version']==3:
                parent=json.loads((args.run/'parent_identity.json').read_text())
                if not payload.get('parent_checkpoint_sha256')==config.get('parent_checkpoint_sha256')==parent['files']['checkpoint.pt']:
                    raise RuntimeError('Offline continuation parent checkpoint identities disagree')
            del payload
        for name, identity in weights.items():
            if sha256(args.run/name) != identity['sha256']:
                raise RuntimeError('Frozen checkpoint changed while loading')
        if any(model.training or any(p.requires_grad for p in model.parameters())
               for model in [exp.vla, exp.encoder, candidate, exp.release_actor] + ([exp.bc_actor] if exp.bc_actor is not None else [])):
            raise RuntimeError('Evaluation model is not frozen')
        before = dict(progress=exp.progress.copy(), replay_size=len(exp.buffer), replay_position=exp.buffer.pos)
        blends = {a: make_blend_actor(candidate, a) for a in args.alphas}
        bc_blends = {a: make_blend_actor(bc_actor, a) for a in args.alphas} if args.blend_bc else {}
        if args.single_build_env:
            from AlphaBrain.training.reinforcement_learning.envs import libero_env
            libero_env._WORKER_SCRIPT = str(ROOT/'scripts/libero_worker_single_build.py')
        close_final = install_persistent_env(exp)
        episodes = []
        for task in args.tasks:
            for initial in args.initials:
                for item in plan:
                    random.seed(42)
                    np.random.seed(42)
                    torch.manual_seed(42)
                    exp.actor = blends[item['alpha']] if item['mode'] == 'candidate' else candidate
                    exp.bc_actor = bc_blends[item['alpha']] if item['mode'] == 'bc_only' and args.blend_bc else bc_actor
                    exp._public_policy, exp._public_alpha = item['label'], item['alpha']
                    video_dir = args.output/item['label'] if args.save_videos else None
                    if video_dir:
                        video_dir.mkdir(exist_ok=True)
                    result = exp.episode(task, initial, item['mode'], collect=False, video_dir=video_dir)
                    result.update(policy=item['label'], alpha=item['alpha'])
                    if 'video' in result:
                        path = video_dir/result['video']
                        if not path.is_file():
                            raise RuntimeError(f'Expected MP4 was not produced: {path}')
                        result['video'] = str(path.relative_to(args.output))
                    episodes.append(result)
                    write_json(args.output/'episodes.json', episodes)
        exp.actor = candidate
        exp.bc_actor = bc_actor
        if before != dict(progress=exp.progress.copy(), replay_size=len(exp.buffer), replay_position=exp.buffer.pos) or not candidate_matches() or not bc_matches():
            raise RuntimeError('Frozen evaluation modified training state, candidate parameters, or BC parameters')
        close_final()
        close_final = None
        counts = {p: dict(successes=sum(e['success'] for e in episodes if e['policy']==p), episodes=sum(e['policy']==p for e in episodes)) for p in labels}
        per_task = {t: {p: dict(successes=sum(e['success'] for e in episodes if e['policy']==p and e['task_id']==t), episodes=len(args.initials)) for p in labels} for t in args.tasks}
        paired_by_candidate = {}
        for item in plan:
            if item['mode'] != 'candidate':
                continue
            p = item['label']
            pairs = []
            for task in args.tasks:
                for initial in args.initials:
                    pair = {e['policy']:e for e in episodes if e['task_id']==task and e['initial_state']==initial}
                    b, c = pair['baseline'], pair[p]
                    pairs.append(dict(task_id=task, initial_state=initial, baseline=b['success'], candidate=c['success'],
                                      candidate_only=bool(c['success'] and not b['success']), baseline_only=bool(b['success'] and not c['success']),
                                      step_delta_if_both_success=c['action_steps']-b['action_steps'] if c['success'] and b['success'] else None))
            paired_by_candidate[p] = pairs
        paired_candidate_vs_bc_same_alpha = {}
        if args.blend_bc:
            for alpha in args.alphas:
                candidate_label, bc_label = label_for_alpha(alpha), label_for_alpha(alpha, 'bc_only')
                pairs = []
                for task in args.tasks:
                    for initial in args.initials:
                        pair = {e['policy']:e for e in episodes if e['task_id']==task and e['initial_state']==initial}
                        b, c = pair[bc_label], pair[candidate_label]
                        pairs.append(dict(task_id=task, initial_state=initial, alpha=alpha, bc=b['success'], candidate=c['success'],
                                          candidate_only=bool(c['success'] and not b['success']), bc_only=bool(b['success'] and not c['success']),
                                          step_delta_if_both_success=c['action_steps']-b['action_steps'] if c['success'] and b['success'] else None))
                paired_candidate_vs_bc_same_alpha[candidate_label] = pairs
        report = dict(split=args.split, initial_states=args.initials, tasks=args.tasks, counts=counts, per_task=per_task,
                      macro_success_rate={p: float(np.mean([per_task[t][p]['successes']/len(args.initials) for t in args.tasks])) for p in labels},
                      paired_by_candidate=paired_by_candidate, paired_candidate_vs_bc_same_alpha=paired_candidate_vs_bc_same_alpha,
                      episodes=episodes, protocol='protocol.json')
        if len(paired_by_candidate) == 1:
            report['paired'] = next(iter(paired_by_candidate.values()))
        write_json(args.output/'summary.json', report)
        write_json(args.output/'run_summary.json', dict(status='complete', mode=protocol['mode'], split=args.split,
                   training_updates=0, replay_unchanged=True, candidate_parameters_unchanged=True, bc_parameters_unchanged=True,
                   wall_seconds=time.monotonic()-started, peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                   peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, counts=counts))
        exp.log('evaluation_complete', counts=counts, output=str(args.output))
    except BaseException as exc:
        write_json(args.output/'run_summary.json', dict(status='failed', mode=protocol['mode'], error=repr(exc)))
        raise
    finally:
        if close_final:
            close_final()


if __name__ == '__main__':
    main()
