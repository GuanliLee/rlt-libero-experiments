#!/usr/bin/env python3
"""Model-independent validation/early-stop decisions; no training or test access.

Call observe() only after a complete frozen validation, then save a checkpoint
when save_best is true and stop collection when stop_requested is true.
The caller must save the initial (episode 0) checkpoint as well. This module
does not by itself modify an existing training loop or save model weights.
"""
from dataclasses import asdict, dataclass, field


@dataclass
class TrainingMonitor:
    run_id: str
    tasks: tuple = tuple(range(10))
    initials: tuple = (31, 32, 33)
    max_steps: int = 300
    min_episodes: int = 90
    max_episodes: int = 300
    patience: int = 3
    min_step_reduction: float = .05
    best_episode: int | None = None
    last_episode: int = -1
    best_successes: int = -1
    best_mask: list = field(default_factory=list)
    best_success_steps: int = 0
    stalled_evaluations: int = 0
    baseline_mask: list = field(default_factory=list)
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.tasks, self.initials = tuple(self.tasks), tuple(self.initials)
        if (not self.run_id or not self.tasks or not self.initials
            or len(set(self.tasks)) != len(self.tasks) or len(set(self.initials)) != len(self.initials)
            or not set(self.tasks) <= set(range(10)) or not set(self.initials) <= set(range(30,40))):
            raise ValueError('Use a named run and unique validation-only scenes (initials30-39).')
        if not 0 <= self.min_episodes <= self.max_episodes or self.patience < 1 or self.max_steps < 1:
            raise ValueError('Invalid stopping budget')
        if not 0 < self.min_step_reduction < 1:
            raise ValueError('Step reduction must be between zero and one')

    def observe(self, *, run_id, episode, episodes, split='validation'):
        if split != 'validation' or run_id != self.run_id:
            raise ValueError('Cannot select on test data or mix training runs')
        if type(episode) is not int or episode <= self.last_episode:
            raise ValueError('Training episode count must strictly increase')
        if self.last_episode < 0 and episode != 0:
            raise ValueError('Evaluate the initial policy at episode0 before training')
        expected = {(t,i) for t in self.tasks for i in self.initials}
        groups = {p:{} for p in ['baseline','candidate','bc_only']}
        for e in episodes:
            if e['policy'] not in groups:
                continue
            key = (e['task_id'],e['initial_state'])
            if key not in expected or key in groups[e['policy']]:
                raise ValueError('Unexpected or duplicate validation scene')
            if type(e['success']) is not bool or type(e['action_steps']) is not int or not 1 <= e['action_steps'] <= self.max_steps:
                raise ValueError('Invalid success/step result')
            groups[e['policy']][key] = e
        if any(set(g) != expected for g in groups.values()):
            raise ValueError('Need complete baseline, candidate, and matched BC validation')
        keys = sorted(expected)
        base = [groups['baseline'][k]['success'] for k in keys]
        mask = [groups['candidate'][k]['success'] for k in keys]
        bc = [groups['bc_only'][k]['success'] for k in keys]
        if self.baseline_mask and base != self.baseline_mask:
            raise ValueError('Baseline outcomes changed; audit evaluation consistency before selecting')
        successes = sum(mask)
        step_sum = sum(groups['candidate'][k]['action_steps'] for k in keys if groups['candidate'][k]['success'])
        initial = self.best_episode is None
        more_success = successes > self.best_successes
        faster_same_successes = (not initial and successes > 0 and mask == self.best_mask
                                 and step_sum <= self.best_success_steps*(1-self.min_step_reduction))
        improved = initial or more_success or faster_same_successes
        if improved:
            self.best_episode, self.best_successes = episode, successes
            self.best_mask, self.best_success_steps = mask, step_sum
            self.stalled_evaluations = 0
        elif episode >= self.min_episodes:
            self.stalled_evaluations += 1
        self.last_episode, self.baseline_mask = episode, base
        budget = episode >= self.max_episodes
        plateau = episode >= self.min_episodes and self.stalled_evaluations >= self.patience
        row = dict(episode=episode,save_best=improved,best_episode=self.best_episode,
                   candidate_successes=successes,baseline_successes=sum(base),bc_successes=sum(bc),
                   pairs=len(keys),net_wins_over_baseline=successes-sum(base),net_wins_over_bc=successes-sum(bc),
                   step_improvement_on_identical_success_scenes=faster_same_successes,
                   stalled_evaluations=self.stalled_evaluations,stop_requested=budget or plateau,
                   stop_reason='budget' if budget else 'validation_plateau' if plateau else None,
                   stable_efficacy_proven=False)
        self.history.append(row)
        return row

    def state_dict(self):
        return asdict(self)
