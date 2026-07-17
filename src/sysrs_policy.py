"""Synchronized Successive Rejects (SySRs / Smart-SR) for matrix bandits.

Ported from https://github.com/zifanlyu/llm-bandits-sysrs (Smart-SR):
fixed-budget successive rejects with a shared without-replacement task sequence
across active arms, plus budget reallocation when late phases would exceed the
number of available tasks.

Hyperparameter-free aside from the fixed pull budget used to build the SR
schedule (supplied by the experiment as ``eval_budget_fraction`` of the matrix).

Adapted to the local step API: each call returns a synchronized batch that
reveals one shared task to every currently active arm (shape ``(2, n_active)``),
or ``None`` when the elimination schedule is finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from simple_regret_recommend import empirical_incumbent, recommend_from_means


def _logbar(n_arms: int) -> float:
    if n_arms < 2:
        return 0.5
    return float(0.5 + (1.0 / np.arange(2, n_arms + 1)).sum())


def successive_rejects_schedule(n_items: int, n_arms: int) -> np.ndarray:
    """Classic SR cumulative per-arm pull schedule ``nk`` (length ``n_arms``)."""
    if n_arms < 2:
        return np.zeros(1, dtype=int)
    logbar = _logbar(n_arms)
    phases = np.arange(1, n_arms)
    nk = np.ceil((n_items - n_arms) / logbar / (n_arms + 1 - phases)).astype(int)
    return np.r_[0, np.maximum.accumulate(nk)]


def reallocate_budget_across_rounds(
    n_items: int,
    nk_original: np.ndarray,
    total_n_tasks: int,
    n_arms: int,
) -> np.ndarray:
    """Redistribute budget from task-saturated SR rounds to earlier rounds.

    Faithful port of ``_reallocate_budget_across_rounds`` from llm-bandits-sysrs.
    """
    nk_current = nk_original.astype(float).copy()
    num_rounds = len(nk_original)

    target_budget = int(n_items)
    saturated_rounds: set[int] = set()
    iteration = 0
    max_iterations = num_rounds

    while iteration < max_iterations:
        iteration += 1
        newly_saturated = {
            t
            for t in range(1, num_rounds)
            if t not in saturated_rounds and nk_current[t] > total_n_tasks
        }
        if not newly_saturated:
            break

        saturated_rounds.update(newly_saturated)

        total_weighted_budget = 0.0
        boundary_budget_constant = 0.0
        for t in range(1, num_rounds):
            if t not in saturated_rounds:
                n_active_arms = n_arms - t + 1
                total_weighted_budget += (nk_original[t] - nk_original[t - 1]) * n_active_arms

        for t in sorted(saturated_rounds):
            if t > 1 and (t - 1) not in saturated_rounds:
                n_active_arms = n_arms - t + 1
                boundary_budget_constant += total_n_tasks * n_active_arms
                total_weighted_budget -= nk_original[t - 1] * n_active_arms

        budget_consecutive_saturated = 0.0
        saturated_sorted = sorted(saturated_rounds)
        for i, t in enumerate(saturated_sorted):
            n_active_arms = n_arms - t + 1
            if i == 0 and t == 1:
                budget_consecutive_saturated += total_n_tasks * n_active_arms

        if total_weighted_budget == 0:
            break

        alpha = (
            target_budget - budget_consecutive_saturated - boundary_budget_constant
        ) / total_weighted_budget

        for t in range(1, num_rounds):
            if t not in saturated_rounds:
                nk_current[t] = nk_original[t] * alpha
        for t in saturated_rounds:
            nk_current[t] = total_n_tasks

    nk_reallocated = np.ceil(nk_current).astype(int)
    return np.maximum.accumulate(nk_reallocated)


def adjust_sysrs_budget(n_items: int, n_arms: int, n_examples: int) -> int:
    """Clamp the planned pull budget to a feasible SR size."""
    n_items = max(int(n_items), int(n_arms))
    logbar = _logbar(n_arms)
    max_theoretical = int((n_examples - 1) * logbar + n_arms) if n_examples >= 1 else n_arms
    max_budget = max(n_arms, (max_theoretical // n_arms) * n_arms)
    return int(min(n_items, max_budget))


@dataclass
class SySRsState:
    """Mutable elimination state shared by the step and recommend callables."""

    n_arms: int
    n_examples: int
    planned_budget: int
    shared_tasks: np.ndarray
    nk: np.ndarray
    rng: np.random.Generator
    active: np.ndarray = field(init=False)
    phase: int = 1
    task_cursor: int = 0
    # Shared tasks remaining in the current phase (revealed one-at-a-time to all active arms).
    phase_tasks: list[int] = field(default_factory=list)
    done: bool = False

    def __post_init__(self) -> None:
        self.active = np.ones(self.n_arms, dtype=bool)
        self._refill_phase_tasks()

    def _active_arms(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    def _refill_phase_tasks(self) -> None:
        self.phase_tasks.clear()
        if self.done or int(self.active.sum()) <= 1:
            self.done = True
            return
        if self.phase >= self.n_arms:
            self.done = True
            return

        extra = int(self.nk[self.phase] - self.nk[self.phase - 1])
        if extra <= 0:
            # No new pulls this phase; ``next_batch`` will eliminate using ``obs``.
            return

        max_tasks = int(self.shared_tasks.shape[0])
        if self.task_cursor >= max_tasks:
            self.done = True
            return

        end = min(self.task_cursor + extra, max_tasks)
        phase_tasks = self.shared_tasks[self.task_cursor : end]
        self.task_cursor = end
        if phase_tasks.size == 0:
            self.done = True
            return
        self.phase_tasks = [int(c) for c in phase_tasks.tolist()]

    def _eliminate_worst_from_obs(self, obs: torch.Tensor) -> None:
        active_idx = self._active_arms()
        if active_idx.size <= 1:
            self.done = True
            return

        mus = torch.nanmean(obs, dim=1).detach().cpu().numpy()
        scores = np.full(self.n_arms, np.inf, dtype=np.float64)
        for arm in active_idx.tolist():
            m = mus[arm]
            scores[arm] = float(m) if np.isfinite(m) else -np.inf
        # Eliminate a random argmin among active arms (paper: random_argmax(-means)).
        worst_score = scores[active_idx].min()
        candidates = active_idx[np.isclose(scores[active_idx], worst_score)]
        worst = int(self.rng.choice(candidates))
        self.active[worst] = False
        self.phase += 1
        if int(self.active.sum()) <= 1 or self.phase >= self.n_arms:
            self.done = True
            return
        self._refill_phase_tasks()

    def next_batch(self, obs: torch.Tensor) -> torch.Tensor | None:
        """Reveal the next shared task to all active arms (synchronized pull)."""
        if self.done:
            return None
        # Drain empty / zero-extra phases by eliminating until work appears or done.
        guard = 0
        while not self.phase_tasks and not self.done and guard < self.n_arms + 1:
            self._eliminate_worst_from_obs(obs)
            guard += 1
        if self.done or not self.phase_tasks:
            return None

        col = self.phase_tasks.pop(0)
        arms = self._active_arms()
        if arms.size == 0:
            self.done = True
            return None
        cols = np.full(arms.shape, col, dtype=np.int64)
        return torch.stack(
            [
                torch.tensor(arms, dtype=torch.long),
                torch.tensor(cols, dtype=torch.long),
            ]
        )

    def recommend(self, obs: torch.Tensor, _aux: Any = None) -> tuple[int, torch.Tensor]:
        mus = torch.nanmean(obs, dim=1)
        active_idx = self._active_arms()
        if active_idx.size == 0:
            return empirical_incumbent(obs, _aux)
        if active_idx.size == 1:
            return int(active_idx[0]), mus
        scores = mus.clone()
        inactive = torch.ones(self.n_arms, dtype=torch.bool, device=mus.device)
        inactive[torch.as_tensor(active_idx, device=mus.device, dtype=torch.long)] = False
        scores = scores.to(torch.float32)
        scores[inactive] = -float("inf")
        return recommend_from_means(scores)


def make_sysrs_policy(
    *,
    n_arms: int,
    n_examples: int,
    planned_budget: int,
    seed: int | None = None,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor | None],
    Callable[[torch.Tensor, Any], tuple[int, torch.Tensor]],
    SySRsState,
]:
    """Build SySRs ``step_fn`` / ``recommend_fn`` closed over a shared state."""
    rng = np.random.default_rng(None if seed is None else int(seed))

    n_arms = int(n_arms)
    n_examples = int(n_examples)
    planned = adjust_sysrs_budget(int(planned_budget), n_arms, n_examples)
    nk_original = successive_rejects_schedule(planned, n_arms)
    nk = reallocate_budget_across_rounds(planned, nk_original, n_examples, n_arms)
    max_tasks = int(min(n_examples, int(nk[-1]) if nk.size else 0))
    if max_tasks <= 0:
        max_tasks = min(n_examples, 1)
    shared_tasks = rng.choice(np.arange(n_examples), size=max_tasks, replace=False)

    state = SySRsState(
        n_arms=n_arms,
        n_examples=n_examples,
        planned_budget=planned,
        shared_tasks=shared_tasks.astype(int),
        nk=nk.astype(int),
        rng=rng,
    )

    def step_fn(obs: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor | None:
        return state.next_batch(obs)

    def recommend_fn(obs: torch.Tensor, aux: Any = None) -> tuple[int, torch.Tensor]:
        return state.recommend(obs, aux)

    return step_fn, recommend_fn, state
