#!/usr/bin/env python3
"""Per-arm evaluation rollouts for traces from `simulate_simple_regret.py`.

This script replays the same policies on the same accuracy matrix (same seed and hyperparameters)
and plots per-arm cumulative **batch pulls**:

- x-axis: batch iteration index (1..T)
- y-axis: cumulative number of batches in which each arm was selected

Supports UCB-E, SySRs, and Gittins (no LRF).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

# Use a repo-local, writable matplotlib cache/config directory when possible.
_repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from banditeval.bandits import upper_confidence_bound_exploration

if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from gittins_policy import gittins_index_exploration  # noqa: E402
from gittins_lookup import compute_roots_lookup_table  # noqa: E402
from gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior  # noqa: E402
from sysrs_policy import make_sysrs_policy  # noqa: E402


def simulate_with_batch_pull_snapshots(
    ground_truth: torch.Tensor,
    step: Callable[..., torch.Tensor | None],
    *,
    step_kwargs: dict,
    seed: int,
    max_evaluations: int | None = None,
    per_arm_original_cost: torch.Tensor | None = None,
    max_original_cost: float | None = None,
    batch_pull_snapshots: list[np.ndarray] | None = None,
) -> None:
    """Replay policy, recording cumulative per-arm batch pulls each iteration."""
    if max_evaluations is None:
        raise ValueError("max_evaluations is required")
    if max_original_cost is not None and per_arm_original_cost is None:
        raise ValueError("per_arm_original_cost is required when max_original_cost is set")

    torch.manual_seed(seed)

    obs = torch.full_like(ground_truth, float("nan"))
    evaluated = 0
    total_original_cost = 0.0
    n_arms = int(ground_truth.shape[0])
    cumulative_batch_pulls = np.zeros(n_arms, dtype=np.int64)

    while True:
        if max_evaluations is not None and evaluated >= max_evaluations:
            break
        call_kw = dict(step_kwargs)
        batch = step(obs, **call_kw)
        if batch is None:
            break
        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch
        if per_arm_original_cost is not None:
            rows = row_idx.to(dtype=torch.long)
            total_original_cost += float(per_arm_original_cost[rows].sum().item())

        if batch_pull_snapshots is not None:
            distinct_arms = {int(x) for x in row_idx.reshape(-1).tolist()}
            for k in distinct_arms:
                cumulative_batch_pulls[k] += 1
            batch_pull_snapshots.append(cumulative_batch_pulls.copy())

        if max_original_cost is not None and total_original_cost >= float(max_original_cost):
            break


def _plot_arm_panel(
    ax: plt.Axes,
    iteration_index: list[int],
    batch_pulls: np.ndarray,
    title: str,
    colors: np.ndarray,
) -> None:
    if not iteration_index or batch_pulls.size == 0:
        ax.set_title(title + " (empty)")
        return
    x = np.asarray(iteration_index, dtype=np.int64)
    for k in range(batch_pulls.shape[1]):
        ax.plot(x, batch_pulls[:, k], color=colors[k], linewidth=0.7, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Iteration index (batch step)")
    ax.set_ylabel("Cumulative batch pulls (this arm)")
    ax.grid(True, alpha=0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Path to .npz written by simulate_simple_regret.py",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Override matrix .npy path from meta (if file moved)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image (default: <traces_stem>_arm_rollouts_batch_pulls.png next to traces)",
    )
    args = parser.parse_args()

    if not args.traces.is_file():
        print(f"Traces not found: {args.traces}", file=sys.stderr)
        return 1
    z = np.load(args.traces)
    matrix_path = args.matrix or Path(str(np.asarray(z["matrix"]).reshape(())))
    if not matrix_path.is_file():
        print(f"Matrix not found: {matrix_path} (pass --matrix to override)", file=sys.stderr)
        return 1
    gt_np = np.load(matrix_path)
    if gt_np.ndim != 2:
        print(f"Expected 2D matrix, got {gt_np.shape}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms = int(ground_truth.shape[0])
    budget_evals = int(np.asarray(z["budget_evals"]).reshape(()))
    cost_aware = bool(np.asarray(z.get("cost_aware", np.array(False))).reshape(()))
    budget_original_cost = float(np.asarray(z.get("budget_original_cost", -1.0)).reshape(()))
    max_original_cost = (
        budget_original_cost if cost_aware and budget_original_cost >= 0.0 else None
    )
    seed = int(np.asarray(z["seed"]).reshape(()))
    ucb_batch_size = int(np.asarray(z.get("ucb_batch_size", 32)).reshape(()))
    ucb_a = float(np.asarray(z.get("ucb_a", 1.0)).reshape(()))
    gittins_batch_size = int(np.asarray(z.get("gittins_batch_size", 32)).reshape(()))
    tau_sq_batch = float(
        np.asarray(z.get("tau_sq_gittins", 1.0 / (4.0 * gittins_batch_size))).reshape(())
    )
    prior_mean = float(np.asarray(z.get("gittins_prior_mean", 0.5)).reshape(()))
    prior_variance = float(np.asarray(z.get("gittins_prior_variance", 0.04)).reshape(()))
    cost_scaling_factor = float(np.asarray(z.get("cost_scaling_factor", 1e-4)).reshape(()))
    cost_per_arm_original = np.asarray(
        z.get("cost_per_arm_original", np.ones(n_arms, dtype=np.float64))
    ).reshape(-1)
    if cost_per_arm_original.shape != (n_arms,):
        print(
            f"cost_per_arm_original must have shape ({n_arms},), got {cost_per_arm_original.shape}",
            file=sys.stderr,
        )
        return 1
    cost_tensor = torch.tensor(cost_per_arm_original, dtype=torch.float64)

    cmap = plt.colormaps["turbo"]
    colors = cmap(np.linspace(0.0, 1.0, n_arms, endpoint=False))

    panels: list[tuple[str, list[int], np.ndarray]] = []

    def _has_trace(prefix: str) -> bool:
        key = f"{prefix}_regret"
        return key in z.files and np.asarray(z[key]).size > 0

    # UCB-E rollout.
    if _has_trace("ucb"):
        snaps: list[np.ndarray] = []
        simulate_with_batch_pull_snapshots(
            ground_truth,
            upper_confidence_bound_exploration,
            step_kwargs={"a": ucb_a, "batch_size": ucb_batch_size, "return_mus": False},
            seed=seed,
            max_evaluations=budget_evals,
            per_arm_original_cost=cost_tensor if max_original_cost is not None else None,
            max_original_cost=max_original_cost,
            batch_pull_snapshots=snaps,
        )
        if snaps:
            it = list(range(1, len(snaps) + 1))
            panels.append(("UCB-E", it, np.stack(snaps, axis=0)))

    # SySRs rollout (hyperparameter-free; schedule budget from trace / cell budget).
    if _has_trace("sysrs"):
        snaps = []
        planned_budget = (
            int(np.asarray(z["sysrs_planned_budget"]).reshape(()))
            if "sysrs_planned_budget" in z.files
            else budget_evals
        )
        sysrs_step, _sysrs_recommend, _sysrs_state = make_sysrs_policy(
            n_arms=n_arms,
            n_examples=int(ground_truth.shape[1]),
            planned_budget=planned_budget,
            seed=seed,
        )
        simulate_with_batch_pull_snapshots(
            ground_truth,
            sysrs_step,
            step_kwargs={},
            seed=seed,
            max_evaluations=budget_evals,
            per_arm_original_cost=cost_tensor if max_original_cost is not None else None,
            max_original_cost=max_original_cost,
            batch_pull_snapshots=snaps,
        )
        if snaps:
            it = list(range(1, len(snaps) + 1))
            panels.append(("SySRs", it, np.stack(snaps, axis=0)))

    # Gittins rollout (per-cell DP with batch-observation model).
    if _has_trace("gittins"):
        snaps = []
        B = int(gittins_batch_size)
        tau_sq_cell = float(tau_sq_batch) * float(B)
        transition_stds = transition_stds_shrinking_gaussian_posterior(
            np.float32(prior_variance), np.float32(tau_sq_cell), int(ground_truth.shape[1])
        )
        dp_costs_per_arm = np.asarray(cost_per_arm_original * cost_scaling_factor, dtype=np.float32)
        roots = compute_roots_lookup_table(
            transition_stds=transition_stds,
            costs_per_arm=dp_costs_per_arm,
            n_points=int(2**10 + 1),
        )
        roots_torch = torch.tensor(np.array(roots), dtype=torch.float32)
        cached_scores = torch.full((n_arms,), float("inf"), dtype=torch.float32)
        prev_arm: int | None = None

        def gittins_step(obs: torch.Tensor, **_) -> torch.Tensor | None:
            nonlocal prev_arm
            recompute = None if prev_arm is None else [prev_arm]
            batch = gittins_index_exploration(
                obs,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                obs_noise_variance=tau_sq_batch,
                cost_per_transition=cost_tensor,
                cost_scaling_factor=cost_scaling_factor,
                n_gittins_grid_points=int(2**10 + 1),
                batch_size=B,
                return_mus=False,
                use_batch_mean_gittins_dp=False,
                batch_observation_model=True,
                allow_early_stop=False,
                roots_lookup_table=roots_torch,
                cached_scores=cached_scores,
                recompute_arms=recompute,
            )
            if batch is not None:
                prev_arm = int(batch[0, 0].item())
            return batch

        simulate_with_batch_pull_snapshots(
            ground_truth,
            gittins_step,
            step_kwargs={},
            seed=seed,
            max_evaluations=budget_evals,
            per_arm_original_cost=cost_tensor if max_original_cost is not None else None,
            max_original_cost=max_original_cost,
            batch_pull_snapshots=snaps,
        )
        if snaps:
            it = list(range(1, len(snaps) + 1))
            panels.append(("Gittins", it, np.stack(snaps, axis=0)))

    if not panels:
        print("No algorithms in meta to plot.", file=sys.stderr)
        return 1

    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.8 * nrows), sharex=False, constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    for ax, (title, xs, counts) in zip(axes, panels):
        _plot_arm_panel(ax, xs, counts, title, colors)

    if max_original_cost is not None:
        budget_line = (
            f"seed={seed}, budget={max_original_cost:.4g} original cost, "
            f"{n_arms} arms (re-simulated from trace hyperparameters)"
        )
    else:
        budget_line = (
            f"seed={seed}, budget={budget_evals} evals, "
            f"{n_arms} arms (re-simulated from trace hyperparameters)"
        )
    fig.suptitle(
        f"Per-arm cumulative batch pulls vs iteration — {matrix_path.name}\n{budget_line}",
        fontsize=11,
    )

    out = args.out
    if out is None:
        stem = args.traces.stem.replace("_traces", "")
        out = args.traces.with_name(f"{stem}_arm_rollouts_batch_pulls.png")
        if out == args.traces:
            out = args.traces.with_name(args.traces.stem + "_arm_rollouts_batch_pulls.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
