#!/usr/bin/env python3
"""Three-way rollout plot for one matrix: UCB vs matched-prior Gittins vs mismatched-prior Gittins.

Each panel shows per-arm cumulative batch pulls across batch iterations.
Use one figure per mode: unit or cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Any

_repo_root = Path(__file__).resolve().parents[1] if "scripts" in str(Path(__file__).resolve()) else Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from banditeval.bandits import upper_confidence_bound_exploration

# Make repo src importable when this script is copied under scripts/ or run from repo root.
for cand in [Path.cwd() / "src", Path(__file__).resolve().parents[1] / "src"]:
    if cand.is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from gittins_policy import gittins_index_exploration  # noqa: E402
from gittins_lookup import compute_roots_lookup_table  # noqa: E402
from gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior  # noqa: E402


def load_cost_vector(path: Path | None, n_arms: int) -> torch.Tensor:
    if path is None:
        return torch.ones((n_arms,), dtype=torch.float64)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        arr = np.squeeze(np.load(path)).astype(np.float64)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            arr = np.asarray(data, dtype=np.float64)
        elif isinstance(data, dict) and "cost_per_arm" in data:
            arr = np.asarray(data["cost_per_arm"], dtype=np.float64)
        elif isinstance(data, dict) and "0" in data:
            arr = np.asarray([
                float(data[str(i)].get("estimated_cost_per_1m_input_tokens", data[str(i)].get("cost")))
                for i in range(n_arms)
            ], dtype=np.float64)
        else:
            raise ValueError(f"unsupported JSON cost format: {path}")
    else:
        raise ValueError(f"cost vector must be .npy or .json, got {path}")
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.shape != (n_arms,):
        raise ValueError(f"cost vector shape must be ({n_arms},), got {arr.shape}")
    return torch.tensor(arr, dtype=torch.float64)


def simulate_batch_pull_snapshots(
    ground_truth: torch.Tensor,
    step_fn: Callable[[torch.Tensor], Any],
    *,
    seed: int,
    max_evaluations: int,
    cost_per_arm: torch.Tensor,
    max_original_cost: float | None,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    obs = torch.full_like(ground_truth, float("nan"))
    n_arms = int(ground_truth.shape[0])
    evaluated = 0
    total_cost = 0.0
    cumulative_batch_pulls = np.zeros(n_arms, dtype=np.int64)
    snapshots: list[np.ndarray] = []

    while evaluated < int(max_evaluations):
        out = step_fn(obs)
        if out is None:
            break
        batch = out[0] if isinstance(out, tuple) else out
        if batch is None:
            break
        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        if n_batch <= 0:
            break
        pulled_arm = int(row_idx.reshape(-1)[0].item())
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch
        total_cost += float(cost_per_arm[pulled_arm].item()) * float(n_batch)

        for k in {int(x) for x in row_idx.reshape(-1).tolist()}:
            cumulative_batch_pulls[k] += 1
        snapshots.append(cumulative_batch_pulls.copy())

        if max_original_cost is not None and total_cost >= float(max_original_cost):
            break

    if not snapshots:
        return np.zeros((0, n_arms), dtype=np.int64)
    return np.stack(snapshots, axis=0)


def make_gittins_step(
    ground_truth: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    batch_size: int,
    cost_per_arm: torch.Tensor,
    cost_scaling_factor: float,
):
    n_arms, n_examples = map(int, ground_truth.shape)
    B = int(batch_size)
    tau_sq_batch = 1.0 / (4.0 * float(B))
    tau_sq_cell = float(tau_sq_batch) * float(B)
    transition_stds = transition_stds_shrinking_gaussian_posterior(
        np.float32(float(prior_variance)),
        np.float32(float(tau_sq_cell)),
        int(n_examples),
    )
    dp_costs_per_arm = (cost_per_arm.to(torch.float32).numpy() * float(cost_scaling_factor)).astype(np.float32)
    roots = compute_roots_lookup_table(
        transition_stds=transition_stds,
        costs_per_arm=dp_costs_per_arm,
        n_points=int(2**10 + 1),
    )
    roots_torch = torch.tensor(np.array(roots), dtype=torch.float32)
    cached_scores = torch.full((n_arms,), float("inf"), dtype=torch.float32)
    prev_arm: int | None = None

    def step(obs: torch.Tensor):
        nonlocal prev_arm
        recompute = None if prev_arm is None else [prev_arm]
        out = gittins_index_exploration(
            obs,
            prior_mean=float(prior_mean),
            prior_variance=float(prior_variance),
            obs_noise_variance=float(tau_sq_batch),
            cost_per_transition=cost_per_arm,
            cost_scaling_factor=float(cost_scaling_factor),
            n_gittins_grid_points=int(2**10 + 1),
            batch_size=B,
            return_mus=False,
            cached_scores=cached_scores,
            recompute_arms=recompute,
            use_batch_mean_gittins_dp=False,
            batch_observation_model=True,
            roots_lookup_table=roots_torch,
            allow_early_stop=False,
        )
        batch = out[0] if isinstance(out, tuple) else out
        if batch is not None:
            prev_arm = int(batch[0, 0].item())
        return out

    return step


def plot_panel(ax: plt.Axes, snapshots: np.ndarray, title: str, colors: np.ndarray) -> None:
    if snapshots.size == 0:
        ax.set_title(title + " (empty)")
        ax.axis("off")
        return
    x = np.arange(1, snapshots.shape[0] + 1)
    for k in range(snapshots.shape[1]):
        ax.plot(x, snapshots[:, k], color=colors[k], linewidth=0.7, alpha=0.80)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Batch iteration")
    ax.set_ylabel("Cumulative batch pulls")
    ax.grid(True, alpha=0.25)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-budget-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cost-vector", type=Path, default=None)
    p.add_argument("--cost-scaling-factor", type=float, default=1e-4)
    p.add_argument("--matched-prior-mean", type=float, default=0.5)
    p.add_argument("--matched-prior-variance", type=float, default=0.04)
    p.add_argument("--mismatched-prior-mean", type=float, default=0.2)
    p.add_argument("--mismatched-prior-variance", type=float, default=0.01)
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()

    if not args.matrix.is_file():
        raise FileNotFoundError(args.matrix)
    gt_np = np.load(args.matrix)
    if gt_np.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {gt_np.shape}")
    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms, n_examples = map(int, ground_truth.shape)
    cost_per_arm = load_cost_vector(args.cost_vector, n_arms)

    cost_mode = args.cost_vector is not None
    if cost_mode:
        max_evaluations = int(n_arms * n_examples)
        max_original_cost = float(args.eval_budget_fraction) * float(n_examples * cost_per_arm.sum().item())
        mode_label = "cost"
    else:
        max_evaluations = int(max(1, round(float(args.eval_budget_fraction) * n_arms * n_examples)))
        max_original_cost = None
        mode_label = "unit"

    colors = plt.colormaps["turbo"](np.linspace(0.0, 1.0, n_arms, endpoint=False))

    ucb_step = lambda obs: upper_confidence_bound_exploration(
        obs, a=1.0, batch_size=int(args.batch_size), return_mus=False
    )
    matched_step = make_gittins_step(
        ground_truth,
        prior_mean=float(args.matched_prior_mean),
        prior_variance=float(args.matched_prior_variance),
        batch_size=int(args.batch_size),
        cost_per_arm=cost_per_arm if cost_mode else torch.ones((n_arms,), dtype=torch.float64),
        cost_scaling_factor=float(args.cost_scaling_factor),
    )
    mismatched_step = make_gittins_step(
        ground_truth,
        prior_mean=float(args.mismatched_prior_mean),
        prior_variance=float(args.mismatched_prior_variance),
        batch_size=int(args.batch_size),
        cost_per_arm=cost_per_arm if cost_mode else torch.ones((n_arms,), dtype=torch.float64),
        cost_scaling_factor=float(args.cost_scaling_factor),
    )

    panels = [
        (f"UCB-E B{args.batch_size}", simulate_batch_pull_snapshots(
            ground_truth, ucb_step, seed=args.seed, max_evaluations=max_evaluations,
            cost_per_arm=cost_per_arm, max_original_cost=max_original_cost,
        )),
        (f"Gittins matched prior B{args.batch_size} ({args.matched_prior_mean}, {args.matched_prior_variance})", simulate_batch_pull_snapshots(
            ground_truth, matched_step, seed=args.seed, max_evaluations=max_evaluations,
            cost_per_arm=cost_per_arm, max_original_cost=max_original_cost,
        )),
        (f"Gittins mismatched prior B{args.batch_size} ({args.mismatched_prior_mean}, {args.mismatched_prior_variance})", simulate_batch_pull_snapshots(
            ground_truth, mismatched_step, seed=args.seed, max_evaluations=max_evaluations,
            cost_per_arm=cost_per_arm, max_original_cost=max_original_cost,
        )),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=False, constrained_layout=True)
    for ax, (title, snaps) in zip(axes, panels):
        plot_panel(ax, snaps, title, colors)
    suptitle = args.title or f"Three-way rollout ({mode_label}) | {args.matrix.name} | seed={args.seed}"
    fig.suptitle(suptitle, fontsize=14)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
