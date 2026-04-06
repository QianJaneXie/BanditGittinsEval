#!/usr/bin/env python3
"""
Simulate UCB-style exploration on a fixed accuracy matrix (rows = models,
columns = i.i.d. samples) by revealing masked entries, and plot simple regret:

    μ* − μ_{âₜ}

where μ* is the best true row mean and âₜ is the model with highest empirical
mean (among rows with at least one observation) after each batch.

Compared algorithms:
  - upper_confidence_bound_exploration (UCB-E)
  - upper_confidence_bound_exploration_low_rank_factorization (UCB-E-LRF)

The horizontal axis is cumulative matrix entries evaluated. By default both
algorithms use the same total budget (``--eval-budget-fraction``, e.g. 10% of
cells). UCB-E runs over the full budget from the first query. UCB-E-LRF spends
``--warmup-percentage`` (e.g. 5%) on uniform random probing, then runs the
low-rank UCB rule for the rest; **only the post–warm-up segment** is drawn for
UCB-E-LRF so the curve begins where that policy starts (at ~5% cumulative evals
when the defaults are 5% warm-up and 10% total).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch

from banditeval.bandits import (
    upper_confidence_bound_exploration,
    upper_confidence_bound_exploration_low_rank_factorization,
)

T = TypeVar("T", int, float)


def _trim_trace_from_cum_eval(xs: list[int], ys: list[T], min_x: int) -> tuple[list[int], list[T]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x >= min_x]
    if not pairs:
        return [], []
    ox, oy = zip(*pairs)
    return list(ox), list(oy)


def incumbent_from_empirical_means(observed: torch.Tensor) -> int:
    """Recommend arm with largest row nan-mean; rows with no data are excluded."""
    row_means = torch.nanmean(observed, dim=1)
    scores = torch.where(torch.isnan(row_means), torch.full_like(row_means, -float("inf")), row_means)
    if not torch.isfinite(scores).any():
        return 0
    return int(torch.argmax(scores).item())


def simulate(
    ground_truth: torch.Tensor,
    step: Callable[..., torch.Tensor | None],
    *,
    step_kwargs: dict,
    seed: int,
    max_evaluations: int,
) -> tuple[list[float], list[int]]:
    """Run until eval budget is reached, the matrix is exhausted, or ``batch is None``.

    Returns parallel lists: regret after each batch, and cumulative number of
    entries revealed (batch sizes summed), including warm-up queries for LRF.

    No new batch is started once cumulative evaluations have reached
    ``max_evaluations`` (total evals never exceed that cap).
    """
    torch.manual_seed(seed)

    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    regrets: list[float] = []
    cum_evaluated: list[int] = []
    evaluated = 0

    while True:
        if max_evaluations is not None and evaluated >= max_evaluations:
            break
        batch = step(obs, **step_kwargs)
        if batch is None:
            break
        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch

        arm = incumbent_from_empirical_means(obs)
        mu_sel = float(true_means[arm].item())
        regrets.append(mu_star - mu_sel)
        cum_evaluated.append(evaluated)

    return regrets, cum_evaluated


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=root / "outputs" / "matrices" / "gsm8k_1_samples_various_models_seed1.npy",
        help="Path to (n_models, n_examples) accuracy matrix (.npy)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "outputs" / "figures" / "ucb_simple_regret_gsm8k_various_models_seed1.png",
        help="Where to save the figure",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for exploration randomness")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--eval-budget-fraction",
        type=float,
        default=0.10,
        help="Stop each run after this fraction of matrix cells have been evaluated (default: 10%%)",
    )
    parser.add_argument("--ucb-a", type=int, default=1, dest="a")
    parser.add_argument(
        "--warmup-percentage",
        type=float,
        default=0.05,
        help="UCB-E-LRF: fraction of matrix that must be observed before low-rank UCB replaces random probing",
    )
    parser.add_argument("--lrf-device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.matrix.is_file():
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    gt_np = np.load(args.matrix)
    if gt_np.ndim != 2:
        print(f"Expected a 2D matrix, got shape {gt_np.shape}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)

    n_cells = int(ground_truth.numel())
    budget_evals = max(1, int(round(args.eval_budget_fraction * n_cells)))
    warmup_evals = int(np.ceil(args.warmup_percentage * n_cells))
    if warmup_evals >= budget_evals:
        print(
            "Warm-up threshold (ceil(warmup %% × n)) must be < eval budget; "
            "raise --eval-budget-fraction or lower --warmup-percentage.",
            file=sys.stderr,
        )
        return 1

    sim_kwargs = {"seed": args.seed, "max_evaluations": budget_evals}
    regrets_ucbe, xs_ucbe = simulate(
        ground_truth,
        upper_confidence_bound_exploration,
        step_kwargs={"a": args.a, "batch_size": args.batch_size, "return_mus": False},
        **sim_kwargs,
    )
    regrets_lrf, xs_lrf = simulate(
        ground_truth,
        upper_confidence_bound_exploration_low_rank_factorization,
        step_kwargs={
            "a": args.a,
            "batch_size": args.batch_size,
            "return_mus": False,
            "warmup_percentage": args.warmup_percentage,
            "device": args.lrf_device,
        },
        **sim_kwargs,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    xs_lrf_plot, regrets_lrf_plot = _trim_trace_from_cum_eval(xs_lrf, regrets_lrf, warmup_evals)

    plt.figure(figsize=(8, 5))
    plt.plot(xs_ucbe, regrets_ucbe, label="UCB-E", linewidth=1.5)
    plt.plot(xs_lrf_plot, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
    plt.xlabel("Cumulative examples evaluated (matrix entries revealed)")
    plt.ylabel("Simple regret")
    plt.title(
        f"Simple regret — {args.matrix.name}\n"
        f"seed={args.seed}, batch={args.batch_size}, budget={args.eval_budget_fraction:.0%} of {n_cells} cells "
        f"(LRF: {args.warmup_percentage:.0%} random warm-up, then low-rank UCB; curve starts at ~{warmup_evals} evals)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    print(f"Wrote {args.out}")
    print(f"UCB-E: {len(regrets_ucbe)} batches, {xs_ucbe[-1] if xs_ucbe else 0} / {budget_evals} budget evals")
    print(
        f"UCB-E-LRF: {len(regrets_lrf)} batches, {xs_lrf[-1] if xs_lrf else 0} / {budget_evals} budget evals "
        f"({len(regrets_lrf_plot)} plotted points from cum_eval ≥ {warmup_evals})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
