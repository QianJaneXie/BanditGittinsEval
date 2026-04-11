#!/usr/bin/env python3
"""
Simulate exploration on a fixed accuracy matrix (rows = models, columns = i.i.d. samples) by
revealing masked entries, and plot simple regret:

    μ* − μ_{âₜ}

where μ* is the best true row mean and âₜ is the model with highest empirical mean (among rows
with at least one observation) after each batch.

Compared algorithms:
  - upper_confidence_bound_exploration (UCB-E)
  - upper_confidence_bound_exploration_low_rank_factorization (UCB-E-LRF)
  - gittins_index_exploration with τ² = 1/(4B), B = ``--gittins-batch-size`` (default 20; worst-case
    bound on [0, 1]). Gittins uses **batch-mean DP** (one DP stage per that batch) by default; pass
    ``--gittins-per-cell-dp`` for one DP stage per matrix cell instead. UCB-E / LRF use ``--batch-size``.

The horizontal axis is cumulative matrix entries evaluated. By default all policies use the same
total budget (``--eval-budget-fraction``, e.g. 10% of cells). UCB-E runs over the full budget from
the first query. UCB-E-LRF spends ``--warmup-percentage`` (e.g. 5%) on uniform random probing, then
runs the low-rank UCB rule for the rest; **only the post–warm-up segment** is drawn for UCB-E-LRF
so the curve begins where that policy starts (at ~5% cumulative evals when the defaults are 5%
warm-up and 10% total).

By default all three algorithms are simulated and plotted. For a quick test (e.g. Gittins only with a
small budget), use ``--algorithms gittins`` and ``--eval-budget-fraction 0.02`` (or similar).

Use ``--verbose`` to print each step: distinct arms in the batch, incumbent arm, and simple regret
(tagged ``ucb`` / ``lrf`` / ``gittins`` when multiple algorithms run).

**Traces:** by default, cumulative-evaluation counts and simple regret series are written next to the
figure as ``<figure_stem>_traces.npz`` (see ``--traces-out`` / ``--no-save-traces``). Arrays:

- ``ucb_x``, ``ucb_regret`` — UCB-E (empty if not run)
- ``lrf_x_full``, ``lrf_regret_full`` — UCB-E-LRF full trace (empty if not run)
- ``lrf_x_plot``, ``lrf_regret_plot`` — LRF segment used in the figure (cum eval ≥ ``warmup_evals``)
- ``gittins_x``, ``gittins_regret`` — Gittins (empty if not run)
- scalars ``warmup_evals``, ``budget_evals``, ``tau_sq_gittins``

Replot without resimulating: ``python scripts/replot_simple_regret_from_traces.py --traces …``
"""

from __future__ import annotations

import argparse
import json
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

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from gittins_policy import gittins_index_exploration

T = TypeVar("T", int, float)


def make_gittins_step_with_score_cache(**gittins_kwargs):
    """Return ``step(obs, **_)`` that reuses per-arm Gittins scores and only recomputes the last pulled arm."""

    cache: dict[str, torch.Tensor | int | None] = {"scores": None, "prev_arm": None}

    def step(obs: torch.Tensor, **_kwargs) -> torch.Tensor | None:
        m = int(obs.shape[0])
        scores = cache["scores"]
        if scores is None:
            scores = torch.empty((m,), dtype=torch.float32)
            cache["scores"] = scores
            recompute_arms = None
        else:
            prev = cache["prev_arm"]
            recompute_arms = None if prev is None else [int(prev)]

        batch = gittins_index_exploration(
            obs,
            cached_scores=scores,
            recompute_arms=recompute_arms,
            **gittins_kwargs,
        )
        if batch is not None:
            cache["prev_arm"] = int(batch[0, 0].item())
        return batch

    return step


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
    verbose: bool = False,
    log_prefix: str = "",
) -> tuple[list[float], list[int]]:
    """Run until eval budget is reached, the matrix is exhausted, or ``batch is None``.

    Returns parallel lists: regret after each batch, and cumulative number of
    entries revealed (batch sizes summed), including warm-up queries for LRF.

    No new batch is started once cumulative evaluations have reached
    ``max_evaluations`` (total evals never exceed that cap).

    If ``verbose``, prints after each batch: distinct row indices (arms) in the batch,
    incumbent arm (empirical best row, used for simple regret), and simple regret.
    """
    torch.manual_seed(seed)

    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    regrets: list[float] = []
    cum_evaluated: list[int] = []
    evaluated = 0
    tag = log_prefix or "sim"

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
        simple_regret = mu_star - mu_sel
        regrets.append(simple_regret)
        cum_evaluated.append(evaluated)

        if verbose:
            distinct_arms = sorted({int(x) for x in row_idx.reshape(-1).tolist()})
            print(
                f"[{tag}] step {len(regrets)}: cum_eval={evaluated} "
                f"batch_arms(distinct)={distinct_arms} incumbent_arm={arm} "
                f"simple_regret={simple_regret:.6f}",
                flush=True,
            )

    return regrets, cum_evaluated


def save_trace_bundle(
    path: Path,
    *,
    xs_ucbe: list[int],
    regrets_ucbe: list[float],
    xs_lrf: list[int],
    regrets_lrf: list[float],
    xs_lrf_plot: list[int],
    regrets_lrf_plot: list[float],
    xs_gittins: list[int],
    regrets_gittins: list[float],
    warmup_evals: int,
    budget_evals: int,
    tau_sq_gittins: float,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ucb_x=np.asarray(xs_ucbe, dtype=np.int64),
        ucb_regret=np.asarray(regrets_ucbe, dtype=np.float64),
        lrf_x_full=np.asarray(xs_lrf, dtype=np.int64),
        lrf_regret_full=np.asarray(regrets_lrf, dtype=np.float64),
        lrf_x_plot=np.asarray(xs_lrf_plot, dtype=np.int64),
        lrf_regret_plot=np.asarray(regrets_lrf_plot, dtype=np.float64),
        gittins_x=np.asarray(xs_gittins, dtype=np.int64),
        gittins_regret=np.asarray(regrets_gittins, dtype=np.float64),
        warmup_evals=np.int64(warmup_evals),
        budget_evals=np.int64(budget_evals),
        tau_sq_gittins=np.float64(tau_sq_gittins),
    )
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
        default=root / "outputs" / "figures" / "simple_regret_gsm8k_various_models_seed1.png",
        help="Where to save the figure",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for exploration randomness")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="UCB-E and UCB-E-LRF: examples per step (default: 32)",
    )
    parser.add_argument(
        "--gittins-batch-size",
        type=int,
        default=20,
        help="Gittins: examples per step and B in τ² = 1/(4B) (default: 20)",
    )
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
    parser.add_argument(
        "--gittins-grid-points",
        type=int,
        default=2**10 + 1,
        help="Tabular Q grid resolution for gittins_index_exploration",
    )
    parser.add_argument(
        "--gittins-cost",
        type=float,
        default=1e-4,
        help="Default Gittins DP cost per transition (scalar for all arms unless --gittins-cost-per-arm)",
    )
    parser.add_argument(
        "--gittins-cost-per-arm",
        type=Path,
        default=None,
        help="Optional .npy vector of shape (n_arms,) — per-arm transition cost; overrides --gittins-cost",
    )
    parser.add_argument(
        "--gittins-per-cell-dp",
        action="store_true",
        help="Gittins: one DP stage per matrix cell (long horizon). Default is batch-mean DP aligned with --gittins-batch-size",
    )
    parser.add_argument(
        "--gittins-prior-mean",
        type=float,
        default=0.7,
        help="μ_0 for θ_k ~ N(μ_0, v_0) in gittins_index_exploration",
    )
    parser.add_argument(
        "--gittins-prior-variance",
        type=float,
        default=0.01,
        help="v_0 for θ_k ~ N(μ_0, v_0) in gittins_index_exploration",
    )
    parser.add_argument(
        "--traces-out",
        type=Path,
        default=None,
        help="Save regret traces as compressed .npz (default: next to --out, stem + _traces.npz)",
    )
    parser.add_argument(
        "--no-save-traces",
        action="store_true",
        help="Do not write trace .npz or .meta.json",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["ucb", "lrf", "gittins"],
        default=["ucb", "lrf", "gittins"],
        metavar="NAME",
        help="Which algorithms to run and plot (default: all three). Example: --algorithms gittins",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each simulation step: arms in batch, incumbent arm, simple regret",
    )
    args = parser.parse_args()

    algorithms = list(dict.fromkeys(args.algorithms))

    if args.batch_size <= 0:
        print("--batch-size must be positive", file=sys.stderr)
        return 1
    if args.gittins_batch_size <= 0:
        print("--gittins-batch-size must be positive", file=sys.stderr)
        return 1

    if not args.matrix.is_file():
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    gt_np = np.load(args.matrix)
    if gt_np.ndim != 2:
        print(f"Expected a 2D matrix, got shape {gt_np.shape}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms = int(ground_truth.shape[0])

    gittins_cost = float(args.gittins_cost)
    if args.gittins_cost_per_arm is not None:
        if not args.gittins_cost_per_arm.is_file():
            print(f"--gittins-cost-per-arm not found: {args.gittins_cost_per_arm}", file=sys.stderr)
            return 1
        c_np = np.squeeze(np.load(args.gittins_cost_per_arm))
        if c_np.shape != (n_arms,):
            print(
                f"--gittins-cost-per-arm must have shape (n_arms,) = ({n_arms},), got {c_np.shape}",
                file=sys.stderr,
            )
            return 1
        gittins_cost = torch.tensor(c_np, dtype=torch.float64)

    n_cells = int(ground_truth.numel())
    budget_evals = max(1, int(round(args.eval_budget_fraction * n_cells)))
    warmup_evals = int(np.ceil(args.warmup_percentage * n_cells))
    if "lrf" in algorithms and warmup_evals >= budget_evals:
        print(
            "Warm-up threshold (ceil(warmup %% × n)) must be < eval budget; "
            "raise --eval-budget-fraction or lower --warmup-percentage.",
            file=sys.stderr,
        )
        return 1

    tau_sq_gittins = 1.0 / (4.0 * float(args.gittins_batch_size))

    sim_kwargs = {
        "seed": args.seed,
        "max_evaluations": budget_evals,
        "verbose": args.verbose,
    }

    regrets_ucbe: list[float] = []
    xs_ucbe: list[int] = []
    regrets_lrf: list[float] = []
    xs_lrf: list[int] = []
    regrets_gittins: list[float] = []
    xs_gittins: list[int] = []

    if "ucb" in algorithms:
        regrets_ucbe, xs_ucbe = simulate(
            ground_truth,
            upper_confidence_bound_exploration,
            step_kwargs={"a": args.a, "batch_size": args.batch_size, "return_mus": False},
            log_prefix="ucb",
            **sim_kwargs,
        )
    if "lrf" in algorithms:
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
            log_prefix="lrf",
            **sim_kwargs,
        )
    if "gittins" in algorithms:
        regrets_gittins, xs_gittins = simulate(
            ground_truth,
            make_gittins_step_with_score_cache(
                batch_size=args.gittins_batch_size,
                return_mus=False,
                obs_noise_variance=tau_sq_gittins,
                cost_per_transition=gittins_cost,
                n_gittins_grid_points=args.gittins_grid_points,
                prior_mean=args.gittins_prior_mean,
                prior_variance=args.gittins_prior_variance,
                use_batch_mean_gittins_dp=not args.gittins_per_cell_dp,
            ),
            step_kwargs={},
            log_prefix="gittins",
            **sim_kwargs,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    xs_lrf_plot, regrets_lrf_plot = _trim_trace_from_cum_eval(xs_lrf, regrets_lrf, warmup_evals)

    plt.figure(figsize=(8, 5))
    if "ucb" in algorithms:
        plt.plot(xs_ucbe, regrets_ucbe, label="UCB-E", linewidth=1.5)
    if "lrf" in algorithms:
        plt.plot(xs_lrf_plot, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
    if "gittins" in algorithms:
        gittins_label = (
            "Gittins (τ² = 1/(4B), per-cell DP)"
            if args.gittins_per_cell_dp
            else "Gittins (τ² = 1/(4B), batch-mean DP)"
        )
        plt.plot(xs_gittins, regrets_gittins, label=gittins_label, linewidth=1.5)
    plt.xlabel("Cumulative examples evaluated (matrix entries revealed)")
    plt.ylabel("Simple regret")
    batch_desc_parts: list[str] = []
    if "ucb" in algorithms or "lrf" in algorithms:
        batch_desc_parts.append(f"UCB/LRF batch={args.batch_size}")
    if "gittins" in algorithms:
        batch_desc_parts.append(f"Gittins batch={args.gittins_batch_size}")
    batch_desc = ", ".join(batch_desc_parts) if batch_desc_parts else f"batch={args.batch_size}"
    sub = (
        f"seed={args.seed}, {batch_desc}, budget={args.eval_budget_fraction:.0%} of {n_cells} cells"
    )
    if "lrf" in algorithms:
        sub += (
            f"\n(LRF: {args.warmup_percentage:.0%} random warm-up, then low-rank UCB; "
            f"curve starts at ~{warmup_evals} evals)"
        )
    sub = f"algorithms={','.join(algorithms)} | " + sub
    plt.title(f"Simple regret — {args.matrix.name}\n{sub}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    if not args.no_save_traces:
        traces_path = args.traces_out
        if traces_path is None:
            traces_path = args.out.with_name(f"{args.out.stem}_traces.npz")
        save_trace_bundle(
            traces_path,
            xs_ucbe=xs_ucbe,
            regrets_ucbe=regrets_ucbe,
            xs_lrf=xs_lrf,
            regrets_lrf=regrets_lrf,
            xs_lrf_plot=xs_lrf_plot,
            regrets_lrf_plot=regrets_lrf_plot,
            xs_gittins=xs_gittins,
            regrets_gittins=regrets_gittins,
            warmup_evals=warmup_evals,
            budget_evals=budget_evals,
            tau_sq_gittins=tau_sq_gittins,
            meta={
                "matrix": str(args.matrix.resolve()),
                "figure": str(args.out.resolve()),
                "traces": str(traces_path.resolve()),
                "seed": args.seed,
                "batch_size": args.batch_size,
                "gittins_batch_size": args.gittins_batch_size,
                "eval_budget_fraction": args.eval_budget_fraction,
                "ucb_a": args.a,
                "warmup_percentage": args.warmup_percentage,
                "lrf_device": args.lrf_device,
                "gittins_grid_points": args.gittins_grid_points,
                "gittins_cost": args.gittins_cost,
                "gittins_cost_per_arm": str(args.gittins_cost_per_arm.resolve())
                if args.gittins_cost_per_arm is not None
                else None,
                "gittins_prior_mean": args.gittins_prior_mean,
                "gittins_prior_variance": args.gittins_prior_variance,
                "gittins_per_cell_dp": args.gittins_per_cell_dp,
                "n_cells": n_cells,
                "algorithms": algorithms,
                "title": f"Simple regret — {args.matrix.name}\n{sub}",
            },
        )
        print(f"Wrote traces {traces_path} and {traces_path.with_suffix('.meta.json')}")

    print(f"Wrote {args.out}")
    if "ucb" in algorithms:
        print(
            f"UCB-E: {len(regrets_ucbe)} batches, {xs_ucbe[-1] if xs_ucbe else 0} / {budget_evals} budget evals"
        )
    if "lrf" in algorithms:
        print(
            f"UCB-E-LRF: {len(regrets_lrf)} batches, {xs_lrf[-1] if xs_lrf else 0} / {budget_evals} budget evals "
            f"({len(regrets_lrf_plot)} plotted points from cum_eval ≥ {warmup_evals})"
        )
    if "gittins" in algorithms:
        print(
            f"Gittins: {len(regrets_gittins)} batches, {xs_gittins[-1] if xs_gittins else 0} / {budget_evals} budget evals "
            f"(B = {args.gittins_batch_size}, τ² = 1/(4B) = {tau_sq_gittins})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
