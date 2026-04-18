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
- scalar ``gittins_stop_cum_eval`` — first cumulative eval count (before that policy step) where the
  top-scoring arm was already fully observed (nominal stopping time); ``-1`` if that never occurred.
  The Gittins regret curve still runs to the eval budget when using the simple-regret script.
- scalars ``warmup_evals``, ``budget_evals``, ``tau_sq_gittins``

**Gittins-only rerun with UCB/LRF unchanged:** pass ``--algorithms gittins`` and
``--merge-ucb-lrf-from PREVIOUS_traces.npz`` to copy UCB-E and UCB-E-LRF series from an earlier
full run and simulate only Gittins (same matrix / seed / budget recommended).

Replot without resimulating: ``python scripts/replot_simple_regret_from_traces.py --traces …``

**Timing (``*_traces.meta.json``):** when traces are saved, ``timing`` records per-method wall-clock
stats from ``time.time()`` (see ``summary.iter_total`` / ``summary.iter_step``). Pass
``--timing-include-per-iter-series`` to also store every iteration's seconds in the meta file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

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

    def step(obs: torch.Tensor, **kwargs) -> torch.Tensor | None:
        sim_cum_eval = kwargs.pop("sim_cum_eval", None)
        m = int(obs.shape[0])
        scores = cache["scores"]
        if scores is None:
            scores = torch.full((m,), float("inf"), dtype=torch.float32)
            cache["scores"] = scores
            recompute_arms = None
        else:
            prev = cache["prev_arm"]
            recompute_arms = None if prev is None else [int(prev)]

        batch = gittins_index_exploration(
            obs,
            cached_scores=scores,
            recompute_arms=recompute_arms,
            sim_cum_eval=sim_cum_eval,
            **gittins_kwargs,
        )
        if batch is not None:
            cache["prev_arm"] = int(batch[0, 0].item())
        return batch

    return step


def make_round_robin_step(*, batch_size: int) -> Callable[[torch.Tensor], torch.Tensor | None]:
    """Round-robin baseline: cycle arms, evaluate ``batch_size`` new columns on that arm.

    Incumbent recommendation is still the sample-mean argmax (handled by ``simulate()``).
    """

    state: dict[str, int] = {"next_arm": 0}

    def step(obs: torch.Tensor, **_: object) -> torch.Tensor | None:
        m, n = obs.shape
        if m <= 0 or n <= 0:
            return None

        start = int(state["next_arm"]) % m
        chosen: int | None = None
        for i in range(m):
            k = (start + i) % m
            if torch.isnan(obs[k]).any():
                chosen = k
                break
        if chosen is None:
            return None

        unobs = torch.isnan(obs[chosen]).nonzero().flatten()
        if unobs.numel() == 0:
            return None
        bsz = min(int(batch_size), int(unobs.numel()))
        perm = torch.randperm(int(unobs.numel()))[:bsz]
        cols = unobs[perm].long()
        rows = torch.full((bsz,), int(chosen), dtype=torch.long)
        state["next_arm"] = (int(chosen) + 1) % m
        return torch.stack([rows, cols])

    return step


def _lists_from_npz_trace(z: Any, xkey: str, ykey: str) -> tuple[list[int], list[float]]:
    x = z[xkey]
    y = z[ykey]
    if x.size == 0:
        return [], []
    return x.astype(np.int64).tolist(), y.astype(np.float64).tolist()


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
    pass_sim_cum_eval: bool = False,
) -> tuple[list[float], list[int], dict[str, Any]]:
    """Run until eval budget is reached, the matrix is exhausted, or ``batch is None``.

    Returns parallel lists: regret after each batch, and cumulative number of
    entries revealed (batch sizes summed), including warm-up queries for LRF.

    If ``pass_sim_cum_eval`` is True, each ``step`` call also receives
    ``sim_cum_eval=<cumulative evals so far>`` (for Gittins nominal stopping-time bookkeeping).

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
    iter_total_s: list[float] = []
    iter_step_s: list[float] = []

    while True:
        if max_evaluations is not None and evaluated >= max_evaluations:
            break
        t_iter0 = time.time()
        call_kw = dict(step_kwargs)
        if pass_sim_cum_eval:
            call_kw["sim_cum_eval"] = evaluated
        t_step0 = time.time()
        batch = step(obs, **call_kw)
        t_step1 = time.time()
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

        t_iter1 = time.time()
        iter_total_s.append(float(t_iter1 - t_iter0))
        iter_step_s.append(float(t_step1 - t_step0))

    timing = {
        "clock": "time.time",
        "unit": "seconds",
        "iter_total_s": iter_total_s,
        "iter_step_s": iter_step_s,
    }
    return regrets, cum_evaluated, timing


def _timing_summary(times_s: list[float]) -> dict[str, float | int]:
    if not times_s:
        return {"n": 0}
    arr = np.asarray(times_s, dtype=np.float64)
    return {
        "n": int(arr.size),
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "median_s": float(np.quantile(arr, 0.5)),
        "p90_s": float(np.quantile(arr, 0.9)),
        "p99_s": float(np.quantile(arr, 0.99)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def _timing_for_meta(
    timing: dict[str, Any] | None, *, include_per_iter_series: bool
) -> dict[str, Any] | None:
    """Serialize simulate() timing for JSON meta (summary always; full series optional)."""
    if timing is None:
        return None
    out: dict[str, Any] = {
        "clock": timing["clock"],
        "unit": timing["unit"],
        "summary": {
            "iter_total": _timing_summary(timing["iter_total_s"]),
            "iter_step": _timing_summary(timing["iter_step_s"]),
        },
    }
    if include_per_iter_series:
        out["iter_total_s"] = list(timing["iter_total_s"])
        out["iter_step_s"] = list(timing["iter_step_s"])
    return out


def save_trace_bundle(
    path: Path,
    *,
    xs_rr: list[int],
    regrets_rr: list[float],
    xs_ucbe: list[int],
    regrets_ucbe: list[float],
    xs_lrf: list[int],
    regrets_lrf: list[float],
    xs_lrf_plot: list[int],
    regrets_lrf_plot: list[float],
    xs_gittins: list[int],
    regrets_gittins: list[float],
    gittins_stop_cum_eval: int | None,
    warmup_evals: int,
    budget_evals: int,
    tau_sq_gittins: float,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_scalar = np.int64(-1 if gittins_stop_cum_eval is None else int(gittins_stop_cum_eval))
    np.savez_compressed(
        path,
        rr_x=np.asarray(xs_rr, dtype=np.int64),
        rr_regret=np.asarray(regrets_rr, dtype=np.float64),
        ucb_x=np.asarray(xs_ucbe, dtype=np.int64),
        ucb_regret=np.asarray(regrets_ucbe, dtype=np.float64),
        lrf_x_full=np.asarray(xs_lrf, dtype=np.int64),
        lrf_regret_full=np.asarray(regrets_lrf, dtype=np.float64),
        lrf_x_plot=np.asarray(xs_lrf_plot, dtype=np.int64),
        lrf_regret_plot=np.asarray(regrets_lrf_plot, dtype=np.float64),
        gittins_x=np.asarray(xs_gittins, dtype=np.int64),
        gittins_regret=np.asarray(regrets_gittins, dtype=np.float64),
        gittins_stop_cum_eval=stop_scalar,
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
        default=root / "data" / "matrices" / "gsm8k_1_samples_various_models_seed1.npy",
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
        default=0.2,
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
        choices=["rr", "ucb", "lrf", "gittins"],
        default=["ucb", "lrf", "gittins"],
        metavar="NAME",
        help="Which algorithms to run and plot (default: all three). Example: --algorithms gittins",
    )
    parser.add_argument(
        "--merge-ucb-lrf-from",
        type=Path,
        default=None,
        help="Use with --algorithms gittins only: load UCB-E and UCB-E-LRF traces from this .npz "
        "and simulate only Gittins. LRF plot trim uses warmup_evals stored in that .npz when present.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each simulation step: arms in batch, incumbent arm, simple regret",
    )
    parser.add_argument(
        "--timing-include-per-iter-series",
        action="store_true",
        help="When saving traces meta, include per-iteration time arrays (iter_total_s / iter_step_s); "
        "default is summary statistics only to keep .meta.json small.",
    )
    args = parser.parse_args()

    algorithms = list(dict.fromkeys(args.algorithms))
    merge_ucb_lrf = args.merge_ucb_lrf_from
    if merge_ucb_lrf is not None:
        if set(algorithms) != {"gittins"}:
            print(
                "With --merge-ucb-lrf-from, use exactly: --algorithms gittins",
                file=sys.stderr,
            )
            return 1
        if not merge_ucb_lrf.is_file():
            print(f"--merge-ucb-lrf-from not found: {merge_ucb_lrf}", file=sys.stderr)
            return 1

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
    warmup_evals_lrf_trim = warmup_evals

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
    regrets_rr: list[float] = []
    xs_rr: list[int] = []
    regrets_gittins: list[float] = []
    xs_gittins: list[int] = []
    gittins_stop_cum_eval: int | None = None
    timing_ucb: dict[str, Any] | None = None
    timing_lrf: dict[str, Any] | None = None
    timing_rr: dict[str, Any] | None = None
    timing_gittins: dict[str, Any] | None = None

    if merge_ucb_lrf is not None:
        z_merge = np.load(merge_ucb_lrf)
        if "warmup_evals" in z_merge.files:
            warmup_evals_lrf_trim = int(z_merge["warmup_evals"].reshape(()))
        xs_ucbe, regrets_ucbe = _lists_from_npz_trace(z_merge, "ucb_x", "ucb_regret")
        xs_lrf, regrets_lrf = _lists_from_npz_trace(z_merge, "lrf_x_full", "lrf_regret_full")
        if not xs_ucbe and not xs_lrf:
            print(
                "Merge file has empty UCB and LRF traces; check --merge-ucb-lrf-from path.",
                file=sys.stderr,
            )
            return 1

    if "ucb" in algorithms:
        regrets_ucbe, xs_ucbe, timing_ucb = simulate(
            ground_truth,
            upper_confidence_bound_exploration,
            step_kwargs={"a": args.a, "batch_size": args.batch_size, "return_mus": False},
            log_prefix="ucb",
            **sim_kwargs,
        )
    if "lrf" in algorithms:
        regrets_lrf, xs_lrf, timing_lrf = simulate(
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
    if "rr" in algorithms:
        regrets_rr, xs_rr, timing_rr = simulate(
            ground_truth,
            make_round_robin_step(batch_size=args.batch_size),
            step_kwargs={},
            log_prefix="rr",
            **sim_kwargs,
        )
    if "gittins" in algorithms:
        gittins_natural_stop_holder: list[int | None] = [None]
        regrets_gittins, xs_gittins, timing_gittins = simulate(
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
                allow_early_stop=False,
                natural_stop_cum_eval_holder=gittins_natural_stop_holder,
            ),
            step_kwargs={},
            log_prefix="gittins",
            pass_sim_cum_eval=True,
            **sim_kwargs,
        )
        gittins_stop_cum_eval = gittins_natural_stop_holder[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plot_algorithms = ["ucb", "lrf", "gittins"] if merge_ucb_lrf is not None else algorithms
    xs_lrf_plot, regrets_lrf_plot = _trim_trace_from_cum_eval(
        xs_lrf, regrets_lrf, warmup_evals_lrf_trim
    )

    plt.figure(figsize=(8, 5))
    if "rr" in plot_algorithms:
        plt.plot(xs_rr, regrets_rr, label="Round-robin (sample mean)", linewidth=1.5)
    if "ucb" in plot_algorithms:
        plt.plot(xs_ucbe, regrets_ucbe, label="UCB-E", linewidth=1.5)
    if "lrf" in plot_algorithms:
        plt.plot(xs_lrf_plot, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
    if "gittins" in plot_algorithms:
        gittins_label = (
            "Gittins (τ² = 1/(4B), per-cell DP)"
            if args.gittins_per_cell_dp
            else "Gittins (τ² = 1/(4B), batch-mean DP)"
        )
        (line_gittins,) = plt.plot(
            xs_gittins, regrets_gittins, label=gittins_label, linewidth=1.5
        )
        if gittins_stop_cum_eval is not None:
            plt.axvline(
                gittins_stop_cum_eval,
                color=line_gittins.get_color(),
                linestyle="--",
                alpha=0.85,
                linewidth=1.2,
                label=f"Gittins nominal stop ({gittins_stop_cum_eval} evals)",
            )
    plt.xlabel("Cumulative examples evaluated (matrix entries revealed)")
    plt.ylabel("Simple regret")
    batch_desc_parts: list[str] = []
    if "rr" in plot_algorithms or "ucb" in plot_algorithms or "lrf" in plot_algorithms:
        batch_desc_parts.append(f"UCB/LRF batch={args.batch_size}")
    if "gittins" in plot_algorithms:
        batch_desc_parts.append(f"Gittins batch={args.gittins_batch_size}")
    batch_desc = ", ".join(batch_desc_parts) if batch_desc_parts else f"batch={args.batch_size}"
    sub = (
        f"seed={args.seed}, {batch_desc}, budget={args.eval_budget_fraction:.0%} of {n_cells} cells"
    )
    if "lrf" in plot_algorithms:
        sub += (
            f"\n(LRF: {args.warmup_percentage:.0%} random warm-up, then low-rank UCB; "
            f"curve starts at ~{warmup_evals_lrf_trim} evals)"
        )
    if merge_ucb_lrf is not None:
        sub += f"\n(UCB/LRF merged from {merge_ucb_lrf.name})"
    sub = f"algorithms={','.join(plot_algorithms)} | " + sub
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
        timing_meta = {
            "rr": _timing_for_meta(
                timing_rr, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "ucb": _timing_for_meta(
                timing_ucb, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "lrf": _timing_for_meta(
                timing_lrf, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "gittins": _timing_for_meta(
                timing_gittins, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "per_iter_series_included": bool(args.timing_include_per_iter_series),
        }
        save_trace_bundle(
            traces_path,
            xs_rr=xs_rr,
            regrets_rr=regrets_rr,
            xs_ucbe=xs_ucbe,
            regrets_ucbe=regrets_ucbe,
            xs_lrf=xs_lrf,
            regrets_lrf=regrets_lrf,
            xs_lrf_plot=xs_lrf_plot,
            regrets_lrf_plot=regrets_lrf_plot,
            xs_gittins=xs_gittins,
            regrets_gittins=regrets_gittins,
            gittins_stop_cum_eval=gittins_stop_cum_eval,
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
                "algorithms": plot_algorithms,
                "title": f"Simple regret — {args.matrix.name}\n{sub}",
                "gittins_stop_cum_eval": gittins_stop_cum_eval,
                "merge_ucb_lrf_from": str(merge_ucb_lrf.resolve()) if merge_ucb_lrf else None,
                "warmup_evals_lrf_plot_trim": warmup_evals_lrf_trim,
                "timing": timing_meta,
            },
        )
        print(f"Wrote traces {traces_path} and {traces_path.with_suffix('.meta.json')}")

    print(f"Wrote {args.out}")
    if "rr" in plot_algorithms:
        print(
            f"Round-robin: {len(regrets_rr)} batches, {xs_rr[-1] if xs_rr else 0} / {budget_evals} budget evals "
            f"(B = {args.batch_size})"
        )
        if "rr" in algorithms:
            s_total = _timing_summary(timing_rr["iter_total_s"])
            s_step = _timing_summary(timing_rr["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    if "ucb" in plot_algorithms:
        src = " (merged)" if merge_ucb_lrf and "ucb" not in algorithms else ""
        print(
            f"UCB-E{src}: {len(regrets_ucbe)} batches, {xs_ucbe[-1] if xs_ucbe else 0} / {budget_evals} budget evals"
        )
        if "ucb" in algorithms:
            s_total = _timing_summary(timing_ucb["iter_total_s"])
            s_step = _timing_summary(timing_ucb["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    if "lrf" in plot_algorithms:
        src = " (merged)" if merge_ucb_lrf and "lrf" not in algorithms else ""
        print(
            f"UCB-E-LRF{src}: {len(regrets_lrf)} batches, {xs_lrf[-1] if xs_lrf else 0} / {budget_evals} budget evals "
            f"({len(regrets_lrf_plot)} plotted points from cum_eval ≥ {warmup_evals_lrf_trim})"
        )
        if "lrf" in algorithms:
            s_total = _timing_summary(timing_lrf["iter_total_s"])
            s_step = _timing_summary(timing_lrf["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    if "gittins" in plot_algorithms:
        stop_msg = (
            f", nominal stop marker at cum_eval={gittins_stop_cum_eval} (curve to budget)"
            if gittins_stop_cum_eval is not None
            else ""
        )
        print(
            f"Gittins: {len(regrets_gittins)} batches, {xs_gittins[-1] if xs_gittins else 0} / {budget_evals} budget evals "
            f"(B = {args.gittins_batch_size}, τ² = 1/(4B) = {tau_sq_gittins}){stop_msg}"
        )
        if "gittins" in algorithms:
            s_total = _timing_summary(timing_gittins["iter_total_s"])
            s_step = _timing_summary(timing_gittins["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
