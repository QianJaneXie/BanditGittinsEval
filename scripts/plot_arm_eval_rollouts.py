#!/usr/bin/env python3
"""
Per-arm evaluation rollouts aligned with ``*_traces.npz`` from ``plot_simple_regret.py``.

The trace bundle only stores regret curves; this script reads the sibling ``.meta.json`` (same stem
as ``--traces``), reloads the accuracy matrix, and **re-runs** the same three policies with the same
parameters so per-arm cumulative sample counts match the original simulation (same seed).

Self-contained: does not modify or rely on ``simulate`` in ``plot_simple_regret.py``.

Plots one column of panels (UCB-E, UCB-E-LRF, Gittins): x = batch iteration index (step
``1 … T``), y = cumulative **batch pulls** per arm — a batch counts once for each distinct arm that
received at least one matrix cell in that batch (one batch can add several cells to the same arm but
still counts as one pull for that arm).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

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

from gittins_policy import gittins_index_exploration  # noqa: E402


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


def _incumbent_from_empirical_means(observed: torch.Tensor) -> int:
    row_means = torch.nanmean(observed, dim=1)
    scores = torch.where(torch.isnan(row_means), torch.full_like(row_means, -float("inf")), row_means)
    if not torch.isfinite(scores).any():
        return 0
    return int(torch.argmax(scores).item())


def simulate_with_batch_pull_snapshots(
    ground_truth: torch.Tensor,
    step: Callable[..., torch.Tensor | None],
    *,
    step_kwargs: dict,
    seed: int,
    max_evaluations: int,
    verbose: bool = False,
    log_prefix: str = "",
    batch_pull_snapshots: list[np.ndarray] | None = None,
) -> tuple[list[float], list[int]]:
    """Same loop as ``plot_simple_regret.simulate``, plus optional cumulative batch-pull snapshots."""
    torch.manual_seed(seed)

    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    regrets: list[float] = []
    cum_evaluated: list[int] = []
    evaluated = 0
    tag = log_prefix or "sim"
    n_arms = int(ground_truth.shape[0])
    cumulative_batch_pulls = np.zeros(n_arms, dtype=np.int64)

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

        arm = _incumbent_from_empirical_means(obs)
        mu_sel = float(true_means[arm].item())
        simple_regret = mu_star - mu_sel
        regrets.append(simple_regret)
        cum_evaluated.append(evaluated)
        if batch_pull_snapshots is not None:
            distinct_arms = {int(x) for x in row_idx.reshape(-1).tolist()}
            for k in distinct_arms:
                cumulative_batch_pulls[k] += 1
            batch_pull_snapshots.append(cumulative_batch_pulls.copy())

        if verbose:
            distinct_arms = sorted({int(x) for x in row_idx.reshape(-1).tolist()})
            print(
                f"[{tag}] step {len(regrets)}: cum_eval={evaluated} "
                f"batch_arms(distinct)={distinct_arms} incumbent_arm={arm} "
                f"simple_regret={simple_regret:.6f}",
                flush=True,
            )

    return regrets, cum_evaluated


def _resolve_matrix(meta_path: Path, meta: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    raw = Path(meta["matrix"])
    repo_root = meta_path.resolve().parents[2]
    candidates = [
        raw,
        repo_root / "data" / "benchmark_matrices" / raw.name,
        repo_root / "data" / "mmlu_matrices" / raw.name,
        repo_root / "data" / "matrices" / raw.name,
        meta_path.parent.parent / "benchmark_matrices" / raw.name,
        meta_path.parent.parent / "matrices" / raw.name,
        repo_root / "outputs" / "matrices" / raw.name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Accuracy matrix not found. Tried: {tried}")


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
        help="Path to *_traces.npz (uses *_traces.meta.json beside it)",
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

    meta_path = args.traces.with_suffix(".meta.json")
    if not meta_path.is_file():
        print(f"Meta not found: {meta_path}", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        matrix_path = _resolve_matrix(meta_path, meta, args.matrix)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    algorithms: list[str] = list(meta.get("algorithms", ["ucb", "lrf", "gittins"]))
    gt_np = np.load(matrix_path)
    if gt_np.ndim != 2:
        print(f"Expected 2D matrix, got {gt_np.shape}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms = int(ground_truth.shape[0])
    n_cells = int(ground_truth.numel())
    budget_evals = int(meta.get("budget_evals", max(1, int(round(float(meta["eval_budget_fraction"]) * n_cells)))))
    seed = int(meta["seed"])
    batch_size = int(meta["batch_size"])
    gittins_batch_size = int(meta.get("gittins_batch_size", 20))
    tau_sq = float(meta.get("tau_sq_gittins", 1.0 / (4.0 * gittins_batch_size)))
    ucb_a = int(meta.get("ucb_a", 1))
    warmup_pct = float(meta["warmup_percentage"])
    lrf_device = str(meta.get("lrf_device", "cpu"))
    gittins_grid = int(meta.get("gittins_grid_points", 1025))
    gittins_cost = float(meta.get("gittins_cost", 1e-4))
    gittins_prior_mean = float(meta.get("gittins_prior_mean", 0.2))
    gittins_prior_variance = float(meta.get("gittins_prior_variance", 0.01))
    gittins_per_cell_dp = bool(meta.get("gittins_per_cell_dp", False))

    sim_kwargs = {
        "seed": seed,
        "max_evaluations": budget_evals,
        "verbose": False,
    }

    cmap = plt.colormaps["turbo"]
    colors = cmap(np.linspace(0.0, 1.0, n_arms, endpoint=False))

    panels: list[tuple[str, list[int], np.ndarray]] = []

    if "ucb" in algorithms:
        snaps: list[np.ndarray] = []
        _, _ = simulate_with_batch_pull_snapshots(
            ground_truth,
            upper_confidence_bound_exploration,
            step_kwargs={"a": ucb_a, "batch_size": batch_size, "return_mus": False},
            log_prefix="ucb",
            batch_pull_snapshots=snaps,
            **sim_kwargs,
        )
        it = list(range(1, len(snaps) + 1))
        panels.append(("UCB-E", it, np.stack(snaps, axis=0)))

    if "lrf" in algorithms:
        snaps = []
        _, _ = simulate_with_batch_pull_snapshots(
            ground_truth,
            upper_confidence_bound_exploration_low_rank_factorization,
            step_kwargs={
                "a": ucb_a,
                "batch_size": batch_size,
                "return_mus": False,
                "warmup_percentage": warmup_pct,
                "device": lrf_device,
            },
            log_prefix="lrf",
            batch_pull_snapshots=snaps,
            **sim_kwargs,
        )
        it = list(range(1, len(snaps) + 1))
        panels.append(("UCB-E-LRF (incl. warm-up)", it, np.stack(snaps, axis=0)))

    if "gittins" in algorithms:
        snaps = []
        _, _ = simulate_with_batch_pull_snapshots(
            ground_truth,
            make_gittins_step_with_score_cache(
                batch_size=gittins_batch_size,
                return_mus=False,
                obs_noise_variance=tau_sq,
                cost_per_transition=gittins_cost,
                n_gittins_grid_points=gittins_grid,
                prior_mean=gittins_prior_mean,
                prior_variance=gittins_prior_variance,
                use_batch_mean_gittins_dp=not gittins_per_cell_dp,
            ),
            step_kwargs={},
            log_prefix="gittins",
            batch_pull_snapshots=snaps,
            **sim_kwargs,
        )
        it = list(range(1, len(snaps) + 1))
        mode = "per-cell DP" if gittins_per_cell_dp else "batch-mean DP"
        panels.append((f"Gittins ({mode})", it, np.stack(snaps, axis=0)))

    if not panels:
        print("No algorithms in meta to plot.", file=sys.stderr)
        return 1

    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.8 * nrows), sharex=False, constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    for ax, (title, xs, counts) in zip(axes, panels):
        _plot_arm_panel(ax, xs, counts, title, colors)

    fig.suptitle(
        f"Per-arm cumulative batch pulls vs iteration — {matrix_path.name}\n"
        f"seed={seed}, budget={budget_evals} evals, {n_arms} arms (re-run from trace meta)",
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
