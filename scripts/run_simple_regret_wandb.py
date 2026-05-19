#!/usr/bin/env python3
"""W&B runner for matrix-bandit simple-regret experiments.

One W&B run = one concrete experiment configuration.

Supported experiment variants:
- ucb_B{B}
- lrf_B{B}
- gittins_unit_B{B}_scale{S}_{default|dataset}
- gittins_aware_B{B}_scale{S}_{default|dataset}

This runner is designed for the current lightweight code path:
- UCB / LRF are run directly from banditeval.bandits.
- Gittins is run through src/gittins_policy.py with a precomputed root lookup table.
- Timing is recorded for Gittins lookup table construction and every policy iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from banditeval.bandits import upper_confidence_bound_exploration

try:
    from banditeval.bandits import upper_confidence_bound_exploration_low_rank_factorization
except Exception:  # pragma: no cover
    upper_confidence_bound_exploration_low_rank_factorization = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gittins_lookup import compute_roots_lookup_table  # noqa: E402
from gittins_policy import gittins_index_exploration  # noqa: E402
from gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior  # noqa: E402


DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_VARIANCE = 0.04

# Dataset-specific priors requested for the current GSM8K / PIQA workflow.
# Note: PIQA variance is set to 0.02, following the final value in the user-provided note.
DATASET_PRIORS: dict[str, tuple[float, float]] = {
    "gsm8k": (0.2, 0.01),
    "piqa": (0.3, 0.02),
}


@dataclass(frozen=True)
class VariantConfig:
    raw: str
    policy_variant: str
    policy_family: str
    cost_mode: str
    batch_size: int
    gittins_batch_size: int
    cost_scaling_factor: float
    prior_type: str


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_experiment_variant(raw: str) -> VariantConfig:
    """Parse compact variant labels into concrete runner parameters."""
    v = raw.strip()

    m = re.fullmatch(r"(ucb|lrf)_B(\d+)", v)
    if m:
        policy = m.group(1)
        b = int(m.group(2))
        return VariantConfig(
            raw=v,
            policy_variant=policy,
            policy_family=policy,
            cost_mode="baseline",
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=1e-4,
            prior_type="default",
        )

    m = re.fullmatch(
        r"gittins_(unit|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)",
        v,
    )
    if m:
        cost_mode = m.group(1)
        b = int(m.group(2))
        scale = float(m.group(3))
        prior_type = m.group(4)
        return VariantConfig(
            raw=v,
            policy_variant=f"gittins_{cost_mode}",
            policy_family="gittins",
            cost_mode=cost_mode,
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=scale,
            prior_type=prior_type,
        )

    # Backward-compatible simple labels from the earlier pilot workflow.
    if v in {"ucb", "lrf", "gittins_unit", "gittins_aware"}:
        if v == "ucb":
            return VariantConfig(v, "ucb", "ucb", "baseline", 20, 20, 1e-4, "default")
        if v == "lrf":
            return VariantConfig(v, "lrf", "lrf", "baseline", 20, 20, 1e-4, "default")
        cost_mode = "unit" if v == "gittins_unit" else "aware"
        return VariantConfig(v, v, "gittins", cost_mode, 20, 20, 1e-4, "default")

    raise ValueError(
        f"Unsupported experiment variant: {raw!r}. Examples: "
        "ucb_B20, lrf_B20, gittins_unit_B20_scale1e-4_default, "
        "gittins_aware_B20_scale1e-4_dataset"
    )


def infer_matrix_seed(matrix: Path) -> str | None:
    m = re.search(r"seed(\d+)", matrix.stem)
    return m.group(1) if m else None


def load_cost_vector(path: Path, n_arms: int) -> torch.Tensor:
    """Load per-arm costs with shape (n_arms,) from .npy or supported .json formats."""
    if not path.is_file():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".npy":
        arr = np.squeeze(np.load(path)).astype(np.float64)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            arr = np.asarray(data, dtype=np.float64)
        elif isinstance(data, dict) and "cost_per_arm" in data and isinstance(data["cost_per_arm"], list):
            arr = np.asarray(data["cost_per_arm"], dtype=np.float64)
        elif isinstance(data, dict) and "0" in data and isinstance(data["0"], dict):
            # Current pricing config format.
            values = []
            for i in range(n_arms):
                row = data.get(str(i))
                if not isinstance(row, dict):
                    raise ValueError(f"missing pricing row for arm {i}")
                if "estimated_cost_per_1m_input_tokens" in row:
                    values.append(float(row["estimated_cost_per_1m_input_tokens"]))
                elif "cost" in row:
                    values.append(float(row["cost"]))
                else:
                    raise ValueError(
                        f"pricing row {i} must contain estimated_cost_per_1m_input_tokens or cost"
                    )
            arr = np.asarray(values, dtype=np.float64)
        else:
            raise ValueError(f"unsupported JSON cost format: {path}")
    else:
        raise ValueError(f"cost vector must be .npy or .json, got {path}")

    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.shape != (n_arms,):
        raise ValueError(f"cost vector shape must be ({n_arms},), got {arr.shape}")
    return torch.tensor(arr, dtype=torch.float64)


def empirical_incumbent(obs: torch.Tensor) -> tuple[int, torch.Tensor]:
    mus = torch.nanmean(obs, dim=1)
    scores = torch.where(torch.isnan(mus), torch.full_like(mus, -float("inf")), mus)
    if not torch.isfinite(scores).any():
        return 0, mus
    return int(torch.argmax(scores).item()), mus


def posterior_means(
    obs: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> torch.Tensor:
    counts = (~obs.isnan()).sum(dim=1).to(torch.float64)
    obs_sum = torch.nan_to_num(obs, nan=0.0).sum(dim=1).to(torch.float64)
    v0 = float(prior_variance)
    prec = 1.0 / v0 + counts / float(tau_sq_cell)
    v_t = 1.0 / prec
    mus = v_t * (float(prior_mean) / v0 + obs_sum / float(tau_sq_cell))
    mus[counts == 0] = float(prior_mean)
    return mus.to(torch.float32)


def timing_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "total_s": 0.0,
            "mean_s": None,
            "median_s": None,
            "p90_s": None,
            "p99_s": None,
            "min_s": None,
            "max_s": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p90_s": float(np.percentile(arr, 90)),
        "p99_s": float(np.percentile(arr, 99)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def save_line_plot(
    path: Path,
    *,
    x: list[float] | list[int],
    y: list[float],
    xlabel: str,
    title: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if x and y:
        plt.plot(x, y, linewidth=1.7, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Simple regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def simulate_timed(
    *,
    ground_truth: torch.Tensor,
    step_fn: Callable[[torch.Tensor, int], Any],
    seed: int,
    max_evaluations: int,
    original_cost_per_arm: torch.Tensor,
    recommend_fn: Callable[[torch.Tensor, Any], tuple[int, torch.Tensor]],
    run: wandb.sdk.wandb_run.Run | None,
    log_step_metrics: bool,
    min_evaluations_before_natural_stop: int = 0,
    should_stop_after_step: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))

    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    x: list[int] = []
    x_original_cost: list[float] = []
    regret: list[float] = []
    recommended_arm: list[int] = []
    recommended_mean: list[float] = []
    iter_step_s: list[float] = []
    iter_total_s: list[float] = []
    batch_cells: list[int] = []

    evaluated = 0
    total_cost = 0.0
    step_idx = 0

    while evaluated < max_evaluations:
        t0 = time.perf_counter()
        ts0 = time.perf_counter()
        out = step_fn(obs, evaluated)
        ts1 = time.perf_counter()

        if out is None:
            break
        if isinstance(out, tuple):
            batch, aux = out
        else:
            batch, aux = out, None
        if batch is None:
            break

        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        if n_batch <= 0:
            break

        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch

        pulled_arm = int(row_idx[0].item())
        total_cost += float(original_cost_per_arm[pulled_arm].item()) * float(n_batch)

        arm, mus = recommend_fn(obs, aux)
        simple_regret = mu_star - float(true_means[arm].item())

        t1 = time.perf_counter()
        step_time = float(ts1 - ts0)
        total_time = float(t1 - t0)

        step_idx += 1
        x.append(int(evaluated))
        x_original_cost.append(float(total_cost))
        regret.append(float(simple_regret))
        recommended_arm.append(int(arm))
        recommended_mean.append(float(mus[arm].item()) if torch.isfinite(mus[arm]) else float("nan"))
        iter_step_s.append(step_time)
        iter_total_s.append(total_time)
        batch_cells.append(n_batch)

        if run is not None and log_step_metrics:
            run.log(
                {
                    "cum_eval": int(evaluated),
                    "cum_original_cost": float(total_cost),
                    "simple_regret": float(simple_regret),
                    "recommended_arm": int(arm),
                    "recommended_mean": recommended_mean[-1],
                    "iter_step_s": step_time,
                    "iter_total_s": total_time,
                    "batch_cells": n_batch,
                    "step_idx": step_idx,
                }
            )

        if (
            should_stop_after_step is not None
            and evaluated >= int(min_evaluations_before_natural_stop)
            and should_stop_after_step(int(evaluated))
        ):
            break

    return {
        "x": x,
        "x_original_cost": x_original_cost,
        "regret": regret,
        "recommended_arm": recommended_arm,
        "recommended_mean": recommended_mean,
        "iter_step_s": iter_step_s,
        "iter_total_s": iter_total_s,
        "batch_cells": batch_cells,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--dataset-tag", "--dataset_tag", dest="dataset_tag", required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument(
        "--experiment-variant",
        "--experiment_variant",
        "--policy-variant",
        "--policy_variant",
        dest="experiment_variant",
        required=True,
    )
    p.add_argument("--run-seed", "--run_seed", "--seed", dest="run_seed", type=int, default=0)

    p.add_argument("--eval-budget-fraction", "--eval_budget_fraction", type=float, default=0.1)
    p.add_argument("--ucb-a", "--ucb_a", dest="ucb_a", type=float, default=1.0)
    p.add_argument("--warmup-percentage", "--warmup_percentage", type=float, default=0.05)
    p.add_argument("--lrf-device", "--lrf_device", default="cpu")

    p.add_argument("--cost-vector", "--cost_vector", type=Path, default=None)
    p.add_argument("--gittins-grid-points", "--gittins_grid_points", type=int, default=2**10 + 1)
    p.add_argument("--gittins-obs-noise-variance", "--gittins_obs_noise_variance", type=float, default=None)

    # Optional explicit overrides. Normally these are resolved from experiment_variant.
    p.add_argument("--gittins-prior-mean", "--gittins_prior_mean", type=float, default=None)
    p.add_argument("--gittins-prior-variance", "--gittins_prior_variance", type=float, default=None)

    p.add_argument("--out-dir", "--out_dir", type=Path, default=Path("outputs") / "wandb_simple_regret")
    p.add_argument("--log-step-metrics", "--log_step_metrics", action="store_true")
    p.add_argument(
        "--extend-gittins-to-natural-stop",
        "--extend_gittins_to_natural_stop",
        action="store_true",
        help=(
            "For Gittins variants, keep running past the nominal eval budget until the "
            "natural stopping time is observed, capped by the full matrix."
        ),
    )

    p.add_argument("--wandb-entity", "--wandb_entity", default=None)
    p.add_argument("--wandb-project", "--wandb_project", default="GittinsBanditEval")
    p.add_argument("--wandb-group", "--wandb_group", default=None)
    p.add_argument("--wandb-name", "--wandb_name", default=None)
    p.add_argument("--wandb-mode", "--wandb_mode", choices=["online", "offline", "disabled"], default="online")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    variant = parse_experiment_variant(args.experiment_variant)
    dataset_tag = safe_token(args.dataset_tag.lower())
    matrix_seed = infer_matrix_seed(args.matrix)

    if not args.matrix.is_file():
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    gt_np = np.load(args.matrix)
    if gt_np.ndim != 2:
        print(f"Expected 2D matrix, got shape {gt_np.shape}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms, n_examples = int(ground_truth.shape[0]), int(ground_truth.shape[1])
    n_cells = int(ground_truth.numel())
    max_evaluations = int(max(1, round(float(args.eval_budget_fraction) * n_cells)))

    if args.cost_vector is not None:
        actual_cost_per_arm = load_cost_vector(args.cost_vector, n_arms)
    else:
        actual_cost_per_arm = torch.ones((n_arms,), dtype=torch.float64)

    if variant.prior_type == "default":
        prior_mean, prior_variance = DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_VARIANCE
    elif variant.prior_type == "dataset":
        prior_mean, prior_variance = DATASET_PRIORS.get(dataset_tag, (DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_VARIANCE))
    else:
        raise ValueError(f"Unsupported prior_type: {variant.prior_type}")

    if args.gittins_prior_mean is not None:
        prior_mean = float(args.gittins_prior_mean)
    if args.gittins_prior_variance is not None:
        prior_variance = float(args.gittins_prior_variance)

    matrix_seed_label = f"seed{matrix_seed}" if matrix_seed is not None else "seedNA"
    run_name = args.wandb_name or f"{dataset_tag}_{matrix_seed_label}_{variant.raw}_runseed{args.run_seed}"


    group = args.wandb_group or f"{dataset_tag}_simple_regret_sweep"

    run: wandb.sdk.wandb_run.Run | None = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=group,
            name=run_name,
            job_type="simple_regret",
            mode=args.wandb_mode,
            config={
                **vars(args),
                **asdict(variant),
                "dataset_tag_resolved": dataset_tag,
                "matrix_seed": matrix_seed,
                "n_arms": n_arms,
                "n_examples": n_examples,
                "n_cells": n_cells,
                "budget_max_evals": max_evaluations,
                "prior_mean_resolved": prior_mean,
                "prior_variance_resolved": prior_variance,
            },
        )
        run.define_metric("cum_eval")
        run.define_metric("simple_regret", step_metric="cum_eval")
        run.define_metric("cum_original_cost")
        run.define_metric("iter_step_s", step_metric="cum_eval")
        run.define_metric("iter_total_s", step_metric="cum_eval")

    out_base = (
        args.out_dir
        / dataset_tag
        / safe_token(variant.raw)
        / f"{Path(args.matrix).stem}__runseed{args.run_seed}__{safe_token(variant.raw)}"
    )
    trace_path = out_base.with_name(out_base.name + "_traces.npz")
    meta_path = out_base.with_name(out_base.name + "_meta.json")
    fig_eval_path = out_base.with_name(out_base.name + "_regret_vs_evals.png")
    fig_cost_path = out_base.with_name(out_base.name + "_regret_vs_cost.png")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    lookup_table_s: float | None = None
    natural_stop_holder: list[int | None] = [None]

    if variant.policy_family == "ucb":
        def step_fn(obs: torch.Tensor, sim_cum_eval: int):
            return upper_confidence_bound_exploration(
                obs,
                a=float(args.ucb_a),
                batch_size=int(variant.batch_size),
                return_mus=False,
            )

        def recommend_fn(obs: torch.Tensor, aux: Any):
            return empirical_incumbent(obs)

    elif variant.policy_family == "lrf":
        if upper_confidence_bound_exploration_low_rank_factorization is None:
            print("LRF policy is unavailable in banditeval.bandits.", file=sys.stderr)
            return 1
        warmup_evals = int(math.ceil(float(args.warmup_percentage) * n_cells))
        if warmup_evals >= max_evaluations:
            print(
                "LRF warmup_evals is >= max_evaluations. "
                "Increase --eval-budget-fraction or decrease --warmup-percentage.",
                file=sys.stderr,
            )
            return 1

        def step_fn(obs: torch.Tensor, sim_cum_eval: int):
            return upper_confidence_bound_exploration_low_rank_factorization(
                obs,
                a=float(args.ucb_a),
                batch_size=int(variant.batch_size),
                return_mus=False,
                warmup_percentage=float(args.warmup_percentage),
                device=str(args.lrf_device),
            )

        def recommend_fn(obs: torch.Tensor, aux: Any):
            return empirical_incumbent(obs)

    elif variant.policy_family == "gittins":
        B = int(variant.gittins_batch_size)
        tau_sq_batch = (
            float(args.gittins_obs_noise_variance)
            if args.gittins_obs_noise_variance is not None
            else 1.0 / (4.0 * float(B))
        )
        tau_sq_cell = float(tau_sq_batch) * float(B)

        if variant.cost_mode == "aware":
            if args.cost_vector is None:
                print("gittins_aware requires --cost-vector", file=sys.stderr)
                return 1
            decision_cost_per_arm = actual_cost_per_arm.clone()
        elif variant.cost_mode == "unit":
            decision_cost_per_arm = torch.ones((n_arms,), dtype=torch.float64)
        else:
            raise ValueError(f"Unsupported Gittins cost_mode: {variant.cost_mode}")

        t_lookup0 = time.perf_counter()
        transition_stds = transition_stds_shrinking_gaussian_posterior(
            np.float32(float(prior_variance)),
            np.float32(float(tau_sq_cell)),
            int(n_examples),
        )
        dp_costs_per_arm = (
            decision_cost_per_arm.to(torch.float32).numpy()
            * float(variant.cost_scaling_factor)
        ).astype(np.float32)
        roots = compute_roots_lookup_table(
            transition_stds=transition_stds,
            costs_per_arm=dp_costs_per_arm,
            n_points=int(args.gittins_grid_points),
        )
        roots_torch = torch.tensor(np.array(roots), dtype=torch.float32)
        lookup_table_s = float(time.perf_counter() - t_lookup0)

        cached_scores = torch.full((n_arms,), float("inf"), dtype=torch.float32)
        prev_arm: int | None = None

        def step_fn(obs: torch.Tensor, sim_cum_eval: int):
            nonlocal prev_arm
            recompute = None if prev_arm is None else [prev_arm]
            out = gittins_index_exploration(
                obs,
                prior_mean=float(prior_mean),
                prior_variance=float(prior_variance),
                obs_noise_variance=float(tau_sq_batch),
                cost_per_transition=decision_cost_per_arm,
                cost_scaling_factor=float(variant.cost_scaling_factor),
                n_gittins_grid_points=int(args.gittins_grid_points),
                batch_size=B,
                return_mus=True,
                cached_scores=cached_scores,
                recompute_arms=recompute,
                use_batch_mean_gittins_dp=False,
                allow_early_stop=False,
                sim_cum_eval=int(sim_cum_eval),
                natural_stop_cum_eval_holder=natural_stop_holder,
                roots_lookup_table=roots_torch,
                batch_observation_model=True,
            )
            if isinstance(out, tuple):
                batch = out[0]
            else:
                batch = out
            if batch is not None:
                prev_arm = int(batch[0, 0].item())
            return out

        def recommend_fn(obs: torch.Tensor, aux: Any):
            # Recommendation is evaluated after the simulator reveals the selected batch.
            # Recompute posterior means from updated observations instead of using pre-reveal aux.
            mus = posterior_means(
                obs,
                prior_mean=float(prior_mean),
                prior_variance=float(prior_variance),
                tau_sq_cell=float(tau_sq_cell),
            )
            scores = torch.where(torch.isnan(mus), torch.full_like(mus, -float("inf")), mus)
            if not torch.isfinite(scores).any():
                return 0, mus
            return int(torch.argmax(scores).item()), mus

    else:
        raise ValueError(f"Unsupported policy family: {variant.policy_family}")

    sim_max_evaluations = (
        int(n_cells)
        if bool(args.extend_gittins_to_natural_stop) and variant.policy_family == "gittins"
        else max_evaluations
    )
    stop_after_step = (
        (lambda evaluated: natural_stop_holder[0] is not None)
        if bool(args.extend_gittins_to_natural_stop) and variant.policy_family == "gittins"
        else None
    )

    sim = simulate_timed(
        ground_truth=ground_truth,
        step_fn=step_fn,
        seed=int(args.run_seed),
        max_evaluations=sim_max_evaluations,
        original_cost_per_arm=actual_cost_per_arm,
        recommend_fn=recommend_fn,
        run=run,
        log_step_metrics=bool(args.log_step_metrics),
        min_evaluations_before_natural_stop=max_evaluations,
        should_stop_after_step=stop_after_step,
    )
    if (
        args.extend_gittins_to_natural_stop
        and variant.policy_family == "gittins"
        and natural_stop_holder[0] is None
        and sim["x"]
        and int(sim["x"][-1]) >= int(n_cells)
    ):
        natural_stop_holder[0] = int(n_cells)

    np.savez(
        trace_path,
        matrix=str(args.matrix),
        dataset_tag=dataset_tag,
        matrix_seed="" if matrix_seed is None else matrix_seed,
        run_seed=int(args.run_seed),
        experiment_variant=variant.raw,
        policy_variant=variant.policy_variant,
        policy_family=variant.policy_family,
        cost_mode=variant.cost_mode,
        batch_size=int(variant.batch_size),
        gittins_batch_size=int(variant.gittins_batch_size),
        cost_scaling_factor=float(variant.cost_scaling_factor),
        prior_type=variant.prior_type,
        extend_gittins_to_natural_stop=bool(args.extend_gittins_to_natural_stop),
        run_max_evals=int(sim_max_evaluations),
        prior_mean=np.asarray(prior_mean, dtype=np.float32),
        prior_variance=np.asarray(prior_variance, dtype=np.float32),
        lookup_table_s=np.asarray(-1.0 if lookup_table_s is None else lookup_table_s, dtype=np.float64),
        x=np.asarray(sim["x"], dtype=np.int32),
        x_original_cost=np.asarray(sim["x_original_cost"], dtype=np.float64),
        regret=np.asarray(sim["regret"], dtype=np.float32),
        recommended_arm=np.asarray(sim["recommended_arm"], dtype=np.int32),
        recommended_mean=np.asarray(sim["recommended_mean"], dtype=np.float32),
        iter_step_s=np.asarray(sim["iter_step_s"], dtype=np.float64),
        iter_total_s=np.asarray(sim["iter_total_s"], dtype=np.float64),
        batch_cells=np.asarray(sim["batch_cells"], dtype=np.int32),
        cost_per_arm_original=np.asarray(actual_cost_per_arm.numpy(), dtype=np.float64),
        gittins_stop_cum_eval=np.asarray(
            -1 if natural_stop_holder[0] is None else int(natural_stop_holder[0]),
            dtype=np.int32,
        ),
    )

    step_summary = timing_summary(sim["iter_step_s"])
    total_summary = timing_summary(sim["iter_total_s"])
    meta = {
        "matrix": str(args.matrix),
        "dataset_tag": dataset_tag,
        "matrix_seed": matrix_seed,
        "run_seed": int(args.run_seed),
        "experiment_variant": variant.raw,
        "variant": asdict(variant),
        "n_arms": n_arms,
        "n_examples": n_examples,
        "n_cells": n_cells,
        "budget_max_evals": max_evaluations,
        "run_max_evals": int(sim_max_evaluations),
        "extend_gittins_to_natural_stop": bool(args.extend_gittins_to_natural_stop),
        "eval_budget_fraction": float(args.eval_budget_fraction),
        "prior_mean_resolved": float(prior_mean),
        "prior_variance_resolved": float(prior_variance),
        "cost_vector": str(args.cost_vector) if args.cost_vector else None,
        "trace": str(trace_path),
        "figure_eval": str(fig_eval_path),
        "figure_cost": str(fig_cost_path),
        "timing": {
            "lookup_table_s": lookup_table_s,
            "iter_step_s": step_summary,
            "iter_total_s": total_summary,
        },
        "final": {
            "final_simple_regret": float(sim["regret"][-1]) if sim["regret"] else None,
            "best_seen_regret": float(min(sim["regret"])) if sim["regret"] else None,
            "final_cum_eval": int(sim["x"][-1]) if sim["x"] else None,
            "final_cum_original_cost": float(sim["x_original_cost"][-1]) if sim["x_original_cost"] else None,
            "num_batches": len(sim["regret"]),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    title = f"{variant.raw} — {args.matrix.name}\nrun_seed={args.run_seed}, budget={args.eval_budget_fraction:.3g}"
    save_line_plot(
        fig_eval_path,
        x=sim["x"],
        y=sim["regret"],
        xlabel="Cumulative examples evaluated",
        title=title,
        label=variant.raw,
    )
    save_line_plot(
        fig_cost_path,
        x=sim["x_original_cost"],
        y=sim["regret"],
        xlabel="Cumulative original cost",
        title=title,
        label=variant.raw,
    )

    print(f"Wrote trace: {trace_path}")
    print(f"Wrote meta: {meta_path}")
    print(f"Wrote figure: {fig_eval_path}")
    print(f"Wrote cost figure: {fig_cost_path}")
    print(f"lookup_table_s={lookup_table_s}")
    print(f"iter_step_mean_s={step_summary['mean_s']}")
    print(f"iter_step_p90_s={step_summary['p90_s']}")
    print(f"final_simple_regret={meta['final']['final_simple_regret']}")

    if run is not None:
        run.summary.update(
            {
                "final_simple_regret": meta["final"]["final_simple_regret"],
                "best_seen_regret": meta["final"]["best_seen_regret"],
                "final_cum_eval": meta["final"]["final_cum_eval"],
                "final_cum_original_cost": meta["final"]["final_cum_original_cost"],
                "num_batches": meta["final"]["num_batches"],
                "lookup_table_s": lookup_table_s,
                "iter_step_mean_s": step_summary["mean_s"],
                "iter_step_median_s": step_summary["median_s"],
                "iter_step_p90_s": step_summary["p90_s"],
                "iter_total_mean_s": total_summary["mean_s"],
                "iter_total_median_s": total_summary["median_s"],
                "iter_total_p90_s": total_summary["p90_s"],
                "prior_mean_resolved": float(prior_mean),
                "prior_variance_resolved": float(prior_variance),
                "matrix_seed": matrix_seed,
                "run_seed": int(args.run_seed),
                "experiment_variant": variant.raw,
            }
        )
        run.log({"regret_vs_evals": wandb.Image(str(fig_eval_path))})
        run.log({"regret_vs_cost": wandb.Image(str(fig_cost_path))})

        artifact = wandb.Artifact(
            name=f"simple-regret-{run.id}",
            type="experiment_outputs",
            description=f"Outputs for {variant.raw}",
        )
        artifact.add_file(str(trace_path))
        artifact.add_file(str(meta_path))
        artifact.add_file(str(fig_eval_path))
        artifact.add_file(str(fig_cost_path))
        run.log_artifact(artifact)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
