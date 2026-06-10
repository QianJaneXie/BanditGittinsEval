#!/usr/bin/env python3
"""W&B runner for matrix-bandit simple-regret experiments.

One W&B run = one concrete experiment configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

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
from gittins_policy import gittins_index_exploration, gittins_post_pull_update  # noqa: E402
from gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior  # noqa: E402
from simple_regret_recommend import empirical_incumbent, posterior_incumbent  # noqa: E402

DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_VARIANCE = 0.04

# Dataset-specific priors requested for the current GSM8K / PIQA workflow.
# Note: PIQA variance is set to 0.02, following the final value in the user-provided note.
DATASET_PRIORS: dict[str, tuple[float, float]] = {
    "gsm8k": (0.2, 0.01),
    "piqa": (0.3, 0.02),
}

MMLU_PRIOR_BY_BUCKET: dict[str, tuple[float, float]] = {
    "low": (0.4, 0.02),
    "medium": (0.6, 0.02),
    "high": (0.75, 0.01),
}

DEFAULT_MMLU_TASK_METADATA = REPO_ROOT / "data" / "MMLU_matrices" / "task_metadata.json"


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
    v = raw.strip()

    m = re.fullmatch(r"(ucb|lrf)_cost_B(\d+)", v)
    if m:
        policy, b = m.group(1), int(m.group(2))
        return VariantConfig(v, f"{policy}_cost", policy, "cost", b, b, 1e-4, "default")

    m = re.fullmatch(r"(ucb|lrf)_aware_B(\d+)", v)
    if m:
        policy, b = m.group(1), int(m.group(2))
        return VariantConfig(v, f"{policy}_cost", policy, "cost", b, b, 1e-4, "default")

    m = re.fullmatch(r"(ucb|lrf)_B(\d+)", v)
    if m:
        policy, b = m.group(1), int(m.group(2))
        return VariantConfig(v, policy, policy, "unit", b, b, 1e-4, "default")

    m = re.fullmatch(r"gittins_(unit|cost|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)", v)
    if m:
        cost_mode = "cost" if m.group(1) == "aware" else m.group(1)
        return VariantConfig(
            v,
            f"gittins_{cost_mode}",
            "gittins",
            cost_mode,
            int(m.group(2)),
            int(m.group(2)),
            float(m.group(3)),
            m.group(4),
        )

    if v in {"ucb", "lrf", "gittins_unit", "gittins_cost", "gittins_aware"}:
        if v == "ucb":
            return VariantConfig(v, "ucb", "ucb", "unit", 20, 20, 1e-4, "default")
        if v == "lrf":
            return VariantConfig(v, "lrf", "lrf", "unit", 20, 20, 1e-4, "default")
        cost_mode = "unit" if v == "gittins_unit" else "cost"
        policy_variant = "gittins_unit" if v == "gittins_unit" else "gittins_cost"
        return VariantConfig(v, policy_variant, "gittins", cost_mode, 20, 20, 1e-4, "default")

    raise ValueError(f"Unsupported experiment variant: {raw!r}")


def load_cost_vector(path: Path, n_arms: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".npy":
        arr = np.squeeze(np.load(path)).astype(np.float64)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            arr = np.asarray(data, dtype=np.float64)
        elif isinstance(data, dict) and "cost_per_arm" in data:
            arr = np.asarray(data["cost_per_arm"], dtype=np.float64)
        elif isinstance(data, dict) and "0" in data:
            arr = np.asarray(
                [
                    float(data[str(i)]["estimated_cost_per_1m_input_tokens"])
                    if "estimated_cost_per_1m_input_tokens" in data[str(i)]
                    else float(data[str(i)]["cost"])
                    for i in range(n_arms)
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unsupported JSON cost format: {path}")
    else:
        raise ValueError(f"cost vector must be .npy or .json, got {path}")
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.shape != (n_arms,):
        raise ValueError(f"cost vector shape must be ({n_arms},), got {arr.shape}")
    return torch.tensor(arr, dtype=torch.float64)


def resolve_prior(
    *,
    dataset_tag: str,
    prior_type: str,
    matrix: Path,
    mmlu_task_prior_buckets: dict[str, str],
) -> tuple[float, float, str | None, str | None, str]:
    stem = matrix.stem.split("_synthetic_", 1)[0] if "_synthetic_" in matrix.stem else matrix.stem
    task = stem if dataset_tag == "mmlu" and stem else None

    if prior_type == "default":
        return DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_VARIANCE, task, None, "default"
    if prior_type != "dataset":
        raise ValueError(f"Unsupported prior_type: {prior_type}")

    if dataset_tag == "mmlu":
        bucket = mmlu_task_prior_buckets.get(task or "")
        if bucket in MMLU_PRIOR_BY_BUCKET:
            mean, variance = MMLU_PRIOR_BY_BUCKET[bucket]
            return float(mean), float(variance), task, bucket, "mmlu_task_bucket"
        return DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_VARIANCE, task, None, "mmlu_bucket_fallback_default"

    if dataset_tag not in DATASET_PRIORS:
        raise ValueError(f"No dataset prior for dataset_tag={dataset_tag!r}")
    mean, variance = DATASET_PRIORS[dataset_tag]
    return float(mean), float(variance), None, dataset_tag, "dataset_global"


def load_mmlu_task_prior_buckets(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["task"]).strip(): str(row["dataset_prior_bucket"]).strip().lower()
        for row in data.get("tasks", [])
        if isinstance(row, dict)
        and row.get("task")
        and str(row.get("dataset_prior_bucket", "")).strip().lower() in MMLU_PRIOR_BY_BUCKET
    }


def build_policy(
    *,
    variant: VariantConfig,
    args: argparse.Namespace,
    ground_truth: torch.Tensor,
    actual_cost_per_arm: torch.Tensor,
    prior_mean: float,
    prior_variance: float,
    max_evaluations: int,
    n_cells: int,
    natural_stop_holder: list[int | None],
    recommendation_aware_stop_holder: list[int | None],
) -> tuple[
    Callable[[torch.Tensor, int], Any],
    Callable[[torch.Tensor, Any], tuple[int, torch.Tensor]],
    Callable[[torch.Tensor, int, int], dict[str, float] | None] | None,
    float | None,
]:
    n_arms = int(ground_truth.shape[0])
    lookup_table_s: float | None = None

    if variant.policy_family == "ucb":
        def step_fn(obs: torch.Tensor, _sim_cum_eval: int):
            return upper_confidence_bound_exploration(
                obs,
                a=float(args.ucb_a),
                batch_size=int(variant.batch_size),
                return_mus=False,
            )

        return step_fn, empirical_incumbent, None, lookup_table_s

    if variant.policy_family == "lrf":
        if upper_confidence_bound_exploration_low_rank_factorization is None:
            raise RuntimeError("LRF policy is unavailable in banditeval.bandits.")
        warmup_evals = int(math.ceil(float(args.warmup_percentage) * n_cells))
        if warmup_evals >= max_evaluations:
            raise RuntimeError("LRF warmup_evals >= max_evaluations.")

        def step_fn(obs: torch.Tensor, _sim_cum_eval: int):
            return upper_confidence_bound_exploration_low_rank_factorization(
                obs,
                a=float(args.ucb_a),
                batch_size=int(variant.batch_size),
                return_mus=False,
                warmup_percentage=float(args.warmup_percentage),
                device=str(args.lrf_device),
            )

        return step_fn, empirical_incumbent, None, lookup_table_s

    if variant.policy_family != "gittins":
        raise ValueError(f"Unsupported policy family: {variant.policy_family}")

    B = int(variant.gittins_batch_size)
    tau_sq_batch = (
        float(args.gittins_obs_noise_variance)
        if args.gittins_obs_noise_variance is not None
        else 1.0 / (4.0 * float(B))
    )
    tau_sq_cell = float(tau_sq_batch) * float(B)
    if variant.cost_mode in {"cost", "aware"}:
        if args.cost_vector is None:
            raise RuntimeError("gittins_cost requires --cost-vector")
        decision_cost_per_arm = actual_cost_per_arm.clone()
    else:
        decision_cost_per_arm = torch.ones((n_arms,), dtype=torch.float64)

    t0 = time.perf_counter()
    transition_stds = transition_stds_shrinking_gaussian_posterior(
        np.float32(float(prior_variance)),
        np.float32(float(tau_sq_cell)),
        int(ground_truth.shape[1]),
    )
    dp_costs_per_arm = (
        decision_cost_per_arm.to(torch.float32).numpy() * float(variant.cost_scaling_factor)
    ).astype(np.float32)
    roots_torch = torch.tensor(
        np.array(
            compute_roots_lookup_table(
                transition_stds=transition_stds,
                costs_per_arm=dp_costs_per_arm,
                n_points=int(args.gittins_grid_points),
            )
        ),
        dtype=torch.float32,
    )
    lookup_table_s = float(time.perf_counter() - t0)

    cached_scores = torch.full((n_arms,), float("inf"), dtype=torch.float32)
    prev_arm: int | None = None
    gittins_post_pull_kw = dict(
        prior_mean=float(prior_mean),
        prior_variance=float(prior_variance),
        obs_noise_variance=float(tau_sq_batch),
        cost_per_transition=decision_cost_per_arm,
        cost_scaling_factor=float(variant.cost_scaling_factor),
        n_gittins_grid_points=int(args.gittins_grid_points),
        batch_size=B,
        use_batch_mean_gittins_dp=False,
        roots_lookup_table=roots_torch,
        batch_observation_model=True,
        natural_stop_cum_eval_holder=natural_stop_holder,
        recommendation_aware_stop_cum_eval_holder=recommendation_aware_stop_holder,
    )

    def step_fn(obs: torch.Tensor, _sim_cum_eval: int):
        nonlocal prev_arm
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
            recompute_arms=None if prev_arm is None else [prev_arm],
            use_batch_mean_gittins_dp=False,
            allow_early_stop=False,
            roots_lookup_table=roots_torch,
            batch_observation_model=True,
        )
        batch = out[0] if isinstance(out, tuple) else out
        if batch is not None:
            prev_arm = int(batch[0, 0].item())
        return out

    def post_pull_fn(obs: torch.Tensor, pulled_arm: int, cum_eval: int) -> dict[str, float]:
        mus_post, scores_post = gittins_post_pull_update(
            obs,
            cached_scores=cached_scores,
            recompute_arms=[int(pulled_arm)],
            sim_cum_eval=int(cum_eval),
            **gittins_post_pull_kw,
        )
        arm = int(pulled_arm)
        return {
            "gittins_index_pulled": float(scores_post[arm].item()),
            "posterior_mean_pulled": float(mus_post[arm].item()),
        }

    recommend_fn = partial(
        posterior_incumbent,
        prior_mean=float(prior_mean),
        prior_variance=float(prior_variance),
        tau_sq_cell=float(tau_sq_cell),
    )
    return step_fn, recommend_fn, post_pull_fn, lookup_table_s


def run_simple_regret_experiment(
    *,
    ground_truth: torch.Tensor,
    original_cost_per_arm: torch.Tensor,
    variant: VariantConfig,
    args: argparse.Namespace,
    prior_mean: float,
    prior_variance: float,
    max_evaluations: int,
    max_original_cost: float | None,
    run: wandb.sdk.wandb_run.Run | None,
    log_step_metrics: bool,
) -> dict[str, Any]:
    n_cells = int(ground_truth.numel())
    natural_stop_holder: list[int | None] = [None]
    recommendation_aware_stop_holder: list[int | None] = [None]

    step_fn, recommend_fn, post_pull_fn, lookup_table_s = build_policy(
        variant=variant,
        args=args,
        ground_truth=ground_truth,
        actual_cost_per_arm=original_cost_per_arm,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        max_evaluations=max_evaluations,
        n_cells=n_cells,
        natural_stop_holder=natural_stop_holder,
        recommendation_aware_stop_holder=recommendation_aware_stop_holder,
    )

    torch.manual_seed(int(args.run_seed))
    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    history: dict[str, list[Any]] = {
        "x": [],
        "x_original_cost": [],
        "regret": [],
        "recommended_arm": [],
        "recommended_mean": [],
        "pulled_arm": [],
        "gittins_index_pulled": [],
        "posterior_mean_pulled": [],
    }
    natural_stop_cum_original_cost: float | None = None
    recommendation_aware_stop_cum_original_cost: float | None = None
    evaluated = 0
    total_cost = 0.0

    while evaluated < max_evaluations:
        out = step_fn(obs, evaluated)
        if out is None:
            break
        batch, aux = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        if batch is None:
            break

        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        if n_batch <= 0:
            break

        pulled_arm = int(row_idx[0].item())
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch
        total_cost += float(original_cost_per_arm[pulled_arm].item()) * float(n_batch)

        gittins_diag: dict[str, float] | None = None
        if post_pull_fn is not None:
            gittins_diag = post_pull_fn(obs, pulled_arm, int(evaluated))

        if natural_stop_holder[0] == int(evaluated):
            natural_stop_cum_original_cost = float(total_cost)
        if recommendation_aware_stop_holder[0] == int(evaluated):
            recommendation_aware_stop_cum_original_cost = float(total_cost)

        arm, mus = recommend_fn(obs, aux)
        simple_regret = mu_star - float(true_means[arm].item())

        history["x"].append(int(evaluated))
        history["x_original_cost"].append(float(total_cost))
        history["regret"].append(float(simple_regret))
        history["recommended_arm"].append(int(arm))
        history["recommended_mean"].append(
            float(mus[arm].item()) if torch.isfinite(mus[arm]) else float("nan")
        )
        history["pulled_arm"].append(int(pulled_arm))
        if gittins_diag is not None:
            history["gittins_index_pulled"].append(float(gittins_diag["gittins_index_pulled"]))
            history["posterior_mean_pulled"].append(float(gittins_diag["posterior_mean_pulled"]))

        if run is not None and log_step_metrics:
            payload: dict[str, Any] = {
                "cum_eval": int(evaluated),
                "cum_original_cost": float(total_cost),
                "simple_regret": float(simple_regret),
                "recommended_arm": int(arm),
                "recommended_mean": history["recommended_mean"][-1],
                "pulled_arm": int(pulled_arm),
            }
            if gittins_diag is not None:
                payload["gittins_index_pulled"] = float(gittins_diag["gittins_index_pulled"])
                payload["posterior_mean_pulled"] = float(gittins_diag["posterior_mean_pulled"])
            run.log(payload)

        if max_original_cost is not None and total_cost >= float(max_original_cost):
            break

    return {
        **history,
        "lookup_table_s": lookup_table_s,
        "natural_stop_cum_original_cost": natural_stop_cum_original_cost,
        "recommendation_aware_stop_cum_original_cost": recommendation_aware_stop_cum_original_cost,
        "natural_stop_cum_eval": natural_stop_holder[0],
        "recommendation_aware_stop_cum_eval": recommendation_aware_stop_holder[0],
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
    p.add_argument("--gittins-prior-mean", "--gittins_prior_mean", type=float, default=None)
    p.add_argument("--gittins-prior-variance", "--gittins_prior_variance", type=float, default=None)
    p.add_argument("--mmlu-task-metadata", "--mmlu_task_metadata", type=Path, default=DEFAULT_MMLU_TASK_METADATA)
    p.add_argument("--log-step-metrics", "--log_step_metrics", dest="log_step_metrics", action="store_true", default=True)
    p.add_argument("--no-log-step-metrics", "--no_log_step_metrics", dest="log_step_metrics", action="store_false")
    p.add_argument("--wandb-entity", "--wandb_entity", default=None)
    p.add_argument("--wandb-project", "--wandb_project", default="GittinsBanditEval")
    p.add_argument("--wandb-group", "--wandb_group", default=None)
    p.add_argument("--wandb-name", "--wandb_name", default=None)
    p.add_argument("--wandb-mode", "--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def main() -> int:
    wall_t0 = time.perf_counter()
    args = parse_args()
    variant = parse_experiment_variant(args.experiment_variant)
    dataset_tag = safe_token(args.dataset_tag.lower())

    if not args.matrix.is_file():
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(np.load(args.matrix), dtype=torch.float32)
    if ground_truth.ndim != 2:
        print(f"Expected 2D matrix, got shape {tuple(ground_truth.shape)}", file=sys.stderr)
        return 1

    n_arms, n_examples = map(int, ground_truth.shape)
    n_cells = int(ground_truth.numel())
    actual_cost_per_arm = (
        load_cost_vector(args.cost_vector, n_arms)
        if args.cost_vector is not None
        else torch.ones((n_arms,), dtype=torch.float64)
    )

    total_bf_cost = float(n_examples * actual_cost_per_arm.sum().item())
    budget_evals = int(max(1, round(float(args.eval_budget_fraction) * n_cells)))
    cost_aware = variant.cost_mode in {"cost", "aware"}
    if cost_aware:
        if args.cost_vector is None:
            print(f"{variant.raw} requires --cost-vector.", file=sys.stderr)
            return 1
        max_evaluations = n_cells
        max_original_cost = float(args.eval_budget_fraction) * total_bf_cost
        budget_original_cost = max_original_cost
    else:
        max_evaluations = budget_evals
        max_original_cost = None
        budget_original_cost = None

    mmlu_task_prior_buckets = (
        load_mmlu_task_prior_buckets(args.mmlu_task_metadata) if dataset_tag == "mmlu" else {}
    )
    prior_mean, prior_variance, mmlu_task, prior_bucket, prior_source = resolve_prior(
        dataset_tag=dataset_tag,
        prior_type=variant.prior_type,
        matrix=args.matrix,
        mmlu_task_prior_buckets=mmlu_task_prior_buckets,
    )
    if args.gittins_prior_mean is not None:
        prior_mean = float(args.gittins_prior_mean)
    if args.gittins_prior_variance is not None:
        prior_variance = float(args.gittins_prior_variance)

    matrix_seed = re.search(r"seed(\d+)", args.matrix.stem)
    matrix_seed = matrix_seed.group(1) if matrix_seed else None
    size_bucket = "small" if n_examples <= 150 else "medium" if n_examples <= 400 else "large"
    if dataset_tag != "mmlu":
        size_bucket = None
    matrix_seed_label = (
        f"task{safe_token(mmlu_task)}"
        if dataset_tag == "mmlu" and mmlu_task
        else f"seed{matrix_seed}" if matrix_seed else "seedNA"
    )
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
                "budget_max_evals": budget_evals,
                "budget_original_cost": budget_original_cost,
                "total_brute_force_original_cost": total_bf_cost,
                "cost_aware_run": cost_aware,
                "sim_max_evaluations": max_evaluations,
                "prior_mean_resolved": prior_mean,
                "prior_variance_resolved": prior_variance,
                "prior_bucket": prior_bucket,
                "prior_source": prior_source,
                "mmlu_task": mmlu_task,
                "mmlu_size_bucket": size_bucket,
            },
        )
        run.define_metric("cum_eval")
        run.define_metric("simple_regret", step_metric="cum_eval")
        run.define_metric("cum_original_cost")

    try:
        result = run_simple_regret_experiment(
            ground_truth=ground_truth,
            original_cost_per_arm=actual_cost_per_arm,
            variant=variant,
            args=args,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_evaluations=max_evaluations,
            max_original_cost=max_original_cost,
            run=run,
            log_step_metrics=bool(args.log_step_metrics),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        if run is not None:
            run.finish(exit_code=1)
        return 1

    total_wall_time_s = float(time.perf_counter() - wall_t0)
    final_simple_regret = float(result["regret"][-1]) if result["regret"] else None
    best_seen_regret = float(min(result["regret"])) if result["regret"] else None

    print(f"final_simple_regret={final_simple_regret}")

    if run is not None:
        run.summary.update(
            {
                "final_simple_regret": final_simple_regret,
                "best_seen_regret": best_seen_regret,
                "final_cum_eval": int(result["x"][-1]) if result["x"] else None,
                "final_cum_original_cost": float(result["x_original_cost"][-1])
                if result["x_original_cost"]
                else None,
                "num_batches": len(result["regret"]),
                "lookup_table_s": result["lookup_table_s"],
                "total_wall_time_s": total_wall_time_s,
                "prior_mean_resolved": float(prior_mean),
                "prior_variance_resolved": float(prior_variance),
                "prior_bucket": prior_bucket,
                "prior_source": prior_source,
                "mmlu_task": mmlu_task,
                "mmlu_size_bucket": size_bucket,
                "matrix_seed": matrix_seed,
                "run_seed": int(args.run_seed),
                "experiment_variant": variant.raw,
                "gittins_stop_cum_eval": result["natural_stop_cum_eval"],
                "gittins_stop_cum_original_cost": result["natural_stop_cum_original_cost"],
                "gittins_recommendation_aware_stop_cum_eval": result["recommendation_aware_stop_cum_eval"],
                "gittins_recommendation_aware_stop_cum_original_cost": result[
                    "recommendation_aware_stop_cum_original_cost"
                ],
            }
        )
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
