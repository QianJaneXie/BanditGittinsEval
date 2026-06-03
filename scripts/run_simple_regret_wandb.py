#!/usr/bin/env python3
"""W&B runner for matrix-bandit simple-regret experiments.

One W&B run = one concrete experiment configuration.

Supported experiment variants:
- ucb_B{B} / ucb_cost_B{B}  (unit-cost vs 10% eval budget / cost budget vs 10% total cost)
- lrf_B{B} / lrf_cost_B{B}
- gittins_unit_B{B}_scale{S}_{default|dataset}
- gittins_cost_B{B}_scale{S}_{default|dataset}

This runner is designed for the current lightweight code path:
- UCB / LRF are run directly from banditeval.bandits.
- Gittins is run through src/gittins_policy.py with a precomputed root lookup table.
- Timing is recorded for Gittins lookup table construction and every policy iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
try:
    import resource
except Exception:  # pragma: no cover
    resource = None


def _load_psutil():
    try:
        return importlib.import_module("psutil")
    except Exception:  # pragma: no cover
        return None


psutil = _load_psutil()

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


# MMLU dataset priors are resolved through task_metadata.json buckets.
# default prior remains N(0.5, 0.04); dataset prior uses these bucket-level values.
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
    """Parse compact variant labels into concrete runner parameters."""
    v = raw.strip()

    m = re.fullmatch(r"(ucb|lrf)_cost_B(\d+)", v)
    if m:
        policy = m.group(1)
        b = int(m.group(2))
        return VariantConfig(
            raw=v,
            policy_variant=f"{policy}_cost",
            policy_family=policy,
            cost_mode="cost",
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=1e-4,
            prior_type="default",
        )

    # Backward-compatible alias for earlier sweep YAMLs.
    m = re.fullmatch(r"(ucb|lrf)_aware_B(\d+)", v)
    if m:
        policy = m.group(1)
        b = int(m.group(2))
        return VariantConfig(
            raw=v,
            policy_variant=f"{policy}_cost",
            policy_family=policy,
            cost_mode="cost",
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=1e-4,
            prior_type="default",
        )

    m = re.fullmatch(r"(ucb|lrf)_B(\d+)", v)
    if m:
        policy = m.group(1)
        b = int(m.group(2))
        return VariantConfig(
            raw=v,
            policy_variant=policy,
            policy_family=policy,
            cost_mode="unit",
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=1e-4,
            prior_type="default",
        )

    m = re.fullmatch(
        r"gittins_(unit|cost|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)",
        v,
    )
    if m:
        cost_mode = m.group(1)
        if cost_mode == "aware":
            cost_mode = "cost"
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
    if v in {"ucb", "lrf", "gittins_unit", "gittins_cost", "gittins_aware"}:
        if v == "ucb":
            return VariantConfig(v, "ucb", "ucb", "unit", 20, 20, 1e-4, "default")
        if v == "lrf":
            return VariantConfig(v, "lrf", "lrf", "unit", 20, 20, 1e-4, "default")
        cost_mode = "unit" if v == "gittins_unit" else "cost"
        policy_variant = "gittins_unit" if v == "gittins_unit" else "gittins_cost"
        return VariantConfig(v, policy_variant, "gittins", cost_mode, 20, 20, 1e-4, "default")

    raise ValueError(
        f"Unsupported experiment variant: {raw!r}. Examples: "
        "ucb_B20, ucb_cost_B20, lrf_B20, lrf_cost_B20, "
        "gittins_unit_B20_scale1e-4_default, gittins_cost_B20_scale1e-4_dataset"
    )


@dataclass(frozen=True)
class ExperimentBudget:
    budget_mode: str
    max_evaluations: int
    max_original_cost: float | None
    budget_evals: int
    budget_original_cost: float | None
    total_brute_force_original_cost: float


def resolve_experiment_budget(
    *,
    n_cells: int,
    n_examples: int,
    cost_per_arm: torch.Tensor,
    eval_budget_fraction: float,
    cost_mode: str,
) -> ExperimentBudget:
    total_brute_force_original_cost = float(n_examples * cost_per_arm.sum().item())
    budget_evals = int(max(1, round(float(eval_budget_fraction) * int(n_cells))))
    if cost_mode in {"cost", "aware"}:
        budget_original_cost = float(eval_budget_fraction) * total_brute_force_original_cost
        return ExperimentBudget(
            budget_mode="cost",
            max_evaluations=int(n_cells),
            max_original_cost=budget_original_cost,
            budget_evals=budget_evals,
            budget_original_cost=budget_original_cost,
            total_brute_force_original_cost=total_brute_force_original_cost,
        )
    return ExperimentBudget(
        budget_mode="evals",
        max_evaluations=budget_evals,
        max_original_cost=None,
        budget_evals=budget_evals,
        budget_original_cost=None,
        total_brute_force_original_cost=total_brute_force_original_cost,
    )


def infer_matrix_seed(matrix: Path) -> str | None:
    m = re.search(r"seed(\d+)", matrix.stem)
    return m.group(1) if m else None


def infer_mmlu_task(matrix: Path) -> str | None:
    """Infer MMLU subject name from matrix filename.

    Current MMLU matrices are expected as data/MMLU_matrices/<subject>.npy.
    For synthetic or suffixed files, keep the stem before "_synthetic_" as the task name.
    """
    stem = matrix.stem
    synthetic_marker = "_synthetic_"
    if synthetic_marker in stem:
        return stem.split(synthetic_marker, 1)[0]
    return stem if stem else None


def load_mmlu_task_prior_buckets(path: Path | None) -> dict[str, str]:
    """Load MMLU task -> dataset prior bucket mapping from task metadata."""
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: dict[str, str] = {}
    for row in data.get("tasks", []):
        if not isinstance(row, dict):
            continue
        task = str(row.get("task", "")).strip()
        bucket = str(row.get("dataset_prior_bucket", "")).strip().lower()
        if task and bucket in MMLU_PRIOR_BY_BUCKET:
            out[task] = bucket
    return out


def mmlu_size_bucket(n_examples: int) -> str:
    if int(n_examples) <= 150:
        return "small"
    if int(n_examples) <= 400:
        return "medium"
    return "large"


def resolve_prior(
    *,
    dataset_tag: str,
    prior_type: str,
    matrix: Path,
    mmlu_task_prior_buckets: dict[str, str],
) -> tuple[float, float, str | None, str | None, str]:
    """Return (prior_mean, prior_variance, mmlu_task, prior_bucket, prior_source)."""
    task = infer_mmlu_task(matrix) if dataset_tag == "mmlu" else None

    if prior_type == "default":
        return DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_VARIANCE, task, None, "default"

    if prior_type != "dataset":
        raise ValueError(f"Unsupported prior_type: {prior_type}")

    if dataset_tag == "mmlu":
        if task is None:
            return DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_VARIANCE, None, None, "mmlu_bucket_fallback_default"
        bucket = mmlu_task_prior_buckets.get(task)
        if bucket is not None and bucket in MMLU_PRIOR_BY_BUCKET:
            mean, variance = MMLU_PRIOR_BY_BUCKET[bucket]
            return float(mean), float(variance), task, bucket, "mmlu_task_bucket"
        return DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_VARIANCE, task, None, "mmlu_bucket_fallback_default"

    if dataset_tag not in DATASET_PRIORS:
        raise ValueError(
            f"No dataset prior is defined for dataset_tag={dataset_tag!r}. "
            "Use *_default or add a DATASET_PRIORS entry."
        )
    mean, variance = DATASET_PRIORS[dataset_tag]
    return float(mean), float(variance), None, dataset_tag, "dataset_global"


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


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _run_git_cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    git_status = _run_git_cmd(["status", "--porcelain"])
    git_dirty: bool | None
    if git_status is None:
        git_dirty = None
    else:
        git_dirty = bool(git_status)

    return {
        "git_commit": _run_git_cmd(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty,
        "matrix_sha256": sha256_file(args.matrix),
        "cost_vector_sha256": sha256_file(args.cost_vector) if args.cost_vector is not None else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
    }


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


def _ru_maxrss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = float(ru.ru_maxrss)
        if sys.platform == "darwin":
            return int(maxrss)
        return int(maxrss * 1024.0)
    except Exception:
        return None


def current_rss_bytes() -> int | None:
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    return None


def peak_rss_bytes() -> int | None:
    ru_peak = _ru_maxrss_bytes()
    if ru_peak is not None:
        return ru_peak
    if psutil is not None:
        try:
            mem = psutil.Process(os.getpid()).memory_info()
            if hasattr(mem, "peak_wset"):
                return int(getattr(mem, "peak_wset"))
            return int(mem.rss)
        except Exception:
            pass
    return None


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
    natural_stop_cum_eval_holder: list[int | None] | None = None,
    recommendation_aware_stop_cum_eval_holder: list[int | None] | None = None,
    max_original_cost: float | None = None,
    post_pull_fn: Callable[[torch.Tensor, int, int], dict[str, Any] | None] | None = None,
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
    # Pull diagnostics: record which arm(s) the policy actually evaluates each step.
    # `pulled_arm` is the primary arm used by the existing cost accounting;
    # `pulled_arms_json` preserves the full set in case a future policy returns a mixed-arm batch.
    pulled_arm_history: list[int] = []
    pulled_arms_json: list[str] = []
    pulled_rows_json: list[str] = []
    pulled_cols_json: list[str] = []
    n_pulled_arms: list[int] = []
    n_pulled_cols: list[int] = []
    # Gittins-only diagnostics (full vectors in trace npz; scalars also logged to W&B).
    gittins_index_pulled: list[float] = []
    posterior_mean_pulled: list[float] = []
    gittins_scores_post_pull: list[np.ndarray] = []
    posterior_mean_post_pull: list[np.ndarray] = []

    evaluated = 0
    total_cost = 0.0
    step_idx = 0
    natural_stop_cum_original_cost: float | None = None
    recommendation_aware_stop_cum_original_cost: float | None = None

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

        pulled_arm = int(row_idx[0].item())

        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch
        total_cost += float(original_cost_per_arm[pulled_arm].item()) * float(n_batch)

        gittins_diag: dict[str, Any] | None = None
        if post_pull_fn is not None:
            gittins_diag = post_pull_fn(obs, pulled_arm, int(evaluated))
        if (
            natural_stop_cum_original_cost is None
            and natural_stop_cum_eval_holder is not None
            and len(natural_stop_cum_eval_holder) == 1
            and natural_stop_cum_eval_holder[0] is not None
            and int(natural_stop_cum_eval_holder[0]) == int(evaluated)
        ):
            natural_stop_cum_original_cost = float(total_cost)
        if (
            recommendation_aware_stop_cum_original_cost is None
            and recommendation_aware_stop_cum_eval_holder is not None
            and len(recommendation_aware_stop_cum_eval_holder) == 1
            and recommendation_aware_stop_cum_eval_holder[0] is not None
            and int(recommendation_aware_stop_cum_eval_holder[0]) == int(evaluated)
        ):
            recommendation_aware_stop_cum_original_cost = float(total_cost)

        if gittins_diag is not None:
            scores_post = np.asarray(gittins_diag["gittins_scores_post_pull"], dtype=np.float32)
            mus_post = np.asarray(gittins_diag["posterior_mean_post_pull"], dtype=np.float32)
            gittins_scores_post_pull.append(scores_post)
            posterior_mean_post_pull.append(mus_post)
            gittins_index_pulled.append(float(scores_post[pulled_arm]))
            posterior_mean_pulled.append(float(mus_post[pulled_arm]))
        pulled_rows = [int(x) for x in row_idx.reshape(-1).tolist()]
        pulled_cols = [int(x) for x in col_idx.reshape(-1).tolist()]
        unique_pulled_arms = sorted(set(pulled_rows))
        unique_pulled_cols = sorted(set(pulled_cols))

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
        pulled_arm_history.append(int(pulled_arm))
        pulled_arms_json.append(json.dumps(unique_pulled_arms, separators=(",", ":")))
        pulled_rows_json.append(json.dumps(pulled_rows, separators=(",", ":")))
        pulled_cols_json.append(json.dumps(pulled_cols, separators=(",", ":")))
        n_pulled_arms.append(int(len(unique_pulled_arms)))
        n_pulled_cols.append(int(len(unique_pulled_cols)))

        if run is not None and log_step_metrics:
            log_payload: dict[str, Any] = {
                    "cum_eval": int(evaluated),
                    "cum_original_cost": float(total_cost),
                    "simple_regret": float(simple_regret),
                    "recommended_arm": int(arm),
                    "recommended_mean": recommended_mean[-1],
                    "pulled_arm": int(pulled_arm),
                    "pulled_arms_json": pulled_arms_json[-1],
                    "pulled_rows_json": pulled_rows_json[-1],
                    "pulled_cols_json": pulled_cols_json[-1],
                    "n_pulled_arms": n_pulled_arms[-1],
                    "n_pulled_cols": n_pulled_cols[-1],
                    "iter_step_s": step_time,
                    "iter_total_s": total_time,
                    "batch_cells": n_batch,
                    "step_idx": step_idx,
            }
            if gittins_diag is not None:
                log_payload["gittins_index_pulled"] = float(gittins_index_pulled[-1])
                log_payload["posterior_mean_pulled"] = float(posterior_mean_pulled[-1])
            run.log(log_payload)

        if max_original_cost is not None and total_cost >= float(max_original_cost):
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
        "pulled_arm": pulled_arm_history,
        "pulled_arms_json": pulled_arms_json,
        "pulled_rows_json": pulled_rows_json,
        "pulled_cols_json": pulled_cols_json,
        "n_pulled_arms": n_pulled_arms,
        "n_pulled_cols": n_pulled_cols,
        "natural_stop_cum_original_cost": natural_stop_cum_original_cost,
        "recommendation_aware_stop_cum_original_cost": recommendation_aware_stop_cum_original_cost,
        "gittins_index_pulled": gittins_index_pulled,
        "posterior_mean_pulled": posterior_mean_pulled,
        "gittins_scores_post_pull": gittins_scores_post_pull,
        "posterior_mean_post_pull": posterior_mean_post_pull,
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
    p.add_argument(
        "--mmlu-task-metadata",
        "--mmlu_task_metadata",
        type=Path,
        default=DEFAULT_MMLU_TASK_METADATA,
        help="MMLU task metadata JSON with dataset_prior_bucket for subject-level dataset priors.",
    )

    p.add_argument("--out-dir", "--out_dir", type=Path, default=Path("outputs") / "wandb_simple_regret")
    p.add_argument(
        "--log-step-metrics",
        "--log_step_metrics",
        dest="log_step_metrics",
        action="store_true",
        default=True,
        help="Enable per-step W&B history logging (default: enabled).",
    )
    p.add_argument(
        "--no-log-step-metrics",
        "--no_log_step_metrics",
        dest="log_step_metrics",
        action="store_false",
        help="Disable per-step W&B history logging.",
    )
    p.add_argument("--wandb-entity", "--wandb_entity", default=None)
    p.add_argument("--wandb-project", "--wandb_project", default="GittinsBanditEval")
    p.add_argument("--wandb-group", "--wandb_group", default=None)
    p.add_argument("--wandb-name", "--wandb_name", default=None)
    p.add_argument("--wandb-mode", "--wandb_mode", choices=["online", "offline", "disabled"], default="online")

    return p.parse_args()


def main() -> int:
    run_wall_t0 = time.perf_counter()
    args = parse_args()
    provenance = collect_provenance(args)
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

    if args.cost_vector is not None:
        actual_cost_per_arm = load_cost_vector(args.cost_vector, n_arms)
    else:
        actual_cost_per_arm = torch.ones((n_arms,), dtype=torch.float64)

    budget = resolve_experiment_budget(
        n_cells=n_cells,
        n_examples=n_examples,
        cost_per_arm=actual_cost_per_arm,
        eval_budget_fraction=float(args.eval_budget_fraction),
        cost_mode=variant.cost_mode,
    )
    max_evaluations = int(budget.max_evaluations)
    max_original_cost = budget.max_original_cost
    cost_aware_run = variant.cost_mode in {"cost", "aware"}

    if cost_aware_run and args.cost_vector is None:
        print(
            f"{variant.raw} requires --cost-vector for cost-aware budgeting.",
            file=sys.stderr,
        )
        return 1

    size_bucket = mmlu_size_bucket(n_examples) if dataset_tag == "mmlu" else None
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
        prior_source = f"{prior_source}+manual_mean_override"
    if args.gittins_prior_variance is not None:
        prior_variance = float(args.gittins_prior_variance)
        prior_source = f"{prior_source}+manual_variance_override"

    if dataset_tag == "mmlu" and mmlu_task is not None:
        matrix_seed_label = f"task{safe_token(mmlu_task)}"
    else:
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
                **provenance,
                "dataset_tag_resolved": dataset_tag,
                "matrix_seed": matrix_seed,
                "n_arms": n_arms,
                "n_examples": n_examples,
                "n_cells": n_cells,
                "budget_mode": budget.budget_mode,
                "budget_max_evals": budget.budget_evals,
                "budget_original_cost": budget.budget_original_cost,
                "total_brute_force_original_cost": budget.total_brute_force_original_cost,
                "cost_aware_run": cost_aware_run,
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
        run.define_metric("iter_step_s", step_metric="cum_eval")
        run.define_metric("iter_total_s", step_metric="cum_eval")
        run.define_metric("pulled_arm", step_metric="cum_eval")
        run.define_metric("n_pulled_arms", step_metric="cum_eval")
        run.define_metric("n_pulled_cols", step_metric="cum_eval")
        run.define_metric("gittins_index_pulled", step_metric="cum_eval")
        run.define_metric("posterior_mean_pulled", step_metric="cum_eval")

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
    lookup_memory_rss_before_mb: float | None = None
    lookup_memory_rss_after_mb: float | None = None
    lookup_memory_rss_delta_mb: float | None = None
    lookup_memory_peak_before_mb: float | None = None
    lookup_memory_peak_after_mb: float | None = None
    lookup_memory_peak_delta_mb: float | None = None
    peak_rss_gb: float | None = None
    extra_peak_memory_gb: float | None = None
    natural_stop_holder: list[int | None] = [None]
    recommendation_aware_stop_holder: list[int | None] = [None]
    post_pull_fn: Callable[[torch.Tensor, int, int], dict[str, Any] | None] | None = None

    if variant.policy_family == "ucb":
        def step_fn(obs: torch.Tensor, sim_cum_eval: int):
            return upper_confidence_bound_exploration(
                obs,
                a=float(args.ucb_a),
                batch_size=int(variant.batch_size),
                return_mus=False,
            )

        recommend_fn = empirical_incumbent

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

        recommend_fn = empirical_incumbent

    elif variant.policy_family == "gittins":
        B = int(variant.gittins_batch_size)
        tau_sq_batch = (
            float(args.gittins_obs_noise_variance)
            if args.gittins_obs_noise_variance is not None
            else 1.0 / (4.0 * float(B))
        )
        tau_sq_cell = float(tau_sq_batch) * float(B)

        if variant.cost_mode in {"cost", "aware"}:
            if args.cost_vector is None:
                print("gittins_cost requires --cost-vector", file=sys.stderr)
                return 1
            decision_cost_per_arm = actual_cost_per_arm.clone()
        elif variant.cost_mode == "unit":
            decision_cost_per_arm = torch.ones((n_arms,), dtype=torch.float64)
        else:
            raise ValueError(f"Unsupported Gittins cost_mode: {variant.cost_mode}")

        rss_before_b = current_rss_bytes()
        peak_before_b = peak_rss_bytes()
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
        rss_after_b = current_rss_bytes()
        peak_after_b = peak_rss_bytes()

        if rss_before_b is not None:
            lookup_memory_rss_before_mb = float(rss_before_b) / (1024.0**2)
        if rss_after_b is not None:
            lookup_memory_rss_after_mb = float(rss_after_b) / (1024.0**2)
        if rss_before_b is not None and rss_after_b is not None:
            lookup_memory_rss_delta_mb = max(0.0, float(rss_after_b - rss_before_b) / (1024.0**2))

        if peak_before_b is not None:
            lookup_memory_peak_before_mb = float(peak_before_b) / (1024.0**2)
        if peak_after_b is not None:
            lookup_memory_peak_after_mb = float(peak_after_b) / (1024.0**2)
        if peak_before_b is not None and peak_after_b is not None:
            lookup_memory_peak_delta_mb = max(0.0, float(peak_after_b - peak_before_b) / (1024.0**2))

        if peak_after_b is not None:
            peak_rss_gb = float(peak_after_b) / (1024.0**3)
        if peak_before_b is not None and peak_after_b is not None:
            extra_peak_memory_gb = max(0.0, float(peak_after_b - peak_before_b) / (1024.0**3))

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

        def post_pull_fn(obs: torch.Tensor, pulled_arm: int, cum_eval: int) -> dict[str, Any]:
            mus_post, scores_post = gittins_post_pull_update(
                obs,
                cached_scores=cached_scores,
                recompute_arms=[int(pulled_arm)],
                sim_cum_eval=int(cum_eval),
                **gittins_post_pull_kw,
            )
            return {
                "gittins_scores_post_pull": scores_post.detach().cpu().numpy().copy(),
                "posterior_mean_post_pull": mus_post.detach().cpu().numpy().copy(),
            }

        recommend_fn = partial(
            posterior_incumbent,
            prior_mean=float(prior_mean),
            prior_variance=float(prior_variance),
            tau_sq_cell=float(tau_sq_cell),
        )

    else:
        raise ValueError(f"Unsupported policy family: {variant.policy_family}")

    sim = simulate_timed(
        ground_truth=ground_truth,
        step_fn=step_fn,
        seed=int(args.run_seed),
        max_evaluations=max_evaluations,
        original_cost_per_arm=actual_cost_per_arm,
        recommend_fn=recommend_fn,
        run=run,
        log_step_metrics=bool(args.log_step_metrics),
        natural_stop_cum_eval_holder=natural_stop_holder,
        recommendation_aware_stop_cum_eval_holder=recommendation_aware_stop_holder,
        max_original_cost=max_original_cost,
        post_pull_fn=post_pull_fn,
    )

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
        cost_aware=np.asarray(bool(cost_aware_run), dtype=np.bool_),
        budget_mode=np.asarray(budget.budget_mode, dtype="<U16"),
        budget_evals=np.asarray(int(budget.budget_evals), dtype=np.int32),
        budget_original_cost=np.asarray(
            -1.0 if budget.budget_original_cost is None else float(budget.budget_original_cost),
            dtype=np.float64,
        ),
        total_brute_force_original_cost=np.asarray(
            float(budget.total_brute_force_original_cost), dtype=np.float64
        ),
        sim_max_evaluations=np.asarray(int(max_evaluations), dtype=np.int32),
        prior_mean=np.asarray(prior_mean, dtype=np.float32),
        prior_variance=np.asarray(prior_variance, dtype=np.float32),
        prior_bucket="" if prior_bucket is None else prior_bucket,
        prior_source=prior_source,
        mmlu_task="" if mmlu_task is None else mmlu_task,
        mmlu_size_bucket="" if size_bucket is None else size_bucket,
        lookup_table_s=np.asarray(-1.0 if lookup_table_s is None else lookup_table_s, dtype=np.float64),
        lookup_memory_rss_before_mb=np.asarray(
            -1.0 if lookup_memory_rss_before_mb is None else lookup_memory_rss_before_mb, dtype=np.float64
        ),
        lookup_memory_rss_after_mb=np.asarray(
            -1.0 if lookup_memory_rss_after_mb is None else lookup_memory_rss_after_mb, dtype=np.float64
        ),
        lookup_memory_rss_delta_mb=np.asarray(
            -1.0 if lookup_memory_rss_delta_mb is None else lookup_memory_rss_delta_mb, dtype=np.float64
        ),
        lookup_memory_peak_before_mb=np.asarray(
            -1.0 if lookup_memory_peak_before_mb is None else lookup_memory_peak_before_mb, dtype=np.float64
        ),
        lookup_memory_peak_after_mb=np.asarray(
            -1.0 if lookup_memory_peak_after_mb is None else lookup_memory_peak_after_mb, dtype=np.float64
        ),
        lookup_memory_peak_delta_mb=np.asarray(
            -1.0 if lookup_memory_peak_delta_mb is None else lookup_memory_peak_delta_mb, dtype=np.float64
        ),
        peak_rss_gb=np.asarray(-1.0 if peak_rss_gb is None else peak_rss_gb, dtype=np.float64),
        extra_peak_memory_gb=np.asarray(
            -1.0 if extra_peak_memory_gb is None else extra_peak_memory_gb, dtype=np.float64
        ),
        x=np.asarray(sim["x"], dtype=np.int32),
        x_original_cost=np.asarray(sim["x_original_cost"], dtype=np.float64),
        regret=np.asarray(sim["regret"], dtype=np.float32),
        recommended_arm=np.asarray(sim["recommended_arm"], dtype=np.int32),
        recommended_mean=np.asarray(sim["recommended_mean"], dtype=np.float32),
        iter_step_s=np.asarray(sim["iter_step_s"], dtype=np.float64),
        iter_total_s=np.asarray(sim["iter_total_s"], dtype=np.float64),
        batch_cells=np.asarray(sim["batch_cells"], dtype=np.int32),
        pulled_arm=np.asarray(sim["pulled_arm"], dtype=np.int32),
        pulled_arms_json=np.asarray(sim["pulled_arms_json"], dtype="<U4096"),
        pulled_rows_json=np.asarray(sim["pulled_rows_json"], dtype="<U4096"),
        pulled_cols_json=np.asarray(sim["pulled_cols_json"], dtype="<U4096"),
        n_pulled_arms=np.asarray(sim["n_pulled_arms"], dtype=np.int32),
        n_pulled_cols=np.asarray(sim["n_pulled_cols"], dtype=np.int32),
        gittins_index_pulled=np.asarray(sim["gittins_index_pulled"], dtype=np.float32),
        posterior_mean_pulled=np.asarray(sim["posterior_mean_pulled"], dtype=np.float32),
        gittins_scores_post_pull=np.stack(sim["gittins_scores_post_pull"], axis=0)
        if sim["gittins_scores_post_pull"]
        else np.zeros((0, n_arms), dtype=np.float32),
        posterior_mean_post_pull=np.stack(sim["posterior_mean_post_pull"], axis=0)
        if sim["posterior_mean_post_pull"]
        else np.zeros((0, n_arms), dtype=np.float32),
        git_commit=np.asarray(provenance["git_commit"] or "", dtype="<U64"),
        git_dirty=np.asarray(False if provenance["git_dirty"] is None else provenance["git_dirty"], dtype=np.bool_),
        matrix_sha256=np.asarray(provenance["matrix_sha256"] or "", dtype="<U64"),
        cost_vector_sha256=np.asarray(provenance["cost_vector_sha256"] or "", dtype="<U64"),
        slurm_job_id=np.asarray(provenance["slurm_job_id"] or "", dtype="<U64"),
        hostname=np.asarray(provenance["hostname"] or "", dtype="<U256"),
        python_version=np.asarray(provenance["python_version"] or "", dtype="<U1024"),
        cost_per_arm_original=np.asarray(actual_cost_per_arm.numpy(), dtype=np.float64),
        gittins_stop_cum_eval=np.asarray(
            -1 if natural_stop_holder[0] is None else int(natural_stop_holder[0]),
            dtype=np.int32,
        ),
        gittins_stop_cum_original_cost=np.asarray(
            -1.0
            if sim["natural_stop_cum_original_cost"] is None
            else float(sim["natural_stop_cum_original_cost"]),
            dtype=np.float64,
        ),
        gittins_recommendation_aware_stop_cum_eval=np.asarray(
            -1
            if recommendation_aware_stop_holder[0] is None
            else int(recommendation_aware_stop_holder[0]),
            dtype=np.int32,
        ),
        gittins_recommendation_aware_stop_cum_original_cost=np.asarray(
            -1.0
            if sim["recommendation_aware_stop_cum_original_cost"] is None
            else float(sim["recommendation_aware_stop_cum_original_cost"]),
            dtype=np.float64,
        ),
    )

    step_summary = timing_summary(sim["iter_step_s"])
    total_summary = timing_summary(sim["iter_total_s"])
    total_wall_time_s = float(time.perf_counter() - run_wall_t0)
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
        "budget_mode": budget.budget_mode,
        "budget_max_evals": budget.budget_evals,
        "budget_original_cost": budget.budget_original_cost,
        "total_brute_force_original_cost": budget.total_brute_force_original_cost,
        "cost_aware_run": cost_aware_run,
        "sim_max_evaluations": max_evaluations,
        "eval_budget_fraction": float(args.eval_budget_fraction),
        "prior_mean_resolved": float(prior_mean),
        "prior_variance_resolved": float(prior_variance),
        "prior_bucket": prior_bucket,
        "prior_source": prior_source,
        "mmlu_task": mmlu_task,
        "mmlu_size_bucket": size_bucket,
        "provenance": provenance,
        "cost_vector": str(args.cost_vector) if args.cost_vector else None,
        "trace": str(trace_path),
        "figure_eval": str(fig_eval_path),
        "figure_cost": str(fig_cost_path),
        "timing": {
            "lookup_table_s": lookup_table_s,
            "lookup_memory_rss_before_mb": lookup_memory_rss_before_mb,
            "lookup_memory_rss_after_mb": lookup_memory_rss_after_mb,
            "lookup_memory_rss_delta_mb": lookup_memory_rss_delta_mb,
            "lookup_memory_peak_before_mb": lookup_memory_peak_before_mb,
            "lookup_memory_peak_after_mb": lookup_memory_peak_after_mb,
            "lookup_memory_peak_delta_mb": lookup_memory_peak_delta_mb,
            "peak_rss_gb": peak_rss_gb,
            "extra_peak_memory_gb": extra_peak_memory_gb,
            "iter_step_s": step_summary,
            "iter_total_s": total_summary,
            "total_wall_time_s": total_wall_time_s,
        },
        "final": {
            "final_simple_regret": float(sim["regret"][-1]) if sim["regret"] else None,
            "best_seen_regret": float(min(sim["regret"])) if sim["regret"] else None,
            "final_cum_eval": int(sim["x"][-1]) if sim["x"] else None,
            "final_cum_original_cost": float(sim["x_original_cost"][-1]) if sim["x_original_cost"] else None,
            "num_batches": len(sim["regret"]),
            "num_distinct_pulled_arms": int(len(set(sim["pulled_arm"]))) if sim["pulled_arm"] else 0,
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
    print(f"peak_rss_gb={peak_rss_gb}")
    print(f"extra_peak_memory_gb={extra_peak_memory_gb}")
    print(f"total_wall_time_s={total_wall_time_s}")
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
                "lookup_memory_rss_before_mb": lookup_memory_rss_before_mb,
                "lookup_memory_rss_after_mb": lookup_memory_rss_after_mb,
                "lookup_memory_rss_delta_mb": lookup_memory_rss_delta_mb,
                "lookup_memory_peak_before_mb": lookup_memory_peak_before_mb,
                "lookup_memory_peak_after_mb": lookup_memory_peak_after_mb,
                "lookup_memory_peak_delta_mb": lookup_memory_peak_delta_mb,
                "peak_rss_gb": peak_rss_gb,
                "extra_peak_memory_gb": extra_peak_memory_gb,
                "total_wall_time_s": total_wall_time_s,
                "iter_step_mean_s": step_summary["mean_s"],
                "iter_step_median_s": step_summary["median_s"],
                "iter_step_p90_s": step_summary["p90_s"],
                "iter_total_mean_s": total_summary["mean_s"],
                "iter_total_median_s": total_summary["median_s"],
                "iter_total_p90_s": total_summary["p90_s"],
                "prior_mean_resolved": float(prior_mean),
                "prior_variance_resolved": float(prior_variance),
                "prior_bucket": prior_bucket,
                "prior_source": prior_source,
                "mmlu_task": mmlu_task,
                "mmlu_size_bucket": size_bucket,
                "matrix_seed": matrix_seed,
                "run_seed": int(args.run_seed),
                "experiment_variant": variant.raw,
                "git_commit": provenance["git_commit"],
                "git_dirty": provenance["git_dirty"],
                "matrix_sha256": provenance["matrix_sha256"],
                "cost_vector_sha256": provenance["cost_vector_sha256"],
                "slurm_job_id": provenance["slurm_job_id"],
                "slurm_array_job_id": provenance["slurm_array_job_id"],
                "slurm_array_task_id": provenance["slurm_array_task_id"],
                "hostname": provenance["hostname"],
                "python_version": provenance["python_version"],
                "gittins_stop_cum_eval": (
                    None if natural_stop_holder[0] is None else int(natural_stop_holder[0])
                ),
                "gittins_stop_cum_original_cost": (
                    None
                    if sim["natural_stop_cum_original_cost"] is None
                    else float(sim["natural_stop_cum_original_cost"])
                ),
                "gittins_recommendation_aware_stop_cum_eval": (
                    None
                    if recommendation_aware_stop_holder[0] is None
                    else int(recommendation_aware_stop_holder[0])
                ),
                "gittins_recommendation_aware_stop_cum_original_cost": (
                    None
                    if sim["recommendation_aware_stop_cum_original_cost"] is None
                    else float(sim["recommendation_aware_stop_cum_original_cost"])
                ),
            }
        )

        artifact = wandb.Artifact(
            name=f"simple-regret-{run.id}",
            type="experiment_outputs",
            description=f"Outputs for {variant.raw}",
        )
        artifact.add_file(str(trace_path))
        artifact.add_file(str(meta_path))
        run.log_artifact(artifact)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
