#!/usr/bin/env python3
"""Run discrete BO baselines with optional W&B logging.

The input is a ``*_bo_inputs.npz`` file from ``convert_matrix_to_bo_inputs.py``.
Rows are complete configurations, not matrix cells: evaluating one candidate reveals
its aggregate score ``Y`` and consumes the full-evaluation ``cost`` stored in the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from stable_pbgi import StableGittinsIndex  # noqa: E402


@dataclass(frozen=True)
class BoData:
    path: Path
    X: torch.Tensor
    Y: torch.Tensor
    cost: torch.Tensor
    arm_ids: np.ndarray
    dataset: str
    matrix_seed: str
    n_examples: int
    cat_dims: list[int]
    default_n_init: int
    metadata: dict[str, Any]


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


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
            cwd=_repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    git_status = _run_git_cmd(["status", "--porcelain"])
    if git_status is None:
        git_dirty: bool | None = None
    else:
        git_dirty = bool(git_status)
    return {
        "git_commit": _run_git_cmd(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty,
        "bo_inputs_sha256": sha256_file(args.bo_inputs),
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
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p90_s": float(np.percentile(arr, 90)),
        "p99_s": float(np.percentile(arr, 99)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bo-inputs", "--bo_inputs", dest="bo_inputs", type=Path, required=True)
    p.add_argument(
        "--acquisition",
        choices=["pbgi"],
        default="pbgi",
        help="Acquisition used after the random initialization design.",
    )
    p.add_argument("--seed", "--run-seed", "--run_seed", dest="seed", type=int, default=0)
    p.add_argument(
        "--n-init",
        "--n_init",
        type=int,
        default=None,
        help="Number of random initial configurations. Defaults to dim + 1.",
    )
    p.add_argument(
        "--n-steps",
        "--n_steps",
        type=int,
        default=20,
        help="Number of BO-selected configurations after initialization.",
    )
    p.add_argument(
        "--cost-aware",
        "--cost_aware",
        action="store_true",
        help="Use costs in acquisition ranking. For PBGI this passes cost_X to the acquisition.",
    )
    p.add_argument(
        "--cost-scaling-factor",
        "--cost_scaling_factor",
        type=float,
        default=1e-4,
        help="Cost scale used by PBGI, matching the bandit Gittins default.",
    )
    p.add_argument("--observation-noise", "--observation_noise", type=float, default=1e-6)
    p.add_argument(
        "--extend-to-natural-stop",
        "--extend_to_natural_stop",
        action="store_true",
        help=(
            "Run past the nominal BO budget until the PBGI natural stop is observed, "
            "capped by evaluating all configurations."
        ),
    )
    p.add_argument("--out-dir", "--out_dir", type=Path, default=Path("outputs") / "bo_baselines")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")

    p.add_argument("--wandb-entity", "--wandb_entity", default=None)
    p.add_argument("--wandb-project", "--wandb_project", default="GittinsBanditEval")
    p.add_argument("--wandb-group", "--wandb_group", default="bo_baseline_sweep")
    p.add_argument("--wandb-name", "--wandb_name", default=None)
    p.add_argument("--wandb-mode", "--wandb_mode", choices=["online", "offline", "disabled"], default="online")
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
    return p.parse_args()


def load_bo_inputs(path: Path, *, dtype: torch.dtype) -> BoData:
    if not path.is_file():
        raise FileNotFoundError(str(path))

    z = np.load(path, allow_pickle=False)
    required = {"X", "Y", "cost", "arm_ids"}
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")

    X_np = np.asarray(z["X"], dtype=np.float64)
    Y_np = np.asarray(z["Y"], dtype=np.float64).reshape(-1, 1)
    cost_np = np.asarray(z["cost"], dtype=np.float64).reshape(-1)

    if X_np.ndim != 2 or X_np.shape[1] <= 0:
        raise ValueError(f"Expected X with shape (n_configs, n_features), got {X_np.shape}")
    if Y_np.shape != (X_np.shape[0], 1):
        raise ValueError(f"Expected Y with shape ({X_np.shape[0]}, 1), got {Y_np.shape}")
    if cost_np.shape != (X_np.shape[0],):
        raise ValueError(f"Expected cost with shape ({X_np.shape[0]},), got {cost_np.shape}")
    if np.any(cost_np <= 0):
        raise ValueError("All costs must be positive for cost-aware acquisition ranking.")

    metadata: dict[str, Any] = {}
    if "metadata_json" in z.files:
        metadata = json.loads(str(np.asarray(z["metadata_json"]).reshape(())))

    dataset = str(np.asarray(z["dataset"]).reshape(())) if "dataset" in z.files else metadata.get("dataset", "")
    matrix_seed = (
        str(np.asarray(z["matrix_seed"]).reshape(()))
        if "matrix_seed" in z.files
        else str(metadata.get("matrix_seed", ""))
    )
    n_examples = int(metadata.get("n_examples", 1))
    cat_dims = list(range(int(X_np.shape[1])))

    return BoData(
        path=path,
        X=torch.tensor(X_np, dtype=dtype),
        Y=torch.tensor(Y_np, dtype=dtype),
        cost=torch.tensor(cost_np, dtype=dtype),
        arm_ids=np.asarray(z["arm_ids"], dtype=np.int64),
        dataset=dataset,
        matrix_seed=matrix_seed,
        n_examples=n_examples,
        cat_dims=cat_dims,
        default_n_init=len(cat_dims) + 1,
        metadata=metadata,
    )


def fit_mixed_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    *,
    observation_noise: float,
    cat_dims: list[int],
) -> MixedSingleTaskGP:
    train_Yvar = torch.full_like(train_Y, float(observation_noise))
    model = MixedSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=cat_dims,
        train_Yvar=train_Yvar,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def score_candidates(
    *,
    acquisition: str,
    model: MixedSingleTaskGP,
    candidate_X: torch.Tensor,
    candidate_cost: torch.Tensor,
    cost_aware: bool,
    cost_scaling_factor: float,
) -> torch.Tensor:
    Xq = candidate_X.unsqueeze(1)
    with torch.no_grad():
        if acquisition != "pbgi":  # pragma: no cover - argparse constrains this.
            raise ValueError(f"Unsupported acquisition: {acquisition}")
        acq = StableGittinsIndex(model, lmbda=float(cost_scaling_factor))
        if cost_aware:
            scores = acq(Xq, cost_X=candidate_cost)
        else:
            scores = acq(Xq)
    # Some PBGI paths may return a broadcast matrix (n_candidates, n_candidates);
    # normalize to one scalar per candidate before ranking.
    if scores.ndim == 2 and scores.shape[0] == scores.shape[1]:
        scores = torch.diagonal(scores, offset=0)
    elif scores.ndim > 1:
        scores = scores.reshape(scores.shape[0], -1).mean(dim=1)
    return scores.reshape(-1).detach()


def observed_incumbent(selected: list[int], Y: torch.Tensor) -> tuple[int, float]:
    selected_arr = torch.tensor(selected, dtype=torch.long)
    observed_y = Y[selected_arr, 0]
    best_pos = int(torch.argmax(observed_y).item())
    arm = int(selected_arr[best_pos].item())
    return arm, float(Y[arm, 0].item())


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


def compute_budget_counts(data: BoData, args: argparse.Namespace) -> tuple[int, int, int, int]:
    n_configs = int(data.X.shape[0])
    requested_n_init = data.default_n_init if args.n_init is None else int(args.n_init)
    n_init = min(max(requested_n_init, 1), n_configs)
    n_steps = min(max(int(args.n_steps), 0), n_configs - n_init)
    nominal_total_configs = min(n_configs, n_init + n_steps)
    return requested_n_init, n_init, n_steps, nominal_total_configs


def variant_name(args: argparse.Namespace) -> str:
    v = str(args.acquisition)
    if bool(args.cost_aware):
        v = f"{v}_cost_aware"
    return v


def run_bo(
    args: argparse.Namespace,
    data: BoData,
    *,
    run: wandb.sdk.wandb_run.Run | None,
    log_step_metrics: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    n_configs = int(data.X.shape[0])
    _, n_init, n_steps, nominal_total_configs = compute_budget_counts(data, args)

    init = rng.choice(n_configs, size=n_init, replace=False).astype(int).tolist()
    selected: list[int] = []
    remaining: set[int] = set(range(n_configs))

    x: list[int] = []
    x_original_cost: list[float] = []
    regret: list[float] = []
    recommended_arm: list[int] = []
    recommended_mean: list[float] = []
    selected_arm: list[int] = []
    observed_y: list[float] = []
    acquisition_value: list[float] = []
    selection_phase: list[str] = []
    iter_fit_s: list[float] = []
    iter_score_s: list[float] = []
    stop_cum_eval: int | None = None
    stop_cum_original_cost: float | None = None
    stop_index_value: float | None = None

    mu_star = float(data.Y[:, 0].max().item())
    total_cost = 0.0
    max_bo_steps = (n_configs - n_init) if bool(args.extend_to_natural_stop) else n_steps

    def record(arm: int, acq_value: float, phase: str, fit_s: float, score_s: float) -> None:
        nonlocal total_cost
        selected.append(int(arm))
        remaining.remove(int(arm))
        total_cost += float(data.cost[arm].item())

        rec_arm, rec_value = observed_incumbent(selected, data.Y)
        x.append(int(len(selected) * data.n_examples))
        x_original_cost.append(float(total_cost))
        regret.append(float(mu_star - float(data.Y[rec_arm, 0].item())))
        recommended_arm.append(int(rec_arm))
        recommended_mean.append(float(rec_value))
        selected_arm.append(int(arm))
        observed_y.append(float(data.Y[arm, 0].item()))
        acquisition_value.append(float(acq_value))
        selection_phase.append(phase)
        iter_fit_s.append(float(fit_s))
        iter_score_s.append(float(score_s))

        if run is not None and bool(log_step_metrics):
            step_idx = int(len(x) - 1)
            run.log(
                {
                    "cum_eval": x[-1],
                    "cum_original_cost": x_original_cost[-1],
                    "simple_regret": regret[-1],
                    "recommended_arm": recommended_arm[-1],
                    "recommended_mean": recommended_mean[-1],
                    "selected_arm": selected_arm[-1],
                    "observed_y": observed_y[-1],
                    "acquisition_value": acquisition_value[-1],
                    "selection_phase": selection_phase[-1],
                    "iter_fit_s": iter_fit_s[-1],
                    "iter_score_s": iter_score_s[-1],
                    "step_idx": step_idx,
                    "num_evaluated_configs": len(selected),
                }
            )

    for arm in init:
        record(int(arm), float("nan"), "random_init", 0.0, 0.0)

    for _ in range(max_bo_steps):
        if not remaining:
            break
        train_idx = torch.tensor(selected, dtype=torch.long)
        train_X = data.X[train_idx]
        train_Y = data.Y[train_idx]

        fit_t0 = time.perf_counter()
        model = fit_mixed_gp(
            train_X,
            train_Y,
            observation_noise=float(args.observation_noise),
            cat_dims=data.cat_dims,
        )
        fit_s = float(time.perf_counter() - fit_t0)

        remaining_idx = torch.tensor(sorted(remaining), dtype=torch.long)
        score_t0 = time.perf_counter()
        scores = score_candidates(
            acquisition=str(args.acquisition),
            model=model,
            candidate_X=data.X[remaining_idx],
            candidate_cost=data.cost[remaining_idx],
            cost_aware=bool(args.cost_aware),
            cost_scaling_factor=float(args.cost_scaling_factor),
        )
        score_s = float(time.perf_counter() - score_t0)

        if not torch.isfinite(scores).any():
            raise RuntimeError("No finite acquisition scores were produced.")
        best_pos = int(torch.argmax(scores).item())
        best_score = float(scores[best_pos].item())
        best_observed = float(train_Y.max().item())
        if stop_cum_eval is None and best_score < best_observed:
            stop_cum_eval = int(len(selected) * data.n_examples)
            stop_cum_original_cost = float(total_cost)
            stop_index_value = best_score
        if (
            bool(args.extend_to_natural_stop)
            and len(selected) >= nominal_total_configs
            and stop_cum_eval is not None
        ):
            break
        arm = int(remaining_idx[best_pos].item())
        record(arm, best_score, str(args.acquisition), fit_s, score_s)

    return {
        "x": x,
        "x_original_cost": x_original_cost,
        "regret": regret,
        "recommended_arm": recommended_arm,
        "recommended_mean": recommended_mean,
        "selected_arm": selected_arm,
        "observed_y": observed_y,
        "acquisition_value": acquisition_value,
        "selection_phase": selection_phase,
        "iter_fit_s": iter_fit_s,
        "iter_score_s": iter_score_s,
        "stop_cum_eval": stop_cum_eval,
        "stop_cum_original_cost": stop_cum_original_cost,
        "stop_index_value": stop_index_value,
        "nominal_total_configs": nominal_total_configs,
    }


def main() -> int:
    run_wall_t0 = time.perf_counter()
    args = parse_args()
    provenance = collect_provenance(args)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(int(args.seed))

    data = load_bo_inputs(args.bo_inputs, dtype=dtype)
    requested_n_init, n_init_value, _n_steps_effective, nominal_total_configs = compute_budget_counts(data, args)
    n_init_rule = "dim_plus_1_default" if args.n_init is None else "user_set"
    variant = variant_name(args)
    dataset = safe_token(data.dataset or "unknown")

    run_name = args.wandb_name or (
        f"{dataset}_seed{data.matrix_seed}_{variant}_runseed{args.seed}"
        f"_ninit{n_init_value}_nsteps{int(args.n_steps)}"
    )

    run: wandb.sdk.wandb_run.Run | None = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            job_type="bo_baseline",
            mode=args.wandb_mode,
            config={
                "bo_inputs": str(args.bo_inputs),
                "dataset_tag": data.dataset,
                "matrix_seed": data.matrix_seed,
                "run_seed": int(args.seed),
                "experiment_variant": variant,
                "policy_family": "bo",
                "acquisition": str(args.acquisition),
                "cost_aware": bool(args.cost_aware),
                "cost_scaling_factor": float(args.cost_scaling_factor),
                "n_init": int(n_init_value),
                "n_init_rule": n_init_rule,
                "n_steps": int(args.n_steps),
                "nominal_total_configs": int(nominal_total_configs),
                "extend_to_natural_stop": bool(args.extend_to_natural_stop),
                "n_examples": int(data.n_examples),
                "n_configs": int(data.X.shape[0]),
                "dtype": str(args.dtype),
                **provenance,
            },
        )
        run.define_metric("cum_eval")
        run.define_metric("simple_regret", step_metric="cum_eval")
        run.define_metric("cum_original_cost")
        run.define_metric("iter_fit_s", step_metric="cum_eval")
        run.define_metric("iter_score_s", step_metric="cum_eval")
        run.define_metric("num_evaluated_configs", step_metric="cum_eval")

    out_base = (
        args.out_dir
        / dataset
        / safe_token(variant)
        / (
            f"{Path(args.bo_inputs).stem}__runseed{args.seed}"
            f"__{safe_token(variant)}__ninit{n_init_value}__nsteps{args.n_steps}"
            f"{'__extendstop' if args.extend_to_natural_stop else ''}"
        )
    )
    trace_path = out_base.with_name(out_base.name + "_traces.npz")
    meta_path = out_base.with_name(out_base.name + "_meta.json")
    fig_eval_path = out_base.with_name(out_base.name + "_regret_vs_evals.png")
    fig_cost_path = out_base.with_name(out_base.name + "_regret_vs_cost.png")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    sim = run_bo(args, data, run=run, log_step_metrics=bool(args.log_step_metrics))

    total_wall_time_s = float(time.perf_counter() - run_wall_t0)
    fit_summary = timing_summary(sim["iter_fit_s"])
    score_summary = timing_summary(sim["iter_score_s"])

    np.savez(
        trace_path,
        bo_inputs=str(args.bo_inputs),
        dataset_tag=data.dataset,
        matrix_seed=data.matrix_seed,
        run_seed=int(args.seed),
        experiment_variant=variant,
        policy_variant=variant,
        policy_family="bo",
        acquisition=str(args.acquisition),
        cost_aware=bool(args.cost_aware),
        cost_scaling_factor=float(args.cost_scaling_factor),
        n_init=int(n_init_value),
        n_init_rule=n_init_rule,
        n_steps=int(args.n_steps),
        nominal_total_configs=np.asarray(sim["nominal_total_configs"], dtype=np.int32),
        extend_to_natural_stop=bool(args.extend_to_natural_stop),
        pbgi_stop_cum_eval=np.asarray(
            -1 if sim["stop_cum_eval"] is None else int(sim["stop_cum_eval"]),
            dtype=np.int32,
        ),
        pbgi_stop_cum_original_cost=np.asarray(
            -1.0
            if sim["stop_cum_original_cost"] is None
            else float(sim["stop_cum_original_cost"]),
            dtype=np.float64,
        ),
        pbgi_stop_index_value=np.asarray(
            np.nan if sim["stop_index_value"] is None else float(sim["stop_index_value"]),
            dtype=np.float64,
        ),
        n_examples=int(data.n_examples),
        n_configs=int(data.X.shape[0]),
        cat_dims=np.asarray(data.cat_dims, dtype=np.int32),
        x=np.asarray(sim["x"], dtype=np.int32),
        x_original_cost=np.asarray(sim["x_original_cost"], dtype=np.float64),
        regret=np.asarray(sim["regret"], dtype=np.float32),
        recommended_arm=np.asarray(sim["recommended_arm"], dtype=np.int32),
        recommended_mean=np.asarray(sim["recommended_mean"], dtype=np.float32),
        selected_arm=np.asarray(sim["selected_arm"], dtype=np.int32),
        observed_y=np.asarray(sim["observed_y"], dtype=np.float32),
        acquisition_value=np.asarray(sim["acquisition_value"], dtype=np.float64),
        selection_phase=np.asarray(sim["selection_phase"], dtype="<U32"),
        iter_fit_s=np.asarray(sim["iter_fit_s"], dtype=np.float64),
        iter_score_s=np.asarray(sim["iter_score_s"], dtype=np.float64),
        cost_per_arm_original=np.asarray(data.cost.numpy(), dtype=np.float64),
        git_commit=np.asarray(provenance["git_commit"] or "", dtype="<U64"),
        git_dirty=np.asarray(False if provenance["git_dirty"] is None else provenance["git_dirty"], dtype=np.bool_),
        bo_inputs_sha256=np.asarray(provenance["bo_inputs_sha256"] or "", dtype="<U64"),
        slurm_job_id=np.asarray(provenance["slurm_job_id"] or "", dtype="<U64"),
        hostname=np.asarray(provenance["hostname"] or "", dtype="<U256"),
        python_version=np.asarray(provenance["python_version"] or "", dtype="<U1024"),
    )

    final = {
        "final_simple_regret": float(sim["regret"][-1]) if sim["regret"] else None,
        "best_seen_regret": float(min(sim["regret"])) if sim["regret"] else None,
        "final_cum_eval": int(sim["x"][-1]) if sim["x"] else None,
        "final_cum_original_cost": float(sim["x_original_cost"][-1]) if sim["x_original_cost"] else None,
        "num_evaluated_configs": len(sim["selected_arm"]),
    }
    meta = {
        "bo_inputs": str(args.bo_inputs),
        "dataset_tag": data.dataset,
        "matrix_seed": data.matrix_seed,
        "run_seed": int(args.seed),
        "experiment_variant": variant,
        "policy_family": "bo",
        "acquisition": str(args.acquisition),
        "cost_aware": bool(args.cost_aware),
        "cost_scaling_factor": float(args.cost_scaling_factor),
        "n_init": int(n_init_value),
        "n_init_rule": n_init_rule,
        "n_steps": int(args.n_steps),
        "nominal_total_configs": int(sim["nominal_total_configs"]),
        "extend_to_natural_stop": bool(args.extend_to_natural_stop),
        "pbgi_stop_cum_eval": None if sim["stop_cum_eval"] is None else int(sim["stop_cum_eval"]),
        "pbgi_stop_cum_original_cost": (
            None if sim["stop_cum_original_cost"] is None else float(sim["stop_cum_original_cost"])
        ),
        "pbgi_stop_index_value": (
            None if sim["stop_index_value"] is None else float(sim["stop_index_value"])
        ),
        "n_examples": int(data.n_examples),
        "n_configs": int(data.X.shape[0]),
        "trace": str(trace_path),
        "figure_eval": str(fig_eval_path),
        "figure_cost": str(fig_cost_path),
        "provenance": provenance,
        "timing": {
            "total_wall_time_s": total_wall_time_s,
            "total_fit_s": fit_summary["total_s"],
            "total_score_s": score_summary["total_s"],
            "iter_fit_mean_s": fit_summary["mean_s"],
            "iter_fit_median_s": fit_summary["median_s"],
            "iter_fit_p90_s": fit_summary["p90_s"],
            "iter_fit_p99_s": fit_summary["p99_s"],
            "iter_score_mean_s": score_summary["mean_s"],
            "iter_score_median_s": score_summary["median_s"],
            "iter_score_p90_s": score_summary["p90_s"],
            "iter_score_p99_s": score_summary["p99_s"],
        },
        "final": final,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    title = f"{variant} - {Path(args.bo_inputs).name}\nrun_seed={args.seed}"
    save_line_plot(
        fig_eval_path,
        x=sim["x"],
        y=sim["regret"],
        xlabel="Cumulative examples evaluated",
        title=title,
        label=variant,
    )
    save_line_plot(
        fig_cost_path,
        x=sim["x_original_cost"],
        y=sim["regret"],
        xlabel="Cumulative full-evaluation cost",
        title=title,
        label=variant,
    )

    if run is not None:
        run.summary.update(
            {
                "final_simple_regret": final["final_simple_regret"],
                "best_seen_regret": final["best_seen_regret"],
                "final_cum_eval": final["final_cum_eval"],
                "final_cum_original_cost": final["final_cum_original_cost"],
                "num_evaluated_configs": final["num_evaluated_configs"],
                "pbgi_stop_cum_eval": meta["pbgi_stop_cum_eval"],
                "pbgi_stop_cum_original_cost": meta["pbgi_stop_cum_original_cost"],
                "pbgi_stop_index_value": meta["pbgi_stop_index_value"],
                "total_wall_time_s": total_wall_time_s,
                "total_fit_s": fit_summary["total_s"],
                "total_score_s": score_summary["total_s"],
                "iter_fit_mean_s": fit_summary["mean_s"],
                "iter_fit_median_s": fit_summary["median_s"],
                "iter_fit_p90_s": fit_summary["p90_s"],
                "iter_fit_p99_s": fit_summary["p99_s"],
                "iter_score_mean_s": score_summary["mean_s"],
                "iter_score_median_s": score_summary["median_s"],
                "iter_score_p90_s": score_summary["p90_s"],
                "iter_score_p99_s": score_summary["p99_s"],
                "git_commit": provenance["git_commit"],
                "git_dirty": provenance["git_dirty"],
                "bo_inputs_sha256": provenance["bo_inputs_sha256"],
                "slurm_job_id": provenance["slurm_job_id"],
                "slurm_array_job_id": provenance["slurm_array_job_id"],
                "slurm_array_task_id": provenance["slurm_array_task_id"],
                "hostname": provenance["hostname"],
                "python_version": provenance["python_version"],
            }
        )
        run.log({"regret_vs_evals": wandb.Image(str(fig_eval_path))})
        run.log({"regret_vs_cost": wandb.Image(str(fig_cost_path))})
        run.finish()

    print(f"Wrote trace: {trace_path}")
    print(f"Wrote meta: {meta_path}")
    print(f"Wrote figure: {fig_eval_path}")
    print(f"Wrote cost figure: {fig_cost_path}")
    print(f"final_simple_regret={final['final_simple_regret']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

