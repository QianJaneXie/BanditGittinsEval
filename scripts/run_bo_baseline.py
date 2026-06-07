#!/usr/bin/env python3
"""Run discrete Bayesian optimization baselines on converted BO inputs.

The input is a ``*_bo_inputs.npz`` file from ``convert_matrix_to_bo_inputs.py``.
Rows are complete configurations, not matrix cells: evaluating one candidate reveals
its aggregate score ``Y`` and consumes the full-evaluation ``cost`` stored in the file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from stable_pbgi import StableGittinsIndex  # noqa: E402
from log_ei_puc import LogExpectedImprovementWithCost  # noqa: E402


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


MAJOR_CATEGORY_COL_BY_DATASET: dict[str, int] = {
    "gsm8k": 0,  # model_id
    "piqa": 0,  # model_id
    "mmlu": 1,  # prompt_idx
}


def major_category_n_init(X_np: np.ndarray, dataset: str) -> int:
    ds = str(dataset).lower()
    if ds not in MAJOR_CATEGORY_COL_BY_DATASET:
        raise ValueError(
            "Unsupported dataset for major-category n_init default: "
            f"{dataset!r}. Supported: {sorted(MAJOR_CATEGORY_COL_BY_DATASET)}"
        )
    col = MAJOR_CATEGORY_COL_BY_DATASET[ds]
    if X_np.ndim != 2 or col >= X_np.shape[1]:
        raise ValueError(
            f"Cannot infer major-category n_init from X shape {X_np.shape} for dataset {dataset!r}"
        )
    return int(len(np.unique(X_np[:, col])))


def resolve_bo_budget(
    *,
    n_configs: int,
    n_init: int,
    n_steps: int | None,
    eval_budget_fraction: float,
    cost_aware: bool,
    total_brute_force_original_cost: float,
) -> tuple[int, int, int, str, float | None]:
    n_init_eff = min(max(int(n_init), 1), int(n_configs))
    budget_original_cost: float | None = None
    if cost_aware:
        budget_original_cost = float(eval_budget_fraction) * float(total_brute_force_original_cost)
        nominal_total = min(
            int(n_configs),
            max(1, int(np.floor(float(eval_budget_fraction) * int(n_configs)))),
        )
        n_steps_eff = max(0, nominal_total - n_init_eff)
        n_steps_rule = "cost_budget_fraction_config_label"
    elif n_steps is None:
        nominal_total = min(
            int(n_configs),
            max(1, int(np.floor(float(eval_budget_fraction) * int(n_configs)))),
        )
        n_steps_eff = max(0, nominal_total - n_init_eff)
        n_steps_rule = "eval_budget_fraction_default"
    else:
        n_steps_eff = min(max(int(n_steps), 0), int(n_configs) - n_init_eff)
        n_steps_rule = "user_set"
        nominal_total = min(int(n_configs), n_init_eff + n_steps_eff)
    return n_init_eff, n_steps_eff, nominal_total, n_steps_rule, budget_original_cost


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bo-inputs", "--bo_inputs", dest="bo_inputs", type=Path, required=True)
    p.add_argument(
        "--acquisition",
        choices=["pbgi", "logei", "logeipc"],
        default="pbgi",
        help="Acquisition used after the random initialization design.",
    )
    p.add_argument("--seed", "--run-seed", "--run_seed", dest="seed", type=int, default=0)
    p.add_argument(
        "--n-init",
        "--n_init",
        type=int,
        default=None,
        help=(
            "Number of random initial configurations. Defaults to the number of "
            "levels in the major categorical dimension (model_id for GSM8K/PIQA, "
            "prompt_idx for MMLU)."
        ),
    )
    p.add_argument(
        "--n-steps",
        "--n_steps",
        type=int,
        default=None,
        help=(
            "Number of BO-selected configurations after initialization. "
            "Defaults to max(0, floor(eval_budget_fraction * n_configs) - n_init)."
        ),
    )
    p.add_argument(
        "--eval-budget-fraction",
        "--eval_budget_fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of the full-evaluation budget. Unit-cost runs stop after this "
            "fraction of configurations (init + BO). Cost-aware runs stop after "
            "cumulative cost reaches this fraction of sum(cost)."
        ),
    )
    p.add_argument(
        "--cost-aware",
        "--cost_aware",
        action="store_true",
        help=(
            "Use costs in acquisition ranking. For PBGI and LogEIPC this passes cost_X "
            "to the acquisition."
        ),
    )
    p.add_argument(
        "--cost-scaling-factor",
        "--cost_scaling_factor",
        type=float,
        default=1e-4,
        help="Cost scale used by PBGI, matching the bandit Gittins default.",
    )
    p.add_argument("--observation-noise", "--observation_noise", type=float, default=1e-6)
    p.add_argument("--out-dir", "--out_dir", type=Path, default=Path("outputs") / "bo_baselines")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
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
        default_n_init=major_category_n_init(X_np, dataset),
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
    best_f: float,
    candidate_X: torch.Tensor,
    candidate_cost: torch.Tensor,
    cost_aware: bool,
    cost_scaling_factor: float,
) -> torch.Tensor:
    Xq = candidate_X.unsqueeze(1)
    with torch.no_grad():
        if acquisition == "pbgi":
            acq = StableGittinsIndex(model, lmbda=float(cost_scaling_factor))
            if cost_aware:
                scores = acq(Xq, cost_X=candidate_cost)
            else:
                scores = acq(Xq)
        elif acquisition == "logei":
            acq = LogExpectedImprovement(model=model, best_f=float(best_f), maximize=True)
            scores = acq(Xq)
        elif acquisition == "logeipc":
            acq = LogExpectedImprovementWithCost(model=model, best_f=float(best_f), maximize=True)
            if cost_aware:
                scores = acq(Xq, cost_X=candidate_cost)
            else:
                scores = acq(Xq)
        else:  # pragma: no cover - argparse constrains this.
            raise ValueError(f"Unsupported acquisition: {acquisition}")
    return scores.reshape(-1).detach()


def observed_incumbent(selected: list[int], Y: torch.Tensor) -> tuple[int, float]:
    selected_arr = torch.tensor(selected, dtype=torch.long)
    observed_y = Y[selected_arr, 0]
    best_pos = int(torch.argmax(observed_y).item())
    arm = int(selected_arr[best_pos].item())
    return arm, float(Y[arm, 0].item())


def bo_should_stop_post_pull(
    *,
    acquisition: str,
    max_remaining_acq: float,
    best_observed: float,
    log_lambda: float,
) -> bool:
    """Stopping rules using acquisition values after the latest eval is in the training set."""
    if acquisition in {"logei", "logeipc"}:
        return bool(max_remaining_acq < log_lambda)
    if acquisition == "pbgi":
        return bool(max_remaining_acq < best_observed)
    raise ValueError(f"Unsupported acquisition: {acquisition}")


def evaluate_bo_stopping_post_pull(
    *,
    acquisition: str,
    data: BoData,
    selected: list[int],
    remaining: set[int],
    observation_noise: float,
    cost_aware: bool,
    cost_scaling_factor: float,
    log_lambda: float,
) -> tuple[bool, float | None, float]:
    """Refit on ``selected``, score remaining candidates, return (should_stop, stop_index, best_observed)."""
    if not remaining:
        train_idx = torch.tensor(selected, dtype=torch.long)
        best_observed = float(data.Y[train_idx].max().item())
        return False, None, best_observed

    train_idx = torch.tensor(selected, dtype=torch.long)
    train_X = data.X[train_idx]
    train_Y = data.Y[train_idx]
    best_observed = float(train_Y.max().item())
    model = fit_mixed_gp(
        train_X,
        train_Y,
        observation_noise=float(observation_noise),
        cat_dims=data.cat_dims,
    )
    remaining_idx = torch.tensor(sorted(remaining), dtype=torch.long)
    scores = score_candidates(
        acquisition=acquisition,
        model=model,
        best_f=best_observed,
        candidate_X=data.X[remaining_idx],
        candidate_cost=data.cost[remaining_idx],
        cost_aware=bool(cost_aware),
        cost_scaling_factor=float(cost_scaling_factor),
    )
    if not torch.isfinite(scores).any():
        return False, None, best_observed
    max_remaining_acq = float(scores.max().item())
    should_stop = bo_should_stop_post_pull(
        acquisition=acquisition,
        max_remaining_acq=max_remaining_acq,
        best_observed=best_observed,
        log_lambda=log_lambda,
    )
    return should_stop, max_remaining_acq, best_observed


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


def run_bo(
    args: argparse.Namespace,
    data: BoData,
    *,
    n_init: int,
    n_steps: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    n_configs = int(data.X.shape[0])
    n_init = min(max(int(n_init), 1), n_configs)
    n_steps = min(max(int(n_steps), 0), n_configs - n_init)

    init = rng.choice(n_configs, size=n_init, replace=False).astype(int).tolist()
    selected: list[int] = []
    remaining: set[int] = set(range(n_configs))

    x: list[int] = []
    x_original_cost: list[float] = []
    regret: list[float] = []
    recommended_arm: list[int] = []
    recommended_mean: list[float] = []
    pulled_arm: list[int] = []
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
    log_lambda = float(np.log(float(args.cost_scaling_factor)))
    effective_cost_aware = bool(args.cost_aware) or str(args.acquisition) == "logeipc"
    total_brute_force_original_cost = float(data.cost.sum().item())
    budget_original_cost: float | None = None
    if effective_cost_aware:
        budget_original_cost = float(args.eval_budget_fraction) * total_brute_force_original_cost
    nominal_total_configs = min(n_configs, n_init + n_steps)
    max_bo_steps = int(n_configs - n_init) if budget_original_cost is not None else int(n_steps)

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
        pulled_arm.append(int(arm))
        observed_y.append(float(data.Y[arm, 0].item()))
        acquisition_value.append(float(acq_value))
        selection_phase.append(phase)
        iter_fit_s.append(float(fit_s))
        iter_score_s.append(float(score_s))

    for arm in init:
        arm_cost = float(data.cost[int(arm)].item())
        if (
            budget_original_cost is not None
            and total_cost + arm_cost > float(budget_original_cost)
        ):
            break
        record(int(arm), float("nan"), "random_init", 0.0, 0.0)

    within_cost_budget = (
        budget_original_cost is None or total_cost < float(budget_original_cost)
    )
    for _ in range(max_bo_steps):
        if not within_cost_budget:
            break
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
            best_f=float(train_Y.max().item()),
            candidate_X=data.X[remaining_idx],
            candidate_cost=data.cost[remaining_idx],
            cost_aware=bool(effective_cost_aware),
            cost_scaling_factor=float(args.cost_scaling_factor),
        )
        score_s = float(time.perf_counter() - score_t0)

        if not torch.isfinite(scores).any():
            raise RuntimeError("No finite acquisition scores were produced.")
        best_pos = int(torch.argmax(scores).item())
        best_score = float(scores[best_pos].item())
        arm = int(remaining_idx[best_pos].item())
        arm_cost = float(data.cost[arm].item())
        if (
            budget_original_cost is not None
            and total_cost + arm_cost > float(budget_original_cost)
        ):
            break
        record(arm, best_score, str(args.acquisition), fit_s, score_s)

        if stop_cum_eval is None:
            should_stop, stop_acq, _ = evaluate_bo_stopping_post_pull(
                acquisition=str(args.acquisition),
                data=data,
                selected=selected,
                remaining=remaining,
                observation_noise=float(args.observation_noise),
                cost_aware=bool(effective_cost_aware),
                cost_scaling_factor=float(args.cost_scaling_factor),
                log_lambda=log_lambda,
            )
            if should_stop and stop_acq is not None:
                stop_cum_eval = int(len(selected) * data.n_examples)
                stop_cum_original_cost = float(total_cost)
                stop_index_value = float(stop_acq)

        within_cost_budget = (
            budget_original_cost is None or total_cost < float(budget_original_cost)
        )

    return {
        "x": x,
        "x_original_cost": x_original_cost,
        "regret": regret,
        "recommended_arm": recommended_arm,
        "recommended_mean": recommended_mean,
        "pulled_arm": pulled_arm,
        "observed_y": observed_y,
        "acquisition_value": acquisition_value,
        "selection_phase": selection_phase,
        "iter_fit_s": iter_fit_s,
        "iter_score_s": iter_score_s,
        "stop_cum_eval": stop_cum_eval,
        "stop_cum_original_cost": stop_cum_original_cost,
        "stop_index_value": stop_index_value,
        "nominal_total_configs": nominal_total_configs,
        "cost_aware": bool(effective_cost_aware),
        "total_brute_force_original_cost": total_brute_force_original_cost,
        "budget_original_cost": budget_original_cost,
    }


def main() -> int:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(int(args.seed))

    data = load_bo_inputs(args.bo_inputs, dtype=dtype)
    requested_n_init = data.default_n_init if args.n_init is None else int(args.n_init)
    effective_cost_aware = bool(args.cost_aware) or str(args.acquisition) == "logeipc"
    total_brute_force_original_cost = float(data.cost.sum().item())
    n_init_value, n_steps_value, nominal_total_value, n_steps_rule, budget_original_cost = (
        resolve_bo_budget(
            n_configs=int(data.X.shape[0]),
            n_init=int(requested_n_init),
            n_steps=args.n_steps,
            eval_budget_fraction=float(args.eval_budget_fraction),
            cost_aware=bool(effective_cost_aware),
            total_brute_force_original_cost=total_brute_force_original_cost,
        )
    )
    variant = str(args.acquisition)
    if effective_cost_aware:
        variant = f"{variant}_cost_aware"

    out_base = (
        args.out_dir
        / safe_token(data.dataset or "unknown")
        / safe_token(variant)
        / (
            f"{Path(args.bo_inputs).stem}__runseed{args.seed}"
            f"__{safe_token(variant)}__ninit{n_init_value}__nsteps{n_steps_value}"
        )
    )
    trace_path = out_base.with_name(out_base.name + "_traces.npz")
    meta_path = out_base.with_name(out_base.name + "_meta.json")
    fig_eval_path = out_base.with_name(out_base.name + "_regret_vs_evals.png")
    fig_cost_path = out_base.with_name(out_base.name + "_regret_vs_cost.png")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    sim = run_bo(args, data, n_init=n_init_value, n_steps=n_steps_value)

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
        cost_aware=bool(effective_cost_aware),
        cost_scaling_factor=float(args.cost_scaling_factor),
        eval_budget_fraction=np.asarray(float(args.eval_budget_fraction), dtype=np.float64),
        n_init=int(n_init_value),
        n_init_rule="major_category_levels_default" if args.n_init is None else "user_set",
        n_steps=int(n_steps_value),
        n_steps_rule=str(n_steps_rule),
        nominal_total_configs=np.asarray(sim["nominal_total_configs"], dtype=np.int32),
        total_brute_force_original_cost=np.asarray(
            float(sim["total_brute_force_original_cost"]), dtype=np.float64
        ),
        budget_original_cost=np.asarray(
            -1.0
            if sim["budget_original_cost"] is None
            else float(sim["budget_original_cost"]),
            dtype=np.float64,
        ),
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
        pulled_arm=np.asarray(sim["pulled_arm"], dtype=np.int32),
        observed_y=np.asarray(sim["observed_y"], dtype=np.float32),
        acquisition_value=np.asarray(sim["acquisition_value"], dtype=np.float64),
        selection_phase=np.asarray(sim["selection_phase"], dtype="<U32"),
        iter_fit_s=np.asarray(sim["iter_fit_s"], dtype=np.float64),
        iter_score_s=np.asarray(sim["iter_score_s"], dtype=np.float64),
        cost_per_arm_original=np.asarray(data.cost.numpy(), dtype=np.float64),
    )

    final = {
        "final_simple_regret": float(sim["regret"][-1]) if sim["regret"] else None,
        "best_seen_regret": float(min(sim["regret"])) if sim["regret"] else None,
        "final_cum_eval": int(sim["x"][-1]) if sim["x"] else None,
        "final_cum_original_cost": float(sim["x_original_cost"][-1]) if sim["x_original_cost"] else None,
        "num_evaluated_configs": len(sim["pulled_arm"]),
    }
    meta = {
        "bo_inputs": str(args.bo_inputs),
        "dataset_tag": data.dataset,
        "matrix_seed": data.matrix_seed,
        "run_seed": int(args.seed),
        "experiment_variant": variant,
        "policy_family": "bo",
        "acquisition": str(args.acquisition),
        "cost_aware": bool(effective_cost_aware),
        "cost_scaling_factor": float(args.cost_scaling_factor),
        "eval_budget_fraction": float(args.eval_budget_fraction),
        "n_init": int(n_init_value),
        "n_init_rule": "major_category_levels_default" if args.n_init is None else "user_set",
        "n_steps": int(n_steps_value),
        "n_steps_rule": str(n_steps_rule),
        "nominal_total_configs": int(sim["nominal_total_configs"]),
        "total_brute_force_original_cost": float(sim["total_brute_force_original_cost"]),
        "budget_original_cost": (
            None if sim["budget_original_cost"] is None else float(sim["budget_original_cost"])
        ),
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

    print(f"Wrote trace: {trace_path}")
    print(f"Wrote meta: {meta_path}")
    print(f"Wrote figure: {fig_eval_path}")
    print(f"Wrote cost figure: {fig_cost_path}")
    print(f"final_simple_regret={final['final_simple_regret']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
