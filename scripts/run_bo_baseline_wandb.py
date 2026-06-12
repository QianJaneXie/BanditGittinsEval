#!/usr/bin/env python3
"""W&B runner for discrete BO baseline simple-regret experiments.

One W&B run = one concrete BO baseline configuration.

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

import numpy as np
import torch
import wandb
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from log_ei_puc import LogExpectedImprovementWithCost  # noqa: E402
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
    dominant_dim: int
    metadata: dict[str, Any]


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


DOMINANT_DIM_COL_BY_DATASET: dict[str, int] = {
    "gsm8k": 0,  # model_id
    "piqa": 0,  # model_id
    "mmlu": 1,  # prompt_idx
}
INIT_BUDGET_FRACTION = 0.4


def dominant_dimension_count(X_np: np.ndarray, dataset: str) -> int:
    ds = str(dataset).lower()
    if ds not in DOMINANT_DIM_COL_BY_DATASET:
        raise ValueError(
            "Unsupported dataset for dominant-dimension n_init default: "
            f"{dataset!r}. Supported: {sorted(DOMINANT_DIM_COL_BY_DATASET)}"
        )
    col = DOMINANT_DIM_COL_BY_DATASET[ds]
    if X_np.ndim != 2 or col >= X_np.shape[1]:
        raise ValueError(
            f"Cannot infer dominant dimension from X shape {X_np.shape} for dataset {dataset!r}"
        )
    return int(len(np.unique(X_np[:, col])))


def n_init_budget_cap(*, n_configs: int, eval_budget_fraction: float) -> int:
    return max(
        1,
        int(
            np.round(
                INIT_BUDGET_FRACTION * float(eval_budget_fraction) * int(n_configs)
            )
        ),
    )


def default_bo_n_init(
    *,
    dominant_dim: int,
    n_configs: int,
    eval_budget_fraction: float,
) -> int:
    return min(int(dominant_dim), n_init_budget_cap(n_configs=n_configs, eval_budget_fraction=eval_budget_fraction))


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
    p.add_argument("--dataset-tag", "--dataset_tag", dest="dataset_tag", default=None)
    p.add_argument(
        "--acquisition",
        choices=["pbgi", "logei", "logeipc"],
        default="pbgi",
        help="Acquisition used after the random initialization design.",
    )
    p.add_argument(
        "--experiment-variant",
        "--experiment_variant",
        "--policy-variant",
        "--policy_variant",
        dest="experiment_variant",
        default=None,
        help=(
            "Optional compact BO variant name. Supported forms include pbgi, logei, "
            "logeipc, pbgi_unit, pbgi_cost, and pbgi_cost_aware. If omitted, the "
            "variant is derived from --acquisition and --cost-aware."
        ),
    )
    p.add_argument("--seed", "--run-seed", "--run_seed", dest="run_seed", type=int, default=0)
    p.add_argument(
        "--n-init",
        "--n_init",
        type=int,
        default=None,
        help=(
            "Number of random initial configurations. Defaults to "
            "min(dominant dimension, 40% of eval_budget_fraction * n_configs). "
            "Dominant dimension is model_id for GSM8K/PIQA and prompt_idx for MMLU."
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
            "to the acquisition. LogEIPC is treated as cost-aware even without this flag."
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
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument(
        "--log-step-metrics",
        "--log_step_metrics",
        dest="log_step_metrics",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-log-step-metrics",
        "--no_log_step_metrics",
        dest="log_step_metrics",
        action="store_false",
    )
    p.add_argument("--wandb-entity", "--wandb_entity", default=None)
    p.add_argument("--wandb-project", "--wandb_project", default="GittinsBanditEval")
    p.add_argument("--wandb-group", "--wandb_group", default=None)
    p.add_argument("--wandb-name", "--wandb_name", default=None)
    p.add_argument(
        "--wandb-mode",
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    return p.parse_args()


def method_label_from_variant(variant: str, acquisition: str) -> str:
    acq = str(acquisition).lower()
    cost_mode = "cost" if ("cost" in str(variant) or acq == "logeipc") else "unit"
    labels = {
        ("pbgi", "unit"): "BO PBGI",
        ("pbgi", "cost"): "BO PBGI (cost)",
        ("logei", "unit"): "BO LogEI",
        ("logeipc", "cost"): "BO LogEIPC",
    }
    return labels.get((acq, cost_mode), f"BO {variant}")


def resolve_bo_variant(args: argparse.Namespace) -> tuple[str, bool, str, str]:
    """Resolve acquisition, cost-awareness, compact variant name, and cost mode."""
    raw = None if args.experiment_variant is None else str(args.experiment_variant).strip()
    if not raw:
        acquisition = str(args.acquisition)
        effective_cost_aware = bool(args.cost_aware) or acquisition == "logeipc"
        cost_mode = "cost" if effective_cost_aware else "unit"
        variant = f"{acquisition}_cost_aware" if effective_cost_aware else acquisition
        return acquisition, effective_cost_aware, variant, cost_mode

    v = raw
    cost_aware_from_variant: bool | None = None
    acquisition = v
    if v.endswith("_cost_aware"):
        acquisition = v[: -len("_cost_aware")]
        cost_aware_from_variant = True
    elif v.endswith("_cost"):
        acquisition = v[: -len("_cost")]
        cost_aware_from_variant = True
    elif v.endswith("_unit"):
        acquisition = v[: -len("_unit")]
        cost_aware_from_variant = False

    if acquisition not in {"pbgi", "logei", "logeipc"}:
        raise ValueError(f"Unsupported BO experiment variant: {raw!r}")

    effective_cost_aware = (
        bool(cost_aware_from_variant)
        if cost_aware_from_variant is not None
        else (bool(args.cost_aware) or acquisition == "logeipc")
    )
    cost_mode = "cost" if effective_cost_aware else "unit"
    variant = v
    args.acquisition = acquisition
    args.cost_aware = bool(effective_cost_aware)
    return acquisition, bool(effective_cost_aware), variant, cost_mode


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
        dominant_dim=dominant_dimension_count(X_np, dataset),
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
    """Refit on ``selected``, score remaining candidates, return stop flag and value."""
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


def run_bo_experiment(
    *,
    args: argparse.Namespace,
    data: BoData,
    n_init: int,
    n_steps: int,
    run: wandb.sdk.wandb_run.Run | None,
    log_step_metrics: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.run_seed))
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
    selected_config_arm_id: list[int] = []
    observed_y: list[float] = []
    acquisition_value: list[float] = []
    selection_phase: list[str] = []
    step_idx: list[int] = []
    iter_fit_s: list[float] = []
    iter_score_s: list[float] = []
    iter_total_s: list[float] = []
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
        step_t0 = time.perf_counter()
        selected.append(int(arm))
        remaining.remove(int(arm))
        total_cost += float(data.cost[arm].item())

        rec_arm, rec_value = observed_incumbent(selected, data.Y)
        cum_eval = int(len(selected) * data.n_examples)
        simple_regret = float(mu_star - float(data.Y[rec_arm, 0].item()))
        config_arm_id = int(data.arm_ids[int(arm)])

        x.append(cum_eval)
        x_original_cost.append(float(total_cost))
        regret.append(simple_regret)
        recommended_arm.append(int(rec_arm))
        recommended_mean.append(float(rec_value))
        pulled_arm.append(int(arm))
        selected_config_arm_id.append(config_arm_id)
        observed_y.append(float(data.Y[arm, 0].item()))
        acquisition_value.append(float(acq_value))
        selection_phase.append(phase)
        step_idx.append(int(len(selected)))
        iter_fit_s.append(float(fit_s))
        iter_score_s.append(float(score_s))
        iter_total = float(fit_s) + float(score_s) + float(time.perf_counter() - step_t0)
        iter_total_s.append(iter_total)

        if run is not None and log_step_metrics:
            run.log(
                {
                    "cum_eval": cum_eval,
                    "cum_original_cost": float(total_cost),
                    "simple_regret": simple_regret,
                    "recommended_arm": int(rec_arm),
                    "recommended_mean": float(rec_value),
                    "pulled_arm": int(arm),
                    "selected_config_arm_id": config_arm_id,
                    "observed_y": float(data.Y[arm, 0].item()),
                    "acquisition_value": float(acq_value),
                    "selection_phase": phase,
                    "step_idx": int(len(selected)),
                    "iter_fit_s": float(fit_s),
                    "iter_score_s": float(score_s),
                    "iter_total_s": iter_total,
                }
            )

    for arm in init:
        arm_cost = float(data.cost[int(arm)].item())
        if budget_original_cost is not None and total_cost + arm_cost > float(budget_original_cost):
            break
        record(int(arm), float("nan"), "random_init", 0.0, 0.0)

    within_cost_budget = budget_original_cost is None or total_cost < float(budget_original_cost)
    for _ in range(max_bo_steps):
        if not within_cost_budget:
            break
        if not remaining:
            break
        if not selected:
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
        if budget_original_cost is not None and total_cost + arm_cost > float(budget_original_cost):
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

        within_cost_budget = budget_original_cost is None or total_cost < float(budget_original_cost)

    return {
        "x": x,
        "x_original_cost": x_original_cost,
        "regret": regret,
        "recommended_arm": recommended_arm,
        "recommended_mean": recommended_mean,
        "pulled_arm": pulled_arm,
        "selected_config_arm_id": selected_config_arm_id,
        "observed_y": observed_y,
        "acquisition_value": acquisition_value,
        "selection_phase": selection_phase,
        "step_idx": step_idx,
        "iter_fit_s": iter_fit_s,
        "iter_score_s": iter_score_s,
        "iter_total_s": iter_total_s,
        "stop_cum_eval": stop_cum_eval,
        "stop_cum_original_cost": stop_cum_original_cost,
        "stop_index_value": stop_index_value,
        "nominal_total_configs": nominal_total_configs,
        "cost_aware": bool(effective_cost_aware),
        "total_brute_force_original_cost": total_brute_force_original_cost,
        "budget_original_cost": budget_original_cost,
    }


def main() -> int:
    wall_t0 = time.perf_counter()
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(int(args.run_seed))

    if not args.bo_inputs.is_file():
        print(f"BO inputs not found: {args.bo_inputs}", file=sys.stderr)
        return 1

    data = load_bo_inputs(args.bo_inputs, dtype=dtype)
    dataset_tag = safe_token((args.dataset_tag or data.dataset).lower())
    if data.dataset and dataset_tag != safe_token(str(data.dataset).lower()):
        print(
            f"Warning: --dataset-tag={dataset_tag!r} differs from BO input dataset={data.dataset!r}.",
            file=sys.stderr,
        )

    try:
        acquisition, effective_cost_aware, variant, cost_mode = resolve_bo_variant(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    n_configs = int(data.X.shape[0])
    n_init_budget_cap_value = n_init_budget_cap(
        n_configs=n_configs,
        eval_budget_fraction=float(args.eval_budget_fraction),
    )
    if args.n_init is None:
        requested_n_init = default_bo_n_init(
            dominant_dim=int(data.dominant_dim),
            n_configs=n_configs,
            eval_budget_fraction=float(args.eval_budget_fraction),
        )
        n_init_rule = "dominant_dim_budget_cap_default"
    else:
        requested_n_init = int(args.n_init)
        n_init_rule = "user_set"
    total_brute_force_original_cost = float(data.cost.sum().item())
    n_init_value, n_steps_value, nominal_total_value, n_steps_rule, budget_original_cost = (
        resolve_bo_budget(
            n_configs=n_configs,
            n_init=int(requested_n_init),
            n_steps=args.n_steps,
            eval_budget_fraction=float(args.eval_budget_fraction),
            cost_aware=bool(effective_cost_aware),
            total_brute_force_original_cost=total_brute_force_original_cost,
        )
    )

    policy_variant = variant
    policy_family = "bo"
    method_label = method_label_from_variant(variant, acquisition)

    matrix_seed = str(data.matrix_seed) if str(data.matrix_seed) else None
    metadata_matrix_path = data.metadata.get("matrix_path")
    mmlu_task = None
    if dataset_tag == "mmlu":
        if metadata_matrix_path:
            mmlu_task = Path(str(metadata_matrix_path)).stem
        else:
            stem = Path(args.bo_inputs).stem
            mmlu_task = stem.removesuffix("_bo_inputs").removesuffix("_bo")
    size_bucket = None
    if dataset_tag == "mmlu":
        size_bucket = "small" if data.n_examples <= 150 else "medium" if data.n_examples <= 400 else "large"

    matrix_seed_label = (
        f"task{safe_token(mmlu_task)}"
        if dataset_tag == "mmlu" and mmlu_task
        else f"seed{matrix_seed}" if matrix_seed else "seedNA"
    )
    run_name = args.wandb_name or f"{dataset_tag}_{matrix_seed_label}_{variant}_runseed{args.run_seed}"
    group = args.wandb_group or f"{dataset_tag}_bo_baseline_sweep"

    run: wandb.sdk.wandb_run.Run | None = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=group,
            name=run_name,
            job_type="bo_baseline",
            mode=args.wandb_mode,
            config={
                **vars(args),
                "dataset_tag_resolved": dataset_tag,
                "matrix_seed": matrix_seed,
                "n_examples": int(data.n_examples),
                "n_configs": int(data.X.shape[0]),
                "n_features": int(data.X.shape[1]),
                "cat_dims": list(map(int, data.cat_dims)),
                "dominant_dim": int(data.dominant_dim),
                "init_budget_fraction": float(INIT_BUDGET_FRACTION),
                "n_init_budget_cap": int(n_init_budget_cap_value),
                "n_init": int(n_init_value),
                "n_init_rule": n_init_rule,
                "n_steps": int(n_steps_value),
                "n_steps_rule": str(n_steps_rule),
                "nominal_total_configs": int(nominal_total_value),
                "total_brute_force_original_cost": total_brute_force_original_cost,
                "budget_original_cost": budget_original_cost,
                "cost_aware_run": bool(effective_cost_aware),
                "cost_mode": cost_mode,
                "policy_variant": policy_variant,
                "policy_family": policy_family,
                "method_label": method_label,
                "experiment_variant": variant,
                "mmlu_task": mmlu_task,
                "mmlu_size_bucket": size_bucket,
            },
        )
        run.define_metric("cum_eval")
        run.define_metric("simple_regret", step_metric="cum_eval")
        run.define_metric("cum_original_cost")

    try:
        result = run_bo_experiment(
            args=args,
            data=data,
            n_init=n_init_value,
            n_steps=n_steps_value,
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
    final_cum_eval = int(result["x"][-1]) if result["x"] else None
    final_cum_original_cost = float(result["x_original_cost"][-1]) if result["x_original_cost"] else None
    num_evaluated_configs = len(result["pulled_arm"])
    total_fit_s = float(np.sum(result["iter_fit_s"])) if result["iter_fit_s"] else 0.0
    total_score_s = float(np.sum(result["iter_score_s"])) if result["iter_score_s"] else 0.0
    total_iter_logged_s = float(np.sum(result["iter_total_s"])) if result["iter_total_s"] else 0.0

    print(f"final_simple_regret={final_simple_regret}")

    if run is not None:
        run.summary.update(
            {
                "final_simple_regret": final_simple_regret,
                "best_seen_regret": best_seen_regret,
                "final_cum_eval": final_cum_eval,
                "final_cum_original_cost": final_cum_original_cost,
                "num_evaluated_configs": int(num_evaluated_configs),
                "num_batches": int(num_evaluated_configs),
                "total_wall_time_s": total_wall_time_s,
                "total_fit_s": total_fit_s,
                "total_score_s": total_score_s,
                "total_iter_logged_s": total_iter_logged_s,
                "matrix_seed": matrix_seed,
                "run_seed": int(args.run_seed),
                "experiment_variant": variant,
                "policy_variant": policy_variant,
                "policy_family": policy_family,
                "acquisition": str(acquisition),
                "cost_mode": cost_mode,
                "cost_aware_run": bool(effective_cost_aware),
                "cost_scaling_factor": float(args.cost_scaling_factor),
                "eval_budget_fraction": float(args.eval_budget_fraction),
                "dominant_dim": int(data.dominant_dim),
                "init_budget_fraction": float(INIT_BUDGET_FRACTION),
                "n_init_budget_cap": int(n_init_budget_cap_value),
                "n_init": int(n_init_value),
                "n_init_rule": n_init_rule,
                "n_steps": int(n_steps_value),
                "n_steps_rule": str(n_steps_rule),
                "nominal_total_configs": int(result["nominal_total_configs"]),
                "total_brute_force_original_cost": float(result["total_brute_force_original_cost"]),
                "budget_original_cost": result["budget_original_cost"],
                "n_examples": int(data.n_examples),
                "n_configs": int(data.X.shape[0]),
                "mmlu_task": mmlu_task,
                "mmlu_size_bucket": size_bucket,
                "bo_stop_cum_eval": result["stop_cum_eval"],
                "bo_stop_cum_original_cost": result["stop_cum_original_cost"],
                "bo_stop_index_value": result["stop_index_value"],
                # Backward-compatible aliases with the original run_bo_baseline.py trace keys.
                "pbgi_stop_cum_eval": result["stop_cum_eval"],
                "pbgi_stop_cum_original_cost": result["stop_cum_original_cost"],
                "pbgi_stop_index_value": result["stop_index_value"],
            }
        )
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
