#!/usr/bin/env python3
"""W&B runner for discrete Bayesian optimization baselines on converted BO inputs.

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

try:
    import resource
except Exception:  # pragma: no cover
    resource = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

_repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
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
    feature_names: list[str]
    model_names: np.ndarray | None = None
    temperatures: np.ndarray | None = None
    max_tokens: np.ndarray | None = None
    prompt_names: np.ndarray | None = None
    prompt_types: np.ndarray | None = None


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


@dataclass(frozen=True)
class BoVariantConfig:
    raw: str
    acquisition: str
    cost_aware: bool


BO_EXPERIMENT_VARIANTS: dict[str, BoVariantConfig] = {
    "bo_pbgi": BoVariantConfig("bo_pbgi", "pbgi", False),
    "bo_logei": BoVariantConfig("bo_logei", "logei", False),
    "bo_pbgi_cost": BoVariantConfig("bo_pbgi_cost", "pbgi", True),
    "bo_logeipc_cost": BoVariantConfig("bo_logeipc_cost", "logeipc", True),
}


def resolve_variant_from_args(args: argparse.Namespace) -> BoVariantConfig:
    """Resolve the readable sweep variant into concrete acquisition/cost settings.

    This keeps the W&B YAML compact (`experiment_variant: bo_pbgi_cost`) while
    preserving backwards-compatible direct CLI usage (`--acquisition pbgi --cost-aware`).
    The resolved values are written back to `args` so the existing code path uses
    one source of truth.
    """
    if args.experiment_variant is not None:
        raw = str(args.experiment_variant)
        if raw not in BO_EXPERIMENT_VARIANTS:
            raise ValueError(
                f"Unsupported BO experiment_variant={raw!r}. Supported: "
                f"{sorted(BO_EXPERIMENT_VARIANTS)}"
            )
        resolved = BO_EXPERIMENT_VARIANTS[raw]
        args.acquisition = resolved.acquisition
        args.cost_aware = bool(resolved.cost_aware)
        return resolved

    effective_cost_aware = bool(args.cost_aware) or str(args.acquisition) == "logeipc"
    if str(args.acquisition) == "logeipc" or effective_cost_aware:
        raw = "bo_logeipc_cost" if str(args.acquisition) == "logeipc" else f"bo_{args.acquisition}_cost"
    else:
        raw = f"bo_{args.acquisition}"
    return BoVariantConfig(raw=raw, acquisition=str(args.acquisition), cost_aware=effective_cost_aware)


def _optional_npz_array(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in z.files:
        return None
    return np.asarray(z[key])


def _string_at(arr: np.ndarray | None, idx: int) -> str | None:
    if arr is None:
        return None
    try:
        value = arr[int(idx)]
    except Exception:
        return None
    s = str(value)
    if not s or s.lower() in {"nan", "none"}:
        return None
    return s


def _float_at(arr: np.ndarray | None, idx: int) -> float | None:
    if arr is None:
        return None
    try:
        value = float(np.asarray(arr[int(idx)]).reshape(()))
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _int_at(arr: np.ndarray | None, idx: int) -> int | None:
    value = _float_at(arr, idx)
    if value is None:
        return None
    return int(value)


def arm_feature_dict(data: BoData, arm: int) -> dict[str, int | float]:
    vals = data.X[int(arm)].detach().cpu().numpy().reshape(-1)
    out: dict[str, int | float] = {}
    for j, value in enumerate(vals):
        name = data.feature_names[j] if j < len(data.feature_names) else f"x{j}"
        fv = float(value)
        out[name] = int(fv) if fv.is_integer() else fv
    return out


def arm_content_dict(data: BoData, arm: int) -> dict[str, Any]:
    arm = int(arm)
    return {
        "arm_id": int(data.arm_ids[arm]) if data.arm_ids.size > arm else arm,
        "features": arm_feature_dict(data, arm),
        "model_name": _string_at(data.model_names, arm),
        "temperature": _float_at(data.temperatures, arm),
        "max_tokens": _int_at(data.max_tokens, arm),
        "prompt_name": _string_at(data.prompt_names, arm),
        "prompt_type": _string_at(data.prompt_types, arm),
        "score_y": float(data.Y[arm, 0].item()),
        "full_eval_cost": float(data.cost[arm].item()),
    }


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _ru_maxrss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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
    peak = _ru_maxrss_bytes()
    if peak is not None:
        return peak
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    return None


def bytes_to_mb(x: int | None) -> float | None:
    return None if x is None else float(x) / (1024.0 ** 2)


def bytes_to_gb(x: int | None) -> float | None:
    return None if x is None else float(x) / (1024.0 ** 3)


def timing_summary(values: list[float], prefix: str) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_total_s": 0.0,
            f"{prefix}_mean_s": None,
            f"{prefix}_median_s": None,
            f"{prefix}_p90_s": None,
            f"{prefix}_p99_s": None,
            f"{prefix}_min_s": None,
            f"{prefix}_max_s": None,
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_total_s": float(arr.sum()),
        f"{prefix}_mean_s": float(arr.mean()),
        f"{prefix}_median_s": float(np.median(arr)),
        f"{prefix}_p90_s": float(np.percentile(arr, 90)),
        f"{prefix}_p99_s": float(np.percentile(arr, 99)),
        f"{prefix}_min_s": float(arr.min()),
        f"{prefix}_max_s": float(arr.max()),
    }


def dataset_content_summary(data: BoData) -> dict[str, Any]:
    best_arm = int(torch.argmax(data.Y[:, 0]).item())
    costs = data.cost.detach().cpu().numpy().astype(float).reshape(-1)
    y = data.Y[:, 0].detach().cpu().numpy().astype(float).reshape(-1)
    unique_feature_counts = {
        name: int(len(np.unique(data.X[:, j].detach().cpu().numpy())))
        for j, name in enumerate(data.feature_names)
    }
    best = arm_content_dict(data, best_arm)
    return {
        "feature_names": list(data.feature_names),
        "feature_names_json": compact_json(list(data.feature_names)),
        "unique_feature_counts_json": compact_json(unique_feature_counts),
        "global_best_arm": int(best_arm),
        "global_best_arm_id": int(best["arm_id"]),
        "global_best_mean": float(best["score_y"]),
        "global_best_model_name": best.get("model_name"),
        "global_best_prompt_name": best.get("prompt_name"),
        "global_best_prompt_type": best.get("prompt_type"),
        "global_best_temperature": best.get("temperature"),
        "global_best_max_tokens": best.get("max_tokens"),
        "global_best_features_json": compact_json(best.get("features", {})),
        "score_y_mean": float(np.mean(y)),
        "score_y_std": float(np.std(y)),
        "score_y_min": float(np.min(y)),
        "score_y_max": float(np.max(y)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "cost_min": float(np.min(costs)),
        "cost_max": float(np.max(costs)),
    }


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
    p.add_argument(
        "--experiment-variant",
        "--experiment_variant",
        "--policy-variant",
        "--policy_variant",
        dest="experiment_variant",
        choices=sorted(BO_EXPERIMENT_VARIANTS),
        default=None,
        help=(
            "Readable W&B sweep variant. If set, it overrides --acquisition and "
            "--cost-aware using the fixed mapping bo_pbgi/bo_logei/bo_pbgi_cost/bo_logeipc_cost."
        ),
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
    p.add_argument("--out-dir", "--out_dir", type=Path, default=Path("outputs") / "bo_baselines_wandb")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument(
        "--log-step-metrics",
        "--log_step_metrics",
        action="store_true",
        help="Log one W&B history row per BO evaluation step.",
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
    p.add_argument(
        "--wandb-save-files",
        "--wandb_save_files",
        action="store_true",
        help="Upload trace/meta/figure files to the W&B run files.",
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
    feature_names = (
        [str(x) for x in np.asarray(z["feature_names"]).reshape(-1).tolist()]
        if "feature_names" in z.files
        else [str(x) for x in metadata.get("feature_names", [])]
    )
    if not feature_names:
        feature_names = [f"x{j}" for j in range(int(X_np.shape[1]))]
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
        feature_names=feature_names,
        model_names=_optional_npz_array(z, "model_names"),
        temperatures=_optional_npz_array(z, "temperatures"),
        max_tokens=_optional_npz_array(z, "max_tokens"),
        prompt_names=_optional_npz_array(z, "prompt_names"),
        prompt_types=_optional_npz_array(z, "prompt_types"),
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
    selected_arm: list[int] = []
    observed_y: list[float] = []
    acquisition_value: list[float] = []
    selection_phase: list[str] = []
    iter_fit_s: list[float] = []
    iter_score_s: list[float] = []
    iter_step_s: list[float] = []
    iter_total_s: list[float] = []
    iter_rss_mb: list[float] = []
    iter_peak_rss_mb: list[float] = []
    selected_cost: list[float] = []
    recommended_cost: list[float] = []
    selected_feature_json: list[str] = []
    recommended_feature_json: list[str] = []
    selected_model_name: list[str] = []
    recommended_model_name: list[str] = []
    selected_prompt_name: list[str] = []
    recommended_prompt_name: list[str] = []
    selected_prompt_type: list[str] = []
    recommended_prompt_type: list[str] = []
    selected_temperature: list[float] = []
    recommended_temperature: list[float] = []
    selected_max_tokens: list[int] = []
    recommended_max_tokens: list[int] = []
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
        selected_content = arm_content_dict(data, int(arm))
        recommended_content = arm_content_dict(data, int(rec_arm))

        recommended_arm.append(int(rec_arm))
        recommended_mean.append(float(rec_value))
        selected_arm.append(int(arm))
        observed_y.append(float(data.Y[arm, 0].item()))
        acquisition_value.append(float(acq_value))
        selection_phase.append(phase)
        iter_fit_s.append(float(fit_s))
        iter_score_s.append(float(score_s))
        iter_step_s.append(float(fit_s) + float(score_s))
        iter_total_s.append(float(fit_s) + float(score_s))
        rss_b = current_rss_bytes()
        peak_b = peak_rss_bytes()
        iter_rss_mb.append(float("nan") if rss_b is None else float(bytes_to_mb(rss_b)))
        iter_peak_rss_mb.append(float("nan") if peak_b is None else float(bytes_to_mb(peak_b)))
        selected_cost.append(float(selected_content["full_eval_cost"]))
        recommended_cost.append(float(recommended_content["full_eval_cost"]))
        selected_feature_json.append(compact_json(selected_content["features"]))
        recommended_feature_json.append(compact_json(recommended_content["features"]))
        selected_model_name.append(str(selected_content.get("model_name") or ""))
        recommended_model_name.append(str(recommended_content.get("model_name") or ""))
        selected_prompt_name.append(str(selected_content.get("prompt_name") or ""))
        recommended_prompt_name.append(str(recommended_content.get("prompt_name") or ""))
        selected_prompt_type.append(str(selected_content.get("prompt_type") or ""))
        recommended_prompt_type.append(str(recommended_content.get("prompt_type") or ""))
        selected_temperature.append(float("nan") if selected_content.get("temperature") is None else float(selected_content["temperature"]))
        recommended_temperature.append(float("nan") if recommended_content.get("temperature") is None else float(recommended_content["temperature"]))
        selected_max_tokens.append(-1 if selected_content.get("max_tokens") is None else int(selected_content["max_tokens"]))
        recommended_max_tokens.append(-1 if recommended_content.get("max_tokens") is None else int(recommended_content["max_tokens"]))

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
        "selected_arm": selected_arm,
        "observed_y": observed_y,
        "acquisition_value": acquisition_value,
        "selection_phase": selection_phase,
        "iter_fit_s": iter_fit_s,
        "iter_score_s": iter_score_s,
        "iter_step_s": iter_step_s,
        "iter_total_s": iter_total_s,
        "iter_rss_mb": iter_rss_mb,
        "iter_peak_rss_mb": iter_peak_rss_mb,
        "selected_cost": selected_cost,
        "recommended_cost": recommended_cost,
        "selected_feature_json": selected_feature_json,
        "recommended_feature_json": recommended_feature_json,
        "selected_model_name": selected_model_name,
        "recommended_model_name": recommended_model_name,
        "selected_prompt_name": selected_prompt_name,
        "recommended_prompt_name": recommended_prompt_name,
        "selected_prompt_type": selected_prompt_type,
        "recommended_prompt_type": recommended_prompt_type,
        "selected_temperature": selected_temperature,
        "recommended_temperature": recommended_temperature,
        "selected_max_tokens": selected_max_tokens,
        "recommended_max_tokens": recommended_max_tokens,
        "stop_cum_eval": stop_cum_eval,
        "stop_cum_original_cost": stop_cum_original_cost,
        "stop_index_value": stop_index_value,
        "nominal_total_configs": nominal_total_configs,
        "cost_aware": bool(effective_cost_aware),
        "total_brute_force_original_cost": total_brute_force_original_cost,
        "budget_original_cost": budget_original_cost,
    }



def make_wandb_run_name(
    *,
    data: BoData,
    bo_inputs: Path,
    variant: str,
    seed: int,
) -> str:
    dataset = safe_token(data.dataset or "unknown")
    stem = safe_token(Path(bo_inputs).stem)
    return f"{dataset}__{stem}__{safe_token(variant)}__runseed{int(seed)}"


def init_wandb_run(
    args: argparse.Namespace,
    *,
    data: BoData,
    variant: str,
    n_init_value: int,
    n_steps_value: int,
    n_steps_rule: str,
    effective_cost_aware: bool,
    budget_original_cost: float | None,
) -> wandb.sdk.wandb_run.Run | None:
    if str(args.wandb_mode) == "disabled":
        return None

    run_name = args.wandb_name or make_wandb_run_name(
        data=data,
        bo_inputs=args.bo_inputs,
        variant=variant,
        seed=int(args.seed),
    )
    content_summary = dataset_content_summary(data)
    config = {
        "bo_inputs": str(args.bo_inputs),
        "dataset_tag": data.dataset,
        "matrix_seed": data.matrix_seed,
        "run_seed": int(args.seed),
        "experiment_variant": variant,
        "policy_variant": variant,
        "policy_family": "bo",
        "acquisition": str(args.acquisition),
        "cost_aware": bool(effective_cost_aware),
        "cost_scaling_factor": float(args.cost_scaling_factor),
        "observation_noise": float(args.observation_noise),
        "eval_budget_fraction": float(args.eval_budget_fraction),
        "n_init": int(n_init_value),
        "n_init_rule": "major_category_levels_default" if args.n_init is None else "user_set",
        "n_steps": int(n_steps_value),
        "n_steps_rule": str(n_steps_rule),
        "budget_original_cost": None if budget_original_cost is None else float(budget_original_cost),
        "n_examples": int(data.n_examples),
        "n_configs": int(data.X.shape[0]),
        "cat_dims": list(map(int, data.cat_dims)),
        "dtype": str(args.dtype),
        **content_summary,
    }
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        mode=args.wandb_mode,
        config=config,
    )
    run.define_metric("cum_eval")
    run.define_metric("cum_original_cost")
    run.define_metric("simple_regret", step_metric="cum_eval")
    run.define_metric("recommended_mean", step_metric="cum_eval")
    run.define_metric("observed_y", step_metric="cum_eval")
    run.define_metric("acquisition_value", step_metric="cum_eval")
    run.define_metric("iter_fit_s", step_metric="cum_eval")
    run.define_metric("iter_score_s", step_metric="cum_eval")
    run.define_metric("iter_step_s", step_metric="cum_eval")
    run.define_metric("iter_total_s", step_metric="cum_eval")
    run.define_metric("iter_rss_mb", step_metric="cum_eval")
    run.define_metric("iter_peak_rss_mb", step_metric="cum_eval")
    run.define_metric("selected_cost", step_metric="cum_eval")
    run.define_metric("recommended_cost", step_metric="cum_eval")
    run.define_metric("cum_configs", step_metric="cum_eval")
    return run


def log_wandb_history(
    run: wandb.sdk.wandb_run.Run | None,
    sim: dict[str, Any],
) -> None:
    if run is None:
        return
    n = len(sim["x"])
    for i in range(n):
        run.log(
            {
                "step_idx": int(i),
                "cum_configs": int(i + 1),
                "cum_eval": int(sim["x"][i]),
                "cum_original_cost": float(sim["x_original_cost"][i]),
                "simple_regret": float(sim["regret"][i]),
                "recommended_arm": int(sim["recommended_arm"][i]),
                "recommended_mean": float(sim["recommended_mean"][i]),
                "selected_arm": int(sim["selected_arm"][i]),
                "observed_y": float(sim["observed_y"][i]),
                "acquisition_value": float(sim["acquisition_value"][i])
                if np.isfinite(sim["acquisition_value"][i])
                else None,
                "selection_phase": str(sim["selection_phase"][i]),
                "iter_fit_s": float(sim["iter_fit_s"][i]),
                "iter_score_s": float(sim["iter_score_s"][i]),
                "iter_step_s": float(sim["iter_step_s"][i]),
                "iter_total_s": float(sim["iter_total_s"][i]),
                "iter_rss_mb": float(sim["iter_rss_mb"][i]) if np.isfinite(sim["iter_rss_mb"][i]) else None,
                "iter_peak_rss_mb": float(sim["iter_peak_rss_mb"][i]) if np.isfinite(sim["iter_peak_rss_mb"][i]) else None,
                "selected_cost": float(sim["selected_cost"][i]),
                "recommended_cost": float(sim["recommended_cost"][i]),
                "selected_feature_json": str(sim["selected_feature_json"][i]),
                "recommended_feature_json": str(sim["recommended_feature_json"][i]),
                "selected_model_name": str(sim["selected_model_name"][i]),
                "recommended_model_name": str(sim["recommended_model_name"][i]),
                "selected_prompt_name": str(sim["selected_prompt_name"][i]),
                "recommended_prompt_name": str(sim["recommended_prompt_name"][i]),
                "selected_prompt_type": str(sim["selected_prompt_type"][i]),
                "recommended_prompt_type": str(sim["recommended_prompt_type"][i]),
                "selected_temperature": float(sim["selected_temperature"][i]) if np.isfinite(sim["selected_temperature"][i]) else None,
                "recommended_temperature": float(sim["recommended_temperature"][i]) if np.isfinite(sim["recommended_temperature"][i]) else None,
                "selected_max_tokens": int(sim["selected_max_tokens"][i]) if int(sim["selected_max_tokens"][i]) >= 0 else None,
                "recommended_max_tokens": int(sim["recommended_max_tokens"][i]) if int(sim["recommended_max_tokens"][i]) >= 0 else None,
            }
        )


def update_wandb_summary_and_files(
    run: wandb.sdk.wandb_run.Run | None,
    *,
    final: dict[str, Any],
    meta: dict[str, Any],
    total_wall_time_s: float,
    trace_path: Path,
    meta_path: Path,
    fig_eval_path: Path,
    fig_cost_path: Path,
    save_files: bool,
) -> None:
    if run is None:
        return
    summary = dict(final)
    summary.update(
        total_wall_time_s=float(total_wall_time_s),
        trace_path=str(trace_path),
        meta_path=str(meta_path),
        figure_eval_path=str(fig_eval_path),
        figure_cost_path=str(fig_cost_path),
        pbgi_stop_cum_eval=meta.get("pbgi_stop_cum_eval"),
        pbgi_stop_cum_original_cost=meta.get("pbgi_stop_cum_original_cost"),
        pbgi_stop_index_value=meta.get("pbgi_stop_index_value"),
        nominal_total_configs=meta.get("nominal_total_configs"),
        budget_original_cost=meta.get("budget_original_cost"),
        total_brute_force_original_cost=meta.get("total_brute_force_original_cost"),
        iter_fit_mean_s=meta.get("iter_fit_mean_s"),
        iter_score_mean_s=meta.get("iter_score_mean_s"),
        iter_step_mean_s=meta.get("iter_step_mean_s"),
        iter_total_mean_s=meta.get("iter_total_mean_s"),
        peak_rss_gb=meta.get("peak_rss_gb"),
        extra_peak_memory_gb=meta.get("extra_peak_memory_gb"),
        rss_before_mb=meta.get("rss_before_mb"),
        rss_after_mb=meta.get("rss_after_mb"),
        rss_delta_mb=meta.get("rss_delta_mb"),
        peak_rss_before_mb=meta.get("peak_rss_before_mb"),
        peak_rss_after_mb=meta.get("peak_rss_after_mb"),
        peak_rss_delta_mb=meta.get("peak_rss_delta_mb"),
        final_recommended_arm=meta.get("final_recommended_arm"),
        final_recommended_arm_id=meta.get("final_recommended_arm_id"),
        final_recommended_mean=meta.get("final_recommended_mean"),
        final_recommended_model_name=meta.get("final_recommended_model_name"),
        final_recommended_prompt_name=meta.get("final_recommended_prompt_name"),
        final_recommended_prompt_type=meta.get("final_recommended_prompt_type"),
        final_recommended_features_json=meta.get("final_recommended_features_json"),
        final_selected_arm=meta.get("final_selected_arm"),
        final_selected_model_name=meta.get("final_selected_model_name"),
        global_best_arm=meta.get("global_best_arm"),
        global_best_mean=meta.get("global_best_mean"),
        global_best_model_name=meta.get("global_best_model_name"),
        global_best_prompt_name=meta.get("global_best_prompt_name"),
        global_best_features_json=meta.get("global_best_features_json"),
    )
    run.summary.update(summary)
    if save_files:
        for path in (trace_path, meta_path, fig_eval_path, fig_cost_path):
            if path.is_file():
                wandb.save(str(path), base_path=str(path.parent))

def main() -> int:
    args = parse_args()
    variant_cfg = resolve_variant_from_args(args)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(int(args.seed))

    data = load_bo_inputs(args.bo_inputs, dtype=dtype)
    requested_n_init = data.default_n_init if args.n_init is None else int(args.n_init)
    effective_cost_aware = bool(variant_cfg.cost_aware)
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
    variant = str(variant_cfg.raw)

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

    run = init_wandb_run(
        args,
        data=data,
        variant=variant,
        n_init_value=n_init_value,
        n_steps_value=n_steps_value,
        n_steps_rule=n_steps_rule,
        effective_cost_aware=effective_cost_aware,
        budget_original_cost=budget_original_cost,
    )

    rss_before_b = current_rss_bytes()
    peak_before_b = peak_rss_bytes()
    t0 = time.perf_counter()
    try:
        sim = run_bo(args, data, n_init=n_init_value, n_steps=n_steps_value)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise

    total_wall_time_s = float(time.perf_counter() - t0)
    rss_after_b = current_rss_bytes()
    peak_after_b = peak_rss_bytes()
    memory_summary = {
        "rss_before_mb": bytes_to_mb(rss_before_b),
        "rss_after_mb": bytes_to_mb(rss_after_b),
        "rss_delta_mb": None if rss_before_b is None or rss_after_b is None else bytes_to_mb(rss_after_b - rss_before_b),
        "peak_rss_before_mb": bytes_to_mb(peak_before_b),
        "peak_rss_after_mb": bytes_to_mb(peak_after_b),
        "peak_rss_delta_mb": None if peak_before_b is None or peak_after_b is None else bytes_to_mb(peak_after_b - peak_before_b),
        "peak_rss_gb": bytes_to_gb(peak_after_b),
        "extra_peak_memory_gb": None if peak_before_b is None or peak_after_b is None else bytes_to_gb(peak_after_b - peak_before_b),
    }

    timing_summaries: dict[str, Any] = {}
    timing_summaries.update(timing_summary(sim["iter_fit_s"], "iter_fit"))
    timing_summaries.update(timing_summary(sim["iter_score_s"], "iter_score"))
    timing_summaries.update(timing_summary(sim["iter_step_s"], "iter_step"))
    timing_summaries.update(timing_summary(sim["iter_total_s"], "iter_total"))
    content_summary = dataset_content_summary(data)

    final_recommended_arm = int(sim["recommended_arm"][-1]) if sim["recommended_arm"] else None
    final_selected_arm = int(sim["selected_arm"][-1]) if sim["selected_arm"] else None
    final_recommended_content = (
        arm_content_dict(data, final_recommended_arm) if final_recommended_arm is not None else {}
    )
    final_selected_content = (
        arm_content_dict(data, final_selected_arm) if final_selected_arm is not None else {}
    )

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
        raw_experiment_variant=str(args.experiment_variant or variant),
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
        selected_arm=np.asarray(sim["selected_arm"], dtype=np.int32),
        observed_y=np.asarray(sim["observed_y"], dtype=np.float32),
        acquisition_value=np.asarray(sim["acquisition_value"], dtype=np.float64),
        selection_phase=np.asarray(sim["selection_phase"], dtype="<U32"),
        iter_fit_s=np.asarray(sim["iter_fit_s"], dtype=np.float64),
        iter_score_s=np.asarray(sim["iter_score_s"], dtype=np.float64),
        iter_step_s=np.asarray(sim["iter_step_s"], dtype=np.float64),
        iter_total_s=np.asarray(sim["iter_total_s"], dtype=np.float64),
        iter_rss_mb=np.asarray(sim["iter_rss_mb"], dtype=np.float64),
        iter_peak_rss_mb=np.asarray(sim["iter_peak_rss_mb"], dtype=np.float64),
        selected_cost=np.asarray(sim["selected_cost"], dtype=np.float64),
        recommended_cost=np.asarray(sim["recommended_cost"], dtype=np.float64),
        selected_feature_json=np.asarray(sim["selected_feature_json"], dtype="<U1024"),
        recommended_feature_json=np.asarray(sim["recommended_feature_json"], dtype="<U1024"),
        selected_model_name=np.asarray(sim["selected_model_name"], dtype="<U128"),
        recommended_model_name=np.asarray(sim["recommended_model_name"], dtype="<U128"),
        selected_prompt_name=np.asarray(sim["selected_prompt_name"], dtype="<U128"),
        recommended_prompt_name=np.asarray(sim["recommended_prompt_name"], dtype="<U128"),
        selected_prompt_type=np.asarray(sim["selected_prompt_type"], dtype="<U64"),
        recommended_prompt_type=np.asarray(sim["recommended_prompt_type"], dtype="<U64"),
        selected_temperature=np.asarray(sim["selected_temperature"], dtype=np.float64),
        recommended_temperature=np.asarray(sim["recommended_temperature"], dtype=np.float64),
        selected_max_tokens=np.asarray(sim["selected_max_tokens"], dtype=np.int32),
        recommended_max_tokens=np.asarray(sim["recommended_max_tokens"], dtype=np.int32),
        feature_names=np.asarray(data.feature_names, dtype="<U64"),
        model_names=np.asarray([] if data.model_names is None else data.model_names, dtype="<U128"),
        prompt_names=np.asarray([] if data.prompt_names is None else data.prompt_names, dtype="<U128"),
        prompt_types=np.asarray([] if data.prompt_types is None else data.prompt_types, dtype="<U64"),
        temperatures=np.asarray([] if data.temperatures is None else data.temperatures, dtype=np.float64),
        max_tokens=np.asarray([] if data.max_tokens is None else data.max_tokens, dtype=np.int32),
        cost_per_arm_original=np.asarray(data.cost.numpy(), dtype=np.float64),
        rss_before_mb=np.asarray(np.nan if memory_summary["rss_before_mb"] is None else memory_summary["rss_before_mb"], dtype=np.float64),
        rss_after_mb=np.asarray(np.nan if memory_summary["rss_after_mb"] is None else memory_summary["rss_after_mb"], dtype=np.float64),
        peak_rss_gb=np.asarray(np.nan if memory_summary["peak_rss_gb"] is None else memory_summary["peak_rss_gb"], dtype=np.float64),
        extra_peak_memory_gb=np.asarray(np.nan if memory_summary["extra_peak_memory_gb"] is None else memory_summary["extra_peak_memory_gb"], dtype=np.float64),
    )

    final = {
        "final_simple_regret": float(sim["regret"][-1]) if sim["regret"] else None,
        "best_seen_regret": float(min(sim["regret"])) if sim["regret"] else None,
        "final_cum_eval": int(sim["x"][-1]) if sim["x"] else None,
        "final_cum_original_cost": float(sim["x_original_cost"][-1]) if sim["x_original_cost"] else None,
        "num_evaluated_configs": len(sim["selected_arm"]),
        "total_wall_time_s": float(total_wall_time_s),
    }
    meta = {
        "bo_inputs": str(args.bo_inputs),
        "dataset_tag": data.dataset,
        "matrix_seed": data.matrix_seed,
        "run_seed": int(args.seed),
        "experiment_variant": variant,
        "raw_experiment_variant": str(args.experiment_variant or variant),
        "policy_variant": variant,
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
        **content_summary,
        **timing_summaries,
        **memory_summary,
        "final_recommended_arm": final_recommended_arm,
        "final_recommended_arm_id": final_recommended_content.get("arm_id"),
        "final_recommended_mean": final_recommended_content.get("score_y"),
        "final_recommended_cost": final_recommended_content.get("full_eval_cost"),
        "final_recommended_model_name": final_recommended_content.get("model_name"),
        "final_recommended_prompt_name": final_recommended_content.get("prompt_name"),
        "final_recommended_prompt_type": final_recommended_content.get("prompt_type"),
        "final_recommended_temperature": final_recommended_content.get("temperature"),
        "final_recommended_max_tokens": final_recommended_content.get("max_tokens"),
        "final_recommended_features_json": compact_json(final_recommended_content.get("features", {})),
        "final_selected_arm": final_selected_arm,
        "final_selected_arm_id": final_selected_content.get("arm_id"),
        "final_selected_mean": final_selected_content.get("score_y"),
        "final_selected_cost": final_selected_content.get("full_eval_cost"),
        "final_selected_model_name": final_selected_content.get("model_name"),
        "final_selected_prompt_name": final_selected_content.get("prompt_name"),
        "final_selected_prompt_type": final_selected_content.get("prompt_type"),
        "final_selected_temperature": final_selected_content.get("temperature"),
        "final_selected_max_tokens": final_selected_content.get("max_tokens"),
        "final_selected_features_json": compact_json(final_selected_content.get("features", {})),
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

    meta["total_wall_time_s"] = total_wall_time_s
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.log_step_metrics:
        log_wandb_history(run, sim)
    update_wandb_summary_and_files(
        run,
        final=final,
        meta=meta,
        total_wall_time_s=total_wall_time_s,
        trace_path=trace_path,
        meta_path=meta_path,
        fig_eval_path=fig_eval_path,
        fig_cost_path=fig_cost_path,
        save_files=bool(args.wandb_save_files),
    )
    if run is not None:
        run.finish()

    print(f"Wrote trace: {trace_path}")
    print(f"Wrote meta: {meta_path}")
    print(f"Wrote figure: {fig_eval_path}")
    print(f"Wrote cost figure: {fig_cost_path}")
    print(f"final_simple_regret={final['final_simple_regret']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
