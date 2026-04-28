#!/usr/bin/env python3
"""Simulate matrix-bandit exploration and save traces (no plotting).

Policies:
- UCB-E: arm selection by UCB bound, recommendation by empirical mean.
- Gittins: arm selection by Gittins index, recommendation by posterior mean E[θ_k | D_t].
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from banditeval.bandits import upper_confidence_bound_exploration

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from gittins_lookup import compute_roots_lookup_table  # noqa: E402
from gittins_policy import gittins_index_exploration  # noqa: E402
from gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior  # noqa: E402


@dataclass(frozen=True)
class Trace:
    x: list[int]
    x_original_cost: list[float]
    regret: list[float]
    recommended_arm: list[int]
    recommended_mean: list[float]


def _recommend_from_means(mus: torch.Tensor) -> int:
    """Argmax, treating NaN as -inf (UCB-E mus may be NaN for unobserved arms)."""
    mus = mus.detach()
    scores = torch.where(
        torch.isnan(mus), torch.full_like(mus, -float("inf")), mus.to(torch.float32)
    )
    if not torch.isfinite(scores).any():
        return 0
    return int(torch.argmax(scores).item())


def simulate_simple_regret(
    *,
    ground_truth: torch.Tensor,
    step: Callable[..., Any],
    step_kwargs: dict[str, Any],
    seed: int,
    max_evaluations: int,
    per_arm_original_cost: torch.Tensor,
) -> Trace:
    torch.manual_seed(int(seed))
    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    regrets: list[float] = []
    cum_evaluated: list[int] = []
    cum_original_cost: list[float] = []
    rec_arm: list[int] = []
    rec_mean: list[float] = []
    evaluated = 0
    total_original_cost = 0.0

    while evaluated < max_evaluations:
        out = step(obs, **step_kwargs)
        if out is None:
            break
        if isinstance(out, tuple):
            batch, mus = out
        else:
            batch, mus = out, None
        if batch is None:
            break
        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch

        pulled_arm = int(row_idx[0].item())
        unit_cost = float(per_arm_original_cost[pulled_arm].item())
        total_original_cost += unit_cost * float(n_batch)

        if mus is None:
            # Policies we use here always set return_mus=True; this is a safety fallback.
            mus = torch.nanmean(obs, dim=1)
        arm = _recommend_from_means(mus)
        simple_regret = mu_star - float(true_means[arm].item())

        regrets.append(float(simple_regret))
        cum_evaluated.append(int(evaluated))
        cum_original_cost.append(float(total_original_cost))
        rec_arm.append(int(arm))
        rec_mean.append(float(mus[arm].item()) if torch.isfinite(mus[arm]) else float("nan"))

    return Trace(
        x=cum_evaluated,
        x_original_cost=cum_original_cost,
        regret=regrets,
        recommended_arm=rec_arm,
        recommended_mean=rec_mean,
    )


def _is_configurations_pricing(data: object) -> bool:
    if not isinstance(data, dict) or "0" not in data:
        return False
    z = data["0"]
    return isinstance(z, dict) and "estimated_cost_per_1m_input_tokens" in z


def _json_configurations_to_1d(data: dict, n_arms: int) -> np.ndarray:
    out: list[float] = []
    for i in range(n_arms):
        row = data.get(str(i))
        if not isinstance(row, dict) or "estimated_cost_per_1m_input_tokens" not in row:
            raise ValueError(f"missing estimated_cost_per_1m_input_tokens for arm {i}")
        out.append(float(row["estimated_cost_per_1m_input_tokens"]))
    return np.asarray(out, dtype=np.float64)


def load_cost_vector(path: Path, n_arms: int) -> torch.Tensor:
    """Load per-arm costs (shape `(n_arms,)`) from `.npy` or `.json`."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.squeeze(np.load(path))
    elif suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if _is_configurations_pricing(data):
            arr = _json_configurations_to_1d(data, n_arms)
        elif isinstance(data, list):
            arr = np.asarray(data, dtype=np.float64)
        elif isinstance(data, dict) and "cost_per_arm" in data and isinstance(
            data["cost_per_arm"], list
        ):
            arr = np.asarray(data["cost_per_arm"], dtype=np.float64)
        else:
            raise ValueError("unsupported JSON cost format")
    else:
        raise ValueError("cost vector must be .npy or .json")
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.shape != (n_arms,):
        raise ValueError(f"cost vector must have shape (n_arms,) = ({n_arms},); got {arr.shape}")
    return torch.tensor(arr, dtype=torch.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True, help="Path to (n_arms, n_examples) .npy")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .npz path (will store ucb_* and gittins_* arrays).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-budget-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32, help="UCB-E batch size (examples per step).")
    p.add_argument(
        "--gittins-batch-size",
        type=int,
        default=32,
        help="Gittins batch size (examples per step).",
    )
    p.add_argument("--ucb-a", type=float, default=1.0, help="UCB-E exploration parameter a.")
    p.add_argument("--gittins-prior-mean", type=float, default=0.5)
    p.add_argument("--gittins-prior-variance", type=float, default=0.04)
    p.add_argument(
        "--cost-vector",
        type=Path,
        default=None,
        help="Optional per-arm costs (JSON or .npy). Homogeneous cost: use a vector of 1.0.",
    )
    p.add_argument(
        "--cost-scaling-factor",
        type=float,
        default=1e-4,
        help="Scale applied to per-arm costs inside the Gittins DP (see gittins_policy).",
    )
    p.add_argument(
        "--gittins-obs-noise-variance",
        type=float,
        default=None,
        help="If omitted, uses 1/(4*B) with B=--gittins-batch-size.",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["ucb", "gittins"],
        choices=["ucb", "gittins"],
    )
    args = p.parse_args()

    mat = np.load(args.matrix)
    ground_truth = torch.tensor(mat, dtype=torch.float32)
    n_arms, n_examples = ground_truth.shape
    max_evaluations = int(max(1, round(float(args.eval_budget_fraction) * n_arms * n_examples)))
    per_arm_original_cost = (
        load_cost_vector(args.cost_vector, n_arms)
        if args.cost_vector is not None
        else torch.ones((n_arms,), dtype=torch.float64)
    )

    out: dict[str, Any] = {
        "matrix": str(args.matrix),
        "seed": int(args.seed),
        "n_arms": int(n_arms),
        "n_examples": int(n_examples),
        "budget_evals": int(max_evaluations),
        "cost_scaling_factor": float(args.cost_scaling_factor),
    }

    if "ucb" in args.algorithms:
        tr = simulate_simple_regret(
            ground_truth=ground_truth,
            step=upper_confidence_bound_exploration,
            step_kwargs={
                "a": float(args.ucb_a),
                "batch_size": int(args.batch_size),
                "return_mus": True,
            },
            seed=int(args.seed),
            max_evaluations=max_evaluations,
            per_arm_original_cost=per_arm_original_cost,
        )
        out.update(
            ucb_x=np.asarray(tr.x, dtype=np.int32),
            ucb_x_original_cost=np.asarray(tr.x_original_cost, dtype=np.float64),
            ucb_regret=np.asarray(tr.regret, dtype=np.float32),
            ucb_recommended_arm=np.asarray(tr.recommended_arm, dtype=np.int32),
            ucb_recommended_mean=np.asarray(tr.recommended_mean, dtype=np.float32),
        )
    else:
        out.update(
            ucb_x=np.asarray([], dtype=np.int32),
            ucb_x_original_cost=np.asarray([], dtype=np.float64),
            ucb_regret=np.asarray([], dtype=np.float32),
            ucb_recommended_arm=np.asarray([], dtype=np.int32),
            ucb_recommended_mean=np.asarray([], dtype=np.float32),
        )

    if "gittins" in args.algorithms:
        B = int(args.gittins_batch_size)
        tau_sq = (
            float(args.gittins_obs_noise_variance)
            if args.gittins_obs_noise_variance is not None
            else 1.0 / (4.0 * float(B))
        )
        transition_stds = transition_stds_shrinking_gaussian_posterior(
            np.float32(float(args.gittins_prior_variance)), np.float32(tau_sq), int(n_examples)
        )
        dp_costs_per_arm = (
            torch.tensor(per_arm_original_cost.numpy(), dtype=torch.float32)
            * float(args.cost_scaling_factor)
        )
        roots = compute_roots_lookup_table(
            transition_stds=transition_stds,
            costs_per_arm=np.asarray(dp_costs_per_arm.numpy(), dtype=np.float32),
            n_points=int(2**10 + 1),
        )
        roots_torch = torch.tensor(np.array(roots), dtype=torch.float32)

        tr = simulate_simple_regret(
            ground_truth=ground_truth,
            step=gittins_index_exploration,
            step_kwargs={
                "prior_mean": float(args.gittins_prior_mean),
                "prior_variance": float(args.gittins_prior_variance),
                "obs_noise_variance": float(tau_sq),
                "cost_per_transition": torch.tensor(
                    per_arm_original_cost.numpy(), dtype=torch.float64
                ),
                "cost_scaling_factor": float(args.cost_scaling_factor),
                "batch_size": int(B),
                "return_mus": True,
                "use_batch_mean_gittins_dp": False,
                "roots_lookup_table": roots_torch,
            },
            seed=int(args.seed),
            max_evaluations=max_evaluations,
            per_arm_original_cost=per_arm_original_cost,
        )
        out.update(
            gittins_x=np.asarray(tr.x, dtype=np.int32),
            gittins_x_original_cost=np.asarray(tr.x_original_cost, dtype=np.float64),
            gittins_regret=np.asarray(tr.regret, dtype=np.float32),
            gittins_recommended_arm=np.asarray(tr.recommended_arm, dtype=np.int32),
            gittins_recommended_mean=np.asarray(tr.recommended_mean, dtype=np.float32),
            tau_sq_gittins=np.asarray(tau_sq, dtype=np.float32),
            gittins_batch_size=np.asarray(B, dtype=np.int32),
        )
    else:
        out.update(
            gittins_x=np.asarray([], dtype=np.int32),
            gittins_x_original_cost=np.asarray([], dtype=np.float64),
            gittins_regret=np.asarray([], dtype=np.float32),
            gittins_recommended_arm=np.asarray([], dtype=np.int32),
            gittins_recommended_mean=np.asarray([], dtype=np.float32),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

