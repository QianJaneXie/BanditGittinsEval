#!/usr/bin/env python3
"""Simulate matrix-bandit exploration and save traces (no plotting).

Policies:
- UCB-E: arm selection by UCB bound, recommendation by empirical mean.
- SySRs: synchronized successive rejects (Smart-SR), recommend among active arms.
- Gittins: arm selection by Gittins index, recommendation by the posterior mean
  of the full fixed test-set average, with an optional posterior predictive std
  penalty for that same average.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from banditeval.bandits import upper_confidence_bound_exploration

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from gittins_lookup import compute_roots_lookup_table  # noqa: E402
from gittins_policy import gittins_index_exploration, gittins_post_pull_update  # noqa: E402
from gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior  # noqa: E402
from simple_regret_recommend import empirical_incumbent, posterior_incumbent  # noqa: E402
from sysrs_policy import make_sysrs_policy  # noqa: E402


@dataclass(frozen=True)
class Trace:
    x: list[int]
    x_original_cost: list[float]
    regret: list[float]
    recommended_arm: list[int]
    recommended_mean: list[float]
    stop_cum_eval: int | None = None
    stop_cum_original_cost: float | None = None
    recommendation_aware_stop_cum_eval: int | None = None
    recommendation_aware_stop_cum_original_cost: float | None = None


def simulate_simple_regret(
    *,
    ground_truth: torch.Tensor,
    step: Callable[..., Any],
    step_kwargs: dict[str, Any],
    seed: int,
    max_evaluations: int,
    per_arm_original_cost: torch.Tensor,
    max_original_cost: float | None = None,
    recommend_fn: Callable[[torch.Tensor, Any], tuple[int, torch.Tensor]] | None = None,
    pass_sim_cum_eval: bool = False,
    natural_stop_cum_eval_holder: list[int | None] | None = None,
    recommendation_aware_stop_cum_eval_holder: list[int | None] | None = None,
    post_pull_fn: Callable[[torch.Tensor, int, int], None] | None = None,
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
    stop_cum_eval: int | None = None
    stop_cum_original_cost: float | None = None
    recommendation_aware_stop_cum_eval: int | None = None
    recommendation_aware_stop_cum_original_cost: float | None = None

    while evaluated < max_evaluations:
        call_kw = dict(step_kwargs)
        if pass_sim_cum_eval:
            call_kw["sim_cum_eval"] = int(evaluated)
        if post_pull_fn is None:
            if natural_stop_cum_eval_holder is not None:
                call_kw["natural_stop_cum_eval_holder"] = natural_stop_cum_eval_holder
            if recommendation_aware_stop_cum_eval_holder is not None:
                call_kw["recommendation_aware_stop_cum_eval_holder"] = (
                    recommendation_aware_stop_cum_eval_holder
                )
        out = step(obs, **call_kw)
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
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch

        rows = row_idx.to(dtype=torch.long)
        total_original_cost += float(per_arm_original_cost[rows].sum().item())
        pulled_arm = int(rows[0].item())

        if post_pull_fn is not None:
            post_pull_fn(obs, pulled_arm, int(evaluated))
        if (
            stop_cum_eval is None
            and natural_stop_cum_eval_holder is not None
            and len(natural_stop_cum_eval_holder) == 1
            and natural_stop_cum_eval_holder[0] is not None
            and int(natural_stop_cum_eval_holder[0]) == int(evaluated)
        ):
            stop_cum_eval = int(natural_stop_cum_eval_holder[0])
            stop_cum_original_cost = float(total_original_cost)
        if (
            recommendation_aware_stop_cum_eval is None
            and recommendation_aware_stop_cum_eval_holder is not None
            and len(recommendation_aware_stop_cum_eval_holder) == 1
            and recommendation_aware_stop_cum_eval_holder[0] is not None
            and int(recommendation_aware_stop_cum_eval_holder[0]) == int(evaluated)
        ):
            recommendation_aware_stop_cum_eval = int(
                recommendation_aware_stop_cum_eval_holder[0]
            )
            recommendation_aware_stop_cum_original_cost = float(total_original_cost)

        if recommend_fn is None:
            arm, mus = empirical_incumbent(obs)
        else:
            arm, mus = recommend_fn(obs, aux)
        simple_regret = mu_star - float(true_means[arm].item())

        regrets.append(float(simple_regret))
        cum_evaluated.append(int(evaluated))
        cum_original_cost.append(float(total_original_cost))
        rec_arm.append(int(arm))
        rec_mean.append(float(mus[arm].item()) if torch.isfinite(mus[arm]) else float("nan"))

        if max_original_cost is not None and total_original_cost >= float(max_original_cost):
            break

    return Trace(
        x=cum_evaluated,
        x_original_cost=cum_original_cost,
        regret=regrets,
        recommended_arm=rec_arm,
        recommended_mean=rec_mean,
        stop_cum_eval=stop_cum_eval,
        stop_cum_original_cost=stop_cum_original_cost,
        recommendation_aware_stop_cum_eval=recommendation_aware_stop_cum_eval,
        recommendation_aware_stop_cum_original_cost=recommendation_aware_stop_cum_original_cost,
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
        help="Output .npz path (will store ucb_*, sysrs_*, and gittins_* arrays).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eval-budget-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of the full-evaluation budget. Without --cost-vector: "
            "stop after this fraction of matrix cells. With --cost-vector: stop after "
            "cumulative cost reaches this fraction of n_examples * sum_k c_k."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="UCB-E batch size (examples per step). Unused by SySRs.",
    )
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
        "--recommendation-std-penalty",
        type=float,
        default=0.0,
        help="Gittins recommendation: maximize the full fixed test-set average's "
        "posterior mean minus this nonnegative multiple of its posterior predictive "
        "std. Use 0 for mean only (default), 1 for mean - std.",
    )
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
        help="Observation noise variance τ² in Y|θ ~ N(θ, τ²). If omitted, uses 1/(4B) with "
        "B=--gittins-batch-size.",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["ucb", "sysrs", "gittins"],
        choices=["ucb", "sysrs", "gittins"],
    )
    args = p.parse_args()
    if (
        not math.isfinite(args.recommendation_std_penalty)
        or args.recommendation_std_penalty < 0.0
    ):
        p.error("--recommendation-std-penalty must be finite and nonnegative")

    mat = np.load(args.matrix)
    ground_truth = torch.tensor(mat, dtype=torch.float32)
    n_arms, n_examples = ground_truth.shape
    per_arm_original_cost = (
        load_cost_vector(args.cost_vector, n_arms)
        if args.cost_vector is not None
        else torch.ones((n_arms,), dtype=torch.float64)
    )
    cost_aware = args.cost_vector is not None
    total_brute_force_original_cost = float(n_examples * per_arm_original_cost.sum().item())
    if cost_aware:
        max_original_cost = float(args.eval_budget_fraction) * total_brute_force_original_cost
        max_evaluations = int(n_arms * n_examples)
    else:
        max_original_cost = None
        max_evaluations = int(
            max(1, round(float(args.eval_budget_fraction) * n_arms * n_examples))
        )

    out: dict[str, Any] = {
        "matrix": str(args.matrix),
        "seed": int(args.seed),
        "n_arms": int(n_arms),
        "n_examples": int(n_examples),
        "cost_aware": bool(cost_aware),
        "budget_evals": int(max_evaluations),
        "total_brute_force_original_cost": np.asarray(
            total_brute_force_original_cost, dtype=np.float64
        ),
        "budget_original_cost": np.asarray(
            -1.0 if max_original_cost is None else float(max_original_cost),
            dtype=np.float64,
        ),
        "cost_scaling_factor": float(args.cost_scaling_factor),
        "cost_per_arm_original": np.asarray(per_arm_original_cost.numpy(), dtype=np.float64),
        "ucb_a": float(args.ucb_a),
        "ucb_batch_size": int(args.batch_size),
        "gittins_prior_mean": float(args.gittins_prior_mean),
        "gittins_prior_variance": float(args.gittins_prior_variance),
        "recommendation_rule": (
            "finite_population_posterior_mean_minus_std"
            if args.recommendation_std_penalty > 0.0
            else "finite_population_posterior_mean"
        ),
        "recommendation_std_penalty": float(args.recommendation_std_penalty),
    }

    if "ucb" in args.algorithms:
        tr = simulate_simple_regret(
            ground_truth=ground_truth,
            step=upper_confidence_bound_exploration,
            step_kwargs={
                "a": float(args.ucb_a),
                "batch_size": int(args.batch_size),
                "return_mus": False,
            },
            seed=int(args.seed),
            max_evaluations=max_evaluations,
            per_arm_original_cost=per_arm_original_cost,
            max_original_cost=max_original_cost,
            recommend_fn=empirical_incumbent,
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

    if "sysrs" in args.algorithms:
        # Plan SySRs against the cell budget fraction (same as unit-cost max_evaluations).
        planned_budget = int(
            max(1, round(float(args.eval_budget_fraction) * n_arms * n_examples))
        )
        sysrs_step, sysrs_recommend, sysrs_state = make_sysrs_policy(
            n_arms=n_arms,
            n_examples=n_examples,
            planned_budget=planned_budget,
            seed=int(args.seed),
        )
        tr = simulate_simple_regret(
            ground_truth=ground_truth,
            step=sysrs_step,
            step_kwargs={},
            seed=int(args.seed),
            max_evaluations=max_evaluations,
            per_arm_original_cost=per_arm_original_cost,
            max_original_cost=max_original_cost,
            recommend_fn=sysrs_recommend,
        )
        out.update(
            sysrs_x=np.asarray(tr.x, dtype=np.int32),
            sysrs_x_original_cost=np.asarray(tr.x_original_cost, dtype=np.float64),
            sysrs_regret=np.asarray(tr.regret, dtype=np.float32),
            sysrs_recommended_arm=np.asarray(tr.recommended_arm, dtype=np.int32),
            sysrs_recommended_mean=np.asarray(tr.recommended_mean, dtype=np.float32),
            sysrs_planned_budget=np.asarray(sysrs_state.planned_budget, dtype=np.int32),
        )
    else:
        out.update(
            sysrs_x=np.asarray([], dtype=np.int32),
            sysrs_x_original_cost=np.asarray([], dtype=np.float64),
            sysrs_regret=np.asarray([], dtype=np.float32),
            sysrs_recommended_arm=np.asarray([], dtype=np.int32),
            sysrs_recommended_mean=np.asarray([], dtype=np.float32),
        )

    if "gittins" in args.algorithms:
        B = int(args.gittins_batch_size)
        tau_sq = (
            float(args.gittins_obs_noise_variance)
            if args.gittins_obs_noise_variance is not None
            else 1.0 / (4.0 * float(B))
        )
        # Batch observation model: user supplies τ² for the batch mean (default 1/(4B)).
        # Convert to an equivalent per-cell variance τ²_cell = τ² * B so the per-cell DP/lookup
        # matches the posterior updates in `gittins_index_exploration(batch_observation_model=True)`.
        tau_sq_cell = float(tau_sq) * float(B)
        transition_stds = transition_stds_shrinking_gaussian_posterior(
            np.float32(float(args.gittins_prior_variance)),
            np.float32(tau_sq_cell),
            int(n_examples),
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
        stop_holder: list[int | None] = [None]
        recommendation_aware_stop_holder: list[int | None] = [None]
        cached_scores = torch.full((n_arms,), float("inf"), dtype=torch.float32)
        prev_arm: int | None = None

        def gittins_step(obs: torch.Tensor, **_kwargs):
            nonlocal prev_arm
            recompute = None if prev_arm is None else [prev_arm]
            out = gittins_index_exploration(
                obs,
                prior_mean=float(args.gittins_prior_mean),
                prior_variance=float(args.gittins_prior_variance),
                obs_noise_variance=float(tau_sq),
                cost_per_transition=torch.tensor(
                    per_arm_original_cost.numpy(), dtype=torch.float64
                ),
                cost_scaling_factor=float(args.cost_scaling_factor),
                batch_size=int(B),
                return_mus=False,
                cached_scores=cached_scores,
                recompute_arms=recompute,
                use_batch_mean_gittins_dp=False,
                batch_observation_model=True,
                roots_lookup_table=roots_torch,
                allow_early_stop=False,
            )
            batch = out[0] if isinstance(out, tuple) else out
            if batch is not None:
                prev_arm = int(batch[0, 0].item())
            return out

        gittins_post_pull_kw = dict(
            prior_mean=float(args.gittins_prior_mean),
            prior_variance=float(args.gittins_prior_variance),
            obs_noise_variance=float(tau_sq),
            cost_per_transition=torch.tensor(
                per_arm_original_cost.numpy(), dtype=torch.float64
            ),
            cost_scaling_factor=float(args.cost_scaling_factor),
            n_gittins_grid_points=int(2**10 + 1),
            batch_size=int(B),
            use_batch_mean_gittins_dp=False,
            roots_lookup_table=roots_torch,
            batch_observation_model=True,
            natural_stop_cum_eval_holder=stop_holder,
            recommendation_aware_stop_cum_eval_holder=recommendation_aware_stop_holder,
            recommendation_std_penalty=float(args.recommendation_std_penalty),
        )

        def gittins_post_pull(obs: torch.Tensor, pulled_arm: int, cum_eval: int) -> None:
            gittins_post_pull_update(
                obs,
                cached_scores=cached_scores,
                recompute_arms=[int(pulled_arm)],
                sim_cum_eval=int(cum_eval),
                **gittins_post_pull_kw,
            )

        tr = simulate_simple_regret(
            ground_truth=ground_truth,
            step=gittins_step,
            step_kwargs={},
            seed=int(args.seed),
            max_evaluations=max_evaluations,
            per_arm_original_cost=per_arm_original_cost,
            max_original_cost=max_original_cost,
            recommend_fn=partial(
                posterior_incumbent,
                prior_mean=float(args.gittins_prior_mean),
                prior_variance=float(args.gittins_prior_variance),
                tau_sq_cell=float(tau_sq_cell),
                std_penalty=float(args.recommendation_std_penalty),
            ),
            pass_sim_cum_eval=False,
            natural_stop_cum_eval_holder=stop_holder,
            recommendation_aware_stop_cum_eval_holder=recommendation_aware_stop_holder,
            post_pull_fn=gittins_post_pull,
        )
        out.update(
            gittins_x=np.asarray(tr.x, dtype=np.int32),
            gittins_x_original_cost=np.asarray(tr.x_original_cost, dtype=np.float64),
            gittins_regret=np.asarray(tr.regret, dtype=np.float32),
            gittins_recommended_arm=np.asarray(tr.recommended_arm, dtype=np.int32),
            gittins_recommended_mean=np.asarray(tr.recommended_mean, dtype=np.float32),
            tau_sq_gittins=np.asarray(tau_sq, dtype=np.float32),
            gittins_batch_size=np.asarray(B, dtype=np.int32),
            gittins_stop_cum_eval=np.asarray(-1 if tr.stop_cum_eval is None else tr.stop_cum_eval, dtype=np.int32),
            gittins_stop_cum_original_cost=np.asarray(
                -1.0 if tr.stop_cum_original_cost is None else tr.stop_cum_original_cost,
                dtype=np.float64,
            ),
            gittins_recommendation_aware_stop_cum_eval=np.asarray(
                -1
                if tr.recommendation_aware_stop_cum_eval is None
                else tr.recommendation_aware_stop_cum_eval,
                dtype=np.int32,
            ),
            gittins_recommendation_aware_stop_cum_original_cost=np.asarray(
                -1.0
                if tr.recommendation_aware_stop_cum_original_cost is None
                else tr.recommendation_aware_stop_cum_original_cost,
                dtype=np.float64,
            ),
        )
    else:
        out.update(
            gittins_x=np.asarray([], dtype=np.int32),
            gittins_x_original_cost=np.asarray([], dtype=np.float64),
            gittins_regret=np.asarray([], dtype=np.float32),
            gittins_recommended_arm=np.asarray([], dtype=np.int32),
            gittins_recommended_mean=np.asarray([], dtype=np.float32),
            gittins_stop_cum_eval=np.asarray(-1, dtype=np.int32),
            gittins_stop_cum_original_cost=np.asarray(-1.0, dtype=np.float64),
            gittins_recommendation_aware_stop_cum_eval=np.asarray(-1, dtype=np.int32),
            gittins_recommendation_aware_stop_cum_original_cost=np.asarray(-1.0, dtype=np.float64),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
