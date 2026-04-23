#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from banditeval.bandits import (
    upper_confidence_bound_exploration,
    upper_confidence_bound_exploration_low_rank_factorization,
)

from plot_simple_regret import (
    _final_cum_cost,
    _resolve_gittins_prior,
    _timing_summary,
    experiment_specs,
    gittins_cost_to_meta_string,
    incumbent_from_empirical_means,
    incumbent_from_gittins_posterior,
    load_gittins_cost_vector_from_file,
    make_gittins_step_with_score_cache,
    make_round_robin_step,
    save_trace_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_COST_XLABEL = "Cumulative cost (USD per 1M input tokens)"

POLICIES = [
    "rr",
    "ucb",
    "lrf",
    "gittins_uniform_cost",
    "gittins_varying_cost",
]

POLICY_LABELS = {
    "rr": "Round-robin (sample mean)",
    "ucb": "UCB-E",
    "lrf": "UCB-E-LRF",
    "gittins_uniform_cost": "Gittins (uniform cost)",
    "gittins_varying_cost": "Gittins (varying cost)",
}

POLICY_FAMILY = {
    "rr": "rr",
    "ucb": "ucb",
    "lrf": "lrf",
    "gittins_uniform_cost": "gittins",
    "gittins_varying_cost": "gittins",
}

POLICY_COST_MODE = {
    "rr": "baseline",
    "ucb": "baseline",
    "lrf": "baseline",
    "gittins_uniform_cost": "uniform",
    "gittins_varying_cost": "varying",
}


def _json_safe_config(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = v
    return out


def _safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _append_suffixes(path: Path, *parts: str) -> Path:
    suffix = "__".join(_safe_token(p) for p in parts if p)
    return path.with_name(f"{path.stem}__{suffix}{path.suffix}")


def _trim_trace_from_cum_eval(
    xs: list[int], ys: list[float], min_x: int
) -> tuple[list[int], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x >= min_x]
    if not pairs:
        return [], []
    ox, oy = zip(*pairs)
    return list(ox), list(oy)


def _trim_cost_from_eval_threshold(
    xs_eval: list[int], xs_cost: list[float], min_x: int
) -> list[float]:
    return [c for x, c in zip(xs_eval, xs_cost) if x >= min_x]


def _infer_dataset_tag(args: argparse.Namespace) -> str:
    if args.dataset_tag:
        return _safe_token(args.dataset_tag)

    # Prefer a short tag derived from the experiment preset name.
    # Example: gsm8k_various_model -> gsm8k
    if args.experiment:
        exp = str(args.experiment)
        if "_" in exp:
            return _safe_token(exp.split("_")[0])
        return _safe_token(exp)

    # Fallback to matrix stem if no preset is available.
    return _safe_token(Path(args.matrix).stem)


def _build_run_name(args: argparse.Namespace) -> str:
    dataset_tag = _infer_dataset_tag(args)
    return f"{args.policy}_seed{args.seed}_{dataset_tag}"


def _define_wandb_metrics(
    run: wandb.sdk.wandb_run.Run,
    *,
    has_cost_axis: bool,
) -> None:
    # Default chart: simple_regret vs cum_eval
    run.define_metric("cum_eval")
    run.define_metric("simple_regret", step_metric="cum_eval")

    # Diagnostics
    run.define_metric("incumbent_arm", step_metric="cum_eval")
    run.define_metric("iter_total_s", step_metric="cum_eval")
    run.define_metric("iter_step_s", step_metric="cum_eval")
    run.define_metric("batch_cells", step_metric="cum_eval")

    # Cost is logged as an alternative x-axis for dashboard use.
    if has_cost_axis:
        run.define_metric("cum_original_cost")


def _make_live_logger(
    run: wandb.sdk.wandb_run.Run | None,
    *,
    log_after_cum_eval: int = 0,
    has_cost_axis: bool,
) -> Callable[[dict[str, Any]], None] | None:
    if run is None:
        return None

    def _logger(row: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "cum_eval": row["cum_eval"],
            "incumbent_arm": row["incumbent_arm"],
            "iter_total_s": row["iter_total_s"],
            "iter_step_s": row["iter_step_s"],
            "batch_cells": row["batch_cells"],
        }

        if has_cost_axis and row.get("cum_original_cost") is not None:
            payload["cum_original_cost"] = row["cum_original_cost"]

        # For LRF, only start the displayed simple_regret after warm-up,
        # matching the existing figure semantics.
        if row["cum_eval"] >= log_after_cum_eval:
            payload["simple_regret"] = row["simple_regret"]

        run.log(payload)

    return _logger


def simulate_live_single(
    ground_truth: torch.Tensor,
    step: Callable[..., torch.Tensor | None],
    *,
    step_kwargs: dict[str, Any],
    seed: int,
    max_evaluations: int,
    verbose: bool = False,
    log_prefix: str = "",
    pass_sim_cum_eval: bool = False,
    per_arm_original_cost: torch.Tensor | None = None,
    incumbent_fn: Callable[[torch.Tensor], int] | None = None,
    step_logger: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[float], list[int], dict[str, Any]]:
    if per_arm_original_cost is not None and tuple(per_arm_original_cost.shape) != (
        ground_truth.shape[0],
    ):
        raise ValueError(
            "per_arm_original_cost must have shape (n_arms,) = "
            f"({ground_truth.shape[0]},); got {tuple(per_arm_original_cost.shape)}"
        )

    torch.manual_seed(seed)

    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    regrets: list[float] = []
    cum_evaluated: list[int] = []
    cum_original_cost: list[float] = []

    evaluated = 0
    total_original_cost = 0.0
    tag = log_prefix or "sim"

    iter_total_s: list[float] = []
    iter_step_s: list[float] = []

    while evaluated < max_evaluations:
        t_iter0 = time.time()
        call_kw = dict(step_kwargs)
        if pass_sim_cum_eval:
            call_kw["sim_cum_eval"] = evaluated

        t_step0 = time.time()
        batch = step(obs, **call_kw)
        t_step1 = time.time()

        if batch is None:
            break

        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch

        if per_arm_original_cost is not None:
            pulled_arm = int(row_idx[0].item())
            unit = float(per_arm_original_cost[pulled_arm].item())
            total_original_cost += unit * n_batch
            cum_original_cost.append(total_original_cost)

        if incumbent_fn is not None:
            arm = incumbent_fn(obs)
        else:
            arm = incumbent_from_empirical_means(obs)

        mu_sel = float(true_means[arm].item())
        simple_regret = mu_star - mu_sel

        regrets.append(simple_regret)
        cum_evaluated.append(evaluated)

        t_iter1 = time.time()
        iter_total = float(t_iter1 - t_iter0)
        iter_step = float(t_step1 - t_step0)
        iter_total_s.append(iter_total)
        iter_step_s.append(iter_step)

        if verbose:
            distinct_arms = sorted({int(x) for x in row_idx.reshape(-1).tolist()})
            print(
                f"[{tag}] step {len(regrets)}: cum_eval={evaluated} "
                f"batch_arms(distinct)={distinct_arms} incumbent_arm={arm} "
                f"simple_regret={simple_regret:.6f}",
                flush=True,
            )

        if step_logger is not None:
            step_logger(
                {
                    "cum_eval": evaluated,
                    "cum_original_cost": total_original_cost if per_arm_original_cost is not None else None,
                    "simple_regret": simple_regret,
                    "incumbent_arm": arm,
                    "iter_total_s": iter_total,
                    "iter_step_s": iter_step,
                    "batch_cells": n_batch,
                }
            )

    timing: dict[str, Any] = {
        "clock": "time.time",
        "unit": "seconds",
        "iter_total_s": iter_total_s,
        "iter_step_s": iter_step_s,
    }
    if per_arm_original_cost is not None:
        timing["cum_original_cost"] = cum_original_cost

    return regrets, cum_evaluated, timing


def _save_single_policy_figure(
    out: Path,
    *,
    x: list[float] | list[int],
    y: list[float],
    xlabel: str,
    matrix_name: str,
    subtitle: str,
    label: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if y:
        plt.plot(x, y, linewidth=1.75, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Simple regret")
    plt.title(f"{label} — {matrix_name}\n{subtitle}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-policy live W&B runner: one run = one policy."
    )
    specs = experiment_specs(REPO_ROOT)

    parser.add_argument(
        "--experiment",
        type=str,
        choices=sorted(specs.keys()),
        default="gsm8k_various_model",
    )
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--dataset-tag", type=str, default=None, help="Short dataset tag used in auto run naming, e.g. gsm8k")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--out-cost",
        type=Path,
        default=None,
        help="Optional separate PNG for the cost-axis figure; default is <out_stem>_cost.png",
    )
    parser.add_argument("--traces-out", type=Path, default=None)

    parser.add_argument(
        "--policy",
        type=str,
        choices=POLICIES,
        required=True,
        help="Exactly one policy per run.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-budget-fraction",
        type=float,
        default=0.10,
        help="Always interpreted as fraction of total matrix cells for this single-policy runner.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gittins-batch-size", type=int, default=20)
    parser.add_argument("--ucb-a", type=int, default=1, dest="a")
    parser.add_argument("--warmup-percentage", type=float, default=0.05)
    parser.add_argument("--lrf-device", type=str, default="cpu")
    parser.add_argument("--gittins-grid-points", type=int, default=2**10 + 1)
    parser.add_argument("--gittins-prior-mean", type=float, default=None)
    parser.add_argument("--gittins-prior-variance", type=float, default=None)
    parser.add_argument("--gittins-per-cell-dp", action="store_true")

    parser.add_argument(
        "--gittins-cost-vector",
        type=Path,
        default=None,
        help="Required for --policy gittins_varying_cost; ignored otherwise.",
    )
    parser.add_argument(
        "--original-cost-vector",
        type=Path,
        default=None,
        help="Optional true-cost vector to log cum_original_cost for dashboard x-axis switching.",
    )

    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional manual override; if omitted, auto name is used.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument(
        "--keep-out-exact",
        action="store_true",
        help="Do not append policy/run id to output paths",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = experiment_specs(REPO_ROOT)
    spec = specs[args.experiment]

    if args.matrix is None:
        args.matrix = spec.matrix
    if args.out is None:
        args.out = spec.out

    if not args.matrix.is_file():
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    if args.batch_size <= 0:
        print("--batch-size must be positive", file=sys.stderr)
        return 1
    if args.gittins_batch_size <= 0:
        print("--gittins-batch-size must be positive", file=sys.stderr)
        return 1

    gt_np = np.load(args.matrix)
    if gt_np.ndim != 2:
        print(f"Expected a 2D matrix, got shape {gt_np.shape}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms = int(ground_truth.shape[0])
    n_examples = int(ground_truth.shape[1])
    n_cells = int(ground_truth.numel())

    budget_max_evals = max(1, int(round(args.eval_budget_fraction * n_cells)))
    warmup_evals = int(np.ceil(args.warmup_percentage * n_cells))

    if args.policy == "lrf" and warmup_evals >= budget_max_evals:
        print(
            "Warm-up threshold must be < evaluation budget; "
            "raise --eval-budget-fraction or lower --warmup-percentage.",
            file=sys.stderr,
        )
        return 1

    gittins_prior_mean, gittins_prior_variance = _resolve_gittins_prior(spec, args)
    tau_sq_gittins = 1.0 / (4.0 * float(args.gittins_batch_size))

    # Cost used by the Gittins policy decision
    if args.policy == "gittins_varying_cost":
        if args.gittins_cost_vector is None:
            print("--policy gittins_varying_cost requires --gittins-cost-vector", file=sys.stderr)
            return 1
        try:
            policy_cost_tensor = load_gittins_cost_vector_from_file(args.gittins_cost_vector, n_arms)
        except Exception as e:
            print(f"gittins cost vector: {e}", file=sys.stderr)
            return 1
    else:
        policy_cost_tensor = torch.full((n_arms,), 1.0, dtype=torch.float64)

    # Cost only for logging/plotting x = cumulative cost
    original_cost_tensor: torch.Tensor | None = None
    if args.original_cost_vector is not None:
        try:
            original_cost_tensor = load_gittins_cost_vector_from_file(args.original_cost_vector, n_arms)
        except Exception as e:
            print(f"original cost vector: {e}", file=sys.stderr)
            return 1
    elif args.policy == "gittins_varying_cost" and args.gittins_cost_vector is not None:
        # If not provided separately, reuse varying-cost vector for cost-axis logging.
        original_cost_tensor = policy_cost_tensor

    has_cost_axis = original_cost_tensor is not None

    # Auto run naming
    auto_run_name = _build_run_name(args)
    run_name = args.wandb_name if args.wandb_name else auto_run_name

    run: wandb.sdk.wandb_run.Run | None = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            job_type="simple_regret_single_policy",
            config=_json_safe_config(
                {
                    **vars(args),
                    "policy_family": POLICY_FAMILY[args.policy],
                    "policy_cost_mode": POLICY_COST_MODE[args.policy],
                    "budget_mode": "evaluation_budget_only",
                    "budget_max_evals": budget_max_evals,
                    "dataset_tag_resolved": _infer_dataset_tag(args),
                    "auto_run_name": auto_run_name,
                    "run_name_resolved": run_name,
                }
            ),
            mode=args.wandb_mode,
        )

    if run is not None and not args.keep_out_exact:
        args.out = _append_suffixes(args.out, args.policy, run.id)
        if args.out_cost is not None:
            args.out_cost = _append_suffixes(args.out_cost, args.policy, run.id)
        if args.traces_out is not None:
            args.traces_out = _append_suffixes(args.traces_out, args.policy, run.id)
    elif not args.keep_out_exact:
        args.out = _append_suffixes(args.out, args.policy)

    if args.out_cost is None and has_cost_axis:
        args.out_cost = args.out.with_name(f"{args.out.stem}_cost{args.out.suffix}")
    if args.traces_out is None:
        args.traces_out = args.out.with_name(f"{args.out.stem}_traces.npz")

    if run is not None:
        run.config.update(
            {
                "resolved_matrix": str(args.matrix.resolve()),
                "resolved_out": str(args.out.resolve()),
                "resolved_out_cost": str(args.out_cost.resolve()) if args.out_cost else None,
                "resolved_traces_out": str(args.traces_out.resolve()),
                "gittins_prior_mean_resolved": gittins_prior_mean,
                "gittins_prior_variance_resolved": gittins_prior_variance,
            },
            allow_val_change=True,
        )
        _define_wandb_metrics(run, has_cost_axis=has_cost_axis)

    policy = args.policy
    label = POLICY_LABELS[policy]

    step: Callable[..., torch.Tensor | None]
    step_kwargs: dict[str, Any]
    incumbent_fn: Callable[[torch.Tensor], int] | None = None
    pass_sim_cum_eval = False
    log_after_cum_eval = 0

    if policy == "rr":
        step = make_round_robin_step(batch_size=args.batch_size)
        step_kwargs = {}
    elif policy == "ucb":
        step = upper_confidence_bound_exploration
        step_kwargs = {"a": args.a, "batch_size": args.batch_size, "return_mus": False}
    elif policy == "lrf":
        step = upper_confidence_bound_exploration_low_rank_factorization
        step_kwargs = {
            "a": args.a,
            "batch_size": args.batch_size,
            "return_mus": False,
            "warmup_percentage": args.warmup_percentage,
            "device": args.lrf_device,
        }
        log_after_cum_eval = warmup_evals
    elif policy in ("gittins_uniform_cost", "gittins_varying_cost"):
        step = make_gittins_step_with_score_cache(
            batch_size=args.gittins_batch_size,
            return_mus=False,
            obs_noise_variance=tau_sq_gittins,
            cost_per_transition=policy_cost_tensor,
            n_gittins_grid_points=args.gittins_grid_points,
            prior_mean=gittins_prior_mean,
            prior_variance=gittins_prior_variance,
            use_batch_mean_gittins_dp=not args.gittins_per_cell_dp,
            allow_early_stop=False,
            natural_stop_cum_eval_holder=[None],
        )
        step_kwargs = {}
        incumbent_fn = lambda obs: incumbent_from_gittins_posterior(
            obs,
            prior_mean=gittins_prior_mean,
            prior_variance=gittins_prior_variance,
            tau_sq=tau_sq_gittins,
        )
        pass_sim_cum_eval = True
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    regrets, xs_eval, timing = simulate_live_single(
        ground_truth,
        step,
        step_kwargs=step_kwargs,
        seed=args.seed,
        max_evaluations=budget_max_evals,
        verbose=args.verbose,
        log_prefix=policy,
        pass_sim_cum_eval=pass_sim_cum_eval,
        per_arm_original_cost=original_cost_tensor,
        incumbent_fn=incumbent_fn,
        step_logger=_make_live_logger(
            run,
            log_after_cum_eval=log_after_cum_eval,
            has_cost_axis=has_cost_axis,
        ),
    )

    xs_cost = list(timing.get("cum_original_cost", []))

    # Plot semantics: LRF figure starts after warm-up, matching current project convention.
    plot_x_eval = xs_eval
    plot_regret = regrets
    plot_x_cost = xs_cost
    if policy == "lrf":
        plot_x_eval, plot_regret = _trim_trace_from_cum_eval(xs_eval, regrets, warmup_evals)
        plot_x_cost = _trim_cost_from_eval_threshold(xs_eval, xs_cost, warmup_evals)

    budget_str = f"eval budget={args.eval_budget_fraction:.0%} of {n_cells} cells (cap={budget_max_evals})"
    subtitle = f"policy={policy}, seed={args.seed}, {budget_str}"

    _save_single_policy_figure(
        args.out,
        x=plot_x_eval,
        y=plot_regret,
        xlabel="Cumulative examples evaluated (matrix entries revealed)",
        matrix_name=args.matrix.name,
        subtitle=subtitle,
        label=label,
    )

    if has_cost_axis and args.out_cost is not None:
        _save_single_policy_figure(
            args.out_cost,
            x=plot_x_cost,
            y=plot_regret,
            xlabel=_COST_XLABEL,
            matrix_name=args.matrix.name,
            subtitle=subtitle,
            label=label,
        )

    # Save trace bundle in the same format as the current project,
    # but populate only the selected policy arrays.
    xs_rr: list[int] = []
    regrets_rr: list[float] = []
    xs_ucb: list[int] = []
    regrets_ucb: list[float] = []
    xs_lrf: list[int] = []
    regrets_lrf: list[float] = []
    xs_lrf_plot: list[int] = []
    regrets_lrf_plot: list[float] = []
    xs_gittins: list[int] = []
    regrets_gittins: list[float] = []

    xs_rr_cost: list[float] = []
    xs_ucb_cost: list[float] = []
    xs_lrf_cost: list[float] = []
    xs_lrf_plot_cost: list[float] = []
    xs_gittins_cost: list[float] = []

    if policy == "rr":
        xs_rr, regrets_rr, xs_rr_cost = xs_eval, regrets, xs_cost
    elif policy == "ucb":
        xs_ucb, regrets_ucb, xs_ucb_cost = xs_eval, regrets, xs_cost
    elif policy == "lrf":
        xs_lrf, regrets_lrf, xs_lrf_cost = xs_eval, regrets, xs_cost
        xs_lrf_plot, regrets_lrf_plot, xs_lrf_plot_cost = plot_x_eval, plot_regret, plot_x_cost
    elif policy in ("gittins_uniform_cost", "gittins_varying_cost"):
        xs_gittins, regrets_gittins, xs_gittins_cost = xs_eval, regrets, xs_cost

    meta = {
        "matrix": str(args.matrix.resolve()),
        "figure": str(args.out.resolve()),
        "figure_cost": str(args.out_cost.resolve()) if args.out_cost else None,
        "traces": str(args.traces_out.resolve()),
        "seed": args.seed,
        "policy": args.policy,
        "policy_family": POLICY_FAMILY[args.policy],
        "policy_cost_mode": POLICY_COST_MODE[args.policy],
        "batch_size": args.batch_size,
        "gittins_batch_size": args.gittins_batch_size,
        "eval_budget_fraction": args.eval_budget_fraction,
        "budget_mode": "evaluation_budget_only",
        "budget_max_evals": budget_max_evals,
        "ucb_a": args.a,
        "warmup_percentage": args.warmup_percentage,
        "lrf_device": args.lrf_device,
        "gittins_grid_points": args.gittins_grid_points,
        "gittins_policy_cost": gittins_cost_to_meta_string(policy_cost_tensor),
        "gittins_cost_vector_file": str(args.gittins_cost_vector.resolve()) if args.gittins_cost_vector else None,
        "original_cost_vector_file": str(args.original_cost_vector.resolve()) if args.original_cost_vector else None,
        "experiment": args.experiment,
        "dataset_tag_resolved": _infer_dataset_tag(args),
        "run_name_resolved": run_name,
        "gittins_prior_mean": gittins_prior_mean,
        "gittins_prior_variance": gittins_prior_variance,
        "gittins_per_cell_dp": args.gittins_per_cell_dp,
        "n_cells": n_cells,
        "title": f"{label} — {args.matrix.name}\n{subtitle}",
        "timing": {args.policy: timing},
    }

    save_trace_bundle(
        args.traces_out,
        xs_rr=xs_rr,
        regrets_rr=regrets_rr,
        xs_ucbe=xs_ucb,
        regrets_ucbe=regrets_ucb,
        xs_lrf=xs_lrf,
        regrets_lrf=regrets_lrf,
        xs_lrf_plot=xs_lrf_plot,
        regrets_lrf_plot=regrets_lrf_plot,
        xs_rr_original_cost=xs_rr_cost if xs_rr_cost else None,
        xs_ucbe_original_cost=xs_ucb_cost if xs_ucb_cost else None,
        xs_lrf_original_cost=xs_lrf_cost if xs_lrf_cost else None,
        xs_lrf_plot_original_cost=xs_lrf_plot_cost if xs_lrf_plot_cost else None,
        xs_gittins=xs_gittins,
        regrets_gittins=regrets_gittins,
        xs_gittins_original_cost=xs_gittins_cost if xs_gittins_cost else None,
        gittins_stop_cum_eval=None,
        warmup_evals=warmup_evals,
        budget_evals=budget_max_evals,
        tau_sq_gittins=tau_sq_gittins,
        meta=meta,
    )

    print(f"Wrote traces {args.traces_out} and {args.traces_out.with_suffix('.meta.json')}")
    print(f"Wrote eval figure {args.out}")
    if has_cost_axis and args.out_cost is not None:
        print(f"Wrote cost figure {args.out_cost}")

    s_total = _timing_summary(timing["iter_total_s"])
    print(
        f"{label}: {len(regrets)} batches, {xs_eval[-1] if xs_eval else 0} evals, "
        f"cum cost {(_final_cum_cost(timing) or 0.0):.6g}"
    )
    print(
        "  timing (per-iteration): "
        f"mean={s_total.get('mean_s', float('nan')):.6f}s "
        f"median={s_total.get('median_s', float('nan')):.6f}s "
        f"p90={s_total.get('p90_s', float('nan')):.6f}s"
    )

    if run is not None:
        run.summary["final_simple_regret"] = float(plot_regret[-1]) if plot_regret else None
        run.summary["final_cum_eval"] = int(xs_eval[-1]) if xs_eval else None
        run.summary["final_cum_original_cost"] = float(xs_cost[-1]) if xs_cost else None

        run.log({"eval_figure": wandb.Image(str(args.out))})
        if has_cost_axis and args.out_cost is not None and args.out_cost.is_file():
            run.log({"cost_figure": wandb.Image(str(args.out_cost))})

        artifact = wandb.Artifact(
            name=f"simple-regret-single-policy-{run.id}",
            type="experiment_outputs",
            description="Single-policy live W&B outputs",
        )
        artifact.add_file(str(args.out))
        if has_cost_axis and args.out_cost is not None and args.out_cost.is_file():
            artifact.add_file(str(args.out_cost))
        artifact.add_file(str(args.traces_out))
        artifact.add_file(str(args.traces_out.with_suffix(".meta.json")))
        run.log_artifact(artifact)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())