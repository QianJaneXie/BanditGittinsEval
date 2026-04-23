#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

ALGORITHM_SETS: dict[str, list[str]] = {
    "full": ["rr", "ucb", "lrf", "gittins"],
    "no_lrf": ["rr", "ucb", "gittins"],
    "gittins_only": ["gittins"],
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


def _append_run_id(path: Path, run_id: str) -> Path:
    return path.with_name(f"{path.stem}__{run_id}{path.suffix}")


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


def _define_wandb_metrics(
    run: wandb.sdk.wandb_run.Run,
    *,
    algorithms: list[str],
    has_cost_axis: bool,
) -> None:
    # Shared x-axis for the combined eval plot
    run.define_metric("simple_regret_eval/x")
    for algo in algorithms:
        run.define_metric(f"simple_regret_eval/{algo}", step_metric="simple_regret_eval/x")

    # Shared x-axis for the combined cost plot
    if has_cost_axis:
        run.define_metric("simple_regret_cost/x")
        for algo in algorithms:
            run.define_metric(f"simple_regret_cost/{algo}", step_metric="simple_regret_cost/x")

    # Keep per-algorithm raw diagnostics too
    for algo in algorithms:
        run.define_metric(f"{algo}_cum_eval")
        run.define_metric(f"{algo}_incumbent_arm", step_metric=f"{algo}_cum_eval")
        run.define_metric(f"{algo}_iter_total_s", step_metric=f"{algo}_cum_eval")
        run.define_metric(f"{algo}_iter_step_s", step_metric=f"{algo}_cum_eval")
        run.define_metric(f"{algo}_batch_cells", step_metric=f"{algo}_cum_eval")

        if has_cost_axis:
            run.define_metric(f"{algo}_cum_original_cost")


def _make_live_logger(
    run: wandb.sdk.wandb_run.Run | None,
    *,
    algo: str,
    has_cost_axis: bool,
) -> Callable[[dict[str, Any]], None] | None:
    if run is None:
        return None

    def _logger(row: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            # shared x for the combined eval chart
            "simple_regret_eval/x": row["cum_eval"],
            f"simple_regret_eval/{algo}": row["simple_regret"],
            # per-algorithm diagnostics
            f"{algo}_cum_eval": row["cum_eval"],
            f"{algo}_incumbent_arm": row["incumbent_arm"],
            f"{algo}_iter_total_s": row["iter_total_s"],
            f"{algo}_iter_step_s": row["iter_step_s"],
            f"{algo}_batch_cells": row["batch_cells"],
        }

        if has_cost_axis and row.get("cum_original_cost") is not None:
            payload["simple_regret_cost/x"] = row["cum_original_cost"]
            payload[f"simple_regret_cost/{algo}"] = row["simple_regret"]
            payload[f"{algo}_cum_original_cost"] = row["cum_original_cost"]

        run.log(payload)

    return _logger


def simulate_live(
    ground_truth: torch.Tensor,
    step: Callable[..., torch.Tensor | None],
    *,
    step_kwargs: dict[str, Any],
    seed: int,
    max_evaluations: int | None = None,
    max_cumulative_cost: float | None = None,
    verbose: bool = False,
    log_prefix: str = "",
    pass_sim_cum_eval: bool = False,
    per_arm_original_cost: torch.Tensor | None = None,
    incumbent_fn: Callable[[torch.Tensor], int] | None = None,
    step_logger: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[float], list[int], dict[str, Any]]:
    if max_evaluations is None and max_cumulative_cost is None:
        raise ValueError("simulate_live requires max_evaluations and/or max_cumulative_cost")
    if max_cumulative_cost is not None and per_arm_original_cost is None:
        raise ValueError("max_cumulative_cost requires per_arm_original_cost")

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

    while True:
        if max_evaluations is not None and evaluated >= max_evaluations:
            break
        if max_cumulative_cost is not None and total_original_cost >= max_cumulative_cost:
            break

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


def _save_eval_figure(
    out: Path,
    *,
    matrix_name: str,
    subtitle: str,
    xs_rr: list[int],
    regrets_rr: list[float],
    xs_ucb: list[int],
    regrets_ucb: list[float],
    xs_lrf_plot: list[int],
    regrets_lrf_plot: list[float],
    xs_gittins: list[int],
    regrets_gittins: list[float],
    gittins_stop_cum_eval: int | None,
    gittins_per_cell_dp: bool,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    if regrets_rr:
        plt.plot(xs_rr, regrets_rr, label="Round-robin (sample mean)", linewidth=1.5)
    if regrets_ucb:
        plt.plot(xs_ucb, regrets_ucb, label="UCB-E", linewidth=1.5)
    if regrets_lrf_plot:
        plt.plot(xs_lrf_plot, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
    if regrets_gittins:
        gittins_label = (
            "Gittins (tau_sq = 1/(4B), per-cell DP)"
            if gittins_per_cell_dp
            else "Gittins (tau_sq = 1/(4B), batch-mean DP)"
        )
        (line_g,) = plt.plot(xs_gittins, regrets_gittins, label=gittins_label, linewidth=1.5)
        if gittins_stop_cum_eval is not None:
            plt.axvline(
                gittins_stop_cum_eval,
                color=line_g.get_color(),
                linestyle="--",
                alpha=0.85,
                linewidth=1.2,
                label=f"Gittins nominal stop ({gittins_stop_cum_eval} evals)",
            )

    plt.xlabel("Cumulative examples evaluated (matrix entries revealed)")
    plt.ylabel("Simple regret")
    plt.title(f"Simple regret — {matrix_name}\n{subtitle}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _save_cost_figure(
    out: Path,
    *,
    matrix_name: str,
    subtitle: str,
    xs_rr_cost: list[float],
    regrets_rr: list[float],
    xs_ucb_cost: list[float],
    regrets_ucb: list[float],
    xs_lrf_cost_plot: list[float],
    regrets_lrf_plot: list[float],
    xs_gittins_cost: list[float],
    regrets_gittins: list[float],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    if regrets_rr and xs_rr_cost:
        plt.plot(xs_rr_cost, regrets_rr, label="Round-robin (sample mean)", linewidth=1.5)
    if regrets_ucb and xs_ucb_cost:
        plt.plot(xs_ucb_cost, regrets_ucb, label="UCB-E", linewidth=1.5)
    if regrets_lrf_plot and xs_lrf_cost_plot:
        plt.plot(xs_lrf_cost_plot, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
    if regrets_gittins and xs_gittins_cost:
        plt.plot(xs_gittins_cost, regrets_gittins, label="Gittins (cost axis)", linewidth=1.5)

    plt.xlabel(_COST_XLABEL)
    plt.ylabel("Simple regret")
    plt.title(f"Simple regret vs cost — {matrix_name}\n{subtitle}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live W&B version of simple-regret simulation with shared x-axis fields."
    )
    specs = experiment_specs(REPO_ROOT)

    parser.add_argument(
        "--experiment",
        type=str,
        choices=sorted(specs.keys()),
        default="gsm8k_various_model",
    )
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--out-cost",
        type=Path,
        default=None,
        help="Optional separate PNG for the cost-axis figure; default is <out_stem>_cost.png",
    )
    parser.add_argument("--traces-out", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gittins-batch-size", type=int, default=20)
    parser.add_argument("--eval-budget-fraction", type=float, default=0.10)
    parser.add_argument("--ucb-a", type=int, default=1, dest="a")
    parser.add_argument("--warmup-percentage", type=float, default=0.05)
    parser.add_argument("--lrf-device", type=str, default="cpu")
    parser.add_argument("--gittins-grid-points", type=int, default=2**10 + 1)

    parser.add_argument(
        "--gittins-cost-mode",
        type=str,
        choices=["unaware", "aware"],
        default="unaware",
    )
    parser.add_argument(
        "--gittins-cost-vector",
        type=Path,
        default=None,
        help="Required for aware Gittins; reused as original-cost vector if --original-cost-vector is omitted",
    )
    parser.add_argument(
        "--original-cost-vector",
        type=Path,
        default=None,
        help="Optional: log cumulative original cost for all algorithms even when Gittins runs in cost-unaware mode",
    )
    parser.add_argument("--gittins-prior-mean", type=float, default=None)
    parser.add_argument("--gittins-prior-variance", type=float, default=None)
    parser.add_argument("--gittins-per-cell-dp", action="store_true")

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["rr", "ucb", "lrf", "gittins"],
        default=None,
        metavar="NAME",
    )
    parser.add_argument(
        "--algorithm-set",
        choices=sorted(ALGORITHM_SETS.keys()),
        default="full",
        help="Convenient preset for local runs and sweeps",
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument(
        "--keep-out-exact",
        action="store_true",
        help="Do not append the W&B run id to output paths",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = experiment_specs(REPO_ROOT)
    spec = specs[args.experiment]

    if args.algorithms is None:
        algorithms = ALGORITHM_SETS[args.algorithm_set]
    else:
        algorithms = list(dict.fromkeys(args.algorithms))

    if args.matrix is None:
        args.matrix = spec.matrix
    if args.out is None:
        args.out = spec.out

    gittins_prior_mean, gittins_prior_variance = _resolve_gittins_prior(spec, args)

    if args.batch_size <= 0:
        print("--batch-size must be positive", file=sys.stderr)
        return 1
    if args.gittins_batch_size <= 0:
        print("--gittins-batch-size must be positive", file=sys.stderr)
        return 1
    if not args.matrix.is_file():
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    run: wandb.sdk.wandb_run.Run | None = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            job_type="simple_regret_live",
            config=_json_safe_config(
                {
                    **vars(args),
                    "algorithms_resolved": algorithms,
                }
            ),
            mode=args.wandb_mode,
        )

    if run is not None and not args.keep_out_exact:
        args.out = _append_run_id(args.out, run.id)
        if args.out_cost is not None:
            args.out_cost = _append_run_id(args.out_cost, run.id)
        if args.traces_out is not None:
            args.traces_out = _append_run_id(args.traces_out, run.id)

    gt_np = np.load(args.matrix)
    if gt_np.ndim != 2:
        print(f"Expected a 2D matrix, got shape {gt_np.shape}", file=sys.stderr)
        if run is not None:
            run.finish(exit_code=1)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms = int(ground_truth.shape[0])
    n_examples = int(ground_truth.shape[1])
    n_cells = int(ground_truth.numel())

    cost_vector_path: Path | None = args.gittins_cost_vector
    if args.gittins_cost_mode == "aware":
        if "gittins" not in algorithms:
            print(
                "--gittins-cost-mode aware requires --algorithms to include gittins",
                file=sys.stderr,
            )
            if run is not None:
                run.finish(exit_code=1)
            return 1
        if cost_vector_path is None:
            print("--gittins-cost-mode aware requires --gittins-cost-vector", file=sys.stderr)
            if run is not None:
                run.finish(exit_code=1)
            return 1
        try:
            gittins_cost_tensor = load_gittins_cost_vector_from_file(cost_vector_path, n_arms)
        except Exception as e:
            print(f"gittins cost vector: {e}", file=sys.stderr)
            if run is not None:
                run.finish(exit_code=1)
            return 1
    else:
        gittins_cost_tensor = torch.full((n_arms,), 1.0, dtype=torch.float64)

    original_cost_tensor: torch.Tensor | None = None
    original_cost_path = args.original_cost_vector
    if original_cost_path is not None:
        try:
            original_cost_tensor = load_gittins_cost_vector_from_file(original_cost_path, n_arms)
        except Exception as e:
            print(f"original cost vector: {e}", file=sys.stderr)
            if run is not None:
                run.finish(exit_code=1)
            return 1
    elif args.gittins_cost_mode == "aware" and cost_vector_path is not None:
        original_cost_tensor = gittins_cost_tensor

    total_full_matrix_cost: float | None = None
    budget_max_cumulative_cost: float | None = None
    if args.gittins_cost_mode == "aware":
        total_full_matrix_cost = float(n_examples * gittins_cost_tensor.sum().item())
        budget_max_cumulative_cost = float(args.eval_budget_fraction) * total_full_matrix_cost
        budget_max_evals = n_cells
    else:
        budget_max_evals = max(1, int(round(args.eval_budget_fraction * n_cells)))

    warmup_evals = int(np.ceil(args.warmup_percentage * n_cells))
    if "lrf" in algorithms and budget_max_cumulative_cost is None and warmup_evals >= budget_max_evals:
        print(
            "Warm-up threshold (ceil(warmup %% × n)) must be < eval budget; "
            "raise --eval-budget-fraction or lower --warmup-percentage.",
            file=sys.stderr,
        )
        if run is not None:
            run.finish(exit_code=1)
        return 1

    if (
        "lrf" in algorithms
        and budget_max_cumulative_cost is not None
        and original_cost_tensor is not None
    ):
        worst_case_lrf_warmup_cost = float(warmup_evals) * float(original_cost_tensor.max().item())
        if worst_case_lrf_warmup_cost > float(budget_max_cumulative_cost):
            print(
                "Warning: worst-case LRF warm-up cost "
                "(warmup_evals x max arm cost) exceeds cost budget "
                f"{budget_max_cumulative_cost:.6g}; consider lowering --warmup-percentage.",
                file=sys.stderr,
            )

    tau_sq_gittins = 1.0 / (4.0 * float(args.gittins_batch_size))
    has_cost_axis = original_cost_tensor is not None

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
        _define_wandb_metrics(run, algorithms=algorithms, has_cost_axis=has_cost_axis)

    sim_kwargs: dict[str, Any] = {
        "seed": args.seed,
        "max_evaluations": budget_max_evals,
        "verbose": args.verbose,
        "per_arm_original_cost": original_cost_tensor,
    }
    if budget_max_cumulative_cost is not None:
        sim_kwargs["max_cumulative_cost"] = budget_max_cumulative_cost

    regrets_rr: list[float] = []
    xs_rr: list[int] = []
    timing_rr: dict[str, Any] | None = None
    xs_rr_original: list[float] = []

    regrets_ucb: list[float] = []
    xs_ucb: list[int] = []
    timing_ucb: dict[str, Any] | None = None
    xs_ucb_original: list[float] = []

    regrets_lrf: list[float] = []
    xs_lrf: list[int] = []
    timing_lrf: dict[str, Any] | None = None
    xs_lrf_original: list[float] = []

    regrets_gittins: list[float] = []
    xs_gittins: list[int] = []
    timing_gittins: dict[str, Any] | None = None
    xs_gittins_original: list[float] = []
    gittins_stop_cum_eval: int | None = None

    if "rr" in algorithms:
        regrets_rr, xs_rr, timing_rr = simulate_live(
            ground_truth,
            make_round_robin_step(batch_size=args.batch_size),
            step_kwargs={},
            log_prefix="rr",
            step_logger=_make_live_logger(run, algo="rr", has_cost_axis=has_cost_axis),
            **sim_kwargs,
        )
        xs_rr_original = list(timing_rr.get("cum_original_cost", []))

    if "ucb" in algorithms:
        regrets_ucb, xs_ucb, timing_ucb = simulate_live(
            ground_truth,
            upper_confidence_bound_exploration,
            step_kwargs={"a": args.a, "batch_size": args.batch_size, "return_mus": False},
            log_prefix="ucb",
            step_logger=_make_live_logger(run, algo="ucb", has_cost_axis=has_cost_axis),
            **sim_kwargs,
        )
        xs_ucb_original = list(timing_ucb.get("cum_original_cost", []))

    if "lrf" in algorithms:
        regrets_lrf, xs_lrf, timing_lrf = simulate_live(
            ground_truth,
            upper_confidence_bound_exploration_low_rank_factorization,
            step_kwargs={
                "a": args.a,
                "batch_size": args.batch_size,
                "return_mus": False,
                "warmup_percentage": args.warmup_percentage,
                "device": args.lrf_device,
            },
            log_prefix="lrf",
            step_logger=_make_live_logger(run, algo="lrf", has_cost_axis=has_cost_axis),
            **sim_kwargs,
        )
        xs_lrf_original = list(timing_lrf.get("cum_original_cost", []))

    if "gittins" in algorithms:
        gittins_natural_stop_holder: list[int | None] = [None]
        regrets_gittins, xs_gittins, timing_gittins = simulate_live(
            ground_truth,
            make_gittins_step_with_score_cache(
                batch_size=args.gittins_batch_size,
                return_mus=False,
                obs_noise_variance=tau_sq_gittins,
                cost_per_transition=gittins_cost_tensor,
                n_gittins_grid_points=args.gittins_grid_points,
                prior_mean=gittins_prior_mean,
                prior_variance=gittins_prior_variance,
                use_batch_mean_gittins_dp=not args.gittins_per_cell_dp,
                allow_early_stop=False,
                natural_stop_cum_eval_holder=gittins_natural_stop_holder,
            ),
            step_kwargs={},
            log_prefix="gittins",
            pass_sim_cum_eval=True,
            incumbent_fn=lambda obs: incumbent_from_gittins_posterior(
                obs,
                prior_mean=gittins_prior_mean,
                prior_variance=gittins_prior_variance,
                tau_sq=tau_sq_gittins,
            ),
            step_logger=_make_live_logger(run, algo="gittins", has_cost_axis=has_cost_axis),
            **sim_kwargs,
        )
        gittins_stop_cum_eval = gittins_natural_stop_holder[0]
        xs_gittins_original = list(timing_gittins.get("cum_original_cost", []))

    xs_lrf_plot, regrets_lrf_plot = _trim_trace_from_cum_eval(xs_lrf, regrets_lrf, warmup_evals)
    xs_lrf_plot_original = _trim_cost_from_eval_threshold(xs_lrf, xs_lrf_original, warmup_evals)

    batch_desc_parts: list[str] = []
    if any(a in algorithms for a in ("rr", "ucb", "lrf")):
        batch_desc_parts.append(f"UCB/LRF batch={args.batch_size}")
    if "gittins" in algorithms:
        batch_desc_parts.append(f"Gittins batch={args.gittins_batch_size}")
    batch_desc = ", ".join(batch_desc_parts) if batch_desc_parts else f"batch={args.batch_size}"

    if args.gittins_cost_mode == "aware" and total_full_matrix_cost is not None and budget_max_cumulative_cost is not None:
        budget_str = (
            f"budget={args.eval_budget_fraction:.0%} of full-matrix cost "
            f"(cap {budget_max_cumulative_cost:.4g} / {total_full_matrix_cost:.4g} cost units)"
        )
    else:
        budget_str = f"budget={args.eval_budget_fraction:.0%} of {n_cells} cells (eval cap)"

    subtitle = f"algorithms={','.join(algorithms)} | seed={args.seed}, {batch_desc}, {budget_str}"
    if "lrf" in algorithms:
        subtitle += (
            f"\n(LRF: {args.warmup_percentage:.0%} random warm-up, "
            f"curve starts at ~{warmup_evals} evals)"
        )

    _save_eval_figure(
        args.out,
        matrix_name=args.matrix.name,
        subtitle=subtitle,
        xs_rr=xs_rr,
        regrets_rr=regrets_rr,
        xs_ucb=xs_ucb,
        regrets_ucb=regrets_ucb,
        xs_lrf_plot=xs_lrf_plot,
        regrets_lrf_plot=regrets_lrf_plot,
        xs_gittins=xs_gittins,
        regrets_gittins=regrets_gittins,
        gittins_stop_cum_eval=gittins_stop_cum_eval,
        gittins_per_cell_dp=args.gittins_per_cell_dp,
    )

    if has_cost_axis and args.out_cost is not None:
        _save_cost_figure(
            args.out_cost,
            matrix_name=args.matrix.name,
            subtitle=subtitle,
            xs_rr_cost=xs_rr_original,
            regrets_rr=regrets_rr,
            xs_ucb_cost=xs_ucb_original,
            regrets_ucb=regrets_ucb,
            xs_lrf_cost_plot=xs_lrf_plot_original,
            regrets_lrf_plot=regrets_lrf_plot,
            xs_gittins_cost=xs_gittins_original,
            regrets_gittins=regrets_gittins,
        )

    meta = {
        "matrix": str(args.matrix.resolve()),
        "figure": str(args.out.resolve()),
        "figure_cost": str(args.out_cost.resolve()) if args.out_cost else None,
        "traces": str(args.traces_out.resolve()),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gittins_batch_size": args.gittins_batch_size,
        "eval_budget_fraction": args.eval_budget_fraction,
        "ucb_a": args.a,
        "warmup_percentage": args.warmup_percentage,
        "lrf_device": args.lrf_device,
        "gittins_grid_points": args.gittins_grid_points,
        "gittins_cost": gittins_cost_to_meta_string(gittins_cost_tensor),
        "gittins_cost_mode": args.gittins_cost_mode,
        "gittins_cost_vector_file": str(cost_vector_path.resolve()) if cost_vector_path else None,
        "original_cost_vector_file": str(original_cost_path.resolve()) if original_cost_path else None,
        "experiment": args.experiment,
        "gittins_prior_mean": gittins_prior_mean,
        "gittins_prior_variance": gittins_prior_variance,
        "gittins_per_cell_dp": args.gittins_per_cell_dp,
        "n_cells": n_cells,
        "budget_stops_by": "cumulative_cost" if args.gittins_cost_mode == "aware" else "evaluations",
        "total_full_matrix_cost": total_full_matrix_cost,
        "budget_max_cumulative_cost": budget_max_cumulative_cost,
        "algorithms": algorithms,
        "title": f"Simple regret — {args.matrix.name}\n{subtitle}",
        "gittins_stop_cum_eval": gittins_stop_cum_eval,
        "timing": {
            "rr": timing_rr,
            "ucb": timing_ucb,
            "lrf": timing_lrf,
            "gittins": timing_gittins,
        },
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
        xs_rr_original_cost=xs_rr_original if xs_rr_original else None,
        xs_ucbe_original_cost=xs_ucb_original if xs_ucb_original else None,
        xs_lrf_original_cost=xs_lrf_original if xs_lrf_original else None,
        xs_lrf_plot_original_cost=xs_lrf_plot_original if xs_lrf_plot_original else None,
        xs_gittins=xs_gittins,
        regrets_gittins=regrets_gittins,
        xs_gittins_original_cost=xs_gittins_original if xs_gittins_original else None,
        gittins_stop_cum_eval=gittins_stop_cum_eval,
        warmup_evals=warmup_evals,
        budget_evals=budget_max_evals,
        tau_sq_gittins=tau_sq_gittins,
        meta=meta,
    )

    print(f"Wrote traces {args.traces_out} and {args.traces_out.with_suffix('.meta.json')}")
    print(f"Wrote eval figure {args.out}")
    if has_cost_axis and args.out_cost is not None:
        print(f"Wrote cost figure {args.out_cost}")

    if regrets_rr:
        s_total = _timing_summary(timing_rr["iter_total_s"])
        print(
            f"Round-robin: {len(regrets_rr)} batches, {xs_rr[-1] if xs_rr else 0} evals, "
            f"cum cost {(_final_cum_cost(timing_rr) or 0.0):.6g}"
        )
        print(
            "  timing (per-iteration): "
            f"mean={s_total.get('mean_s', float('nan')):.6f}s "
            f"median={s_total.get('median_s', float('nan')):.6f}s "
            f"p90={s_total.get('p90_s', float('nan')):.6f}s"
        )

    if regrets_ucb:
        s_total = _timing_summary(timing_ucb["iter_total_s"])
        print(
            f"UCB-E: {len(regrets_ucb)} batches, {xs_ucb[-1] if xs_ucb else 0} evals, "
            f"cum cost {(_final_cum_cost(timing_ucb) or 0.0):.6g}"
        )
        print(
            "  timing (per-iteration): "
            f"mean={s_total.get('mean_s', float('nan')):.6f}s "
            f"median={s_total.get('median_s', float('nan')):.6f}s "
            f"p90={s_total.get('p90_s', float('nan')):.6f}s"
        )

    if regrets_lrf:
        s_total = _timing_summary(timing_lrf["iter_total_s"])
        print(
            f"UCB-E-LRF: {len(regrets_lrf)} batches, {xs_lrf[-1] if xs_lrf else 0} evals, "
            f"cum cost {(_final_cum_cost(timing_lrf) or 0.0):.6g} "
            f"({len(regrets_lrf_plot)} plotted points from cum_eval >= {warmup_evals})"
        )
        print(
            "  timing (per-iteration): "
            f"mean={s_total.get('mean_s', float('nan')):.6f}s "
            f"median={s_total.get('median_s', float('nan')):.6f}s "
            f"p90={s_total.get('p90_s', float('nan')):.6f}s"
        )

    if regrets_gittins:
        s_total = _timing_summary(timing_gittins["iter_total_s"])
        tau_msg = f"tau_sq = 1/(4B) = {tau_sq_gittins}"
        stop_msg = (
            f", nominal stop marker at cum_eval={gittins_stop_cum_eval}"
            if gittins_stop_cum_eval is not None
            else ""
        )
        print(
            f"Gittins: {len(regrets_gittins)} batches, {xs_gittins[-1] if xs_gittins else 0} evals, "
            f"cum cost {(_final_cum_cost(timing_gittins) or 0.0):.6g} "
            f"(B = {args.gittins_batch_size}, {tau_msg}){stop_msg}"
        )
        print(
            "  timing (per-iteration): "
            f"mean={s_total.get('mean_s', float('nan')):.6f}s "
            f"median={s_total.get('median_s', float('nan')):.6f}s "
            f"p90={s_total.get('p90_s', float('nan')):.6f}s"
        )

    if run is not None:
        run.summary["final_rr_simple_regret"] = float(regrets_rr[-1]) if regrets_rr else None
        run.summary["final_ucb_simple_regret"] = float(regrets_ucb[-1]) if regrets_ucb else None
        run.summary["final_lrf_simple_regret"] = float(regrets_lrf_plot[-1]) if regrets_lrf_plot else None
        run.summary["final_gittins_simple_regret"] = float(regrets_gittins[-1]) if regrets_gittins else None

        run.summary["final_rr_cum_eval"] = int(xs_rr[-1]) if xs_rr else None
        run.summary["final_ucb_cum_eval"] = int(xs_ucb[-1]) if xs_ucb else None
        run.summary["final_lrf_cum_eval"] = int(xs_lrf[-1]) if xs_lrf else None
        run.summary["final_gittins_cum_eval"] = int(xs_gittins[-1]) if xs_gittins else None

        if has_cost_axis:
            run.summary["final_rr_cum_original_cost"] = float(xs_rr_original[-1]) if xs_rr_original else None
            run.summary["final_ucb_cum_original_cost"] = float(xs_ucb_original[-1]) if xs_ucb_original else None
            run.summary["final_lrf_cum_original_cost"] = float(xs_lrf_original[-1]) if xs_lrf_original else None
            run.summary["final_gittins_cum_original_cost"] = float(xs_gittins_original[-1]) if xs_gittins_original else None

        run.log({"media/eval_figure": wandb.Image(str(args.out))})
        if has_cost_axis and args.out_cost is not None and args.out_cost.is_file():
            run.log({"media/cost_figure": wandb.Image(str(args.out_cost))})

        artifact = wandb.Artifact(
            name=f"simple-regret-live-{run.id}",
            type="experiment_outputs",
            description="Live W&B simple-regret outputs",
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