#!/usr/bin/env python3
"""Compare recommendation rules on identical, local Gittins sampling trajectories.

Example (no W&B or network):
  .venv/bin/python scripts/compare_recommendation_rules.py \
    --matrix data/MMLU_matrices/abstract_algebra.npy \
             data/MMLU_matrices/computer_security.npy \
    --prior-types default dataset --seeds 10

The recommendation never changes sampling or stopping. Compare latent-arm and
full-test-set posterior means. Add --include-lcb to also compare std penalties.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gittins_policy import gittins_index_exploration  # noqa: E402
from gittins_shrinking_posterior import compute_finite_population_roots_lookup_table  # noqa: E402
from simple_regret_recommend import (  # noqa: E402
    finite_population_posterior_moments,
    posterior_moments,
    recommend_from_posterior,
)

BUCKET_PRIORS = {"low": (0.4, 0.02), "medium": (0.6, 0.02), "high": (0.75, 0.01)}
DATASET_PRIORS = {"gsm8k": (0.2, 0.01)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_se(values: np.ndarray) -> tuple[float, float | None]:
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return float(values.mean()), se


def dataset_prior(matrix: Path, dataset_tag: str, buckets: dict) -> tuple[float, float]:
    if dataset_tag == "auto":
        if matrix.stem.lower().startswith("gsm8k_"):
            dataset_tag = "gsm8k"
        elif matrix.stem in buckets:
            dataset_tag = "mmlu"
        else:
            raise ValueError(f"Cannot infer dataset prior for {matrix}; provide --dataset-tag.")
    if dataset_tag == "mmlu":
        if matrix.stem not in buckets:
            raise ValueError(f"MMLU task missing from metadata: {matrix.stem}")
        return BUCKET_PRIORS[buckets[matrix.stem]]
    return DATASET_PRIORS[dataset_tag]


def run_trajectory(truth, roots, prior, args, rules, seed):
    torch.manual_seed(seed)
    obs = torch.full_like(truth, float("nan"))
    true_means = truth.to(torch.float64).mean(dim=1)
    best_mean = float(true_means.max())
    max_evaluations = int(np.ceil(truth.numel() * args.budget_fraction))
    cache = torch.full((truth.shape[0],), float("inf"), dtype=torch.float32)
    previous_arm = None
    evaluated = 0
    counts = torch.zeros(truth.shape[0], dtype=torch.int64)
    history = {key: [] for key in (
        "evaluations", "pulled_arm", "regret", "recommendation",
        "recommended_mean", "recommended_std", "recommended_count",
        "finite_mean_affine_max_abs_error",
    )}
    while evaluated < max_evaluations:
        batch = gittins_index_exploration(
            obs, prior_mean=prior[0], prior_variance=prior[1],
            obs_noise_variance=args.tau_sq_cell / args.batch_size,
            batch_size=args.batch_size, batch_observation_model=True,
            cost_scaling_factor=args.cost_scaling_factor,
            n_gittins_grid_points=args.grid_points, roots_lookup_table=roots,
            cached_scores=cache,
            recompute_arms=None if previous_arm is None else [previous_arm],
            allow_early_stop=False,
        )
        if batch is None:
            break
        row, col = batch
        obs[row, col] = truth[row, col]
        previous_arm = int(row[0])
        evaluated += len(row)
        counts[previous_arm] += len(row)
        moments = {
            target: helper(
                obs, prior_mean=prior[0], prior_variance=prior[1], tau_sq_cell=args.tau_sq_cell,
            )
            for target, helper in (
                ("latent", posterior_moments),
                ("finite", finite_population_posterior_moments),
            )
        }
        recommendations, regrets = [], []
        recommended_means, recommended_stds, recommended_counts = [], [], []
        for _, rule_kw in rules:
            means, variances = moments[rule_kw["target"]]
            arm, _ = recommend_from_posterior(means, variances, std_penalty=rule_kw["std_penalty"])
            recommendations.append(arm)
            regrets.append(best_mean - float(true_means[arm]))
            recommended_means.append(float(means[arm]))
            recommended_stds.append(float(variances[arm].sqrt()))
            recommended_counts.append(int(counts[arm]))
        history["evaluations"].append(evaluated)
        history["pulled_arm"].append(previous_arm)
        history["regret"].append(regrets)
        history["recommendation"].append(recommendations)
        history["recommended_mean"].append(recommended_means)
        history["recommended_std"].append(recommended_stds)
        history["recommended_count"].append(recommended_counts)
        affine_means = prior[0] + (1 + args.tau_sq_cell / (truth.shape[1] * prior[1])) * (
            moments["latent"][0].to(torch.float64) - prior[0]
        )
        history["finite_mean_affine_max_abs_error"].append(
            float((moments["finite"][0].to(torch.float64) - affine_means).abs().max())
        )
    return {key: np.asarray(value) for key, value in history.items()}


def align_histories(histories: list[dict], budget: int) -> tuple[np.ndarray, dict]:
    """Align post-pull states without looking ahead or smoothing between pulls.

    An exhausted arm can produce a partial batch, so seeds need not share their
    evaluation grid. Keep only the common observed domain and carry each seed's
    most recently completed state forward onto the union of event times.
    """
    start = max(int(history["evaluations"][0]) for history in histories)
    end = min(budget, *(int(history["evaluations"][-1]) for history in histories))
    if end < start:
        raise ValueError("No shared post-pull evaluation domain within the budget.")
    events = np.unique(np.concatenate([history["evaluations"] for history in histories]))
    x = np.unique(np.r_[events[(events >= start) & (events <= end)], end])
    indices = [np.searchsorted(history["evaluations"], x, side="right") - 1
               for history in histories]
    stacked = {
        key: np.stack([history[key][index] for history, index in zip(histories, indices)])
        for key in histories[0] if key != "evaluations"
    }
    # Retain the source event times, distinguishing carried states from new pulls.
    stacked["source_evaluations"] = np.stack([
        history["evaluations"][index] for history, index in zip(histories, indices)
    ])
    stacked["evaluations"] = np.broadcast_to(x, (len(histories), len(x))).copy()
    return x, stacked


def plot_results(results, rules, out):
    os.environ.setdefault("MPLCONFIGDIR", str((out / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((out / ".cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#34445d", "#008579", "#c06a35", "#8d62a8"]
    labels = [name for name, _ in rules]
    columns = min(2, len(results))
    rows = int(np.ceil(len(results) / columns))
    for paired in (False, True):
        fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 3.8 * rows), squeeze=False)
        for ax, result in zip(axes.flat, results):
            x = result["evaluations"] / result["n_cells"] * 100
            regrets = result["regret"]
            for rule_index in range(1 if paired else 0, len(rules)):
                values = regrets[:, :, rule_index]
                if paired:
                    values = values - regrets[:, :, 0]
                means = values.mean(axis=0)
                ax.step(x, means, where="post", color=colors[rule_index], label=labels[rule_index],
                        linewidth=1.2, linestyle="--" if rule_index % 2 else "-")
                if len(values) > 1:
                    errors = values.std(axis=0, ddof=1) / np.sqrt(len(values))
                    ax.fill_between(x, means-errors, means+errors, color=colors[rule_index], alpha=0.13, step="post")
            if paired:
                ax.axhline(0, color="#303a52", linewidth=0.7)
            else:
                ax.set_ylim(bottom=0)
            ax.set_title(
                f"{result['dataset']}\n"
                f"{result['prior_type']} prior N({result['prior'][0]}, {result['prior'][1]})"
                f" · batch size {result['batch_size']}", fontsize=10,
            )
            ax.set_xlabel("Evaluation budget (% of all matrix cells)")
            ax.set_ylabel("Simple regret difference vs. baseline" if paired else "Simple regret (accuracy gap)")
            ax.grid(alpha=0.18)
        for ax in list(axes.flat)[len(results):]:
            ax.set_visible(False)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=min(2, len(legend_labels)), frameon=False)
        trajectory_label = ("Paired Gittins trajectories; mean ± 1 SE across seeds"
                            if results[0]["regret"].shape[0] > 1
                            else "Same Gittins observations within each prior; one run seed; unsmoothed")
        fig.suptitle(trajectory_label + ("; negative = better" if paired else ""), fontsize=11)
        fig.tight_layout(rect=(0, 0.11, 1, 0.93))
        fig.savefig(out / ("paired_regret_difference.png" if paired else "simple_regret.png"), dpi=170)
        plt.close(fig)


    if results[0]["regret"].shape[0] == 1:
        for cumulative in (False, True):
            fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 3.8 * rows), squeeze=False)
            for ax, result in zip(axes.flat, results):
                x = result["evaluations"] / result["n_cells"] * 100
                for rule_index, label in enumerate(labels):
                    arms = result["recommendation"][0, :, rule_index]
                    values = np.r_[0, np.cumsum(arms[1:] != arms[:-1])] if cumulative else arms
                    ax.step(x, values, where="post", color=colors[rule_index], label=label, linewidth=1.1,
                            alpha=0.85, linestyle="--" if rule_index == 0 else "-")
                ax.set_title(
                    f"{result['dataset']}\n{result['prior_type']} prior"
                    f" · batch size {result['batch_size']}", fontsize=10,
                )
                ax.set_xlabel("Evaluation budget (% of all matrix cells)")
                ax.set_ylabel("Cumulative recommendation switches" if cumulative else "Recommended arm index")
                ax.grid(alpha=0.18)
            for ax in list(axes.flat)[len(results):]:
                ax.set_visible(False)
            handles, legend_labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="lower center", ncol=min(2, len(legend_labels)), frameon=False)
            fig.suptitle("One run seed; raw post-pull recommendations", fontsize=11)
            fig.tight_layout(rect=(0, 0.11, 1, 0.93))
            fig.savefig(out / ("recommendation_switches.png" if cumulative else "recommended_arm.png"), dpi=170)
            plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, nargs="+", required=True)
    parser.add_argument("--prior-types", choices=("default", "dataset"), nargs="+", default=["default", "dataset"])
    parser.add_argument("--dataset-tag", choices=("auto", "mmlu", "gsm8k"), default="auto")
    parser.add_argument("--task-metadata", type=Path, default=REPO_ROOT / "data/MMLU_matrices/task_metadata.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/recommendation_lcb_pilot")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tau-sq-cell", type=float, default=0.25)
    parser.add_argument("--cost-scaling-factor", type=float, default=1e-4)
    parser.add_argument("--grid-points", type=int, default=1025)
    parser.add_argument("--std-penalty", type=float, default=1.0)
    parser.add_argument("--include-lcb", action="store_true",
                        help="Also replay latent and full-test mean minus std-penalty times std.")
    args = parser.parse_args()
    if args.seeds < 1 or args.batch_size < 1 or not 0 < args.budget_fraction <= 1:
        parser.error("Require seeds >= 1, batch-size >= 1, and budget-fraction in (0, 1].")
    for name in ("tau_sq_cell", "cost_scaling_factor"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not np.isfinite(args.std_penalty) or args.std_penalty < 0 or args.grid_points < 2:
        parser.error("Require finite std-penalty >= 0 and grid-points >= 2.")
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = (json.loads(args.task_metadata.read_text())
                if "dataset" in args.prior_types and args.dataset_tag != "gsm8k" else {})
    buckets = {row["task"]: row["dataset_prior_bucket"] for row in metadata.get("tasks", [])}
    rules = [
        ("Latent mean (baseline)", {"target": "latent", "std_penalty": 0.0}),
        ("Full-test mean", {"target": "finite", "std_penalty": 0.0}),
    ]
    if args.include_lcb:
        rules.extend([
        (f"Latent mean − {args.std_penalty:g} × std", {"target": "latent", "std_penalty": args.std_penalty}),
        (f"Full-test mean − {args.std_penalty:g} × std", {"target": "finite", "std_penalty": args.std_penalty}),
        ])
    checkpoints, summaries, results, setups = [], [], [], []
    started = time.perf_counter()
    for matrix in args.matrix:
        arr = np.load(matrix)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError(f"Expected a finite 2D matrix: {matrix}")
        if min(arr.shape) < 1:
            raise ValueError(f"Expected a nonempty matrix: {matrix}")
        truth = torch.tensor(arr, dtype=torch.float32)
        for prior_type in args.prior_types:
            prior = (0.5, 0.04) if prior_type == "default" else dataset_prior(matrix, args.dataset_tag, buckets)
            print(f"{matrix.stem}/{prior_type}: {tuple(truth.shape)}, prior={prior}; computing lookup", flush=True)
            roots = torch.tensor(
                np.array(compute_finite_population_roots_lookup_table(
                    prior_variance=prior[1], tau_sq_cell=args.tau_sq_cell,
                    n_examples=truth.shape[1],
                    costs_per_arm=np.float32(args.cost_scaling_factor),
                    n_points=args.grid_points,
                )), dtype=torch.float32,
            )
            histories = []
            raw_dir = args.out_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                seed_started = time.perf_counter()
                histories.append(run_trajectory(truth, roots, prior, args, rules, seed))
                np.savez_compressed(
                    raw_dir / f"{matrix.stem}_{prior_type}_seed{seed}.npz",
                    **histories[-1], seed=np.asarray(seed),
                    rules=np.array([name for name, _ in rules]),
                    gittins_index_target=np.asarray("finite_population_mean"),
                )
                print(f"  seed {seed}: {len(histories[-1]['evaluations'])} pulls in {time.perf_counter()-seed_started:.1f}s", flush=True)
            requested_budget = int(np.ceil(truth.numel() * args.budget_fraction))
            x, stacked = align_histories(histories, requested_budget)
            np.savez_compressed(
                args.out_dir / f"{matrix.stem}_{prior_type}.npz", **stacked,
                rules=np.array([name for name, _ in rules]),
                seeds=np.arange(args.seed_start, args.seed_start + args.seeds),
                gittins_index_target=np.asarray("finite_population_mean"),
            )
            result = dict(
                dataset=matrix.stem, prior_type=prior_type, prior=prior,
                batch_size=int(args.batch_size),
                n_cells=truth.numel(), evaluations=x, regret=stacked["regret"],
                recommendation=stacked["recommendation"],
            )
            results.append(result)
            setups.append(dict(
                dataset=matrix.stem, matrix=str(matrix),
                sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
                shape=list(truth.shape), prior_type=prior_type,
                prior_mean=prior[0], prior_variance=prior[1], evaluation_budget=int(x[-1]),
                raw_final_evaluations=[int(history["evaluations"][-1]) for history in histories],
                raw_history_pattern=str(raw_dir / f"{matrix.stem}_{prior_type}_seed{{seed}}.npz"),
                raw_post_pull_states=sum(len(history["evaluations"]) for history in histories),
                mean_only_recommendation_disagreements=sum(int(np.count_nonzero(
                    history["recommendation"][:, 0] != history["recommendation"][:, 1]
                )) for history in histories),
                finite_mean_affine_max_abs_error=max(float(history["finite_mean_affine_max_abs_error"].max())
                                                     for history in histories),
            ))
            targets = sorted({
                4, 20, 40, 100, 500, 1000, int(x[-1]),
                *[int(truth.numel()*fraction) for fraction in (.005, .01, .02, .05, .1)],
            })
            checkpoint_indices = sorted({
                int(np.searchsorted(x, target, side="right")-1)
                for target in targets if x[0] <= target <= x[-1]
            })
            for rule_index, (rule_name, _) in enumerate(rules):
                regrets = stacked["regret"][:, :, rule_index]
                differences = regrets - stacked["regret"][:, :, 0]
                diff_rec = stacked["recommendation"][:, :, rule_index] != stacked["recommendation"][:, :, 0]
                final_mean, final_se = mean_se(regrets[:, -1])
                delta_mean, delta_se = mean_se(differences[:, -1])
                shared = dict(dataset=matrix.stem, prior_type=prior_type, rule=rule_name)
                summary = dict(
                    **shared, seeds=args.seeds, final_evaluations=int(x[-1]),
                    final_regret_mean=final_mean, final_regret_se=final_se,
                    final_paired_delta_mean=delta_mean, final_paired_delta_se=delta_se,
                    mean_regret_over_steps=float(regrets.mean()),
                    budget_weighted_mean_regret=(float(np.average(regrets[:, :-1], axis=1, weights=np.diff(x)).mean())
                                                 if len(x) > 1 else None),
                    mean_paired_delta_over_steps=float(differences.mean()),
                    mean_paired_delta_over_steps_se=mean_se(differences.mean(axis=1))[1],
                    differing_recommendation_fraction=float(diff_rec.mean()),
                )
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
                for index in checkpoint_indices:
                    mean, se = mean_se(regrets[:, index])
                    delta, delta_se = mean_se(differences[:, index])
                    checkpoints.append(dict(
                        **shared, evaluations=int(x[index]),
                        budget_fraction=float(x[index]/truth.numel()),
                        regret_mean=mean, regret_se=se,
                        paired_delta_mean=delta, paired_delta_se=delta_se,
                        differing_recommendation_fraction=float(diff_rec[:, index].mean()),
                    ))
            write_csv(args.out_dir / "summary.csv", summaries)
            write_csv(args.out_dir / "checkpoints.csv", checkpoints)
    config = {
        key: str(value) if isinstance(value, Path) else
        [str(v) for v in value] if key == "matrix" else value
        for key, value in vars(args).items()
    }
    payload = dict(
        config=config,
        sampling_policy="Full-test-set Gittins, unit cost; fixed budget; recommendations replayed on identical observations",
        acquisition_target="full_test_set_mean",
        gittins_index_target="finite_population_mean",
        acquisition_cost_semantics="The numerical cost scaling is unchanged and is expressed in full-test-set-mean reward units.",
        recommendation_score="Target-specific posterior mean - std_penalty * sqrt(target-specific posterior variance)",
        recommendation_rules=[dict(name=name, **rule) for name, rule in rules],
        posterior_targets={
            "latent": "Latent arm mean theta; normal-normal posterior moments",
            "finite": "Full fixed test-set mean: (observed_sum + (N-n)*latent_mean)/N; variance=((N-n)/N)^2*latent_variance + (N-n)*tau_sq_cell/N^2",
        },
        standard_error=("sample standard deviation across seeds / sqrt(number of seeds)"
                        if args.seeds > 1 else "undefined for one seed; reported as null; no error shading"),
        budget_convention="Raw runs complete the final batch crossing the requested budget, as in simulate_simple_regret.py; aggregate stops at the requested budget",
        aggregation="Union of completed-pull evaluation times within the shared observed domain, carrying each seed's last completed state forward; no interpolation or smoothing",
        mean_regret_over_steps_convention="Mean over aligned grid points; use budget_weighted_mean_regret for the budget-weighted average from the first shared observation to the budget",
        setups=setups, summary=summaries, elapsed_seconds=time.perf_counter()-started,
    )
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(results, rules, args.out_dir)
    print(f"Saved {args.out_dir}; total {time.perf_counter()-started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
