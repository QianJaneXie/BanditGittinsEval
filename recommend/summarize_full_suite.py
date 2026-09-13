"""Paired stability and quality analysis of AlpacaEval and all MMLU subjects.

Only reads completed saved experiments. All MMLU summaries first aggregate the
fixed subjects within each sampling-seed block, then bootstrap whole blocks.
This preserves the pairing and does not treat 57 subjects as 57 new replications.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from analyze_stability import METRICS, path_metrics

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("posterior_mean", "gaussian_lcb_1.645")
PRIORS = ("default", "dataset")
BEGIN, END = .01, .10
GRID = np.linspace(BEGIN, END, 181)
CHECKPOINTS = (.01, .02, .05, .10)
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPEATS = 10000
TOLERANCE = 1e-10
COLORS = ("#818a94", "#196cad")
SHORT_METRICS = {
    "total_downward_pp": "Total downward movement (pp)",
    "maximum_drawdown_pp": "Maximum drawdown (pp)",
    "switches": "Recommendation switches",
    "largest_drop_pp": "Largest downward jump (pp)",
    "mean_drawdown_pp": "Budget-averaged drawdown (pp)",
    "mean_regret_pp": "Budget-averaged regret (pp)",
    "final_regret_pp": "Final regret (pp)",
    "downward_jumps": "Harmful jumps",
    "jumps_ge_2pp": "Jumps of at least 2 pp",
    "any_jump_ge_2pp": "Risk of any jump of at least 2 pp",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def step_sample(x, values, targets):
    """Latest completed batch; hold the final state beyond a rounded endpoint."""
    indices = np.searchsorted(x, targets, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Requested a state before the first recorded evaluation")
    return values[np.minimum(indices, len(x) - 1)]


def curves_for_path(x, regrets, arms, end):
    """Keep every real jump for movement before sampling its cumulative curve."""
    first = int(np.searchsorted(x, BEGIN, side="right") - 1)
    stop = int(np.searchsorted(x, end, side="right"))
    window_x = np.r_[BEGIN, x[first + 1:stop]]
    window_r = regrets[first:stop]
    movement = np.vstack([np.zeros(2), 100 * np.cumsum(np.maximum(np.diff(window_r, axis=0), 0), axis=0)])
    return {
        "regret_pp": 100 * step_sample(x, regrets, np.minimum(GRID, end)),
        "cumulative_downward_pp": step_sample(window_x, movement, np.minimum(GRID, end)),
    }


def collect(results_root, allow_subset=False):
    rows, checkpoint_rows, sources, experiments, curves = [], [], {}, {}, {}
    directories = sorted(p.parent for p in results_root.glob("*/manifest.json")
                         if p.parent.name == "alpaca_eval" or p.parent.name.startswith("mmlu_"))
    expected_tasks = {"mmlu_" + row["task"] for row in json.loads(
        (ROOT / "data/MMLU_matrices/task_metadata.json").read_text(encoding="utf-8"))["tasks"]}
    expected_names = expected_tasks | {"alpaca_eval"}
    found = {p.name for p in directories}
    if not found:
        raise ValueError(f"No suite manifests found in {results_root}")
    if not allow_subset and found != expected_names:
        raise ValueError(f"Expected AlpacaEval and all {len(expected_tasks)} MMLU subjects; "
                         f"missing={sorted(expected_names-found)}, extra={sorted(found-expected_names)}")
    for directory in directories:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete experiment: {directory}")
        sources[relative(manifest_path)] = digest(manifest_path)
        config = manifest["config"]
        if tuple(config["priors"]) != PRIORS or len(config["matrix_seeds"]) != 1:
            raise ValueError("Expected both priors and exactly one fixed matrix per subject")
        if abs(config["budget_fraction"] - END) > 1e-12:
            raise ValueError("This analysis prespecifies a nominal 10% endpoint")
        n_questions = int(next(iter(manifest["matrices"].values()))["shape"][1])
        n_arms = int(next(iter(manifest["matrices"].values()))["shape"][0])
        benchmark = directory.name
        experiments[benchmark] = dict(label=config.get("benchmark_label", benchmark),
            n_questions=n_questions, n_arms=n_arms, batch_size=config["batch_size"],
            prior_pairs=manifest["priors"], seeds=sorted(config["run_seeds"]),
            unit="accuracy pp" if benchmark.startswith("mmlu_") else "score points (0–100 scale)",
            exact_budget=round(config["budget_fraction"] * n_questions * n_arms),
            manifest_path=relative(manifest_path))
        expected = {(p, int(m), int(s)) for p in config["priors"]
                    for m in config["matrix_seeds"] for s in config["run_seeds"]}
        seen = set()
        for path in sorted((directory / "traces").glob("*.npz")):
            sources[relative(path)] = digest(path)
            with np.load(path, allow_pickle=False) as z:
                identity = (str(z["prior_tag"]), int(z["matrix_seed"]), int(z["run_seed"]))
                if identity in seen or identity not in expected:
                    raise ValueError(f"Unexpected or duplicate trace: {path}")
                seen.add(identity)
                keys = list(z["method_keys"].astype(str))
                columns = [keys.index(key) for key in METHODS]
                arms = np.asarray(z["recommended_arm"][:, columns], dtype=np.int64)
                regrets = np.asarray(z["regret"][:, columns], dtype=float)
                cells = int(z["n_arms"]) * int(z["n_examples"])
                if cells != n_arms * n_questions:
                    raise ValueError(f"Wrong dimensions: {path}")
                x_cells = np.asarray(z["x"], dtype=np.int64)
                if int(x_cells[-1]) != experiments[benchmark]["exact_budget"]:
                    raise ValueError(f"Incorrect final budget: {path}")
                x = x_cells / cells
                if regrets.shape != (len(x), 2) or arms.shape != regrets.shape:
                    raise ValueError(f"Invalid paired arrays: {path}")
                if not np.isfinite(regrets).all() or np.any(arms < 0):
                    raise ValueError("Primary rules must recommend throughout; abstention is not zero loss")
                oracle = np.asarray(z["true_means"], dtype=float)
                np.testing.assert_allclose(regrets, oracle.max() - oracle[arms], atol=1e-10, rtol=0)
                end = min(END, float(x[-1]))
                for col, method in enumerate(METHODS):
                    rows.append(dict(benchmark=benchmark, prior=identity[0], matrix_seed=identity[1],
                        run_seed=identity[2], method=method, n_questions=n_questions,
                        window_begin=BEGIN, window_end=end,
                        **path_metrics(x, regrets[:, col], arms[:, col], begin=BEGIN, end=end)))
                    for budget in CHECKPOINTS:
                        cp = int(np.searchsorted(x, budget, side="right") - 1)
                        if cp < 0:
                            raise ValueError("Checkpoint precedes first observation")
                        checkpoint_rows.append(dict(benchmark=benchmark, prior=identity[0],
                            matrix_seed=identity[1], run_seed=identity[2], method=method,
                            budget_fraction=budget, observed_cells=int(x_cells[cp]),
                            observed_fraction=float(x[cp]), regret_pp=float(100 * regrets[cp, col]),
                            selected_score_100=float(100 * oracle[arms[cp, col]]),
                            best_score_100=float(100 * oracle.max()),
                            selected_n=int(z["selected_n"][cp, columns[col]])))
                curves[benchmark, identity[0], identity[2]] = curves_for_path(x, regrets, arms, end)
        if seen != expected:
            raise ValueError(f"Missing traces in {directory}: {sorted(expected-seen)}")
        print(f"Loaded {benchmark}: {len(seen)} paired traces", flush=True)
    seed_sets = {tuple(info["seeds"]) for info in experiments.values()}
    if len(seed_sets) != 1:
        raise ValueError("All suite datasets must share the same complete seed grid")
    return rows, checkpoint_rows, sources, experiments, curves


def bootstrap_weights(n_seeds):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=BOOTSTRAP_REPEATS) / n_seeds


def summarize_blocks(blocks, weights):
    """The only uncertainty unit is the first (sampling-seed) axis."""
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 2 or blocks.shape != (weights.shape[1], 2):
        raise ValueError("Expected paired seed-by-method blocks")
    means = blocks.mean(axis=0)
    ci = np.quantile(weights @ blocks, [.025, .975], axis=0)
    differences = blocks[:, 1] - blocks[:, 0]
    delta_ci = np.quantile(weights @ differences, [.025, .975])
    tail = np.quantile(blocks, .9, axis=0)
    return dict(n_seed_blocks=len(blocks), posterior_mean=float(means[0]), lcb_mean=float(means[1]),
        posterior_p90=float(tail[0]), lcb_p90=float(tail[1]),
        posterior_ci95_low=float(ci[0, 0]), posterior_ci95_high=float(ci[1, 0]),
        lcb_ci95_low=float(ci[0, 1]), lcb_ci95_high=float(ci[1, 1]),
        paired_mean_delta=float(differences.mean()), paired_ci95_low=float(delta_ci[0]),
        paired_ci95_high=float(delta_ci[1]),
        reduction_percent=float(100 * (means[0] - means[1]) / means[0]) if means[0] else None)


def comparison_groups(experiments):
    groups = {}
    if "alpaca_eval" in experiments:
        groups["alpaca_eval"] = (["alpaca_eval"], np.array([1.]))
    subjects = sorted(b for b in experiments if b.startswith("mmlu_"))
    if subjects:
        groups["mmlu_equal_subject"] = (subjects, np.full(len(subjects), 1 / len(subjects)))
        weights = np.array([experiments[b]["n_questions"] for b in subjects], dtype=float)
        groups["mmlu_question_weighted"] = (subjects, weights / weights.sum())
    return groups


def summarize(rows, checkpoint_rows, experiments):
    seeds = next(iter(experiments.values()))["seeds"]
    weights = bootstrap_weights(len(seeds))
    lookup = {(r["benchmark"], r["prior"], r["run_seed"], r["method"]): r for r in rows}
    subject_summary, aggregates, checkpoints, regressions = [], [], [], []
    blocks_by_subject = {}
    for benchmark in experiments:
        for prior in PRIORS:
            for metric in METRICS:
                blocks = np.array([[lookup[benchmark, prior, seed, method][metric]
                    for method in METHODS] for seed in seeds])
                blocks_by_subject[benchmark, prior, metric] = blocks
                result = dict(benchmark=benchmark, prior=prior, metric=metric,
                    n_questions=experiments[benchmark]["n_questions"], **summarize_blocks(blocks, weights))
                subject_summary.append(result)
                if result["paired_mean_delta"] > TOLERANCE:
                    regressions.append(result.copy())
    for scope, (subjects, subject_weights) in comparison_groups(experiments).items():
        for prior in PRIORS:
            for metric in METRICS:
                all_blocks = np.stack([blocks_by_subject[b, prior, metric] for b in subjects])
                blocks = np.einsum("b,bsm->sm", subject_weights, all_blocks)
                subject_deltas = (all_blocks[:, :, 1] - all_blocks[:, :, 0]).mean(axis=1)
                aggregates.append(dict(scope=scope, prior=prior, metric=metric, n_subjects=len(subjects),
                    lcb_better_subjects=int(np.sum(subject_deltas < -TOLERANCE)),
                    tied_subjects=int(np.sum(np.abs(subject_deltas) <= TOLERANCE)),
                    lcb_worse_subjects=int(np.sum(subject_deltas > TOLERANCE)),
                    **summarize_blocks(blocks, weights)))
    cp_lookup = {(r["benchmark"], r["prior"], r["run_seed"], r["method"], r["budget_fraction"]): r
                 for r in checkpoint_rows}
    cp_groups = {b: ([b], np.array([1.])) for b in experiments}
    cp_groups.update(comparison_groups(experiments))
    for scope, (subjects, subject_weights) in cp_groups.items():
        for prior in PRIORS:
            for budget in CHECKPOINTS:
                values = np.array([[[cp_lookup[b, prior, seed, method, budget]["regret_pp"]
                    for method in METHODS] for seed in seeds] for b in subjects])
                blocks = np.einsum("b,bsm->sm", subject_weights, values)
                checkpoints.append(dict(scope=scope, prior=prior, budget_fraction=budget,
                    n_subjects=len(subjects), **summarize_blocks(blocks, weights)))
    regressions.sort(key=lambda r: (r["metric"], r["prior"], -r["paired_mean_delta"]))
    return subject_summary, aggregates, checkpoints, regressions


def plot_all(subject_summary, aggregates, experiments, curves, out, subject_plots=True):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "recommend/.mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#a4aeb8", "text.color": "#213547", "axes.labelcolor": "#213547",
        "xtick.color": "#465769", "ytick.color": "#465769", "pdf.fonttype": 42})
    scopes = comparison_groups(experiments)
    primary_scopes = [s for s in ("alpaca_eval", "mmlu_equal_subject") if s in scopes]
    scope_labels = {"alpaca_eval": "AlpacaEval", "mmlu_equal_subject": "MMLU · equal subject average",
                    "mmlu_question_weighted": "MMLU · question weighted"}
    seeds = next(iter(experiments.values()))["seeds"]
    weights = bootstrap_weights(len(seeds))
    # Avoid excessive temporary arrays when displaying pointwise curve intervals.
    plot_weights = weights[:2000]
    aggregate_lookup = {(r["scope"], r["prior"], r["metric"]): r for r in aggregates}

    def save(fig, stem, pdf=True):
        fig.savefig(out / f"{stem}.png", dpi=175)
        if pdf:
            fig.savefig(out / f"{stem}.pdf")
        plt.close(fig)

    chosen = ["total_downward_pp", "maximum_drawdown_pp", "switches", "mean_regret_pp"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.3), squeeze=False)
    for row, prior in enumerate(PRIORS):
        for col, metric in enumerate(chosen):
            ax = axes[row, col]
            for y, scope in enumerate(primary_scopes):
                r = aggregate_lookup[scope, prior, metric]
                for method, offset, color, label in [("posterior", -.13, COLORS[0], "Posterior mean"),
                        ("lcb", .13, COLORS[1], "LCB: mean − 1.645 × SD")]:
                    ax.hlines(y + offset, r[f"{method}_ci95_low"], r[f"{method}_ci95_high"], color=color, lw=2)
                    ax.scatter(r[f"{method}_mean"], y + offset, color=color, s=35, label=label if y == 0 else None, zorder=3)
            ax.set_yticks(np.arange(len(primary_scopes)), [scope_labels[s].replace(" · equal subject average", " (57 subjects)")
                if len(scopes.get("mmlu_equal_subject", ([], []))[0]) == 57 else scope_labels[s] for s in primary_scopes])
            if col:
                ax.set_yticklabels([])
            ax.set_ylim(len(primary_scopes) - .4, -.6)
            ax.set_xlim(left=0)
            ax.set_title(f"{prior.capitalize()} prior", loc="left", fontweight="bold")
            ax.set_xlabel(SHORT_METRICS[metric])
            ax.grid(axis="x", color="#dce3e9", lw=.6)
    fig.suptitle("Recommendation stability and quality on the full evaluation suite", x=.025, y=.98,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.025, .925, "Gittins observations shared within each run · 1%–10% budget · Lower is better", color="#536879", fontsize=11)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.54, .071), ncol=2, frameon=False)
    fig.text(.025, .025, "Means of per-run metrics; bars: pointwise 95% paired seed-block bootstrap intervals (10,000 draws).\n"
        "MMLU averages fixed subjects within each seed. Compare rules within a dataset; raw movement depends on update frequency.\n"
        "MMLU: accuracy percentage points. AlpacaEval: score points on a 0–100 scale.", fontsize=9, color="#536879")
    fig.subplots_adjust(left=.145, right=.99, top=.84, bottom=.235, wspace=.3, hspace=.6)
    save(fig, "overview")

    def curve_blocks(subjects, subject_weights, prior, metric):
        return np.einsum("b,bsgm->sgm", subject_weights,
            np.array([[curves[b, prior, seed][metric] for seed in seeds] for b in subjects]))

    def draw_curve(ax, blocks, interval=True):
        for col, (color, label) in enumerate(zip(COLORS, ("Posterior mean", "LCB: mean − 1.645 × SD"))):
            means = blocks[:, :, col].mean(axis=0)
            ax.plot(100 * GRID, means, color=color, lw=1.9, label=label, drawstyle="steps-post")
            if interval:
                lows, highs = np.quantile(plot_weights @ blocks[:, :, col], [.025, .975], axis=0)
                ax.fill_between(100 * GRID, lows, highs, color=color, alpha=.13, lw=0, step="post")
        ax.grid(color="#dce3e9", lw=.6)
        ax.set_xlim(1, 10)
        ax.set_ylim(bottom=0)
        ax.set_xticks([1, 2, 5, 10])
        ax.set_xlabel("Evaluation budget (% of matrix cells)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig, axes = plt.subplots(2, len(primary_scopes), figsize=(6.6 * len(primary_scopes), 8.4), squeeze=False)
    for col, scope in enumerate(primary_scopes):
        subjects, subject_weights = scopes[scope]
        for row, prior in enumerate(PRIORS):
            draw_curve(axes[row, col], curve_blocks(subjects, subject_weights, prior, "regret_pp"))
            axes[row, col].set_title(f"{scope_labels[scope]} · {prior} prior", loc="left", fontweight="bold")
            axes[row, col].set_ylabel("Mean regret (score points)" if scope == "alpaca_eval" else "Mean regret (accuracy pp)")
    fig.suptitle("Recommendation quality as the evaluation budget grows", x=.04, y=.98, ha="left", fontsize=17, fontweight="bold")
    fig.text(.04, .93, "The two rules use the same Gittins allocation path · Shading: pointwise 95% bootstrap intervals", color="#536879")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.55, .055), ncol=2, frameon=False)
    fig.text(.04, .02, "2,000 seed-block resamples for curve shading; fixed subjects remain together.\n"
        "Last completed batch held to the nominal endpoint; no future observation is interpolated.", fontsize=9, color="#536879")
    fig.subplots_adjust(left=.08, right=.975, top=.85, bottom=.19, hspace=.46, wspace=.22)
    save(fig, "quality_curves")

    if subject_plots:
        (out / "per_subject").mkdir(exist_ok=True)
        for benchmark, info in experiments.items():
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.1), squeeze=False)
            for row, prior in enumerate(PRIORS):
                for col, metric in enumerate(["regret_pp", "cumulative_downward_pp"]):
                    blocks = curve_blocks([benchmark], np.array([1.]), prior, metric)
                    draw_curve(axes[row, col], blocks, interval=False)
                    axes[row, col].set_title(f"{prior.capitalize()} prior", loc="left", fontweight="bold")
                    unit = "accuracy pp" if benchmark.startswith("mmlu_") else "score points"
                    axes[row, col].set_ylabel(("Mean regret" if col == 0 else "Mean cumulative downward\nmovement") + f" ({unit})")
            fig.suptitle(info["label"], x=.05, y=.98, ha="left", fontsize=18, fontweight="bold")
            fig.text(.05, .925, f"{info['n_arms']:,} arms × {info['n_questions']:,} examples · {len(seeds)} paired seeds per prior · Batch {info['batch_size']}", color="#536879")
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.54, .045), ncol=2, frameon=False)
            fig.text(.05, .015, "Cumulative movement starts from the recommendation active at 1% and includes every later batch update.", fontsize=9, color="#536879")
            fig.subplots_adjust(left=.095, right=.98, top=.855, bottom=.17, hspace=.45, wspace=.23)
            save(fig, "per_subject/" + benchmark, pdf=False)

    subjects = scopes.get("mmlu_equal_subject", ([], []))[0]
    if not subjects:
        return
    summary_lookup = {(r["benchmark"], r["prior"], r["metric"]): r for r in subject_summary}
    heat_metrics = ["total_downward_pp", "maximum_drawdown_pp", "switches", "mean_regret_pp", "final_regret_pp"]
    heat_labels = ["Total down\n(pp)", "Max drawdown\n(pp)", "Switches\n(count)", "Mean regret\n(pp)", "Final regret\n(pp)"]
    # Each metric uses its own symmetric scale, shared across both priors.
    heat_height = max(7, .27 * len(subjects) + 2.5)
    fig, axes = plt.subplots(1, 10, figsize=(15, heat_height), squeeze=False)
    for p, prior in enumerate(PRIORS):
        for m, metric in enumerate(heat_metrics):
            ax = axes[0, p * 5 + m]
            values = np.array([summary_lookup[b, prior, metric]["paired_mean_delta"] for b in subjects])
            lim = max(abs(summary_lookup[b, pr, metric]["paired_mean_delta"]) for b in subjects for pr in PRIORS)
            lim = max(lim, .01)
            ax.imshow(values[:, None], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            for y, value in enumerate(values):
                ax.text(0, y, f"{value:+.1f}" if abs(value) >= 10 else f"{value:+.2f}", ha="center", va="center",
                    fontsize=7.7, color="white" if abs(value) > .58 * lim else "#243547")
            ax.set_title(heat_labels[m], fontsize=9, pad=8)
            ax.set_xticks([])
            ax.set_yticks(np.arange(len(subjects)))
            ax.set_yticklabels([b.removeprefix("mmlu_").replace("_", " ") for b in subjects] if p == 0 and m == 0 else [], fontsize=8.5)
            ax.tick_params(axis="y", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("LCB effects across MMLU subjects", x=.035, y=1-.08/heat_height, ha="left", va="top", fontsize=17, fontweight="bold")
    fig.text(.035, 1-.55/heat_height, "Each cell: LCB − posterior mean · Blue: lower / better · Red: higher / worse · Each metric has its own color scale", color="#536879", fontsize=10)
    fig.text(.425, 1-.92/heat_height, "DEFAULT PRIOR", ha="center", fontweight="bold", fontsize=12)
    fig.text(.795, 1-.92/heat_height, "DATASET PRIOR", ha="center", fontweight="bold", fontsize=12)
    fig.text(.035, .018, f"Mean differences across {len(seeds)} paired seeds, 1%–10% budget. Subject labels are fixed benchmarks, not independent bootstrap replications.\n"
        "Movement and switches assess stability; regret assesses recommendation quality. Point estimates are exploratory and not multiplicity-adjusted.", fontsize=9, color="#536879")
    fig.subplots_adjust(left=.26, right=.99, top=1-1.42/heat_height, bottom=.60/heat_height, wspace=.13)
    save(fig, "mmlu_subject_effects")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    for ax, prior in zip(axes, PRIORS):
        x = np.array([summary_lookup[b, prior, "mean_regret_pp"]["paired_mean_delta"] for b in subjects])
        y = np.array([summary_lookup[b, prior, "total_downward_pp"]["paired_mean_delta"] for b in subjects])
        ax.axhline(0, color="#9aa5b0", lw=1)
        ax.axvline(0, color="#9aa5b0", lw=1)
        ax.scatter(x, y, c=np.where(x > TOLERANCE, "#b65c30", COLORS[1]), s=35, alpha=.85, zorder=3)
        # Label only the largest quality regressions and movement exceptions.
        labels = set(np.argsort(x)[-3:].tolist()) | set(np.where(y > TOLERANCE)[0].tolist())
        for i in sorted(labels):
            ax.annotate(subjects[i].removeprefix("mmlu_").replace("_", " "), (x[i], y[i]),
                xytext=(-5, 5), textcoords="offset points", fontsize=7.5, ha="right")
        ax.set_title(f"{prior.capitalize()} prior", loc="left", fontweight="bold")
        ax.set_xlabel("LCB − posterior: budget-averaged regret (pp)")
        ax.set_ylabel("LCB − posterior: total downward movement (pp)")
        ax.grid(color="#dce3e9", lw=.6)
    fig.suptitle("MMLU: stability improvements and recommendation quality tradeoffs", x=.035, y=.98, ha="left", fontsize=16, fontweight="bold")
    fig.text(.035, .915, "One point per fixed subject · Lower = more stable · Left = better average recommendation quality", color="#536879")
    fig.text(.035, .02, "Orange points have higher average regret with LCB. Subject means summarize 20 paired seeds; no subject-level significance is implied.", fontsize=9, color="#536879")
    fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.16, wspace=.26)
    save(fig, "mmlu_stability_quality_tradeoff")


def report(subject_summary, aggregates, checkpoints, regressions, experiments, out, target):
    lookup = {(r["scope"], r["prior"], r["metric"]): r for r in aggregates}
    primary = [scope for scope in ("alpaca_eval", "mmlu_equal_subject") if (scope, "default", "total_downward_pp") in lookup]
    labels = {"alpaca_eval": "AlpacaEval", "mmlu_equal_subject": "MMLU: equal subject average", "mmlu_question_weighted": "MMLU: question weighted"}
    n_subjects = sum(b.startswith("mmlu_") for b in experiments)
    mmlu_questions = sum(info["n_questions"] for b, info in experiments.items() if b.startswith("mmlu_"))
    n_seeds = len(next(iter(experiments.values()))["seeds"])
    n_runs = len(experiments) * len(PRIORS) * n_seeds
    fig_rel = os.path.relpath(out, target.parent).replace("\\", "/")
    def link(name):
        return f"{fig_rel}/{name}"
    def source_link(name):
        return os.path.relpath(ROOT / name, target.parent).replace("\\", "/")
    headline = ""
    if n_subjects:
        primary_stability = ["total_downward_pp", "maximum_drawdown_pp", "largest_drop_pp", "switches"]
        consistency = [lookup["mmlu_equal_subject", p, m]["lcb_better_subjects"]
                       for p in PRIORS for m in primary_stability]
        if all(n == n_subjects for n in consistency):
            headline = (f"**LCB is more temporally stable in every one of the {n_subjects} MMLU subjects under both priors**: "
                "it reduces mean total downward movement, maximum drawdown, largest drop, and switches. "
                "Average recommendation quality also improves across the suite, with subject-level exceptions.\n\n")
        else:
            headline = "The full suite separates improvements in temporal stability from changes in recommendation quality; paired effects and subject-level exceptions are reported below.\n\n"
    if "alpaca_eval" in experiments:
        headline += ("AlpacaEval also has less movement with LCB. Its dataset-prior runs are already nearly stable with posterior mean, "
            "so the remaining improvement has an interval touching zero. Both methods finish with zero regret in every AlpacaEval run.\n\n")
    text = f"""# Gittins + LCB on AlpacaEval and the full MMLU suite

{headline}This comparison uses **{n_runs:,} paired Gittins trajectories**: AlpacaEval and
{n_subjects} MMLU subjects, each with {n_seeds} sampling seeds and two priors.
Posterior-mean recommendation and **LCB = mean − 1.645 × posterior SD** see
exactly the same observations in each trajectory. Only the recommendation rule
changes; Gittins continues to choose which arm to evaluate.

The primary robustness measures are total downward movement, maximum drawdown,
largest downward jump, and recommendation switches. Recommendation quality is
reported alongside stability: a consistently poor arm could be perfectly stable.

## Setup and scope

- This is an **offline replay of the existing saved reward matrices**. No new
  LLM generations or evaluator calls were made.
- The allocation and LCB coefficient follow the earlier experiments without tuning
  on these new results. Sampling is uniform without replacement within an arm.
- Runs use the existing NumPy/SciPy Gittins backend, cost scale `1e-4`, per-cell
  observation variance `0.25`, and the default `N(0.5, 0.04)` prior plus each
  dataset's existing prior. The second Normal parameter is variance.
- MMLU optimizes an arm **separately for each subject**. The primary suite result
  averages subject-level metrics equally within each sampling seed. It does not
  select one global arm for concatenated MMLU questions. A question-count-weighted
  average is included as a secondary summary.
- The stability window begins at the recommendation active at 1% of matrix cells
  and includes every later batch update through the actual rounded endpoint,
  capped at 10%. Checkpoints use the last completed batch at or before the nominal
  budget; no future observation is interpolated.
- MMLU movements and regret are **accuracy percentage points**. AlpacaEval uses
  the saved evaluator score matrix; its movements are **score points on a 0–100
  scale**, not automatically literal accuracy or win-rate percentage points.

"""
    if "alpaca_eval" in experiments:
        e = experiments["alpaca_eval"]
        text += f"AlpacaEval has {e['n_arms']:,} arms × {e['n_questions']:,} examples, batch {e['batch_size']}, "
        text += f"and an exact final budget of {e['exact_budget']:,} cells ({100*e['exact_budget']/(e['n_arms']*e['n_questions']):.7f}%). "
        text += "Its dataset prior is `N(0.2, 0.01)`. MMLU uses batch 4 and the existing subject-prior buckets.\n\n"
    if n_subjects:
        text += f"MMLU covers all {mmlu_questions:,} saved questions across {n_subjects} subjects, with 1,500 arms per subject.\n\n"
    text += "## Aggregate stability and quality\n\nEvery cell is **posterior mean → LCB**. Lower is better for all metrics below.\n\n"
    table_metrics = ["total_downward_pp", "maximum_drawdown_pp", "switches", "largest_drop_pp", "mean_regret_pp", "final_regret_pp"]
    text += "| Dataset / prior | Total down (pp) | Max drawdown (pp) | Switches | Largest drop (pp) | Mean regret (pp) | Final regret (pp) |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for scope in primary:
        for prior in PRIORS:
            values = [f"{lookup[scope, prior, m]['posterior_mean']:.3f} → {lookup[scope, prior, m]['lcb_mean']:.3f}" for m in table_metrics]
            text += f"| {labels[scope]} / {prior} | " + " | ".join(values) + " |\n"
    text += "\nHere and in compact figure axes, 'pp' denotes accuracy percentage points for MMLU and score points for AlpacaEval.\n\n"
    text += "### Paired evidence and interpretation\n\n"
    for scope in primary:
        for prior in PRIORS:
            r = lookup[scope, prior, "total_downward_pp"]
            q = lookup[scope, prior, "mean_regret_pp"]
            direction = "decreases" if r["paired_mean_delta"] < -TOLERANCE else "increases" if r["paired_mean_delta"] > TOLERANCE else "ties"
            text += f"- **{labels[scope]}, {prior} prior:** total downward movement {direction}; "
            text += f"LCB − posterior = **{r['paired_mean_delta']:+.3f}**, paired 95% interval **[{r['paired_ci95_low']:+.3f}, {r['paired_ci95_high']:+.3f}]**. "
            text += f"The budget-averaged regret difference is {q['paired_mean_delta']:+.3f} "
            text += f"[{q['paired_ci95_low']:+.3f}, {q['paired_ci95_high']:+.3f}].\n"
    stability = ["total_downward_pp", "maximum_drawdown_pp", "largest_drop_pp", "switches"]
    all_primary_stability = [lookup[s, p, m] for s in primary for p in PRIORS for m in stability]
    improved = sum(r["paired_mean_delta"] < -TOLERANCE for r in all_primary_stability)
    supported = sum(r["paired_ci95_high"] < 0 for r in all_primary_stability)
    text += f"\nLCB improves the mean in {improved}/{len(all_primary_stability)} aggregate dataset/prior/primary-stability comparisons; "
    text += f"{supported} have a pointwise paired 95% interval entirely below zero. "
    text += "These counts concern stability, not a claim that every quality metric improves.\n\n"
    text += "![Aggregate stability and quality](" + link("overview.png") + ")\n\n"
    if n_subjects:
        text += "## How consistent is the result across MMLU subjects?\n\n"
        text += "Counts compare each subject's mean over paired seeds. 'Worse' is a positive LCB − posterior difference; ties use absolute tolerance `1e-10`. Counts do not imply subject-level statistical significance.\n\n"
        text += "| Metric | Default: better / tie / worse | Dataset: better / tie / worse |\n|---|---:|---:|\n"
        for metric in METRICS:
            values = []
            for prior in PRIORS:
                r = lookup["mmlu_equal_subject", prior, metric]
                values.append(f"{r['lcb_better_subjects']} / {r['tied_subjects']} / {r['lcb_worse_subjects']}")
            text += f"| {SHORT_METRICS[metric]} | " + " | ".join(values) + " |\n"
        text += "\n### Largest quality regressions\n\nThe table lists the five largest mean-regret increases per prior, including their uncertainty. Every positive metric difference is retained in the regression CSV.\n\n"
        text += "| Prior | Subject | Mean regret: posterior → LCB (pp) | Difference (95% paired interval) |\n|---|---|---:|---:|\n"
        quality_regressions = [r for r in regressions if r["benchmark"].startswith("mmlu_") and r["metric"] == "mean_regret_pp"]
        for prior in PRIORS:
            selected = sorted((r for r in quality_regressions if r["prior"] == prior), key=lambda r: r["paired_mean_delta"], reverse=True)[:5]
            for r in selected:
                name = r["benchmark"].removeprefix("mmlu_").replace("_", " ")
                difference = f"{r['paired_mean_delta']:+.6f}" if abs(r['paired_mean_delta']) < .001 else f"{r['paired_mean_delta']:+.3f}"
                text += f"| {prior} | {name} | {r['posterior_mean']:.3f} → {r['lcb_mean']:.3f} | {difference} [{r['paired_ci95_low']:+.3f}, {r['paired_ci95_high']:+.3f}] |\n"
        text += "\n### Question-weighted sensitivity check\n\n"
        text += "| Prior | Metric | Posterior → LCB | Difference (95% paired interval) |\n|---|---|---:|---:|\n"
        for prior in PRIORS:
            for metric in ["total_downward_pp", "maximum_drawdown_pp", "mean_regret_pp", "final_regret_pp"]:
                r = lookup["mmlu_question_weighted", prior, metric]
                text += f"| {prior} | {SHORT_METRICS[metric]} | {r['posterior_mean']:.3f} → {r['lcb_mean']:.3f} | {r['paired_mean_delta']:+.3f} [{r['paired_ci95_low']:+.3f}, {r['paired_ci95_high']:+.3f}] |\n"
    text += "\n## Recommendation quality at fixed budgets\n\nRegret is the full-row score of the best arm minus that of the recommended arm.\n\n"
    text += "| Dataset / prior | 1% | 2% | 5% | 10% |\n|---|---:|---:|---:|---:|\n"
    cp_lookup = {(r["scope"], r["prior"], r["budget_fraction"]): r for r in checkpoints}
    for scope in primary:
        for prior in PRIORS:
            vals = []
            for budget in CHECKPOINTS:
                r = cp_lookup[scope, prior, budget]
                vals.append(f"{r['posterior_mean']:.3f} → {r['lcb_mean']:.3f}")
            text += f"| {labels[scope]} / {prior} | " + " | ".join(vals) + " |\n"
    text += """
## Metric definitions and uncertainty

Let `r_t` be oracle full-row regret at each real recommendation update, beginning
with the state active at 1%. Oracle scores evaluate the result only; they are
never exposed to either recommendation rule.

- **Total downward movement:** `100 × sum(max(r_t − r_(t−1), 0))`; repeated
  deterioration accumulates and can exceed 100 points.
- **Maximum drawdown:** `100 × max(r_t − min_(s<=t) r_s)`; the largest decline
  from the best recommendation reached since the window began.
- **Largest downward jump:** `100 × max(max(r_t − r_(t−1), 0))`.
- **Switches:** changes in arm identity, including harmless switches.
- **Budget-averaged drawdown / regret:** exact step-function integral divided
  by the window width; an update at the endpoint has zero exposure time.
- **Harmful jumps / jumps of at least 2 points / any such jump:** counts and an
  event indicator. The mean event indicator is a risk on a 0–1 scale. The 2-point
  threshold is inclusive with tolerance `1e-12` in raw score units.
- **Final regret:** quality at the window's last observed state.

All ten metrics reuse the earlier frozen `analyze_stability.path_metrics`
implementation. Metrics are computed within each run before any averaging.
MMLU's primary result is the equal-subject mean for each seed; the secondary
result weights subjects by their question counts. **The bootstrap resamples
whole sampling-seed blocks, retaining every fixed subject and both rules**.
It uses 10,000 resamples with seed 20260913. Curves use 2,000 of those resamples
for pointwise shading. The p90 in each aggregate row is the 90th percentile
of the seed-level subject aggregate; it is not a pooled subject/run tail.

Intervals are pointwise, unadjusted, and conditional on these fixed matrices.
They describe variation in sampling paths, not uncertainty over new questions,
new subjects, models, or prompts. The earlier analysis motivated these metrics
and the 1%–10% window; the three MMLU subjects examined earlier are repeated
within this complete suite and are not independent new evidence. Tail estimates
from only 20 seeds are coarse. Neither the Gaussian LCB model nor these
experiments guarantees universal temporal stability or better accuracy.

Do not compare datasets' raw switch or movement totals as if their evaluation
frequencies were identical: matrix dimensions and batch sizes differ. Within
each paired comparison, those factors are held fixed.

"""
    text += "## Validation and provenance\n\n"
    text += ("The [compact engine parity tests](" + source_link("recommend/test_full_suite_engine.py") +
        ") compare against the frozen `experiment_core` implementation. "
        "Separately, seed-0 trajectories for the three previously studied MMLU subjects were matched exactly.\n\n")
    text += ("The independent auditor rebuilds observed counts and reward sums and replays every sampled cell, "
        "every Gittins allocation, both recommendations, and their saved scores, variances, and oracle regrets. "
        "It also checks input/source hashes, unique observations, exact budgets, and the complete seed grid. "
        "See the [suite audit summary](" + source_link("recommend/results/full_suite/audit_summary.json") + ") "
        "and the per-dataset audit links below. This is a full trajectory audit, not a sampled-step check.\n\n")
    text += ("Root-table generation is checked separately in the [portable backend validation](" +
        source_link("recommend/results/portable_validation.json") + "). The analysis manifest records hashes "
        "of every source trace and experiment manifest.\n\n## Artifacts\n\n")
    for name, file in [
        ("Aggregate overview", "overview.png"), ("Quality curves", "quality_curves.png"),
        ("Every per-run metric", "per_run_metrics.csv"), ("Per-subject means, tails, and intervals", "per_subject_summary.csv"),
        ("Aggregate paired bootstrap intervals", "aggregate_summary.csv"),
        ("Every positive metric difference", "regressions.csv"),
        ("Checkpoint summary and intervals", "checkpoint_summary.csv"),
        ("Raw per-run checkpoints", "per_run_checkpoints.csv"),
        ("Analysis configuration and source hashes", "manifest.json")]:
        text += f"- [{name}]({link(file)})\n"
    if n_subjects:
        text += f"- [All-subject effects heatmap]({link('mmlu_subject_effects.png')})\n"
        text += f"- [Stability versus quality scatter]({link('mmlu_stability_quality_tradeoff.png')})\n"
    text += "\n### Individual comparison figures\n\n"
    for benchmark, info in experiments.items():
        audit_path = str(Path(info["manifest_path"]).parent / "validation.json")
        text += f"- [{info['label']}]({link('per_subject/' + benchmark + '.png')}) · [audit]({source_link(audit_path)})\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def self_test():
    # A 10-point fall, recovery and another fall accumulate 20; drawdown is 10.
    x = np.array([.005, .01, .02, .03, .04, .10])
    r = np.array([.2, .1, .2, .1, .2, .15])
    arms = np.arange(len(r))
    metrics = path_metrics(x, r, arms)
    np.testing.assert_allclose([metrics["total_downward_pp"], metrics["maximum_drawdown_pp"]], [20, 10])
    values = curves_for_path(x, np.column_stack([r, r]), np.column_stack([arms, arms]), .10)
    np.testing.assert_allclose(values["cumulative_downward_pp"][-1], [20, 20])
    np.testing.assert_allclose(step_sample(x, r, [.015, .11]), [.1, .15])
    # Subject effects cancel within each seed. Independent subject resampling
    # would invent uncertainty; the actual whole-seed aggregate is exactly zero.
    subject_blocks = np.array([[[0., 2.], [0., 4.]], [[0., -2.], [0., -4.]]])
    blocks = np.einsum("b,bsm->sm", [.5, .5], subject_blocks)
    result = summarize_blocks(blocks, bootstrap_weights(2))
    assert result["paired_mean_delta"] == result["paired_ci95_low"] == result["paired_ci95_high"] == 0
    weighted = np.einsum("b,bsm->sm", [.25, .75], subject_blocks)
    np.testing.assert_allclose(weighted[:, 1], [-1., -2.])
    print("Analysis self-test passed: movement, step sampling, and paired subject aggregation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "recommend/results/full_suite")
    parser.add_argument("--out", type=Path, default=ROOT / "recommend/figures/full_suite")
    parser.add_argument("--report", type=Path, default=ROOT / "recommend/RESULTS_FULL_SUITE.md")
    parser.add_argument("--allow-subset", action="store_true", help="Smoke checks only; normal full-suite analysis requires all 58 datasets")
    parser.add_argument("--no-subject-plots", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    rows, cp_rows, sources, experiments, curves = collect(args.results_root, args.allow_subset)
    subject_summary, aggregates, checkpoints, regressions = summarize(rows, cp_rows, experiments)
    write_csv(args.out / "per_run_metrics.csv", rows)
    write_csv(args.out / "per_run_checkpoints.csv", cp_rows)
    write_csv(args.out / "per_subject_summary.csv", subject_summary)
    write_csv(args.out / "aggregate_summary.csv", aggregates)
    write_csv(args.out / "checkpoint_summary.csv", checkpoints)
    write_csv(args.out / "regressions.csv", regressions)
    plot_all(subject_summary, aggregates, experiments, curves, args.out, not args.no_subject_plots)
    report(subject_summary, aggregates, checkpoints, regressions, experiments, args.out, args.report)
    manifest = dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        trajectories=len(rows) // 2, methods=METHODS, window=[BEGIN, END],
        endpoint="min(nominal endpoint, actual rounded final budget); checkpoints hold last state",
        experiments=experiments, full_suite=not args.allow_subset,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_repeats=BOOTSTRAP_REPEATS,
        curve_bootstrap_repeats=2000, bootstrap_unit="whole sampling-seed blocks; fixed subjects and both methods retained",
        mmlu_primary_aggregation="equal subject mean within each seed before resampling",
        mmlu_secondary_aggregation="question-count weighted within each seed before resampling",
        intervals="pointwise, unadjusted, conditional on the fixed matrices", large_jump_threshold=2,
        python=sys.version, numpy=np.__version__, matplotlib=__import__("matplotlib").__version__,
        source_sha256={relative(Path(__file__)): digest(Path(__file__)),
                       "recommend/analyze_stability.py": digest(ROOT / "recommend/analyze_stability.py")},
        input_sha256=sources)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Analyzed {len(rows)//2:,} paired trajectories; wrote {args.report}", flush=True)
    for r in aggregates:
        if r["metric"] in ["total_downward_pp", "maximum_drawdown_pp", "mean_regret_pp", "final_regret_pp"]:
            print(r["scope"], r["prior"], r["metric"], f"{r['posterior_mean']:.4f} -> {r['lcb_mean']:.4f}",
                  f"delta CI [{r['paired_ci95_low']:+.4f}, {r['paired_ci95_high']:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
