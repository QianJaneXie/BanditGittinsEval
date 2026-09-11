#!/usr/bin/env python3
"""Analyze raw paired recommendation traces without smoothing or rebinning.

Example:
  .venv/bin/python scripts/analyze_recommendation_oscillation.py \
    --out-dir outputs/recommendation_lcb_gsm8k_seed1_20seeds

The source traces are produced by compare_recommendation_rules.py. A seed is a
sampling run on one fixed matrix, not an independently generated reward matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

ZERO_TOL = 1e-12


def mean_se(values):
    values = np.asarray([x for x in values if x is not None], dtype=float)
    return {
        "mean": float(values.mean()) if values.size else None,
        "se": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else None,
        "n": int(values.size),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def held_indices(evaluations, grid):
    indices = np.searchsorted(evaluations, grid, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Cannot hold a recommendation before its first observation.")
    return indices


def clipped_trace(evaluations, values, start, end):
    """Include both domain endpoints; hold the most recent post-pull value."""
    grid = np.unique(np.r_[start, evaluations[(evaluations > start) & (evaluations < end)], end])
    return grid, values[held_indices(evaluations, grid)]


def budget_mean(evaluations, values):
    return float(np.dot(np.diff(evaluations), values[:-1]) / (evaluations[-1] - evaluations[0]))


def regret_metrics(evaluations, regret):
    delta = np.diff(regret)
    zero = regret <= ZERO_TOL
    first_zero = int(evaluations[np.flatnonzero(zero)[0]]) if zero.any() else None
    last_nonzero = np.flatnonzero(~zero)
    settles_zero = (int(evaluations[last_nonzero[-1] + 1]) if last_nonzero.size else int(evaluations[0])) if zero[-1] else None
    return {
        "upward_regret_jumps": int(np.sum(delta > ZERO_TOL)),
        "regret_total_variation": float(np.abs(delta).sum()),
        "budget_weighted_mean_regret": budget_mean(evaluations, regret),
        "final_regret": float(regret[-1]),
        "first_zero_evaluations": first_zero,
        "settles_zero_evaluations": settles_zero,
    }


def per_seed_metrics(raw, rule, start, end):
    evaluations = raw["evaluations"]
    grid, regret = clipped_trace(evaluations, raw["regret"][:, rule], start, end)
    indices = held_indices(evaluations, grid)
    arms = raw["recommendation"][indices, rule]
    unobserved = raw["recommended_count"][indices, rule] == 0
    raw_mask = (evaluations >= start) & (evaluations <= end)
    return {
        "raw_post_pull_states": int(raw_mask.sum()),
        "recommendation_switches": int(np.count_nonzero(arms[1:] != arms[:-1])),
        **regret_metrics(grid, regret),
        "unobserved_recommendation_fraction_raw_states": float((raw["recommended_count"][raw_mask, rule] == 0).mean()),
        "unobserved_recommendation_budget_fraction": budget_mean(grid, unobserved.astype(float)),
    }


def load_sources(out, prior_type, expected_seeds):
    summary = json.loads((out / "summary.json").read_text())
    setups = [x for x in summary["setups"] if x["prior_type"] == prior_type]
    if len(setups) != 1:
        raise ValueError("Analysis expects exactly one matrix for the requested prior.")
    setup = setups[0]
    with np.load(out / f"{setup['dataset']}_{prior_type}.npz", allow_pickle=False) as data:
        aggregate = {key: data[key] for key in data.files}
    seeds = aggregate["seeds"].astype(int)
    seed_start = int(summary["config"].get("seed_start", 0))
    if not np.array_equal(seeds, np.arange(seed_start, seed_start + expected_seeds)):
        raise ValueError(f"Expected {expected_seeds} consecutive run seeds from {seed_start}, found {seeds.tolist()}.")
    raw = []
    for seed in seeds:
        path = out / "raw" / f"{setup['dataset']}_{prior_type}_seed{seed}.npz"
        with np.load(path, allow_pickle=False) as data:
            history = {key: data[key] for key in data.files}
        x = history["evaluations"]
        if x.ndim != 1 or x.size < 2 or not np.all(np.diff(x) > 0):
            raise ValueError(f"Invalid raw evaluation sequence: {path}")
        for key in ("regret", "recommendation", "recommended_count"):
            if history[key].shape != (len(x), len(aggregate["rules"])):
                raise ValueError(f"Invalid {key} shape in {path}")
        if not np.isfinite(history["regret"]).all():
            raise ValueError(f"Nonfinite regret in {path}")
        raw.append(history)
    return summary, setup, aggregate, seeds, raw


def plot_artifacts(out, setup, config, seeds, rules, grid, regrets, raw):
    os.environ.setdefault("MPLCONFIGDIR", str((out / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((out / ".cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#34445d", "#008579", "#c06a35"]
    n_cells = int(np.prod(setup["shape"]))
    x = grid / n_cells * 100
    n = len(seeds)
    prior_label = "general" if setup["prior_type"] == "default" else "data-specific"
    dataset = setup["dataset"]
    dataset_label = "GSM8K" if dataset.lower().startswith("gsm8k") else dataset.replace("_", " ")
    matrix_seed = re.search(r"(?:^|_)seed(\d+)(?:_|$)", dataset)
    matrix_label = f"fixed matrix seed {matrix_seed.group(1)}" if matrix_seed else "fixed reward matrix"
    title = (f"{dataset_label} · {matrix_label} · {n} run seeds {seeds[0]}–{seeds[-1]}\n"
             f"{prior_label.capitalize()} prior N({setup['prior_mean']:g}, {setup['prior_variance']:g})"
             f" · batch size {config['batch_size']} · paired Gittins observations")
    early_percent = float(config["analysis_early_fraction"]) * 100
    full_percent = grid[-1] / n_cells * 100

    def band(ax, values, color, label):
        mean = values.mean(axis=0)
        error = values.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        ax.step(x, mean, where="post", color=color, label=label, linewidth=1.35)
        ax.fill_between(x, mean - error, mean + error, step="post", color=color, alpha=.15, linewidth=0)

    for paired in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
        for ax, limit in zip(axes, (full_percent, early_percent)):
            for rule in range(1 if paired else 0, len(rules)):
                values = regrets[:, :, rule] - regrets[:, :, 0] if paired else regrets[:, :, rule]
                band(ax, values, colors[rule % len(colors)], str(rules[rule]))
            if paired:
                ax.axhline(0, color="#687180", linewidth=.8)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlim(0, limit)
            ax.set_title(f"0–{limit:g}% evaluation budget")
            ax.set_xlabel("Evaluated matrix cells (%)")
            ax.set_ylabel("LCB − posterior mean: simple regret" if paired else "Simple regret (accuracy gap)")
            ax.grid(alpha=.18)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .045), ncol=len(labels), frameon=False)
        fig.suptitle(title, fontsize=11)
        fig.text(.5, .012, "Unsmoothed mean ± 1 SE across paired run seeds; negative differences favor LCB."
                 if paired else "Unsmoothed mean ± 1 SE across run seeds; exact event grid with post-pull values held between events.",
                 ha="center", fontsize=9, color="#4c5665")
        fig.tight_layout(rect=(0, .12, 1, .87))
        name = "aggregate_paired_regret_difference" if paired else "aggregate_simple_regret"
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=190)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    for rule in range(len(rules)):
        switches, variations = [], []
        for history in raw:
            indices = held_indices(history["evaluations"], grid)
            arms = history["recommendation"][:, rule]
            regret = history["regret"][:, rule]
            switches.append(np.r_[0, np.cumsum(arms[1:] != arms[:-1])][indices])
            variations.append(np.r_[0., np.cumsum(np.abs(np.diff(regret)))][indices])
        for ax, values in zip(axes, (switches, variations)):
            band(ax, np.asarray(values), colors[rule % len(colors)], str(rules[rule]))
    for ax, ylabel in zip(axes, ("Cumulative recommendation switches", "Cumulative simple-regret total variation")):
        ax.set_xlim(0, full_percent)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Evaluated matrix cells (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .045), ncol=len(labels), frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.text(.5, .012, "Each seed is counted on its raw trajectory, then averaged; shading is ± 1 SE across seeds.",
             ha="center", fontsize=9, color="#4c5665")
    fig.tight_layout(rect=(0, .12, 1, .87))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"aggregate_oscillation.{extension}", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prior-type", choices=("default", "dataset"), default="default")
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--early-fraction", type=float, default=.03)
    args = parser.parse_args()
    summary, setup, aggregate, seeds, raw = load_sources(args.out_dir, args.prior_type, args.expected_seeds)
    n_cells = int(np.prod(setup["shape"]))
    start = max(int(history["evaluations"][0]) for history in raw)
    end = min(int(setup["evaluation_budget"]), min(int(history["evaluations"][-1]) for history in raw))
    early_end = min(end, int(n_cells * args.early_fraction))
    if not start < early_end <= end:
        parser.error("Early window must end after the first observation and at or before the common end budget.")
    grid = np.unique(np.r_[start, np.concatenate([history["evaluations"] for history in raw]), end])
    grid = grid[(grid >= start) & (grid <= end)]
    regrets = np.stack([history["regret"][held_indices(history["evaluations"], grid)] for history in raw])
    saved_x = aggregate["evaluations"]
    saved_x = saved_x[0] if saved_x.ndim == 2 else saved_x
    saved_indices = np.searchsorted(saved_x, grid)
    if not np.array_equal(saved_x[saved_indices], grid):
        raise ValueError("Saved aggregate grid does not include every raw event.")
    if not np.array_equal(aggregate["regret"][:, saved_indices], regrets):
        raise ValueError("Saved aggregate regret is inconsistent with raw forward-filled observations.")

    rules = [str(x) for x in aggregate["rules"]]
    per_seed, aggregate_rows, distribution_rows, paired_rows = [], [], [], []
    windows = [("full", start, end), ("early", start, early_end)]
    if start < int(.01 * n_cells) < end:
        windows.append(("after_1pct", int(.01 * n_cells), end))
    for window, left, right in windows:
        window_rows = []
        for seed, history in zip(seeds, raw):
            for rule, label in enumerate(rules):
                row = dict(window=window, start_evaluations=left, end_evaluations=right,
                           seed=int(seed), rule=label, **per_seed_metrics(history, rule, left, right))
                window_rows.append(row)
                per_seed.append(row)
        for rule, label in enumerate(rules):
            aggregate_x, aggregate_regret = clipped_trace(grid, regrets[:, :, rule].mean(axis=0), left, right)
            aggregate_rows.append(dict(window=window, rule=label, start_evaluations=left,
                                       end_evaluations=right, **regret_metrics(aggregate_x, aggregate_regret)))
            rows = [row for row in window_rows if row["rule"] == label]
            for metric in rows[0]:
                if metric in ("window", "start_evaluations", "end_evaluations", "seed", "rule"):
                    continue
                distribution_rows.append(dict(window=window, rule=label, metric=metric,
                                              **mean_se([row[metric] for row in rows])))
        for rule in range(1, len(rules)):
            baseline = [row for row in window_rows if row["rule"] == rules[0]]
            comparison = [row for row in window_rows if row["rule"] == rules[rule]]
            for metric in baseline[0]:
                if metric in ("window", "start_evaluations", "end_evaluations", "seed", "rule", "raw_post_pull_states"):
                    continue
                pairs = [(a[metric], b[metric]) for a, b in zip(baseline, comparison)
                         if a[metric] is not None and b[metric] is not None]
                deltas = np.asarray([b - a for a, b in pairs], dtype=float)
                paired_rows.append(dict(window=window, rule=rules[rule], baseline_rule=rules[0], metric=metric,
                                        **mean_se(deltas), improved_seeds=int(np.sum(deltas < -ZERO_TOL)),
                                        tied_seeds=int(np.sum(np.abs(deltas) <= ZERO_TOL)),
                                        worsened_seeds=int(np.sum(deltas > ZERO_TOL))))

    config = dict(summary["config"], analysis_early_fraction=args.early_fraction)
    payload = {
        "setup": setup, "run_seeds": seeds.tolist(), "rules": rules,
        "common_domain_evaluations": [start, end], "exact_union_grid_points": int(grid.size),
        "definitions": {
            "recommendation": "Post-pull posterior mean, or posterior mean minus posterior standard deviation; paired on identical sampling observations within each seed.",
            "regret": "Best full-matrix arm mean minus recommended arm full-matrix mean.",
            "alignment": "Exact union of raw evaluation events; forward hold the latest post-pull recommendation. No smoothing, interpolation, or rebinning.",
            "budget_weighted_mean_regret": "Integral of the post-pull step-held regret over the stated domain, divided by domain length. The final point has no positive-duration interval.",
            "raw_counts": "Per-seed switches and upward jumps are counted from actual post-pull traces, not duplicated union-grid states; no initial recommendation is invented before the first batch.",
            "total_variation": "Sum of absolute consecutive simple-regret differences. Aggregate curve TV is computed after averaging, and differs from mean per-seed TV.",
            "first_zero_evaluations": "First point with regret <= 1e-12 in the clipped window (including a held boundary value); null if absent.",
            "settles_zero_evaluations": "First point after which regret remains <= 1e-12 through the clipped window; null if final regret is nonzero. This describes the observed finite window only.",
            "paired_statistics": "Comparison minus baseline per seed. SE is sample SD/sqrt(n); improved means a smaller metric. First/settles-zero pairs omit seeds with missing values; n is reported.",
            "unobserved_fraction": "Both raw-state fraction and evaluation-budget-weighted fraction are reported; these are not interchangeable.",
            "uncertainty": "Mean ± 1 SE across sampling run seeds on one fixed reward matrix; not a confidence band across independent matrices.",
        },
        "aggregate_curve_metrics": aggregate_rows,
        "per_seed_metric_distributions": distribution_rows,
        "paired_seed_metric_deltas": paired_rows,
        "per_seed_metrics": per_seed,
    }
    (args.out_dir / "oscillation_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for name, rows in (("oscillation_per_seed", per_seed), ("oscillation_aggregate", aggregate_rows),
                       ("oscillation_seed_distributions", distribution_rows), ("oscillation_paired_deltas", paired_rows)):
        write_csv(args.out_dir / f"{name}.csv", rows)
    plot_artifacts(args.out_dir, setup, config, seeds, rules, grid, regrets, raw)
    print(json.dumps({"aggregate_curve_metrics": aggregate_rows,
                      "paired_seed_metric_deltas_full": [x for x in paired_rows if x["window"] == "full"]}, indent=2))
    print(f"Saved analysis and PNG/PDF plots in {args.out_dir}")


if __name__ == "__main__":
    main()
