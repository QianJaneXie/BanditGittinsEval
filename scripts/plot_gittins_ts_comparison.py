#!/usr/bin/env python3
"""Plot the paired, paper-matched GSM8K Gittins/TS experiment.

Figures follow the paper's 350-point linear interpolation and missing-range
convention. Numerical checkpoints and budget-average regret use completed
observations only, with exact step-hold integration and no future feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

METHODS = ("gittins", "ts")
PRIORS = {"G": (0.5, 0.04), "S": (0.2, 0.01)}
MODES = ("unit", "cost")
CHECKPOINTS = (0.01, 0.02, 0.05, 0.10)
CASE_PATTERN = re.compile(r"matrix(\d+)_seed(\d+)_([GS])_(unit|cost)\.npz$")


def paper_protocol() -> dict:
    """Fixed protocol: keep separate from mutable progress/results metadata."""
    return {
        "dataset": "GSM8K", "n_arms": 122, "n_items": 1000,
        "matrix_seeds": [1, 2, 3, 4, 5], "run_seeds": list(range(20)),
        "paired_trials_per_panel": 100, "panels": 4, "policy_trajectories": 800,
        "batch_size": 8, "nominal_exhaustive_budget_fraction": 0.1,
        "priors": {name: {"mean": mean, "variance": variance}
                   for name, (mean, variance) in PRIORS.items()},
        "target": "complete_fixed_benchmark_empirical_mean",
        "recommendation_for_both_methods": "posterior_mean_minus_one_posterior_std",
        "recommendation_std_penalty": 1.0,
        "per_cell_noise_variance": 0.25, "batch_mean_noise_variance": 0.03125,
        "gittins_cost_scaling_factor": 1e-4, "gittins_grid_points": 1025,
        "gittins_model_version": "batch_level_paper_v1",
        "gittins_allocation_implementation": "batch_level_empirical_target_DP",
        "gittins_dp_horizon": 125,
        "gittins_root_columns": 126,
        "gittins_transition_cost": "lambda * actual_batch_size * raw_per_cell_arm_price",
        "allocation_discretization_note": (
            "Each transition observes one batch with noise tau_cell_squared / actual_batch_size "
            "and pays lambda * actual_batch_size * raw_per_cell_arm_price. Roots are indexed "
            "by completed local batches; completed arms have terminal root zero. Numerical "
            "gridding, continuation past stopping, and LCB recommendation remain outside "
            "the paper's exact batch-level optimality claim."
        ),
        "ts_cost_allocation": "cost_unaware_accuracy_posterior_draws",
        "cost_budget_for_both_methods": "actual_sum_of_new_items_times_arm_price",
        "heterogeneous_cost_proxy": "input_price_plus_2_times_output_price",
        "unit_exhaustive_cost": 122000.0,
        "heterogeneous_exhaustive_cost": 83900.0,
        "stopping": "record_natural_stop_but_continue_to_nominal_budget",
        "figure_aggregation": {
            "kind": "linear_interpolation", "grid_points": 350,
            "outside_observed_range": "missing_without_right_extension",
            "bands": "one_standard_error_over_available_runs",
        },
        "numerical_metrics": {
            "checkpoints": "last_completed_observation_at_or_before_budget",
            "AUC": "exact_step_hold_integral_divided_by_budget_interval",
            "before_first_observation": "prior_tie_recommends_arm_0",
            "over_budget_observations": "excluded",
        },
        "uncertainty_note": (
            "100 paired algorithm trials are nested within five response matrices; "
            "they are not 100 independent datasets. Paired-run SE is descriptive for "
            "these repetitions. A two-way bootstrap independently resamples five "
            "matrix IDs and 20 trial-seed IDs, using the same sampled seed IDs across "
            "matrices and keeping policy pairs together. With only five matrices, "
            "these intervals are descriptive, not evidence from 100 independent datasets."
        ),
        "paired_difference_bootstrap": {"repetitions": 5000, "seed": 20261106,
            "scheme": "two_way_matrix_and_shared_trial_seed_resampling"},
        "paper_sources": [
            {"path": "../BanditGittinsEval_Overleaf/ICLR/experiment.tex", "lines": [13, 18, 20, 28, 41]},
            {"path": "../BanditGittinsEval_Overleaf/ICLR/method.tex", "lines": [257, 266, 270, 282]},
            {"path": "../BanditGittinsEval_Overleaf/appendices/experiment_setup.tex", "lines": [8, 12, 15, 27, 47, 117, 118]},
            {"path": "../BanditGittinsEval_Overleaf/appendices/experiment_results.tex", "lines": [58]},
            {"path": "../BanditGittinsEval_Overleaf/appendices/anytime_recommendation.tex", "lines": [82, 105, 110]},
        ],
    }


def mean_and_se(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    centered = values - np.expand_dims(mean, axis=axis)
    squares = np.nansum(centered**2, axis=axis)
    variance = np.divide(squares, count - 1, out=np.full_like(total, np.nan), where=count > 1)
    se = np.sqrt(np.divide(variance, count, out=np.full_like(total, np.nan), where=count > 1))
    return mean, se


def scalar(archive, key: str, fallback=None):
    return np.asarray(archive[key]).item() if key in archive.files else fallback


def previous_regret(x: np.ndarray, y: np.ndarray, budget: float, initial: float) -> float:
    tolerance = 32 * np.finfo(float).eps * max(1.0, abs(budget))
    index = int(np.searchsorted(x, budget + tolerance, side="right") - 1)
    return float(initial if index < 0 else y[index])


def budget_average_regret(x: np.ndarray, y: np.ndarray, end: float, initial: float) -> float:
    """Exact mean regret over [0, end], using recommendations after feedback."""
    tolerance = 32 * np.finfo(float).eps * max(1.0, abs(end))
    valid = (x > 0) & (x <= end + tolerance)
    points = np.r_[0.0, np.minimum(x[valid], end)]
    values = np.r_[initial, y[valid]]
    if points[-1] < end:
        points = np.r_[points, end]
        values = np.r_[values, values[-1]]
    return float(np.dot(values[:-1], np.diff(points)) / end)


def load_cases(directory: Path) -> list[dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for key, expected in (("batch_size", 8), ("recommendation_std_penalty", 1.0),
                          ("per_cell_noise_variance", 0.25), ("cost_scaling_factor", 1e-4),
                          ("gittins_grid_points", 1025), ("budget_fraction", 0.1),
                          ("n_models", 122), ("n_examples", 1000),
                          ("gittins_model_version", "batch_level_paper_v1"),
                          ("gittins_dp_transitions", "batch"), ("gittins_dp_horizon", 125)):
        if manifest.get(key) != expected:
            raise ValueError(f"Input manifest does not match the paper protocol: {key}")
    if manifest.get("gittins_batch_sizes") != [8] * 125:
        raise ValueError("Input manifest does not contain the paper's B=8 batch schedule")
    if manifest.get("priors") != {name: list(values) for name, values in PRIORS.items()}:
        raise ValueError("Input priors differ from the paper's fixed prior settings")
    with np.load(directory / "roots.npz", allow_pickle=False) as roots:
        if scalar(roots, "settings_hash") != manifest.get("settings_hash"):
            raise ValueError("Root cache does not match the input manifest")
        for prior in PRIORS:
            for mode in MODES:
                values = roots[f"{prior}_{mode}_roots"]
                if values.shape != (122, 126) or not np.isfinite(values).all():
                    raise ValueError("Expected batch-level roots with 126 stage columns")
                np.testing.assert_array_equal(values[:, -1], 0.0)
    matrices = {
        seed: np.asarray(np.load(ROOT / "data/BanditEval_matrices" /
            f"gsm8k_1_samples_various_models_seed{seed}.npy", allow_pickle=False), dtype=np.float64)
        for seed in range(1, 6)
    }
    cases = []
    for path in sorted((directory / "runs").glob("matrix*_seed*_*.npz")):
        match = CASE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        matrix_seed, run_seed = map(int, match.group(1, 2))
        prior, mode = match.group(3, 4)
        means = matrices[matrix_seed].mean(axis=1)
        best, initial = float(means.max()), float(means.max() - means[0])
        expected_full = 122000.0 if mode == "unit" else 83900.0
        record = {"matrix_seed": matrix_seed, "run_seed": run_seed,
                  "prior": prior, "mode": mode, "path": str(path), "methods": {}}
        with np.load(path, allow_pickle=False) as z:
            if scalar(z, "settings_hash") != manifest.get("settings_hash"):
                raise ValueError(f"Case settings do not match the immutable input manifest: {path}")
            for field, expected in (("matrix_seed", matrix_seed), ("run_seed", run_seed),
                                    ("prior_name", prior), ("cost_mode", mode)):
                if scalar(z, field, expected) != expected:
                    raise ValueError(f"Case identity mismatch: {path}: {field}")
            for method in METHODS:
                key = f"{method}_x" if mode == "unit" else f"{method}_x_original_cost"
                full = float(scalar(z, f"{method}_full_budget", expected_full))
                if not np.isclose(full, expected_full, rtol=0, atol=1e-6):
                    raise ValueError(f"Unexpected exhaustive cost: {path}: {full}")
                raw_x = np.asarray(z[key], dtype=np.float64)
                arms = np.asarray(z[f"{method}_recommended_arm"], dtype=np.int64)
                recorded = np.asarray(z[f"{method}_regret"], dtype=np.float64)
                if raw_x.ndim != 1 or not raw_x.size or raw_x.shape != arms.shape:
                    raise ValueError(f"Missing or unaligned raw trajectory: {path}: {method}")
                if not np.isfinite(raw_x).all() or not np.all(np.diff(raw_x) > 0):
                    raise ValueError(f"Invalid budget sequence: {path}: {method}")
                if (arms < 0).any() or (arms >= len(means)).any():
                    raise ValueError(f"Invalid recommendation: {path}: {method}")
                y = best - means[arms]
                np.testing.assert_allclose(y, recorded, rtol=0, atol=1e-6)
                x = raw_x / full
                metrics = {f"regret_at_{100*b:g}%": previous_regret(x, y, b, initial)
                           for b in CHECKPOINTS}
                metrics.update({
                    "auc_0_2%": budget_average_regret(x, y, 0.02, initial),
                    "auc_0_10%": budget_average_regret(x, y, 0.10, initial),
                })
                if f"{method}_auc_normalized" in z.files:
                    np.testing.assert_allclose(metrics["auc_0_10%"],
                        float(scalar(z, f"{method}_auc_normalized")), rtol=0, atol=1e-12)
                record["methods"][method] = {
                    "x": x, "y": y, "initial_regret": initial, "metrics": metrics,
                    "actual_final_budget_fraction": float(x[-1]),
                    "actual_final_arm": int(arms[-1]),
                    "observation_rounds": int(len(x)),
                }
        cases.append(record)
    expected = {(matrix, seed, prior, mode)
                for matrix in range(1, 6) for seed in range(20)
                for prior in PRIORS for mode in MODES}
    identities = [(c["matrix_seed"], c["run_seed"], c["prior"], c["mode"]) for c in cases]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        missing = sorted(expected - set(identities))
        extra = sorted(set(identities) - expected)
        raise ValueError(f"Expected 400 completed cases; found {len(cases)}. Missing={missing[:8]}, extra={extra[:8]}")
    return cases


def paired_stats(difference: np.ndarray, matrix_ids: np.ndarray) -> dict:
    mean, se = mean_and_se(difference)
    matrices = []
    for matrix in sorted(set(matrix_ids.tolist())):
        block = difference[matrix_ids == matrix]
        local_mean, local_se = mean_and_se(block)
        matrices.append({"matrix_seed": int(matrix), "n_pairs": len(block),
                         "mean_TS_minus_Gittins": float(local_mean), "paired_se": float(local_se)})
    # Panel cases are sorted by matrix seed and then trial seed. Resampling
    # columns jointly across rows accounts for reuse of trial-seed identifiers.
    if len(matrices) != 5 or any(m["n_pairs"] != 20 for m in matrices):
        raise ValueError("Two-way bootstrap requires the complete 5 x 20 panel")
    table = difference.reshape(5, 20)
    rng = np.random.default_rng(20261106)
    row_ids = rng.integers(0, 5, size=(5000, 5))
    column_ids = rng.integers(0, 20, size=(5000, 20))
    bootstrap = table[row_ids[:, :, None], column_ids[:, None, :]].mean(axis=(1, 2))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n_pairs": len(difference), "mean_TS_minus_Gittins": float(mean),
        "paired_run_se": float(se), "ci95_lower": float(mean - 1.96 * se),
        "ci95_upper": float(mean + 1.96 * se),
        "TS_lower_fraction": float(np.mean(difference < -1e-12)),
        "tied_fraction": float(np.mean(np.abs(difference) <= 1e-12)),
        "Gittins_lower_fraction": float(np.mean(difference > 1e-12)),
        "ci95_two_way_bootstrap_lower": float(lower),
        "ci95_two_way_bootstrap_upper": float(upper),
        "bootstrap_repetitions": 5000, "bootstrap_seed": 20261106,
        "per_matrix": matrices,
    }


def summarize(cases: list[dict], directory: Path) -> dict:
    summary = {"protocol": paper_protocol(),
               "input_manifest": json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
               "panels": {}}
    checkpoint_rows, difference_rows, matrix_rows, run_rows = [], [], [], []
    for mode in MODES:
        for prior in PRIORS:
            panel_cases = sorted((c for c in cases if c["mode"] == mode and c["prior"] == prior),
                                 key=lambda c: (c["matrix_seed"], c["run_seed"]))
            panel = {"n_paired_trials": len(panel_cases), "methods": {}, "paired_differences": {}}
            matrix_ids = np.array([c["matrix_seed"] for c in panel_cases])
            metrics = list(panel_cases[0]["methods"]["gittins"]["metrics"])
            for method in METHODS:
                values = {metric: np.array([c["methods"][method]["metrics"][metric] for c in panel_cases])
                          for metric in metrics}
                method_summary = {}
                for metric, data in values.items():
                    mean, se = mean_and_se(data)
                    item = {"mean": float(mean), "se": float(se),
                            "ci95_lower": float(mean - 1.96 * se), "ci95_upper": float(mean + 1.96 * se)}
                    if metric.startswith("regret_at_"):
                        item["exact_best_recommendation_rate"] = float(np.mean(data <= 1e-12))
                    method_summary[metric] = item
                    checkpoint_rows.append({"cost_mode": mode, "prior": prior, "method": method,
                                            "metric": metric, "n": len(data), **item})
                panel["methods"][method] = method_summary
                for case in panel_cases:
                    raw = case["methods"][method]
                    run_rows.append({"cost_mode": mode, "prior": prior, "method": method,
                        "matrix_seed": case["matrix_seed"], "run_seed": case["run_seed"],
                        **raw["metrics"], "actual_final_budget_fraction": raw["actual_final_budget_fraction"],
                        "observation_rounds": raw["observation_rounds"]})
            for metric in metrics:
                delta = np.array([c["methods"]["ts"]["metrics"][metric]
                                  - c["methods"]["gittins"]["metrics"][metric] for c in panel_cases])
                item = paired_stats(delta, matrix_ids)
                panel["paired_differences"][metric] = item
                difference_rows.append({"cost_mode": mode, "prior": prior, "metric": metric,
                                        **{k: v for k, v in item.items() if k != "per_matrix"}})
                matrix_rows.extend({"cost_mode": mode, "prior": prior, "metric": metric, **row}
                                   for row in item["per_matrix"])
            summary["panels"][f"{mode}_{prior}"] = panel
    for name, rows in (("checkpoints.csv", checkpoint_rows), ("paired_differences.csv", difference_rows),
                       ("per_matrix_differences.csv", matrix_rows), ("run_metrics.csv", run_rows)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render(cases: list[dict], directory: Path, xmax: float, stem: str) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#253044", "text.color": "#253044",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"gittins": "#2363a5", "ts": "#16876b"}
    names = {"gittins": "Gittins", "ts": "Gaussian Thompson sampling"}
    grid = np.linspace(0.0, xmax, 350)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), sharex=True, sharey=True)
    plot_data = {"grid_fraction": grid}
    upper = 0.0
    for row, mode in enumerate(MODES):
        for col, prior in enumerate(PRIORS):
            ax = axes[row, col]
            panel_cases = [c for c in cases if c["mode"] == mode and c["prior"] == prior]
            for method in METHODS:
                runs = np.stack([np.interp(grid, c["methods"][method]["x"],
                    c["methods"][method]["y"], left=np.nan, right=np.nan) for c in panel_cases])
                mean, se = mean_and_se(runs)
                ax.plot(grid * 100, mean, label=names[method], color=colors[method], linewidth=1.8)
                ax.fill_between(grid * 100, np.maximum(0, mean - se), mean + se,
                                color=colors[method], alpha=0.17, linewidth=0)
                plot_data[f"{mode}_{prior}_{method}_mean"] = mean
                plot_data[f"{mode}_{prior}_{method}_se"] = se
                plot_data[f"{mode}_{prior}_{method}_n"] = np.isfinite(runs).sum(0)
                if np.isfinite(mean + se).any():
                    upper = max(upper, float(np.nanmax(mean + se)))
            mu, variance = PRIORS[prior]
            if row == 0:
                ax.set_title(f"{'General' if prior == 'G' else 'GSM8K'} prior: N({mu:g}, {variance:g})",
                             fontsize=12, fontweight="semibold", pad=12)
            if col == 0:
                ax.set_ylabel(f"{'Unit cost' if mode == 'unit' else 'Heterogeneous cost'}\nSimple regret", labelpad=9)
            ax.grid(axis="y", color="#dce4ed", linewidth=0.7)
            ax.set_xlim(0, xmax * 100)
            ax.set_axisbelow(True)
    for ax in axes.flat:
        ax.set_ylim(-0.003, max(0.02, upper * 1.06))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("GSM8K: empirical-target Gittins vs Gaussian TS", x=0.085, y=0.975,
                 ha="left", fontsize=17, fontweight="semibold")
    fig.text(0.085, 0.934,
        "Batch-level Gittins; both use LCB M - sqrt(V), B=8; five matrices x 20 paired trials",
        color="#64748b", fontsize=10.8)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.905),
               ncol=2, frameon=False)
    fig.supxlabel("Exhaustive-evaluation budget used (%)", y=0.065, fontsize=11)
    fig.text(0.085, 0.018,
        "Bands: +/-1 SE. Paper-style linear interpolation on 350 points; no extension outside each observed range.",
        color="#64748b", fontsize=8.8)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.13, hspace=0.20, wspace=0.15)
    fig.savefig(directory / f"{stem}.png", dpi=230, facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    np.savez_compressed(directory / f"{stem}_curves.npz", **plot_data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/gsm8k_gittins_ts_batch_lcb100")
    args = parser.parse_args()
    directory = args.results_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    protocol_path = directory / "paper_setup.json"
    protocol = paper_protocol()
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("paper_setup.json differs from the fixed protocol; use a new results directory")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    cases = load_cases(directory)
    summary = summarize(cases, directory)
    render(cases, directory, 0.10, "comparison")
    render(cases, directory, 0.02, "comparison_early")
    for name, panel in summary["panels"].items():
        final = panel["paired_differences"]["regret_at_10%"]
        auc = panel["paired_differences"]["auc_0_10%"]
        print(f"{name}: final TS-G={final['mean_TS_minus_Gittins']:.6g}; "
              f"budget-average TS-G={auc['mean_TS_minus_Gittins']:.6g}")
    print(f"Wrote paper-style plots and strict paired metrics under {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
