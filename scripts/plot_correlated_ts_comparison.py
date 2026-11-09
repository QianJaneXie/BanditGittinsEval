#!/usr/bin/env python3
"""Plot and summarize paired independent/correlated TS without future feedback."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

from plot_gittins_ts_comparison import (  # noqa: E402
    budget_average_regret, load_cases as load_gittins_cases, mean_and_se, previous_regret,
)
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ("ts", "correlated_ts")
THREE_WAY_METHODS = ("gittins", "ts", "correlated_ts")
PRIORS = {"G": (0.5, 0.04), "S": (0.2, 0.01)}
MODES = ("unit", "cost")
CASE_PATTERN = re.compile(r"matrix(\d+)_seed(\d+)_([GS])_(unit|cost)\.npz$")
CHECKPOINTS = (0.01, 0.02, 0.05, 0.10)


def _scalar(archive, key, default=None):
    return np.asarray(archive[key]).item() if key in archive.files else default


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


@contextmanager
def ts_trajectory_archive(archive):
    """Read the original TS trajectory in place without copying it to results."""
    if "ts_x" in archive:
        yield archive
        return
    if "baseline_reference_case" not in archive:
        raise ValueError("Missing original TS trajectory reference")
    path = Path(str(_scalar(archive, "baseline_reference_case")))
    expected = _scalar(archive, "baseline_reference_case_sha256")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Original baseline reference is missing or changed: {path}")
    with np.load(path, allow_pickle=False) as source:
        if _scalar(source, "settings_hash") != _scalar(archive, "baseline_reference_settings_hash"):
            raise ValueError("Original TS reference settings do not match saved provenance")
        for field in ("matrix_seed", "run_seed", "prior_name", "cost_mode"):
            if _scalar(source, field) != _scalar(archive, field):
                raise ValueError(f"Original TS reference has a different case identity: {field}")
        yield source


def load_cases(directory: Path, *, allow_smoke: bool = False) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for key, expected in (("model_version", "categorical_empirical_correlated_ts_v1"),
                          ("methods", list(METHODS)), ("recommendation_std_penalty", 1.0),
                          ("per_cell_noise_variance", 0.25)):
        if manifest.get(key) != expected:
            raise ValueError(f"Unexpected comparison protocol: {key}")
    if manifest.get("max_steps") is not None and not allow_smoke:
        raise ValueError("Step-capped smoke runs are not complete experiments; use --allow-smoke explicitly")
    matrices = {}
    for seed in manifest["matrix_seeds"]:
        path = ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{seed}.npy"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["matrix_sha256"][str(seed)]:
            raise ValueError(f"Response matrix changed since simulation: {path}")
        matrices[seed] = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    cases = []
    for path in sorted((directory / "runs").glob("matrix*_seed*_*.npz")):
        match = CASE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        matrix_seed, run_seed = map(int, match.group(1, 2))
        prior, mode = match.group(3, 4)
        if matrix_seed not in matrices:
            raise ValueError(f"Case matrix is outside the immutable manifest: {path}")
        truth = matrices[matrix_seed].mean(1)
        initial = float(truth.max() - truth[0])
        full_expected = float(manifest["unit_exhaustive_cost" if mode == "unit" else "heterogeneous_exhaustive_cost"])
        case = dict(matrix_seed=matrix_seed, run_seed=run_seed, prior=prior, mode=mode, methods={})
        with np.load(path, allow_pickle=False) as z, ts_trajectory_archive(z) as ts_source:
            if _scalar(z, "settings_hash") != manifest["settings_hash"]:
                raise ValueError(f"Case settings differ from manifest: {path}")
            for field, expected in (("matrix_seed", matrix_seed), ("run_seed", run_seed),
                                    ("prior_name", prior), ("cost_mode", mode)):
                if _scalar(z, field) != expected:
                    raise ValueError(f"Case identity differs from filename: {path}: {field}")
            for method in METHODS:
                raw = ts_source if method == "ts" else z
                full = float(_scalar(raw, f"{method}_full_budget"))
                np.testing.assert_allclose(full, full_expected, rtol=0, atol=1e-6)
                axis_name = "x" if mode == "unit" else "x_original_cost"
                x = np.asarray(raw[f"{method}_{axis_name}"], dtype=np.float64) / full
                arms = np.asarray(raw[f"{method}_recommended_arm"], dtype=np.int64)
                if not x.size or x.ndim != 1 or arms.shape != x.shape or not np.isfinite(x).all() or (np.diff(x) <= 0).any():
                    raise ValueError(f"Invalid raw history: {path}: {method}")
                if (arms < 0).any() or (arms >= len(truth)).any():
                    raise ValueError(f"Invalid recommendation: {path}: {method}")
                y = truth.max() - truth[arms]
                np.testing.assert_allclose(y, raw[f"{method}_regret"], rtol=0, atol=1e-12)
                metrics = {f"regret_at_{100*b:g}%": previous_regret(x, y, b, initial)
                           for b in CHECKPOINTS if b <= manifest["budget_fraction"]}
                for end in (0.02, 0.10):
                    if end <= manifest["budget_fraction"]:
                        metrics[f"auc_0_{100*end:g}%"] = budget_average_regret(x, y, end, initial)
                computed_auc = budget_average_regret(x, y, manifest["budget_fraction"], initial)
                np.testing.assert_allclose(computed_auc, float(_scalar(raw, f"{method}_auc_normalized")), rtol=0, atol=1e-12)
                case["methods"][method] = dict(
                    x=x, y=y, metrics=metrics,
                    final_budget_fraction=float(x[-1]), observation_rounds=len(x),
                    elapsed_s=float(_scalar(raw, f"{method}_elapsed_s")),
                    fit_count=int(_scalar(raw, f"{method}_fit_count", 0)),
                    stop_reason=str(_scalar(raw, f"{method}_stop_reason")),
                )
        cases.append(case)
    identities = [(c["matrix_seed"], c["run_seed"], c["prior"], c["mode"]) for c in cases]
    expected = {(m, s, p, mode) for m in manifest["matrix_seeds"]
                for s in manifest["run_seeds"] for p in PRIORS for mode in MODES}
    if len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError(f"Expected {len(expected)} complete cases; found {len(cases)}")
    return manifest, cases


def paired_stats(values: np.ndarray, cases: list[dict]) -> dict:
    mean, se = mean_and_se(values)
    matrix_ids = sorted({c["matrix_seed"] for c in cases})
    seeds = sorted({c["run_seed"] for c in cases})
    table = values.reshape(len(matrix_ids), len(seeds))
    rng = np.random.default_rng(20261107)
    rows = rng.integers(0, len(matrix_ids), size=(5000, len(matrix_ids)))
    columns = rng.integers(0, len(seeds), size=(5000, len(seeds)))
    bootstrap = table[rows[:, :, None], columns[:, None, :]].mean((1, 2))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return dict(
        n_pairs=len(values), mean_correlated_minus_independent=float(mean), paired_run_se=float(se),
        correlated_lower_fraction=float(np.mean(values < -1e-12)),
        tied_fraction=float(np.mean(np.abs(values) <= 1e-12)),
        independent_lower_fraction=float(np.mean(values > 1e-12)),
        ci95_two_way_bootstrap_lower=float(lower), ci95_two_way_bootstrap_upper=float(upper),
        bootstrap_repetitions=5000, bootstrap_seed=20261107,
        per_matrix=[dict(matrix_seed=m, mean_correlated_minus_independent=float(table[i].mean()))
                    for i, m in enumerate(matrix_ids)],
    )


def validate_comparison_manifests(manifest: dict, gittins_manifest: dict) -> None:
    """Require the same complete paper experiment before reusing old Gittins."""
    for key, expected in (("matrix_seeds", [1, 2, 3, 4, 5]),
                          ("run_seeds", list(range(20))), ("batch_size", 8),
                          ("budget_fraction", .1), ("per_cell_noise_variance", .25),
                          ("recommendation_std_penalty", 1.0),
                          ("priors", {key: list(value) for key, value in PRIORS.items()})):
        if manifest.get(key) != expected or gittins_manifest.get(key) != expected:
            raise ValueError(f"Three-way comparison requires matching paper setup: {key}")
    if manifest.get("max_steps") is not None:
        raise ValueError("Three-way comparison cannot include a capped smoke experiment")
    if manifest.get("n_arms") != gittins_manifest.get("n_models") or manifest.get("n_examples") != gittins_manifest.get("n_examples"):
        raise ValueError("Three-way comparison has different matrix dimensions")
    if manifest.get("matrix_sha256") != gittins_manifest.get("matrix_sha256"):
        raise ValueError("Three-way comparison has different response matrix hashes")
    if (not manifest.get("ordered_per_cell_cost_sha256") or
            manifest.get("ordered_per_cell_cost_sha256") != gittins_manifest.get("ordered_cost_vector_sha256")):
        raise ValueError("Three-way comparison has different ordered cost vector hashes")
    for current, previous in (("unit_exhaustive_cost", "full_evals"),
                              ("heterogeneous_exhaustive_cost", "full_original_cost")):
        if not np.isclose(float(manifest[current]), float(gittins_manifest[previous]), rtol=0, atol=1e-6):
            raise ValueError(f"Three-way comparison has different exhaustive cost: {current}")
    if (manifest.get("target") != "complete_fixed_benchmark_empirical_mean" or
            gittins_manifest.get("gittins_target") != "finite_population_mean" or
            gittins_manifest.get("ts_target") != "finite_population_mean"):
        raise ValueError("Three-way comparison requires empirical targets for all methods")
    if (gittins_manifest.get("gittins_model_version") != "batch_level_paper_v1" or
            gittins_manifest.get("gittins_dp_transitions") != "batch" or
            gittins_manifest.get("gittins_dp_horizon") != 125):
        raise ValueError("Three-way comparison requires the paper's batch-level Gittins")


def validate_shared_ts_trajectory(current, previous, *, case_name: str = "") -> list[str]:
    """Check raw actions and recommendations, rather than merely a final score."""
    required = ("x", "x_original_cost", "pulled_arm", "n_new_samples",
                "recommended_arm", "recommended_mean", "regret", "budget", "full_budget")
    checked = []
    for field in required:
        key = f"ts_{field}"
        if key not in current or key not in previous:
            raise ValueError(f"Missing shared TS trajectory field: {case_name}: {key}")
        if not np.array_equal(current[key], previous[key]):
            raise ValueError(f"Shared TS trajectory differs: {case_name}: {key}")
        checked.append(key)
    for field in ("sampled_columns", "counts", "observed_sums"):
        key = f"ts_{field}"
        if key in current and key in previous:
            if not np.array_equal(current[key], previous[key]):
                raise ValueError(f"Shared TS trajectory differs: {case_name}: {key}")
            checked.append(key)
    return checked


def load_three_way_cases(
    directory: Path, gittins_directory: Path, manifest: dict, cases: list[dict],
) -> tuple[dict, list[dict], dict]:
    """Validate both studies independently, then join Gittins by paired case ID."""
    gittins_manifest = json.loads((gittins_directory / "manifest.json").read_text())
    validate_comparison_manifests(manifest, gittins_manifest)
    # This loader also validates H+1 roots, terminal zeros, paper parameters,
    # all400 case identities, immutable settings hashes and recorded regrets.
    old_cases = load_gittins_cases(gittins_directory)
    identity = lambda c: (c["matrix_seed"], c["run_seed"], c["prior"], c["mode"])
    old_by_identity = {identity(c): c for c in old_cases}
    if set(old_by_identity) != {identity(c) for c in cases}:
        raise ValueError("Three-way comparison has different paired case identities")
    joined, checked_fields = [], set()
    for case in cases:
        matrix, seed, prior, mode = identity(case)
        name = f"matrix{matrix}_seed{seed}_{prior}_{mode}.npz"
        old = old_by_identity[identity(case)]
        merged, fields = join_three_way_case(
            case, old, directory / "runs" / name, gittins_directory / "runs" / name,
        )
        checked_fields.update(fields)
        joined.append(merged)
    audit = dict(
        reused_gittins_directory=str(gittins_directory.resolve()), reused_cases=len(joined),
        shared_independent_TS_trajectories_identical=True,
        required_raw_fields=list(sorted(checked_fields)),
        sampled_columns_checked_when_present_in_both_archives=True,
        gittins_recomputed=False,
    )
    return gittins_manifest, joined, audit


def join_three_way_case(
    current_case: dict, gittins_case: dict, current_path: Path, gittins_path: Path,
) -> tuple[dict, list[str]]:
    """Join one already validated pair only when its full TS histories agree."""
    for field in ("matrix_seed", "run_seed", "prior", "mode"):
        if current_case[field] != gittins_case[field]:
            raise ValueError(f"Three-way case identities differ: {field}")
    with np.load(current_path, allow_pickle=False) as current_raw, np.load(
        gittins_path, allow_pickle=False,
    ) as old_raw, ts_trajectory_archive(current_raw) as current_ts:
        checked = validate_shared_ts_trajectory(current_ts, old_raw, case_name=current_path.name)
        trace = gittins_case["methods"]["gittins"]
        trace = {**trace, "final_budget_fraction": trace["actual_final_budget_fraction"],
                 "elapsed_s": float(_scalar(old_raw, "gittins_elapsed_s")), "fit_count": 0,
                 "stop_reason": "historical_complete_budget_run"}
    return {**current_case, "methods": {**current_case["methods"], "gittins": trace}}, checked


def summarize(manifest: dict, cases: list[dict], directory: Path) -> dict:
    summary = dict(
        input_manifest=manifest, figure_aggregation="350-point linear interpolation, no extrapolation, mean +/-1 SE",
        smoke_test=manifest.get("max_steps") is not None,
        numerical_metrics="last completed observation at/before checkpoint; exact held regret integrated from prior tie at arm0",
        uncertainty="paired runs nested within response matrices; two-way matrix/shared-trial-seed bootstrap is descriptive",
        paired_difference="correlated_TS_minus_independent_TS; negative favors correlated_TS", panels={},
    )
    rows, difference_rows, per_run = [], [], []
    for mode in MODES:
        for prior in PRIORS:
            panel_cases = sorted((c for c in cases if c["mode"] == mode and c["prior"] == prior),
                                 key=lambda c: (c["matrix_seed"], c["run_seed"]))
            panel = dict(n_paired_trials=len(panel_cases), methods={}, paired_differences={})
            metrics = list(panel_cases[0]["methods"]["ts"]["metrics"])
            for method in METHODS:
                stats = {}
                for metric in metrics:
                    values = np.array([c["methods"][method]["metrics"][metric] for c in panel_cases])
                    mean, se = mean_and_se(values)
                    item = dict(mean=float(mean), se=float(se))
                    if metric.startswith("regret_at_"):
                        item["exact_best_recommendation_rate"] = float(np.mean(values <= 1e-12))
                    stats[metric] = item
                    rows.append(dict(cost_mode=mode, prior=prior, method=method, metric=metric, n=len(values), **item))
                panel["methods"][method] = stats
                for case in panel_cases:
                    raw = case["methods"][method]
                    per_run.append(dict(cost_mode=mode, prior=prior, method=method,
                        matrix_seed=case["matrix_seed"], run_seed=case["run_seed"], **raw["metrics"],
                        elapsed_s=raw["elapsed_s"], fit_count=raw["fit_count"],
                        observation_rounds=raw["observation_rounds"], final_budget_fraction=raw["final_budget_fraction"]))
            for metric in metrics:
                differences = np.array([c["methods"]["correlated_ts"]["metrics"][metric]
                                        - c["methods"]["ts"]["metrics"][metric] for c in panel_cases])
                item = paired_stats(differences, panel_cases)
                panel["paired_differences"][metric] = item
                difference_rows.append(dict(cost_mode=mode, prior=prior, metric=metric,
                                            **{k: v for k, v in item.items() if k != "per_matrix"}))
            summary["panels"][f"{mode}_{prior}"] = panel
    for filename, content in (("checkpoints.csv", rows), ("paired_differences.csv", difference_rows),
                              ("run_metrics.csv", per_run)):
        fields = list(dict.fromkeys(k for row in content for k in row))
        with (directory / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(content)
    (directory / "summary.json").write_text(json.dumps(_finite(summary), indent=2, allow_nan=False) + "\n")
    return summary


def summarize_three_way(
    manifest: dict, gittins_manifest: dict, cases: list[dict], audit: dict, directory: Path,
) -> dict:
    """Report all methods and paired correlated-minus-comparator differences."""
    summary = dict(
        input_manifest=manifest, gittins_input_manifest=gittins_manifest,
        reuse_validation=audit, methods=list(THREE_WAY_METHODS),
        display_names={"gittins": "Gittins", "ts": "Correlation-unaware TS", "correlated_ts": "Correlated TS"},
        gittins_protocol={"model_version": gittins_manifest["gittins_model_version"],
                          "batch_size": gittins_manifest["batch_size"],
                          "cost_scaling_factor": gittins_manifest.get("cost_scaling_factor"),
                          "grid_points": gittins_manifest.get("gittins_grid_points"),
                          "dp_horizon": gittins_manifest["gittins_dp_horizon"],
                          "settings_hash": gittins_manifest.get("settings_hash")},
        numerical_metrics="last completed observation at/before checkpoint; exact step-hold integral from the prior recommendation",
        uncertainty="100 paired algorithm trials nested within five response matrices; descriptive two-way matrix/shared-seed bootstrap",
        difference_convention="correlated_TS_minus_comparator; negative favors correlated_TS",
        panels={},
    )
    checkpoints, differences, runs = [], [], []
    for mode in MODES:
        for prior in PRIORS:
            panel_cases = sorted((c for c in cases if c["mode"] == mode and c["prior"] == prior),
                                 key=lambda c: (c["matrix_seed"], c["run_seed"]))
            metrics = list(panel_cases[0]["methods"]["ts"]["metrics"])
            panel = dict(n_paired_trials=len(panel_cases), methods={}, paired_differences={})
            for method in THREE_WAY_METHODS:
                method_stats = {}
                for metric in metrics:
                    values = np.array([c["methods"][method]["metrics"][metric] for c in panel_cases])
                    mean, se = mean_and_se(values)
                    item = dict(mean=float(mean), se=float(se))
                    if metric.startswith("regret_at_"):
                        item["exact_best_recommendation_rate"] = float(np.mean(values <= 1e-12))
                    method_stats[metric] = item
                    checkpoints.append(dict(cost_mode=mode, prior=prior, method=method, metric=metric,
                                            n=len(values), **item))
                panel["methods"][method] = method_stats
                for case in panel_cases:
                    trace = case["methods"][method]
                    runs.append(dict(cost_mode=mode, prior=prior, method=method,
                                     matrix_seed=case["matrix_seed"], run_seed=case["run_seed"],
                                     **trace["metrics"], final_budget_fraction=trace["final_budget_fraction"],
                                     observation_rounds=trace["observation_rounds"]))
            for comparator in ("gittins", "ts"):
                pair = f"correlated_ts_minus_{comparator}"
                pair_stats = {}
                for metric in metrics:
                    delta = np.array([c["methods"]["correlated_ts"]["metrics"][metric]
                                      - c["methods"][comparator]["metrics"][metric] for c in panel_cases])
                    item = paired_stats(delta, panel_cases)
                    item["mean_difference"] = item.pop("mean_correlated_minus_independent")
                    item["comparator_lower_fraction"] = item.pop("independent_lower_fraction")
                    item["comparator"] = comparator
                    for entry in item["per_matrix"]:
                        entry["mean_difference"] = entry.pop("mean_correlated_minus_independent")
                    pair_stats[metric] = item
                    differences.append(dict(cost_mode=mode, prior=prior, comparison=pair, metric=metric,
                                            **{k: v for k, v in item.items() if k != "per_matrix"}))
                panel["paired_differences"][pair] = pair_stats
            summary["panels"][f"{mode}_{prior}"] = panel
    for name, rows in (("three_way_checkpoints.csv", checkpoints),
                       ("three_way_pairwise_differences.csv", differences),
                       ("three_way_run_metrics.csv", runs)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (directory / "three_way_summary.json").write_text(json.dumps(_finite(summary), indent=2, allow_nan=False) + "\n")
    return summary


def render(
    manifest: dict, cases: list[dict], directory: Path, xmax: float, stem: str,
    *, methods: tuple[str, ...] = METHODS,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"gittins": "#2363a5", "ts": "#16876b", "correlated_ts": "#7c4bb0"}
    labels = {"gittins": "Batch-level Gittins", "ts": "Independent Gaussian TS", "correlated_ts": "Correlated Gaussian TS"}
    if "gittins" in methods:
        labels = {"gittins": "Gittins", "ts": "Correlation-unaware TS", "correlated_ts": "Correlated TS"}
    grid = np.linspace(0, xmax, 350)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), sharex=True, sharey=True)
    curves, upper = {"grid_fraction": grid}, 0.0
    for row, mode in enumerate(MODES):
        for column, prior in enumerate(PRIORS):
            ax = axes[row, column]
            panel = [c for c in cases if c["mode"] == mode and c["prior"] == prior]
            for method in methods:
                values = np.stack([np.interp(grid, c["methods"][method]["x"],
                                             c["methods"][method]["y"], left=np.nan, right=np.nan) for c in panel])
                mean, se = mean_and_se(values)
                ax.plot(grid * 100, mean, color=colors[method], label=labels[method], linewidth=1.8)
                ax.fill_between(grid * 100, np.maximum(0, mean - se), mean + se, color=colors[method], alpha=.17, linewidth=0)
                for name, value in (("mean", mean), ("se", se), ("n", np.isfinite(values).sum(0))):
                    curves[f"{mode}_{prior}_{method}_{name}"] = value
                if np.isfinite(mean).any():
                    upper = max(upper, float(np.nanmax(mean + np.nan_to_num(se))))
            mu, variance = PRIORS[prior]
            if row == 0:
                ax.set_title(f"{'General' if prior == 'G' else 'GSM8K'} prior: N({mu:g}, {variance:g})", fontsize=12, fontweight="semibold", pad=12)
            if column == 0:
                ax.set_ylabel(f"{'Unit cost' if mode == 'unit' else 'Heterogeneous cost'}\nSimple regret")
            ax.grid(axis="y", color="#dce4ed", linewidth=.7)
            ax.set_xlim(0, xmax * 100)
            ax.set_axisbelow(True)
    for ax in axes.flat:
        ax.set_ylim(-.003, max(.02, 1.06 * upper))
    n_matrices, n_seeds = len(manifest["matrix_seeds"]), len(manifest["run_seeds"])
    title = "GSM8K: independent vs correlated Gaussian TS"
    if "gittins" in methods:
        title = "GSM8K: Gittins, independent TS and correlated TS"
    if manifest.get("max_steps") is not None:
        title = f"SMOKE ({manifest['max_steps']} batches): " + title
    fig.suptitle(title, x=.085, y=.975, ha="left", fontsize=17, fontweight="semibold")
    kind = "online-fitted" if manifest["kernel_mode"] == "fit" else "fixed-default"
    fig.text(.085, .934, f"Categorical kernel ({kind}); empirical target, LCB=1, B={manifest['batch_size']}; {n_matrices} matrices x {n_seeds} trials", fontsize=10.5, color="#64748b")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.54, .905), ncol=len(methods), frameon=False)
    fig.supxlabel("Exhaustive-evaluation budget used (%)", y=.065, fontsize=11)
    fig.text(.085, .018, "Bands: +/-1 SE. Linear interpolation on 350 points; no extension outside each observed range.", color="#64748b", fontsize=8.8)
    fig.subplots_adjust(left=.085, right=.985, top=.82, bottom=.13, hspace=.20, wspace=.15)
    fig.savefig(directory / f"{stem}.png", dpi=230, facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    np.savez_compressed(directory / f"{stem}_curves.npz", **curves)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/gsm8k_correlated_ts_reuse_lcb100")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--gittins-results-dir", type=Path,
                        help="Also produce three-way figures using a validated paper batch-level Gittins experiment.")
    args = parser.parse_args()
    directory = args.results_dir.resolve()
    manifest, cases = load_cases(directory, allow_smoke=args.allow_smoke)
    summary = summarize(manifest, cases, directory)
    render(manifest, cases, directory, manifest["budget_fraction"], "comparison")
    render(manifest, cases, directory, min(.02, manifest["budget_fraction"]), "comparison_early")
    if args.gittins_results_dir is not None:
        gittins_manifest, joined, audit = load_three_way_cases(
            directory, args.gittins_results_dir.resolve(), manifest, cases,
        )
        summarize_three_way(manifest, gittins_manifest, joined, audit, directory)
        render(manifest, joined, directory, .1, "comparison_three_way", methods=THREE_WAY_METHODS)
        render(manifest, joined, directory, .02, "comparison_three_way_early", methods=THREE_WAY_METHODS)
    for name, panel in summary["panels"].items():
        metric = "auc_0_10%" if "auc_0_10%" in panel["paired_differences"] else next(iter(panel["paired_differences"]))
        delta = panel["paired_differences"][metric]["mean_correlated_minus_independent"]
        print(f"{name}: {metric}, correlated - independent = {delta:.8g}")
    print(f"Wrote plots and paired metrics under {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
