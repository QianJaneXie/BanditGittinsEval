#!/usr/bin/env python3
"""Plot paired recommendation experiments on a fixed Gittins allocation.

Abstentions stay missing in regret curves. Intervals quantify Monte Carlo
uncertainty over allocation seeds conditional on the fixed input matrices;
they do not quantify uncertainty over a population of evaluation datasets.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


BASELINE = "posterior_mean"
PRIOR_ORDER = ("default", "dataset")
PRIOR_LABELS = {"default": "Default prior", "dataset": "Dataset prior"}
GRID = np.linspace(0.0, 0.10, 201)
COLORS = ["#2166ac", "#b35806", "#1b7837", "#762a83", "#c51b7d", "#008b8b", "#8c6d31", "#5e4fa2"]
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260911
_BOOTSTRAP_WEIGHTS: dict[int, np.ndarray] = {}


@dataclass
class Trace:
    prior: str
    matrix_seed: int
    run_seed: int
    method_keys: list[str]
    x: np.ndarray
    recommended_arm: np.ndarray
    regret: np.ndarray
    selected_n: np.ndarray
    selected_variance: np.ndarray
    scores: np.ndarray
    n_cells: int


def scalar(z, key):
    return np.asarray(z[key]).item()


def load_results(results_dir: Path) -> tuple[dict, list[dict], list[Trace]]:
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Refusing to summarize an incomplete experiment.")
    methods = manifest["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("manifest.json must contain a nonempty methods list.")
    keys = [m["key"] for m in methods]
    if len(set(keys)) != len(keys) or BASELINE not in keys:
        raise ValueError("Method keys must be unique and include posterior_mean.")
    traces = []
    seen = set()
    for path in sorted((results_dir / "traces").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            prior = str(scalar(z, "prior_tag"))
            if prior not in PRIOR_ORDER:
                continue
            matrix_seed = int(scalar(z, "matrix_seed"))
            run_seed = int(scalar(z, "run_seed"))
            identity = (prior, matrix_seed, run_seed)
            if identity in seen:
                raise ValueError(f"Duplicate trace identity: {identity}")
            seen.add(identity)
            trace_keys = np.asarray(z["method_keys"]).astype(str).tolist()
            if set(trace_keys) != set(keys):
                raise ValueError(f"{path}: trace and manifest method keys differ.")
            x = np.asarray(z["x"], dtype=float)
            regret = np.asarray(z["regret"], dtype=float)
            arms = np.asarray(z["recommended_arm"], dtype=int)
            if x.ndim != 1 or x.size == 0 or np.any(np.diff(x) <= 0):
                raise ValueError(f"{path}: x must be nonempty and strictly increasing.")
            if regret.shape != (x.size, len(keys)) or arms.shape != regret.shape:
                raise ValueError(f"{path}: invalid recommendation array shapes.")
            regret = np.where(arms >= 0, regret, np.nan)
            traces.append(Trace(
                prior, matrix_seed, run_seed, trace_keys, x, arms, regret,
                np.asarray(z["selected_n"], dtype=float),
                np.asarray(z["selected_variance"], dtype=float),
                np.asarray(z["scores"], dtype=float),
                int(scalar(z, "n_arms")) * int(scalar(z, "n_examples")),
            ))
    if not traces:
        raise ValueError(f"No supported traces in {results_dir / 'traces'}")
    config = manifest["config"]
    expected = {(p, m, s) for p in config["priors"] for m in config["matrix_seeds"] for s in config["run_seeds"]}
    if seen != expected:
        raise ValueError(f"Incomplete or unexpected seed grid: missing={expected-seen}, extra={seen-expected}")
    return manifest, methods, traces


def hold_last(trace: Trace, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Carry the most recent *state*, including abstention, without lookahead."""
    col = trace.method_keys.index(key)
    indices = np.searchsorted(trace.x / trace.n_cells, GRID, side="right") - 1
    valid = indices >= 0
    regret = np.full(GRID.shape, np.nan)
    coverage = np.full(GRID.shape, np.nan)
    regret[valid] = trace.regret[indices[valid], col]
    coverage[valid] = (trace.recommended_arm[indices[valid], col] >= 0).astype(float)
    return regret, coverage


def finite_mean(values: np.ndarray, axis=0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.where(np.isfinite(values), values, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def stratified_mean_ci(values: np.ndarray, matrix_seeds: np.ndarray, run_seeds: np.ndarray | None = None):
    """Conditional mean and seed-block bootstrap uncertainty on fixed matrices.

    Each random-seed block contains all five matrix runs. Bootstrap draws
    resample blocks, then divide eligible sums by eligible counts, preserving
    the changing denominator for abstaining rules. Shared draws are reused
    across methods and budgets. No matrix is resampled.
    """
    del matrix_seeds  # matrices are fixed within each resampled run-seed block
    values = np.asarray(values, dtype=float)
    if run_seeds is None:
        run_seeds = np.arange(len(values))
    clusters = np.unique(run_seeds)
    n_clusters = len(clusters)
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    sums = np.zeros((n_clusters, flat.shape[1]))
    counts = np.zeros_like(sums)
    for i, cluster in enumerate(clusters):
        v = flat[run_seeds == cluster]
        sums[i] = np.where(np.isfinite(v), v, 0.0).sum(axis=0)
        counts[i] = np.isfinite(v).sum(axis=0)
    average = finite_mean(values)
    if n_clusters < 2:
        missing = np.full(shape, np.nan)
        return average, missing.copy(), missing.copy(), missing.copy()
    if n_clusters not in _BOOTSTRAP_WEIGHTS:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        _BOOTSTRAP_WEIGHTS[n_clusters] = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=BOOTSTRAP_REPEATS).astype(float)
    weights = _BOOTSTRAP_WEIGHTS[n_clusters]
    numerator, denominator = weights @ sums, weights @ counts
    boot = np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)
    low, high, se = (np.full(flat.shape[1], np.nan) for _ in range(3))
    has_data = np.isfinite(boot).any(axis=0)
    if has_data.any():
        b = boot[:, has_data]
        low[has_data], high[has_data] = np.nanquantile(b, [0.025, 0.975], axis=0)
        bmean = finite_mean(b)
        residual = np.where(np.isfinite(b), b - bmean, 0.0)
        n = np.isfinite(b).sum(axis=0)
        se[has_data] = np.sqrt(np.divide(np.square(residual).sum(axis=0), n - 1, out=np.full(n.shape, np.nan), where=n > 1))
    return average, se.reshape(shape), low.reshape(shape), high.reshape(shape)


def curve_statistics(traces: list[Trace], methods: list[dict]) -> dict:
    out = {}
    for prior in PRIOR_ORDER:
        runs = [t for t in traces if t.prior == prior]
        if not runs:
            continue
        matrix_seeds = np.asarray([t.matrix_seed for t in runs])
        run_seeds = np.asarray([t.run_seed for t in runs])
        out[prior] = {}
        for method in methods:
            curves = [hold_last(t, method["key"]) for t in runs]
            regret = np.asarray([c[0] for c in curves])
            coverage = np.asarray([c[1] for c in curves])
            mean, se, low, high = stratified_mean_ci(regret, matrix_seeds, run_seeds)
            cover, _, _, _ = stratified_mean_ci(coverage, matrix_seeds, run_seeds)
            out[prior][method["key"]] = dict(mean=mean, se=se, low=low, high=high, coverage=cover)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_metrics(traces: list[Trace], methods: list[dict], out_dir: Path) -> list[dict]:
    per_run = []
    for trace in traces:
        baseline_col = trace.method_keys.index(BASELINE)
        baseline_regret = trace.regret[-1, baseline_col]
        baseline_eligible = trace.recommended_arm[-1, baseline_col] >= 0 and np.isfinite(baseline_regret)
        for method in methods:
            key = method["key"]
            col = trace.method_keys.index(key)
            eligible = trace.recommended_arm[-1, col] >= 0 and np.isfinite(trace.regret[-1, col])
            value = float(trace.regret[-1, col]) if eligible else np.nan
            per_run.append(dict(
                prior_tag=trace.prior, matrix_seed=trace.matrix_seed, run_seed=trace.run_seed,
                method_key=key, method_label=method.get("label", key),
                family=method.get("family", ""), parameter=method.get("parameter", ""),
                final_evaluations=int(trace.x[-1]), final_budget_fraction=float(trace.x[-1] / trace.n_cells),
                final_recommended_arm=int(trace.recommended_arm[-1, col]),
                final_regret=value, final_coverage=int(eligible),
                top1_correct=float(value <= 1e-7) if eligible else np.nan,
                final_selected_n=float(trace.selected_n[-1, col]) if eligible else np.nan,
                final_selected_variance=float(trace.selected_variance[-1, col]) if eligible else np.nan,
                final_score=float(trace.scores[-1, col]) if eligible else np.nan,
                paired_final_delta=value - baseline_regret if eligible and baseline_eligible else np.nan,
            ))
    write_csv(out_dir / "per_run_metrics.csv", per_run)
    summary = []
    for prior in PRIOR_ORDER:
        for method in methods:
            rows = [r for r in per_run if r["prior_tag"] == prior and r["method_key"] == method["key"]]
            if not rows:
                continue
            seeds = np.asarray([r["matrix_seed"] for r in rows])
            run_seeds = np.asarray([r["run_seed"] for r in rows])
            def stats(field):
                return stratified_mean_ci(np.asarray([r[field] for r in rows]), seeds, run_seeds)
            mean, se, lo, hi = stats("final_regret")
            delta, delta_se, delta_lo, delta_hi = stats("paired_final_delta")
            eligible_values = np.asarray([r["final_regret"] for r in rows])
            eligible_values = eligible_values[np.isfinite(eligible_values)]
            summary.append(dict(
                prior_tag=prior, method_key=method["key"], method_label=method.get("label", method["key"]),
                family=method.get("family", ""), parameter=method.get("parameter", ""),
                n_runs=len(rows), n_matrices=len(np.unique(seeds)), n_eligible=int(eligible_values.size),
                mean_final_regret=float(mean), final_regret_mc_se=float(se),
                final_regret_ci95_low=float(lo), final_regret_ci95_high=float(hi),
                p90_final_regret=float(np.quantile(eligible_values, 0.9)) if eligible_values.size else np.nan,
                top1_accuracy=float(stats("top1_correct")[0]),
                final_coverage=float(stats("final_coverage")[0]),
                mean_selected_n=float(stats("final_selected_n")[0]),
                paired_n=sum(np.isfinite(r["paired_final_delta"]) for r in rows),
                paired_mean_final_delta=float(delta), paired_mc_se=float(delta_se),
                paired_ci95_low=float(delta_lo), paired_ci95_high=float(delta_hi),
            ))
    write_csv(out_dir / "summary.csv", summary)
    return summary


def export_checkpoints(traces: list[Trace], methods: list[dict], out_dir: Path) -> None:
    rows = []
    for fraction in (0.01, 0.02, 0.05, 0.10):
        if fraction > GRID[-1] + 1e-12:
            continue
        for prior in PRIOR_ORDER:
            runs = [t for t in traces if t.prior == prior]
            if not runs:
                continue
            matrix_seeds = np.asarray([t.matrix_seed for t in runs])
            run_seeds = np.asarray([t.run_seed for t in runs])
            for method in methods:
                regrets, coverage, counts, deltas = [], [], [], []
                for trace in runs:
                    idx = np.searchsorted(trace.x / trace.n_cells, fraction, side="right") - 1
                    col = trace.method_keys.index(method["key"])
                    baseline_col = trace.method_keys.index(BASELINE)
                    if idx < 0:
                        regrets.append(np.nan)
                        coverage.append(np.nan)
                        counts.append(np.nan)
                        deltas.append(np.nan)
                        continue
                    eligible = trace.recommended_arm[idx, col] >= 0 and np.isfinite(trace.regret[idx, col])
                    value = float(trace.regret[idx, col]) if eligible else np.nan
                    baseline_value = trace.regret[idx, baseline_col]
                    baseline_eligible = trace.recommended_arm[idx, baseline_col] >= 0 and np.isfinite(baseline_value)
                    regrets.append(value)
                    coverage.append(float(eligible))
                    counts.append(float(trace.selected_n[idx, col]) if eligible else np.nan)
                    deltas.append(value - baseline_value if eligible and baseline_eligible else np.nan)
                def stats(values):
                    return stratified_mean_ci(np.asarray(values), matrix_seeds, run_seeds)
                mean, se, lo, hi = stats(regrets)
                delta, delta_se, delta_lo, delta_hi = stats(deltas)
                eligible_values = np.asarray(regrets)[np.isfinite(regrets)]
                rows.append(dict(
                    prior_tag=prior, budget_fraction=fraction, method_key=method["key"],
                    method_label=method.get("label", method["key"]), family=method.get("family", ""),
                    parameter=method.get("parameter", ""), n_runs=len(runs), n_eligible=int(eligible_values.size),
                    mean_regret=float(mean), p90_regret=float(np.quantile(eligible_values, 0.9)) if eligible_values.size else np.nan,
                    regret_mc_se=float(se), regret_ci95_low=float(lo), regret_ci95_high=float(hi),
                    coverage=float(stats(coverage)[0]), mean_selected_n=float(stats(counts)[0]),
                    paired_n=int(np.isfinite(deltas).sum()), paired_mean_delta=float(delta),
                    paired_mc_se=float(delta_se), paired_ci95_low=float(delta_lo), paired_ci95_high=float(delta_hi),
                ))
    write_csv(out_dir / "budget_checkpoints.csv", rows)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8d959e", "axes.labelcolor": "#243447", "text.color": "#243447",
        "xtick.color": "#465769", "ytick.color": "#465769", "grid.color": "#dce2e8",
        "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def plot_label(method: dict) -> str:
    key, family, parameter = method["key"], method.get("family"), method.get("parameter")
    if key == BASELINE:
        return "Posterior mean"
    if key == "empirical_mean":
        return "Sample mean"
    if family == "variance_gate":
        return f"Variance gate ρ = {parameter:g}"
    if family == "mean_variance":
        return f"Mean − {parameter:g} × variance"
    if family == "gaussian_lcb":
        return f"Gaussian LCB z = {parameter:g}"
    if family == "finite_gaussian_lcb":
        return f"Finite-row Gaussian z = {parameter:g}"
    if key == "finite_population_lcb":
        return "Anytime finite-population LCB"
    return method.get("label", key)


def make_comparison(keys: list[str], title: str, methods: list[dict], curves: dict, target: Path, *, context: dict, pdf=False) -> None:
    labels = {m["key"]: plot_label(m) for m in methods}
    keys = list(dict.fromkeys([BASELINE] + [k for k in keys if k in labels]))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.0), sharex=True, sharey="row", gridspec_kw={"height_ratios": [3.2, 1.05]})
    color_map = {BASELINE: "#626b75"}
    for i, key in enumerate(k for k in keys if k != BASELINE):
        color_map[key] = COLORS[i % len(COLORS)]
    for col, prior in enumerate(PRIOR_ORDER):
        axes[0, col].set_title(PRIOR_LABELS[prior], loc="left", fontweight="bold", pad=11)
        if prior not in curves:
            axes[0, col].text(0.5, 0.5, "No results available", ha="center", va="center", transform=axes[0, col].transAxes)
        for key in keys:
            if prior not in curves or key not in curves[prior]:
                continue
            c = curves[prior][key]
            color = color_map[key]
            linestyle = "--" if key == BASELINE else "-"
            axes[0, col].plot(GRID, 100 * c["mean"], color=color, lw=1.9, ls=linestyle, drawstyle="steps-post", label=labels[key])
            axes[0, col].fill_between(GRID, 100 * np.maximum(0, c["low"]), 100 * c["high"], color=color, alpha=0.12, linewidth=0, step="post")
            axes[1, col].plot(GRID, c["coverage"], color=color, lw=1.65, ls=linestyle, drawstyle="steps-post")
        for row in range(2):
            axes[row, col].grid(True, axis="y")
            axes[row, col].set_xlim(0, GRID[-1])
            axes[row, col].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axes[row, col].set_xticks(np.linspace(0, GRID[-1], 6))
        axes[0, col].set_ylim(bottom=0)
        axes[0, col].yaxis.set_major_locator(MaxNLocator(6, min_n_ticks=3))
        axes[1, col].set_ylim(-0.03, 1.05)
        axes[1, col].set_yticks([0, 0.5, 1])
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[1, col].set_xlabel("Evaluations / full matrix")
    axes[0, 0].set_ylabel("Mean simple regret (accuracy pp)")
    axes[1, 0].set_ylabel("Recommendation\ncoverage")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    ncol = min(3, max(1, len(keys)))
    legend_rows = int(np.ceil(len(keys) / ncol))
    fig.suptitle(title, x=0.065, ha="left", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.065, 0.937, context["subtitle"], fontsize=10.5, color="#556778")
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.061), ncol=ncol, frameon=False, fontsize=9, columnspacing=1.8)
    fig.text(0.065, 0.025, "Regret averages eligible recommendations; abstentions stay missing. Bands: 95% seed-block bootstrap intervals\n" + context["uncertainty_note"] + " Coverage is the fraction of runs recommending. No dataset resampling.", fontsize=8.2, color="#556778", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.98, top=0.865, bottom=0.15 + 0.031 * legend_rows, wspace=0.12, hspace=0.12)
    fig.savefig(target, dpi=190)
    if pdf:
        fig.savefig(target.with_suffix(".pdf"))
    plt.close(fig)


def paired_delta_plot(summary: list[dict], methods: list[dict], out_dir: Path, *, context: dict) -> None:
    keys = [m["key"] for m in methods if m["key"] != BASELINE]
    if not keys:
        return
    labels = {m["key"]: plot_label(m) for m in methods}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(5.4, 0.31 * len(keys) + 2.2)), sharey=True)
    for col, prior in enumerate(PRIOR_ORDER):
        by_key = {r["method_key"]: r for r in summary if r["prior_tag"] == prior}
        for pos, key in enumerate(keys):
            row = by_key.get(key)
            if not row or not np.isfinite(row["paired_mean_final_delta"]):
                continue
            value = 100 * row["paired_mean_final_delta"]
            lo, hi = 100 * row["paired_ci95_low"], 100 * row["paired_ci95_high"]
            axes[col].plot(value, pos, "o", ms=5, color=COLORS[col])
            if np.isfinite(lo) and np.isfinite(hi):
                axes[col].hlines(pos, lo, hi, color=COLORS[col], lw=1.2)
                axes[col].vlines([lo, hi], pos - 0.10, pos + 0.10, color=COLORS[col], lw=1.0)
        axes[col].axvline(0, color="#8d959e", lw=1, ls="--")
        axes[col].grid(True, axis="x")
        axes[col].set_title(PRIOR_LABELS[prior], loc="left", fontweight="bold")
        axes[col].set_xlabel("Final regret difference from posterior mean (pp)\n← lower regret")
    axes[0].set_yticks(np.arange(len(keys)), [labels[k] for k in keys])
    axes[0].invert_yaxis()
    fig.suptitle("Paired recommendation effects at the final budget", x=0.04, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.04, 0.04, "Matched eligible runs only; 95% seed-block bootstrap intervals " + context["uncertainty_note"] + "\n" + context["subtitle"], fontsize=9, color="#556778")
    fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.20, wspace=0.13)
    fig.savefig(out_dir / "paired_final_delta.png", dpi=190)
    plt.close(fig)


def main() -> int:
    global GRID, PRIOR_LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("recommend/results/gsm8k"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, methods, traces = load_results(args.results_dir)
    config = manifest["config"]
    GRID = np.linspace(0.0, config["budget_fraction"], 201)
    n_matrices = len(manifest["matrices"])
    n_seeds = len(config["run_seeds"])
    label = config.get("benchmark_label", "GSM8K")
    context = {
        "subtitle": f"{label}  ·  Unit-cost {100*config['budget_fraction']:g}% budget  ·  Batch size {config['batch_size']}  ·  Fixed Gittins allocation",
        "uncertainty_note": f"(1,000 draws; {n_matrices} fixed {'matrix' if n_matrices == 1 else 'matrices'}; {n_seeds} sampling seeds).",
    }
    PRIOR_LABELS = {key: f"{'Default' if key == 'default' else 'Dataset'} prior: N({values[0]:g}, {values[1]:g})" for key, values in manifest["priors"].items()}
    setup_style()
    curves = curve_statistics(traces, methods)
    summary = export_metrics(traces, methods, out_dir)
    export_checkpoints(traces, methods, out_dir)
    for method in methods:
        make_comparison([method["key"]], method.get("label", method["key"]), methods, curves, out_dir / f"method_{safe_name(method['key'])}.png", context=context)
    families = list(dict.fromkeys(m.get("family", "other") for m in methods if m["key"] != BASELINE))
    for family in families:
        keys = [m["key"] for m in methods if m.get("family", "other") == family]
        title = str(family).replace("_", " ").capitalize() + " sensitivity"
        make_comparison(keys, title, methods, curves, out_dir / f"family_{safe_name(str(family))}.png", context=context)
    representatives = [BASELINE, "variance_gate_0.1", "mean_variance_5", "gaussian_lcb_1.645", "finite_gaussian_lcb_1.645", "finite_population_lcb"]
    make_comparison(representatives, "Recommendation rules on the same observations", methods, curves, out_dir / "overview.png", context=context, pdf=True)
    paired_delta_plot(summary, methods, out_dir, context=context)
    print(f"Wrote {len(methods)} individual comparisons, {len(families)} family plots, overview PNG/PDF, paired effects, and CSV metrics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
