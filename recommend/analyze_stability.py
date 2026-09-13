"""Temporal recommendation stability on existing paired Gittins traces.

This is an exploratory secondary analysis. No allocation or recommendation is
changed, and no new model evaluations are performed. Both compared rules must
recommend throughout the window; missing recommendations are never treated as
zero deterioration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "recommend/figures/stability"
BENCHMARKS = {
    "gsm8k": "GSM8K", "piqa": "PIQA",
    "mmlu_computer_security": "Computer Security",
    "mmlu_anatomy": "Anatomy", "mmlu_business_ethics": "Business Ethics",
}
METHODS = ["posterior_mean", "gaussian_lcb_1.645"]
PRIORS = ["default", "dataset"]
BEGIN, END = .01, .10
BOOTSTRAP_SEED, BOOTSTRAP_REPEATS = 20260913, 10000
METRICS = {
    "total_downward_pp": "Total downward movement (pp)",
    "maximum_drawdown_pp": "Maximum drawdown (pp)",
    "switches": "Recommendation switches",
    "largest_drop_pp": "Largest downward jump (pp)",
    "mean_drawdown_pp": "Budget-averaged drawdown (pp)",
    "mean_regret_pp": "Budget-averaged regret (pp)",
    "final_regret_pp": "Final regret (pp)",
    "downward_jumps": "Number of harmful jumps",
    "jumps_ge_2pp": "Number of jumps of at least 2 pp",
    "any_jump_ge_2pp": "Probability of any jump of at least 2 pp",
}


def path_metrics(x, regret, arms, begin=BEGIN, end=END):
    """Use the state active at begin, then every real update through end.

The recommendation is a right-continuous step function of evaluation budget.
A change exactly at end counts as a jump but contributes zero exposure time.
"""
    x, regret, arms = np.asarray(x), np.asarray(regret), np.asarray(arms)
    if x.ndim != 1 or len(x) == 0 or regret.shape != x.shape or arms.shape != x.shape:
        raise ValueError("Expected nonempty matching one-dimensional arrays")
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0) or not x[0] <= begin < end <= x[-1]:
        raise ValueError("Need increasing budgets covering the complete analysis window")
    start = int(np.searchsorted(x, begin, side="right") - 1)
    stop = int(np.searchsorted(x, end, side="right"))
    r, a = regret[start:stop], arms[start:stop]
    if not np.isfinite(r).all() or np.any(a < 0):
        raise ValueError("Abstaining rules require a separate coverage-aware analysis")
    deterioration = np.maximum(np.diff(r), 0)
    drawdown = r - np.minimum.accumulate(r)
    durations = np.diff(np.r_[begin, x[start + 1:stop], end])
    assert np.all(durations >= 0)
    np.testing.assert_allclose(durations.sum(), end - begin, atol=1e-12)
    return {
        "total_downward_pp": float(100 * deterioration.sum()),
        "maximum_drawdown_pp": float(100 * drawdown.max()),
        "switches": int(np.count_nonzero(np.diff(a))),
        "largest_drop_pp": float(100 * deterioration.max(initial=0)),
        "mean_drawdown_pp": float(100 * np.dot(durations, drawdown) / (end - begin)),
        "mean_regret_pp": float(100 * np.dot(durations, r) / (end - begin)),
        "final_regret_pp": float(100 * r[-1]),
        "downward_jumps": int(np.sum(deterioration > 1e-12)),
        "jumps_ge_2pp": int(np.sum(deterioration >= .02 - 1e-12)),
        "any_jump_ge_2pp": int(np.any(deterioration >= .02 - 1e-12)),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    rows, sources = [], {}
    for benchmark in BENCHMARKS:
        directory = ROOT / "recommend/results" / benchmark
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "complete"
        config = manifest["config"]
        expected = {(p, m, s) for p in config["priors"] for m in config["matrix_seeds"] for s in config["run_seeds"]}
        seen = set()
        sources[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
        for path in sorted((directory / "traces").glob("*.npz")):
            sources[str(path.relative_to(ROOT))] = digest(path)
            with np.load(path, allow_pickle=False) as z:
                identity = (str(z["prior_tag"]), int(z["matrix_seed"]), int(z["run_seed"]))
                assert identity not in seen
                seen.add(identity)
                x = z["x"] / (int(z["n_arms"]) * int(z["n_examples"]))
                keys, regrets, rec = list(z["method_keys"]), z["regret"], z["recommended_arm"]
                for method in METHODS:
                    col = keys.index(method)
                    rows.append(dict(benchmark=benchmark, prior=identity[0], matrix_seed=identity[1],
                        run_seed=identity[2], method=method,
                        **path_metrics(x, regrets[:, col], rec[:, col])))
        assert seen == expected
    return rows, sources


def summarize(rows):
    result = []
    weights_by_size = {}
    for benchmark in BENCHMARKS:
        for prior in PRIORS:
            group = [r for r in rows if r["benchmark"] == benchmark and r["prior"] == prior]
            seeds = sorted({r["run_seed"] for r in group})
            if len(seeds) not in weights_by_size:
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                weights_by_size[len(seeds)] = rng.multinomial(len(seeds), np.full(len(seeds), 1/len(seeds)), size=BOOTSTRAP_REPEATS) / len(seeds)
            weights = weights_by_size[len(seeds)]
            for metric in METRICS:
                # Each seed block retains all fixed matrices and both paired rules.
                blocks = np.array([[np.mean([r[metric] for r in group if r["run_seed"] == seed and r["method"] == method]) for method in METHODS] for seed in seeds])
                samples = weights @ blocks
                delta = blocks[:, 1] - blocks[:, 0]
                ci = np.quantile(weights @ delta, [.025, .975])
                means = blocks.mean(axis=0)
                cis = np.quantile(samples, [.025, .975], axis=0)
                p90 = [float(np.quantile([r[metric] for r in group if r["method"] == method], .9)) for method in METHODS]
                result.append(dict(benchmark=benchmark, prior=prior, metric=metric,
                    n_runs=len(group)//2, n_seed_blocks=len(seeds),
                    posterior_mean=float(means[0]), lcb_mean=float(means[1]),
                    posterior_p90=p90[0], lcb_p90=p90[1],
                    posterior_ci95_low=float(cis[0, 0]), posterior_ci95_high=float(cis[1, 0]),
                    lcb_ci95_low=float(cis[0, 1]), lcb_ci95_high=float(cis[1, 1]),
                    paired_mean_delta=float(delta.mean()), paired_ci95_low=float(ci[0]), paired_ci95_high=float(ci[1]),
                    reduction_percent=float(100*(means[0]-means[1])/means[0]) if means[0] else None))
    return result


def plot(summary):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "recommend/.mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#a1aab5", "text.color": "#243447", "axes.labelcolor": "#243447",
        "xtick.color": "#465769", "ytick.color": "#465769", "pdf.fonttype": 42})

    def panels(metrics, target, size):
        fig, axes = plt.subplots(2, len(metrics), figsize=size, squeeze=False, sharey=True)
        for row, prior in enumerate(PRIORS):
            for col, metric in enumerate(metrics):
                ax = axes[row, col]
                data = [next(r for r in summary if r["benchmark"] == b and r["prior"] == prior and r["metric"] == metric) for b in BENCHMARKS]
                y = np.arange(len(data))
                for method, offset, color, label in [("posterior", -.14, "#76818d", "Posterior mean"), ("lcb", .14, "#2166ac", "LCB: mean − 1.645 × SD")]:
                    vals = np.array([r[f"{method}_mean"] for r in data])
                    lows = np.array([r[f"{method}_ci95_low"] for r in data])
                    highs = np.array([r[f"{method}_ci95_high"] for r in data])
                    ax.hlines(y+offset, lows, highs, color=color, lw=1.6)
                    ax.scatter(vals, y+offset, color=color, s=26, label=label, zorder=3)
                ax.set_title(f"{'Default' if prior == 'default' else 'Dataset'} prior", loc="left", fontsize=12, fontweight="bold")
                ax.set_xlabel(METRICS[metric], fontsize=10)
                ax.set_xlim(left=0)
                ax.grid(axis="x", color="#dce2e8", lw=.6)
                ax.set_yticks(y, list(BENCHMARKS.values()))
        axes[0, 0].invert_yaxis()
        title = "Recommendation stability across the saved evaluation paths" if len(metrics)>1 else METRICS[metrics[0]]
        fig.suptitle(title, x=.035, y=.985, ha="left", fontsize=17, fontweight="bold")
        fig.text(.035, .934, "1%–10% budget · Every actual recommendation update · Lower is better", fontsize=11, color="#556778")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.55, .061), ncol=2, frameon=False)
        footer = "Points: mean of per-run metrics; bars: pointwise 95% seed-block bootstrap intervals (10,000 draws).\nCompare methods within each dataset: matrix sizes and update frequencies differ. Exploratory analysis on fixed matrices."
        if len(metrics)==1:
            footer = "Points: mean of per-run metrics; bars: pointwise 95% bootstrap intervals.\n10,000 resamples of whole seed blocks. Compare methods within each dataset.\nExploratory analysis on fixed matrices; update frequencies differ."
        fig.text(.035, .021, footer, fontsize=9, color="#556778")
        fig.subplots_adjust(left=.13 if len(metrics)>1 else .30, right=.985, top=.865, bottom=.21, hspace=.45, wspace=.25)
        fig.savefig(OUT / f"{target}.png", dpi=180)
        fig.savefig(OUT / f"{target}.pdf")
        plt.close(fig)

    panels(["total_downward_pp", "maximum_drawdown_pp", "switches", "mean_regret_pp"], "overview", (15.5, 8))
    for metric in METRICS:
        panels([metric], metric, (8.4, 8))


def report(summary):
    lookup = {(r["benchmark"], r["prior"], r["metric"]): r for r in summary}
    text = """# Temporal recommendation stability — September 13, 2026

LCB (`mean − 1.645 × SD`) has less total downward movement, smaller maximum
drawdown, and fewer switches than posterior-mean recommendation in all ten
tested dataset/prior settings. This concerns temporal stability of the
recommendation rule on fixed Gittins observations, not a universal accuracy claim.

This analysis reuses all 520 saved trajectories (200 GSM8K + 200 PIQA + 120 MMLU).
The window is 1%–10% of the full matrix. Its initial state is the recommendation
already active at 1%; every actual subsequent batch update is included. No new
LLM calls or Gittins trajectories were needed. Both rules always recommend.

## Metric definitions

Let `q_t` be the recommended arm's full-row benchmark accuracy and
`r_t = q_best − q_t`. Each metric is computed within each run before aggregation.

- **Total downward movement:** `100 * sum(max(r_t − r_(t−1), 0))`.
  All harmful jumps accumulate. For accuracy 80% → 70% → 80% → 70%, this is
  20 pp, although the net loss is only 10 pp. It can exceed 100 pp.
- **Maximum drawdown:** `100 * max(r_t − min_(s<=t) r_s)`. This is the worst
  decline from the best recommendation reached since the window began.
- **Largest downward jump:** the biggest individual increase in regret, in pp.
- **Switches:** number of changes in arm identity, including harmless switches.
- **Budget-averaged drawdown:** drawdown integrated over evaluation budget and
  divided by the window width; captures the duration as well as size of setbacks.
- **Budget-averaged regret:** the same time-weighted calculation for regret;
  preserves the distinction between being stable and being consistently good.
- **Harmful jumps / jumps of at least 2 pp / any jump of at least 2 pp:** counts
  or per-run event indicators. The threshold includes exactly 2 pp, within
  floating-point tolerance. The summary mean of the event indicator is a risk.
- **Final regret:** recommendation quality at 10%, included as a separate
  accuracy check. A change exactly at 10% counts as a jump but has zero duration
  in the time-weighted metrics.

All movements use oracle full-row accuracy for evaluation only, not the evolving
posterior score. Areas hold the recommendation constant between actual updates;
there is no interpolation using future observations. No gap is treated as zero
movement: abstaining rules would need a separate analysis reporting coverage.

## Mean results

Every cell is **posterior mean → LCB**, averaging per-run quantities. Quality
movements are percentage points (pp); switches are counts. GSM8K/PIQA average
100 runs per prior; each MMLU subject averages 20.
"""
    for prior in PRIORS:
        text += f"\n### {prior.capitalize()} prior\n\n"
        text += "| Dataset | Total downward movement (pp) | Maximum drawdown (pp) | Switches | Largest drop (pp) | Budget-averaged regret (pp) |\n|---|---:|---:|---:|---:|---:|\n"
        for b, name in BENCHMARKS.items():
            vals=[]
            for metric in ["total_downward_pp", "maximum_drawdown_pp", "switches", "largest_drop_pp", "mean_regret_pp"]:
                r=lookup[b, prior, metric]
                vals.append(f"{r['posterior_mean']:.3f} → {r['lcb_mean']:.3f}")
            text += "| " + name + " | " + " | ".join(vals) + " |\n"
    text += "\n## Paired effects on cumulative deterioration and drawdown\n\nNegative differences favor LCB. These are differences of mean per-run metrics.\n\n| Dataset | Prior | Total-downward difference [95% interval], pp | Drawdown difference [95% interval], pp |\n|---|---|---:|---:|\n"
    for b,name in BENCHMARKS.items():
        for prior in PRIORS:
            vals=[]
            for metric in ["total_downward_pp", "maximum_drawdown_pp"]:
                r=lookup[b,prior,metric]
                vals.append(f"{r['paired_mean_delta']:.3f} [{r['paired_ci95_low']:.3f}, {r['paired_ci95_high']:.3f}]")
            text+=f"| {name} | {prior} | " + " | ".join(vals) + " |\n"
    text += "\n## Tail behavior and frequency of large drops\n\nDataset priors; p90 is computed across individual runs, not updates. The CSV\ncontains the default-prior results too. These tail/risk estimates are descriptive.\n\n| Dataset | P90 total downward movement, pp | P90 drawdown, pp | Probability of any ≥2 pp jump | Budget-averaged drawdown, pp |\n|---|---:|---:|---:|---:|\n"
    for b,name in BENCHMARKS.items():
        vals=[]
        for metric in ["total_downward_pp","maximum_drawdown_pp"]:
            r=lookup[b,"dataset",metric]
            vals.append(f"{r['posterior_p90']:.3f} → {r['lcb_p90']:.3f}")
        r=lookup[b,"dataset","any_jump_ge_2pp"]
        vals.append(f"{100*r['posterior_mean']:.0f}% → {100*r['lcb_mean']:.0f}%")
        r=lookup[b,"dataset","mean_drawdown_pp"]
        vals.append(f"{r['posterior_mean']:.3f} → {r['lcb_mean']:.3f}")
        text+=f"| {name} | " + " | ".join(vals) + " |\n"
    text += """
## Interpretation and limitations

Lower cumulative deterioration means less repeated backtracking; smaller
drawdown means smaller setbacks from previously achieved quality. Those are
directly relevant to recommendations that jump as evaluations arrive. The
comparison is paired within each trajectory and does not claim a benefit from
changing the allocation policy to LCB.

With dataset priors, mean total downward movement falls by approximately
43%–78%. The paired intervals for its mean reduction, and for mean maximum
drawdown and switch counts, exclude zero in all ten dataset/prior settings.
These intervals remain exploratory and unadjusted.

The differences in budget-averaged regret are much smaller and sometimes favor
posterior mean: for example, dataset-prior PIQA is 0.923 → 0.928 pp, despite
its substantially lower movement. Budget-averaged drawdown decreases in all
ten settings, but its default-prior Business Ethics interval includes zero.
Thus fewer and smaller jumps do not imply improvement on every quality or
duration-sensitive metric.

The analysis window, metric family, and 2 pp event threshold were chosen
exploratorily after examining the experiments. Intervals are pointwise and
unadjusted for multiple comparisons. They resample 20 whole sampling-seed blocks
with replacement (10,000 draws, seed 20260913), retaining all fixed matrices and
both rules in each block. They describe Monte Carlo variation on these matrices,
not new datasets or questions. Tail estimates from 20 MMLU seeds are coarse.

Do not rank datasets by raw total movement or switch counts: their matrix
dimensions and batch sizes differ. Compare the paired rules within the same
dataset, prior, and budget window. A stable but poor arm could score perfectly
on movement, so budget-averaged and final regret are retained alongside stability.

## Files

- [Overview PNG](figures/stability/overview.png) / [PDF](figures/stability/overview.pdf)
- [Per-run metrics](figures/stability/per_run_metrics.csv)
- [Summary with paired intervals and p90s](figures/stability/summary.csv)
- [Analysis configuration and input hashes](figures/stability/manifest.json)

### Figures for every metric

"""
    for metric, label in METRICS.items():
        text+=f"- [{label}](figures/stability/{metric}.png)\n"
    text+="\n![Stability comparison](figures/stability/overview.png)\n"
    (ROOT / "recommend/ROBUSTNESS_RESULTS.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources = collect()
    summary = summarize(rows)
    write_csv(OUT / "per_run_metrics.csv", rows)
    write_csv(OUT / "summary.csv", summary)
    plot(summary)
    report(summary)
    manifest = {"window": [BEGIN, END], "methods": METHODS, "datasets": BENCHMARKS,
        "trajectories": len(rows)//2, "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_unit": "whole sampling-seed blocks, matrices fixed",
        "intervals": "pointwise, unadjusted, exploratory", "large_jump_threshold_pp": 2,
        "threshold_comparison": ">=", "python": sys.version, "numpy": np.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "source_sha256": digest(Path(__file__).resolve()),
        "test_source_sha256": digest(ROOT / "recommend/test_stability.py"), "input_sha256": sources,
        "status": "complete"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Analyzed {len(rows)//2} trajectories, {len(METRICS)} metrics, and both priors; wrote {OUT}")
    for r in summary:
        if r["metric"] in ["mean_regret_pp", "mean_drawdown_pp"]:
            print(r["benchmark"], r["prior"], r["metric"], round(r["posterior_mean"],3), "->", round(r["lcb_mean"],3), "CI delta", round(r["paired_ci95_low"],3), round(r["paired_ci95_high"],3))


if __name__ == "__main__":
    main()
