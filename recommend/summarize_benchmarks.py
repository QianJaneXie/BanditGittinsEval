"""Export cross-dataset comparisons without choosing LCB strength from results."""
from pathlib import Path
import csv
import json
import numpy as np
import plot_results as plots
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ["piqa", "mmlu_computer_security", "mmlu_anatomy", "mmlu_business_ethics"]
LABELS = ["PIQA", "Computer Security", "Anatomy", "Business Ethics"]
PRIMARY = "gaussian_lcb_1.645"
BASELINE = "posterior_mean"
KEYS = [BASELINE, "empirical_mean", PRIMARY, "finite_gaussian_lcb_1.645", "finite_population_lcb", "variance_gate_0.1"]


def main():
    plots.setup_style()
    out = ROOT / "recommend/figures/piqa_mmlu"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 11.5), sharex=True)
    effects, effect_axes = plt.subplots(4, 2, figsize=(12.5, 11.5), sharex=True)
    rows = []
    diagnostics = []
    for row, (benchmark, label) in enumerate(zip(BENCHMARKS, LABELS)):
        manifest, methods, traces = plots.load_results(ROOT / f"recommend/results/{benchmark}")
        with (ROOT / f"recommend/figures/{benchmark}/budget_checkpoints.csv").open() as handle:
            rows.extend(dict(benchmark=benchmark, benchmark_label=label, **r) for r in csv.DictReader(handle))
        for col, prior in enumerate(plots.PRIOR_ORDER):
            runs = [t for t in traces if t.prior == prior]
            matrix_seeds, run_seeds = np.array([t.matrix_seed for t in runs]), np.array([t.run_seed for t in runs])
            arrays = {key: np.array([plots.hold_last(t, key)[0] for t in runs]) for key in [BASELINE, PRIMARY]}
            ax, dax = axes[row, col], effect_axes[row, col]
            mu0, v0 = manifest["priors"][prior]
            title = f"{label}  ·  {'Default' if prior == 'default' else 'Dataset'} prior N({mu0:g}, {v0:g})"
            for key, color, style, name in [(BASELINE, "#66717e", "--", "Posterior mean"), (PRIMARY, "#2166ac", "-", "LCB: mean − 1.645 × SD")]:
                mean, _, low, high = plots.stratified_mean_ci(arrays[key], matrix_seeds, run_seeds)
                ax.plot(100 * plots.GRID, 100 * mean, color=color, lw=1.65, ls=style, drawstyle="steps-post", label=name)
                ax.fill_between(100 * plots.GRID, 100 * np.maximum(low, 0), 100 * high, color=color, alpha=.11, linewidth=0, step="post")
            delta, _, low, high = plots.stratified_mean_ci(arrays[PRIMARY] - arrays[BASELINE], matrix_seeds, run_seeds)
            dax.axhline(0, color="#66717e", lw=1, ls="--")
            dax.plot(100 * plots.GRID, 100 * delta, color="#2166ac", lw=1.7, drawstyle="steps-post")
            dax.fill_between(100 * plots.GRID, 100 * low, 100 * high, color="#2166ac", alpha=.15, linewidth=0, step="post")
            for panel in (ax, dax):
                panel.set_title(title, loc="left", fontsize=11, fontweight="bold")
                panel.grid(True, axis="y")
                panel.set_xlim(1, 10)
                panel.set_xticks([1, 2, 5, 10])
                panel.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
                if row == 3:
                    panel.set_xlabel("Evaluations / full matrix")
            ax.set_ylim(bottom=0)
            if col == 0:
                ax.set_ylabel("Mean regret (accuracy pp)")
                dax.set_ylabel("LCB − mean regret (pp)")
            for budget in [.01, .02, .05, .1]:
                for key in KEYS:
                    samples = []
                    for trace in runs:
                        ix = np.searchsorted(trace.x / trace.n_cells, budget, side="right") - 1
                        j = trace.method_keys.index(key)
                        j0 = trace.method_keys.index(BASELINE)
                        arm = trace.recommended_arm[ix, j]
                        regret = trace.regret[ix, j]
                        baseline = trace.regret[ix, j0]
                        samples.append((arm >= 0, trace.selected_n[ix, j], regret, baseline))
                    values = np.array(samples)
                    valid = values[:, 0] > 0
                    diagnostics.append(dict(benchmark=benchmark, prior=prior, budget_fraction=budget, method=key,
                        coverage=float(valid.mean()), unseen_recommendation_rate=float(np.mean(valid & (values[:, 1] == 0))),
                        baseline_regret_on_same_eligible_runs=float(np.mean(values[valid, 3])) if valid.any() else None,
                        mean_regret=float(np.mean(values[valid, 2])) if valid.any() else None,
                        paired_better_runs=int(np.sum(values[valid, 2] < values[valid, 3] - 1e-12)),
                        paired_worse_runs=int(np.sum(values[valid, 2] > values[valid, 3] + 1e-12)),
                        paired_tied_runs=int(np.sum(np.abs(values[valid, 2] - values[valid, 3]) <= 1e-12))))
        # Common vertical scale within each dataset makes the prior panels comparable.
        for panels in (axes, effect_axes):
            lo = min(panels[row, c].get_ylim()[0] for c in [0, 1])
            hi = max(panels[row, c].get_ylim()[1] for c in [0, 1])
            # Data below the displayed 1% budget must not inflate the y scale.
            if panels is axes:
                vals = []
                for c in [0, 1]:
                    for collection in panels[row, c].collections:
                        for path in collection.get_paths():
                            verts = path.vertices
                            vals.extend(verts[(verts[:, 0] >= 1) & (verts[:, 0] <= 10), 1])
                hi = max(np.nanmax(vals) * 1.05, .1)
                lo = 0
            else:
                # Recompute from visible bands and line vertices to include the CI.
                vals = []
                for c in [0, 1]:
                    for collection in panels[row, c].collections:
                        for path in collection.get_paths():
                            verts = path.vertices
                            vals.extend(verts[(verts[:, 0] >= 1) & (verts[:, 0] <= 10), 1])
                span = max(np.nanmax(np.abs(vals)), .02)
                lo, hi = -span * 1.08, span * 1.08
            for c in [0, 1]:
                panels[row, c].set_ylim(lo, hi)
    for figure, title, subtitle, filename in [
        (fig, "Does LCB improve recommendations on other datasets?", "Same observations for both rules. Lower regret is better. Budgets shown: 1–10%.", "cross_dataset_lcb"),
        (effects, "LCB's effect depends on the dataset, prior, and budget", "Paired regret difference. Below zero favors LCB; above zero favors posterior mean.", "cross_dataset_lcb_effect"),
    ]:
        figure.suptitle(title, x=.07, ha="left", y=.982, fontsize=18, fontweight="bold")
        figure.text(.07, .95, subtitle, fontsize=11, color="#556778")
        figure.text(.07, .021, "LCB strength fixed at z = 1.645. Bands: pointwise 95% paired seed-block bootstrap intervals (1,000 draws).\nPIQA: 5 fixed matrices × 20 seeds per prior; each MMLU subject: 1 fixed matrix × 20 seeds per prior. No dataset resampling.", fontsize=9, color="#556778")
        if figure is fig:
            figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .059), ncol=2, frameon=False)
        figure.subplots_adjust(left=.075, right=.98, top=.91, bottom=.12, hspace=.36, wspace=.20)
        figure.savefig(out / f"{filename}.png", dpi=180)
        figure.savefig(out / f"{filename}.pdf")
        plt.close(figure)
    plots.write_csv(out / "all_methods_checkpoints.csv", rows)
    primary = [r for r in rows if r["method_key"] in [BASELINE, PRIMARY]]
    plots.write_csv(out / "primary_lcb_checkpoints.csv", primary)
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"Wrote cross-dataset figures, checkpoints, and diagnostics to {out}")


if __name__ == "__main__":
    main()
