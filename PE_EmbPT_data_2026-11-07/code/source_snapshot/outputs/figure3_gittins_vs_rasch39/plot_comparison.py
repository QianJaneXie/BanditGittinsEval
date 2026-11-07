"""Plot the Rasch39 experiments with the supplied Figure 3 Gittins curves on actual spend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RASCH_DIR = HERE.parent / "mmlu_rasch39" / "results"
FEATURE_DIR = HERE.parent / "mmlu_feature_comparison"
PRIOR_FIGURE = HERE.parent / "figure3_gittins_vs_logistic" / "plot_comparison.py"
sys.path.insert(0, str(FEATURE_DIR))
from plot_results import BUDGET_TOLERANCE, PLOT_GRID, SEEDS, SPEND_FIELDS, Trace, load_subjects, mean_and_se, write_csv


CHECKPOINTS = (2.0, 4.0, 6.0, 8.0, 9.9)
RASCH_BLOCKS = ("rasch39_random", "rasch39_balanced")
BAI_METHODS = ("logistic_39", *RASCH_BLOCKS)
GITTINS_METHODS = ("gittins_s", "gittins_g")
PRIMARY = (*GITTINS_METHODS, "rasch39_balanced", "logistic_39")
METHODS = {
    "gittins_s": ("GittinsEval-S", "#e57716", "-"),
    "gittins_g": ("GittinsEval-G", "#278745", "-"),
    "rasch39_balanced": ("PE-EmbPT (joint task, 39 arm features + question effects)", "#a92f63", "-"),
    "logistic_39": ("Logistic BAI (39 features)", "#194c82", "-"),
    "rasch39_random": ("PE-EmbPT (random questions)", "#74539a", "--"),
}
PAIRS = (
    ("rasch_random_minus_logistic", 1, 0),
    ("rasch_balanced_minus_random", 2, 1),
    ("rasch_balanced_minus_logistic", 2, 0),
)


def load_bai(subjects, tasks: list[str], rasch_dir: Path, logistic_dir: Path):
    archives = {task: rasch_dir / f"bai_results_MMLU_{task}_combined.npy" for task in tasks}
    missing = [task for task, path in archives.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed Rasch archives for {len(missing)} subjects: {', '.join(missing)}")

    traces = {}
    sources = {}
    for task, path in archives.items():
        subject = subjects[task]
        raw = np.load(path, allow_pickle=True).item()
        rasch = raw["reproduction"]
        logistic = np.load(logistic_dir / path.name, allow_pickle=True).item()["reproduction"]
        expected_feature = logistic["feature_sha256"]["pretrained_embedding"]
        if (
            [tuple(job) for job in raw["jobs"]] != [(task, seed) for seed in SEEDS]
            or list(raw["seeds"]) != list(SEEDS)
            or tuple(raw["feature_names"]) != RASCH_BLOCKS
            or len(raw["out"]) != len(SEEDS)
            or not raw["combine_models"]
            or raw["n_models_stacked"] != 15
            or not raw["costs_recorded"]
            or rasch["prompt_feature_sha256"] != expected_feature
            or rasch["reference_logistic_archive_sha256"] != subject.archive_sha256
            or rasch["reference_logistic_fingerprint"] != logistic["fingerprint"]
            or rasch["matrix_sha256"] != logistic["matrix_sha256"]
            or rasch["input_sha256"] != logistic["input_sha256"]
            or rasch["prompt_feature_width"] != 39
            or rasch["question_feature_width"] != subject.example_count
            or rasch["fitted_coefficients"] != 39 + subject.example_count
        ):
            raise ValueError(f"The completed Rasch archive does not match the logistic reference or protocol: {path}")

        task_traces = []
        denominators = np.array([subject.exhaustive_observations, subject.exhaustive_price])[:, None]
        for seed, run in zip(SEEDS, raw["out"], strict=True):
            if len(run) != 1 or len(run[0]) != len(RASCH_BLOCKS):
                raise ValueError(f"{task} seed {seed} does not have both Rasch variants.")
            logistic_trace = subject.traces[seed][2]
            seed_traces = [logistic_trace]
            for phases in run[0]:
                if [phase["phase"] for phase in phases] != list(range(5)):
                    raise ValueError(f"{task} seed {seed} does not have all five phase records.")
                spend = np.array([[phase[field] for phase in phases] for field in SPEND_FIELDS], dtype=float)
                regret = np.array([phase["simple_regret"] for phase in phases], dtype=float)
                if (
                    not np.isfinite(spend).all() or not np.isfinite(regret).all()
                    or np.any(spend <= 0) or np.any(np.diff(spend, axis=1) < 0)
                    or np.any(regret < -BUDGET_TOLERANCE) or np.any(regret > 1 + BUDGET_TOLERANCE)
                ):
                    raise ValueError(f"Invalid spend or regret in {task} seed {seed}.")
                seed_traces.append(Trace(100.0 * spend / denominators, regret))
            if not np.allclose(seed_traces[1].spend_percent[:, 0], logistic_trace.spend_percent[:, 0],
                               atol=BUDGET_TOLERANCE, rtol=0):
                raise ValueError(f"The controlled random sampler changed phase-one spend in {task} seed {seed}.")
            task_traces.append(tuple(seed_traces))
        traces[task] = tuple(task_traces)
        sources[task] = {
            "rasch_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rasch_fingerprint": rasch["fingerprint"],
            "logistic_archive_sha256": subject.archive_sha256,
            "shared_X39_feature_sha256": expected_feature,
            "question_effect_count": subject.example_count,
        }
    return traces, sources


def bai_at(task_traces, mode_index: int, grid: np.ndarray) -> np.ndarray:
    """Return exact phase-held regret indexed by BAI method, seed, and spend."""
    result = np.full((len(BAI_METHODS), len(SEEDS), len(grid)), np.nan)
    for seed, traces in enumerate(task_traces):
        for method, trace in enumerate(traces):
            observed = np.searchsorted(trace.spend_percent[mode_index], grid + BUDGET_TOLERANCE, side="right") - 1
            available = observed >= 0
            result[method, seed, available] = trace.regret[observed[available]]
    return result


def cache_at(x: np.ndarray, mean: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, mean, left=np.nan, right=np.nan)


def make_curves(previous, groups, subjects, traces):
    supplement = previous["SUPPLEMENT"]
    read_csv = previous["read_csv"]
    prices = {(row["mode"], row["task"]): float(row["full_evaluation_cost"])
              for row in read_csv(supplement / "prompteval_denominators.csv")}
    raw_cache = {}
    paths = {}
    curves = {}
    for mode_index, (mode, _, raw_x_field) in enumerate(previous["MODES"]):
        for group, tasks in groups.items():
            for task in tasks:
                subject = subjects[task]
                local = subject.exhaustive_observations if mode == "unit" else subject.exhaustive_price
                denominator = prices[mode, task]
                if not np.isclose(denominator, local, atol=1e-8, rtol=1e-10):
                    raise ValueError(f"The provided and local exhaustive denominators differ for {task}, {mode}.")
                batch_size = 2 if group.endswith("Small") else 8
                raw_cache[mode, task], source_paths = previous["load_gittins"](task, mode, raw_x_field, denominator, batch_size)
                paths.update(source_paths)

            for method in GITTINS_METHODS:
                per_task = [raw_cache[mode, task][method] for task in tasks]
                start = max(x[0] for x, _ in per_task)
                end = min(10.0, min(x[-1] for x, _ in per_task))
                grid = np.r_[start, PLOT_GRID[(PLOT_GRID > start) & (PLOT_GRID < end)], end]
                task_values = np.array([cache_at(x, mean, grid) for x, mean in per_task])
                if not np.isfinite(task_values).all():
                    raise ValueError(f"The {method} cohort curve would require extrapolation for {group}, {mode}.")
                mean, se, _ = mean_and_se(task_values, axis=0)
                curves[mode, group, method] = (grid, mean, se)

            for method_index, method in enumerate(BAI_METHODS):
                runs = np.array([bai_at(traces[task], mode_index, PLOT_GRID)[method_index] for task in tasks])
                complete = np.isfinite(runs).all(axis=(0, 1))
                task_values = runs[:, :, complete].mean(axis=1)
                mean, se, _ = mean_and_se(task_values, axis=0)
                curves[mode, group, method] = (PLOT_GRID[complete], mean, se)
    return curves, raw_cache, paths


def export_checkpoints(out_dir, previous, groups, traces, raw_cache, *, checkpoints=CHECKPOINTS, prefix=""):
    grid = np.array(checkpoints)
    gittins_methods = GITTINS_METHODS if raw_cache else ()
    subject_rows = []
    aggregate_rows = []
    for mode_index, (mode, _, _) in enumerate(previous["MODES"]):
        for group, tasks in groups.items():
            bai = np.array([bai_at(traces[task], mode_index, grid) for task in tasks])
            values, seed_se, seed_count = mean_and_se(bai, axis=2)
            gittins = np.array([[cache_at(*raw_cache[mode, task][method], grid) for method in gittins_methods]
                                for task in tasks])
            paired = np.array([bai[:, minuend] - bai[:, subtrahend] for _, minuend, subtrahend in PAIRS])
            deltas, delta_seed_se, paired_seed_count = mean_and_se(paired, axis=2)

            for budget_index, budget in enumerate(grid):
                base = {"mode": mode, "group": group, "x_percent_exhaustive": float(budget)}
                aggregate = {**base, "cohort_subjects": len(tasks)}
                for subject_index, task in enumerate(tasks):
                    row = {**base, "subject": task}
                    for index, method in enumerate(gittins_methods):
                        value = gittins[subject_index, index, budget_index]
                        row[f"{method}_mean_regret"] = finite(value)
                        row[f"{method}_contributing_seeds"] = len(SEEDS) if np.isfinite(value) else 0
                    for index, method in enumerate(BAI_METHODS):
                        at = subject_index, index, budget_index
                        row[f"{method}_mean_regret"] = finite(values[at])
                        row[f"{method}_seed_se"] = finite(seed_se[at])
                        row[f"{method}_contributing_seeds"] = int(seed_count[at])
                    for index, (pair, _, _) in enumerate(PAIRS):
                        at = index, subject_index, budget_index
                        row[pair] = finite(deltas[at])
                        row[f"{pair}_seed_se"] = finite(delta_seed_se[at])
                        row[f"{pair}_paired_seeds"] = int(paired_seed_count[at])
                    subject_rows.append(row)

                for index, method in enumerate(gittins_methods):
                    estimates = gittins[:, index, budget_index]
                    add_full_subject_stats(aggregate, method, estimates, np.isfinite(estimates))
                for index, method in enumerate(BAI_METHODS):
                    estimates = values[:, index, budget_index]
                    add_full_subject_stats(aggregate, method, estimates, seed_count[:, index, budget_index] == len(SEEDS))
                for index, (pair, _, _) in enumerate(PAIRS):
                    estimates = deltas[index, :, budget_index]
                    complete = paired_seed_count[index, :, budget_index] == len(SEEDS)
                    add_full_subject_stats(aggregate, pair, estimates, complete)
                aggregate_rows.append(aggregate)

    write_csv(out_dir / f"{prefix}checkpoints.csv", aggregate_rows)
    write_csv(out_dir / f"{prefix}subject_checkpoints.csv", subject_rows)


def finite(value):
    return float(value) if np.isfinite(value) else None


def add_full_subject_stats(row, label, estimates, complete):
    row[f"{label}_full_coverage_subjects"] = int(complete.sum())
    if complete.all():
        mean, se, _ = mean_and_se(estimates, axis=0)
        row[f"{label}_mean"] = finite(mean)
        row[f"{label}_subject_se"] = finite(se)
    else:
        row[f"{label}_mean"] = None
        row[f"{label}_subject_se"] = None


def render(out_dir, previous, curves, groups, *, sensitivity: bool):
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.1), sharex=True, sharey="col")
    methods = (*PRIMARY, "rasch39_random") if sensitivity else PRIMARY
    modes = previous["MODES"]
    for row, (mode, label, _) in enumerate(modes):
        for column, (group, tasks) in enumerate(groups.items()):
            ax = axes[row, column]
            for method in methods:
                name, color, style = METHODS[method]
                if sensitivity and method == "rasch39_balanced":
                    name += " · balanced questions"
                x, mean, se = curves[mode, group, method]
                is_bai = method in BAI_METHODS
                ax.plot(x, mean, label=name, color=color, lw=2.0, linestyle=style,
                        drawstyle="steps-post" if is_bai else "default")
                ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.13,
                                step="post" if is_bai else None, linewidth=0)
                if is_bai:
                    ax.plot(x[-1:], mean[-1:], marker="o", color=color, markersize=3.2)
            ax.set(xlim=(0, 10.12), xticks=(0, 5, 10))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.grid(axis="y", linewidth=0.65, color="#c8cdd4", alpha=0.52)
            if row == 0:
                ax.set_title(f"{group.replace('-', '–')} ({len(tasks)})", fontweight="semibold", pad=8)
            if column == 0:
                ax.set_ylabel(f"{label}\nSimple regret")
    for column, group in enumerate(groups):
        ymax = max(float(np.max(curves[mode, group, method][1] + curves[mode, group, method][2]))
                   for mode, _, _ in modes for method in methods)
        for ax in axes[:, column]:
            ax.set_ylim(0, 1.04 * ymax)
            previous["expand_lower_regret"](ax, threshold=0.05, fraction=0.65)

    title = "PE-EmbPT with random and balanced questions" if sensitivity else "GittinsEval, PE-EmbPT, and logistic BAI on MMLU"
    fig.suptitle(title, fontsize=17, fontweight="semibold", y=0.978)
    fig.text(0.5, 0.922, "Actual budgets  ·  19 Figure 3 subjects  ·  20 seeds per subject  ·  Equal subject weights", ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.067), ncol=3 if sensitivity else 2,
               frameon=False, fontsize=9.1)
    fig.supxlabel("Percent of exhaustive evaluation", y=0.137)
    fig.text(0.5, 0.048, "Regret below 0.05 is expanded. Shading shows ±1 subject SE. GittinsEval ends at full-run support (about 9.97–10%).",
             ha="center", fontsize=8.8, color="#4a515a")
    fig.text(0.5, 0.025, "Price weighting reuses the BAI allocation histories. Final BAI recommendations carry forward to 10%.",
             ha="center", fontsize=8.8, color="#4a515a")
    fig.subplots_adjust(left=0.074, right=0.987, bottom=0.195, top=0.864, wspace=0.29, hspace=0.24)
    name = "gittins_vs_rasch39_sampling" if sensitivity else "gittins_vs_rasch39"
    fig.savefig(out_dir / f"{name}.png", dpi=230)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasch-results", type=Path, default=RASCH_DIR)
    parser.add_argument("--logistic-results", type=Path, default=FEATURE_DIR / "results")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--all-subject-checkpoints", action="store_true",
                        help="Also export 57-subject BAI-only checkpoints at 2, 4, 6, 8, and 10 percent.")
    args = parser.parse_args()
    previous = runpy.run_path(str(PRIOR_FIGURE), run_name="_audited_figure3_loader")
    groups = previous["load_cohorts"]()
    tasks = [task for cohort in groups.values() for task in cohort]
    subjects = {subject.name: subject for subject in load_subjects(args.logistic_results)}
    loaded_tasks = sorted(subjects) if args.all_subject_checkpoints else tasks
    traces, bai_sources = load_bai(subjects, loaded_tasks, args.rasch_results, args.logistic_results)
    curves, raw_cache, gittins_sources = make_curves(previous, groups, subjects, traces)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    export_checkpoints(args.out_dir, previous, groups, traces, raw_cache)
    if args.all_subject_checkpoints:
        export_checkpoints(args.out_dir, previous, {"All 57 subjects": loaded_tasks}, traces, {},
                           checkpoints=(2.0, 4.0, 6.0, 8.0, 10.0), prefix="all57_bai_")
    plot_rows = [
        {"mode": mode, "group": group, "method": method, "x_percent_exhaustive": float(spend),
         "mean_simple_regret": float(regret), "subject_se": float(error), "subjects": len(groups[group]),
         "seeds_per_subject": len(SEEDS)}
        for (mode, group, method), (x, mean, se) in curves.items()
        for spend, regret, error in zip(x, mean, se, strict=True)
    ]
    write_csv(args.out_dir / "plotted_curves.csv", plot_rows)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    render(args.out_dir, previous, curves, groups, sensitivity=False)
    render(args.out_dir, previous, curves, groups, sensitivity=True)

    provenance = {
        "cohorts": groups,
        "comparison": "Joint 1,500-arm, 5-phase MMLU BAI using the same X39 arm design. These are adaptations of PromptEval estimators to the joint task and budget schedule.",
        "pe_embpt_scope": "The Rasch variants use PE-EmbPT's pretrained prompt representation and question effects, with 14 model indicators added for the joint task. The replay uses five phases and reports the highest-scoring active arm at each phase. The original PromptEval experiment evaluated prompts separately for each model.",
        "variants": {
            "logistic_39": "Historical arm-only logistic BAI with random unseen questions and arm-score ranking.",
            "rasch39_random": "X39 arm covariates plus a question indicator per item. Rank by full-row means combining observed labels and fitted probabilities. Use the historical random-question sampling rule.",
            "rasch39_balanced": "Same Rasch estimator and scoring with the upstream-style least-covered-question sampling rule.",
        },
        "axis": "Actual cumulative observations or quoted input-token price weights divided by each subject's exhaustive denominator.",
        "gittins_source": "Supplied Figure 3 per-subject LCB caches in raw spend units. Linearly interpolate contiguous all-20-run subject means within their support only. Require all cohort subjects and stop at the earliest full-coverage endpoint or 10%.",
        "bai_aggregation": "Use each run's latest available phase and carry its final recommendation forward. Require all seeds and all cohort subjects for plotted points and aggregate checkpoint cells. Subject checkpoint rows retain available-case seed means and counts.",
        "uncertainty": "The figures show ±1 SE across equally weighted subject means. Paired BAI checkpoint contrasts first pair seeds within a subject, then aggregate the subject differences. Gittins per-seed histories are unavailable.",
        "checkpoints_percent": list(CHECKPOINTS),
        "all57_bai_checkpoint_percentages": [2.0, 4.0, 6.0, 8.0, 10.0] if args.all_subject_checkpoints else None,
        "plot_grid_step_percent": float(PLOT_GRID[1] - PLOT_GRID[0]),
        "support": {f"{mode}/{group}/{method}": [float(x[0]), float(x[-1])]
                    for (mode, group, method), (x, _, _) in curves.items()},
        "gittins_source_sha256": gittins_sources,
        "bai_sources": bai_sources,
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote both figures and checkpoints for {len(tasks)} subjects to {args.out_dir}")


if __name__ == "__main__":
    main()
