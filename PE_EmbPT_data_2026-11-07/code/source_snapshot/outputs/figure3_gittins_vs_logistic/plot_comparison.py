"""Reaggregate the Figure 3 cohorts on actual evaluation budgets."""

from __future__ import annotations

import csv
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
BUNDLE = REPO / "Figure3_data_2026-11-07"
SUPPLEMENT = BUNDLE / "supplementary"
MODEL_DIR = HERE.parent / "mmlu_model_only_logistic"
FEATURE_DIR = HERE.parent / "mmlu_feature_comparison"
sys.path.insert(0, str(BUNDLE / "code"))
from piecewise_regret_axis import expand_lower_regret


GRID = np.linspace(0.0, 10.0, 401)
CHECKPOINTS = (2.0, 4.0, 6.0, 8.0, 9.9)
SEED_COUNT = 20
GROUP_SIZES = {"Easy-Small": 6, "Easy-Large": 2, "Hard-Small": 9, "Hard-Large": 2}
MODES = (("unit", "Unit cost", "cum_eval"), ("aware", "Price-weighted cost", "cum_original_cost"))
METHODS = {
    "gittins_s": ("GittinsEval-S", "#e57716", "linear"),
    "gittins_g": ("GittinsEval-G", "#278745", "linear"),
    "logistic_39": ("Logistic BAI (39 features)", "#194c82", "step"),
    "logistic_14": ("Logistic BAI (14 features)", "#765594", "step"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cohorts() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {group: [] for group in GROUP_SIZES}
    for row in read_csv(BUNDLE / "data" / "selected_tasks.csv"):
        groups[row["group"]].append(row["task"])
    if {group: len(tasks) for group, tasks in groups.items()} != GROUP_SIZES or len(set(sum(groups.values(), []))) != 19:
        raise ValueError("The supplied selected-task table does not define the expected 19 distinct subjects.")
    return groups


def load_gittins(
    task: str, mode: str, raw_x_field: str, denominator: float, batch_size: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    stem = SUPPLEMENT / "per_task" / f"{mode}_{task}"
    curve_file = stem.with_name(stem.name + "_curves.csv")
    meta_file = stem.with_name(stem.name + "_meta.json")
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    replacement = metadata["lcb_replacement"]
    if (
        metadata["mmlu_task"] != task or metadata["mode"] != mode or metadata["x_label"] != raw_x_field
        or replacement["recommendation_rule"] != "lcb" or replacement["batch_size"] != batch_size
        or set(replacement["replaced_method_kinds"]) != {"gittins_s", "gittins_g"}
    ):
        raise ValueError(f"Unexpected per-subject cache provenance in {meta_file}.")
    records = read_csv(curve_file)
    curves = {}
    for method in ("gittins_s", "gittins_g"):
        rows = [row for row in records if row["method_kind"] == method]
        if len(rows) != 320 or any(row["mmlu_task"] != task or row["mode"] != mode or row["x_label"] != raw_x_field for row in rows):
            raise ValueError(f"Unexpected source grid for {task}, {mode}, {method}.")
        full = np.array([int(row["n_runs"]) == SEED_COUNT for row in rows])
        supported = np.flatnonzero(full)
        if len(supported) < 2 or np.any(np.diff(supported) != 1):
            raise ValueError(f"Missing full-coverage interval for {task}, {mode}, {method}.")
        x = np.array([float(row["x"]) for row in rows])[full] / denominator * 100.0
        y = np.array([float(row["mean"]) if row["mean"] else np.nan for row in rows])[full]
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
            raise ValueError(f"Invalid full-coverage curve for {task}, {mode}, {method}.")
        curves[method] = (x, y)
    return curves, {path.relative_to(REPO).as_posix(): digest(path) for path in (curve_file, meta_file)}


def summarize(logistic_report, groups, comparisons):
    denominators = {
        (row["mode"], row["task"]): float(row["full_evaluation_cost"])
        for row in read_csv(SUPPLEMENT / "prompteval_denominators.csv")
    }
    if len(denominators) != 38:
        raise ValueError("Expected exhaustive denominators for both cost views and all 19 subjects.")
    indexed = {comparison.subject.name: index for index, comparison in enumerate(comparisons)}
    mean_and_se = logistic_report["mean_and_se"]
    values, _ = logistic_report["measure"](comparisons, GRID)
    plot_rows = []
    data = {}
    source_files = {}
    for mode_index, (mode, _, raw_x_field) in enumerate(MODES):
        for group, tasks in groups.items():
            members = [indexed[task] for task in tasks]
            cache = {"gittins_s": [], "gittins_g": []}
            for task, index in zip(tasks, members, strict=True):
                subject = comparisons[index].subject
                expected = subject.exhaustive_observations if mode == "unit" else subject.exhaustive_price
                denominator = denominators[(mode, task)]
                if not np.isclose(denominator, expected, atol=1e-8, rtol=1e-10):
                    raise ValueError(f"Budget denominators differ for {task}, {mode}.")
                batch_size = 2 if group.endswith("Small") else 8
                curves, files = load_gittins(task, mode, raw_x_field, denominator, batch_size)
                source_files.update(files)
                for method, curve in curves.items():
                    cache[method].append(curve)

            for method, curves in cache.items():
                start = max(x[0] for x, _ in curves)
                stop = min(10.0, min(x[-1] for x, _ in curves))
                x = np.r_[start, GRID[(GRID > start) & (GRID < stop)], stop]
                task_means = np.array([np.interp(x, old_x, y, left=np.nan, right=np.nan) for old_x, y in curves])
                if not np.isfinite(task_means).all():
                    raise ValueError(f"The grouped {method} curve would require extrapolation in {group}, {mode}.")
                mean, se, _ = mean_and_se(task_means, axis=0)
                data[(mode, group, method)] = (x, mean, se)

            for design, method in enumerate(("logistic_14", "logistic_39")):
                traces = values[mode_index, members, design]
                complete = np.isfinite(traces).all(axis=(0, 1))
                task_means, _, _ = mean_and_se(traces[:, :, complete], axis=1)
                mean, se, _ = mean_and_se(task_means, axis=0)
                data[(mode, group, method)] = (GRID[complete], mean, se)

            for method in METHODS:
                x, mean, se = data[(mode, group, method)]
                plot_rows.extend({
                    "mode": mode, "group": group, "method": method,
                    "x_percent_exhaustive": float(spend), "mean_simple_regret": float(regret),
                    "subject_se": float(error), "subjects": len(tasks), "seeds_per_subject": SEED_COUNT,
                } for spend, regret, error in zip(x, mean, se, strict=True))
    return data, plot_rows, source_files


def checkpoint_rows(data, groups) -> list[dict]:
    rows = []
    for mode, _, _ in MODES:
        for group, tasks in groups.items():
            for budget in CHECKPOINTS:
                record = {"mode": mode, "group": group, "x_percent_exhaustive": budget, "subjects": len(tasks)}
                for method in METHODS:
                    x, mean, se = data[mode, group, method]
                    match = np.flatnonzero(np.isclose(x, budget, atol=1e-10, rtol=0))
                    record[f"{method}_mean_simple_regret"] = float(mean[match[0]]) if len(match) else None
                    record[f"{method}_subject_se"] = float(se[match[0]]) if len(match) else None
                rows.append(record)
    return rows


def render(data, groups, *, extended: bool) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.7), sharex=True, sharey="col")
    shown = tuple(METHODS) if extended else tuple(METHODS)[:3]
    for row, (mode, row_label, _) in enumerate(MODES):
        for column, (group, tasks) in enumerate(groups.items()):
            ax = axes[row, column]
            for method in shown:
                label, color, curve = METHODS[method]
                x, mean, se = data[mode, group, method]
                step = "post" if curve == "step" else None
                line = "--" if method == "logistic_14" else "-"
                ax.plot(x, mean, label=label, color=color, lw=2.0, linestyle=line,
                        drawstyle="steps-post" if step else "default")
                ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.13, step=step, linewidth=0)
                if step:
                    ax.plot(x[-1:], mean[-1:], marker="o", color=color, markersize=3.2)
            ax.set(xlim=(0, 10.12), xticks=(0, 5, 10))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.grid(axis="y", linewidth=0.65, color="#c8cdd4", alpha=0.52)
            if row == 0:
                ax.set_title(f"{group.replace('-', '–')} ({len(tasks)})", fontweight="semibold", pad=8)
            if column == 0:
                ax.set_ylabel(f"{row_label}\nSimple regret")
    for column, group in enumerate(groups):
        ymax = max(float(np.max(data[mode, group, method][1] + data[mode, group, method][2]))
                   for mode, _, _ in MODES for method in shown)
        for ax in axes[:, column]:
            ax.set_ylim(0, 1.04 * ymax)
            expand_lower_regret(ax, threshold=0.05, fraction=0.65)

    title = "GittinsEval and the logistic BAI feature ablation" if extended else "GittinsEval vs. current logistic BAI on MMLU"
    fig.suptitle(title, fontsize=17, fontweight="semibold", y=0.978)
    fig.text(0.5, 0.92, "Recalculated on actual budgets  ·  19 Figure 3 subjects  ·  20 seeds per subject  ·  Equal subject weights", ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.061), ncol=2 if extended else 3, frameon=False)
    fig.supxlabel("Percent of exhaustive evaluation", y=0.152 if extended else 0.121)
    fig.text(0.5, 0.043, "Regret below 0.05 is expanded. Shading shows ±1 subject SE. GittinsEval ends at full-run support (about 9.97–10%).",
             ha="center", fontsize=8.8, color="#4a515a")
    fig.text(0.5, 0.02, "Price weighting uses the same logistic allocation histories. Final logistic recommendations carry forward to 10%.",
             ha="center", fontsize=8.8, color="#4a515a")
    fig.subplots_adjust(left=0.073, right=0.988, bottom=0.203 if extended else 0.178, top=0.858, wspace=0.29, hspace=0.24)
    name = "gittins_vs_logistic_ablation" if extended else "gittins_vs_pretrained_logistic"
    fig.savefig(HERE / f"{name}.png", dpi=230)
    fig.savefig(HERE / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    report = runpy.run_path(str(MODEL_DIR / "plot_results.py"), run_name="_verified_model_only_report")
    comparisons = report["load_comparisons"](MODEL_DIR / "results", FEATURE_DIR / "results")
    groups = load_cohorts()
    data, rows, source_files = summarize(report, groups, comparisons)
    report["write_csv"](HERE / "plotted_curves.csv", rows)
    report["write_csv"](HERE / "checkpoints.csv", checkpoint_rows(data, groups))
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    render(data, groups, extended=False)
    render(data, groups, extended=True)

    auxiliary = (BUNDLE / "data" / "selected_tasks.csv", SUPPLEMENT / "prompteval_denominators.csv")
    source_files.update({path.relative_to(REPO).as_posix(): digest(path) for path in auxiliary})
    chosen_tasks = set(sum(groups.values(), []))
    provenance = {
        "cohorts": groups,
        "axis": "Actual cumulative observations or fixed input-token price-weighted spend divided by each subject's exhaustive denominator. The denominators match the supplied Figure 3 supplement. The published group curves use endpoint-normalized coordinates and are separate from this reaggregation.",
        "gittins_source": "GittinsEval-S and GittinsEval-G from the supplied per-subject LCB caches. Use the contiguous interval where all 20 runs contribute, interpolate each subject mean linearly only within its supported range, then average all subjects in the cohort. The group stops at the earliest end of support and at most 10%.",
        "logistic_source": "The local five-phase logistic BAI traces. Display the last recommendation at or before each budget, require all 20 seeds of all cohort subjects at the start, and hold the last recommendation after a trajectory ends. The 39-feature design includes pretrained DPR prompt coordinates and LLM ID. The 14-feature design includes LLM ID only. Both are logistic BAI adaptations.",
        "standard_errors": "One sample standard error across the equally weighted subject means in each cohort. These are unpaired visual bands.",
        "vertical_scale": "The Figure 3 helper expands regret from zero to 0.05 to 65% of the positive plot height. The upper segment is linear and all data remain in raw regret units.",
        "checkpoint_table": "The table gives values at 2%, 4%, 6%, 8%, and 9.9%. Missing entries are outside full-run support. Price-weighted logistic curves begin slightly after 2% for complete coverage.",
        "plotted_support": {
            f"{mode}/{group}/{method}": {"start_percent": float(x[0]), "end_percent": float(x[-1])}
            for (mode, group, method), (x, _, _) in data.items()
        },
        "bundle_source_sha256": source_files,
        "logistic_source_archives": {comparison.subject.name: comparison.provenance for comparison in comparisons
                                     if comparison.subject.name in chosen_tasks},
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote the three-method comparison, ablation, and plotted data to {HERE}")


if __name__ == "__main__":
    main()
