"""Recreate the MMLU interaction comparison figures from their bundled CSVs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


MODES = (("unit", "Unit cost"), ("aware", "Price-weighted cost"))
GROUP_COUNTS = {"Easy-Small": 6, "Easy-Large": 2, "Hard-Small": 9, "Hard-Large": 2}
METHODS = {
    "gittins_s": ("GittinsEval-S", "#d76814", "-"),
    "gittins_g": ("GittinsEval-G", "#228443", "-"),
    "logistic_39": ("Logistic BAI (39)", "#545d69", "-"),
    "rasch39_balanced": ("Concatenated (39), balanced", "#1e5f9c", "-"),
    "rasch389_balanced": ("Interactions (389), balanced", "#a22c67", "-"),
    "rasch39_random": ("Concatenated (39), random", "#1e5f9c", "--"),
    "rasch389_random": ("Interactions (389), random", "#a22c67", "--"),
}
PRIMARY = ("gittins_s", "gittins_g", "rasch39_balanced", "rasch389_balanced")
SAMPLING = (*PRIMARY, "rasch39_random", "rasch389_random")


def load_groups(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {group: [] for group in GROUP_COUNTS}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group, task = row["group"], row["task"]
            if group not in groups or not task:
                raise ValueError(f"Invalid cohort row in {path}: {row}")
            groups[group].append(task)
    tasks = [task for members in groups.values() for task in members]
    if {group: len(members) for group, members in groups.items()} != GROUP_COUNTS or len(set(tasks)) != 19:
        raise ValueError(f"Expected the 19 distinct Figure 3 subjects and four cohort sizes in {path}.")
    return groups


def load_curves(path: Path, groups: dict[str, list[str]]) -> dict[tuple[str, str, str], tuple[np.ndarray, ...]]:
    expected = {(mode, group, method) for mode, _ in MODES for group in groups for method in METHODS}
    points = defaultdict(list)
    fields = ("x_percent_exhaustive", "mean_simple_regret", "subject_se")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["mode"], row["group"], row["method"]
            if key not in expected or int(row["subjects"]) != len(groups[key[1]]) or int(row["seeds_per_subject"]) != 20:
                raise ValueError(f"Unexpected curve or coverage in {path}: {key}")
            points[key].append(tuple(float(row[field]) for field in fields))
    if set(points) != expected:
        raise ValueError(f"Missing curves in {path}: {sorted(expected - set(points))}")

    curves = {}
    for key, rows in points.items():
        x, mean, se = np.array(rows, dtype=float).T
        if (
            len(x) < 2 or not np.isfinite([x, mean, se]).all() or np.any(np.diff(x) <= 0)
            or x[0] < 0 or x[-1] > 10 or np.any(mean < 0) or np.any(se < 0)
        ):
            raise ValueError(f"Invalid plotted coordinates in {path}: {key}")
        curves[key] = (x, mean, se)
    return curves


def draw_figure(out_dir, curves, groups, expand_lower_regret, *, sampling: bool) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.2), sharex=True, sharey="col")
    methods = SAMPLING if sampling else PRIMARY
    for row, (mode, mode_label) in enumerate(MODES):
        for column, (group, subjects) in enumerate(groups.items()):
            ax = axes[row, column]
            for method in methods:
                label, color, linestyle = METHODS[method]
                x, mean, se = curves[mode, group, method]
                step = method.startswith("rasch")
                subdued = sampling and not step
                ax.plot(
                    x, mean, label=label, color=color, linestyle=linestyle,
                    lw=1.6 if subdued else 2.1, alpha=0.76 if subdued else 1.0,
                    drawstyle="steps-post" if step else "default",
                )
                ax.fill_between(
                    x, mean - se, mean + se, color=color, alpha=0.08 if subdued else 0.12,
                    step="post" if step else None, linewidth=0,
                )
                if step:
                    ax.plot(x[-1:], mean[-1:], marker="o", color=color, markersize=3)
            ax.set(xlim=(0, 10.12), xticks=(0, 5, 10))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.grid(axis="y", linewidth=0.65, color="#c8cdd4", alpha=0.52)
            if row == 0:
                ax.set_title(f"{group.replace('-', '–')} ({len(subjects)})", fontweight="semibold", pad=8)
            if column == 0:
                ax.set_ylabel(f"{mode_label}\nSimple regret")
    for column, group in enumerate(groups):
        ymax = max(float(np.max(curves[mode, group, method][1] + curves[mode, group, method][2]))
                   for mode, _ in MODES for method in methods)
        for ax in axes[:, column]:
            ax.set_ylim(0, 1.04 * ymax)
            expand_lower_regret(ax, threshold=0.05, fraction=0.65)

    title = "PE-EmbPT design and question-sampling comparison" if sampling else "GittinsEval and PE-EmbPT interaction designs on MMLU"
    fig.suptitle(title, fontsize=17, fontweight="semibold", y=0.978)
    fig.text(0.5, 0.925, "Actual budgets  ·  19 Figure 3 subjects  ·  20 seeds per subject  ·  Equal subject weights", ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.066), ncol=3 if sampling else 4,
               frameon=False, fontsize=9.1)
    fig.supxlabel("Percent of exhaustive evaluation", y=0.142 if sampling else 0.118)
    fig.text(0.5, 0.045, "Shading shows ±1 subject SE. Scale marks expand regret below 0.05. GittinsEval ends at full-run support.",
             ha="center", fontsize=8.8, color="#4a515a")
    fig.text(0.5, 0.022, "The price-weighted view reuses the PE-EmbPT allocations. Final PE-EmbPT recommendations carry forward to 10%.",
             ha="center", fontsize=8.8, color="#4a515a")
    fig.subplots_adjust(left=0.073, right=0.987, bottom=0.196 if sampling else 0.176, top=0.866, wspace=0.30, hspace=0.24)
    name = "Figure3_PE_EmbPT_interaction_sampling" if sampling else "Figure3_PE_EmbPT_interaction"
    fig.savefig(out_dir / f"{name}.png", dpi=230)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def replot(bundle_dir: Path, out_dir: Path) -> None:
    groups = load_groups(bundle_dir / "data" / "selected_tasks.csv")
    curves = load_curves(bundle_dir / "data" / "plotted_curves.csv", groups)
    helper = runpy.run_path(str(bundle_dir / "code" / "piecewise_regret_axis.py"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for sampling in (False, True):
        draw_figure(out_dir, curves, groups, helper["expand_lower_regret"], sampling=sampling)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_bundle = here if (here / "data" / "plotted_curves.csv").is_file() else here.parents[1] / "PE_EmbPT_interaction_data_2026-11-07"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=default_bundle)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.bundle_dir / "reproduced"
    replot(args.bundle_dir, out_dir)
    print(f"Wrote both figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
