"""Recreate both bundled figures from the plotted-point and cohort CSVs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from piecewise_regret_axis import expand_lower_regret


MODES = (("unit", "Unit cost"), ("aware", "Price-weighted cost"))
GROUP_COUNTS = {"Easy-Small": 6, "Easy-Large": 2, "Hard-Small": 9, "Hard-Large": 2}
PRIMARY = ("gittins_s", "gittins_g", "rasch39_balanced", "logistic_39")
BAI_METHODS = ("logistic_39", "rasch39_random", "rasch39_balanced")
METHODS = {
    "gittins_s": ("GittinsEval-S", "#e57716", "-"),
    "gittins_g": ("GittinsEval-G", "#278745", "-"),
    "rasch39_balanced": ("PE-EmbPT (joint task, 39 arm features + question effects)", "#a92f63", "-"),
    "logistic_39": ("Logistic BAI (39 features)", "#194c82", "-"),
    "rasch39_random": ("PE-EmbPT (random questions)", "#74539a", "--"),
}


def load_groups(path: Path) -> dict[str, list[str]]:
    groups = {group: [] for group in GROUP_COUNTS}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group, task = row["group"], row["task"]
            if group not in groups or not task:
                raise ValueError(f"Invalid group or subject in {path}: {row}")
            groups[group].append(task)
    tasks = [task for group in groups.values() for task in group]
    if {group: len(members) for group, members in groups.items()} != GROUP_COUNTS or len(set(tasks)) != 19:
        raise ValueError(f"Expected the 19 distinct Figure 3 subjects and four cohort sizes in {path}.")
    return groups


def load_curves(path: Path, groups: dict[str, list[str]]) -> dict[tuple[str, str, str], tuple[np.ndarray, ...]]:
    expected = {(mode, group, method) for mode, _ in MODES for group in groups for method in METHODS}
    points = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["mode"], row["group"], row["method"]
            if key not in expected or int(row["subjects"]) != len(groups[key[1]]) or int(row["seeds_per_subject"]) != 20:
                raise ValueError(f"Unexpected curve or coverage in {path}: {key}")
            points[key].append(tuple(float(row[field]) for field in ("x_percent_exhaustive", "mean_simple_regret", "subject_se")))
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


def render(out_dir, curves, groups, *, sensitivity: bool):
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.1), sharex=True, sharey="col")
    methods = (*PRIMARY, "rasch39_random") if sensitivity else PRIMARY
    for row, (mode, label) in enumerate(MODES):
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
                   for mode, _ in MODES for method in methods)
        for ax in axes[:, column]:
            ax.set_ylim(0, 1.04 * ymax)
            expand_lower_regret(ax, threshold=0.05, fraction=0.65)

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
    name = "Figure3_PE_EmbPT_sampling" if sensitivity else "Figure3_PE_EmbPT"
    fig.savefig(out_dir / f"{name}.png", dpi=230)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    groups = load_groups(ROOT / "data" / "selected_tasks.csv")
    curves = load_curves(ROOT / "data" / "plotted_curves.csv", groups)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    render(args.out_dir, curves, groups, sensitivity=False)
    render(args.out_dir, curves, groups, sensitivity=True)
    print(f"Wrote both figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
