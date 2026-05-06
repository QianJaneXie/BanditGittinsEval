#!/usr/bin/env python3
"""Plot MMLU matrix arm-quality distributions.

Each MMLU matrix has shape:
    (n_arms, n_questions)

For each task, this script computes:
    arm_quality = mean accuracy over questions for each arm

Then it plots histogram distributions.

Designed for paper appendix/grid use:
- Times New Roman font
- no x/y axis labels by default
- no legend
- no vertical reference lines
- large task titles and tick labels
- can plot one task or all 57 tasks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def configure_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times New Roman PS MT",
        "Times",
        "DejaVu Serif",
    ]
    plt.rcParams["axes.titlesize"] = args.title_fontsize
    plt.rcParams["xtick.labelsize"] = args.tick_fontsize
    plt.rcParams["ytick.labelsize"] = args.tick_fontsize
    plt.rcParams["figure.dpi"] = args.dpi
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["axes.unicode_minus"] = False


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def normalize_token(text: str) -> str:
    s = text.strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_manifest_info(manifest_path: Path) -> tuple[list[str], dict[str, dict]]:
    manifest = load_json(manifest_path)
    tasks = manifest["tasks"]
    file_info = {row["task"]: row for row in manifest["files"]}
    return tasks, file_info


def build_task_lookup(tasks: list[str], display_map: dict[str, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for task in tasks:
        display = display_map.get(task, task)
        aliases = {
            task,
            task.replace("_", " "),
            display,
            Path(task).stem,
        }
        for alias in aliases:
            lookup[normalize_token(alias)] = task
    return lookup


def resolve_tasks(
    requested: list[str] | None,
    manifest_tasks: list[str],
    display_map: dict[str, str],
) -> list[str]:
    if not requested:
        return list(manifest_tasks)

    lookup = build_task_lookup(manifest_tasks, display_map)
    resolved = []
    seen = set()

    for item in requested:
        key = normalize_token(item)
        if key not in lookup:
            raise ValueError(
                f"Unknown task: {item}\n"
                f"Please use a canonical task name from manifest.json, "
                f"for example: us_foreign_policy, computer_security, high_school_computer_science."
            )
        task = lookup[key]
        if task not in seen:
            resolved.append(task)
            seen.add(task)

    return resolved


def compute_task_stats(matrix_path: Path) -> dict:
    arr = np.load(matrix_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {matrix_path}, got shape {arr.shape}")

    arm_means = arr.mean(axis=1)

    return {
        "n_arms": int(arr.shape[0]),
        "n_questions": int(arr.shape[1]),
        "empirical_mean": float(arm_means.mean()),
        "best_arm": float(arm_means.max()),
        "arm_means": arm_means,
    }


def plot_histogram_panel(
    ax: plt.Axes,
    arm_means: np.ndarray,
    title: str,
    *,
    bins: int,
    xlim: tuple[float, float],
    title_fontsize: int,
    tick_fontsize: int,
    show_xtick_labels: bool,
    show_ytick_labels: bool,
) -> None:
    edges = np.linspace(xlim[0], xlim[1], bins + 1)

    ax.hist(
        arm_means,
        bins=edges,
        color="#5B8DB8",
        alpha=0.92,
        edgecolor="white",
        linewidth=0.35,
    )

    ax.set_title(title, fontsize=title_fontsize, pad=6)
    ax.set_xlim(*xlim)

    # Keep only a few clean tick labels.
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    ax.tick_params(axis="both", labelsize=tick_fontsize, length=3.5, width=0.9)

    if not show_xtick_labels:
        ax.tick_params(axis="x", labelbottom=False)

    if not show_ytick_labels:
        ax.tick_params(axis="y", labelleft=False)

    # No axis labels, no legend, no vertical lines.
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Light grid, enough to read distribution but not too distracting.
    ax.grid(True, alpha=0.18, linewidth=0.55)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def save_single_plot(
    out_path: Path,
    row: dict,
    *,
    bins: int,
    xlim: tuple[float, float],
    fig_width: float,
    fig_height: float,
    title_fontsize: int,
    tick_fontsize: int,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plot_histogram_panel(
        ax,
        row["arm_means"],
        row["display_name"],
        bins=bins,
        xlim=xlim,
        title_fontsize=title_fontsize,
        tick_fontsize=tick_fontsize,
        show_xtick_labels=True,
        show_ytick_labels=True,
    )

    fig.tight_layout(pad=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_grid_plot(
    out_path: Path,
    rows: list[dict],
    *,
    cols: int,
    bins: int,
    xlim: tuple[float, float],
    fig_width_per_col: float,
    fig_height_per_row: float,
    title_fontsize: int,
    tick_fontsize: int,
    show_inner_ticks: bool,
) -> None:
    n = len(rows)
    ncols = min(cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width_per_col * ncols, fig_height_per_row * nrows),
        sharex=True,
        squeeze=False,
    )

    for i, row in enumerate(rows):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        if show_inner_ticks:
            show_xtick_labels = True
            show_ytick_labels = True
        else:
            show_xtick_labels = r == nrows - 1
            show_ytick_labels = c == 0

        plot_histogram_panel(
            ax,
            row["arm_means"],
            row["display_name"],
            bins=bins,
            xlim=xlim,
            title_fontsize=title_fontsize,
            tick_fontsize=tick_fontsize,
            show_xtick_labels=show_xtick_labels,
            show_ytick_labels=show_ytick_labels,
        )

    # Remove unused axes.
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    # No global x/y labels here; teacher will integrate them later.
    fig.tight_layout(pad=0.25, w_pad=0.35, h_pad=0.60)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "display_name",
                "file",
                "n_arms",
                "n_questions",
                "empirical_mean",
                "best_arm",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "task": row["task"],
                    "display_name": row["display_name"],
                    "file": row["file"],
                    "n_arms": row["n_arms"],
                    "n_questions": row["n_questions"],
                    "empirical_mean": f"{row['empirical_mean']:.6f}",
                    "best_arm": f"{row['best_arm']:.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data") / "MMLU_matrices" / "manifest.json",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "MMLU_matrices",
    )
    p.add_argument(
        "--name-map",
        type=Path,
        default=Path("data") / "MMLU_matrices" / "task_display_names.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "matrix_quality_distributions",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument("--task", type=str, default=None, help="Plot one task")
    group.add_argument("--tasks", nargs="+", default=None, help="Plot selected tasks")
    group.add_argument("--all", action="store_true", help="Plot all tasks in manifest")

    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=1.0)

    # Bigger default fonts for your current request.
    p.add_argument("--title-fontsize", type=int, default=24)
    p.add_argument("--tick-fontsize", type=int, default=22)

    # Single plot size.
    p.add_argument("--single-width", type=float, default=7.2)
    p.add_argument("--single-height", type=float, default=4.8)

    # Grid panel size. Increase these if titles/ticks look crowded.
    p.add_argument("--fig-width-per-col", type=float, default=3.2)
    p.add_argument("--fig-height-per-row", type=float, default=2.45)

    p.add_argument("--dpi", type=int, default=260)

    # By default, dense grids only show x ticks on bottom row and y ticks on left column.
    p.add_argument(
        "--show-inner-ticks",
        action="store_true",
        help="Show tick labels on every subplot. Usually not recommended for dense grids.",
    )

    p.add_argument(
        "--save-individual",
        action="store_true",
        help="When plotting multiple tasks, also save individual single-task histograms.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib(args)

    manifest_tasks, file_info = load_manifest_info(args.manifest)

    if args.name_map.is_file():
        display_map = load_json(args.name_map)
    else:
        display_map = {}

    if args.task is not None:
        requested = [args.task]
    elif args.tasks is not None:
        requested = args.tasks
    else:
        requested = None

    selected_tasks = resolve_tasks(requested, manifest_tasks, display_map)

    rows = []
    for task in selected_tasks:
        info = file_info[task]
        matrix_path = args.data_dir / info["file"]

        if not matrix_path.is_file():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        stats = compute_task_stats(matrix_path)

        rows.append(
            {
                "task": task,
                "display_name": display_map.get(task, task.replace("_", " ")),
                "file": info["file"],
                "matrix_path": str(matrix_path),
                "n_arms": stats["n_arms"],
                "n_questions": stats["n_questions"],
                "empirical_mean": stats["empirical_mean"],
                "best_arm": stats["best_arm"],
                "arm_means": stats["arm_means"],
            }
        )

    summary_path = args.out_dir / "summary" / f"summary_{len(rows)}tasks.csv"
    save_summary_csv(rows, summary_path)

    xlim = (args.x_min, args.x_max)

    if len(rows) == 1:
        row = rows[0]
        out_path = args.out_dir / "singles_clean" / f"{row['task']}.png"
        save_single_plot(
            out_path,
            row,
            bins=args.bins,
            xlim=xlim,
            fig_width=args.single_width,
            fig_height=args.single_height,
            title_fontsize=args.title_fontsize,
            tick_fontsize=args.tick_fontsize,
        )
        print(f"Wrote single plot: {out_path}")
        print(f"Wrote summary CSV: {summary_path}")
        return 0

    grid_path = args.out_dir / "grids_clean" / f"mmlu_quality_distributions_{len(rows)}tasks_cols{args.cols}.png"
    save_grid_plot(
        grid_path,
        rows,
        cols=args.cols,
        bins=args.bins,
        xlim=xlim,
        fig_width_per_col=args.fig_width_per_col,
        fig_height_per_row=args.fig_height_per_row,
        title_fontsize=args.title_fontsize,
        tick_fontsize=args.tick_fontsize,
        show_inner_ticks=args.show_inner_ticks,
    )

    print(f"Wrote grid plot: {grid_path}")
    print(f"Wrote summary CSV: {summary_path}")

    if args.save_individual:
        single_dir = args.out_dir / "singles_clean"
        for row in rows:
            out_path = single_dir / f"{row['task']}.png"
            save_single_plot(
                out_path,
                row,
                bins=args.bins,
                xlim=xlim,
                fig_width=args.single_width,
                fig_height=args.single_height,
                title_fontsize=args.title_fontsize,
                tick_fontsize=args.tick_fontsize,
            )
        print(f"Wrote {len(rows)} individual plots to: {single_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())