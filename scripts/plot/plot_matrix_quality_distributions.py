#!/usr/bin/env python3
"""
Plot per-arm mean accuracy distributions for MMLU (or BanditEval) matrices.

Each panel is a histogram of row means (one value per arm / model×prompt row).
Designed for paper appendix/grid use:
- Times New Roman font
- no x/y axis labels by default
- small legend interpreting vertical reference markers (corner for single plots,
  above multi-panel grid figures)
- per-panel vertical markers: empirical mean, best-arm quality, informative
  prior mean for that subject's bucket (low 0.4, medium 0.6, high 0.75;
  see docs/mmlu_prior_buckets.md). Bucket-split grids pass the prior
  explicitly so each panel draws all three markers.
- large task titles and tick labels
- Multi-task plots split into three figures by prior bucket (low / medium / high)
  from task_display_names.json; each row fits up to 4 datasets by default
- ``--bandit-eval``: GSM8K/PIQA matrices ``*_1_samples_various_models_seed*.npy``
  under ``data/BanditEval_matrices`` (configurable): prior mean 0.2 (GSM8K) /
  0.3 (PIQA) per panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

PRIOR_BUCKET_ORDER = ("low", "medium", "high")

# Informative prior means (docs/mmlu_prior_buckets.md), aligned with bucket split figures.
PRIOR_MEAN_BY_BUCKET: dict[str, float] = {
    "low": 0.4,
    "medium": 0.6,
    "high": 0.75,
}

# BanditEval (GSM8K / PIQA) 1-sample various-models splits; filenames from repo data/.
BANDITEVAL_GLOB_GSM8K = "gsm8k_1_samples_various_models_seed*.npy"
BANDITEVAL_GLOB_PIQA = "piqa_1_samples_various_models_seed*.npy"
BANDITEVAL_ALPACA_FILE = (
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_no_rounding_debias.npy"
)

BANDITEVAL_PRIOR_MEAN_GSM8K = 0.2
BANDITEVAL_PRIOR_MEAN_PIQA = 0.3
BANDITEVAL_PRIOR_MEAN_ALPACA = 0.1

# Vertical reference aesthetics: (color, linestyle).
REF_EMPIRICAL_STYLE = ("#D62728", "-")  # red solid
REF_BEST_ARM_STYLE = ("#1B9E77", ":")  # green dotted
REF_PRIOR_STYLE = ("#4A148C", "--")  # dark purple dashed


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_task_plot_metadata(
    path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Returns (display_name_by_task, prior_bucket_by_task).

    Supports legacy entries that map a task id to a plain title string,
    or dict entries with ``display_name`` and optional ``bucket`` (``low``
    | ``medium`` | ``high``).
    """
    raw = load_json(path)
    displays: dict[str, str] = {}
    buckets: dict[str, str] = {}

    for task, v in raw.items():
        if isinstance(v, str):
            displays[task] = v
        elif isinstance(v, dict):
            dn = v.get("display_name")
            displays[task] = (
                dn if isinstance(dn, str) and dn else task.replace("_", " ")
            )
            b = v.get("bucket")
            if isinstance(b, str) and b.strip():
                buckets[task] = b.strip().lower()
        else:
            displays[task] = task.replace("_", " ")

    return displays, buckets


def safe_token(s: str) -> str:
    token = re.sub(r"[^\w.-]+", "_", s.strip()).strip("_")
    return token or "plot"


def load_manifest_info(
    manifest_path: Path,
) -> tuple[list[str], dict[str, dict]]:
    manifest = load_json(manifest_path)
    tasks = list(manifest["tasks"])
    file_info = {row["task"]: row for row in manifest["files"]}
    return tasks, file_info


def resolve_tasks(
    requested: list[str] | None,
    manifest_tasks: list[str],
    display_map: dict[str, str],
) -> list[str]:
    if not requested:
        return list(manifest_tasks)

    manifest_set = set(manifest_tasks)
    resolved: list[str] = []
    for task in requested:
        if task in manifest_set:
            resolved.append(task)
            continue
        raise ValueError(
            f"Unknown task {task!r}. Expected one of the {len(manifest_tasks)} "
            f"tasks in the manifest (e.g. {manifest_tasks[0]!r})."
        )
    return resolved


def banditeval_prior_mean_for_stem(stem: str) -> float:
    lower = stem.strip().lower()
    if lower.startswith("gsm8k"):
        return BANDITEVAL_PRIOR_MEAN_GSM8K
    if lower.startswith("piqa"):
        return BANDITEVAL_PRIOR_MEAN_PIQA
    if lower.startswith("alpaca"):
        return BANDITEVAL_PRIOR_MEAN_ALPACA
    raise ValueError(
        f"BanditEval stem must start with 'gsm8k', 'piqa', or 'alpaca' (got {stem!r})"
    )


_BANDITEVAL_STEM_RE = re.compile(
    r"^(?P<prefix>gsm8k|piqa)_1_samples_various_models_seed(?P<seed>\d+)$",
    re.IGNORECASE,
)


def banditeval_display_name(stem: str) -> str:
    m = _BANDITEVAL_STEM_RE.match(stem.strip())
    if m:
        label = "GSM8K" if m.group("prefix").lower() == "gsm8k" else "PIQA"
        return f"{label}, seed {m.group('seed')}"
    if stem == Path(BANDITEVAL_ALPACA_FILE).stem:
        return "Alpaca"
    return stem.replace("_", " ")


def banditeval_row_sort_key(row: dict) -> tuple[int, int]:
    stem = row["task"].lower()
    m = _BANDITEVAL_STEM_RE.match(stem)
    if m:
        dataset_order = 0 if m.group("prefix").lower() == "gsm8k" else 1
        return (dataset_order, int(m.group("seed")))
    if stem == Path(BANDITEVAL_ALPACA_FILE).stem:
        return (2, 1)
    return (99, 0)


def collect_banditeval_quality_matrix_paths(root: Path) -> list[Path]:
    """Paths for GSM8K/PIQA plus the Alpaca no-rounding-debias matrix."""
    gsm = sorted(root.glob(BANDITEVAL_GLOB_GSM8K), key=lambda p: p.name.casefold())
    piq = sorted(root.glob(BANDITEVAL_GLOB_PIQA), key=lambda p: p.name.casefold())
    alpaca_path = root / BANDITEVAL_ALPACA_FILE
    alpaca = [alpaca_path] if alpaca_path.is_file() else []
    merged = gsm + piq + alpaca
    merged.sort(key=lambda p: banditeval_row_sort_key({"task": p.stem}))
    seen: set[str] = set()
    out: list[Path] = []
    for path in merged:
        if path.name in seen:
            continue
        seen.add(path.name)
        out.append(path)
    return out


def compute_task_stats(matrix_path: Path) -> dict:
    matrix = np.load(matrix_path)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {matrix_path}, got shape {matrix.shape}")

    arm_means = matrix.mean(axis=1)
    return {
        "n_arms": int(matrix.shape[0]),
        "n_questions": int(matrix.shape[1]),
        "empirical_mean": float(np.mean(arm_means)),
        "best_arm": float(np.max(arm_means)),
        "arm_means": arm_means,
    }


def reference_line_specs_from_row(
    row: dict,
    *,
    xlim: tuple[float, float],
    prior_mean: float | None = None,
) -> list[tuple[float, str, str]]:
    """(x position, color, linestyle) clipped to histogram x-range.

    If ``prior_mean`` is set (e.g. one value for the whole prior-bucket figure),
    it is used for the informative-prior line; otherwise the row's ``prior_bucket``
    selects 0.4 / 0.6 / 0.75.
    """
    lo, hi = xlim
    specs: list[tuple[float, str, str]] = [
        (float(row["empirical_mean"]), *REF_EMPIRICAL_STYLE),
        (float(row["best_arm"]), *REF_BEST_ARM_STYLE),
    ]
    if prior_mean is not None:
        specs.append((float(prior_mean), *REF_PRIOR_STYLE))
    else:
        pb = (row.get("prior_bucket") or "").strip().lower()
        pm = PRIOR_MEAN_BY_BUCKET.get(pb)
        if pm is not None:
            specs.append((float(pm), *REF_PRIOR_STYLE))

    return [(x, c, ls) for x, c, ls in specs if lo <= x <= hi]


def reference_legend_handles(
    *, include_prior: bool
) -> tuple[list[Line2D], list[str]]:
    legend_linewidth_scale = 1.25
    handles = [
        Line2D(
            [0],
            [0],
            color=REF_EMPIRICAL_STYLE[0],
            linestyle=REF_EMPIRICAL_STYLE[1],
            linewidth=2.8 * legend_linewidth_scale,
        ),
        Line2D(
            [0],
            [0],
            color=REF_BEST_ARM_STYLE[0],
            linestyle=REF_BEST_ARM_STYLE[1],
            linewidth=2.8 * legend_linewidth_scale,
        ),
    ]
    labels = ["Empirical mean", "Best arm"]
    if include_prior:
        handles.append(
            Line2D(
                [0],
                [0],
                color=REF_PRIOR_STYLE[0],
                linestyle=REF_PRIOR_STYLE[1],
                linewidth=3.2 * legend_linewidth_scale,
            )
        )
        labels.append("Prior mean")
    return handles, labels


def configure_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": args.tick_fontsize,
            "axes.titlesize": args.caption_fontsize,
            "xtick.labelsize": args.tick_fontsize,
            "ytick.labelsize": args.tick_fontsize,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
        }
    )


def plot_histogram_panel(
    ax: plt.Axes,
    row: dict,
    title: str,
    *,
    bins: int,
    xlim: tuple[float, float],
    caption_fontsize: int,
    tick_fontsize: int,
    show_xtick_labels: bool,
    show_ytick_labels: bool,
    prior_mean: float | None = None,
) -> None:
    arm_means = row["arm_means"]
    edges = np.linspace(xlim[0], xlim[1], bins + 1)

    ax.hist(
        arm_means,
        bins=edges,
        color="#5B8DB8",
        alpha=0.92,
        edgecolor="white",
        linewidth=0.35,
        zorder=1,
    )

    ax.set_title(title, fontsize=caption_fontsize, pad=2)
    ax.set_xlim(*xlim)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.tick_params(axis="both", labelsize=tick_fontsize, length=3.5, width=0.9)

    if not show_xtick_labels:
        ax.tick_params(axis="x", labelbottom=False)

    if not show_ytick_labels:
        ax.tick_params(axis="y", left=False, labelleft=False)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.18, linewidth=0.55, zorder=0)

    for x, color, ls in reference_line_specs_from_row(
        row, xlim=xlim, prior_mean=prior_mean
    ):
        ax.axvline(
            x,
            color=color,
            linestyle=ls,
            linewidth=2.7,
            alpha=0.95,
            zorder=6,
        )

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
    caption_fontsize: int,
    tick_fontsize: int,
    legend_fontsize: float,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pm_panel = row.get("panel_prior_mean")
    if pm_panel is None:
        pm_panel = PRIOR_MEAN_BY_BUCKET.get(
            (row.get("prior_bucket") or "").strip().lower(), None
        )

    plot_histogram_panel(
        ax,
        row,
        row["display_name"],
        bins=bins,
        xlim=xlim,
        caption_fontsize=caption_fontsize,
        tick_fontsize=tick_fontsize,
        show_xtick_labels=True,
        show_ytick_labels=True,
        prior_mean=pm_panel,
    )

    include_prior = pm_panel is not None
    leg_handles, leg_labels = reference_legend_handles(include_prior=include_prior)
    ax.legend(
        leg_handles,
        leg_labels,
        loc="upper left",
        fontsize=legend_fontsize,
        framealpha=0.92,
        handlelength=2.6,
        handletextpad=0.5,
        borderpad=0.35,
        labelspacing=0.35,
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
    caption_fontsize: int,
    tick_fontsize: int,
    legend_fontsize: float,
    show_inner_ticks: bool,
    prior_mean: float | None = None,
    grid_h_pad: float = 1.2,
    grid_w_pad: float = 0.45,
) -> None:
    n = len(rows)
    ncols = max(1, cols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width_per_col * ncols, fig_height_per_row * nrows),
        sharex=False,
        squeeze=False,
    )

    for j, row in enumerate(rows):
        r = j // ncols
        c = j % ncols
        ax = axes[r][c]

        show_xtick_labels = True
        if show_inner_ticks:
            show_ytick_labels = True
        else:
            show_ytick_labels = c == 0

        pm_panel = row.get("panel_prior_mean")
        if pm_panel is None:
            pm_panel = prior_mean

        plot_histogram_panel(
            ax,
            row,
            row["display_name"],
            bins=bins,
            xlim=xlim,
            caption_fontsize=caption_fontsize,
            tick_fontsize=tick_fontsize,
            show_xtick_labels=show_xtick_labels,
            show_ytick_labels=show_ytick_labels,
            prior_mean=pm_panel,
        )

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    include_prior_any = prior_mean is not None or any(
        (row.get("prior_bucket") or "").strip().lower() in PRIOR_MEAN_BY_BUCKET
        for row in rows
    ) or any(row.get("panel_prior_mean") is not None for row in rows)
    leg_handles, leg_labels = reference_legend_handles(
        include_prior=include_prior_any
    )
    fig.legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(leg_labels),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=2.0,
        handletextpad=0.6,
        handlelength=3.1,
    )

    fig.tight_layout(pad=0.25, w_pad=grid_w_pad, h_pad=grid_h_pad)
    fig.subplots_adjust(bottom=0.12)

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
                "prior_bucket",
                "informative_prior_mean",
                "file",
                "n_arms",
                "n_questions",
                "empirical_mean",
                "best_arm",
            ],
        )
        writer.writeheader()

        for row in rows:
            ipp = row.get("panel_prior_mean")
            ipp_s = "" if ipp is None else f"{float(ipp):.6f}"
            writer.writerow(
                {
                    "task": row["task"],
                    "display_name": row["display_name"],
                    "prior_bucket": row.get("prior_bucket", ""),
                    "informative_prior_mean": ipp_s,
                    "file": row["file"],
                    "n_arms": row["n_arms"],
                    "n_questions": row["n_questions"],
                    "empirical_mean": f"{row['empirical_mean']:.6f}",
                    "best_arm": f"{row['best_arm']:.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-arm mean distributions for MMLU or BanditEval matrices."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data") / "MMLU_matrix" / "manifest.json",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "MMLU_matrix",
    )
    p.add_argument(
        "--name-map",
        type=Path,
        default=Path("data") / "MMLU_matrix" / "task_display_names.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "matrix_quality_distributions",
    )

    p.add_argument(
        "--bandit-eval",
        action="store_true",
        help=(
            "Plot GSM8K/PIQA matrices matching "
            f"{BANDITEVAL_GLOB_GSM8K!r} and {BANDITEVAL_GLOB_PIQA!r}, plus the "
            "Alpaca no-rounding-debias matrix, under --bandit-eval-data-dir "
            "(informative prior 0.2 GSM8K, 0.3 PIQA, 0.1 Alpaca per panel). "
            "Ignores MMLU --manifest / --task / --tasks / --all."
        ),
    )
    p.add_argument(
        "--bandit-eval-data-dir",
        type=Path,
        default=Path("data") / "BanditEval_matrices",
        help="Directory searched when --bandit-eval is set.",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument("--task", type=str, help="Single MMLU task id.")
    group.add_argument("--tasks", nargs="+", help="Several MMLU task ids.")
    group.add_argument("--all", action="store_true", help="All tasks in the manifest.")

    p.add_argument("--bins", type=int, default=20)
    p.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Panels per row in each prior-bucket grid (also used for legacy single-grid mode).",
    )
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=1.0)
    p.add_argument("--dpi", type=int, default=150)

    p.add_argument("--fig-width-per-col", type=float, default=3.2)
    p.add_argument("--fig-height-per-row", type=float, default=2.45)
    p.add_argument(
        "--grid-h-pad",
        type=float,
        default=1.2,
        help="Vertical spacing between subplot rows (larger -> more row gap).",
    )
    p.add_argument(
        "--grid-w-pad",
        type=float,
        default=0.45,
        help="Horizontal spacing between subplot columns.",
    )
    p.add_argument("--single-width", type=float, default=5.5)
    p.add_argument("--single-height", type=float, default=4.0)

    p.add_argument(
        "--caption-fontsize",
        type=int,
        default=24,
        help="Font size for dataset labels (panel titles).",
    )
    p.add_argument("--tick-fontsize", type=int, default=22)
    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=20.0,
        help="Font size for the Empirical/Best/Prior legend text.",
    )
    p.add_argument(
        "--show-inner-ticks",
        action="store_true",
        help="Show y-axis tick labels on every panel (default: left column only).",
    )
    p.add_argument(
        "--save-individual",
        action="store_true",
        help="When plotting multiple tasks, also save individual single-task histograms.",
    )
    p.add_argument(
        "--legacy-single-grid",
        action="store_true",
        help="Emit one combined grid instead of splitting by prior bucket (old layout).",
    )

    return p.parse_args()


def run_bandit_eval_quality_flow(args: argparse.Namespace) -> int:
    root = args.bandit_eval_data_dir
    if not root.is_dir():
        raise FileNotFoundError(f"BanditEval data directory not found: {root}")

    paths = collect_banditeval_quality_matrix_paths(root)
    if not paths:
        raise FileNotFoundError(
            f"No matrices matching {BANDITEVAL_GLOB_GSM8K}, "
            f"{BANDITEVAL_GLOB_PIQA}, or {BANDITEVAL_ALPACA_FILE} under {root}"
        )

    rows: list[dict] = []
    for path in paths:
        stem = path.stem
        stats = compute_task_stats(path)
        rows.append(
            {
                "task": stem,
                "display_name": banditeval_display_name(stem),
                "prior_bucket": "",
                "panel_prior_mean": banditeval_prior_mean_for_stem(stem),
                "file": path.name,
                "matrix_path": str(path),
                "n_arms": stats["n_arms"],
                "n_questions": stats["n_questions"],
                "empirical_mean": stats["empirical_mean"],
                "best_arm": stats["best_arm"],
                "arm_means": stats["arm_means"],
            }
        )

    rows.sort(key=banditeval_row_sort_key)

    summary_path = (
        args.out_dir
        / "summary"
        / f"summary_banditeval_gsm8k_piqa_alpaca_{len(rows)}panels.csv"
    )
    save_summary_csv(rows, summary_path)

    grids_dir = args.out_dir / "grids_clean"
    xlim = (args.x_min, args.x_max)
    base = grids_dir / f"banditeval_quality_gsm8k_piqa_alpaca_{len(rows)}panels_cols{args.cols}"

    if len(rows) == 1:
        row = rows[0]
        single_path = args.out_dir / "singles_clean" / f"{row['task']}.png"
        save_single_plot(
            single_path,
            row,
            bins=args.bins,
            xlim=xlim,
            fig_width=args.single_width,
            fig_height=args.single_height,
            caption_fontsize=args.caption_fontsize,
            tick_fontsize=args.tick_fontsize,
            legend_fontsize=args.legend_fontsize,
        )
        print(f"Wrote single plot: {single_path}")
        grid_path = Path(f"{base}.png")
        save_grid_plot(
            grid_path,
            rows,
            cols=args.cols,
            bins=args.bins,
            xlim=xlim,
            fig_width_per_col=args.fig_width_per_col,
            fig_height_per_row=args.fig_height_per_row,
            caption_fontsize=args.caption_fontsize,
            tick_fontsize=args.tick_fontsize,
            legend_fontsize=args.legend_fontsize,
            show_inner_ticks=args.show_inner_ticks,
            prior_mean=None,
            grid_h_pad=args.grid_h_pad,
            grid_w_pad=args.grid_w_pad,
        )
        print(f"Wrote BanditEval grid: {grid_path}")
        print(f"Wrote summary CSV: {summary_path}")
        return 0

    grid_path = Path(f"{base}.png")
    save_grid_plot(
        grid_path,
        rows,
        cols=args.cols,
        bins=args.bins,
        xlim=xlim,
        fig_width_per_col=args.fig_width_per_col,
        fig_height_per_row=args.fig_height_per_row,
        caption_fontsize=args.caption_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        show_inner_ticks=args.show_inner_ticks,
        prior_mean=None,
        grid_h_pad=args.grid_h_pad,
        grid_w_pad=args.grid_w_pad,
    )
    print(f"Wrote BanditEval grid: {grid_path}")
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
                caption_fontsize=args.caption_fontsize,
                tick_fontsize=args.tick_fontsize,
                legend_fontsize=args.legend_fontsize,
            )
        print(f"Wrote {len(rows)} individual plots to: {single_dir}")

    return 0


def main() -> int:
    args = parse_args()
    configure_matplotlib(args)

    if args.bandit_eval:
        return run_bandit_eval_quality_flow(args)

    manifest_tasks, file_info = load_manifest_info(args.manifest)

    if args.name_map.is_file():
        display_map, prior_bucket_map = load_task_plot_metadata(args.name_map)
    else:
        display_map, prior_bucket_map = {}, {}

    if args.task:
        requested = [args.task]
    elif args.tasks:
        requested = list(args.tasks)
    elif args.all:
        requested = list(manifest_tasks)
    else:
        requested = None

    selected_tasks = resolve_tasks(requested, manifest_tasks, display_map)
    selected_tasks = sorted(
        selected_tasks,
        key=lambda task: display_map.get(task, task.replace("_", " ")).casefold(),
    )

    rows: list[dict] = []
    for task in selected_tasks:
        info = file_info[task]
        matrix_path = args.data_dir / info["file"]
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Matrix not found for task {task!r}: {matrix_path}")

        stats = compute_task_stats(matrix_path)
        rows.append(
            {
                "task": task,
                "display_name": display_map.get(task, task.replace("_", " ")),
                "prior_bucket": prior_bucket_map.get(task, ""),
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
    bucket_split_eligible = (
        not args.legacy_single_grid
        and prior_bucket_map
        and all(r.get("prior_bucket") for r in rows)
    )

    grids_dir = args.out_dir / "grids_clean"

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
            caption_fontsize=args.caption_fontsize,
            tick_fontsize=args.tick_fontsize,
            legend_fontsize=args.legend_fontsize,
        )
        print(f"Wrote single plot: {out_path}")
        if bucket_split_eligible:
            pb = row["prior_bucket"].strip().lower()
            grid_path = (
                grids_dir
                / f"mmlu_quality_distributions_prior_{pb}_1tasks_cols{args.cols}.png"
            )
            save_grid_plot(
                grid_path,
                rows,
                cols=args.cols,
                bins=args.bins,
                xlim=xlim,
                fig_width_per_col=args.fig_width_per_col,
                fig_height_per_row=args.fig_height_per_row,
                caption_fontsize=args.caption_fontsize,
                tick_fontsize=args.tick_fontsize,
                legend_fontsize=args.legend_fontsize,
                show_inner_ticks=args.show_inner_ticks,
                prior_mean=PRIOR_MEAN_BY_BUCKET[pb],
                grid_h_pad=args.grid_h_pad,
                grid_w_pad=args.grid_w_pad,
            )
            print(f"Wrote prior-bucket grid: {grid_path}")
        print(f"Wrote summary CSV: {summary_path}")
        return 0

    if bucket_split_eligible:
        for b in PRIOR_BUCKET_ORDER:
            grp = [r for r in rows if r["prior_bucket"] == b]
            if not grp:
                continue
            grid_path = (
                grids_dir
                / f"mmlu_quality_distributions_prior_{b}_{len(grp)}tasks_cols{args.cols}.png"
            )
            save_grid_plot(
                grid_path,
                grp,
                cols=args.cols,
                bins=args.bins,
                xlim=xlim,
                fig_width_per_col=args.fig_width_per_col,
                fig_height_per_row=args.fig_height_per_row,
                caption_fontsize=args.caption_fontsize,
                tick_fontsize=args.tick_fontsize,
                legend_fontsize=args.legend_fontsize,
                show_inner_ticks=args.show_inner_ticks,
                prior_mean=PRIOR_MEAN_BY_BUCKET[b],
                grid_h_pad=args.grid_h_pad,
                grid_w_pad=args.grid_w_pad,
            )
            print(f"Wrote prior-bucket grid: {grid_path}")
        print(f"Wrote summary CSV: {summary_path}")
    else:
        if len(rows) > 1 and not args.legacy_single_grid:
            missing = [
                r["task"] for r in rows if prior_bucket_map and not r.get("prior_bucket")
            ]
            if missing:
                print(
                    "Note: Some tasks lack `bucket` in --name-map; "
                    "using one combined grid instead of three prior-bucket figures. "
                    f"Examples: {', '.join(missing[:6])}"
                    + (" ..." if len(missing) > 6 else "")
                )
        grid_path = (
            grids_dir / f"mmlu_quality_distributions_{len(rows)}tasks_cols{args.cols}.png"
        )
        save_grid_plot(
            grid_path,
            rows,
            cols=args.cols,
            bins=args.bins,
            xlim=xlim,
            fig_width_per_col=args.fig_width_per_col,
            fig_height_per_row=args.fig_height_per_row,
            caption_fontsize=args.caption_fontsize,
            tick_fontsize=args.tick_fontsize,
            legend_fontsize=args.legend_fontsize,
            show_inner_ticks=args.show_inner_ticks,
            grid_h_pad=args.grid_h_pad,
            grid_w_pad=args.grid_w_pad,
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
                caption_fontsize=args.caption_fontsize,
                tick_fontsize=args.tick_fontsize,
                legend_fontsize=args.legend_fontsize,
            )
        print(f"Wrote {len(rows)} individual plots to: {single_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
