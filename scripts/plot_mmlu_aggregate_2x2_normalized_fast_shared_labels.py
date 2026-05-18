#!/usr/bin/env python3
"""Fast paper-style MMLU aggregate 2x2 normalized simple-regret figure.

Optimized version:
  - reads each task history file ONCE
  - reads only the columns needed for plotting
  - filters variants immediately after reading
  - prints task-level progress

Layout:
  columns: Small, Large
  rows:    Easy, Hard

Default:
  x-axis: normalized cumulative evaluations/cost
  y-axis: normalized simple regret = r(t) / r(first)
  band:   mean ± 2 SE
  no stopping lines

Difficulty mapping:
  high prior bucket -> Easy
  low prior bucket  -> Hard
  medium prior bucket is omitted from this 2x2 main-text aggregate.

Large LRF:
  Large panels do not plot LRF by default because the current large data do not
  include LRF runs. Small panels keep LRF by default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


STYLE_BY_KIND = {
    "gittins_data": {
        "color": "tab:orange",
        "label": "Gittins-S",
        "linewidth": 3.2,
        "zorder": 5,
    },
    "gittins_default": {
        "color": "tab:green",
        "label": "Gittins-G",
        "linewidth": 3.2,
        "zorder": 4,
    },
    "ucb": {
        "color": "tab:blue",
        "label": "UCB-E",
        "linewidth": 3.0,
        "zorder": 3,
    },
    "lrf": {
        "color": "tab:purple",
        "label": "LRF",
        "linewidth": 3.0,
        "zorder": 3,
    },
}

LINEWIDTH_MULT = 3.6
GITTINS_LINE_EXTRA_MULT = 1.0


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--small-root", type=Path, default=Path(r"outputs\wandb_downloads\mmlu_small_merged_finished_with_stopping"))
    p.add_argument("--large-root", type=Path, default=Path(r"outputs\wandb_downloads\mmlu_large_merged_finished_nostop"))
    p.add_argument("--task-metadata", type=Path, default=Path(r"data\MMLU_matrices\task_metadata.json"))
    p.add_argument("--out-dir", type=Path, default=Path(r"outputs\wandb_plots\paper_figures"))

    p.add_argument("--cost-mode", choices=["unit", "aware"], default="unit")
    p.add_argument("--small-batch-size", type=int, default=4)
    p.add_argument("--large-batch-size", type=int, default=16)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--grid-size", type=int, default=350)

    p.add_argument("--normalize-x", choices=["final", "none"], default="final")
    p.add_argument("--normalize-y", choices=["initial", "none"], default="initial")
    p.add_argument("--y-eps", type=float, default=1e-8)

    p.add_argument("--range", choices=["stderr", "std", "none"], default="stderr")
    p.add_argument("--stderr-k", type=float, default=2.0)

    p.add_argument("--include-small-lrf", action="store_true", default=True)
    p.add_argument("--no-include-small-lrf", dest="include_small_lrf", action="store_false")
    p.add_argument("--include-large-lrf", action="store_true", default=False)

    # LRF warmup crop is consistent with earlier MMLU regret-panel handling.
    p.add_argument("--crop-lrf-warmup", action="store_true", default=True)
    p.add_argument("--no-crop-lrf-warmup", dest="crop_lrf_warmup", action="store_false")
    p.add_argument("--warmup-percentage-default", type=float, default=0.05)

    # Paper-style layout, matching the GSM8K/PIQA figure family.
    p.add_argument("--fig-width", type=float, default=26.8)
    p.add_argument("--fig-height", type=float, default=25.9)
    p.add_argument("--title-size", type=float, default=81)
    p.add_argument("--label-size", type=float, default=81)
    p.add_argument("--tick-size", type=float, default=57)
    p.add_argument("--legend-size", type=float, default=63)
    p.add_argument("--row-label-size", type=float, default=81)

    # Shared labels. Do not put x/y labels inside every panel.
    # Defaults are tuned so the shared labels visually sit at the same
    # distance from the panel grid as the per-panel labels in
    # plot_gsm8k_piqa_paper_grid.py (matching the paper's main figure).
    p.add_argument("--shared-x-label-size", type=float, default=81)
    p.add_argument("--shared-y-label-size", type=float, default=81)
    p.add_argument("--shared-x-label-y", type=float, default=0.150)
    p.add_argument("--shared-y-label-x", type=float, default=-0.023)

    p.add_argument("--y-limit-min", type=float, default=None)
    p.add_argument("--y-limit-max", type=float, default=None)

    # Match GSM8K/PIQA panel geometry exactly so the four MMLU panels share
    # the same aspect ratio and spacing as the paper's main figure.
    p.add_argument("--left", type=float, default=0.045)
    p.add_argument("--right", type=float, default=0.995)
    p.add_argument("--top", type=float, default=0.765)
    p.add_argument("--bottom", type=float, default=0.205)
    p.add_argument("--wspace", type=float, default=0.24)
    p.add_argument("--hspace", type=float, default=0.34)

    p.add_argument("--legend-y", type=float, default=0.040)
    p.add_argument("--legend-ncol", type=int, default=5)
    p.add_argument("--dpi", type=int, default=260)

    return p.parse_args()


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": args.title_size,
            "axes.labelsize": args.label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_task_metadata(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("tasks", []):
        if isinstance(row, dict) and "task" in row:
            out[str(row["task"])] = row
    return out


def tasks_for_group(metadata: dict[str, dict[str, Any]], *, size_bucket: str, difficulty: str) -> list[str]:
    diff_to_prior = {"Easy": "high", "Hard": "low"}
    target_prior = diff_to_prior[difficulty]
    tasks = []
    for task, row in metadata.items():
        if str(row.get("size_bucket")) == size_bucket and str(row.get("dataset_prior_bucket")) == target_prior:
            tasks.append(task)
    return sorted(tasks)


def task_history_path(root: Path, task: str) -> Path:
    candidates = [
        root / task / "runs_history.csv.gz",
        root / safe_token(task) / "runs_history.csv.gz",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing runs_history.csv.gz for task={task} under {root}")


def variant_names(batch_size: int, scale: str, cost_mode: str) -> dict[str, str]:
    b = int(batch_size)
    return {
        "gittins_data": f"gittins_{cost_mode}_B{b}_scale{scale}_dataset",
        "gittins_default": f"gittins_{cost_mode}_B{b}_scale{scale}_default",
        "ucb": f"ucb_B{b}",
        "lrf": f"lrf_B{b}",
    }


def header_columns(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        header = f.readline().strip()
    return header.split(",")


def read_filtered_history(path: Path, wanted_variants: set[str], x_col: str) -> pd.DataFrame:
    available = set(header_columns(path))
    desired = [
        "run_id",
        "experiment_variant",
        x_col,
        "cum_eval",
        "simple_regret",
        "n_cells",
        "warmup_percentage",
        "eval_budget_fraction",
    ]
    usecols = [c for c in desired if c in available]

    needed = {"run_id", "experiment_variant", x_col, "simple_regret"}
    missing = sorted(needed - set(usecols))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        low_memory=True,
    )
    df = df[df["experiment_variant"].astype(str).isin(wanted_variants)].copy()
    if df.empty:
        return df

    for col in [x_col, "cum_eval", "simple_regret", "n_cells", "warmup_percentage", "eval_budget_fraction"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def first_finite(g: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if col not in g.columns:
        return default
    vals = pd.to_numeric(g[col], errors="coerce").dropna()
    if vals.empty:
        return default
    return float(vals.iloc[0])


def lrf_warmup_evals(g: pd.DataFrame, default_warmup: float) -> float:
    warmup = first_finite(g, "warmup_percentage", default_warmup)
    n_cells = first_finite(g, "n_cells", float("nan"))
    if math.isfinite(n_cells) and n_cells > 0:
        return float(math.ceil(float(warmup) * float(n_cells)))

    eval_budget_fraction = first_finite(g, "eval_budget_fraction", 0.1)
    if "cum_eval" in g.columns:
        max_eval = pd.to_numeric(g["cum_eval"], errors="coerce").max()
        if pd.notna(max_eval) and eval_budget_fraction > 0:
            return float(math.ceil(float(warmup) * float(max_eval) / float(eval_budget_fraction)))
    return float("nan")


def run_to_curve(
    rg: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    normalize_x: str,
    normalize_y: str,
    y_eps: float,
    grid_size: int,
) -> np.ndarray | None:
    gg = rg[[x_col, y_col]].dropna().sort_values(x_col)
    if gg.empty:
        return None

    gg = gg.groupby(x_col, as_index=False)[y_col].last()
    x = pd.to_numeric(gg[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(gg[y_col], errors="coerce").to_numpy(dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 2:
        return None

    # Normalized x: first logged point -> 0, last logged point -> 1.
    # This avoids empty slices at x=0 because W&B history starts after the first batch.
    if normalize_x == "final":
        x0 = float(x[0])
        x1 = float(x[-1])
        denom = x1 - x0
        if not np.isfinite(denom) or denom <= 0:
            return None
        x = (x - x0) / denom
    elif normalize_x == "none":
        # For raw mode, we still interpolate over each run's own x range. Not recommended for aggregate.
        pass
    else:
        raise ValueError(normalize_x)

    if normalize_y == "initial":
        denom_y = max(abs(float(y[0])), float(y_eps))
        y = y / denom_y
    elif normalize_y == "none":
        pass
    else:
        raise ValueError(normalize_y)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].last()
    x = tmp["x"].to_numpy(dtype=float)
    y = tmp["y"].to_numpy(dtype=float)
    if len(x) < 2:
        return None

    if normalize_x == "final":
        x_grid = np.linspace(0.0, 1.0, int(grid_size))
    else:
        x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), int(grid_size))

    yi = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
    return yi


def collect_curves_from_task_df(
    df: pd.DataFrame,
    *,
    method_variants: dict[str, str],
    x_col: str,
    grid_size: int,
    normalize_x: str,
    normalize_y: str,
    y_eps: float,
    crop_lrf_warmup: bool,
    warmup_default: float,
) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {kind: [] for kind in method_variants}

    if df.empty:
        return out

    for kind, variant in method_variants.items():
        sub = df[df["experiment_variant"].astype(str) == variant].copy()
        if sub.empty:
            continue

        if kind == "lrf" and crop_lrf_warmup:
            pieces = []
            for _, vg in sub.groupby("run_id", sort=False):
                warmup = lrf_warmup_evals(vg, default_warmup=warmup_default)
                if math.isfinite(warmup) and "cum_eval" in vg.columns:
                    vg = vg[pd.to_numeric(vg["cum_eval"], errors="coerce") >= warmup]
                pieces.append(vg)
            sub = pd.concat(pieces, ignore_index=False) if pieces else sub.iloc[0:0]

        for _, rg in sub.groupby("run_id", sort=False):
            yi = run_to_curve(
                rg,
                x_col=x_col,
                y_col="simple_regret",
                normalize_x=normalize_x,
                normalize_y=normalize_y,
                y_eps=y_eps,
                grid_size=grid_size,
            )
            if yi is not None:
                out[kind].append(yi)

    return out


def aggregate_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.linspace(0.0, 1.0, int(len(curves[0]))) if curves else np.array([])
    if not curves:
        return x_grid, np.array([]), np.array([]), np.array([]), np.array([])

    arr = np.vstack(curves)
    valid = np.isfinite(arr)
    n = valid.sum(axis=0)

    mean = np.full(arr.shape[1], np.nan, dtype=float)
    std = np.full(arr.shape[1], np.nan, dtype=float)
    stderr = np.full(arr.shape[1], np.nan, dtype=float)

    ok = n > 0
    if np.any(ok):
        mean[ok] = np.nanmean(arr[:, ok], axis=0)
    ok_std = n > 1
    if np.any(ok_std):
        std[ok_std] = np.nanstd(arr[:, ok_std], axis=0, ddof=1)
        stderr[ok_std] = std[ok_std] / np.sqrt(n[ok_std])
    if np.any(ok & ~ok_std):
        std[ok & ~ok_std] = 0.0
        stderr[ok & ~ok_std] = 0.0

    return x_grid, mean, std, stderr, n.astype(float)


def plot_group_panel(
    ax: plt.Axes,
    *,
    group_label: str,
    root: Path,
    tasks: list[str],
    method_variants: dict[str, str],
    x_col: str,
    args: argparse.Namespace,
    legend_handles: dict[str, Line2D],
    stats_rows: list[dict[str, Any]],
) -> None:
    group_curves: dict[str, list[np.ndarray]] = {kind: [] for kind in method_variants}
    group_task_counts: dict[str, int] = {kind: 0 for kind in method_variants}

    print(f"\n[{group_label}] reading {len(tasks)} tasks from {root}")
    t_group = time.time()

    wanted_variants = set(method_variants.values())

    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        try:
            path = task_history_path(root, task)
            df = read_filtered_history(path, wanted_variants, x_col=x_col)
        except Exception as e:
            print(f"  {i:02d}/{len(tasks):02d} {task}: SKIP ({e})")
            continue

        task_curves = collect_curves_from_task_df(
            df,
            method_variants=method_variants,
            x_col=x_col,
            grid_size=int(args.grid_size),
            normalize_x=args.normalize_x,
            normalize_y=args.normalize_y,
            y_eps=float(args.y_eps),
            crop_lrf_warmup=bool(args.crop_lrf_warmup),
            warmup_default=float(args.warmup_percentage_default),
        )

        msg_parts = []
        for kind, curves in task_curves.items():
            if curves:
                group_curves[kind].extend(curves)
                group_task_counts[kind] += 1
            msg_parts.append(f"{STYLE_BY_KIND[kind]['label']}={len(curves)}")

        print(f"  {i:02d}/{len(tasks):02d} {task}: rows={len(df):,}; " + ", ".join(msg_parts) + f"; {time.time() - t0:.1f}s")

        # Free memory aggressively between tasks.
        del df

    print(f"[{group_label}] finished in {time.time() - t_group:.1f}s")

    for kind in ["gittins_data", "gittins_default", "ucb", "lrf"]:
        if kind not in method_variants:
            continue
        curves = group_curves.get(kind, [])
        if not curves:
            print(f"WARNING: no curves for {group_label} / {STYLE_BY_KIND[kind]['label']}")
            continue

        x_grid, mean, std, stderr, n = aggregate_curves(curves)
        style = STYLE_BY_KIND[kind]
        extra = float(GITTINS_LINE_EXTRA_MULT) if kind in ("gittins_data", "gittins_default") else 1.0

        line = ax.plot(
            x_grid,
            mean,
            color=style["color"],
            linewidth=float(style["linewidth"]) * float(LINEWIDTH_MULT) * extra,
            label=style["label"],
            zorder=style["zorder"],
        )[0]
        legend_handles[kind] = line

        if args.range != "none":
            band = stderr if args.range == "stderr" else std
            if args.range == "stderr":
                band = band * float(args.stderr_k)
            ax.fill_between(
                x_grid,
                mean - band,
                mean + band,
                color=style["color"],
                alpha=0.15,
                linewidth=0,
                zorder=style["zorder"] - 0.5,
            )

        for x, m, s, se, nn in zip(x_grid, mean, std, stderr, n):
            stats_rows.append(
                {
                    "group": group_label,
                    "method_kind": kind,
                    "method": style["label"],
                    "variant": method_variants[kind],
                    "n_tasks_available": int(len(tasks)),
                    "n_tasks_used": int(group_task_counts[kind]),
                    "n_curves_total": int(len(curves)),
                    "n_curves_at_x": int(nn),
                    "x": float(x),
                    "mean": float(m) if np.isfinite(m) else np.nan,
                    "std": float(s) if np.isfinite(s) else np.nan,
                    "stderr": float(se) if np.isfinite(se) else np.nan,
                }
            )

    if args.y_limit_min is not None or args.y_limit_max is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(
            args.y_limit_min if args.y_limit_min is not None else lo,
            args.y_limit_max if args.y_limit_max is not None else hi,
        )

    ax.grid(True, alpha=0.23, linewidth=0.9)
    ax.tick_params(axis="both", labelsize=args.tick_size, width=1.2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib(args)

    metadata = load_task_metadata(args.task_metadata)

    group_defs = {
        ("Easy", "Small"): {
            "root": args.small_root,
            "size_bucket": "small",
            "batch_size": int(args.small_batch_size),
            "include_lrf": bool(args.include_small_lrf),
        },
        ("Easy", "Large"): {
            "root": args.large_root,
            "size_bucket": "large",
            "batch_size": int(args.large_batch_size),
            "include_lrf": bool(args.include_large_lrf),
        },
        ("Hard", "Small"): {
            "root": args.small_root,
            "size_bucket": "small",
            "batch_size": int(args.small_batch_size),
            "include_lrf": bool(args.include_small_lrf),
        },
        ("Hard", "Large"): {
            "root": args.large_root,
            "size_bucket": "large",
            "batch_size": int(args.large_batch_size),
            "include_lrf": bool(args.include_large_lrf),
        },
    }

    selected_tasks: dict[str, list[str]] = {}
    for (difficulty, size), cfg in group_defs.items():
        tasks = tasks_for_group(
            metadata,
            size_bucket=str(cfg["size_bucket"]),
            difficulty=difficulty,
        )
        selected_tasks[f"{difficulty}-{size}"] = tasks
        print(f"{difficulty}-{size}: {len(tasks)} tasks -> {', '.join(tasks)}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(float(args.fig_width), float(args.fig_height)),
        sharey=False,
        constrained_layout=False,
    )

    row_labels = ["Easy", "Hard"]
    col_labels = ["Small", "Large"]

    x_col = "cum_eval" if args.cost_mode == "unit" else "cum_original_cost"

    legend_handles: dict[str, Line2D] = {}
    stats_rows: list[dict[str, Any]] = []

    for r, difficulty in enumerate(row_labels):
        for c, size in enumerate(col_labels):
            cfg = group_defs[(difficulty, size)]
            batch_size = int(cfg["batch_size"])
            method_variants = variant_names(batch_size, str(args.scale), str(args.cost_mode))
            if not bool(cfg["include_lrf"]):
                method_variants.pop("lrf", None)

            group_label = f"{difficulty}-{size}"
            plot_group_panel(
                axes[r, c],
                group_label=group_label,
                root=Path(cfg["root"]),
                tasks=selected_tasks[group_label],
                method_variants=method_variants,
                x_col=x_col,
                args=args,
                legend_handles=legend_handles,
                stats_rows=stats_rows,
            )

            if r == 0:
                axes[r, c].set_title(size, fontsize=args.title_size, fontweight="normal", pad=8)

    # Shared axis labels only. Individual panel x/y labels are removed to avoid overlap.
    ylabel = "Normalized simple regret" if args.normalize_y != "none" else "Simple regret"
    xlabel = "Normalized cumulative evaluations" if args.cost_mode == "unit" else "Normalized cumulative cost"

    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.text(
        float(args.shared_y_label_x),
        0.5 * (float(args.bottom) + float(args.top)),
        ylabel,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=float(args.shared_y_label_size),
    )
    fig.text(
        0.5 * (float(args.left) + float(args.right)),
        float(args.shared_x_label_y),
        xlabel,
        ha="center",
        va="center",
        fontsize=float(args.shared_x_label_size),
    )

    # Right-side row labels.
    axes[0, 1].text(
        1.035,
        0.5,
        "Easy",
        transform=axes[0, 1].transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=args.row_label_size,
        fontweight="normal",
    )
    axes[1, 1].text(
        1.035,
        0.5,
        "Hard",
        transform=axes[1, 1].transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=args.row_label_size,
        fontweight="normal",
    )

    legend_order = ["gittins_data", "gittins_default", "ucb", "lrf"]
    final_handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    final_labels = [STYLE_BY_KIND[k]["label"] for k in legend_order if k in legend_handles]

    if args.range != "none":
        final_handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.18))
        if args.range == "stderr":
            k = float(args.stderr_k)
            k_txt = str(int(k)) if abs(k - round(k)) < 1e-9 else f"{k:g}"
            final_labels.append(f"±{k_txt} SE band")
        else:
            final_labels.append("±1 std band")

    fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        ncol=min(int(args.legend_ncol), len(final_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, float(args.legend_y)),
        fontsize=args.legend_size,
        handlelength=2.0,
        handletextpad=0.35,
        columnspacing=1.15,
        borderpad=0.55,
    )

    fig.subplots_adjust(
        left=float(args.left),
        right=float(args.right),
        top=float(args.top),
        bottom=float(args.bottom),
        wspace=float(args.wspace),
        hspace=float(args.hspace),
    )

    stem = (
        f"mmlu_aggregate_2x2_{args.cost_mode}"
        f"_Bsmall{args.small_batch_size}_Blarge{args.large_batch_size}"
        f"_scale{safe_token(str(args.scale))}"
        f"_x{args.normalize_x}_y{args.normalize_y}"
        f"_fast"
    )
    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"
    out_csv = args.out_dir / f"{stem}_aggregated.csv"
    out_meta = args.out_dir / f"{stem}_groups.json"

    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.22)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(out_csv, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "cost_mode": args.cost_mode,
                "normalize_x": args.normalize_x,
                "normalize_y": args.normalize_y,
                "small_batch_size": args.small_batch_size,
                "large_batch_size": args.large_batch_size,
                "scale": args.scale,
                "selected_tasks": selected_tasks,
                "note": "high prior bucket is plotted as Easy; low prior bucket is plotted as Hard; medium prior bucket is omitted. Large LRF is omitted by default.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
