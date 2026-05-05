#!/usr/bin/env python3
"""Paper-style grid plots for MMLU small (grid only, no bottom overlap).

This script generates only the combined multi-panel figures:
- 1 unit-cost grid using x=cum_eval
- 1 cost-aware grid using x=cum_original_cost

Main layout choices in this version:
- shared x label for the full figure
- shared y label for the full figure
- larger bottom margin so the shared x label and bottom legend do not overlap
- legend moved lower
- no repeated x-axis labels on the bottom row panels
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLOR_UCB = "tab:blue"
COLOR_LRF = "tab:purple"
COLOR_GITTINS_DEFAULT = "tab:green"
COLOR_GITTINS_DATA = "tab:orange"

STYLE_BY_KIND = {
    "gittins_data": {"color": COLOR_GITTINS_DATA, "label": "Gittins (data prior)", "lw": 2.4, "z": 5},
    "gittins_default": {"color": COLOR_GITTINS_DEFAULT, "label": "Gittins (default prior)", "lw": 2.4, "z": 4},
    "ucb": {"color": COLOR_UCB, "label": "UCB-E", "lw": 2.1, "z": 3},
    "lrf": {"color": COLOR_LRF, "label": "LRF", "lw": 2.1, "z": 3},
}

STOP_BAND_PROXY = Patch(facecolor="0.75", edgecolor="none", alpha=0.18, label="Stop IQR")
STOP_LINE_PROXY = Line2D([0], [0], color="0.35", linestyle="--", linewidth=2.0, label="Mean stop")


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root containing the merged MMLU small task folders. "
            r"Default: outputs\wandb_downloads\mmlu_small_merged_finished_with_stopping"
        ),
    )
    p.add_argument("--selected-root", type=Path, default=None, help="Legacy root containing selected task folders.")
    p.add_argument("--remaining-root", type=Path, default=None, help="Legacy root containing remaining task folders.")
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(r"outputs\wandb_plots\mmlu_small_22_tasks_paper"),
        help="Output root for grid figures.",
    )

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--grid-size", type=int, default=320)
    p.add_argument("--range", choices=["stderr", "std", "iqr", "none"], default="stderr")

    p.add_argument("--show-stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")
    p.add_argument("--stop-band", choices=["iqr", "none"], default="iqr")
    p.add_argument("--stop-alpha", type=float, default=0.16)
    p.add_argument("--stop-line-alpha", type=float, default=0.70)

    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--font-size", type=float, default=22)
    p.add_argument("--title-size", type=float, default=22)
    p.add_argument("--tick-size", type=float, default=20)
    p.add_argument("--legend-size", type=float, default=20)
    p.add_argument("--shared-label-size", type=float, default=22)

    p.add_argument("--grid-cols", type=int, default=6)
    p.add_argument("--grid-fig-width", type=float, default=18.0)
    p.add_argument("--grid-fig-height", type=float, default=10.8)
    p.add_argument("--grid-wspace", type=float, default=0.34)
    p.add_argument("--grid-hspace", type=float, default=0.62)
    p.add_argument("--legend-columnspacing", type=float, default=2.0)
    p.add_argument("--legend-handlelength", type=float, default=3.0)

    # Layout controls chosen to avoid overlap.
    p.add_argument("--grid-left", type=float, default=0.09)
    p.add_argument("--grid-right", type=float, default=0.985)
    p.add_argument("--grid-top", type=float, default=0.94)
    p.add_argument("--grid-bottom", type=float, default=0.27)
    p.add_argument("--shared-y-label-x", type=float, default=0.02)
    p.add_argument("--shared-x-label-y", type=float, default=0.16)
    p.add_argument("--legend-y", type=float, default=0.03)
    p.add_argument("--legend-ncol", type=int, default=4)

    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--cost-mode", choices=["both", "unit", "aware"], default="both")
    p.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task names for a smaller grid, e.g. anatomy,college_biology.",
    )

    args = p.parse_args()
    if args.data_root is None and args.selected_root is None and args.remaining_root is None:
        args.data_root = Path(r"outputs\wandb_downloads\mmlu_small_merged_finished_with_stopping")
    return args


def read_history(task_dir: Path) -> pd.DataFrame:
    path = task_dir / "runs_history.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, compression="gzip")


def read_stopping(task_dir: Path) -> pd.DataFrame:
    path = task_dir / "runs_stopping.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def is_lrf_variant(v: str) -> bool:
    vl = str(v).lower()
    return vl.startswith("lrf") or "lrf" in vl


def first_numeric(g: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if col not in g.columns:
        return default
    vals = pd.to_numeric(g[col], errors="coerce").dropna()
    if vals.empty:
        return default
    return float(vals.iloc[0])


def lrf_warmup_evals(g: pd.DataFrame, default_warmup: float = 0.05, default_budget_fraction: float = 0.10) -> float:
    warmup = first_numeric(g, "warmup_percentage", default_warmup)
    n_cells = first_numeric(g, "n_cells", float("nan"))
    if not math.isfinite(n_cells) or n_cells <= 0:
        eval_budget_fraction = first_numeric(g, "eval_budget_fraction", default_budget_fraction)
        max_eval = pd.to_numeric(g.get("cum_eval"), errors="coerce").max()
        if pd.notna(max_eval) and eval_budget_fraction > 0:
            n_cells = float(max_eval) / float(eval_budget_fraction)
    if not math.isfinite(n_cells) or n_cells <= 0:
        return float("nan")
    return float(math.ceil(float(warmup) * float(n_cells)))


def crop_lrf_after_warmup(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for variant, g in df.groupby("experiment_variant", sort=False):
        if is_lrf_variant(str(variant)):
            warmup = lrf_warmup_evals(g)
            if math.isfinite(warmup) and "cum_eval" in g.columns:
                g = g[pd.to_numeric(g["cum_eval"], errors="coerce") >= warmup]
        pieces.append(g)
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=False)


def variant_names(batch_size: int, scale: str, cost_mode: str) -> dict[str, str]:
    b = int(batch_size)
    if cost_mode == "unit":
        return {
            "gittins_data": f"gittins_unit_B{b}_scale{scale}_dataset",
            "gittins_default": f"gittins_unit_B{b}_scale{scale}_default",
            "ucb": f"ucb_B{b}",
            "lrf": f"lrf_B{b}",
        }
    if cost_mode == "aware":
        return {
            "gittins_data": f"gittins_aware_B{b}_scale{scale}_dataset",
            "gittins_default": f"gittins_aware_B{b}_scale{scale}_default",
            "ucb": f"ucb_B{b}",
            "lrf": f"lrf_B{b}",
        }
    raise ValueError(cost_mode)


def _aggregate_runs_to_grid(df: pd.DataFrame, x_col: str, y_col: str, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    runs = []
    min_x = float("inf")
    max_x = -float("inf")
    for _, g in df.groupby("run_id"):
        gg = g[[x_col, y_col]].dropna().sort_values(x_col)
        if gg.empty:
            continue
        gg = gg.groupby(x_col, as_index=False)[y_col].last()
        x = gg[x_col].to_numpy(dtype=float)
        y = gg[y_col].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if len(x) < 2:
            continue
        runs.append((x, y))
        min_x = min(min_x, float(np.nanmin(x)))
        max_x = max(max_x, float(np.nanmax(x)))
    if not runs or not np.isfinite(min_x) or not np.isfinite(max_x) or max_x <= min_x:
        return np.array([]), np.empty((0, 0))

    x_grid = np.linspace(min_x, max_x, int(grid_size))
    ys = []
    for x, y in runs:
        yi = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
        ys.append(yi)
    return x_grid, np.vstack(ys)


def _band_from_yarr(y_arr: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y_arr.size == 0:
        empty = np.array([])
        return empty, empty, empty
    if mode == "iqr":
        center = np.nanmedian(y_arr, axis=0)
        lo = np.nanquantile(y_arr, 0.25, axis=0)
        hi = np.nanquantile(y_arr, 0.75, axis=0)
        return center, lo, hi
    mean = np.nanmean(y_arr, axis=0)
    std = np.nanstd(y_arr, axis=0)
    n = np.sum(~np.isnan(y_arr), axis=0)
    if mode == "stderr":
        band = std / np.sqrt(np.maximum(n, 1))
    elif mode == "std":
        band = std
    else:
        band = np.zeros_like(mean)
    return mean, mean - band, mean + band


def _stop_col_for_xaxis(x_axis: str) -> str | None:
    if x_axis == "cum_eval":
        return "gittins_stop_cum_eval"
    if x_axis == "cum_original_cost":
        return "gittins_stop_cum_original_cost"
    return None


def draw_stopping(ax: plt.Axes, stopping: pd.DataFrame, variants: dict[str, str], x_axis: str, args: argparse.Namespace) -> None:
    if (not args.show_stopping) or stopping.empty:
        return
    col = _stop_col_for_xaxis(x_axis)
    if col is None or col not in stopping.columns:
        return

    for kind in ("gittins_data", "gittins_default"):
        v = variants.get(kind)
        if not v:
            continue
        sub = stopping[stopping["experiment_variant"].astype(str) == str(v)]
        vals = pd.to_numeric(sub[col], errors="coerce")
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        if vals.empty:
            continue
        arr = vals.to_numpy(dtype=float)
        mean = float(np.mean(arr))
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        color = STYLE_BY_KIND[kind]["color"]

        if args.stop_band == "iqr" and q3 > q1:
            ax.axvspan(q1, q3, color=color, alpha=float(args.stop_alpha), linewidth=0, zorder=0.5)
        ax.axvline(mean, color=color, linestyle="--", alpha=float(args.stop_line_alpha), linewidth=2.0, zorder=0.8)


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": str(args.font_family),
            "font.serif": [str(args.font_family)],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": float(args.font_size),
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prettify_task_name(task: str) -> str:
    return task.replace("_", " ")


def _collect_tasks(args: argparse.Namespace) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    roots: list[Path] = []
    if args.data_root is not None:
        roots.append(args.data_root)
    if args.selected_root is not None:
        roots.append(args.selected_root)
    if args.remaining_root is not None:
        roots.append(args.remaining_root)

    print("Reading task folders from:")
    for root in roots:
        print(f"  - {root}")
        if not root.exists():
            print(f"    WARNING: root does not exist, skipped: {root}", flush=True)
            continue
        for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            if (d / "runs_history.csv.gz").is_file():
                tasks.append((d.name, d))

    seen = set()
    out = []
    for name, path in tasks:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, path))
    return out


def plot_grid(*, tasks: list[tuple[str, Path]], cost_mode: str, x_axis: str, out_path: Path, args: argparse.Namespace) -> None:
    n = len(tasks)
    ncols = max(1, int(args.grid_cols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(args.grid_fig_width, args.grid_fig_height), squeeze=False)

    variants_cache = variant_names(args.batch_size, args.scale, cost_mode)
    order = ["gittins_data", "gittins_default", "ucb", "lrf"]

    for idx, (task, task_dir) in enumerate(tasks):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        history = crop_lrf_after_warmup(read_history(task_dir))
        stopping = read_stopping(task_dir)

        for kind in order:
            v = variants_cache[kind]
            vg = history[history["experiment_variant"].astype(str) == str(v)].copy()
            if vg.empty:
                continue
            x_grid, y_arr = _aggregate_runs_to_grid(vg, x_axis, "simple_regret", args.grid_size)
            if x_grid.size == 0:
                continue
            center, lo, hi = _band_from_yarr(y_arr, args.range)
            st = STYLE_BY_KIND[kind]
            ax.plot(x_grid, center, color=st["color"], linewidth=float(st["lw"]), zorder=float(st["z"]))
            if args.range != "none":
                ax.fill_between(x_grid, lo, hi, color=st["color"], alpha=0.15, linewidth=0, zorder=float(st["z"]) - 0.6)

        draw_stopping(ax, stopping, variants_cache, x_axis, args)

        ax.set_title(prettify_task_name(task), fontsize=args.title_size, pad=5)
        ax.grid(True, alpha=0.18, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=args.tick_size, width=1.0, length=4)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    method_handles = [
        Line2D([0], [0], color=STYLE_BY_KIND["gittins_data"]["color"], linewidth=STYLE_BY_KIND["gittins_data"]["lw"]),
        Line2D([0], [0], color=STYLE_BY_KIND["gittins_default"]["color"], linewidth=STYLE_BY_KIND["gittins_default"]["lw"]),
        Line2D([0], [0], color=STYLE_BY_KIND["ucb"]["color"], linewidth=STYLE_BY_KIND["ucb"]["lw"]),
        Line2D([0], [0], color=STYLE_BY_KIND["lrf"]["color"], linewidth=STYLE_BY_KIND["lrf"]["lw"]),
    ]
    method_labels = [
        STYLE_BY_KIND["gittins_data"]["label"],
        STYLE_BY_KIND["gittins_default"]["label"],
        STYLE_BY_KIND["ucb"]["label"],
        STYLE_BY_KIND["lrf"]["label"],
    ]

    stop_items: list[tuple[object, str]] = []
    if args.show_stopping and args.stop_band == "iqr":
        stop_items.append((STOP_BAND_PROXY, "Stop IQR"))
    if args.show_stopping:
        stop_items.append((STOP_LINE_PROXY, "Mean stop"))

    final_handles = []
    final_labels = []
    final_handles.append(method_handles[0]); final_labels.append(method_labels[0])
    if len(stop_items) >= 1:
        final_handles.append(stop_items[0][0]); final_labels.append(stop_items[0][1])
    final_handles.append(method_handles[1]); final_labels.append(method_labels[1])
    if len(stop_items) >= 2:
        final_handles.append(stop_items[1][0]); final_labels.append(stop_items[1][1])
    final_handles.append(method_handles[2]); final_labels.append(method_labels[2])
    final_handles.append(method_handles[3]); final_labels.append(method_labels[3])

    fig.subplots_adjust(
        left=float(args.grid_left),
        right=float(args.grid_right),
        top=float(args.grid_top),
        bottom=float(args.grid_bottom),
        wspace=float(args.grid_wspace),
        hspace=float(args.grid_hspace),
    )

    fig.text(
        float(args.shared_y_label_x),
        0.5,
        "Simple regret",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=float(args.shared_label_size),
    )

    fig.text(
        0.5,
        float(args.shared_x_label_y),
        "Cumulative evaluations" if x_axis == "cum_eval" else "Cumulative cost",
        ha="center",
        va="center",
        fontsize=float(args.shared_label_size),
    )

    fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        ncol=int(args.legend_ncol),
        frameon=False,
        bbox_to_anchor=(0.5, float(args.legend_y)),
        fontsize=args.legend_size,
        handlelength=float(args.legend_handlelength),
        columnspacing=float(args.legend_columnspacing),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_matplotlib(args)
    args.out_root.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(args)
    if args.tasks:
        wanted = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
        task_lookup = {name: path for name, path in tasks}
        missing = [t for t in wanted if t not in task_lookup]
        if missing:
            available = ", ".join(name for name, _ in tasks)
            raise SystemExit(f"Tasks not found: {missing}\nAvailable tasks: {available}")
        tasks = [(t, task_lookup[t]) for t in wanted]

    if not tasks:
        raise SystemExit("No task folders with runs_history.csv.gz were found.")
    if len(tasks) != 22 and not args.tasks:
        print(f"WARNING: expected 22 tasks, found {len(tasks)}.")

    if args.cost_mode in ("both", "unit"):
        out_grid_unit = args.out_root / "grids" / f"mmlu_small_{len(tasks)}_unit_B{args.batch_size}_scale{args.scale}.png"
        plot_grid(tasks=tasks, cost_mode="unit", x_axis="cum_eval", out_path=out_grid_unit, args=args)
        print(f"Wrote unit grid: {out_grid_unit}")

    if args.cost_mode in ("both", "aware"):
        out_grid_cost = args.out_root / "grids" / f"mmlu_small_{len(tasks)}_aware_B{args.batch_size}_scale{args.scale}.png"
        plot_grid(tasks=tasks, cost_mode="aware", x_axis="cum_original_cost", out_path=out_grid_cost, args=args)
        print(f"Wrote cost-aware grid: {out_grid_cost}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())