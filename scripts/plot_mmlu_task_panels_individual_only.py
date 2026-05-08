#!/usr/bin/env python3
"""Plot fixed-size individual MMLU task simple-regret panels.

This script intentionally does ONLY individual-panel plotting.
It does not assemble grids. Use `assemble_mmlu_task_panels_grid.py` after this.

Input layout:
    DATA_ROOT/task_name/runs_history.csv.gz
    DATA_ROOT/task_name/runs_stopping.csv   optional

Default MMLU-small setting:
    batch size = 4
    scale      = 1e-4

Output layout:
    OUT_ROOT/individual/unit/<task>_unit_B4_scale1e-4.png
    OUT_ROOT/individual/aware/<task>_aware_B4_scale1e-4.png

Important for later stitching:
    Images are saved WITHOUT bbox_inches='tight'.
    If --single-width, --single-height, and --dpi are unchanged, every output PNG
    has exactly the same pixel size.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter


COLOR_UCB = "tab:blue"
COLOR_LRF = "tab:purple"
COLOR_GITTINS_S = "tab:orange"  # dataset/data prior
COLOR_GITTINS_G = "tab:green"   # default/general prior

STYLE_BY_KIND = {
    "gittins_data": {"color": COLOR_GITTINS_S, "label": "Gittins-S", "lw": 2.4, "z": 5},
    "gittins_default": {"color": COLOR_GITTINS_G, "label": "Gittins-G", "lw": 2.4, "z": 4},
    "ucb": {"color": COLOR_UCB, "label": "UCB-E", "lw": 2.1, "z": 3},
    "lrf": {"color": COLOR_LRF, "label": "LRF", "lw": 2.1, "z": 3},
}


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def normalize_task_name(s: str) -> str:
    s = str(s).strip().replace(".npy", "").replace(" ", "_")
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path(r"outputs\wandb_downloads\mmlu_small_merged_finished_with_stopping"))
    p.add_argument("--out-root", type=Path, default=Path(r"outputs\wandb_plots\mmlu_task_panels\small"))
    p.add_argument("--name-map", type=Path, default=Path(r"data\MMLU_matrices\task_display_names.json"))

    g = p.add_mutually_exclusive_group()
    g.add_argument("--task", type=str, default=None, help="One task, e.g. econometrics")
    g.add_argument("--tasks", nargs="+", default=None, help="Several tasks")
    g.add_argument("--tasks-file", type=Path, default=None, help="Text file, one task per line")

    p.add_argument("--cost-mode", choices=["both", "unit", "aware"], default="both")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--grid-size", type=int, default=320)

    # Curve uncertainty band.
    p.add_argument("--range", choices=["stderr", "std", "iqr", "none"], default="stderr")
    p.add_argument("--se-mult", type=float, default=2.0, help="Multiplier for SE bands when --range stderr or --stop-band stderr.")
    p.add_argument("--curve-alpha", type=float, default=0.15)

    # Gittins stopping visualization.
    p.add_argument("--show-stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")
    p.add_argument("--stop-band", choices=["stderr", "std", "iqr", "none"], default="stderr")
    p.add_argument("--stop-alpha", type=float, default=0.12)
    p.add_argument("--stop-line-alpha", type=float, default=0.72)
    p.add_argument("--fallback-stop-to-max-x", action="store_true", default=True)
    p.add_argument("--no-fallback-stop-to-max-x", dest="fallback_stop_to_max_x", action="store_false")

    # Fixed-size individual panel style.
    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--single-width", type=float, default=2.25)
    p.add_argument("--single-height", type=float, default=1.65)
    p.add_argument("--dpi", type=int, default=260)
    # Slightly larger text for small panels (paper readability).
    p.add_argument("--title-size", type=float, default=17.5)
    # Axis tick labels (numbers) slightly smaller than titles.
    p.add_argument("--tick-size", type=float, default=14)
    p.add_argument("--line-width-scale", type=float, default=1.0)

    # Fixed margins inside every individual PNG. Keep these unchanged for equal layout.
    p.add_argument("--single-left", type=float, default=0.20)
    p.add_argument("--single-right", type=float, default=0.97)
    p.add_argument("--single-bottom", type=float, default=0.17)
    p.add_argument("--single-top", type=float, default=0.82)

    # Small panels normally only need title + tick labels.
    p.add_argument("--show-axis-labels", action="store_true", help="Normally off; grid assembly adds shared labels later.")
    p.add_argument("--show-legend", action="store_true", help="Normally off; grid assembly adds one shared legend later.")

    # Optional fixed limits if you want all panels to share the same axis ranges.
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--x-max", type=float, default=None)
    p.add_argument("--y-min", type=float, default=None)
    p.add_argument("--y-max", type=float, default=None)
    return p.parse_args()


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.family": str(args.font_family),
        "font.serif": [str(args.font_family), "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def display_name_for_task(task: str, name_map: dict) -> str:
    return str(name_map.get(task, task.replace("_", " ")))


def read_history(task_dir: Path) -> pd.DataFrame:
    path = task_dir / "runs_history.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing history file: {path}")
    return pd.read_csv(path, compression="gzip")


def read_stopping(task_dir: Path) -> pd.DataFrame:
    path = task_dir / "runs_stopping.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def first_numeric(g: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if col not in g.columns:
        return default
    vals = pd.to_numeric(g[col], errors="coerce").dropna()
    if vals.empty:
        return default
    return float(vals.iloc[0])


def is_lrf_variant(v: str) -> bool:
    vl = str(v).lower()
    return vl.startswith("lrf") or "lrf" in vl


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


def aggregate_runs_to_grid(df: pd.DataFrame, x_col: str, y_col: str, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
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
    y_arr = np.vstack([np.interp(x_grid, x, y, left=np.nan, right=np.nan) for x, y in runs])
    return x_grid, y_arr


def band_from_yarr(y_arr: np.ndarray, mode: str, se_mult: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        band = float(se_mult) * (std / np.sqrt(np.maximum(n, 1)))
    elif mode == "std":
        band = std
    else:
        band = np.zeros_like(mean)
    return mean, mean - band, mean + band


def stop_col_for_xaxis(x_axis: str) -> str | None:
    if x_axis == "cum_eval":
        return "gittins_stop_cum_eval"
    if x_axis == "cum_original_cost":
        return "gittins_stop_cum_original_cost"
    return None


def fallback_stop_x(history: pd.DataFrame, variant: str, x_axis: str) -> tuple[float | None, float]:
    finals = []
    sub = history[history["experiment_variant"].astype(str) == str(variant)]
    for _, g in sub.groupby("run_id"):
        vals = pd.to_numeric(g.get(x_axis), errors="coerce").dropna()
        vals = vals[np.isfinite(vals)]
        if not vals.empty:
            finals.append(float(vals.max()))
    if not finals:
        return None, 0.0
    arr = np.asarray(finals, dtype=float)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def draw_stopping(ax: plt.Axes, history: pd.DataFrame, stopping: pd.DataFrame, variants: dict[str, str], x_axis: str, args: argparse.Namespace) -> None:
    if not args.show_stopping:
        return
    col = stop_col_for_xaxis(x_axis)
    if col is None:
        return

    for kind in ("gittins_data", "gittins_default"):
        variant = variants[kind]
        arr = None
        if (not stopping.empty) and col in stopping.columns and "experiment_variant" in stopping.columns:
            sub = stopping[stopping["experiment_variant"].astype(str) == str(variant)]
            vals = pd.to_numeric(sub[col], errors="coerce")
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if not vals.empty:
                arr = vals.to_numpy(dtype=float)

        if arr is None or len(arr) == 0:
            if not args.fallback_stop_to_max_x:
                continue
            mean_stop, band_value = fallback_stop_x(history, variant, x_axis)
            if mean_stop is None:
                continue
            if args.stop_band == "stderr":
                band_value = float(args.se_mult) * float(band_value)
        else:
            mean_stop = float(np.mean(arr))
            if args.stop_band == "stderr":
                band_value = float(args.se_mult) * (float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)
            elif args.stop_band == "std":
                band_value = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            elif args.stop_band == "iqr":
                band_value = None
            else:
                band_value = 0.0

        if args.stop_band != "none":
            if arr is not None and len(arr) > 0 and args.stop_band == "iqr":
                q1 = float(np.quantile(arr, 0.25))
                q3 = float(np.quantile(arr, 0.75))
                if np.isfinite(q1) and np.isfinite(q3) and q3 > q1:
                    ax.axvspan(q1, q3, color=STYLE_BY_KIND[kind]["color"], alpha=float(args.stop_alpha), linewidth=0, zorder=0.5)
            else:
                lo = float(mean_stop) - float(band_value)
                hi = float(mean_stop) + float(band_value)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.axvspan(lo, hi, color=STYLE_BY_KIND[kind]["color"], alpha=float(args.stop_alpha), linewidth=0, zorder=0.5)

        ax.axvline(
            float(mean_stop),
            color=STYLE_BY_KIND[kind]["color"],
            linestyle="--",
            alpha=float(args.stop_line_alpha),
            linewidth=1.6,
            zorder=0.8,
        )


def legend_handles_labels(args: argparse.Namespace) -> tuple[list[object], list[str]]:
    handles: list[object] = [
        Line2D([0], [0], color=STYLE_BY_KIND["gittins_data"]["color"], linewidth=STYLE_BY_KIND["gittins_data"]["lw"]),
        Line2D([0], [0], color=STYLE_BY_KIND["gittins_default"]["color"], linewidth=STYLE_BY_KIND["gittins_default"]["lw"]),
        Line2D([0], [0], color=STYLE_BY_KIND["ucb"]["color"], linewidth=STYLE_BY_KIND["ucb"]["lw"]),
        Line2D([0], [0], color=STYLE_BY_KIND["lrf"]["color"], linewidth=STYLE_BY_KIND["lrf"]["lw"]),
    ]
    labels: list[str] = [
        STYLE_BY_KIND["gittins_data"]["label"],
        STYLE_BY_KIND["gittins_default"]["label"],
        STYLE_BY_KIND["ucb"]["label"],
        STYLE_BY_KIND["lrf"]["label"],
    ]
    if args.range != "none" or (args.show_stopping and args.stop_band != "none"):
        handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.18))
        labels.append(f"±{args.se_mult:g} SE band" if args.range == "stderr" and args.stop_band == "stderr" else "Uncertainty band")
    if args.show_stopping:
        handles.append(Line2D([0], [0], color=COLOR_GITTINS_S, linestyle="--", linewidth=2.0, alpha=args.stop_line_alpha))
        labels.append("Gittins-S mean stop")
        handles.append(Line2D([0], [0], color=COLOR_GITTINS_G, linestyle="--", linewidth=2.0, alpha=args.stop_line_alpha))
        labels.append("Gittins-G mean stop")
    return handles, labels


def plot_task_panel(task: str, task_dir: Path, title: str, cost_mode: str, x_axis: str, args: argparse.Namespace, out_path: Path) -> None:
    history = crop_lrf_after_warmup(read_history(task_dir))
    stopping = read_stopping(task_dir)
    variants = variant_names(args.batch_size, args.scale, cost_mode)

    fig, ax = plt.subplots(figsize=(args.single_width, args.single_height))

    for kind in ("gittins_data", "gittins_default", "ucb", "lrf"):
        variant = variants[kind]
        vg = history[history["experiment_variant"].astype(str) == str(variant)].copy()
        if vg.empty:
            continue
        x_grid, y_arr = aggregate_runs_to_grid(vg, x_axis, "simple_regret", args.grid_size)
        if x_grid.size == 0:
            continue
        center, lo, hi = band_from_yarr(y_arr, args.range, args.se_mult)
        st = STYLE_BY_KIND[kind]
        ax.plot(x_grid, center, color=st["color"], linewidth=float(st["lw"]) * float(args.line_width_scale), zorder=float(st["z"]))
        if args.range != "none":
            ax.fill_between(x_grid, lo, hi, color=st["color"], alpha=float(args.curve_alpha), linewidth=0, zorder=float(st["z"]) - 0.6)

    draw_stopping(ax, history, stopping, variants, x_axis, args)

    ax.set_title(title, fontsize=float(args.title_size), pad=3)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=float(args.tick_size), width=0.9, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    # Keep y-axis tick labels concise across tasks.
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    if args.x_min is not None or args.x_max is not None:
        old = ax.get_xlim()
        ax.set_xlim(args.x_min if args.x_min is not None else old[0], args.x_max if args.x_max is not None else old[1])
    if args.y_min is not None or args.y_max is not None:
        old = ax.get_ylim()
        ax.set_ylim(args.y_min if args.y_min is not None else old[0], args.y_max if args.y_max is not None else old[1])

    if args.show_axis_labels:
        ax.set_ylabel("Simple regret")
        ax.set_xlabel("Cumulative evaluations" if x_axis == "cum_eval" else "Cumulative cost")
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")

    if args.show_legend:
        handles, labels = legend_handles_labels(args)
        ax.legend(handles, labels, fontsize=max(8, int(args.tick_size) - 2), frameon=False)

    # Fixed subplot position, then save WITHOUT bbox_inches='tight'.
    # This guarantees identical pixel dimensions for all output panels when figsize/dpi are fixed.
    fig.subplots_adjust(
        left=float(args.single_left),
        right=float(args.single_right),
        bottom=float(args.single_bottom),
        top=float(args.single_top),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)


def collect_tasks(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if not args.data_root.exists():
        raise SystemExit(f"data-root does not exist: {args.data_root}")
    available = []
    for d in sorted([p for p in args.data_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if (d / "runs_history.csv.gz").is_file():
            available.append((d.name, d))
    lookup = {name: path for name, path in available}

    if args.task:
        wanted = [normalize_task_name(args.task)]
    elif args.tasks:
        wanted = [normalize_task_name(t) for t in args.tasks]
    elif args.tasks_file:
        wanted = [
            normalize_task_name(x)
            for x in args.tasks_file.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.strip().startswith("#")
        ]
    else:
        wanted = [name for name, _ in available]

    missing = [t for t in wanted if t not in lookup]
    if missing:
        raise SystemExit(
            f"Tasks not found under {args.data_root}: {missing}\n"
            f"Available tasks:\n{', '.join(sorted(lookup.keys()))}"
        )
    return [(t, lookup[t]) for t in wanted]


def main() -> int:
    args = parse_args()
    setup_matplotlib(args)
    name_map = read_json(args.name_map)
    tasks = collect_tasks(args)
    args.out_root.mkdir(parents=True, exist_ok=True)

    modes = []
    if args.cost_mode in ("both", "unit"):
        modes.append(("unit", "cum_eval"))
    if args.cost_mode in ("both", "aware"):
        modes.append(("aware", "cum_original_cost"))

    for task, task_dir in tasks:
        title = display_name_for_task(task, name_map)
        for mode, x_axis in modes:
            out = args.out_root / "individual" / mode / f"{safe_token(task)}_{mode}_B{args.batch_size}_scale{safe_token(args.scale)}.png"
            plot_task_panel(task, task_dir, title, mode, x_axis, args, out)
            print(f"Wrote {mode} panel: {out}")

    print(f"Done. Individual panels plotted: {len(tasks)} tasks × {len(modes)} mode(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
