#!/usr/bin/env python3
"""Paper-style 2x2 figure for GSM8K and PIQA.

Figure layout:
  columns: GSM8K, PIQA
  rows:    unit-cost, cost-aware

No-default version:
- keeps only one Gittins curve: Gittins-S
- curve band = mean ± k·standard error (default k=2)
- stopping-location band = mean stop ± k·standard error (default k=2)

Legend:
- Gittins-S
- UCB-E
- LRF
- ±k SE band
- Gittins-S mean stop
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLOR_UCB = "tab:blue"
COLOR_LRF = "tab:purple"
COLOR_GITTINS_S = "tab:orange"
COLOR_GITTINS_G = "tab:green"  # kept only for consistency; not plotted in this no-default version.

STYLE_BY_KIND = {
    "gittins_data": {
        "color": COLOR_GITTINS_S,
        "label": "Gittins-S",
        "linewidth": 3.2,
        "zorder": 5,
    },
    "gittins_default": {
        "color": COLOR_GITTINS_G,
        "label": "Gittins-G",
        "linewidth": 3.2,
        "zorder": 4,
    },
    "ucb": {
        "color": COLOR_UCB,
        "label": "UCB-E",
        "linewidth": 3.0,
        "zorder": 3,
    },
    "lrf": {
        "color": COLOR_LRF,
        "label": "LRF",
        "linewidth": 3.0,
        "zorder": 3,
    },
}

# Thicken curve/stop lines for readability.
LINEWIDTH_MULT = 3.6
GITTINS_LINE_EXTRA_MULT = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--gsm8k-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads\gsm8k_406hah4y"),
        help="Folder containing GSM8K runs_history.csv.gz, runs_summary.csv, runs_stopping.csv.",
    )
    p.add_argument(
        "--piqa-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads\piqa_hzpntjfe"),
        help="Folder containing PIQA runs_history.csv.gz, runs_summary.csv, runs_stopping.csv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"outputs\wandb_plots\paper_figures"),
    )

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--grid-size", type=int, default=350)

    # Curve uncertainty band.
    p.add_argument("--range", choices=["stderr", "std", "none"], default="stderr")
    p.add_argument(
        "--stderr-k",
        type=float,
        default=2.0,
        help="Multiplier for standard-error bands when using stderr-based ranges/stop bands.",
    )

    p.add_argument("--show-stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")

    # Stopping-location uncertainty band.
    p.add_argument(
        "--stop-band",
        choices=["stderr", "std", "iqr", "none"],
        default="stderr",
        help="Uncertainty band for stopping x-location across seeds.",
    )
    p.add_argument("--stop-alpha", type=float, default=0.12)
    p.add_argument("--stop-line-alpha", type=float, default=0.72)

    # Make panels larger on canvas (scale up proportionally).
    p.add_argument("--fig-width", type=float, default=23.8)
    p.add_argument("--fig-height", type=float, default=23.9)

    # Larger paper-style fonts.
    # Increase paper fonts for readability (except tick labels).
    p.add_argument("--title-size", type=float, default=81)
    p.add_argument("--label-size", type=float, default=81)
    # Axis tick labels (numbers) should NOT be enlarged further.
    p.add_argument("--tick-size", type=float, default=57)
    p.add_argument("--legend-size", type=float, default=63)
    # Keep row labels (Unit-cost / Cost-aware) same size as panel titles.
    p.add_argument("--row-label-size", type=float, default=81)

    p.add_argument("--y-limit-min", type=float, default=None)
    p.add_argument("--y-limit-max", type=float, default=None)

    # Layout controls for quick tuning without editing code.
    p.add_argument("--left", type=float, default=0.070)
    p.add_argument("--right", type=float, default=0.965)
    p.add_argument("--top", type=float, default=0.765)
    p.add_argument("--bottom", type=float, default=0.245)
    p.add_argument("--wspace", type=float, default=0.24)
    p.add_argument("--hspace", type=float, default=0.46)
    p.add_argument("--legend-y", type=float, default=-0.005)

    return p.parse_args()


def read_history(folder: Path) -> pd.DataFrame:
    path = folder / "runs_history.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    return pd.read_csv(path, compression="gzip")


def read_stopping(folder: Path) -> pd.DataFrame:
    path = folder / "runs_stopping.csv"
    if not path.exists():
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
    if cost_mode in {"aware", "cost"}:
        return {
            "gittins_data": f"gittins_cost_B{b}_scale{scale}_dataset",
            "gittins_default": f"gittins_cost_B{b}_scale{scale}_default",
            "ucb": f"ucb_cost_B{b}",
            "lrf": f"lrf_cost_B{b}",
        }
    raise ValueError(cost_mode)


def as_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=str)
    return df[col].astype(str)


def filter_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    dataset = dataset.lower()
    mask = (as_str_series(df, "dataset_tag_resolved").str.lower() == dataset) | (
        as_str_series(df, "dataset_tag").str.lower() == dataset
    )
    return df[mask].copy()


def first_numeric(g: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if col not in g.columns:
        return default
    vals = pd.to_numeric(g[col], errors="coerce").dropna()
    if vals.empty:
        return default
    return float(vals.iloc[0])


def is_lrf_variant(variant: str) -> bool:
    v = str(variant).lower()
    return v.startswith("lrf") or "lrf" in v


def lrf_warmup_evals(
    g: pd.DataFrame,
    default_warmup: float = 0.05,
    default_budget_fraction: float = 0.10,
) -> float:
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


def aggregate_variant(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runs: list[tuple[np.ndarray, np.ndarray]] = []
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
        empty = np.array([])
        return empty, empty, empty, empty, empty

    x_grid = np.linspace(min_x, max_x, grid_size)

    ys = []
    for x, y in runs:
        yi = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
        ys.append(yi)

    y_arr = np.vstack(ys)
    mean = np.nanmean(y_arr, axis=0)
    std = np.nanstd(y_arr, axis=0)
    n = np.sum(~np.isnan(y_arr), axis=0)
    stderr = std / np.sqrt(np.maximum(n, 1))

    return x_grid, mean, std, stderr, n


def stopping_col_for_xaxis(x_axis: str) -> str | None:
    if x_axis == "cum_eval":
        return "gittins_stop_cum_eval"
    if x_axis == "cum_original_cost":
        return "gittins_stop_cum_original_cost"
    return None


def draw_stopping(
    ax: plt.Axes,
    stop_df: pd.DataFrame,
    variants: dict[str, str],
    x_axis: str,
    *,
    kinds_to_draw: list[str],
    show_stopping: bool,
    stop_band: str,
    stop_alpha: float,
    stop_line_alpha: float,
    stderr_k: float,
) -> None:
    if not show_stopping or stop_df.empty:
        return

    stop_col = stopping_col_for_xaxis(x_axis)
    if stop_col is None or stop_col not in stop_df.columns:
        return

    for kind in kinds_to_draw:
        band_color = STYLE_BY_KIND[kind]["color"]
        variant = variants[kind]
        sdf = stop_df[stop_df["experiment_variant"].astype(str) == variant].copy()
        if sdf.empty:
            continue

        vals = pd.to_numeric(sdf[stop_col], errors="coerce")
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        if vals.empty:
            continue

        arr = vals.to_numpy(dtype=float)
        mean_stop = float(np.mean(arr))

        if stop_band == "stderr":
            band = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            band *= float(stderr_k)
            lo = mean_stop - band
            hi = mean_stop + band
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.axvspan(lo, hi, color=band_color, alpha=stop_alpha, linewidth=0, zorder=1)

        elif stop_band == "std":
            band = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            lo = mean_stop - band
            hi = mean_stop + band
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.axvspan(lo, hi, color=band_color, alpha=stop_alpha, linewidth=0, zorder=1)

        elif stop_band == "iqr":
            lo = float(np.quantile(arr, 0.25))
            hi = float(np.quantile(arr, 0.75))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.axvspan(lo, hi, color=band_color, alpha=stop_alpha, linewidth=0, zorder=1)

        if np.isfinite(mean_stop):
            extra = float(GITTINS_LINE_EXTRA_MULT) if kind == "gittins_data" else 1.0
            ax.axvline(
                mean_stop,
                color=STYLE_BY_KIND[kind]["color"],
                linestyle="--",
                linewidth=2.6 * float(LINEWIDTH_MULT) * extra,
                alpha=stop_line_alpha,
                zorder=2,
            )


def prepare_panel_df(history: pd.DataFrame, dataset: str, variants: dict[str, str]) -> pd.DataFrame:
    df = filter_dataset(history, dataset)
    want = set(variants.values())
    df = df[df["experiment_variant"].astype(str).isin(want)].copy()

    for col in ["cum_eval", "cum_original_cost", "simple_regret"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["experiment_variant", "run_id", "simple_regret"])
    df = crop_lrf_after_warmup(df)
    return df


def plot_panel(
    ax: plt.Axes,
    history: pd.DataFrame,
    stopping: pd.DataFrame,
    *,
    dataset: str,
    dataset_title: str,
    cost_mode: str,
    x_axis: str,
    args: argparse.Namespace,
) -> dict[str, plt.Line2D]:
    variants = variant_names(args.batch_size, args.scale, cost_mode)
    df = prepare_panel_df(history, dataset, variants)

    handles: dict[str, plt.Line2D] = {}
    order = ["gittins_data", "ucb", "lrf"]

    for kind in order:
        variant = variants[kind]
        vg = df[df["experiment_variant"].astype(str) == variant].copy()

        if vg.empty:
            print(f"WARNING: missing variant for {dataset_title} {cost_mode}: {variant}")
            continue

        x_grid, mean, std, stderr, n = aggregate_variant(vg, x_axis, "simple_regret", args.grid_size)
        if x_grid.size == 0:
            print(f"WARNING: no plottable points for {dataset_title} {cost_mode}: {variant}")
            continue

        style = STYLE_BY_KIND[kind]
        extra = float(GITTINS_LINE_EXTRA_MULT) if kind == "gittins_data" else 1.0
        line = ax.plot(
            x_grid,
            mean,
            color=style["color"],
            linewidth=float(style["linewidth"]) * float(LINEWIDTH_MULT) * extra,
            label=style["label"],
            zorder=style["zorder"],
        )[0]
        handles[kind] = line

        if args.range != "none":
            band = stderr if args.range == "stderr" else std
            if args.range == "stderr":
                band *= float(args.stderr_k)
            ax.fill_between(
                x_grid,
                mean - band,
                mean + band,
                color=style["color"],
                alpha=0.15,
                linewidth=0,
                zorder=style["zorder"] - 0.5,
            )

    draw_stopping(
        ax,
        stopping,
        variants,
        x_axis,
        kinds_to_draw=["gittins_data"],
        show_stopping=args.show_stopping,
        stop_band=args.stop_band,
        stop_alpha=args.stop_alpha,
        stop_line_alpha=args.stop_line_alpha,
        stderr_k=args.stderr_k,
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

    return handles


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
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


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib(args)

    datasets = [
        {"key": "gsm8k", "title": "GSM8K", "dir": args.gsm8k_dir},
        {"key": "piqa", "title": "PIQA", "dir": args.piqa_dir},
    ]

    histories = {}
    stoppings = {}
    for d in datasets:
        histories[d["key"]] = read_history(d["dir"])
        stoppings[d["key"]] = read_stopping(d["dir"])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(args.fig_width, args.fig_height),
        sharey=False,
        constrained_layout=False,
    )

    legend_handles: dict[str, plt.Line2D] = {}

    for col, d in enumerate(datasets):
        handles = plot_panel(
            axes[0, col],
            histories[d["key"]],
            stoppings[d["key"]],
            dataset=d["key"],
            dataset_title=d["title"],
            cost_mode="unit",
            x_axis="cum_eval",
            args=args,
        )
        legend_handles.update(handles)

        handles = plot_panel(
            axes[1, col],
            histories[d["key"]],
            stoppings[d["key"]],
            dataset=d["key"],
            dataset_title=d["title"],
            cost_mode="aware",
            x_axis="cum_original_cost",
            args=args,
        )
        legend_handles.update(handles)

        # Keep some separation from axes while preventing title cropping.
        axes[0, col].set_title(d["title"], fontsize=args.title_size, fontweight="normal", pad=4)

    axes[0, 0].set_ylabel("Simple regret", fontsize=args.label_size)
    axes[1, 0].set_ylabel("Simple regret", fontsize=args.label_size)

    for col in range(2):
        axes[0, col].set_xlabel("Cumulative evaluations", fontsize=args.label_size)
        axes[1, col].set_xlabel("Cumulative cost", fontsize=args.label_size)

    axes[0, 1].text(
        1.055,
        0.5,
        "Unit-cost",
        transform=axes[0, 1].transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=args.row_label_size,
        fontweight="normal",
    )
    axes[1, 1].text(
        1.055,
        0.5,
        "Cost-aware",
        transform=axes[1, 1].transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=args.row_label_size,
        fontweight="normal",
    )

    legend_order = ["gittins_data", "ucb", "lrf"]
    final_handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    final_labels = [STYLE_BY_KIND[k]["label"] for k in legend_order if k in legend_handles]

    if args.show_stopping:
        final_handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_GITTINS_S,
                linestyle="--",
                linewidth=2.6 * float(LINEWIDTH_MULT) * float(GITTINS_LINE_EXTRA_MULT),
                alpha=args.stop_line_alpha,
            )
        )
        final_labels.append("Gittins-S mean stop")

    # One neutral patch for all semi-transparent standard-error bands.
    if args.range != "none" or (args.show_stopping and args.stop_band != "none"):
        final_handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.18))
        k = float(args.stderr_k)
        if args.range == "stderr" or args.stop_band == "stderr":
            k_txt = str(int(k)) if abs(k - round(k)) < 1e-9 else f"{k:g}"
            final_labels.append(f"±{k_txt} SE band")
        else:
            final_labels.append("±1 std band")

    fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, args.legend_y),
        fontsize=args.legend_size,
        handlelength=2.0,
        handletextpad=0.35,
        columnspacing=1.15,
        borderpad=0.55,
    )

    fig.subplots_adjust(
        left=args.left,
        right=args.right,
        top=args.top,
        bottom=0.245 if args.bottom == 0.305 else args.bottom,
        wspace=args.wspace,
        hspace=args.hspace,
    )

    stem = f"figure1_gsm8k_piqa_no_default_B{args.batch_size}_scale{args.scale}".replace(".", "")
    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"

    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
