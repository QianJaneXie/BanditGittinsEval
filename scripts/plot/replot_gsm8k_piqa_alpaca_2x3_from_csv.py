#!/usr/bin/env python3
"""Restyle the paper 2x3 figure from its saved aggregate CSV files."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

import plot_gsm8k_piqa_paper_grid as pg


READABLE_STYLE = {
    "figure": {
        "width": 10.8,
        "height": 4.15,
        "png_dpi": 360,
    },
    "font": {
        "title": 20.0,
        "axis_label": 20.0,
        "tick": 17.0,
        "legend": 15.5,
        "row_label": 19.0,
    },
    "axes": {
        "grid_alpha": 0.14,
        "grid_linewidth": 0.55,
        "spine_linewidth": 0.9,
        "tick_width": 0.9,
        "tick_length": 4.8,
        # Leave a small gap so 0% is not flush against the y-axis spine.
        "x_pad_pct": 0.25,
        "x_right": 10.5,
        "y_pad_frac": 0.015,
    },
    "uncertainty": {
        "primary_alpha": 0.10,
        "baseline_alpha": 0.065,
        "stop_band_alpha": 0.07,
        "stop_line_alpha": 0.70,
        # Mean-stop dashes use each method's solid linewidth (see METHOD_STYLE).
        "stop_linewidth": None,
    },
    "legend": {
        "handlelength": 2.8,
        "handletextpad": 0.55,
        "columnspacing": 1.15,
        "labelspacing": 0.55,
    },
}

# Exact exhaustive-evaluation denominators from the experiment run configs.
# Unit-cost is K*L; cost-aware is L*sum(per-arm original evaluation cost).
FULL_EVALUATION_COST = {
    "gsm8k": {"unit": 122000.0, "aware": 83900.0},
    "piqa": {"unit": 103000.0, "aware": 23000.0},
    "alpaca": {"unit": 123165.0, "aware": 5505234.0},
}

# Keep all wording alternatives here while terminology is still under review.
TEXT_PRESETS = {
    "default": {
        "row_labels": ("Unit-cost", "Cost-aware"),
        "shared_y_label": "Simple Regret",
        "shared_x_label": "Cumulative Evaluation Cost",
    },
    "percentage_preview": {
        "row_labels": ("Unit-cost", "Cost-Aware"),
        "shared_y_label": "Simple Regret",
        "shared_x_label": "Exhaustive Evaluation Cost (%)",
    },
}

# Line style, alpha, and z-order remain redundant with color, without markers.
METHOD_STYLE = {
    "gittins_data": {
        "linewidth": 2.55,
        "linestyle": "-",
        "alpha": 0.98,
        "zorder": 7,
    },
    "gittins_default": {
        "linewidth": 2.45,
        "linestyle": "-",
        "alpha": 0.96,
        "zorder": 6,
    },
    "sysrs": {
        "linewidth": 2.25,
        "linestyle": "-",
        "alpha": 0.94,
        "zorder": 5,
    },
    "ucb": {
        "linewidth": 2.15,
        "linestyle": "-",
        "alpha": 0.90,
        "zorder": 4,
    },
    "lrf": {
        "linewidth": 2.1,
        "linestyle": "-",
        "alpha": 0.88,
        "zorder": 3.8,
    },
    "bo_pbgi_unit": {
        "linewidth": 2.1,
        "linestyle": "-",
        "alpha": 0.88,
        "zorder": 3.4,
    },
    "bo_pbgi_cost": {
        "linewidth": 2.1,
        "linestyle": "-",
        "alpha": 0.88,
        "zorder": 3.4,
    },
    "bo_logei_unit": {
        "linewidth": 2.05,
        "linestyle": "-",
        "alpha": 0.86,
        "zorder": 3.2,
    },
    "bo_logeipc_cost": {
        "linewidth": 2.05,
        "linestyle": "-",
        "alpha": 0.86,
        "zorder": 3.2,
    },
}

LEGEND_ORDER = [
    "gittins_data",
    "gittins_default",
    "sysrs",
    "ucb",
    "lrf",
    "bo_pbgi_unit",
    "bo_logei_unit",
]

# Linear legend order: Gittins → BO(+stops) → UCB/LRF → SySRs → SE.
LEGEND_ENTRIES = [
    ("method", "gittins_data"),
    ("method", "gittins_default"),
    ("stop", "gittins_data"),
    ("stop", "gittins_default"),
    ("method", "bo_pbgi_unit"),
    ("method", "bo_logei_unit"),
    ("stop", "bo_pbgi_unit"),
    ("stop", "bo_logei_unit"),
    ("method", "ucb"),
    ("method", "lrf"),
    ("method", "sysrs"),
    ("band", None),
]

STOP_LABELS = {
    "gittins_data": "Gittins-S mean stop",
    "gittins_default": "Gittins-G mean stop",
    "bo_pbgi_unit": "BO-PBGI mean stop",
    "bo_logei_unit": "BO-LogEI(PC) mean stop",
}


def parse_args() -> argparse.Namespace:
    default_dir = Path(
        "outputs/figure/new_figure/final/"
        "gsm8k_piqa_alpaca_2x3_latest_Gs_sysrs"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=default_dir)
    p.add_argument("--out-dir", type=Path, default=default_dir)
    p.add_argument(
        "--stem",
        default="figure_gsm8k_piqa_alpaca_2x3_B8_LRFB32_scale1e-4_sysrs",
    )
    p.add_argument("--color-gittins-s", default="tab:orange")
    p.add_argument("--color-gittins-g", default="tab:green")
    p.add_argument("--color-sysrs", default="tab:pink")
    p.add_argument("--color-ucb", default="tab:blue")
    p.add_argument("--color-lrf", default="tab:purple")
    p.add_argument("--color-bo-pbgi", default="tab:olive")
    p.add_argument("--color-bo-logei", default="tab:brown")
    p.add_argument("--no-range", action="store_true")
    p.add_argument("--no-show-stopping", action="store_true")
    p.add_argument(
        "--stop-band", choices=["stderr", "std", "iqr", "none"], default="stderr"
    )
    p.add_argument("--stderr-k", type=float, default=1.0)
    p.add_argument(
        "--stop-alpha",
        type=float,
        default=READABLE_STYLE["uncertainty"]["stop_band_alpha"],
    )
    p.add_argument(
        "--stop-line-alpha",
        type=float,
        default=READABLE_STYLE["uncertainty"]["stop_line_alpha"],
    )
    p.add_argument("--fig-width", type=float, default=READABLE_STYLE["figure"]["width"])
    p.add_argument("--fig-height", type=float, default=READABLE_STYLE["figure"]["height"])
    p.add_argument("--title-size", type=float, default=READABLE_STYLE["font"]["title"])
    p.add_argument("--label-size", type=float, default=READABLE_STYLE["font"]["axis_label"])
    p.add_argument("--tick-size", type=float, default=READABLE_STYLE["font"]["tick"])
    p.add_argument("--legend-size", type=float, default=READABLE_STYLE["font"]["legend"])
    p.add_argument("--row-label-size", type=float, default=READABLE_STYLE["font"]["row_label"])
    p.add_argument("--png-dpi", type=int, default=READABLE_STYLE["figure"]["png_dpi"])
    return p.parse_args()


def set_colors(args: argparse.Namespace) -> None:
    colors = {
        "gittins_data": args.color_gittins_s,
        "gittins_default": args.color_gittins_g,
        "sysrs": args.color_sysrs,
        "ucb": args.color_ucb,
        "lrf": args.color_lrf,
        "bo_pbgi_unit": args.color_bo_pbgi,
        "bo_pbgi_cost": args.color_bo_pbgi,
        "bo_logei_unit": args.color_bo_logei,
        "bo_logeipc_cost": args.color_bo_logei,
    }
    for kind, color in colors.items():
        pg.STYLE_BY_KIND[kind]["color"] = color
    pg.COLOR_GITTINS_S = args.color_gittins_s
    pg.COLOR_GITTINS_G = args.color_gittins_g


def as_full_cost_percentage(
    values: float | pd.Series | np.ndarray,
    dataset: str,
    cost_mode: str,
) -> float | np.ndarray:
    denominator = FULL_EVALUATION_COST[dataset][cost_mode]
    converted = np.asarray(values, dtype=float) / denominator * 100.0
    return float(converted) if converted.ndim == 0 else converted


def draw_stops(
    ax: plt.Axes,
    stops: pd.DataFrame,
    args: argparse.Namespace,
    dataset: str,
    cost_mode: str,
) -> None:
    if args.no_show_stopping or stops.empty:
        return
    for row in stops.itertuples(index=False):
        kind = str(row.kind)
        color = pg.STYLE_BY_KIND[kind]["color"]
        mean_raw = float(row.mean_stop)
        if args.stop_band == "stderr":
            delta = float(row.stderr_stop) * args.stderr_k
            lo_raw, hi_raw = mean_raw - delta, mean_raw + delta
        elif args.stop_band == "std":
            lo_raw = mean_raw - float(row.std_stop)
            hi_raw = mean_raw + float(row.std_stop)
        elif args.stop_band == "iqr":
            lo_raw, hi_raw = float(row.q25_stop), float(row.q75_stop)
        else:
            lo_raw = hi_raw = mean_raw
        mean = as_full_cost_percentage(mean_raw, dataset, cost_mode)
        lo = as_full_cost_percentage(lo_raw, dataset, cost_mode)
        hi = as_full_cost_percentage(hi_raw, dataset, cost_mode)
        if hi > lo:
            ax.axvspan(
                lo,
                hi,
                color=color,
                alpha=args.stop_alpha,
                linewidth=0,
                zorder=1.0,
            )
        stop_lw = float(METHOD_STYLE[kind]["linewidth"])
        ax.axvline(
            mean,
            color=color,
            linestyle="--",
            linewidth=stop_lw,
            alpha=args.stop_line_alpha,
            zorder=2.0,
        )


def make_legend(
    fig: plt.Figure,
    available_kinds: set[str],
    args: argparse.Namespace,
) -> None:
    show_stops = not args.no_show_stopping
    show_band = (not args.no_range) or (
        show_stops and args.stop_band != "none"
    )
    handles: list[object] = []
    labels: list[str] = []
    for entry_type, kind in LEGEND_ENTRIES:
        if entry_type == "method":
            assert kind is not None
            if kind not in available_kinds:
                continue
            style = METHOD_STYLE[kind]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=pg.STYLE_BY_KIND[kind]["color"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                )
            )
            labels.append(pg.STYLE_BY_KIND[kind]["label"])
        elif entry_type == "stop":
            assert kind is not None
            if not show_stops or kind not in available_kinds:
                continue
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=pg.STYLE_BY_KIND[kind]["color"],
                    linewidth=float(METHOD_STYLE[kind]["linewidth"]),
                    linestyle="--",
                    alpha=args.stop_line_alpha,
                )
            )
            labels.append(STOP_LABELS[kind])
        elif entry_type == "band":
            if not show_band:
                continue
            band_name = "SE" if args.stop_band == "stderr" else "std"
            handles.append(Patch(facecolor="0.55", edgecolor="none", alpha=0.14))
            labels.append(rf"$\pm${args.stderr_k:g} {band_name} band")

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.755, 0.50),
        ncol=1,
        frameon=False,
        fontsize=args.legend_size,
        handlelength=READABLE_STYLE["legend"]["handlelength"],
        handletextpad=READABLE_STYLE["legend"]["handletextpad"],
        columnspacing=READABLE_STYLE["legend"]["columnspacing"],
        labelspacing=READABLE_STYLE["legend"]["labelspacing"],
    )


def exact_curve_overlaps(curves: pd.DataFrame) -> list[str]:
    """Report only byte-for-byte-equivalent cached x/mean curves."""
    overlaps: list[str] = []
    panel_cols = ["dataset", "cost_mode", "x_axis"]
    for panel_key, panel in curves.groupby(panel_cols, sort=False):
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for kind, group in panel.groupby("kind", sort=False):
            ordered = group.sort_values("x")
            series[str(kind)] = (
                ordered["x"].to_numpy(dtype=float),
                ordered["mean"].to_numpy(dtype=float),
            )
        for left, right in combinations(series, 2):
            lx, ly = series[left]
            rx, ry = series[right]
            if np.array_equal(lx, rx) and np.array_equal(ly, ry):
                overlaps.append(
                    f"{panel_key[0]}/{panel_key[1]}: {left} == {right}"
                )
    return overlaps


def draw_figure(
    curves: pd.DataFrame,
    stops: pd.DataFrame,
    args: argparse.Namespace,
    text_config: dict[str, object],
) -> plt.Figure:
    datasets = [
        ("gsm8k", "GSM8K"),
        ("piqa", "PIQA"),
        ("alpaca", "AlpacaEval"),
    ]
    panels = [
        ("unit", "cum_eval"),
        ("aware", "cum_original_cost"),
    ]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(args.fig_width + 2.6, args.fig_height),
        sharey=False,
        constrained_layout=False,
    )
    available_kinds: set[str] = set()

    for col, (dataset, title) in enumerate(datasets):
        for row, (cost_mode, x_axis) in enumerate(panels):
            ax = axes[row, col]
            panel = curves[
                (curves["dataset"] == dataset)
                & (curves["cost_mode"] == cost_mode)
                & (curves["x_axis"] == x_axis)
            ]
            for kind in METHOD_STYLE:
                kg = panel[panel["kind"] == kind].sort_values("x")
                if kg.empty:
                    continue
                style = METHOD_STYLE[kind]
                color = pg.STYLE_BY_KIND[kind]["color"]
                x_percentage = as_full_cost_percentage(
                    kg["x"], dataset, cost_mode
                )
                ax.plot(
                    x_percentage,
                    kg["mean"],
                    color=color,
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    label=pg.STYLE_BY_KIND[kind]["label"],
                    zorder=style["zorder"],
                    solid_capstyle="round",
                )
                available_kinds.add(kind)
                if not args.no_range:
                    band_alpha = (
                        READABLE_STYLE["uncertainty"]["primary_alpha"]
                        if kind.startswith("gittins_")
                        else READABLE_STYLE["uncertainty"]["baseline_alpha"]
                    )
                    ax.fill_between(
                        x_percentage,
                        kg["band_lo"].to_numpy(dtype=float),
                        kg["band_hi"].to_numpy(dtype=float),
                        color=color,
                        alpha=band_alpha,
                        linewidth=0,
                        zorder=style["zorder"] - 0.8,
                    )

            panel_stops = (
                stops[
                    (stops["dataset"] == dataset)
                    & (stops["cost_mode"] == cost_mode)
                    & (stops["x_axis"] == x_axis)
                ]
                if not stops.empty
                else stops
            )
            draw_stops(ax, panel_stops, args, dataset, cost_mode)
            ax.grid(
                True,
                color="0.45",
                alpha=READABLE_STYLE["axes"]["grid_alpha"],
                linewidth=READABLE_STYLE["axes"]["grid_linewidth"],
            )
            ax.tick_params(
                axis="both",
                labelsize=args.tick_size,
                width=READABLE_STYLE["axes"]["tick_width"],
                length=READABLE_STYLE["axes"]["tick_length"],
            )
            x_pad = float(READABLE_STYLE["axes"]["x_pad_pct"])
            ax.set_xlim(-x_pad, float(READABLE_STYLE["axes"]["x_right"]))
            ax.set_xticks([0.0, 5.0, 10.0])
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.yaxis.set_major_formatter(FuncFormatter(pg.clean_tick_label))
            # Mirror the x-axis inset with a small bottom padding below y=0.
            y_lo, y_hi = ax.get_ylim()
            y_pad = float(READABLE_STYLE["axes"]["y_pad_frac"]) * (y_hi - y_lo)
            ax.set_ylim(min(0.0, y_lo) - y_pad, y_hi)
            for spine in ax.spines.values():
                spine.set_linewidth(READABLE_STYLE["axes"]["spine_linewidth"])
        axes[0, col].set_title(
            title, fontsize=args.title_size, fontweight="normal", pad=5
        )

    fig.supylabel(
        str(text_config["shared_y_label"]),
        x=0.010,
        y=0.55,
        fontsize=args.title_size,
    )
    fig.supxlabel(
        str(text_config["shared_x_label"]),
        x=0.395,
        y=-0.01,
        fontsize=args.title_size,
    )
    for row, text in enumerate(text_config["row_labels"]):
        axes[row, 2].text(
            1.04,
            0.5,
            text,
            transform=axes[row, 2].transAxes,
            rotation=-90,
            va="center",
            ha="left",
            fontsize=args.row_label_size,
            fontweight="normal",
        )

    make_legend(fig, available_kinds, args)
    fig.subplots_adjust(
        left=0.078,
        right=0.725,
        top=0.905,
        bottom=0.165,
        wspace=0.28,
        hspace=0.28,
    )
    return fig


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.data_dir / "plot_data_curves.csv")
    stops_path = args.data_dir / "plot_data_stopping.csv"
    stops = pd.read_csv(stops_path) if stops_path.exists() else pd.DataFrame()

    set_colors(args)
    pg.setup_matplotlib(args)

    fig = draw_figure(curves, stops, args, TEXT_PRESETS["percentage_preview"])
    suffix = "_readable_shared_axes_legend_right_percentage"
    out_png = args.out_dir / f"{args.stem}{suffix}.png"
    out_pdf = args.out_dir / f"{args.stem}{suffix}.pdf"
    fig.savefig(
        out_png,
        dpi=args.png_dpi,
        bbox_inches="tight",
        pad_inches=0.16,
    )
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    for path in (out_png, out_pdf):
        print(f"Wrote {path}")
    overlaps = exact_curve_overlaps(curves)
    if overlaps:
        print("Exact cached-curve overlaps:")
        for item in overlaps:
            print(f"  {item}")
    else:
        print("Exact cached-curve overlaps: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
