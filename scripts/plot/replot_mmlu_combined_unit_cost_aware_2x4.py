#!/usr/bin/env python3
"""Combine MMLU unit-cost and cost-aware aggregate caches into one 2x4 paper figure.

Reads the existing with_sysrs aggregated CSVs / stopping caches produced by
``plot_mmlu_aggregate_2x2_normalized_fast_shared_labels.py`` (and SySRs merge).
Does not recompute experiment results. Display x only: cached final-normalized
fraction of the 10% budget run is shown as percentage of full evaluation cost.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

import plot_gsm8k_piqa_paper_grid as pg


# MMLU runs use eval_budget_fraction=0.1; caches normalize x to that final budget
# (normalize_x=final), so fraction_of_full = cached_x * 0.1 and percent = *100.
EVAL_BUDGET_FRACTION = 0.1

READABLE_STYLE = {
    "figure": {
        # Match 2x3 panel aspect, with modest bottom room for x-label + legend.
        "width": 13.4,
        "height": 5.55,
        "png_dpi": 360,
    },
    "font": {
        "title": 20.0,
        "axis_label": 20.0,
        "tick": 17.0,
        "legend": 17.0,
        "row_label": 19.0,
    },
    "axes": {
        "grid_alpha": 0.14,
        "grid_linewidth": 0.55,
        "spine_linewidth": 0.9,
        "tick_width": 0.9,
        "tick_length": 4.8,
        "x_pad_pct": 0.25,
        "x_right": 10.5,
        "y_pad_frac": 0.04,
    },
    "layout": {
        "left": 0.078,
        "right": 0.955,
        "top": 0.905,
        "bottom": 0.365,
        "wspace": 0.28,
        "hspace": 0.28,
    },
    "labels": {
        "shared_y_x": 0.010,
        "shared_y_y": 0.620,
        "shared_x_x": 0.516,
        "shared_x_y": 0.195,
        "legend_anchor_y": 0.105,
    },
    "uncertainty": {
        "primary_alpha": 0.10,
        "baseline_alpha": 0.065,
        "stop_band_alpha": 0.07,
        "stop_line_alpha": 0.70,
        "stop_linewidth": None,
    },
    "legend": {
        "handlelength": 2.6,
        "handletextpad": 0.5,
        "columnspacing": 1.35,
        "labelspacing": 0.45,
        "ncol": 3,
    },
}

# Sparse y-ticks; always keep 0.
Y_TICKS_BY_GROUP = {
    "Easy-Small": [0.0, 0.1],
    "Easy-Large": [0.0, 0.05, 0.1],
    "Hard-Small": [0.0, 0.1, 0.2, 0.3],
    "Hard-Large": [0.0, 0.1, 0.2, 0.3],
}

TEXT_CONFIG = {
    "row_labels": ("Unit-cost", "Cost-Aware"),
    "shared_y_label": "Simple Regret",
    "shared_x_label": "Exhaustive Evaluation Cost (%)",
    "column_titles": (
        "Easy–Small",
        "Easy–Large",
        "Hard–Small",
        "Hard–Large",
    ),
}

COLUMN_GROUPS = (
    "Easy-Small",
    "Easy-Large",
    "Hard-Small",
    "Hard-Large",
)

COST_MODES = (
    ("unit", "Unit-cost"),
    ("aware", "Cost-Aware"),
)

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

# Map cost-aware BO kinds onto the shared legend keys used by the 2x3 figure.
LEGEND_KIND_ALIAS = {
    "bo_pbgi_cost": "bo_pbgi_unit",
    "bo_logeipc_cost": "bo_logei_unit",
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

# Bottom legend: 4 rows x 3 columns (same appendix format).
# Matplotlib fills legends column-major, so list Col1 (top→bottom), then Col2, then Col3:
#   Col1: Gittins-S, Gittins-G, BO-PBGI, BO-LogEI(PC)
#   Col2: matching mean stops
#   Col3: UCB-E, LRF, SySRs, ±SE
LEGEND_ENTRIES = [
    ("method", "gittins_data"),
    ("method", "gittins_default"),
    ("method", "bo_pbgi_unit"),
    ("method", "bo_logei_unit"),
    ("stop", "gittins_data"),
    ("stop", "gittins_default"),
    ("stop", "bo_pbgi_unit"),
    ("stop", "bo_logei_unit"),
    ("method", "ucb"),
    ("method", "lrf"),
    ("method", "sysrs"),
    ("band", None),
]

DISPLAY_LABEL = {
    "gittins_data": "Gittins-S",
    "gittins_default": "Gittins-G",
    "sysrs": "SySRs",
    "ucb": "UCB-E",
    "lrf": "LRF",
    "bo_pbgi_unit": "BO-PBGI",
    "bo_logei_unit": "BO-LogEI(PC)",
    "bo_pbgi_cost": "BO-PBGI",
    "bo_logeipc_cost": "BO-LogEI(PC)",
}

STOP_LABELS = {
    "gittins_data": "Gittins-S mean stop",
    "gittins_default": "Gittins-G mean stop",
    "bo_pbgi_unit": "BO-PBGI mean stop",
    "bo_logei_unit": "BO-LogEI(PC) mean stop",
}

STOP_LEGEND = [
    ("gittins_data", "Gittins-S mean stop"),
    ("gittins_default", "Gittins-G mean stop"),
    ("bo_pbgi_unit", "BO-PBGI mean stop"),
    ("bo_logei_unit", "BO-LogEI(PC) mean stop"),
]

DRAW_ORDER = [
    "gittins_data",
    "gittins_default",
    "sysrs",
    "ucb",
    "lrf",
    "bo_pbgi_unit",
    "bo_logei_unit",
    "bo_pbgi_cost",
    "bo_logeipc_cost",
]

DEFAULT_STEM = (
    "mmlu_aggregate_2x2_{mode}_Bsmall2_Blarge8_scale1e-4_"
    "xfinal_ynone_fast_lrf_bo_xoffset_bo_after_init_with_sysrs"
)


def parse_args() -> argparse.Namespace:
    data_dir = Path("outputs/figure/new_figure/final/mmlu2_2")
    out_dir = Path("outputs/figure/new_figure/final/mmlucombined")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=data_dir)
    p.add_argument("--out-dir", type=Path, default=out_dir)
    p.add_argument(
        "--unit-curve",
        type=Path,
        default=None,
        help="Override unit aggregated CSV path.",
    )
    p.add_argument(
        "--aware-curve",
        type=Path,
        default=None,
        help="Override cost-aware aggregated CSV path.",
    )
    p.add_argument("--unit-stop", type=Path, default=None)
    p.add_argument("--aware-stop", type=Path, default=None)
    p.add_argument(
        "--budget-fraction",
        type=float,
        default=EVAL_BUDGET_FRACTION,
        help="Experiment budget as a fraction of full evaluation (default 0.1).",
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
    p.add_argument(
        "--y-share",
        choices=["global", "column"],
        default="column",
        help="Share y-limits globally or per column (unit+aware). Default column "
        "because Easy vs Hard regret scales differ by ~2x.",
    )
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
        if kind in pg.STYLE_BY_KIND:
            pg.STYLE_BY_KIND[kind]["color"] = color
    pg.COLOR_GITTINS_S = args.color_gittins_s
    pg.COLOR_GITTINS_G = args.color_gittins_g


def canonical_kind(kind: str) -> str:
    return LEGEND_KIND_ALIAS.get(kind, kind)


def as_full_cost_percentage(
    values: float | pd.Series | np.ndarray,
    budget_fraction: float,
) -> float | np.ndarray:
    """Convert final-budget-normalized x in [0,1] to % of full evaluation cost."""
    converted = np.asarray(values, dtype=float) * float(budget_fraction) * 100.0
    return float(converted) if converted.ndim == 0 else converted


def default_paths(data_dir: Path, mode: str) -> tuple[Path, Path, Path]:
    stem = DEFAULT_STEM.format(mode=mode)
    curve = data_dir / f"{stem}_aggregated.csv"
    groups = data_dir / f"{stem}_groups.json"
    # Stopping cache is shared with the pre-SySRs aggregate stem (SySRs has no early stop).
    stop_stem = stem.replace("_with_sysrs", "")
    stop = data_dir / f"{stop_stem}_stops.csv"
    return curve, stop, groups


def load_mode_tables(
    data_dir: Path,
    mode: str,
    curve_override: Path | None,
    stop_override: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    curve_path, stop_path, groups_path = default_paths(data_dir, mode)
    if curve_override is not None:
        curve_path = curve_override
    if stop_override is not None:
        stop_path = stop_override
    if not curve_path.is_file():
        raise FileNotFoundError(f"missing {mode} curve cache: {curve_path}")
    curves = pd.read_csv(curve_path)
    stops = pd.read_csv(stop_path) if stop_path.is_file() else pd.DataFrame()
    groups: dict = {}
    if groups_path.is_file():
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
    print(f"[LOAD] {mode} curves: {curve_path}")
    print(f"[LOAD] {mode} stops:  {stop_path if stop_path.is_file() else '(none)'}")
    return curves, stops, groups


def compute_y_limits(
    unit_curves: pd.DataFrame,
    aware_curves: pd.DataFrame,
    share: str,
) -> dict[str, tuple[float, float]]:
    """Return mapping group -> (ymin, ymax), matching the 2x3 bottom-only pad."""
    pad = float(READABLE_STYLE["axes"]["y_pad_frac"])
    limits: dict[str, tuple[float, float]] = {}

    def padded(lo: float, hi: float) -> tuple[float, float]:
        span = max(hi - lo, 1e-9)
        # Same as 2x3: inset 0 slightly above the bottom spine; no extra top pad.
        return (min(0.0, lo) - pad * span, hi)

    if share == "global":
        y_all = pd.concat(
            [unit_curves["mean"], aware_curves["mean"]], ignore_index=True
        ).to_numpy(dtype=float)
        y0, y1 = padded(float(np.nanmin(y_all)), float(np.nanmax(y_all)))
        for group in COLUMN_GROUPS:
            limits[group] = (y0, y1)
        return limits

    for group in COLUMN_GROUPS:
        parts = []
        for df in (unit_curves, aware_curves):
            g = df[df["group"].astype(str) == group]
            if not g.empty:
                parts.append(g["mean"].to_numpy(dtype=float))
        if not parts:
            limits[group] = (0.0, 1.0)
            continue
        y = np.concatenate(parts)
        limits[group] = padded(float(np.nanmin(y)), float(np.nanmax(y)))
    return limits


def draw_stops(
    ax: plt.Axes,
    stops: pd.DataFrame,
    group: str,
    args: argparse.Namespace,
) -> set[str]:
    seen: set[str] = set()
    if args.no_show_stopping or stops.empty:
        return seen
    group_stops = stops[stops["group"].astype(str) == group]
    for row in group_stops.itertuples(index=False):
        kind = str(row.method_kind)
        if kind not in METHOD_STYLE:
            continue
        color = pg.STYLE_BY_KIND[kind]["color"]
        mean = as_full_cost_percentage(float(row.x), args.budget_fraction)
        band = float(getattr(row, "band", 0.0) or 0.0)
        band_pct = as_full_cost_percentage(band, args.budget_fraction)
        if np.isfinite(band_pct) and band_pct > 0:
            ax.axvspan(
                mean - band_pct,
                mean + band_pct,
                color=color,
                alpha=args.stop_alpha,
                linewidth=0,
                zorder=1.0,
            )
        ax.axvline(
            mean,
            color=color,
            linestyle="--",
            linewidth=float(METHOD_STYLE[kind]["linewidth"]),
            alpha=args.stop_line_alpha,
            zorder=2.0,
        )
        seen.add(canonical_kind(kind))
    return seen


def draw_panel(
    ax: plt.Axes,
    *,
    curves: pd.DataFrame,
    stops: pd.DataFrame,
    group: str,
    args: argparse.Namespace,
    y_lim: tuple[float, float],
) -> set[str]:
    available: set[str] = set()
    group_df = curves[curves["group"].astype(str) == group]
    for kind in DRAW_ORDER:
        kd = group_df[group_df["method_kind"].astype(str) == kind].sort_values("x")
        if kd.empty:
            continue
        style = METHOD_STYLE[kind]
        color = pg.STYLE_BY_KIND[kind]["color"]
        x_pct = as_full_cost_percentage(kd["x"], args.budget_fraction)
        mean = kd["mean"].to_numpy(dtype=float)
        ax.plot(
            x_pct,
            mean,
            color=color,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            label=DISPLAY_LABEL[kind],
            zorder=style["zorder"],
            solid_capstyle="round",
        )
        available.add(canonical_kind(kind))
        if not args.no_range and "stderr" in kd.columns:
            band = kd["stderr"].to_numpy(dtype=float) * float(args.stderr_k)
            band_alpha = (
                READABLE_STYLE["uncertainty"]["primary_alpha"]
                if kind.startswith("gittins_")
                else READABLE_STYLE["uncertainty"]["baseline_alpha"]
            )
            ax.fill_between(
                x_pct,
                mean - band,
                mean + band,
                color=color,
                alpha=band_alpha,
                linewidth=0,
                zorder=style["zorder"] - 0.8,
            )

    available |= draw_stops(ax, stops, group, args)

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
    # Keep compact labels: 0, 0.05, 0.1, 0.2, 0.3 (no forced trailing zeros).
    ax.yaxis.set_major_formatter(FuncFormatter(pg.clean_tick_label))
    ax.set_yticks(Y_TICKS_BY_GROUP.get(group, [0.0, 0.1, 0.2, 0.3]))
    ax.set_ylim(*y_lim)
    ax.tick_params(axis="y", pad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(READABLE_STYLE["axes"]["spine_linewidth"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    return available


def make_legend(
    fig: plt.Figure,
    available_kinds: set[str],
    stop_kinds: set[str],
    args: argparse.Namespace,
) -> None:
    # Keep a fixed 4x3 column-major legend template; do not drop mid-list
    # entries or Matplotlib will scramble the intended columns.
    _ = available_kinds, stop_kinds
    show_stops = not args.no_show_stopping
    show_band = not args.no_range
    handles: list[object] = []
    labels: list[str] = []
    for entry_type, kind in LEGEND_ENTRIES:
        if entry_type == "method":
            assert kind is not None
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
            labels.append(DISPLAY_LABEL[kind])
        elif entry_type == "stop":
            assert kind is not None
            if not show_stops:
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
            handles.append(Patch(facecolor="0.55", edgecolor="none", alpha=0.14))
            labels.append(rf"$\pm${args.stderr_k:g} SE band")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.516, float(READABLE_STYLE["labels"]["legend_anchor_y"])),
        ncol=int(READABLE_STYLE["legend"]["ncol"]),
        frameon=False,
        fontsize=args.legend_size,
        handlelength=READABLE_STYLE["legend"]["handlelength"],
        handletextpad=READABLE_STYLE["legend"]["handletextpad"],
        columnspacing=READABLE_STYLE["legend"]["columnspacing"],
        labelspacing=READABLE_STYLE["legend"]["labelspacing"],
    )


def exact_curve_overlaps(curves: pd.DataFrame, cost_mode: str) -> list[str]:
    overlaps: list[str] = []
    for group, panel in curves.groupby("group", sort=False):
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for kind, kg in panel.groupby("method_kind", sort=False):
            ordered = kg.sort_values("x")
            series[str(kind)] = (
                ordered["x"].to_numpy(dtype=float),
                ordered["mean"].to_numpy(dtype=float),
            )
        for left, right in combinations(series, 2):
            lx, ly = series[left]
            rx, ry = series[right]
            if np.array_equal(lx, rx) and np.array_equal(ly, ry):
                overlaps.append(f"{cost_mode}/{group}: {left} == {right}")
    return overlaps


def plot_combined_mmlu_unit_cost_cost_aware(
    unit_curves: pd.DataFrame,
    aware_curves: pd.DataFrame,
    unit_stops: pd.DataFrame,
    aware_stops: pd.DataFrame,
    args: argparse.Namespace,
    selected_tasks: dict[str, list[str]] | None = None,
) -> plt.Figure:
    """Main combined-figure entry: 2 rows (unit / cost-aware) x 4 columns."""
    y_limits = compute_y_limits(unit_curves, aware_curves, args.y_share)
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(args.fig_width, args.fig_height),
        sharey=False,
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1, 1, 1, 1], "height_ratios": [1, 1]},
    )
    available_kinds: set[str] = set()
    stop_kinds: set[str] = set()
    mode_tables = {
        "unit": (unit_curves, unit_stops),
        "aware": (aware_curves, aware_stops),
    }

    for row, (mode, _row_name) in enumerate(COST_MODES):
        curves, stops = mode_tables[mode]
        for col, group in enumerate(COLUMN_GROUPS):
            ax = axes[row, col]
            seen = draw_panel(
                ax,
                curves=curves,
                stops=stops,
                group=group,
                args=args,
                y_lim=y_limits[group],
            )
            available_kinds |= seen
            if not args.no_show_stopping and not stops.empty:
                gstops = stops[stops["group"].astype(str) == group]
                for kind in gstops["method_kind"].astype(str).unique():
                    stop_kinds.add(canonical_kind(str(kind)))
            if row == 0:
                axes[0, col].set_title(
                    TEXT_CONFIG["column_titles"][col],
                    fontsize=args.title_size,
                    fontweight="normal",
                    pad=5,
                )

    labels_cfg = READABLE_STYLE["labels"]
    fig.supylabel(
        str(TEXT_CONFIG["shared_y_label"]),
        x=float(labels_cfg["shared_y_x"]),
        y=float(labels_cfg["shared_y_y"]),
        fontsize=args.title_size,
    )
    fig.supxlabel(
        str(TEXT_CONFIG["shared_x_label"]),
        x=float(labels_cfg["shared_x_x"]),
        y=float(labels_cfg["shared_x_y"]),
        fontsize=args.title_size,
    )
    for row, text in enumerate(TEXT_CONFIG["row_labels"]):
        axes[row, -1].text(
            1.045,
            0.5,
            text,
            transform=axes[row, -1].transAxes,
            rotation=-90,
            va="center",
            ha="left",
            fontsize=args.row_label_size,
            fontweight="normal",
        )

    make_legend(fig, available_kinds, stop_kinds, args)
    layout = READABLE_STYLE["layout"]
    fig.subplots_adjust(
        left=float(layout["left"]),
        right=float(layout["right"]),
        top=float(layout["top"]),
        bottom=float(layout["bottom"]),
        wspace=float(layout["wspace"]),
        hspace=float(layout["hspace"]),
    )
    # Re-snap columns so spine-to-spine gutters are exactly equal.
    equalize_column_gutters(axes)

    if selected_tasks:
        print("[GROUPS] column mapping (high prior→Easy, low prior→Hard):")
        for group, title in zip(COLUMN_GROUPS, TEXT_CONFIG["column_titles"]):
            tasks = selected_tasks.get(group, [])
            print(f"  {title}: {', '.join(tasks)}")
    return fig


def equalize_column_gutters(axes: np.ndarray) -> None:
    """Force identical panel widths and identical inter-column spine gaps."""
    n_rows, n_cols = axes.shape
    ref = [axes[0, c].get_position() for c in range(n_cols)]
    left0 = float(ref[0].x0)
    right_last = float(ref[-1].x1)
    total = right_last - left0
    gaps = [float(ref[c + 1].x0 - ref[c].x1) for c in range(n_cols - 1)]
    gap = float(np.mean(gaps)) if gaps else 0.0
    width = (total - (n_cols - 1) * gap) / n_cols
    for r in range(n_rows):
        for c in range(n_cols):
            pos = axes[r, c].get_position()
            x0 = left0 + c * (width + gap)
            axes[r, c].set_position([x0, pos.y0, width, pos.height])

def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_colors(args)
    pg.setup_matplotlib(args)

    unit_curves, unit_stops, unit_groups = load_mode_tables(
        args.data_dir, "unit", args.unit_curve, args.unit_stop
    )
    aware_curves, aware_stops, aware_groups = load_mode_tables(
        args.data_dir, "aware", args.aware_curve, args.aware_stop
    )

    # Report x coverage before percentage conversion.
    for mode, df in (("unit", unit_curves), ("aware", aware_curves)):
        xmax = float(df["x"].max())
        pct_max = as_full_cost_percentage(xmax, args.budget_fraction)
        print(
            f"[X-RANGE] {mode}: cached x in "
            f"[{float(df['x'].min()):.4f}, {xmax:.4f}] "
            f"→ display % of full ≈ [{as_full_cost_percentage(float(df['x'].min()), args.budget_fraction):.2f}, "
            f"{pct_max:.2f}] (budget_fraction={args.budget_fraction})"
        )

    selected = unit_groups.get("selected_tasks") or aware_groups.get("selected_tasks")
    fig = plot_combined_mmlu_unit_cost_cost_aware(
        unit_curves,
        aware_curves,
        unit_stops,
        aware_stops,
        args,
        selected_tasks=selected,
    )

    out_png = args.out_dir / "mmlu_combined_unit_cost_cost_aware.png"
    out_pdf = args.out_dir / "mmlu_combined_unit_cost_cost_aware.pdf"
    fig.savefig(out_png, dpi=args.png_dpi, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Wrote:\n  {out_png}\n  {out_pdf}")

    # Algorithm / seed coverage comparison.
    def coverage(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(["group", "method_kind"], sort=True)["n_curves_total"]
            .first()
            .unstack()
        )

    print("[COVERAGE] unit n_curves_total:")
    print(coverage(unit_curves))
    print("[COVERAGE] aware n_curves_total:")
    print(coverage(aware_curves))
    unit_kinds = set(unit_curves["method_kind"].astype(str))
    aware_kinds = set(aware_curves["method_kind"].astype(str))
    print(f"[DIFF] unit-only kinds: {sorted(unit_kinds - aware_kinds)}")
    print(f"[DIFF] aware-only kinds: {sorted(aware_kinds - unit_kinds)}")

    overlaps = exact_curve_overlaps(unit_curves, "unit") + exact_curve_overlaps(
        aware_curves, "aware"
    )
    if overlaps:
        print("Exact cached-curve overlaps:")
        for item in overlaps:
            print(f"  {item}")
    else:
        print("Exact cached-curve overlaps: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
