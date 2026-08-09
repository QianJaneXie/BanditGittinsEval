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
        "title": 21.0,
        "axis_label": 22.0,
        "tick": 17.0,
        "legend": 15.5,
        "row_label": 20.0,
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
        "legend_anchor_y": 0.160,
    },
    "uncertainty": {
        "primary_alpha": 0.20,
        "baseline_alpha": 0.155,
        "stop_band_alpha": 0.07,
        "stop_line_alpha": 0.70,
        "stop_linewidth": None,
    },
    "legend": {
        "handlelength": 2.6,
        "handletextpad": 0.5,
        "columnspacing": 1.15,
        "labelspacing": 0.45,
        "ncol": 6,
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
    "shared_y_label": "Normalized Simple Regret",
    "shared_x_label": "Percentage of Exhaustive Evaluation Cost",
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
    "prompteval_bai": {
        "linewidth": 2.25,
        "linestyle": "-",
        "alpha": 0.94,
        "zorder": 4.8,
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
    "prompteval_bai",
]

# Bottom legend: 2 rows x 6 columns (same distribution as the GSM8K-style flat legend).
# Matplotlib fills legends column-major, so list Col1 (top→bottom), then Col2, ...:
#   Col1: Gittins-S, Gittins-G
#   Col2: matching mean stops
#   Col3: BO-PBGI, BO-LogEI(PC)
#   Col4: matching mean stops
#   Col5: UCB-E, LRF
#   Col6: SySRs, ±SE
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
    ("method", "prompteval_bai"),
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
    "prompteval_bai": "PromptEval",
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
    "prompteval_bai",
]

DEFAULT_STEM = (
    "mmlu_aggregate_2x2_{mode}_Bsmall2_Blarge8_scale1e-4_"
    "xfinal_ynone_fast_lrf_bo_xoffset_bo_after_init_with_sysrs"
)

PROMPTEVAL_RESULTS_DIR = Path(
    r"outputs\prompteval_downloads\wallclock_mmlu_gsm8k_piqa_20260807\results\formal\mmlu"
)
MMLU_MATRIX_DIR = Path("data/MMLU_matrices")
MMLU_COST_VECTOR = Path("data_analysis/pricing/mmlu_prompt_eval_configurations_input_price.json")
_COST_VECTOR_CACHE: np.ndarray | None = None


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
    p.add_argument("--color-prompteval", default="tab:cyan")
    p.add_argument(
        "--prompteval-aware-bin-width-percent",
        type=float,
        default=0.25,
        help="Cost-aware PromptEval raw-point bin width in full-cost percent.",
    )
    p.add_argument(
        "--prompteval-bo-style-extend",
        action="store_true",
        help=(
            "Aggregate PromptEval like BO coverage: use a common x grid per group, "
            "left-extend each task with its first value and right-extend with its last value."
        ),
    )
    p.add_argument(
        "--prompteval-average-initial-align",
        action="store_true",
        help=(
            "For --prompteval-bo-style-extend, start each group at the average "
            "initial PromptEval x, matching the BO average-initial alignment idea."
        ),
    )
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
        "prompteval_bai": args.color_prompteval,
    }
    for kind, color in colors.items():
        if kind == "prompteval_bai" and kind not in pg.STYLE_BY_KIND:
            pg.STYLE_BY_KIND[kind] = {"color": color}
        if kind in pg.STYLE_BY_KIND:
            pg.STYLE_BY_KIND[kind]["color"] = color
    pg.COLOR_GITTINS_S = args.color_gittins_s
    pg.COLOR_GITTINS_G = args.color_gittins_g


def load_mmlu_cost_vector() -> np.ndarray:
    global _COST_VECTOR_CACHE
    if _COST_VECTOR_CACHE is not None:
        return _COST_VECTOR_CACHE
    payload = json.loads(MMLU_COST_VECTOR.read_text(encoding="utf-8"))
    _COST_VECTOR_CACHE = np.asarray(
        [float(payload[str(i)]["estimated_cost_per_1m_input_tokens"]) for i in range(len(payload))],
        dtype=float,
    )
    return _COST_VECTOR_CACHE


def full_evaluation_cost(task: str, mode: str) -> float:
    matrix = np.load(MMLU_MATRIX_DIR / f"{task}.npy")
    n_arms, n_examples = int(matrix.shape[0]), int(matrix.shape[1])
    if mode == "unit":
        return float(n_arms * n_examples)
    costs = load_mmlu_cost_vector()
    if costs.size != n_arms:
        raise ValueError(f"Cost vector length {costs.size} != n_arms {n_arms} for {task}")
    return float(n_examples) * float(costs.sum())


def prompteval_curve_path(task: str, mode: str) -> Path:
    suffix = "combined" if mode == "unit" else "costaware_combined"
    return PROMPTEVAL_RESULTS_DIR / f"bai_processed_results_MMLU_{task}_{suffix}.npy"


def load_prompteval_curve(task: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    path = prompteval_curve_path(task, mode)
    if not path.is_file():
        return np.array([]), np.array([])
    payload = np.load(path, allow_pickle=True).item()
    curve = np.asarray(payload["curves"], dtype=float)[0, 0, 0]
    y = np.asarray(curve[0], dtype=float)
    x = np.asarray(curve[1], dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def prompteval_aggregate(
    selected_tasks: dict[str, list[str]] | None,
    mode: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not selected_tasks:
        return pd.DataFrame()
    if bool(getattr(args, "prompteval_bo_style_extend", False)):
        return prompteval_aggregate_bo_style_extend(selected_tasks, mode, args)
    rows: list[dict[str, float | str | int]] = []
    for group, tasks in selected_tasks.items():
        for task in tasks:
            x_raw, y = load_prompteval_curve(task, mode)
            if x_raw.size == 0:
                continue
            full_cost = full_evaluation_cost(task, mode)
            pct_full = x_raw / full_cost * 100.0
            if mode == "aware":
                width = float(args.prompteval_aware_bin_width_percent)
                pct_full = (np.floor(pct_full / width) + 0.5) * width
            x_cached = pct_full / (float(args.budget_fraction) * 100.0)
            for x_value, y_value in zip(x_cached, y):
                rows.append({
                    "group": group,
                    "method_kind": "prompteval_bai",
                    "x": float(x_value),
                    "simple_regret": float(y_value),
                })
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    grouped = raw.groupby(["group", "method_kind", "x"], as_index=False)
    out = grouped["simple_regret"].agg(["mean", "std", "count"]).reset_index()
    out["stderr"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    out["n_curves_total"] = out.groupby(["group", "method_kind"])["count"].transform("sum")
    return out[["group", "method_kind", "x", "mean", "stderr", "n_curves_total"]]


def prompteval_aggregate_bo_style_extend(
    selected_tasks: dict[str, list[str]] | None,
    mode: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not selected_tasks:
        return pd.DataFrame()
    rows: list[dict[str, float | str | int]] = []
    for group, tasks in selected_tasks.items():
        curves: list[tuple[np.ndarray, np.ndarray]] = []
        for task in tasks:
            x_raw, y = load_prompteval_curve(task, mode)
            if x_raw.size == 0:
                continue
            full_cost = full_evaluation_cost(task, mode)
            pct_full = x_raw / full_cost * 100.0
            if mode == "aware":
                width = float(args.prompteval_aware_bin_width_percent)
                pct_full = (np.floor(pct_full / width) + 0.5) * width
            x_cached = pct_full / (float(args.budget_fraction) * 100.0)
            good = np.isfinite(x_cached) & np.isfinite(y)
            x_cached = np.asarray(x_cached[good], dtype=float)
            y = np.asarray(y[good], dtype=float)
            if x_cached.size == 0:
                continue
            order = np.argsort(x_cached)
            x_cached = x_cached[order]
            y = y[order]
            unique_x, last_idx = np.unique(x_cached, return_index=True)
            if unique_x.size != x_cached.size:
                # Keep the last value at duplicate x bins, matching the cached aggregate behavior.
                tmp = pd.DataFrame({"x": x_cached, "y": y}).groupby("x", as_index=False)["y"].last()
                x_cached = tmp["x"].to_numpy(dtype=float)
                y = tmp["y"].to_numpy(dtype=float)
            curves.append((x_cached, y))

        if not curves:
            continue
        target = float(np.mean([float(x_arr[0]) for x_arr, _ in curves]))
        raw_grid = np.asarray(sorted({float(x) for x_arr, _ in curves for x in x_arr}), dtype=float)
        if bool(getattr(args, "prompteval_average_initial_align", False)):
            raw_grid = raw_grid[raw_grid > target]
            x_grid = np.concatenate([[target], raw_grid])
        else:
            x_grid = raw_grid
        values: list[np.ndarray] = []
        for x_arr, y_arr in curves:
            if bool(getattr(args, "prompteval_average_initial_align", False)):
                if target < float(x_arr[0]):
                    x_arr = np.concatenate([[target], x_arr])
                    y_arr = np.concatenate([[float(y_arr[0])], y_arr])
                elif target > float(x_arr[0]):
                    y0 = float(np.interp(target, x_arr, y_arr))
                    keep = x_arr > target
                    x_arr = np.concatenate([[target], x_arr[keep]])
                    y_arr = np.concatenate([[y0], y_arr[keep]])
            idx = np.searchsorted(x_arr, x_grid, side="right") - 1
            idx = np.clip(idx, 0, len(y_arr) - 1)
            values.append(y_arr[idx])
        arr = np.vstack(values)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        stderr = std / np.sqrt(arr.shape[0]) if arr.shape[0] > 0 else np.zeros_like(mean)
        for x_value, mean_value, stderr_value in zip(x_grid, mean, stderr, strict=True):
            rows.append({
                "group": group,
                "method_kind": "prompteval_bai",
                "x": float(x_value),
                "mean": float(mean_value),
                "stderr": float(stderr_value),
                "n_curves_total": int(arr.shape[0]),
            })
    return pd.DataFrame(rows)


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
        stderr = (
            kd["stderr"].to_numpy(dtype=float)
            if "stderr" in kd.columns
            else None
        )
        if kind == "prompteval_bai" and len(x_pct) > 0:
            x_right = float(READABLE_STYLE["axes"]["x_right"])
            if float(x_pct[-1]) < x_right:
                x_pct = np.append(x_pct, x_right)
                mean = np.append(mean, mean[-1])
                if stderr is not None:
                    stderr = np.append(stderr, stderr[-1])
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
            drawstyle="steps-post" if kind == "prompteval_bai" else "default",
        )
        available.add(canonical_kind(kind))
        if not args.no_range and stderr is not None:
            band = stderr * float(args.stderr_k)
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
                step="post" if kind == "prompteval_bai" else None,
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
    # Keep a fixed 2x6 column-major legend template; do not drop mid-list
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
            handles.append(Patch(facecolor="0.55", edgecolor="none", alpha=0.18))
            labels.append(rf"$\pm$ {args.stderr_k:g} SE band")

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
        fontsize=args.label_size,
    )
    fig.supxlabel(
        str(TEXT_CONFIG["shared_x_label"]),
        x=float(labels_cfg["shared_x_x"]),
        y=float(labels_cfg["shared_x_y"]),
        fontsize=args.label_size,
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
    selected = unit_groups.get("selected_tasks") or aware_groups.get("selected_tasks")
    unit_bai = prompteval_aggregate(selected, "unit", args)
    aware_bai = prompteval_aggregate(selected, "aware", args)
    if not unit_bai.empty:
        unit_curves = pd.concat([unit_curves, unit_bai], ignore_index=True, sort=False)
    if not aware_bai.empty:
        aware_curves = pd.concat([aware_curves, aware_bai], ignore_index=True, sort=False)

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
