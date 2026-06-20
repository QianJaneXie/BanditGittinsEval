#!/usr/bin/env python3
"""Paper-style 2x2 figure for GSM8K and PIQA.

Figure layout:
  columns: GSM8K, PIQA
  rows:    unit-cost, cost-aware

This version keeps two Gittins curves:
- orange = Gittins-S
- green  = Gittins-G

Uncertainty:
- curve band = mean ± k·standard error (default k=2)
- stopping-location band = mean stop ± k·standard error (default k=2)

Legend:
- ±k SE band
- Gittins-S mean stop
- Gittins-G mean stop
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
from matplotlib.ticker import FuncFormatter


COLOR_UCB = "tab:blue"
COLOR_LRF = "tab:purple"
COLOR_GITTINS_S = "tab:orange"
COLOR_GITTINS_G = "tab:green"
COLOR_BO_PBGI = "tab:olive"
COLOR_BO_LOGEI = "tab:brown"

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
    "bo_pbgi_unit": {
        "color": COLOR_BO_PBGI,
        "label": "BO-PBGI",
        "linewidth": 3.0,
        "zorder": 2,
    },
    "bo_logei_unit": {
        "color": COLOR_BO_LOGEI,
        "label": "BO-LogEI(PC)",
        "linewidth": 3.0,
        "zorder": 2,
    },
    "bo_pbgi_cost": {
        "color": COLOR_BO_PBGI,
        "label": "BO-PBGI",
        "linewidth": 3.0,
        "zorder": 2,
    },
    "bo_logeipc_cost": {
        "color": COLOR_BO_LOGEI,
        "label": "BO-LogEI(PC)",
        "linewidth": 3.0,
        "zorder": 2,
    },
}

# Thicken curve/stop lines for readability.
LINEWIDTH_MULT = 3.6
GITTINS_LINE_EXTRA_MULT = 1.2


def clean_tick_label(value: float, _pos: int) -> str:
    if abs(float(value)) < 1e-12:
        return "0"
    return f"{float(value):.2f}"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--gsm8k-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads_new\ucb_gittins\gsm8k"),
        help="Folder containing GSM8K runs_history.csv.gz, runs_summary.csv, runs_stopping.csv.",
    )
    p.add_argument(
        "--gsm8k-lrf-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads_new\lrf\gsm8k_lrf"),
        help="Optional folder containing GSM8K LRF runs_history.csv.gz.",
    )
    p.add_argument(
        "--gsm8k-bo-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads_new\bo_baseline\gsm8k_bo"),
        help="Optional folder containing GSM8K BO baseline runs_history.csv.gz.",
    )
    p.add_argument(
        "--piqa-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads_new\ucb_gittins\piqa"),
        help="Folder containing PIQA runs_history.csv.gz, runs_summary.csv, runs_stopping.csv.",
    )
    p.add_argument(
        "--piqa-lrf-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads_new\lrf\piqa_lrf"),
        help="Optional folder containing PIQA LRF runs_history.csv.gz.",
    )
    p.add_argument(
        "--piqa-bo-dir",
        type=Path,
        default=Path(r"outputs\wandb_downloads_new\bo_baseline\piqa_bo"),
        help="Optional folder containing PIQA BO baseline runs_history.csv.gz.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"outputs\wandb_plots\paper_figures"),
    )

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lrf-batch-size", type=int, default=32)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--bo-pbgi-unit-variant", default="pbgi_unit")
    p.add_argument("--bo-logei-unit-variant", default="logei_unit")
    p.add_argument("--bo-pbgi-cost-variant", default="pbgi_cost")
    p.add_argument("--bo-logei-cost-variant", default="logeipc_cost")
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
    p.add_argument("--fig-width", type=float, default=26.8)
    p.add_argument("--fig-height", type=float, default=25.9)

    # Slightly larger than previous version.
    # Increase paper fonts for readability (except tick labels).
    p.add_argument("--title-size", type=float, default=81)
    p.add_argument("--label-size", type=float, default=81)
    # Axis tick labels (numbers) should NOT be enlarged further.
    p.add_argument("--tick-size", type=float, default=57)
    p.add_argument("--legend-size", type=float, default=58)
    # Keep row labels (Unit-cost / Cost-aware) same size as panel titles.
    p.add_argument("--row-label-size", type=float, default=81)

    p.add_argument("--y-limit-min", type=float, default=None)
    p.add_argument("--y-limit-max", type=float, default=None)

    return p.parse_args()


def read_history(folder: Path) -> pd.DataFrame:
    path = folder / "runs_history.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    return pd.read_csv(path, compression="gzip")


def read_histories(folders: list[Path]) -> pd.DataFrame:
    pieces = []
    for folder in folders:
        if folder is None:
            continue
        path = folder / "runs_history.csv.gz"
        if not path.exists():
            print(f"WARNING: missing history file, skipping: {path}")
            continue
        pieces.append(pd.read_csv(path, compression="gzip"))
    if not pieces:
        raise FileNotFoundError("No runs_history.csv.gz files found in requested folders")
    return pd.concat(pieces, ignore_index=True, sort=False)


def read_stopping(folder: Path) -> pd.DataFrame:
    stop_cols = [
        "run_id",
        "experiment_variant",
        "gittins_stop_cum_eval",
        "gittins_stop_cum_original_cost",
        "gittins_recommendation_aware_stop_cum_eval",
        "gittins_recommendation_aware_stop_cum_original_cost",
        "bo_stop_cum_eval",
        "bo_stop_cum_original_cost",
    ]

    stopping_path = folder / "runs_stopping.csv"
    if stopping_path.exists():
        sdf = pd.read_csv(stopping_path)
        if any(c in sdf.columns for c in stop_cols[2:]):
            return sdf

    # New downloader path: stopping metrics are written directly to runs_summary.csv.
    summary_path = folder / "runs_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()

    header = pd.read_csv(summary_path, nrows=0)
    usecols = [c for c in stop_cols if c in header.columns]
    if not usecols:
        return pd.DataFrame()

    return pd.read_csv(summary_path, usecols=usecols)


def read_stoppings(folders: list[Path]) -> pd.DataFrame:
    pieces = []
    for folder in folders:
        if folder is None:
            continue
        sdf = read_stopping(folder)
        if not sdf.empty:
            pieces.append(sdf)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True, sort=False)


def variant_names(
    batch_size: int,
    lrf_batch_size: int,
    scale: str,
    cost_mode: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    b = int(batch_size)
    lb = int(lrf_batch_size)
    if cost_mode == "unit":
        return {
            "gittins_data": f"gittins_unit_B{b}_scale{scale}_dataset",
            "gittins_default": f"gittins_unit_B{b}_scale{scale}_default",
            "ucb": f"ucb_B{b}",
            "lrf": f"lrf_B{lb}",
            "bo_pbgi_unit": str(args.bo_pbgi_unit_variant),
            "bo_logei_unit": str(args.bo_logei_unit_variant),
        }
    if cost_mode in {"aware", "cost"}:
        return {
            "gittins_data": f"gittins_cost_B{b}_scale{scale}_dataset",
            "gittins_default": f"gittins_cost_B{b}_scale{scale}_default",
            "ucb": f"ucb_cost_B{b}",
            "lrf": f"lrf_cost_B{lb}",
            "bo_pbgi_cost": str(args.bo_pbgi_cost_variant),
            "bo_logeipc_cost": str(args.bo_logei_cost_variant),
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


def is_bo_variant(variant: str) -> bool:
    v = str(variant).lower()
    return v.startswith(("pbgi", "logei", "logeipc")) or v.startswith("bo")


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


def crop_bo_after_random_init(df: pd.DataFrame) -> pd.DataFrame:
    if "selection_phase" not in df.columns:
        return df

    pieces = []
    for variant, g in df.groupby("experiment_variant", sort=False):
        if is_bo_variant(str(variant)):
            phase = g["selection_phase"].astype(str).str.lower()
            g = g[phase != "random_init"]
        pieces.append(g)

    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=False)


def bo_average_initial_x(df: pd.DataFrame, x_col: str) -> float:
    if "selection_phase" not in df.columns or x_col not in df.columns:
        return float("nan")

    phase = df["selection_phase"].astype(str).str.lower()
    init = df[phase == "random_init"].copy()
    if init.empty:
        return float("nan")

    vals = []
    for _, rg in init.groupby("run_id", sort=False):
        xs = pd.to_numeric(rg[x_col], errors="coerce").dropna()
        if not xs.empty:
            vals.append(float(xs.max()))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def bo_post_init_aligned_to_average_start(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    target = bo_average_initial_x(df, x_col)
    if "selection_phase" not in df.columns:
        return df

    phase = df["selection_phase"].astype(str).str.lower()
    post = df[phase != "random_init"].copy()
    if post.empty or not math.isfinite(target):
        return post

    rows: list[pd.DataFrame] = []
    for run_id, rg in post.groupby("run_id", sort=False):
        gg = rg[[x_col, y_col]].dropna().sort_values(x_col)
        if gg.empty:
            continue
        gg = gg.groupby(x_col, as_index=False)[y_col].last()
        x = pd.to_numeric(gg[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(gg[y_col], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if len(x) == 0 or target > float(x[-1]):
            continue

        if target < float(x[0]):
            x = np.concatenate([[target], x])
            y = np.concatenate([[float(y[0])], y])
        elif target > float(x[0]):
            y0 = float(np.interp(target, x, y))
            keep = x > target
            x = np.concatenate([[target], x[keep]])
            y = np.concatenate([[y0], y[keep]])

        rows.append(pd.DataFrame({"run_id": run_id, x_col: x, y_col: y}))

    if not rows:
        return post.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True, sort=False)


def aggregate_variant(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    grid_size: int,
    *,
    extend_right: bool = False,
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
        right = float(y[-1]) if extend_right else np.nan
        yi = np.interp(x_grid, x, y, left=np.nan, right=right)
        ys.append(yi)

    y_arr = np.vstack(ys)
    mean = np.nanmean(y_arr, axis=0)
    std = np.nanstd(y_arr, axis=0)
    n = np.sum(~np.isnan(y_arr), axis=0)
    stderr = std / np.sqrt(np.maximum(n, 1))

    return x_grid, mean, std, stderr, n


def stopping_cols_for_kind(kind: str, x_axis: str) -> list[str]:
    if kind.startswith("bo_"):
        if x_axis == "cum_eval":
            return ["bo_stop_cum_eval"]
        if x_axis == "cum_original_cost":
            return ["bo_stop_cum_original_cost"]
        return []
    if x_axis == "cum_eval":
        return ["gittins_stop_cum_eval"]
    if x_axis == "cum_original_cost":
        return ["gittins_stop_cum_original_cost"]
    return []


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

    for kind in kinds_to_draw:
        if kind not in variants:
            continue
        band_color = STYLE_BY_KIND[kind]["color"]
        variant = variants[kind]
        sdf = stop_df[stop_df["experiment_variant"].astype(str) == variant].copy()
        if sdf.empty:
            continue

        vals = pd.Series(dtype=float)
        for stop_col in stopping_cols_for_kind(kind, x_axis):
            if stop_col not in sdf.columns:
                continue
            vals = pd.to_numeric(sdf[stop_col], errors="coerce")
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if not vals.empty:
                break
        if vals.empty:
            print(f"WARNING: no stopping values for {STYLE_BY_KIND[kind]['label']} ({variant}) on {x_axis}")
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
            extra = float(GITTINS_LINE_EXTRA_MULT) if kind in ("gittins_data", "gittins_default") else 1.0
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
    variants = variant_names(args.batch_size, args.lrf_batch_size, args.scale, cost_mode, args)
    df = prepare_panel_df(history, dataset, variants)

    handles: dict[str, plt.Line2D] = {}
    order = [
        "gittins_data",
        "gittins_default",
        "ucb",
        "lrf",
        "bo_pbgi_unit",
        "bo_logei_unit",
        "bo_pbgi_cost",
        "bo_logeipc_cost",
    ]

    for kind in order:
        if kind not in variants:
            continue
        variant = variants[kind]
        vg = df[df["experiment_variant"].astype(str) == variant].copy()

        if vg.empty:
            print(f"WARNING: missing variant for {dataset_title} {cost_mode}: {variant}")
            continue

        if kind.startswith("bo_"):
            vg = bo_post_init_aligned_to_average_start(
                vg,
                x_col=x_axis,
                y_col="simple_regret",
            )
            if vg.empty:
                print(f"WARNING: no BO post-init points for {dataset_title} {cost_mode}: {variant}")
                continue

        x_grid, mean, std, stderr, n = aggregate_variant(
            vg,
            x_axis,
            "simple_regret",
            args.grid_size,
            extend_right=kind.startswith("bo_"),
        )
        if x_grid.size == 0:
            print(f"WARNING: no plottable points for {dataset_title} {cost_mode}: {variant}")
            continue

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
        kinds_to_draw=[
            "gittins_data",
            "gittins_default",
            "bo_pbgi_unit",
            "bo_logei_unit",
            "bo_pbgi_cost",
            "bo_logeipc_cost",
        ],
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
    ax.xaxis.set_major_formatter(FuncFormatter(clean_tick_label))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_tick_label))

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
        {
            "key": "gsm8k",
            "title": "GSM8K",
            "dir": args.gsm8k_dir,
            "history_dirs": [args.gsm8k_dir, args.gsm8k_lrf_dir, args.gsm8k_bo_dir],
            "stopping_dirs": [args.gsm8k_dir, args.gsm8k_bo_dir],
        },
        {
            "key": "piqa",
            "title": "PIQA",
            "dir": args.piqa_dir,
            "history_dirs": [args.piqa_dir, args.piqa_lrf_dir, args.piqa_bo_dir],
            "stopping_dirs": [args.piqa_dir, args.piqa_bo_dir],
        },
    ]

    histories = {}
    stoppings = {}
    for d in datasets:
        histories[d["key"]] = read_histories(d["history_dirs"])
        stoppings[d["key"]] = read_stoppings(d["stopping_dirs"])

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
        axes[0, col].set_title(d["title"], fontsize=args.title_size, fontweight="normal", pad=8)

    axes[0, 0].set_ylabel("Simple regret", fontsize=args.label_size)
    axes[1, 0].set_ylabel("Simple regret", fontsize=args.label_size)

    for col in range(2):
        axes[0, col].set_xlabel("Cumulative evaluations", fontsize=args.label_size)
        axes[1, col].set_xlabel("Cumulative cost", fontsize=args.label_size)

    axes[0, 1].text(
        1.035,
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
        1.035,
        0.5,
        "Cost-aware",
        transform=axes[1, 1].transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=args.row_label_size,
        fontweight="normal",
    )

    legend_order = [
        "gittins_data",
        "gittins_default",
        "ucb",
        "lrf",
        "bo_pbgi_unit",
        "bo_logei_unit",
    ]
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

        final_handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_GITTINS_G,
                linestyle="--",
                linewidth=2.6 * float(LINEWIDTH_MULT) * float(GITTINS_LINE_EXTRA_MULT),
                alpha=args.stop_line_alpha,
            )
        )
        final_labels.append("Gittins-G mean stop")

        for kind in ["bo_pbgi_unit", "bo_logei_unit"]:
            if kind not in legend_handles:
                continue
            final_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=STYLE_BY_KIND[kind]["color"],
                    linestyle="--",
                    linewidth=2.6 * float(LINEWIDTH_MULT),
                    alpha=args.stop_line_alpha,
                )
            )
            final_labels.append(f"{STYLE_BY_KIND[kind]['label']} mean stop")

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
        bbox_to_anchor=(0.5, 0.045),
        fontsize=args.legend_size,
        handlelength=2.0,
        handletextpad=0.35,
        columnspacing=1.15,
        borderpad=0.55,
    )

    # Layout: slightly more space between panels, but use more of the full canvas.
    fig.subplots_adjust(
        left=0.060,
        right=0.985,
        top=0.765,
        bottom=0.375,
        wspace=0.24,
        hspace=0.74,
    )

    stem = f"figure1_gsm8k_piqa_B{args.batch_size}_LRFB{args.lrf_batch_size}_scale{args.scale}".replace(".", "")
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
