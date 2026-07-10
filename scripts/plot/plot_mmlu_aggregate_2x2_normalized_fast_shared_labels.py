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
  y-axis: normalized simple regret = r(t) / mean first simple regret of bandit methods
  band:   mean ± 1 SE
  no stopping lines

Difficulty mapping:
  high prior bucket -> Easy
  low prior bucket  -> Hard
  medium prior bucket is omitted from this 2x2 main-text aggregate.

LRF:
  LRF is plotted whenever downloaded runs are available. Incomplete LRF task
  coverage is allowed; aggregate panels use the available task/run curves.
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

_repo_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


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
    "bo_pbgi_unit": {
        "color": "tab:olive",
        "label": "BO-PBGI",
        "linewidth": 3.0,
        "zorder": 2,
    },
    "bo_logei_unit": {
        "color": "tab:brown",
        "label": "BO-LogEI",
        "linewidth": 3.0,
        "zorder": 2,
    },
    "bo_pbgi_cost": {
        "color": "tab:olive",
        "label": "BO-PBGI",
        "linewidth": 3.0,
        "zorder": 2,
    },
    "bo_logeipc_cost": {
        "color": "tab:brown",
        "label": "BO-LogEIPC",
        "linewidth": 3.0,
        "zorder": 2,
    },
}

LINEWIDTH_MULT = 3.6
GITTINS_LINE_EXTRA_MULT = 1.2


def clean_tick_label(value: float, _pos: int) -> str:
    if abs(float(value)) < 1e-12:
        return "0"
    return f"{float(value):g}"


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--small-root", type=Path, default=Path(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_small"))
    p.add_argument("--large-root", type=Path, default=Path(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_large"))
    p.add_argument("--small-lrf-root", type=Path, default=Path(r"outputs\wandb_downloads_new\lrf\mmlu_small_lrf"))
    p.add_argument("--large-lrf-root", type=Path, default=Path(r"outputs\wandb_downloads_new\lrf\mmlu_large_lrf"))
    p.add_argument("--small-bo-root", type=Path, default=Path(r"outputs\wandb_downloads_new\bo_baseline_5pct\mmlu_small_bo"))
    p.add_argument("--large-bo-root", type=Path, default=Path(r"outputs\wandb_downloads_new\bo_baseline_5pct\mmlu_large_bo"))
    p.add_argument("--task-metadata", type=Path, default=Path(r"data\MMLU_matrices\task_metadata.json"))
    p.add_argument("--out-dir", type=Path, default=Path(r"outputs\wandb_plots\paper_figures"))
    p.add_argument("--cache-curves-only", action="store_true", help="Write aggregated curve/stopping cache, but do not save figures.")
    p.add_argument("--plot-from-cache", action="store_true", help="Replot from existing aggregated curve/stopping cache without reading raw histories.")

    p.add_argument("--cost-mode", choices=["unit", "aware"], default="unit")
    p.add_argument("--small-batch-size", type=int, default=4)
    p.add_argument("--large-batch-size", type=int, default=16)
    p.add_argument("--small-lrf-batch-size", type=int, default=32)
    p.add_argument("--large-lrf-batch-size", type=int, default=32)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--bo-pbgi-unit-variant", default="pbgi_unit")
    p.add_argument("--bo-logei-unit-variant", default="logei_unit")
    p.add_argument("--bo-pbgi-cost-variant", default="pbgi_cost")
    p.add_argument("--bo-logei-cost-variant", default="logeipc_cost")
    p.add_argument("--grid-size", type=int, default=350)

    p.add_argument("--normalize-x", choices=["final", "none"], default="final")
    p.add_argument(
        "--normalize-y",
        choices=["initial", "task_initial", "bandit_initial_mean", "none"],
        default="bandit_initial_mean",
    )
    p.add_argument("--y-eps", type=float, default=1e-8)
    p.add_argument("--preserve-lrf-bo-x-offset", action="store_true", default=False)
    p.add_argument("--crop-bo-random-init", action="store_true", default=False)

    p.add_argument("--range", choices=["stderr", "std", "none"], default="stderr")
    p.add_argument("--stderr-k", type=float, default=1.0)

    p.add_argument("--include-small-lrf", action="store_true", default=True)
    p.add_argument("--no-include-small-lrf", dest="include_small_lrf", action="store_false")
    p.add_argument("--include-large-lrf", action="store_true", default=True)
    p.add_argument("--no-include-large-lrf", dest="include_large_lrf", action="store_false")

    # LRF warmup crop is consistent with earlier MMLU regret-panel handling.
    p.add_argument("--crop-lrf-warmup", action="store_true", default=True)
    p.add_argument("--no-crop-lrf-warmup", dest="crop_lrf_warmup", action="store_false")
    p.add_argument("--warmup-percentage-default", type=float, default=0.05)

    # Paper-style layout, matching the GSM8K/PIQA figure family.
    p.add_argument("--fig-width", type=float, default=26.8)
    p.add_argument("--fig-height", type=float, default=30.223417)
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
    p.add_argument("--shared-x-label-y", type=float, default=0.270)
    p.add_argument("--shared-y-label-x", type=float, default=-0.023)

    p.add_argument("--y-limit-min", type=float, default=None)
    p.add_argument("--y-limit-max", type=float, default=None)

    # Match GSM8K/PIQA panel geometry exactly so the four MMLU panels share
    # the same aspect ratio and spacing as the paper's main figure.
    p.add_argument("--left", type=float, default=0.045)
    p.add_argument("--right", type=float, default=0.995)
    p.add_argument("--top", type=float, default=0.765)
    p.add_argument("--bottom", type=float, default=0.325)
    p.add_argument("--wspace", type=float, default=0.24)
    p.add_argument("--hspace", type=float, default=0.34)

    p.add_argument("--legend-y", type=float, default=0.055)
    p.add_argument("--legend-ncol", type=int, default=3)
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--show-stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")
    p.add_argument("--stop-alpha", type=float, default=0.12)
    p.add_argument("--stop-line-alpha", type=float, default=0.72)

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
        row_size = row.get("size_bucket")
        size_ok = row_size is None or str(row_size) == size_bucket
        if size_ok and str(row.get("dataset_prior_bucket")) == target_prior:
            tasks.append(task)
    return sorted(tasks)


def tasks_for_size(root: Path) -> set[str]:
    summary_path = root / "runs_summary.csv"
    if not summary_path.is_file():
        return set()
    header = pd.read_csv(summary_path, nrows=0)
    usecols = [c for c in ["mmlu_task", "matrix_task"] if c in header.columns]
    if not usecols:
        return set()
    df = pd.read_csv(summary_path, usecols=usecols)
    tasks: set[str] = set()
    for col in usecols:
        tasks.update(normalize_task_name(str(v)) for v in df[col].dropna().unique() if str(v) and str(v) != "nan")
    return tasks


def normalize_task_name(value: str) -> str:
    s = str(value).replace("\\", "/").rstrip("/")
    return s.split("/")[-1]


def task_history_path(root: Path, task: str) -> Path:
    candidates = [
        root / task / "runs_history.csv.gz",
        root / safe_token(task) / "runs_history.csv.gz",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing runs_history.csv.gz for task={task} under {root}")


def variant_names(
    batch_size: int,
    lrf_batch_size: int,
    scale: str,
    cost_mode: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    b = int(batch_size)
    lb = int(lrf_batch_size)
    token = "cost" if cost_mode in {"aware", "cost"} else "unit"
    return {
        "gittins_data": f"gittins_{token}_B{b}_scale{scale}_dataset",
        "gittins_default": f"gittins_{token}_B{b}_scale{scale}_default",
        "ucb": f"ucb_cost_B{b}" if token == "cost" else f"ucb_B{b}",
        "lrf": f"lrf_cost_B{lb}" if token == "cost" else f"lrf_B{lb}",
        "bo_pbgi_cost" if token == "cost" else "bo_pbgi_unit": str(
            args.bo_pbgi_cost_variant if token == "cost" else args.bo_pbgi_unit_variant
        ),
        "bo_logeipc_cost" if token == "cost" else "bo_logei_unit": str(
            args.bo_logei_cost_variant if token == "cost" else args.bo_logei_unit_variant
        ),
    }


def header_columns(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        header = f.readline().strip()
    return header.split(",")


def read_filtered_history(
    path: Path,
    wanted_variants: set[str],
    x_col: str,
    tasks: set[str] | None = None,
) -> pd.DataFrame:
    available = set(header_columns(path))
    desired = [
        "run_id",
        "experiment_variant",
        "mmlu_task",
        "matrix_task",
        "benchmark_key",
        x_col,
        "cum_eval",
        "simple_regret",
        "n_cells",
        "warmup_percentage",
        "eval_budget_fraction",
        "selection_phase",
        "step_idx",
    ]
    usecols = [c for c in desired if c in available]

    needed = {"run_id", "experiment_variant", x_col, "simple_regret"}
    missing = sorted(needed - set(usecols))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    chunks = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        low_memory=True,
        chunksize=50_000,
    ):
        chunk = chunk[chunk["experiment_variant"].astype(str).isin(wanted_variants)].copy()
        if chunk.empty:
            continue
        if tasks:
            task_mask = pd.Series(False, index=chunk.index)
            if "mmlu_task" in chunk.columns:
                task_mask |= chunk["mmlu_task"].astype(str).map(normalize_task_name).isin(tasks)
            if "matrix_task" in chunk.columns:
                task_mask |= chunk["matrix_task"].astype(str).map(normalize_task_name).isin(tasks)
            if "benchmark_key" in chunk.columns:
                bkey = chunk["benchmark_key"].astype(str)
                for task in tasks:
                    task_mask |= bkey.str.endswith("_" + task) | bkey.str.contains(task, regex=False)
            chunk = chunk[task_mask].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(chunks, ignore_index=True, sort=False)
    for col in [x_col, "cum_eval", "simple_regret", "n_cells", "warmup_percentage", "eval_budget_fraction"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def task_column(df: pd.DataFrame, known_tasks: list[str] | None = None) -> pd.Series:
    if "mmlu_task" in df.columns:
        s = df["mmlu_task"].astype(str).map(normalize_task_name)
        if ((s != "nan") & (s != "")).any():
            return s
    if "matrix_task" in df.columns:
        s = df["matrix_task"].astype(str).map(normalize_task_name)
        if ((s != "nan") & (s != "")).any():
            return s
    if "benchmark_key" in df.columns:
        keys = df["benchmark_key"].astype(str)
        if known_tasks:
            out = pd.Series([""] * len(df), index=df.index, dtype=str)
            for task in known_tasks:
                mask = keys.str.endswith("_" + task) | keys.str.contains(task, regex=False)
                out[mask] = task
            return out
        return keys
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def read_stopping_summary(root: Path, wanted_variants: set[str], tasks: set[str]) -> pd.DataFrame:
    path = root / "runs_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    desired = [
        "run_id",
        "experiment_variant",
        "mmlu_task",
        "matrix_task",
        "gittins_stop_cum_eval",
        "gittins_stop_cum_original_cost",
        "bo_stop_cum_eval",
        "bo_stop_cum_original_cost",
    ]
    usecols = [c for c in desired if c in header.columns]
    if "run_id" not in usecols or "experiment_variant" not in usecols:
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["experiment_variant"].astype(str).isin(wanted_variants)].copy()
    if tasks:
        task_mask = pd.Series(False, index=df.index)
        if "mmlu_task" in df.columns:
            task_mask |= df["mmlu_task"].astype(str).map(normalize_task_name).isin(tasks)
        if "matrix_task" in df.columns:
            task_mask |= df["matrix_task"].astype(str).map(normalize_task_name).isin(tasks)
        df = df[task_mask].copy()
    for col in [
        "gittins_stop_cum_eval",
        "gittins_stop_cum_original_cost",
        "bo_stop_cum_eval",
        "bo_stop_cum_original_cost",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def stop_column_for_kind(kind: str, x_col: str) -> str | None:
    if kind.startswith("bo_"):
        return "bo_stop_cum_eval" if x_col == "cum_eval" else "bo_stop_cum_original_cost"
    if kind in {"gittins_data", "gittins_default"}:
        return "gittins_stop_cum_eval" if x_col == "cum_eval" else "gittins_stop_cum_original_cost"
    return None


def collect_normalized_stops(
    df: pd.DataFrame,
    stop_df: pd.DataFrame,
    *,
    kind: str,
    variant: str,
    x_col: str,
    normalize_x: str,
    preserve_lrf_bo_x_offset: bool,
) -> list[float]:
    stop_col = stop_column_for_kind(kind, x_col)
    if stop_col is None or stop_df.empty or stop_col not in stop_df.columns:
        return []
    sdf = stop_df[stop_df["experiment_variant"].astype(str) == variant].copy()
    if sdf.empty:
        return []
    stops_by_run = pd.to_numeric(sdf.set_index("run_id")[stop_col], errors="coerce")
    out: list[float] = []
    for run_id, rg in df.groupby("run_id", sort=False):
        if run_id not in stops_by_run.index:
            continue
        stop = float(stops_by_run.loc[run_id])
        if not np.isfinite(stop):
            continue
        xs = pd.to_numeric(rg[x_col], errors="coerce").dropna()
        if xs.empty:
            continue
        if normalize_x == "final":
            x1 = float(xs.max())
            if preserve_lrf_bo_x_offset and kind.startswith("bo_"):
                denom = x1
                if not np.isfinite(denom) or denom <= 0:
                    continue
                stop = stop / denom
            else:
                x0 = float(xs.min())
                denom = x1 - x0
                if not np.isfinite(denom) or denom <= 0:
                    continue
                stop = (stop - x0) / denom
            if stop < 0.0 or stop > 1.0:
                continue
        out.append(float(stop))
    return out


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


def run_to_curve(
    rg: pd.DataFrame,
    *,
    kind: str,
    x_col: str,
    y_col: str,
    normalize_x: str,
    normalize_y: str,
    y_denominator: float | None,
    preserve_lrf_bo_x_offset: bool,
    y_eps: float,
    grid_size: int,
    extend_right: bool = False,
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

    if normalize_x == "final":
        x1 = float(x[-1])
        if preserve_lrf_bo_x_offset and (kind == "lrf" or kind.startswith("bo_")):
            denom = x1
            if not np.isfinite(denom) or denom <= 0:
                return None
            x = x / denom
        else:
            x0 = float(x[0])
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
    elif normalize_y in {"task_initial", "bandit_initial_mean"}:
        if y_denominator is None or not np.isfinite(float(y_denominator)):
            return None
        denom_y = max(abs(float(y_denominator)), float(y_eps))
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

    right = float(y[-1]) if extend_right else np.nan
    yi = np.interp(x_grid, x, y, left=np.nan, right=right)
    return yi


def collect_curves_from_task_df(
    df: pd.DataFrame,
    *,
    method_variants: dict[str, str],
    x_col: str,
    grid_size: int,
    normalize_x: str,
    normalize_y: str,
    y_denominator: float | None,
    preserve_lrf_bo_x_offset: bool,
    y_eps: float,
    crop_lrf_warmup: bool,
    crop_bo_random_init: bool,
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

        if kind.startswith("bo_") and crop_bo_random_init:
            sub = bo_post_init_aligned_to_average_start(
                sub,
                x_col=x_col,
                y_col="simple_regret",
            )

        for _, rg in sub.groupby("run_id", sort=False):
            yi = run_to_curve(
                rg,
                kind=kind,
                x_col=x_col,
                y_col="simple_regret",
                normalize_x=normalize_x,
                normalize_y=normalize_y,
                y_denominator=y_denominator,
                preserve_lrf_bo_x_offset=preserve_lrf_bo_x_offset,
                y_eps=y_eps,
                grid_size=grid_size,
                extend_right=kind.startswith("bo_"),
            )
            if yi is not None:
                out[kind].append(yi)

    return out


def compute_task_initial_denominators(
    loaded_sources: list[dict[str, Any]],
    *,
    x_col: str,
    y_eps: float,
) -> dict[str, float]:
    """One denominator per task, shared by all policies/runs for that task."""
    per_task: dict[str, list[float]] = {}
    for source in loaded_sources:
        df = source.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        needed = {"_mmlu_task_for_plot", "run_id", "experiment_variant", x_col, "simple_regret"}
        if not needed.issubset(df.columns):
            continue
        for (_, _, run_id), rg in df.groupby(["_mmlu_task_for_plot", "experiment_variant", "run_id"], sort=False):
            if pd.isna(run_id):
                continue
            gg = rg[[x_col, "simple_regret"]].dropna().sort_values(x_col)
            if gg.empty:
                continue
            task = str(rg["_mmlu_task_for_plot"].iloc[0])
            val = float(gg["simple_regret"].iloc[0])
            if np.isfinite(val):
                per_task.setdefault(task, []).append(abs(val))

    out: dict[str, float] = {}
    for task, vals in per_task.items():
        finite = [float(v) for v in vals if np.isfinite(v) and float(v) > float(y_eps)]
        if finite:
            out[task] = max(finite)
    return out


def compute_bandit_initial_mean_denominators(
    loaded_sources: list[dict[str, Any]],
    *,
    x_col: str,
    y_eps: float,
) -> dict[str, float]:
    """One denominator per task from first simple regret of bandit methods."""
    bandit_kinds = {"gittins_data", "gittins_default", "ucb"}
    per_task: dict[str, list[float]] = {}
    for source in loaded_sources:
        df = source.get("df")
        kinds = set(source.get("kinds", []))
        kind_to_variant = source.get("kind_to_variant", {})
        wanted_variants = {
            str(kind_to_variant[kind])
            for kind in kinds & bandit_kinds
            if kind in kind_to_variant
        }
        if not wanted_variants:
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        needed = {"_mmlu_task_for_plot", "run_id", "experiment_variant", x_col, "simple_regret"}
        if not needed.issubset(df.columns):
            continue
        df = df[df["experiment_variant"].astype(str).isin(wanted_variants)].copy()
        if df.empty:
            continue
        for (_, _, run_id), rg in df.groupby(["_mmlu_task_for_plot", "experiment_variant", "run_id"], sort=False):
            if pd.isna(run_id):
                continue
            gg = rg[[x_col, "simple_regret"]].dropna().sort_values(x_col)
            if gg.empty:
                continue
            task = str(rg["_mmlu_task_for_plot"].iloc[0])
            val = float(gg["simple_regret"].iloc[0])
            if np.isfinite(val) and abs(val) > float(y_eps):
                per_task.setdefault(task, []).append(abs(val))

    out: dict[str, float] = {}
    for task, vals in per_task.items():
        finite = [float(v) for v in vals if np.isfinite(v) and float(v) > float(y_eps)]
        if finite:
            out[task] = float(np.mean(finite))
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


def output_stem(args: argparse.Namespace) -> str:
    stem = (
        f"mmlu_aggregate_2x2_{args.cost_mode}"
        f"_Bsmall{args.small_batch_size}_Blarge{args.large_batch_size}"
        f"_scale{safe_token(str(args.scale))}"
        f"_x{args.normalize_x}_y{args.normalize_y}"
        f"_fast"
    )
    if args.preserve_lrf_bo_x_offset:
        stem += "_lrf_bo_xoffset"
    if args.crop_bo_random_init:
        stem += "_bo_after_init"
    return stem


def plot_group_panel(
    ax: plt.Axes,
    *,
    group_label: str,
    root: Path,
    method_roots: dict[str, Path],
    tasks: list[str],
    method_variants: dict[str, str],
    x_col: str,
    args: argparse.Namespace,
    legend_handles: dict[str, Line2D],
    stats_rows: list[dict[str, Any]],
    stop_rows: list[dict[str, Any]],
) -> None:
    group_curves: dict[str, list[np.ndarray]] = {kind: [] for kind in method_variants}
    group_task_counts: dict[str, int] = {kind: 0 for kind in method_variants}
    group_stops: dict[str, list[float]] = {kind: [] for kind in method_variants}

    print(f"\n[{group_label}] reading {len(tasks)} tasks from {root}")
    t_group = time.time()

    task_set = set(tasks)

    source_to_kinds: dict[Path, list[str]] = {}
    for kind in method_variants:
        source_to_kinds.setdefault(Path(method_roots.get(kind, root)), []).append(kind)

    loaded_sources: list[dict[str, Any]] = []
    for source_root, kinds in source_to_kinds.items():
        wanted = {method_variants[kind] for kind in kinds}
        t0 = time.time()
        try:
            merged_path = source_root / "runs_history.csv.gz"
            stop_df = read_stopping_summary(source_root, wanted, task_set)
            if merged_path.is_file():
                df = read_filtered_history(merged_path, wanted, x_col=x_col, tasks=task_set)
            else:
                pieces = []
                for task in tasks:
                    path = task_history_path(source_root, task)
                    pieces.append(read_filtered_history(path, wanted, x_col=x_col))
                df = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
        except Exception as e:
            labels = ", ".join(STYLE_BY_KIND[kind]["label"] for kind in kinds)
            print(f"  {labels}: SKIP ({e})")
            continue

        if df.empty:
            labels = ", ".join(STYLE_BY_KIND[kind]["label"] for kind in kinds)
            print(f"  {labels}: rows=0; variants={sorted(wanted)}; {time.time() - t0:.1f}s")
            continue

        df["_mmlu_task_for_plot"] = task_column(df, tasks)
        loaded_sources.append(
            {
                "source_root": source_root,
                "kinds": kinds,
                "kind_to_variant": {kind: method_variants[kind] for kind in kinds},
                "df": df,
                "stop_df": stop_df,
                "elapsed": time.time() - t0,
            }
        )

    task_denominators: dict[str, float] = {}
    if args.normalize_y == "task_initial":
        task_denominators = compute_task_initial_denominators(
            loaded_sources,
            x_col=x_col,
            y_eps=float(args.y_eps),
        )
        print(f"  task_initial denominators: {len(task_denominators)}/{len(tasks)} tasks")
    elif args.normalize_y == "bandit_initial_mean":
        task_denominators = compute_bandit_initial_mean_denominators(
            loaded_sources,
            x_col=x_col,
            y_eps=float(args.y_eps),
        )
        print(f"  bandit_initial_mean denominators: {len(task_denominators)}/{len(tasks)} tasks")

    for source in loaded_sources:
        source_root = source["source_root"]
        kinds = source["kinds"]
        df = source["df"]
        stop_df = source["stop_df"]
        source_msgs = []
        for kind in kinds:
            variant = method_variants[kind]
            used_tasks = 0
            total_curves = 0
            kind_df = df[df["experiment_variant"].astype(str) == variant].copy()
            group_stops[kind].extend(
                collect_normalized_stops(
                    kind_df,
                    stop_df,
                    kind=kind,
                    variant=variant,
                    x_col=x_col,
                    normalize_x=args.normalize_x,
                    preserve_lrf_bo_x_offset=bool(args.preserve_lrf_bo_x_offset),
                )
            )
            for task in tasks:
                task_df = kind_df[kind_df["_mmlu_task_for_plot"].astype(str) == task].copy()
                if task_df.empty:
                    continue
                task_curves = collect_curves_from_task_df(
                    task_df,
                    method_variants={kind: variant},
                    x_col=x_col,
                    grid_size=int(args.grid_size),
                    normalize_x=args.normalize_x,
                    normalize_y=args.normalize_y,
                    y_denominator=task_denominators.get(task)
                    if args.normalize_y in {"task_initial", "bandit_initial_mean"}
                    else None,
                    preserve_lrf_bo_x_offset=bool(args.preserve_lrf_bo_x_offset),
                    y_eps=float(args.y_eps),
                    crop_lrf_warmup=bool(args.crop_lrf_warmup),
                    crop_bo_random_init=bool(args.crop_bo_random_init),
                    warmup_default=float(args.warmup_percentage_default),
                )
                curves = task_curves.get(kind, [])
                if curves:
                    group_curves[kind].extend(curves)
                    used_tasks += 1
                    total_curves += len(curves)
            group_task_counts[kind] = used_tasks
            source_msgs.append(
                f"{STYLE_BY_KIND[kind]['label']}: tasks={used_tasks}/{len(tasks)}, "
                f"curves={total_curves}, variant={variant}"
            )

        print(f"  {source_root}: rows={len(df):,}; " + "; ".join(source_msgs) + f"; {source['elapsed']:.1f}s")
    del loaded_sources

    print(f"[{group_label}] finished in {time.time() - t_group:.1f}s")

    for kind in [
        "gittins_data",
        "gittins_default",
        "ucb",
        "lrf",
        "bo_pbgi_unit",
        "bo_logei_unit",
        "bo_pbgi_cost",
        "bo_logeipc_cost",
    ]:
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

    if args.show_stopping:
        for kind in ["gittins_data", "gittins_default", "bo_pbgi_unit", "bo_logei_unit", "bo_pbgi_cost", "bo_logeipc_cost"]:
            if kind not in method_variants:
                continue
            vals = np.asarray(group_stops.get(kind, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                print(f"WARNING: no stopping values for {group_label} / {STYLE_BY_KIND[kind]['label']}")
                continue
            mean_stop = float(np.mean(vals))
            if vals.size > 1:
                band = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) * float(args.stderr_k)
            else:
                band = 0.0
            style = STYLE_BY_KIND[kind]
            stop_rows.append(
                {
                    "group": group_label,
                    "method_kind": kind,
                    "method": style["label"],
                    "variant": method_variants[kind],
                    "x": mean_stop,
                    "band": band,
                    "n_stops": int(vals.size),
                }
            )
            if band > 0:
                ax.axvspan(
                    mean_stop - band,
                    mean_stop + band,
                    color=style["color"],
                    alpha=float(args.stop_alpha),
                    linewidth=0,
                    zorder=1,
                )
            extra = float(GITTINS_LINE_EXTRA_MULT) if kind in ("gittins_data", "gittins_default") else 1.0
            ax.axvline(
                mean_stop,
                color=style["color"],
                linestyle="--",
                linewidth=2.6 * float(LINEWIDTH_MULT) * extra,
                alpha=float(args.stop_line_alpha),
                zorder=2,
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


def draw_cached_panel(
    ax: plt.Axes,
    *,
    group_label: str,
    curve_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    args: argparse.Namespace,
    legend_handles: dict[str, Line2D],
) -> None:
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
    group_df = curve_df[curve_df["group"].astype(str) == group_label]
    for kind in order:
        kd = group_df[group_df["method_kind"].astype(str) == kind].sort_values("x")
        if kd.empty:
            continue
        style = STYLE_BY_KIND[kind]
        extra = float(GITTINS_LINE_EXTRA_MULT) if kind in ("gittins_data", "gittins_default") else 1.0
        x = kd["x"].to_numpy(float)
        mean = kd["mean"].to_numpy(float)
        line = ax.plot(
            x,
            mean,
            color=style["color"],
            linewidth=float(style["linewidth"]) * float(LINEWIDTH_MULT) * extra,
            label=style["label"],
            zorder=style["zorder"],
        )[0]
        legend_handles[kind] = line

        if args.range != "none":
            band_col = "stderr" if args.range == "stderr" else "std"
            if band_col in kd.columns:
                band = kd[band_col].to_numpy(float)
                if args.range == "stderr":
                    band = band * float(args.stderr_k)
                ax.fill_between(
                    x,
                    mean - band,
                    mean + band,
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0,
                    zorder=style["zorder"] - 0.5,
                )

    if args.show_stopping and not stop_df.empty:
        group_stops = stop_df[stop_df["group"].astype(str) == group_label]
        for _, row in group_stops.iterrows():
            kind = str(row["method_kind"])
            if kind not in STYLE_BY_KIND:
                continue
            x_stop = float(row["x"])
            band = float(row.get("band", 0.0))
            style = STYLE_BY_KIND[kind]
            if np.isfinite(band) and band > 0:
                ax.axvspan(
                    x_stop - band,
                    x_stop + band,
                    color=style["color"],
                    alpha=float(args.stop_alpha),
                    linewidth=0,
                    zorder=1,
                )
            extra = float(GITTINS_LINE_EXTRA_MULT) if kind in ("gittins_data", "gittins_default") else 1.0
            ax.axvline(
                x_stop,
                color=style["color"],
                linestyle="--",
                linewidth=2.6 * float(LINEWIDTH_MULT) * extra,
                alpha=float(args.stop_line_alpha),
                zorder=2,
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


def finalize_figure(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    args: argparse.Namespace,
    legend_handles: dict[str, Line2D],
    out_png: Path | None,
    out_pdf: Path | None,
) -> None:
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

    legend_order = [
        "gittins_data",
        "gittins_default",
        "ucb",
        "lrf",
        "bo_pbgi_unit",
        "bo_logei_unit",
        "bo_pbgi_cost",
        "bo_logeipc_cost",
    ]
    final_handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    final_labels = [STYLE_BY_KIND[k]["label"] for k in legend_order if k in legend_handles]

    if args.show_stopping:
        for kind in [
            "gittins_data",
            "gittins_default",
            "bo_pbgi_unit",
            "bo_logei_unit",
            "bo_pbgi_cost",
            "bo_logeipc_cost",
        ]:
            if kind not in legend_handles:
                continue
            final_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=STYLE_BY_KIND[kind]["color"],
                    linestyle="--",
                    linewidth=2.6
                    * float(LINEWIDTH_MULT)
                    * (float(GITTINS_LINE_EXTRA_MULT) if kind in ("gittins_data", "gittins_default") else 1.0),
                    alpha=float(args.stop_line_alpha),
                )
            )
            final_labels.append(f"{STYLE_BY_KIND[kind]['label']} mean stop")

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

    if out_png is not None and out_pdf is not None:
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.22)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.22)
        print(f"Wrote:\n  {out_png}\n  {out_pdf}")


def plot_from_cache(args: argparse.Namespace) -> int:
    stem = output_stem(args)
    curve_path = args.out_dir / f"{stem}_aggregated.csv"
    stop_path = args.out_dir / f"{stem}_stops.csv"
    if not curve_path.is_file():
        raise FileNotFoundError(f"missing cache curve CSV: {curve_path}")

    curve_df = pd.read_csv(curve_path)
    stop_df = pd.read_csv(stop_path) if stop_path.is_file() else pd.DataFrame()
    print(f"[PLOT-CACHE] reading {curve_path}")
    if stop_path.is_file():
        print(f"[PLOT-CACHE] reading {stop_path}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(float(args.fig_width), float(args.fig_height)),
        sharey=False,
        constrained_layout=False,
    )
    legend_handles: dict[str, Line2D] = {}
    row_labels = ["Easy", "Hard"]
    col_labels = ["Small", "Large"]
    for r, difficulty in enumerate(row_labels):
        for c, size in enumerate(col_labels):
            group_label = f"{difficulty}-{size}"
            draw_cached_panel(
                axes[r, c],
                group_label=group_label,
                curve_df=curve_df,
                stop_df=stop_df,
                args=args,
                legend_handles=legend_handles,
            )
            if r == 0:
                axes[r, c].set_title(size, fontsize=args.title_size, fontweight="normal", pad=8)

    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"
    finalize_figure(fig, axes, args=args, legend_handles=legend_handles, out_png=out_png, out_pdf=out_pdf)
    plt.close(fig)
    return 0


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib(args)

    if args.plot_from_cache:
        return plot_from_cache(args)

    metadata = load_task_metadata(args.task_metadata)

    group_defs = {
        ("Easy", "Small"): {
            "root": args.small_root,
            "lrf_root": args.small_lrf_root,
            "bo_root": args.small_bo_root,
            "size_bucket": "small",
            "batch_size": int(args.small_batch_size),
            "lrf_batch_size": int(args.small_lrf_batch_size),
            "include_lrf": bool(args.include_small_lrf),
        },
        ("Easy", "Large"): {
            "root": args.large_root,
            "lrf_root": args.large_lrf_root,
            "bo_root": args.large_bo_root,
            "size_bucket": "large",
            "batch_size": int(args.large_batch_size),
            "lrf_batch_size": int(args.large_lrf_batch_size),
            "include_lrf": bool(args.include_large_lrf),
        },
        ("Hard", "Small"): {
            "root": args.small_root,
            "lrf_root": args.small_lrf_root,
            "bo_root": args.small_bo_root,
            "size_bucket": "small",
            "batch_size": int(args.small_batch_size),
            "lrf_batch_size": int(args.small_lrf_batch_size),
            "include_lrf": bool(args.include_small_lrf),
        },
        ("Hard", "Large"): {
            "root": args.large_root,
            "lrf_root": args.large_lrf_root,
            "bo_root": args.large_bo_root,
            "size_bucket": "large",
            "batch_size": int(args.large_batch_size),
            "lrf_batch_size": int(args.large_lrf_batch_size),
            "include_lrf": bool(args.include_large_lrf),
        },
    }

    selected_tasks: dict[str, list[str]] = {}
    tasks_by_size = {
        "small": tasks_for_size(args.small_root),
        "large": tasks_for_size(args.large_root),
    }
    for (difficulty, size), cfg in group_defs.items():
        difficulty_tasks = tasks_for_group(
            metadata,
            size_bucket=str(cfg["size_bucket"]),
            difficulty=difficulty,
        )
        size_tasks = tasks_by_size.get(str(cfg["size_bucket"]), set())
        tasks = sorted([t for t in difficulty_tasks if not size_tasks or t in size_tasks])
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
    stop_rows: list[dict[str, Any]] = []

    for r, difficulty in enumerate(row_labels):
        for c, size in enumerate(col_labels):
            cfg = group_defs[(difficulty, size)]
            batch_size = int(cfg["batch_size"])
            method_variants = variant_names(
                batch_size,
                int(cfg["lrf_batch_size"]),
                str(args.scale),
                str(args.cost_mode),
                args,
            )
            if not bool(cfg["include_lrf"]):
                method_variants.pop("lrf", None)
            method_roots = {
                "gittins_data": Path(cfg["root"]),
                "gittins_default": Path(cfg["root"]),
                "ucb": Path(cfg["root"]),
                "lrf": Path(cfg["lrf_root"]),
                "bo_pbgi_unit": Path(cfg["bo_root"]),
                "bo_logei_unit": Path(cfg["bo_root"]),
                "bo_pbgi_cost": Path(cfg["bo_root"]),
                "bo_logeipc_cost": Path(cfg["bo_root"]),
            }

            group_label = f"{difficulty}-{size}"
            plot_group_panel(
                axes[r, c],
                group_label=group_label,
                root=Path(cfg["root"]),
                method_roots=method_roots,
                tasks=selected_tasks[group_label],
                method_variants=method_variants,
                x_col=x_col,
                args=args,
                legend_handles=legend_handles,
                stats_rows=stats_rows,
                stop_rows=stop_rows,
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

    legend_order = [
        "gittins_data",
        "gittins_default",
        "ucb",
        "lrf",
        "bo_pbgi_unit",
        "bo_logei_unit",
        "bo_pbgi_cost",
        "bo_logeipc_cost",
    ]
    final_handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    final_labels = [STYLE_BY_KIND[k]["label"] for k in legend_order if k in legend_handles]

    if args.show_stopping:
        for kind in [
            "gittins_data",
            "gittins_default",
            "bo_pbgi_unit",
            "bo_logei_unit",
            "bo_pbgi_cost",
            "bo_logeipc_cost",
        ]:
            if kind not in legend_handles:
                continue
            final_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=STYLE_BY_KIND[kind]["color"],
                    linestyle="--",
                    linewidth=2.6
                    * float(LINEWIDTH_MULT)
                    * (float(GITTINS_LINE_EXTRA_MULT) if kind in ("gittins_data", "gittins_default") else 1.0),
                    alpha=float(args.stop_line_alpha),
                )
            )
            final_labels.append(f"{STYLE_BY_KIND[kind]['label']} mean stop")

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

    stem = output_stem(args)
    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"
    out_csv = args.out_dir / f"{stem}_aggregated.csv"
    out_stops = args.out_dir / f"{stem}_stops.csv"
    out_meta = args.out_dir / f"{stem}_groups.json"

    if not args.cache_curves_only:
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.22)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(out_csv, index=False)
    pd.DataFrame(stop_rows).to_csv(out_stops, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "cost_mode": args.cost_mode,
                "normalize_x": args.normalize_x,
                "normalize_y": args.normalize_y,
                "small_batch_size": args.small_batch_size,
                "large_batch_size": args.large_batch_size,
                "small_lrf_batch_size": args.small_lrf_batch_size,
                "large_lrf_batch_size": args.large_lrf_batch_size,
                "bo_pbgi_unit_variant": args.bo_pbgi_unit_variant,
                "bo_logei_unit_variant": args.bo_logei_unit_variant,
                "bo_pbgi_cost_variant": args.bo_pbgi_cost_variant,
                "bo_logei_cost_variant": args.bo_logei_cost_variant,
                "scale": args.scale,
                "selected_tasks": selected_tasks,
                "note": "high prior bucket is plotted as Easy; low prior bucket is plotted as Hard; medium prior bucket is omitted. LRF aggregates use available task/run curves and allow incomplete coverage.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.cache_curves_only:
        print(f"\nWrote {out_png}")
        print(f"Wrote {out_pdf}")
    print(f"[CACHE] wrote {out_csv}")
    print(f"[CACHE] wrote {out_stops}")
    print(f"[CACHE] wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
