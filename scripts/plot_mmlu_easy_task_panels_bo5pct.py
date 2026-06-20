#!/usr/bin/env python3
"""Plot MMLU easy-subject task panels with 5% BO post-init curves.

This script is intentionally scoped to the paper appendix easy-subject grids:
it first writes fixed-size individual task panels, then assembles them into the
unit-cost and cost-aware grid figures.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_mmlu_aggregate_2x2_normalized_fast_shared_labels import (
    normalize_task_name,
    read_filtered_history,
    task_column,
)


TASKS_BY_BUCKET: dict[str, list[str]] = {
    "easy": [
    "computer_security",
    "high_school_biology",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_psychology",
    "high_school_us_history",
    "high_school_world_history",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "management",
    "marketing",
    "miscellaneous",
    "sociology",
    "us_foreign_policy",
    "world_religions",
    ],
    "medium": [
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_medicine",
        "conceptual_physics",
        "electrical_engineering",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "human_aging",
        "medical_genetics",
        "moral_disputes",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
    ],
    "hard": [
        "abstract_algebra",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "econometrics",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_chemistry",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
        "moral_scenarios",
        "professional_accounting",
        "professional_law",
        "virology",
    ],
}

TASK_DISPLAY: dict[str, str] = {
    "computer_security": "Comp Security",
    "high_school_biology": "HS Biology",
    "high_school_geography": "HS Geography",
    "high_school_government_and_politics": "HS Gov & Pol",
    "high_school_psychology": "HS Psychology",
    "high_school_us_history": "HS US Hist",
    "high_school_world_history": "HS World Hist",
    "human_sexuality": "Human Sexuality",
    "international_law": "Int'l Law",
    "jurisprudence": "Jurisprudence",
    "logical_fallacies": "Logical Fallacies",
    "management": "Management",
    "marketing": "Marketing",
    "miscellaneous": "Misc",
    "sociology": "Sociology",
    "us_foreign_policy": "US Foreign Policy",
    "world_religions": "World Religions",
    "anatomy": "Anatomy",
    "astronomy": "Astronomy",
    "business_ethics": "Business Ethics",
    "clinical_knowledge": "Clinical Know.",
    "college_biology": "College Biology",
    "college_medicine": "College Medicine",
    "conceptual_physics": "Conceptual Phys",
    "electrical_engineering": "Electrical Eng",
    "high_school_computer_science": "HS CS",
    "high_school_european_history": "HS Euro Hist",
    "high_school_macroeconomics": "HS Macroecon",
    "high_school_microeconomics": "HS Microecon",
    "human_aging": "Human Aging",
    "medical_genetics": "Medical Genetics",
    "moral_disputes": "Moral Disputes",
    "nutrition": "Nutrition",
    "philosophy": "Philosophy",
    "prehistory": "Prehistory",
    "professional_medicine": "Prof Medicine",
    "professional_psychology": "Prof Psychology",
    "public_relations": "Public Relations",
    "security_studies": "Security Studies",
    "abstract_algebra": "Abstract Alg.",
    "college_chemistry": "College Chem",
    "college_computer_science": "College CS",
    "college_mathematics": "College Math",
    "college_physics": "College Physics",
    "econometrics": "Econometrics",
    "elementary_mathematics": "Elementary Math",
    "formal_logic": "Formal Logic",
    "global_facts": "Global Facts",
    "high_school_chemistry": "HS Chem",
    "high_school_mathematics": "HS Math",
    "high_school_physics": "HS Physics",
    "high_school_statistics": "HS Statistics",
    "machine_learning": "Machine Learning",
    "moral_scenarios": "Moral Scenarios",
    "professional_accounting": "Prof Accounting",
    "professional_law": "Prof Law",
    "virology": "Virology",
}

TASK_SIZE: dict[str, str] = {
    "computer_security": "S",
    "human_sexuality": "S",
    "international_law": "S",
    "jurisprudence": "S",
    "management": "S",
    "us_foreign_policy": "S",
    "high_school_biology": "M",
    "high_school_geography": "M",
    "high_school_government_and_politics": "M",
    "high_school_us_history": "M",
    "high_school_world_history": "M",
    "logical_fallacies": "M",
    "marketing": "M",
    "sociology": "M",
    "world_religions": "M",
    "high_school_psychology": "L",
    "miscellaneous": "L",
    "anatomy": "S",
    "astronomy": "M",
    "business_ethics": "S",
    "clinical_knowledge": "M",
    "college_biology": "S",
    "college_medicine": "M",
    "conceptual_physics": "M",
    "electrical_engineering": "S",
    "high_school_computer_science": "S",
    "high_school_european_history": "M",
    "high_school_macroeconomics": "M",
    "high_school_microeconomics": "M",
    "human_aging": "M",
    "medical_genetics": "S",
    "moral_disputes": "M",
    "nutrition": "M",
    "philosophy": "M",
    "prehistory": "M",
    "professional_medicine": "M",
    "professional_psychology": "L",
    "public_relations": "S",
    "security_studies": "M",
    "abstract_algebra": "S",
    "college_chemistry": "S",
    "college_computer_science": "S",
    "college_mathematics": "S",
    "college_physics": "S",
    "econometrics": "S",
    "elementary_mathematics": "M",
    "formal_logic": "S",
    "global_facts": "S",
    "high_school_chemistry": "M",
    "high_school_mathematics": "M",
    "high_school_physics": "M",
    "high_school_statistics": "M",
    "machine_learning": "S",
    "moral_scenarios": "L",
    "professional_accounting": "M",
    "professional_law": "L",
    "virology": "M",
}

TASK_GROUP: dict[str, str] = {
    task: ("small" if size == "S" else "medium" if size == "M" else "large")
    for task, size in TASK_SIZE.items()
}

GROUP_BATCH: dict[str, int] = {"small": 2, "medium": 4, "large": 8}

COLOR_UCB = "tab:blue"
COLOR_LRF = "tab:purple"
COLOR_GITTINS_S = "tab:orange"
COLOR_GITTINS_G = "tab:green"
COLOR_BO_PBGI = "tab:olive"
COLOR_BO_LOGEI = "tab:brown"

STYLE_BY_KIND = {
    "gittins_data": {"color": COLOR_GITTINS_S, "label": "Gittins-S", "lw": 2.4, "z": 6},
    "gittins_default": {"color": COLOR_GITTINS_G, "label": "Gittins-G", "lw": 2.4, "z": 5},
    "ucb": {"color": COLOR_UCB, "label": "UCB-E", "lw": 1.8, "z": 4},
    "lrf": {"color": COLOR_LRF, "label": "LRF", "lw": 1.8, "z": 3},
    "bo_pbgi": {"color": COLOR_BO_PBGI, "label": "BO-PBGI", "lw": 1.9, "z": 3.5},
    "bo_logei": {"color": COLOR_BO_LOGEI, "label": "BO-LogEI", "lw": 1.9, "z": 3.4},
}


def clean_tick_label(value: float, _pos: int) -> str:
    if abs(float(value)) < 1e-12:
        return "0"
    return f"{float(value):.2f}"


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=Path(r"outputs\wandb_plots_new\paper_figures\mmlu_easy_bo5pct_panels"))
    p.add_argument("--prior-bucket", choices=sorted(TASKS_BY_BUCKET), default="easy")
    p.add_argument("--cost-mode", choices=["both", "unit", "aware"], default="both")
    p.add_argument("--assemble-only", action="store_true", help="Reuse existing individual panels and only reassemble grids.")
    p.add_argument("--individual-only", action="store_true", help="Write individual task panels and skip grid assembly.")
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--grid-size", type=int, default=320)
    p.add_argument("--se-mult", type=float, default=2.0)
    p.add_argument("--range", choices=["stderr", "none"], default="stderr")
    p.add_argument("--curve-alpha", type=float, default=0.15)
    p.add_argument("--show-stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")
    p.add_argument("--stop-alpha", type=float, default=0.12)
    p.add_argument("--stop-line-alpha", type=float, default=0.72)
    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--single-width", type=float, default=2.25)
    p.add_argument("--single-height", type=float, default=1.65)
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--title-size", type=float, default=14.5)
    p.add_argument("--tick-size", type=float, default=14)
    p.add_argument("--shared-label-size", type=float, default=35)
    p.add_argument("--legend-size", type=float, default=30)
    p.add_argument("--legend-ncol", type=int, default=4)
    p.add_argument("--fig-width", type=float, default=18.0)
    p.add_argument("--grid-cols", type=int, default=4)
    p.add_argument("--left", type=float, default=0.058)
    p.add_argument("--right", type=float, default=0.985)
    p.add_argument("--top", type=float, default=0.965)
    p.add_argument("--shared-y-label-x", type=float, default=0.042)
    p.add_argument("--shared-x-label-pad-in", type=float, default=0.48)
    p.add_argument("--legend-pad-in", type=float, default=0.95)
    p.add_argument("--bottom-floor-in", type=float, default=0.75)
    p.add_argument("--shared-x-label-x-offset", type=float, default=0.01)
    p.add_argument("--shared-y-label-y-offset", type=float, default=0.01)
    return p.parse_args()


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.family": str(args.font_family),
        "font.serif": [str(args.font_family), "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def variants_for_group(group: str, mode: str, scale: str) -> dict[str, str]:
    batch = GROUP_BATCH[group]
    if mode == "unit":
        return {
            "gittins_data": f"gittins_unit_B{batch}_scale{scale}_dataset",
            "gittins_default": f"gittins_unit_B{batch}_scale{scale}_default",
            "ucb": f"ucb_B{batch}",
            "lrf": "lrf_B32",
            "bo_pbgi": "pbgi_unit",
            "bo_logei": "logei_unit",
        }
    return {
        "gittins_data": f"gittins_cost_B{batch}_scale{scale}_dataset",
        "gittins_default": f"gittins_cost_B{batch}_scale{scale}_default",
        "ucb": f"ucb_cost_B{batch}",
        "lrf": "lrf_cost_B32",
        "bo_pbgi": "pbgi_cost",
        "bo_logei": "logeipc_cost",
    }


def roots_for_group(group: str) -> dict[str, Path]:
    return {
        "bandit": Path(rf"outputs\wandb_downloads_new\ucb_gittins\mmlu_{group}"),
        "lrf": Path(rf"outputs\wandb_downloads_new\lrf\mmlu_{group}_lrf"),
        "bo": Path(rf"outputs\wandb_downloads_new\bo_baseline_5pct\mmlu_{group}_bo"),
    }


def lrf_warmup_evals(g: pd.DataFrame) -> float:
    warmup = pd.to_numeric(g.get("warmup_percentage"), errors="coerce").dropna()
    warmup_value = float(warmup.iloc[0]) if not warmup.empty else 0.05
    n_cells = pd.to_numeric(g.get("n_cells"), errors="coerce").dropna()
    if not n_cells.empty and float(n_cells.iloc[0]) > 0:
        return float(math.ceil(warmup_value * float(n_cells.iloc[0])))
    return float("nan")


def crop_lrf_after_warmup(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for variant, g in df.groupby("experiment_variant", sort=False):
        if "lrf" in str(variant).lower():
            warmup = lrf_warmup_evals(g)
            if math.isfinite(warmup) and "cum_eval" in g.columns:
                g = g[pd.to_numeric(g["cum_eval"], errors="coerce") >= warmup]
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else df


def crop_bo_after_random_init(df: pd.DataFrame) -> pd.DataFrame:
    if "selection_phase" not in df.columns:
        return df
    is_bo = df["experiment_variant"].astype(str).isin({"pbgi_unit", "logei_unit", "pbgi_cost", "logeipc_cost"})
    keep = (~is_bo) | (df["selection_phase"].astype(str).str.lower() != "random_init")
    return df[keep].copy()


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


def aggregate_runs_to_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    grid_size: int,
    *,
    extend_right: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    runs = []
    min_x = float("inf")
    max_x = -float("inf")
    for _, g in df.groupby("run_id", sort=False):
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
    y_arr = np.vstack([
        np.interp(x_grid, x, y, left=np.nan, right=float(y[-1]) if extend_right else np.nan)
        for x, y in runs
    ])
    return x_grid, y_arr


def band_from_yarr(y_arr: np.ndarray, se_mult: float, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y_arr.size == 0:
        empty = np.array([])
        return empty, empty, empty
    mean = np.nanmean(y_arr, axis=0)
    if mode == "none":
        return mean, mean, mean
    std = np.nanstd(y_arr, axis=0)
    n = np.sum(~np.isnan(y_arr), axis=0)
    band = float(se_mult) * (std / np.sqrt(np.maximum(n, 1)))
    return mean, mean - band, mean + band


def load_group_mode(group: str, mode: str, tasks: list[str], args: argparse.Namespace) -> pd.DataFrame:
    roots = roots_for_group(group)
    variants = variants_for_group(group, mode, args.scale)
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"

    pieces: list[pd.DataFrame] = []
    source_variants = {
        roots["bandit"]: {variants["gittins_data"], variants["gittins_default"], variants["ucb"]},
        roots["lrf"]: {variants["lrf"]},
        roots["bo"]: {variants["bo_pbgi"], variants["bo_logei"]},
    }
    for root, wanted in source_variants.items():
        path = root / "runs_history.csv.gz"
        if not path.is_file():
            print(f"SKIP missing {path}")
            continue
        print(f"Reading {group}/{mode}: {path} variants={sorted(wanted)} tasks={len(tasks)}")
        df = read_filtered_history(path, wanted, x_col=x_col, tasks=set(tasks))
        if df.empty:
            print(f"  rows=0 for {path}")
            continue
        df["_mmlu_task_for_plot"] = task_column(df, tasks)
        pieces.append(df)
        print(f"  rows={len(df):,}")

    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True, sort=False)
    df = crop_lrf_after_warmup(df)
    return df


def stop_column_for_kind(kind: str, x_col: str) -> str | None:
    if kind.startswith("bo_"):
        return "bo_stop_cum_eval" if x_col == "cum_eval" else "bo_stop_cum_original_cost"
    if kind in {"gittins_data", "gittins_default"}:
        return "gittins_stop_cum_eval" if x_col == "cum_eval" else "gittins_stop_cum_original_cost"
    return None


def load_stop_summary_group_mode(group: str, mode: str, tasks: list[str], args: argparse.Namespace) -> pd.DataFrame:
    roots = roots_for_group(group)
    variants = variants_for_group(group, mode, args.scale)
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"
    stop_kinds = ["gittins_data", "gittins_default", "bo_pbgi", "bo_logei"]

    pieces: list[pd.DataFrame] = []
    source_variants = {
        roots["bandit"] / "runs_summary.csv": {variants["gittins_data"], variants["gittins_default"]},
        roots["bo"] / "runs_summary.csv": {variants["bo_pbgi"], variants["bo_logei"]},
    }
    desired_base = [
        "run_id",
        "experiment_variant",
        "mmlu_task",
        "matrix_task",
        "benchmark_key",
    ]
    desired_stop = sorted({c for kind in stop_kinds if (c := stop_column_for_kind(kind, x_col))})
    known_tasks = set(tasks)

    for path, wanted in source_variants.items():
        if not path.is_file():
            print(f"SKIP missing summary {path}")
            continue
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in desired_base + desired_stop if c in header.columns]
        if "experiment_variant" not in usecols:
            continue
        df = pd.read_csv(path, usecols=usecols)
        df = df[df["experiment_variant"].astype(str).isin(wanted)].copy()
        if df.empty:
            continue

        task_mask = pd.Series(False, index=df.index)
        if "mmlu_task" in df.columns:
            task_mask |= df["mmlu_task"].astype(str).map(normalize_task_name).isin(known_tasks)
        if "matrix_task" in df.columns:
            task_mask |= df["matrix_task"].astype(str).map(normalize_task_name).isin(known_tasks)
        if "benchmark_key" in df.columns:
            bkey = df["benchmark_key"].astype(str)
            for task in known_tasks:
                task_mask |= bkey.str.endswith("_" + task) | bkey.str.contains(task, regex=False)
        df = df[task_mask].copy()
        if df.empty:
            continue

        df["_mmlu_task_for_plot"] = task_column(df, tasks)
        for col in desired_stop:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        pieces.append(df)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True, sort=False)


def draw_stopping(
    ax: plt.Axes,
    task_stop_df: pd.DataFrame,
    variants: dict[str, str],
    x_col: str,
    args: argparse.Namespace,
) -> None:
    if not bool(args.show_stopping) or task_stop_df.empty:
        return

    for kind in ["gittins_data", "gittins_default", "bo_pbgi", "bo_logei"]:
        stop_col = stop_column_for_kind(kind, x_col)
        if stop_col is None or stop_col not in task_stop_df.columns:
            continue
        variant = variants[kind]
        vals = pd.to_numeric(
            task_stop_df.loc[task_stop_df["experiment_variant"].astype(str) == variant, stop_col],
            errors="coerce",
        )
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        if vals.empty:
            continue

        arr = vals.to_numpy(dtype=float)
        mean_stop = float(np.mean(arr))
        if not np.isfinite(mean_stop):
            continue

        color = STYLE_BY_KIND[kind]["color"]
        if len(arr) > 1:
            band = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) * float(args.se_mult)
            lo = mean_stop - band
            hi = mean_stop + band
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.axvspan(lo, hi, color=color, alpha=float(args.stop_alpha), linewidth=0, zorder=1)

        ax.axvline(
            mean_stop,
            color=color,
            linestyle="--",
            linewidth=max(1.1, float(STYLE_BY_KIND[kind]["lw"]) * 0.9),
            alpha=float(args.stop_line_alpha),
            zorder=2,
        )


def panel_path(args: argparse.Namespace, task: str, mode: str) -> Path:
    group = TASK_GROUP[task]
    batch = GROUP_BATCH[group]
    return args.out_root / "individual" / mode / f"{safe_token(task)}_{mode}_B{batch}_scale{safe_token(args.scale)}.png"


def plot_panel(task: str, mode: str, df: pd.DataFrame, stop_df: pd.DataFrame, args: argparse.Namespace) -> Path:
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"
    fig, ax = plt.subplots(figsize=(args.single_width, args.single_height))
    group = TASK_GROUP[task]
    variants = variants_for_group(group, mode, args.scale)
    task_df = df[df["_mmlu_task_for_plot"].astype(str) == task].copy()
    task_stop_df = stop_df[stop_df["_mmlu_task_for_plot"].astype(str) == task].copy() if not stop_df.empty else stop_df

    for kind in ["gittins_data", "gittins_default", "ucb", "lrf", "bo_pbgi", "bo_logei"]:
        variant = variants[kind]
        vg = task_df[task_df["experiment_variant"].astype(str) == variant].copy()
        if vg.empty:
            continue
        if kind.startswith("bo_"):
            vg = bo_post_init_aligned_to_average_start(
                vg,
                x_col=x_col,
                y_col="simple_regret",
            )
            if vg.empty:
                continue
        x_grid, y_arr = aggregate_runs_to_grid(
            vg,
            x_col,
            "simple_regret",
            int(args.grid_size),
            extend_right=kind.startswith("bo_"),
        )
        if x_grid.size == 0:
            continue
        center, lo, hi = band_from_yarr(y_arr, float(args.se_mult), args.range)
        style = STYLE_BY_KIND[kind]
        ax.plot(x_grid, center, color=style["color"], linewidth=style["lw"], zorder=style["z"])
        if args.range != "none":
            ax.fill_between(x_grid, lo, hi, color=style["color"], alpha=float(args.curve_alpha), linewidth=0, zorder=style["z"] - 0.5)

    draw_stopping(ax, task_stop_df, variants, x_col, args)

    ax.set_title(f"{TASK_DISPLAY[task]} ({TASK_SIZE[task]})", fontsize=float(args.title_size), pad=3)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=float(args.tick_size), width=0.9, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FuncFormatter(clean_tick_label))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_tick_label))
    x_left, x_right = ax.get_xlim()
    ax.set_xlim(x_left, x_right + 0.08 * (x_right - x_left))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.17, top=0.82)

    out = panel_path(args, task, mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi))
    plt.close(fig)
    return out


def legend_handles_labels(args: argparse.Namespace, mode: str) -> tuple[list[object], list[str]]:
    kinds = ["gittins_data", "gittins_default", "ucb", "lrf", "bo_pbgi", "bo_logei"]
    handles: list[object] = [
        Line2D([0], [0], color=STYLE_BY_KIND[k]["color"], linewidth=STYLE_BY_KIND[k]["lw"])
        for k in kinds
    ]
    labels = [STYLE_BY_KIND[k]["label"] for k in kinds]
    if mode == "aware":
        labels = ["BO-LogEIPC" if label == "BO-LogEI" else label for label in labels]
    if args.range != "none":
        handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.18))
        labels.append(f"\u00b1{args.se_mult:g} SE band")
    if args.show_stopping:
        handles.append(Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.9))
        labels.append("Mean stop")
    return handles, labels


def selected_tasks(args: argparse.Namespace) -> list[str]:
    return TASKS_BY_BUCKET[str(args.prior_bucket)]


def assemble_grid(mode: str, args: argparse.Namespace) -> Path:
    tasks = selected_tasks(args)
    n = len(tasks)
    ncols = int(args.grid_cols)
    nrows = int(math.ceil(n / ncols))
    panel_aspect = float(args.single_width) / float(args.single_height)
    panel_area_width_in = (float(args.right) - float(args.left)) * float(args.fig_width)
    row_height_in = (panel_area_width_in / ncols) / panel_aspect
    panel_area_height_in = row_height_in * nrows
    bottom_in = float(args.shared_x_label_pad_in) + float(args.legend_pad_in) + float(args.bottom_floor_in)
    fig_height = (panel_area_height_in + bottom_in) / float(args.top)
    bottom_frac = bottom_in / fig_height

    panel_paths = [panel_path(args, task, mode) for task in tasks]
    panel_sizes = []
    for path in panel_paths:
        with Image.open(path) as image:
            panel_sizes.append(image.size)
    target_width = max(width for width, _ in panel_sizes)
    target_height = max(height for _, height in panel_sizes)

    fig, axes = plt.subplots(nrows, ncols, figsize=(float(args.fig_width), fig_height), squeeze=False)
    first_shape = None
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        path = panel_paths[idx]
        with Image.open(path) as src:
            if src.size != (target_width, target_height):
                canvas = Image.new("RGBA", (target_width, target_height), "white")
                offset = ((target_width - src.width) // 2, (target_height - src.height) // 2)
                canvas.paste(src.convert("RGBA"), offset)
                img = np.asarray(canvas)
            else:
                img = np.asarray(src.convert("RGBA"))
        if first_shape is None:
            first_shape = img.shape
        elif img.shape != first_shape:
            print(f"WARNING: panel size differs: {path} {img.shape} vs {first_shape}")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.subplots_adjust(left=float(args.left), right=float(args.right), top=float(args.top), bottom=bottom_frac, wspace=0.0, hspace=0.0)
    panel_center_x = 0.5 * (float(args.left) + float(args.right))
    panel_center_y = 0.5 * (bottom_frac + float(args.top))
    shared_x_y = bottom_frac - float(args.shared_x_label_pad_in) / fig_height
    legend_y = shared_x_y - float(args.legend_pad_in) / fig_height

    fig.text(
        float(args.shared_y_label_x),
        panel_center_y + float(args.shared_y_label_y_offset),
        "Simple regret",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=float(args.shared_label_size),
    )
    fig.text(
        panel_center_x + float(args.shared_x_label_x_offset),
        shared_x_y,
        "Cumulative evaluations" if mode == "unit" else "Cumulative cost",
        ha="center",
        va="center",
        fontsize=float(args.shared_label_size),
    )
    handles, labels = legend_handles_labels(args, mode)
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=min(int(args.legend_ncol), len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
        fontsize=float(args.legend_size),
        handlelength=2.1,
        handletextpad=0.6,
        columnspacing=1.0,
    )

    out = args.out_root / f"mmlu_{args.prior_bucket}_{mode}_bo5pct.png"
    fig.savefig(out, dpi=int(args.dpi))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    setup_matplotlib(args)
    args.out_root.mkdir(parents=True, exist_ok=True)

    modes = []
    if args.cost_mode in {"both", "unit"}:
        modes.append("unit")
    if args.cost_mode in {"both", "aware"}:
        modes.append("aware")
    for mode in modes:
        if not bool(args.assemble_only):
            for group in ["small", "medium", "large"]:
                tasks = [task for task in selected_tasks(args) if TASK_GROUP[task] == group]
                loaded = load_group_mode(group, mode, tasks, args)
                stop_df = load_stop_summary_group_mode(group, mode, tasks, args)
                for task in tasks:
                    out = plot_panel(task, mode, loaded, stop_df, args)
                    print(f"Wrote panel: {out}")
        if not bool(args.individual_only):
            grid = assemble_grid(mode, args)
            print(f"Wrote grid: {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
