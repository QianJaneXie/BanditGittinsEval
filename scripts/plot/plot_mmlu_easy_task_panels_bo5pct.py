#!/usr/bin/env python3
"""Plot MMLU easy-subject task panels with 5% BO post-init curves.

This script is intentionally scoped to the paper appendix easy-subject grids:
it first writes fixed-size individual task panels, then assembles them into the
unit-cost and cost-aware grid figures.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path

_plot_dir = Path(__file__).resolve().parent
if str(_plot_dir) not in sys.path:
    sys.path.insert(0, str(_plot_dir))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from plot_mmlu_aggregate_2x2_normalized_fast_shared_labels import (
    normalize_task_name,
    read_filtered_history,
    task_column,
)

DEFAULT_MATRIX_DIR = Path("data/MMLU_matrices")
DEFAULT_COST_VECTOR = Path(
    "data_analysis/pricing/mmlu_prompt_eval_configurations_input_price.json"
)
# Match the 2x3 / 2x4 paper figures: show 0–10% of exhaustive evaluation cost.
PERCENT_XTICKS = (0.0, 5.0, 10.0)
PERCENT_X_RIGHT = 10.5
PERCENT_X_PAD = 0.25
SHARED_X_LABEL_PERCENT = "Percentage of Exhaustive Evaluation Cost"


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
COLOR_SYSRS = "tab:pink"
COLOR_BO_PBGI = "tab:olive"
COLOR_BO_LOGEI = "tab:brown"
COLOR_PROMPTEVAL = "tab:cyan"
PROMPTEVAL_RESULTS_DIR = Path(
    r"outputs\prompteval_downloads\wallclock_mmlu_gsm8k_piqa_20260807\results\formal\mmlu"
)

STYLE_BY_KIND = {
    # Match the main paper figures (2x3 / MMLU combined).
    "gittins_data": {"color": COLOR_GITTINS_S, "label": "Gittins-S", "lw": 2.55, "z": 7},
    "gittins_default": {"color": COLOR_GITTINS_G, "label": "Gittins-G", "lw": 2.45, "z": 6},
    "sysrs": {"color": COLOR_SYSRS, "label": "SySRs", "lw": 2.25, "z": 5},
    "ucb": {"color": COLOR_UCB, "label": "UCB-E", "lw": 2.15, "z": 4},
    "lrf": {"color": COLOR_LRF, "label": "LRF", "lw": 2.1, "z": 3.8},
    "bo_pbgi": {"color": COLOR_BO_PBGI, "label": "BO-PBGI", "lw": 2.1, "z": 3.4},
    "bo_logei": {"color": COLOR_BO_LOGEI, "label": "BO-LogEI(PC)", "lw": 2.05, "z": 3.2},
    "prompteval_bai": {"color": COLOR_PROMPTEVAL, "label": "PromptEval", "lw": 2.25, "z": 4.8},
}

# Bottom legend: 4 rows x 3 columns.
# Matplotlib fills legends column-major, so list Col1 (top→bottom), then Col2, then Col3:
#   Col1: Gittins-S, Gittins-G, BO-PBGI, BO-LogEI(PC)
#   Col2: matching mean stops
#   Col3: UCB-E, LRF, SySRs, ±SE
LEGEND_ENTRIES = [
    ("method", "gittins_data"),
    ("method", "gittins_default"),
    ("method", "bo_pbgi"),
    ("method", "bo_logei"),
    ("stop", "gittins_data"),
    ("stop", "gittins_default"),
    ("stop", "bo_pbgi"),
    ("stop", "bo_logei"),
    ("method", "ucb"),
    ("method", "lrf"),
    ("method", "sysrs"),
    ("method", "prompteval_bai"),
]

STOP_LABELS = {
    "gittins_data": "Gittins-S mean stop",
    "gittins_default": "Gittins-G mean stop",
    "bo_pbgi": "BO-PBGI mean stop",
    "bo_logei": "BO-LogEI(PC) mean stop",
}

X_START_ZERO_LEFT_PAD_FRAC = 0.02
LEGEND_LINE_MULT = 1.35


def cache_method_kind(kind: str, mode: str) -> str:
    if kind == "gittins_data":
        return "gittins_s"
    if kind == "gittins_default":
        return "gittins_g"
    if kind == "ucb":
        return "ucbe"
    if kind == "bo_logei" and mode == "aware":
        return "bo_logeipc"
    return kind


def style_kind_from_cache(method_kind: str) -> str:
    return {
        "gittins_s": "gittins_data",
        "gittins_g": "gittins_default",
        "ucbe": "ucb",
        "bo_logeipc": "bo_logei",
    }.get(str(method_kind), str(method_kind))


def method_label_for_kind(kind: str, mode: str) -> str:
    return str(STYLE_BY_KIND[kind]["label"])


def prompteval_curve_path(task: str, mode: str) -> Path:
    suffix = "combined" if mode == "unit" else "costaware_combined"
    return PROMPTEVAL_RESULTS_DIR / f"bai_processed_results_MMLU_{task}_{suffix}.npy"


def load_prompteval_curve(task: str, mode: str) -> pd.DataFrame:
    path = prompteval_curve_path(task, mode)
    if not path.is_file():
        return pd.DataFrame()
    payload = np.load(path, allow_pickle=True).item()
    curve = np.asarray(payload["curves"], dtype=float)[0, 0, 0]
    if curve.shape[0] != 2:
        raise ValueError(f"Unexpected PromptEval curve shape at {path}: {curve.shape}")
    y = np.asarray(curve[0], dtype=float)
    x = np.asarray(curve[1], dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return pd.DataFrame()
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"
    return pd.DataFrame({
        "mmlu_task": task,
        "mode": mode,
        "method_kind": "prompteval_bai",
        "method_label": STYLE_BY_KIND["prompteval_bai"]["label"],
        "x": x,
        "mean": y,
        "lo": y,
        "hi": y,
        "n_runs": 1,
        "x_label": x_col,
    })


def clean_tick_label(value: float, _pos: int) -> str:
    if abs(float(value)) < 1e-12:
        return "0"
    return f"{float(value):g}"


def clean_x_tick_label(value: float, _pos: int) -> str:
    if abs(float(value)) < 1e-12:
        return "0"
    return f"{float(value):g}"


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


_COST_VECTOR_CACHE: np.ndarray | None = None
_FULL_COST_CACHE: dict[tuple[str, str], float] = {}


def load_mmlu_cost_vector(path: Path) -> np.ndarray:
    global _COST_VECTOR_CACHE
    if _COST_VECTOR_CACHE is not None:
        return _COST_VECTOR_CACHE
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    n = len(payload)
    costs = np.asarray(
        [float(payload[str(i)]["estimated_cost_per_1m_input_tokens"]) for i in range(n)],
        dtype=float,
    )
    _COST_VECTOR_CACHE = costs
    return costs


def full_evaluation_cost(task: str, mode: str, args: argparse.Namespace) -> float:
    key = (str(task), str(mode))
    if key in _FULL_COST_CACHE:
        return _FULL_COST_CACHE[key]
    matrix_path = Path(args.matrix_dir) / f"{task}.npy"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing MMLU matrix for percentage x-axis: {matrix_path}")
    arr = np.load(matrix_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {matrix_path}, got shape {arr.shape}")
    n_arms, n_examples = int(arr.shape[0]), int(arr.shape[1])
    if mode == "unit":
        denom = float(n_arms * n_examples)
    else:
        costs = load_mmlu_cost_vector(Path(args.cost_vector))
        if costs.size != n_arms:
            raise ValueError(
                f"Cost vector length {costs.size} != n_arms {n_arms} for {task}"
            )
        denom = float(n_examples) * float(costs.sum())
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid full-evaluation denominator for {task}/{mode}: {denom}")
    _FULL_COST_CACHE[key] = denom
    return denom


def as_full_cost_percentage(
    values: float | np.ndarray | list[float],
    task: str,
    mode: str,
    args: argparse.Namespace,
) -> float | np.ndarray:
    denom = full_evaluation_cost(task, mode, args)
    converted = np.asarray(values, dtype=float) / denom * 100.0
    return float(converted) if converted.ndim == 0 else converted


def cache_curve_path(args: argparse.Namespace, task: str, mode: str) -> Path:
    return args.out_root / "processed_curves" / mode / f"{safe_token(task)}_curves.csv"


def cache_meta_path(args: argparse.Namespace, task: str, mode: str) -> Path:
    return args.out_root / "processed_curves" / mode / f"{safe_token(task)}_meta.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=Path(r"outputs\wandb_plots_new\paper_figures\mmlu_easy_bo5pct_panels"))
    p.add_argument("--prior-bucket", choices=sorted(TASKS_BY_BUCKET), default="easy")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional subset of tasks within --prior-bucket (default: all).",
    )
    p.add_argument("--cost-mode", choices=["both", "unit", "aware"], default="both")
    p.add_argument("--assemble-only", action="store_true", help="Reuse existing individual panels and only reassemble grids.")
    p.add_argument("--individual-only", action="store_true", help="Write individual task panels and skip grid assembly.")
    p.add_argument("--cache-curves-only", action="store_true", help="Write processed curve CSV/JSON files and do not plot panels or assemble grids.")
    p.add_argument("--plot-from-cache", action="store_true", help="Plot individual panels from processed_curves without reading raw history.")
    p.add_argument("--x-start-zero", action="store_true", help="Force individual panel x-axis to start at 0.")
    p.add_argument(
        "--x-as-percent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display x as %% of exhaustive evaluation cost with ticks at 0/5/10 (default: on).",
    )
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--cost-vector", type=Path, default=DEFAULT_COST_VECTOR)
    p.add_argument("--percent-x-right", type=float, default=PERCENT_X_RIGHT)
    p.add_argument("--percent-x-pad", type=float, default=PERCENT_X_PAD)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--grid-size", type=int, default=320)
    p.add_argument("--se-mult", type=float, default=1.0)
    p.add_argument("--range", choices=["stderr", "none"], default="stderr")
    p.add_argument("--curve-alpha", type=float, default=0.15)
    p.add_argument("--show-stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")
    p.add_argument("--stop-alpha", type=float, default=0.12)
    p.add_argument("--stop-line-alpha", type=float, default=0.70)
    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--single-width", type=float, default=2.25)
    p.add_argument("--single-height", type=float, default=1.65)
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--title-size", type=float, default=14.5)
    p.add_argument("--tick-size", type=float, default=14)
    p.add_argument("--shared-label-size", type=float, default=35)
    p.add_argument("--legend-size", type=float, default=26)
    p.add_argument("--legend-ncol", type=int, default=3)
    p.add_argument("--fig-width", type=float, default=18.0)
    p.add_argument("--grid-cols", type=int, default=4)
    p.add_argument("--left", type=float, default=0.058)
    p.add_argument("--right", type=float, default=0.985)
    p.add_argument("--top", type=float, default=0.965)
    p.add_argument("--shared-y-label-x", type=float, default=0.042)
    p.add_argument("--shared-x-label-pad-in", type=float, default=0.58)
    # Smaller pad pulls the legend closer to the x-label (slightly upward).
    p.add_argument("--legend-pad-in", type=float, default=1.55)
    p.add_argument("--bottom-floor-in", type=float, default=1.70)
    p.add_argument("--shared-x-label-x-offset", type=float, default=0.01)
    p.add_argument("--shared-y-label-y-offset", type=float, default=0.01)
    p.add_argument("--output-suffix", default="")
    p.add_argument("--sysrs-color", default="tab:pink")
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


def stop_lines_from_summary(
    task_stop_df: pd.DataFrame,
    variants: dict[str, str],
    x_col: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    if not bool(args.show_stopping) or task_stop_df.empty:
        return []

    stop_lines: list[dict[str, object]] = []
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

        entry: dict[str, object] = {
            "method_kind": cache_method_kind(kind, "aware" if x_col == "cum_original_cost" else "unit"),
            "x": mean_stop,
            "label": method_label_for_kind(kind, "aware" if x_col == "cum_original_cost" else "unit"),
        }
        if len(arr) > 1:
            band = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) * float(args.se_mult)
            lo = mean_stop - band
            hi = mean_stop + band
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                entry["lo"] = lo
                entry["hi"] = hi
        stop_lines.append(entry)
    return stop_lines


def draw_cached_stopping(ax: plt.Axes, stop_lines: list[dict[str, object]], args: argparse.Namespace) -> None:
    if not bool(args.show_stopping):
        return

    for stop in stop_lines:
        kind = style_kind_from_cache(str(stop.get("method_kind", "")))
        if kind not in STYLE_BY_KIND:
            continue
        color = STYLE_BY_KIND[kind]["color"]
        lo = stop.get("lo")
        hi = stop.get("hi")
        if lo is not None and hi is not None:
            lo_f = float(lo)
            hi_f = float(hi)
            if np.isfinite(lo_f) and np.isfinite(hi_f) and hi_f > lo_f:
                ax.axvspan(lo_f, hi_f, color=color, alpha=float(args.stop_alpha), linewidth=0, zorder=1)

        mean_stop = float(stop.get("x", float("nan")))
        if not np.isfinite(mean_stop):
            continue
        ax.axvline(
            mean_stop,
            color=STYLE_BY_KIND[kind]["color"],
            linestyle="--",
            linewidth=float(STYLE_BY_KIND[kind]["lw"]),
            alpha=float(args.stop_line_alpha),
            zorder=2,
        )


def draw_stopping(
    ax: plt.Axes,
    task_stop_df: pd.DataFrame,
    variants: dict[str, str],
    x_col: str,
    args: argparse.Namespace,
) -> None:
    draw_cached_stopping(ax, stop_lines_from_summary(task_stop_df, variants, x_col, args), args)


def panel_path(args: argparse.Namespace, task: str, mode: str) -> Path:
    group = TASK_GROUP[task]
    batch = GROUP_BATCH[group]
    return args.out_root / "individual" / mode / f"{safe_token(task)}_{mode}_B{batch}_scale{safe_token(args.scale)}.png"


def panel_title(task: str) -> str:
    return f"{TASK_DISPLAY[task]} ({TASK_SIZE[task]})"


def compute_panel_payload(
    task: str,
    mode: str,
    df: pd.DataFrame,
    stop_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"
    group = TASK_GROUP[task]
    variants = variants_for_group(group, mode, args.scale)
    task_df = df[df["_mmlu_task_for_plot"].astype(str) == task].copy()
    task_stop_df = stop_df[stop_df["_mmlu_task_for_plot"].astype(str) == task].copy() if not stop_df.empty else stop_df

    rows: list[pd.DataFrame] = []
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
        n_runs = np.sum(~np.isnan(y_arr), axis=0)
        rows.append(pd.DataFrame({
            "mmlu_task": task,
            "mode": mode,
            "method_kind": cache_method_kind(kind, mode),
            "method_label": method_label_for_kind(kind, mode),
            "x": x_grid,
            "mean": center,
            "lo": lo,
            "hi": hi,
            "n_runs": n_runs,
            "x_label": x_col,
        }))
    prompteval_curve = load_prompteval_curve(task, mode)
    if not prompteval_curve.empty:
        rows.append(prompteval_curve)

    curves = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(
        columns=["mmlu_task", "mode", "method_kind", "method_label", "x", "mean", "lo", "hi", "n_runs", "x_label"]
    )
    stop_lines = stop_lines_from_summary(task_stop_df, variants, x_col, args)

    meta: dict[str, object] = {
        "mmlu_task": task,
        "mode": mode,
        "x_label": x_col,
        "y_label": "simple_regret",
        "title": panel_title(task),
        "xlim": None,
        "ylim": None,
        "stop_lines": stop_lines,
        "band_style": {
            "range": args.range,
            "se_mult": float(args.se_mult),
            "curve_alpha": float(args.curve_alpha),
            "stop_alpha": float(args.stop_alpha),
            "stop_line_alpha": float(args.stop_line_alpha),
        },
        "style_keys_present": sorted(curves["method_kind"].dropna().astype(str).unique().tolist()) if not curves.empty else [],
    }
    return curves, meta


def apply_panel_axes_style(
    ax: plt.Axes,
    task: str,
    args: argparse.Namespace,
    *,
    x_as_percent: bool,
) -> None:
    ax.set_title(panel_title(task), fontsize=float(args.title_size), pad=3)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=float(args.tick_size), width=0.9, length=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_tick_label))
    if x_as_percent:
        ax.set_xticks(list(PERCENT_XTICKS))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(clean_x_tick_label))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def convert_stop_lines_to_percent(
    stop_lines: list[dict[str, object]],
    task: str,
    mode: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for stop in stop_lines:
        converted = dict(stop)
        for key in ("x", "lo", "hi"):
            if key in converted and converted[key] is not None:
                converted[key] = float(
                    as_full_cost_percentage(float(converted[key]), task, mode, args)
                )
        out.append(converted)
    return out


def draw_panel_payload(
    task: str,
    mode: str,
    curves: pd.DataFrame,
    meta: dict[str, object],
    args: argparse.Namespace,
) -> Path:
    fig, ax = plt.subplots(figsize=(args.single_width, args.single_height))
    x_as_percent = bool(args.x_as_percent)
    plot_curves = curves
    stop_lines = list(meta.get("stop_lines", []))
    if x_as_percent and not curves.empty:
        plot_curves = curves.copy()
        plot_curves["x"] = as_full_cost_percentage(
            pd.to_numeric(plot_curves["x"], errors="coerce").to_numpy(dtype=float),
            task,
            mode,
            args,
        )
        stop_lines = convert_stop_lines_to_percent(stop_lines, task, mode, args)

    for method_kind, cg in plot_curves.groupby("method_kind", sort=False):
        kind = style_kind_from_cache(str(method_kind))
        if kind not in STYLE_BY_KIND:
            continue
        style = STYLE_BY_KIND[kind]
        cg = cg.sort_values("x")
        x = pd.to_numeric(cg["x"], errors="coerce").to_numpy(dtype=float)
        mean = pd.to_numeric(cg["mean"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(cg["lo"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(cg["hi"], errors="coerce").to_numpy(dtype=float)
        if kind == "prompteval_bai" and x_as_percent and len(x) > 0:
            x_right = float(args.percent_x_right)
            if float(x[-1]) < x_right:
                x = np.append(x, x_right)
                mean = np.append(mean, mean[-1])
                lo = np.append(lo, lo[-1])
                hi = np.append(hi, hi[-1])
        ax.plot(
            x,
            mean,
            color=style["color"],
            linewidth=style["lw"],
            zorder=style["z"],
            drawstyle="steps-post" if kind == "prompteval_bai" else "default",
        )
        if args.range != "none":
            band_alpha = 0.20 if kind.startswith("gittins_") else 0.155
            ax.fill_between(
                x,
                lo,
                hi,
                color=style["color"],
                alpha=band_alpha,
                linewidth=0,
                zorder=style["z"] - 0.5,
                step="post" if kind == "prompteval_bai" else None,
            )

    draw_cached_stopping(ax, stop_lines, args)
    apply_panel_axes_style(ax, task, args, x_as_percent=x_as_percent)
    if x_as_percent:
        ax.set_xlim(-float(args.percent_x_pad), float(args.percent_x_right))
    elif bool(args.x_start_zero):
        if meta.get("xlim") is not None:
            right = float(list(meta["xlim"])[1])
        else:
            x_left, x_right = ax.get_xlim()
            right = x_right + 0.08 * (x_right - x_left)
        left = -float(X_START_ZERO_LEFT_PAD_FRAC) * right if right > 0 else 0.0
        ax.set_xlim(left, right)
        meta["xlim"] = [left, right]
    elif meta.get("xlim") is not None:
        ax.set_xlim(*meta["xlim"])
    else:
        x_left, x_right = ax.get_xlim()
        ax.set_xlim(x_left, x_right + 0.08 * (x_right - x_left))
        meta["xlim"] = [float(v) for v in ax.get_xlim()]
    if meta.get("ylim") is not None:
        ax.set_ylim(*meta["ylim"])
    else:
        meta["ylim"] = [float(v) for v in ax.get_ylim()]
    bottom = 0.20 if x_as_percent else 0.17
    fig.subplots_adjust(left=0.20, right=0.90, bottom=bottom, top=0.82)

    out = panel_path(args, task, mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi))
    plt.close(fig)
    return out


def write_panel_cache(task: str, mode: str, curves: pd.DataFrame, meta: dict[str, object], args: argparse.Namespace) -> None:
    if meta.get("xlim") is None or meta.get("ylim") is None:
        xs: list[float] = []
        ys: list[float] = []
        if not curves.empty:
            xs.extend(pd.to_numeric(curves["x"], errors="coerce").dropna().astype(float).tolist())
            for col in ["lo", "hi", "mean"]:
                ys.extend(pd.to_numeric(curves[col], errors="coerce").dropna().astype(float).tolist())
        for stop in meta.get("stop_lines", []):
            for key in ["lo", "hi", "x"]:
                if key in stop and np.isfinite(float(stop[key])):
                    xs.append(float(stop[key]))
        if xs and meta.get("xlim") is None:
            x_left = 0.0 if bool(args.x_start_zero) else float(np.min(xs))
            x_right = float(np.max(xs))
            pad = 0.08 * (x_right - x_left) if x_right > x_left else max(1.0, abs(x_right) * 0.08)
            meta["xlim"] = [x_left, x_right + pad]
        if ys and meta.get("ylim") is None:
            y_lo = float(np.min(ys))
            y_hi = float(np.max(ys))
            pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else max(0.01, abs(y_hi) * 0.05)
            meta["ylim"] = [y_lo - pad, y_hi + pad]

    curve_path = cache_curve_path(args, task, mode)
    meta_path = cache_meta_path(args, task, mode)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(curve_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[CACHE] wrote {curve_path}")
    print(f"[CACHE] wrote {meta_path}")


def plot_panel(task: str, mode: str, df: pd.DataFrame, stop_df: pd.DataFrame, args: argparse.Namespace) -> Path:
    curves, meta = compute_panel_payload(task, mode, df, stop_df, args)
    out = draw_panel_payload(task, mode, curves, meta, args)
    if not bool(args.cache_curves_only):
        write_panel_cache(task, mode, curves, meta, args)
    return out


def cache_panel(task: str, mode: str, df: pd.DataFrame, stop_df: pd.DataFrame, args: argparse.Namespace) -> None:
    curves, meta = compute_panel_payload(task, mode, df, stop_df, args)
    write_panel_cache(task, mode, curves, meta, args)


def plot_panel_from_cache(task: str, mode: str, args: argparse.Namespace) -> Path:
    curve_path = cache_curve_path(args, task, mode)
    meta_path = cache_meta_path(args, task, mode)
    if not curve_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing processed cache for {task}/{mode}: {curve_path} or {meta_path}")
    curves = pd.read_csv(curve_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "method_kind" not in curves.columns or "prompteval_bai" not in set(curves["method_kind"].astype(str)):
        prompteval_curve = load_prompteval_curve(task, mode)
        if not prompteval_curve.empty:
            curves = pd.concat([curves, prompteval_curve], ignore_index=True, sort=False)
    out = draw_panel_payload(task, mode, curves, meta, args)
    print(f"[PLOT-CACHE] wrote {out}")
    return out


def legend_handles_labels(args: argparse.Namespace, mode: str) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    for entry_type, kind in LEGEND_ENTRIES:
        if entry_type == "method":
            assert kind is not None
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=STYLE_BY_KIND[kind]["color"],
                    linewidth=float(STYLE_BY_KIND[kind]["lw"]) * float(LEGEND_LINE_MULT),
                    linestyle="-",
                )
            )
            labels.append(str(STYLE_BY_KIND[kind]["label"]))
        elif entry_type == "stop":
            assert kind is not None
            if not bool(args.show_stopping):
                continue
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=STYLE_BY_KIND[kind]["color"],
                    linewidth=float(STYLE_BY_KIND[kind]["lw"]) * float(LEGEND_LINE_MULT),
                    linestyle="--",
                    alpha=float(args.stop_line_alpha),
                )
            )
            labels.append(STOP_LABELS[kind])
        elif entry_type == "band":
            if args.range == "none":
                continue
            handles.append(Patch(facecolor="0.55", edgecolor="none", alpha=0.18))
            labels.append(rf"$\pm${args.se_mult:g} SE band")
        elif entry_type == "blank":
            handles.append(Line2D([0], [0], color="none", linewidth=0, alpha=0))
            labels.append("")
    return handles, labels


def selected_tasks(args: argparse.Namespace) -> list[str]:
    tasks = list(TASKS_BY_BUCKET[str(args.prior_bucket)])
    if args.tasks:
        wanted = {normalize_task_name(t) for t in args.tasks}
        tasks = [task for task in tasks if task in wanted]
        missing = wanted - set(tasks)
        if missing:
            raise ValueError(
                f"Unknown tasks for prior-bucket={args.prior_bucket}: {sorted(missing)}"
            )
    return tasks


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
    panel_top = float(args.top)

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

    fig.subplots_adjust(left=float(args.left), right=float(args.right), top=panel_top, bottom=bottom_frac, wspace=0.0, hspace=0.0)
    panel_center_x = 0.5 * (float(args.left) + float(args.right))
    panel_center_y = 0.5 * (bottom_frac + panel_top)
    shared_x_y = bottom_frac - float(args.shared_x_label_pad_in) / fig_height
    legend_y = shared_x_y - float(args.legend_pad_in) / fig_height
    if bool(args.x_as_percent):
        shared_x_label = SHARED_X_LABEL_PERCENT
    else:
        shared_x_label = "Cumulative evaluations" if mode == "unit" else "Cumulative cost"

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
        shared_x_label,
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
        handlelength=3.0,
        handletextpad=0.5,
        columnspacing=1.35,
        labelspacing=0.45,
    )

    suffix = str(args.output_suffix)
    out = args.out_root / f"mmlu_{args.prior_bucket}_{mode}{suffix}.png"
    fig.savefig(out, dpi=int(args.dpi))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    STYLE_BY_KIND["sysrs"]["color"] = str(args.sysrs_color)
    setup_matplotlib(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if bool(args.cache_curves_only) and bool(args.plot_from_cache):
        raise ValueError("--cache-curves-only and --plot-from-cache cannot be used together")

    modes = []
    if args.cost_mode in {"both", "unit"}:
        modes.append("unit")
    if args.cost_mode in {"both", "aware"}:
        modes.append("aware")
    for mode in modes:
        if bool(args.plot_from_cache):
            if not bool(args.assemble_only):
                for task in selected_tasks(args):
                    plot_panel_from_cache(task, mode, args)
            if not bool(args.individual_only):
                grid = assemble_grid(mode, args)
                print(f"[ASSEMBLE] wrote {grid}")
            continue

        if not bool(args.assemble_only):
            for group in ["small", "medium", "large"]:
                tasks = [task for task in selected_tasks(args) if TASK_GROUP[task] == group]
                if not tasks:
                    continue
                loaded = load_group_mode(group, mode, tasks, args)
                stop_df = load_stop_summary_group_mode(group, mode, tasks, args)
                for task in tasks:
                    if bool(args.cache_curves_only):
                        cache_panel(task, mode, loaded, stop_df, args)
                    else:
                        out = plot_panel(task, mode, loaded, stop_df, args)
                        print(f"Wrote panel: {out}")
                del loaded
                del stop_df
                gc.collect()
        if not bool(args.individual_only) and not bool(args.cache_curves_only):
            grid = assemble_grid(mode, args)
            print(f"[ASSEMBLE] wrote {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
