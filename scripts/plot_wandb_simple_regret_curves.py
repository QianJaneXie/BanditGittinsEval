#!/usr/bin/env python3
"""Plot aggregated W&B simple-regret curves with readable labels, LRF warmup cropping,
and Gittins stopping bands.

Designed for the current GSM8K / PIQA simple-regret workflow:
- Aggregates across all run_id values left after filtering.
- Crops LRF to the post-warmup region by default.
- Draws Gittins stopping as a mean vertical line plus an optional distribution band.
- Uses short, readable legend labels and clearer axis names by default.
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


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--history-csv", required=True, type=Path)
    p.add_argument(
        "--summary-csv",
        default=None,
        type=Path,
        help=(
            "runs_summary.csv or runs_stopping.csv. If runs_summary.csv is passed and a sibling "
            "runs_stopping.csv exists, the script will automatically use the stopping file."
        ),
    )
    p.add_argument("--out-dir", required=True, type=Path)

    p.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. gsm8k or piqa")
    p.add_argument("--matrix-seed", default=None, help="Leave unset to aggregate all matrix seeds")
    p.add_argument("--run-seed", default=None, help="Leave unset to aggregate all run seeds")
    p.add_argument("--group-by", default="experiment_variant")
    p.add_argument("--variant-regex", default=None)
    p.add_argument("--variants", default=None, help="Comma-separated exact variants to plot, in order")
    p.add_argument("--max-variants", type=int, default=12)
    p.add_argument("--min-runs", type=int, default=1)

    p.add_argument("--x-axis", choices=["cum_eval", "cum_original_cost", "step_idx"], default="cum_eval")
    p.add_argument("--y-axis", default="simple_regret")
    p.add_argument("--grid-size", type=int, default=300)
    p.add_argument(
        "--range",
        choices=["std", "stderr", "none"],
        default="stderr",
        help="Curve band: std = run-to-run variation; stderr = uncertainty of the mean.",
    )

    p.add_argument("--crop-lrf-warmup", dest="crop_lrf_warmup", action="store_true", default=True)
    p.add_argument("--no-crop-lrf-warmup", dest="crop_lrf_warmup", action="store_false")
    p.add_argument("--warmup-percentage", type=float, default=0.05)
    p.add_argument("--default-eval-budget-fraction", type=float, default=0.10)

    p.add_argument("--show-stopping", dest="show_stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")
    p.add_argument(
        "--stop-band",
        choices=["iqr", "std", "stderr", "none"],
        default="iqr",
        help="Stopping-time band: iqr = Q1-Q3; std = mean±std; stderr = mean±stderr.",
    )
    p.add_argument("--stop-q-low", type=float, default=0.25)
    p.add_argument("--stop-q-high", type=float, default=0.75)
    p.add_argument("--stop-band-alpha", type=float, default=0.10)
    p.add_argument("--stop-line-alpha", type=float, default=0.70)

    p.add_argument("--title", default=None)
    p.add_argument(
        "--subtitle",
        default=None,
        help="Default subtitle is short and based on --range / LRF cropping. Use '' to remove it.",
    )
    p.add_argument("--fig-width", type=float, default=12.0)
    p.add_argument("--fig-height", type=float, default=6.8)
    # Paper-sized defaults: bump all text a notch larger.
    p.add_argument("--title-size", type=float, default=27)
    p.add_argument("--label-size", type=float, default=23)
    p.add_argument("--tick-size", type=float, default=22)
    p.add_argument("--legend-size", type=float, default=22)
    p.add_argument("--line-width", type=float, default=2.6)
    p.add_argument("--stop-line-width", type=float, default=1.8)
    p.add_argument(
        "--paper-colors",
        dest="paper_colors",
        action="store_true",
        default=True,
        help="Use paper-style colors via tab:* names: tab:blue (UCB), tab:purple (LRF), tab:green (Gittins default prior), tab:orange (Gittins dataset prior).",
    )
    p.add_argument("--no-paper-colors", dest="paper_colors", action="store_false")
    p.add_argument(
        "--variant-colors",
        default=None,
        help=(
            "Optional JSON mapping from exact variant names to matplotlib colors, e.g. "
            "'{\"gittins_unit_B4_scale1e-4_dataset\":\"#1f77b4\"}'. "
            "These colors override --paper-colors for matching variants."
        ),
    )
    p.add_argument(
        "--emphasize-variant",
        default=None,
        help="Exact experiment_variant string(s) to highlight: thicker line, higher z-order, bold legend label. Comma-separated.",
    )
    p.add_argument("--emphasize-line-width", type=float, default=3.4)
    p.add_argument("--emphasize-zorder", type=float, default=5.0)
    p.add_argument("--base-zorder", type=float, default=2.0)
    p.add_argument("--legend-loc", default="best")
    p.add_argument("--legend-ncol", type=int, default=1)
    p.add_argument("--no-short-labels", dest="short_labels", action="store_false", default=True)
    p.add_argument("--show-n", dest="show_n", action="store_true", default=False)
    p.add_argument("--no-show-n", dest="show_n", action="store_false")
    p.add_argument("--y-limit-min", type=float, default=None)
    p.add_argument("--y-limit-max", type=float, default=None)

    return p.parse_args()



def _as_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=str)
    return df[col].astype(str)


def _first_numeric(g: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if col not in g.columns:
        return default
    vals = pd.to_numeric(g[col], errors="coerce").dropna()
    if vals.empty:
        return default
    return float(vals.iloc[0])


def apply_common_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.dataset is not None:
        dataset = args.dataset.lower()
        mask = (_as_str_series(out, "dataset_tag_resolved").str.lower() == dataset) | (
            _as_str_series(out, "dataset_tag").str.lower() == dataset
        )
        out = out[mask]
    if args.matrix_seed is not None and "matrix_seed" in out.columns:
        out = out[out["matrix_seed"].astype(str) == str(args.matrix_seed)]
    if args.run_seed is not None and "run_seed" in out.columns:
        out = out[out["run_seed"].astype(str) == str(args.run_seed)]
    return out


def apply_variant_filters(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str] | None]:
    out = df.copy()
    exact_order: list[str] | None = None
    if args.variants:
        exact_order = [v.strip() for v in args.variants.split(",") if v.strip()]
        out = out[out[args.group_by].astype(str).isin(exact_order)]
    if args.variant_regex is not None:
        rx = re.compile(args.variant_regex)
        out = out[out[args.group_by].astype(str).map(lambda x: bool(rx.search(x)))]
    return out, exact_order


def _variant_is_lrf(variant: str, df: pd.DataFrame) -> bool:
    v = str(variant).lower()
    if v.startswith("lrf") or "lrf" in v:
        return True
    if "policy_family" in df.columns:
        fams = set(df["policy_family"].dropna().astype(str).str.lower())
        return fams == {"lrf"}
    return False


def _get_lrf_warmup_evals(g: pd.DataFrame, args: argparse.Namespace) -> float:
    warmup = _first_numeric(g, "warmup_percentage", args.warmup_percentage)
    n_cells = _first_numeric(g, "n_cells", float("nan"))

    if not math.isfinite(n_cells) or n_cells <= 0:
        eval_budget_fraction = _first_numeric(g, "eval_budget_fraction", args.default_eval_budget_fraction)
        max_eval = pd.to_numeric(g.get("cum_eval"), errors="coerce").max()
        if pd.notna(max_eval) and eval_budget_fraction > 0:
            n_cells = float(max_eval) / float(eval_budget_fraction)

    if not math.isfinite(n_cells) or n_cells <= 0:
        return float("nan")
    return float(math.ceil(float(warmup) * float(n_cells)))


def crop_lrf_warmup(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.crop_lrf_warmup:
        return df
    pieces = []
    for variant, g in df.groupby(args.group_by, sort=False):
        if _variant_is_lrf(str(variant), g):
            warmup_evals = _get_lrf_warmup_evals(g, args)
            if math.isfinite(warmup_evals) and "cum_eval" in g.columns:
                g = g[pd.to_numeric(g["cum_eval"], errors="coerce") >= warmup_evals]
        pieces.append(g)
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, axis=0, ignore_index=False)


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


def choose_variants(df: pd.DataFrame, args: argparse.Namespace, exact_order: list[str] | None) -> list[str]:
    counts = df.groupby(args.group_by)["run_id"].nunique().to_dict()
    variants = [str(v) for v in df[args.group_by].dropna().astype(str).unique().tolist()]
    variants = [v for v in variants if counts.get(v, 0) >= args.min_runs]

    if exact_order is not None:
        available = set(variants)
        ordered = [v for v in exact_order if v in available]
        missing = [v for v in exact_order if v not in available]
        if missing:
            print("WARNING: requested variants not found after filtering:")
            for v in missing:
                print("  ", v)
        return ordered

    variants = sorted(variants)
    if len(variants) <= args.max_variants:
        return variants

    final_rows = []
    for variant, vg in df.groupby(args.group_by):
        vals = []
        for _, rg in vg.groupby("run_id"):
            rr = rg[[args.x_axis, args.y_axis]].dropna().sort_values(args.x_axis)
            if not rr.empty:
                vals.append(float(rr[args.y_axis].iloc[-1]))
        if vals:
            final_rows.append((str(variant), float(np.mean(vals))))

    selected = [v for v, _ in sorted(final_rows, key=lambda t: t[1])[: args.max_variants]]
    print(f"Too many variants; plotting top {len(selected)} by mean final {args.y_axis}:")
    for v in selected:
        print("  ", v)
    return selected


def _candidate_stopping_csvs(summary_csv: Path | None) -> list[Path]:
    if summary_csv is None:
        return []
    candidates = [summary_csv]
    sibling = summary_csv.with_name("runs_stopping.csv")
    if sibling != summary_csv:
        candidates.append(sibling)
    seen: set[Path] = set()
    out = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _read_stopping_source(args: argparse.Namespace, stop_col: str) -> pd.DataFrame:
    for path in _candidate_stopping_csvs(args.summary_csv):
        if not path.is_file():
            continue
        try:
            sdf = pd.read_csv(path)
        except Exception as e:
            print(f"WARNING: could not read {path}: {e}")
            continue
        if stop_col in sdf.columns:
            if path != args.summary_csv:
                print(f"Using stopping data from sibling file: {path}")
            return sdf
    return pd.DataFrame()


def load_stopping_summary(args: argparse.Namespace, variants: list[str]) -> pd.DataFrame:
    if not args.show_stopping or args.summary_csv is None:
        return pd.DataFrame()

    if args.x_axis == "cum_eval":
        stop_col = "gittins_stop_cum_eval"
    elif args.x_axis == "cum_original_cost":
        stop_col = "gittins_stop_cum_original_cost"
    else:
        return pd.DataFrame()

    sdf = _read_stopping_source(args, stop_col)
    if sdf.empty:
        print(f"WARNING: {stop_col} not found; stopping visualization skipped")
        return pd.DataFrame()

    sdf = apply_common_filters(sdf, args)
    if args.group_by not in sdf.columns:
        print(f"WARNING: group-by column {args.group_by!r} not found in stopping CSV")
        return pd.DataFrame()
    sdf = sdf[sdf[args.group_by].astype(str).isin(variants)]

    rows = []
    q_low = min(max(float(args.stop_q_low), 0.0), 1.0)
    q_high = min(max(float(args.stop_q_high), 0.0), 1.0)
    if q_low > q_high:
        q_low, q_high = q_high, q_low

    for variant, g in sdf.groupby(args.group_by):
        vals = pd.to_numeric(g[stop_col], errors="coerce")
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        if vals.empty:
            continue
        arr = vals.to_numpy(dtype=float)
        n_stopped = int(len(arr))
        std = float(np.std(arr, ddof=0))
        rows.append(
            {
                "variant": str(variant),
                "x_axis": args.x_axis,
                "stop_col": stop_col,
                "mean_stop_x": float(np.mean(arr)),
                "median_stop_x": float(np.median(arr)),
                "std_stop_x": std,
                "stderr_stop_x": float(std / np.sqrt(max(n_stopped, 1))),
                "q_low": q_low,
                "q_high": q_high,
                "q_low_stop_x": float(np.quantile(arr, q_low)),
                "q_high_stop_x": float(np.quantile(arr, q_high)),
                "min_stop_x": float(np.min(arr)),
                "max_stop_x": float(np.max(arr)),
                "n_stopped": n_stopped,
                "n_total": int(g["run_id"].nunique()) if "run_id" in g.columns else int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _stop_band_bounds(row: pd.Series, mode: str) -> tuple[float, float] | None:
    mean = float(row["mean_stop_x"])
    if not np.isfinite(mean):
        return None
    if mode == "none":
        return None
    if mode == "iqr":
        lo = float(row["q_low_stop_x"])
        hi = float(row["q_high_stop_x"])
    elif mode == "std":
        width = float(row["std_stop_x"])
        lo, hi = mean - width, mean + width
    elif mode == "stderr":
        width = float(row["stderr_stop_x"])
        lo, hi = mean - width, mean + width
    else:
        return None
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_variant_label(variant: str) -> dict[str, str]:
    v = str(variant)
    m = re.fullmatch(r"(ucb|lrf)_B(\d+)", v)
    if m:
        fam, b = m.group(1), m.group(2)
        return {
            "family": "UCB-E" if fam == "ucb" else "LRF",
            "batch": b,
            "prior": "",
            "prior_short": "",
            "cost": "",
        }

    m = re.fullmatch(r"gittins_(unit|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)", v)
    if m:
        cost_mode, b, scale, prior = m.groups()
        # The unit/cost-aware setting is already clear from the figure title and x-axis,
        # so legend labels only need to distinguish the prior.
        prior_label = "data prior" if prior == "dataset" else "default prior"
        prior_short = "data" if prior == "dataset" else "default"
        return {
            "family": "Gittins",
            "batch": b,
            "prior": prior_label,
            "prior_short": prior_short,
            "cost": cost_mode,
            "scale": scale,
        }

    return {"family": v, "batch": "", "prior": "", "prior_short": "", "cost": ""}


def display_variant_label(variant: str, n_runs: int | None, args: argparse.Namespace) -> str:
    if not args.short_labels:
        return f"{variant} (n={n_runs})" if args.show_n and n_runs is not None else str(variant)

    info = _parse_variant_label(variant)
    label = info["family"]
    if info.get("batch"):
        label += f" B{info['batch']}"
    if info.get("prior"):
        label += f", {info['prior']}"
    if args.show_n and n_runs is not None:
        label += f" (n={n_runs})"
    return label


def display_stop_label(variant: str, kind: str, n_stopped: int, args: argparse.Namespace) -> str:
    if not args.short_labels:
        suffix = f" (n={n_stopped})" if args.show_n else ""
        return f"{variant} {kind}{suffix}"

    info = _parse_variant_label(variant)
    prior = info.get("prior_short") or info.get("family", "")
    if kind == "mean":
        label = f"Mean stop ({prior})" if prior else "Mean stop"
    else:
        label = f"Stop {kind} ({prior})" if prior else f"Stop {kind}"
    if args.show_n:
        label += f" (n={n_stopped})"
    return label


def axis_label(col: str) -> str:
    return {
        "cum_eval": "Cumulative evaluations",
        "cum_original_cost": "Cumulative cost",
        "step_idx": "Batch step",
        "simple_regret": "Simple regret",
    }.get(col, col)


def range_label(mode: str) -> str:
    if mode == "stderr":
        return "mean ± SE"
    if mode == "std":
        return "mean ± SD"
    return "mean"


# Matplotlib tab10 named colors (stable across backends; avoids custom hex clashes).
_COLOR_UCB = "tab:blue"
_COLOR_LRF = "tab:purple"
_COLOR_GITTINS_DEFAULT_PRIOR = "tab:green"
_COLOR_GITTINS_DATASET_PRIOR = "tab:orange"
_COLOR_FALLBACK = "tab:gray"


def _emphasized_variants(args: argparse.Namespace) -> set[str]:
    raw = getattr(args, "emphasize_variant", None)
    if not raw:
        return set()
    return {v.strip() for v in str(raw).split(",") if v.strip()}


def _parse_variant_colors(raw: str | None) -> dict[str, str]:
    """Parse optional exact variant -> color mapping.

    This accepts both strict JSON, e.g.
        {"gittins_unit_B4_scale1e-4_dataset":"#1f77b4"}

    and a PowerShell-stripped form that sometimes appears after command-line
    quoting, e.g.
        {gittins_unit_B4_scale1e-4_dataset:#1f77b4}

    If the argument is not provided, the original paper-color behavior is
    unchanged.
    """
    if raw is None or str(raw).strip() == "":
        return {}

    text = str(raw).strip()

    # Preferred path: valid JSON.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        if not isinstance(parsed, dict):
            raise SystemExit("--variant-colors must be a JSON object mapping variant names to colors")
        return {str(k): str(v) for k, v in parsed.items()}

    # Fallback for PowerShell-stripped mappings like:
    #   {variant:#1f77b4,variant2:#ff7f0e}
    # This is intentionally conservative and only supports comma-separated
    # key:value pairs, which is enough for color maps.
    stripped = text
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1]

    out: dict[str, str] = {}
    if stripped.strip() == "":
        return out

    for item in stripped.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"Could not parse --variant-colors item: {item!r}")
        key, value = item.split(":", 1)
        key = key.strip().strip("'\"")
        value = value.strip().strip("'\"")
        if not key:
            raise SystemExit(f"Could not parse empty --variant-colors key in item: {item!r}")
        if not value:
            raise SystemExit(f"Could not parse empty --variant-colors value in item: {item!r}")
        out[key] = value

    return out


def _gittins_variant_numeric_parts(variant: str) -> dict[str, float | str] | None:
    """Return parsed fields for compact Gittins variants.

    Used only for automatic colors in pure Gittins batch/scale ablations.
    It does not change UCB/LRF/Gittins comparison colors.
    """
    m = re.fullmatch(
        r"gittins_(unit|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)",
        str(variant),
    )
    if not m:
        return None
    cost_mode, batch, scale, prior = m.groups()
    try:
        scale_value = float(scale)
    except ValueError:
        return None
    return {
        "cost_mode": cost_mode,
        "batch": float(batch),
        "scale": scale_value,
        "prior": prior,
    }


def _auto_ordered_gittins_colors(variants: list[str]) -> dict[str, str]:
    """Assign standard colors by numeric value for pure Gittins ablations.

    Rule requested for batch/scale plots:
      smallest value -> tab:blue, middle -> tab:orange, largest -> tab:green.

    This function only activates when all plotted variants are Gittins variants
    from the same cost mode and prior. Mixed UCB/LRF/Gittins paper-style plots
    keep their original colors.
    """
    if len(variants) < 2:
        return {}

    parsed = {v: _gittins_variant_numeric_parts(v) for v in variants}
    if any(p is None for p in parsed.values()):
        return {}

    cost_modes = {str(p["cost_mode"]) for p in parsed.values() if p is not None}
    priors = {str(p["prior"]) for p in parsed.values() if p is not None}
    batches = {float(p["batch"]) for p in parsed.values() if p is not None}
    scales = {float(p["scale"]) for p in parsed.values() if p is not None}

    if len(cost_modes) != 1 or len(priors) != 1:
        return {}

    if len(batches) > 1 and len(scales) == 1:
        key = "batch"
    elif len(scales) > 1 and len(batches) == 1:
        key = "scale"
    else:
        return {}

    standard_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    ordered = sorted(variants, key=lambda v: float(parsed[v][key]))  # type: ignore[index]
    return {v: standard_colors[i % len(standard_colors)] for i, v in enumerate(ordered)}


def _maybe_order_gittins_ablation_variants(variants: list[str]) -> list[str]:
    """Order pure Gittins batch/scale ablations by the numeric value in the legend."""
    parsed = {v: _gittins_variant_numeric_parts(v) for v in variants}
    if not variants or any(p is None for p in parsed.values()):
        return variants

    cost_modes = {str(p["cost_mode"]) for p in parsed.values() if p is not None}
    priors = {str(p["prior"]) for p in parsed.values() if p is not None}
    batches = {float(p["batch"]) for p in parsed.values() if p is not None}
    scales = {float(p["scale"]) for p in parsed.values() if p is not None}

    if len(cost_modes) != 1 or len(priors) != 1:
        return variants
    if len(batches) > 1 and len(scales) == 1:
        return sorted(variants, key=lambda v: float(parsed[v]["batch"]))  # type: ignore[index]
    if len(scales) > 1 and len(batches) == 1:
        return sorted(variants, key=lambda v: float(parsed[v]["scale"]))  # type: ignore[index]
    return variants


def variant_plot_style(variant: str, args: argparse.Namespace) -> dict[str, float | str | None]:
    """Line color / width / z-order for a variant."""
    lw = float(args.line_width)
    z = float(args.base_zorder)
    color: str | None = None

    if args.paper_colors:
        v = str(variant)
        vl = v.lower()
        if vl.startswith("ucb"):
            color = _COLOR_UCB
        elif vl.startswith("lrf"):
            color = _COLOR_LRF
        else:
            info = _parse_variant_label(v)
            if info.get("family") == "Gittins":
                ps = info.get("prior_short", "")
                if ps == "data":
                    color = _COLOR_GITTINS_DATASET_PRIOR
                elif ps == "default":
                    color = _COLOR_GITTINS_DEFAULT_PRIOR
        if color is None:
            color = _COLOR_FALLBACK

    variant_colors = getattr(args, "variant_colors_map", {}) or {}
    if str(variant) in variant_colors:
        color = str(variant_colors[str(variant)])

    if str(variant) in _emphasized_variants(args):
        lw = float(args.emphasize_line_width)
        z = float(args.emphasize_zorder)

    return {"color": color, "linewidth": lw, "zorder": z}


def default_title(args: argparse.Namespace) -> str:
    dataset = (args.dataset or "All").upper()
    setting = "Cost-aware" if args.x_axis == "cum_original_cost" else "Unit-cost"
    return f"{dataset} {setting}"


def default_subtitle(args: argparse.Namespace) -> str:
    if args.subtitle is not None:
        return args.subtitle
    return ""


def _plot_stopping(ax: plt.Axes, stopping_df: pd.DataFrame, color_by_variant: dict[str, str], args: argparse.Namespace) -> None:
    if not args.show_stopping or stopping_df.empty:
        return

    for _, r in stopping_df.iterrows():
        variant = str(r["variant"])
        stop_x = float(r["mean_stop_x"])
        if not np.isfinite(stop_x) or stop_x < 0:
            continue
        color = color_by_variant.get(variant, None)
        n_stopped = int(r["n_stopped"])

        bounds = _stop_band_bounds(r, args.stop_band)
        if bounds is not None:
            lo, hi = bounds
            if hi > lo:
                if args.stop_band == "iqr":
                    kind = "IQR"
                elif args.stop_band == "std":
                    kind = "±SD"
                else:
                    kind = "±SE"
                ax.axvspan(
                    lo,
                    hi,
                    color=color,
                    alpha=float(args.stop_band_alpha),
                    linewidth=0,
                    label=display_stop_label(variant, kind, n_stopped, args),
                )

        ax.axvline(
            stop_x,
            color=color,
            linestyle="--",
            alpha=float(args.stop_line_alpha),
            linewidth=float(args.stop_line_width),
            label=display_stop_label(variant, "mean", n_stopped, args),
        )


def main() -> int:
    args = parse_args()
    args.variant_colors_map = _parse_variant_colors(args.variant_colors)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": args.tick_size,
            "axes.titlesize": args.title_size,
            "axes.labelsize": args.label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
        }
    )

    df = pd.read_csv(args.history_csv)
    df = apply_common_filters(df, args)
    df, exact_order = apply_variant_filters(df, args)

    required = [args.x_axis, args.y_axis, args.group_by, "run_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in history CSV: {missing}")

    df[args.x_axis] = pd.to_numeric(df[args.x_axis], errors="coerce")
    df[args.y_axis] = pd.to_numeric(df[args.y_axis], errors="coerce")
    df = df.dropna(subset=[args.x_axis, args.y_axis, args.group_by, "run_id"])
    df = crop_lrf_warmup(df, args)

    variants = choose_variants(df, args, exact_order)
    variants = _maybe_order_gittins_ablation_variants(variants)
    if not variants:
        raise SystemExit("No variants left after filtering.")

    # For pure Gittins batch-size or scale ablations, bind colors to numeric values:
    # smallest -> tab:blue, middle -> tab:orange, largest -> tab:green.
    # Explicit --variant-colors still has the highest priority if provided.
    for variant, color in _auto_ordered_gittins_colors(variants).items():
        args.variant_colors_map.setdefault(variant, color)

    stopping_df = load_stopping_summary(args, variants)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    aggregate_rows = []
    run_count_rows = []
    color_by_variant: dict[str, str] = {}
    emphasis_legend_labels: set[str] = set()
    emph_vars = _emphasized_variants(args)

    for variant in variants:
        vg = df[df[args.group_by].astype(str) == variant]
        n_runs = int(vg["run_id"].nunique())
        run_count_rows.append({"variant": variant, "n_runs": n_runs})
        x_grid, mean, std, stderr, n_grid = aggregate_variant(vg, args.x_axis, args.y_axis, args.grid_size)
        if x_grid.size == 0:
            print(f"WARNING: no plottable points for {variant}")
            continue

        legend_label = display_variant_label(variant, n_runs, args)
        if str(variant) in emph_vars:
            emphasis_legend_labels.add(legend_label)

        st = variant_plot_style(str(variant), args)
        plot_kw: dict[str, float | str] = {
            "linewidth": float(st["linewidth"]),
            "zorder": float(st["zorder"]),
        }
        if st["color"] is not None:
            plot_kw["color"] = str(st["color"])

        line = ax.plot(x_grid, mean, label=legend_label, **plot_kw)[0]
        color_by_variant[variant] = line.get_color()

        if args.range != "none":
            band = std if args.range == "std" else stderr
            ax.fill_between(
                x_grid,
                mean - band,
                mean + band,
                alpha=0.15,
                color=line.get_color(),
                zorder=float(st["zorder"]) - 0.5,
                linewidth=0,
            )

        for x, m, s, se, n in zip(x_grid, mean, std, stderr, n_grid):
            aggregate_rows.append(
                {
                    "variant": variant,
                    "x_axis": args.x_axis,
                    "x": float(x),
                    "mean": float(m),
                    "std": float(s),
                    "stderr": float(se),
                    "n": int(n),
                }
            )

    _plot_stopping(ax, stopping_df, color_by_variant, args)

    title = args.title or default_title(args)
    subtitle = default_subtitle(args)
    if subtitle:
        title = f"{title}\n{subtitle}"

    ax.set_title(title, pad=12)
    ax.set_xlabel(axis_label(args.x_axis))
    ax.set_ylabel(axis_label(args.y_axis))
    if args.y_limit_min is not None or args.y_limit_max is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(args.y_limit_min if args.y_limit_min is not None else lo, args.y_limit_max if args.y_limit_max is not None else hi)

    ax.grid(True, alpha=0.25)
    leg = ax.legend(
        loc=args.legend_loc,
        ncol=max(1, int(args.legend_ncol)),
        frameon=True,
        framealpha=0.90,
        borderpad=0.6,
        handlelength=2.4,
    )
    if emphasis_legend_labels:
        for text in leg.get_texts():
            if text.get_text() in emphasis_legend_labels:
                text.set_fontweight("bold")
    fig.tight_layout()

    name_parts = [args.dataset or "all", f"seed{args.matrix_seed or 'all'}", args.y_axis, "vs", args.x_axis]
    if args.variant_regex:
        name_parts.append(safe_token(args.variant_regex)[:80])
    if args.variants:
        name_parts.append("selected")
    if args.stop_band != "none":
        name_parts.append(f"stop_{args.stop_band}")
    stem = "_".join(name_parts)

    out_png = args.out_dir / f"{stem}.png"
    out_curves_csv = args.out_dir / f"{stem}_aggregated_curves.csv"
    out_stopping_csv = args.out_dir / f"{stem}_stopping_summary.csv"
    out_counts_csv = args.out_dir / f"{stem}_run_counts.csv"

    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    pd.DataFrame(aggregate_rows).to_csv(out_curves_csv, index=False)
    pd.DataFrame(run_count_rows).to_csv(out_counts_csv, index=False)

    if not stopping_df.empty:
        stopping_df.to_csv(out_stopping_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "variant",
                "x_axis",
                "stop_col",
                "mean_stop_x",
                "median_stop_x",
                "std_stop_x",
                "stderr_stop_x",
                "q_low",
                "q_high",
                "q_low_stop_x",
                "q_high_stop_x",
                "min_stop_x",
                "max_stop_x",
                "n_stopped",
                "n_total",
            ]
        ).to_csv(out_stopping_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_curves_csv}")
    print(f"Wrote {out_stopping_csv}")
    print(f"Wrote {out_counts_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
