#!/usr/bin/env python3
"""Generate six timing panels in one command.

Outputs one fixed-size timing panel for each pair:

  datasets/groups: GSM8K, PIQA, MMLU-small
  cost modes:      unit, aware

So the script writes 6 PNG files and 6 CSV files:
  OUT_ROOT/unit/gsm8k_unit_B16_scale1e-4_iter_step_median_s_timing.png
  OUT_ROOT/aware/gsm8k_aware_B16_scale1e-4_iter_step_median_s_timing.png
  ...

Default error bars are black ±2 SE across completed runs, matching the paper
simple-regret convention.

Input layouts supported:
  1. ROOT/runs_summary.csv
  2. ROOT/<task>/runs_summary.csv  (for MMLU merged folders)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


COLOR_UCB = "tab:blue"
COLOR_LRF = "tab:purple"
COLOR_GITTINS_S = "tab:orange"  # dataset / specific prior
COLOR_GITTINS_G = "tab:green"   # general default prior

METHOD_STYLE = {
    "Gittins-S": {"color": COLOR_GITTINS_S},
    "Gittins-G": {"color": COLOR_GITTINS_G},
    "UCB-E": {"color": COLOR_UCB},
    "LRF": {"color": COLOR_LRF},
}

METHOD_ORDER = ["Gittins-S", "Gittins-G", "UCB-E", "LRF"]


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--gsm8k-root", type=Path, required=True)
    p.add_argument("--piqa-root", type=Path, required=True)
    p.add_argument("--mmlu-small-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path(r"outputs\wandb_plots\timing\panels_2se"))

    p.add_argument("--gsm8k-batch-size", type=int, default=16)
    p.add_argument("--piqa-batch-size", type=int, default=16)
    p.add_argument("--mmlu-small-batch-size", type=int, default=4)
    p.add_argument("--scale", default="1e-4")

    p.add_argument(
        "--metric",
        default="iter_step_median_s",
        choices=[
            "iter_step_median_s",
            "iter_step_mean_s",
            "iter_step_p90_s",
            "iter_total_median_s",
            "iter_total_mean_s",
            "iter_total_p90_s",
            "lookup_table_s",
        ],
        help="Timing field in runs_summary.csv.",
    )

    # Default paper style: black ±2 SE error bars.
    p.add_argument("--se-mult", type=float, default=2.0)
    p.add_argument("--skip-empty-methods", action="store_true", default=True)
    p.add_argument("--no-skip-empty-methods", dest="skip_empty_methods", action="store_false")

    # Fixed-size small-panel style, matching MMLU panel workflow.
    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--single-width", type=float, default=2.25)
    p.add_argument("--single-height", type=float, default=1.65)
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--title-size", type=float, default=14.5)
    p.add_argument("--tick-size", type=float, default=13.5)

    p.add_argument("--single-left", type=float, default=0.22)
    p.add_argument("--single-right", type=float, default=0.97)
    p.add_argument("--single-bottom", type=float, default=0.17)
    p.add_argument("--single-top", type=float, default=0.82)

    p.add_argument("--show-axis-labels", action="store_true", help="Normally off; assembled plot can add shared labels.")
    p.add_argument("--show-method-labels", action="store_true", help="Normally off; legend is added by assemble script.")
    p.add_argument("--log-y", action="store_true", help="Use log y-scale if small methods are invisible.")
    p.add_argument("--y-min", type=float, default=0.0)
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


def load_summary_root(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    direct = root / "runs_summary.csv"
    if direct.is_file():
        pieces.append(pd.read_csv(direct))

    # MMLU merged layout: ROOT/task/runs_summary.csv
    for p in sorted(root.glob("*/runs_summary.csv")):
        df = pd.read_csv(p)
        if "matrix_task" not in df.columns:
            df["matrix_task"] = p.parent.name
        else:
            df["matrix_task"] = df["matrix_task"].fillna(p.parent.name)
        pieces.append(df)

    if not pieces:
        raise FileNotFoundError(f"No runs_summary.csv found under {root}")

    out = pd.concat(pieces, ignore_index=True)
    if "state" in out.columns:
        out = out[out["state"].astype(str).str.lower().eq("finished")].copy()
    return out


def variant_to_method(batch_size: int, cost_mode: str, scale: str) -> dict[str, str]:
    b = int(batch_size)
    return {
        f"gittins_{cost_mode}_B{b}_scale{scale}_dataset": "Gittins-S",
        f"gittins_{cost_mode}_B{b}_scale{scale}_default": "Gittins-G",
        f"ucb_B{b}": "UCB-E",
        f"lrf_B{b}": "LRF",
    }


def summarize_timing(
    *,
    df: pd.DataFrame,
    label: str,
    batch_size: int,
    cost_mode: str,
    scale: str,
    metric: str,
    se_mult: float,
    skip_empty_methods: bool,
) -> pd.DataFrame:
    if "experiment_variant" not in df.columns:
        raise ValueError("runs_summary.csv must contain experiment_variant.")
    if metric not in df.columns:
        raise ValueError(f"runs_summary.csv does not contain metric column: {metric}")

    mapping = variant_to_method(batch_size, cost_mode, scale)
    sub = df[df["experiment_variant"].astype(str).isin(mapping.keys())].copy()
    sub["method"] = sub["experiment_variant"].astype(str).map(mapping)
    sub["timing_s"] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub[np.isfinite(sub["timing_s"])].copy()

    rows = []
    for method in METHOD_ORDER:
        vals = sub.loc[sub["method"] == method, "timing_s"].dropna().to_numpy(dtype=float)
        if skip_empty_methods and len(vals) == 0:
            continue

        center = float(np.median(vals)) if len(vals) else np.nan
        mean = float(np.mean(vals)) if len(vals) else np.nan
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        se = std / np.sqrt(max(len(vals), 1)) if len(vals) else 0.0
        err = float(se_mult) * float(se)

        rows.append({
            "panel": label,
            "method": method,
            "cost_mode": cost_mode,
            "batch_size": int(batch_size),
            "scale": scale,
            "metric": metric,
            "error_bar": f"±{se_mult:g} SE",
            "n_runs": int(len(vals)),
            "median_s": center,
            "mean_s": mean,
            "std_s": std,
            "se_s": se,
            "err_low_s": err if np.isfinite(center) else 0.0,
            "err_high_s": err if np.isfinite(center) else 0.0,
        })
    return pd.DataFrame(rows)


def y_label_for(metric: str) -> str:
    if metric == "lookup_table_s":
        return "Lookup precompute time (s)"
    if "total" in metric:
        return "Per-batch total time (s)"
    if "p90" in metric:
        return "Per-batch decision time, p90 (s)"
    if "mean" in metric:
        return "Per-batch decision time, mean (s)"
    return "Per-batch decision time, median (s)"


def plot_one_panel(
    *,
    stats: pd.DataFrame,
    label: str,
    out_dir: Path,
    cost_mode: str,
    batch_size: int,
    scale: str,
    metric: str,
    args: argparse.Namespace,
) -> None:
    if stats.empty:
        raise SystemExit(f"No matching timing rows for {label} {cost_mode}. Check batch size / scale / data root.")

    token = f"{safe_token(label.lower())}_{cost_mode}_B{batch_size}_scale{safe_token(scale)}_{safe_token(metric)}"
    out_png = out_dir / cost_mode / f"{token}_timing.png"
    out_pdf = out_png.with_suffix(".pdf")
    out_csv = out_dir / cost_mode / f"{token}_timing.csv"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(float(args.single_width), float(args.single_height)))

    x = np.arange(len(stats), dtype=float)
    heights = stats["median_s"].to_numpy(dtype=float)
    yerr = np.vstack([
        stats["err_low_s"].to_numpy(dtype=float),
        stats["err_high_s"].to_numpy(dtype=float),
    ])
    colors = [METHOD_STYLE[m]["color"] for m in stats["method"].tolist()]

    ax.bar(
        x,
        heights,
        yerr=yerr,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        ecolor="black",
        capsize=3,
        width=0.72,
    )

    ax.set_title(label, fontsize=float(args.title_size), pad=3)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=float(args.tick_size), width=0.9, length=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    if args.show_method_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(stats["method"].tolist(), rotation=25, ha="right", fontsize=max(8, float(args.tick_size) - 2))
    else:
        ax.set_xticks([])

    if args.show_axis_labels:
        ax.set_ylabel(y_label_for(metric))
        ax.set_xlabel("Method")
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")

    if args.log_y:
        ax.set_yscale("log")
    else:
        if args.y_min is not None:
            ax.set_ylim(bottom=float(args.y_min))
    if args.y_max is not None:
        ax.set_ylim(top=float(args.y_max))

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    fig.subplots_adjust(
        left=float(args.single_left),
        right=float(args.single_right),
        bottom=float(args.single_bottom),
        top=float(args.single_top),
    )
    fig.savefig(out_png, dpi=int(args.dpi))
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Wrote timing panel: {out_png}")
    print(f"Wrote timing stats: {out_csv}")


def main() -> int:
    args = parse_args()
    setup_matplotlib(args)
    args.out_root.mkdir(parents=True, exist_ok=True)

    groups = [
        ("GSM8K", args.gsm8k_root, int(args.gsm8k_batch_size)),
        ("PIQA", args.piqa_root, int(args.piqa_batch_size)),
        ("MMLU-small", args.mmlu_small_root, int(args.mmlu_small_batch_size)),
    ]

    # Read once per group, then generate unit and aware panels.
    for label, root, batch_size in groups:
        df = load_summary_root(root)
        for cost_mode in ("unit", "aware"):
            stats = summarize_timing(
                df=df,
                label=label,
                batch_size=batch_size,
                cost_mode=cost_mode,
                scale=str(args.scale),
                metric=str(args.metric),
                se_mult=float(args.se_mult),
                skip_empty_methods=bool(args.skip_empty_methods),
            )
            plot_one_panel(
                stats=stats,
                label=label,
                out_dir=args.out_root,
                cost_mode=cost_mode,
                batch_size=batch_size,
                scale=str(args.scale),
                metric=str(args.metric),
                args=args,
            )

    print("Done. Wrote 3 groups × 2 cost modes = 6 timing panels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
