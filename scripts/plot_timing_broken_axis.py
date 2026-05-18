#!/usr/bin/env python3
"""Per-batch decision time as a broken-y-axis bar chart.

Each dataset is one column.  Within a column the y-axis is split:

  - Top sub-panel:    "Time (s)"       shows the LRF bar (slow).
  - Bottom sub-panel: "Time (microseconds)" shows Gittins-S, Gittins-G, UCB-E.

The bottom sub-panel's upper limit (in microseconds) and the top
sub-panel's lower limit (in seconds) are tied to the same physical
value, so the diagonal break marks line up exactly.  By default LRF is
de-emphasised with reduced alpha because it is so much larger than the
other methods that comparing them is the point of the plot.

Inputs are the same `runs_summary.csv` files used by
`plot_timing_six_panels.py`, so MMLU merged folders work out of the box.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


COLOR_UCB = "tab:blue"
COLOR_LRF = "tab:purple"
COLOR_GITTINS_S = "tab:orange"
COLOR_GITTINS_G = "tab:green"

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

    p.add_argument(
        "--out",
        type=Path,
        default=Path(r"outputs\wandb_plots\timing\timing_broken_axis.png"),
        help="Output PNG path.  PDF is written next to it automatically.",
    )

    p.add_argument("--cost-mode", choices=["unit", "aware"], default="unit")
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
    )
    p.add_argument("--se-mult", type=float, default=2.0)

    # Figure layout.
    p.add_argument("--fig-width", type=float, default=12.0)
    p.add_argument("--fig-height", type=float, default=4.6)
    p.add_argument("--height-ratio-top", type=float, default=1.0)
    p.add_argument("--height-ratio-bottom", type=float, default=1.4)

    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--title-size", type=float, default=18)
    p.add_argument("--tick-size", type=float, default=14)
    p.add_argument("--ylabel-size", type=float, default=17)
    p.add_argument("--legend-size", type=float, default=16)
    p.add_argument("--dpi", type=int, default=260)

    p.add_argument("--left", type=float, default=0.075)
    p.add_argument("--right", type=float, default=0.985)
    p.add_argument("--top", type=float, default=0.88)
    p.add_argument("--bottom", type=float, default=0.18)
    p.add_argument("--wspace", type=float, default=0.26)
    p.add_argument("--hspace", type=float, default=0.06)
    p.add_argument("--legend-y", type=float, default=0.02)

    # Y-limit options.
    p.add_argument(
        "--shared-bottom",
        action="store_true",
        default=True,
        help="If set, all microsecond panels share the same upper limit.",
    )
    p.add_argument("--no-shared-bottom", dest="shared_bottom", action="store_false")
    p.add_argument(
        "--shared-top",
        action="store_true",
        default=False,
        help="If set, all second panels share the same upper limit (LRF max).",
    )
    p.add_argument(
        "--bottom-max-us",
        type=float,
        default=None,
        help="If set, fixed upper limit of the microsecond panel (in μs).",
    )
    p.add_argument(
        "--top-max-s",
        type=float,
        default=None,
        help="If set, fixed upper limit of the second panel (in s).",
    )
    p.add_argument(
        "--top-min-frac",
        type=float,
        default=0.55,
        help="When LRF is the only thing in the top panel, top_min = LRF * top-min-frac.  "
             "Lower values make the LRF bar appear taller within the top sub-panel.",
    )

    # LRF de-emphasis (off by default: use solid tab colors).
    p.add_argument("--de-emphasize-lrf", action="store_true", default=False)
    p.add_argument("--no-de-emphasize-lrf", dest="de_emphasize_lrf", action="store_false")
    p.add_argument("--lrf-alpha", type=float, default=0.55)

    # Misc.
    p.add_argument(
        "--break-mark-d",
        type=float,
        default=0.018,
        help="Half-length of the diagonal break tick (axes-fraction units).",
    )

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
    batch_size: int,
    cost_mode: str,
    scale: str,
    metric: str,
    se_mult: float,
) -> pd.DataFrame:
    if "experiment_variant" not in df.columns:
        raise ValueError("runs_summary.csv must contain experiment_variant.")
    if metric not in df.columns:
        raise ValueError(f"runs_summary.csv missing metric: {metric}")

    mapping = variant_to_method(batch_size, cost_mode, scale)
    sub = df[df["experiment_variant"].astype(str).isin(mapping.keys())].copy()
    sub["method"] = sub["experiment_variant"].astype(str).map(mapping)
    sub["t"] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub[np.isfinite(sub["t"])]

    rows = []
    for method in METHOD_ORDER:
        vals = sub.loc[sub["method"] == method, "t"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            rows.append({"method": method, "median_s": np.nan, "err_s": 0.0, "n": 0})
            continue
        median = float(np.median(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        se = std / np.sqrt(len(vals))
        rows.append({
            "method": method,
            "median_s": median,
            "err_s": se_mult * se,
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows).set_index("method").reindex(METHOD_ORDER).reset_index()


def compute_panel_limits(
    all_stats: list[tuple[str, pd.DataFrame]],
    args: argparse.Namespace,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Return per-panel (bottom_max_us, (top_min_s, top_max_s))."""
    bottom_lims: list[float] = []
    top_lims: list[tuple[float, float]] = []

    for _, stats in all_stats:
        non_lrf = stats[stats["method"] != "LRF"].dropna(subset=["median_s"])
        if len(non_lrf) > 0:
            top_us = float(((non_lrf["median_s"] + non_lrf["err_s"]) * 1e6).max()) * 1.30
        else:
            top_us = 100.0
        bottom_lims.append(top_us)

        lrf = stats[stats["method"] == "LRF"].dropna(subset=["median_s"])
        if len(lrf) > 0:
            v = float(lrf["median_s"].iloc[0])
            err = float(lrf["err_s"].iloc[0])
            top_max = (v + err) * 1.18
        else:
            top_max = 1.0
        top_lims.append((0.0, top_max))  # placeholder, top_min recomputed after sharing

    if args.shared_bottom and bottom_lims:
        v = max(bottom_lims)
        bottom_lims = [v] * len(bottom_lims)
    if args.shared_top and top_lims:
        v = max(t for _, t in top_lims)
        top_lims = [(b, v) for b, _ in top_lims]
    if args.bottom_max_us is not None:
        bottom_lims = [float(args.bottom_max_us)] * len(bottom_lims)
    if args.top_max_s is not None:
        top_lims = [(b, float(args.top_max_s)) for b, _ in top_lims]

    # Tie top_min_s to bottom_max_us so the break is at the same physical value.
    top_lims = [
        (bottom_lims[i] * 1e-6, top_lims[i][1])
        for i in range(len(top_lims))
    ]
    return bottom_lims, top_lims


def draw_break_marks(ax_top, ax_bot, *, d: float) -> None:
    kwargs = dict(color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)


def y_label_units() -> tuple[str, str]:
    """Return (top_unit_label, bottom_unit_label)."""
    return "Time (s)", "Time (\u03bcs)"  # μs


def plot(args: argparse.Namespace) -> None:
    setup_matplotlib(args)

    groups = [
        ("GSM8K", args.gsm8k_root, int(args.gsm8k_batch_size)),
        ("PIQA", args.piqa_root, int(args.piqa_batch_size)),
        ("MMLU-small", args.mmlu_small_root, int(args.mmlu_small_batch_size)),
    ]

    all_stats: list[tuple[str, pd.DataFrame]] = []
    for label, root, batch_size in groups:
        df = load_summary_root(root)
        stats = summarize_timing(
            df=df,
            batch_size=batch_size,
            cost_mode=str(args.cost_mode),
            scale=str(args.scale),
            metric=str(args.metric),
            se_mult=float(args.se_mult),
        )
        all_stats.append((label, stats))

    bottom_lims, top_lims = compute_panel_limits(all_stats, args)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(float(args.fig_width), float(args.fig_height)),
        gridspec_kw={
            "height_ratios": [float(args.height_ratio_top), float(args.height_ratio_bottom)],
        },
    )

    top_unit, bot_unit = y_label_units()
    x = np.arange(len(METHOD_ORDER), dtype=float)
    colors = [METHOD_STYLE[m]["color"] for m in METHOD_ORDER]
    alphas = [
        float(args.lrf_alpha) if (m == "LRF" and bool(args.de_emphasize_lrf)) else 1.0
        for m in METHOD_ORDER
    ]

    for col, (label, stats) in enumerate(all_stats):
        ax_top = axes[0][col]
        ax_bot = axes[1][col]

        heights_s = stats["median_s"].to_numpy(dtype=float)
        err_s = stats["err_s"].to_numpy(dtype=float)

        # Top sub-panel: seconds (LRF dominates here).
        bars_top = ax_top.bar(
            x, heights_s, yerr=err_s,
            color=colors, edgecolor="black", linewidth=0.7,
            ecolor="black", capsize=3, width=0.72,
        )
        for patch, a in zip(bars_top, alphas):
            patch.set_alpha(a)

        # Bottom sub-panel: microseconds (small methods are visible here).
        bars_bot = ax_bot.bar(
            x, heights_s * 1e6, yerr=err_s * 1e6,
            color=colors, edgecolor="black", linewidth=0.7,
            ecolor="black", capsize=3, width=0.72,
        )
        for patch, a in zip(bars_bot, alphas):
            patch.set_alpha(a)

        ax_bot.set_ylim(0.0, float(bottom_lims[col]))
        ax_top.set_ylim(float(top_lims[col][0]), float(top_lims[col][1]))

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_bot.set_xticks([])

        draw_break_marks(ax_top, ax_bot, d=float(args.break_mark_d))

        ax_top.set_title(label, fontsize=float(args.title_size), pad=4)

        for ax in (ax_top, ax_bot):
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=float(args.tick_size), width=0.9, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=2))
        ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=3))

        if col == 0:
            ax_top.set_ylabel(top_unit, fontsize=float(args.ylabel_size))
            ax_bot.set_ylabel(bot_unit, fontsize=float(args.ylabel_size))

    fig.subplots_adjust(
        left=float(args.left),
        right=float(args.right),
        top=float(args.top),
        bottom=float(args.bottom),
        wspace=float(args.wspace),
        hspace=float(args.hspace),
    )

    handles: list[object] = [
        Line2D(
            [0], [0],
            color=METHOD_STYLE[m]["color"],
            linewidth=8.0,
            solid_capstyle="butt",
            alpha=(float(args.lrf_alpha) if (m == "LRF" and bool(args.de_emphasize_lrf)) else 1.0),
        )
        for m in METHOD_ORDER
    ]
    handles.append(
        Line2D([0], [0], color="black", marker="_", markersize=18, linewidth=1.4)
    )
    labels = list(METHOD_ORDER) + [f"\u00b1{args.se_mult:g} SE"]

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, float(args.legend_y)),
        fontsize=float(args.legend_size),
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=1.4,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi))
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.out.with_suffix('.pdf')}")


def main() -> int:
    args = parse_args()
    plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
