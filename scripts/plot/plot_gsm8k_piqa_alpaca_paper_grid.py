#!/usr/bin/env python3
"""Paper-style 2x3 figure for GSM8K, PIQA, and AlpacaEval.

Figure layout:
  columns: GSM8K, PIQA, AlpacaEval
  rows:    unit-cost, cost-aware

This script reuses the plotting/aggregation helpers from
``plot_gsm8k_piqa_paper_grid.py`` and only changes the layout plus input
directories. Alpaca defaults to the var0.02 Gittins-S download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_gsm8k_piqa_paper_grid as pg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--gsm8k-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/ucb_gittins/gsm8k"))
    p.add_argument("--gsm8k-lrf-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/lrf/gsm8k_lrf"))
    p.add_argument("--gsm8k-bo-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/bo_baseline_5pct/gsm8k_bo"))

    p.add_argument("--piqa-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/ucb_gittins/piqa"))
    p.add_argument("--piqa-lrf-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/lrf/piqa_lrf"))
    p.add_argument("--piqa-bo-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/bo_baseline_5pct/piqa_bo"))

    p.add_argument("--alpaca-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/ucb_gittins/alpaca_var0.02"))
    p.add_argument("--alpaca-lrf-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/lrf/alpaca_lrf"))
    p.add_argument("--alpaca-bo-dir", type=Path, default=Path(r"outputs/wandb_downloads_new/bo_baseline_5pct/alpaca_bo"))

    p.add_argument("--out-dir", type=Path, default=Path(r"outputs/figure/new_figure/final/gsm8k_piqa_alpaca_2x3"))

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lrf-batch-size", type=int, default=32)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--bo-pbgi-unit-variant", default="pbgi_unit")
    p.add_argument("--bo-logei-unit-variant", default="logei_unit")
    p.add_argument("--bo-pbgi-cost-variant", default="pbgi_cost")
    p.add_argument("--bo-logei-cost-variant", default="logeipc_cost")
    p.add_argument("--grid-size", type=int, default=350)

    p.add_argument("--range", choices=["stderr", "std", "none"], default="stderr")
    p.add_argument("--stderr-k", type=float, default=1.0)
    p.add_argument("--show-stopping", action="store_true", default=True)
    p.add_argument("--no-show-stopping", dest="show_stopping", action="store_false")
    p.add_argument("--stop-band", choices=["stderr", "std", "iqr", "none"], default="stderr")
    p.add_argument("--stop-alpha", type=float, default=0.12)
    p.add_argument("--stop-line-alpha", type=float, default=0.72)

    p.add_argument("--fig-width", type=float, default=35.0)
    p.add_argument("--fig-height", type=float, default=13.131295)
    p.add_argument("--title-size", type=float, default=54)
    p.add_argument("--label-size", type=float, default=49)
    p.add_argument("--tick-size", type=float, default=34)
    p.add_argument("--legend-size", type=float, default=39)
    p.add_argument("--row-label-size", type=float, default=50)
    p.add_argument("--line-width-mult", type=float, default=3.0)

    p.add_argument("--y-limit-min", type=float, default=None)
    p.add_argument("--y-limit-max", type=float, default=None)

    return p.parse_args()


def make_legend(
    fig: plt.Figure,
    legend_handles: dict[str, plt.Line2D],
    args: argparse.Namespace,
) -> None:
    legend_order = [
        "gittins_data",
        "gittins_default",
        "ucb",
        "lrf",
        "bo_pbgi_unit",
        "bo_logei_unit",
    ]
    final_handles = []
    for k in legend_order:
        if k not in legend_handles:
            continue
        is_gittins = k in {"gittins_data", "gittins_default"}
        handle = Line2D(
            [0],
            [0],
            color=pg.STYLE_BY_KIND[k]["color"],
            linestyle="-",
            linewidth=7.2 if is_gittins else 6.0,
        )
        final_handles.append(handle)
    final_labels = [pg.STYLE_BY_KIND[k]["label"] for k in legend_order if k in legend_handles]

    if args.show_stopping:
        final_handles.append(
            Line2D(
                [0],
                [0],
                color=pg.COLOR_GITTINS_S,
                linestyle="--",
                linewidth=6.8,
                alpha=args.stop_line_alpha,
            )
        )
        final_labels.append("Gittins-S mean stop")

        final_handles.append(
            Line2D(
                [0],
                [0],
                color=pg.COLOR_GITTINS_G,
                linestyle="--",
                linewidth=6.8,
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
                    color=pg.STYLE_BY_KIND[kind]["color"],
                    linestyle="--",
                    linewidth=5.8,
                    alpha=args.stop_line_alpha,
                )
            )
            final_labels.append(f"{pg.STYLE_BY_KIND[kind]['label']} mean stop")

    if args.range != "none" or (args.show_stopping and args.stop_band != "none"):
        final_handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.18))
        k = float(args.stderr_k)
        k_txt = str(int(k)) if abs(k - round(k)) < 1e-9 else f"{k:g}"
        band_kind = "SE" if args.range == "stderr" or args.stop_band == "stderr" else "std"
        final_labels.append(rf"$\pm${k_txt} {band_kind} band")

    fig.legend(
        final_handles,
        final_labels,
        loc="center left",
        bbox_to_anchor=(0.840, 0.49),
        ncol=1,
        frameon=False,
        fontsize=args.legend_size,
        handlelength=1.85,
        handletextpad=0.42,
        labelspacing=0.66,
        borderpad=0.4,
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pg.LINEWIDTH_MULT = float(args.line_width_mult)
    pg.GITTINS_LINE_EXTRA_MULT = 1.2
    pg.STYLE_BY_KIND["gittins_data"]["linewidth"] = 3.0
    pg.STYLE_BY_KIND["gittins_default"]["linewidth"] = 3.0

    pg.setup_matplotlib(args)

    datasets = [
        {
            "key": "gsm8k",
            "title": "GSM8K",
            "history_dirs": [args.gsm8k_dir, args.gsm8k_lrf_dir, args.gsm8k_bo_dir],
            "stopping_dirs": [args.gsm8k_dir, args.gsm8k_bo_dir],
        },
        {
            "key": "piqa",
            "title": "PIQA",
            "history_dirs": [args.piqa_dir, args.piqa_lrf_dir, args.piqa_bo_dir],
            "stopping_dirs": [args.piqa_dir, args.piqa_bo_dir],
        },
        {
            "key": "alpaca",
            "title": "AlpacaEval",
            "history_dirs": [args.alpaca_dir, args.alpaca_lrf_dir, args.alpaca_bo_dir],
            "stopping_dirs": [args.alpaca_dir, args.alpaca_bo_dir],
        },
    ]

    histories = {d["key"]: pg.read_histories(d["history_dirs"]) for d in datasets}
    stoppings = {d["key"]: pg.read_stoppings(d["stopping_dirs"]) for d in datasets}

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(args.fig_width, args.fig_height),
        sharey=False,
        constrained_layout=False,
    )

    legend_handles: dict[str, plt.Line2D] = {}
    for col, d in enumerate(datasets):
        handles = pg.plot_panel(
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

        handles = pg.plot_panel(
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

        axes[0, col].set_title(d["title"], fontsize=args.title_size, fontweight="normal", pad=6)

    axes[0, 0].set_ylabel("Simple regret", fontsize=args.label_size)
    axes[1, 0].set_ylabel("Simple regret", fontsize=args.label_size)
    for col in range(3):
        axes[0, col].set_xlabel("Cumulative evaluations", fontsize=args.label_size)
        axes[1, col].set_xlabel("Cumulative cost", fontsize=args.label_size)

    axes[0, 2].text(
        1.04,
        0.5,
        "Unit-cost",
        transform=axes[0, 2].transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=args.row_label_size,
        fontweight="normal",
    )
    axes[1, 2].text(
        1.04,
        0.5,
        "Cost-aware",
        transform=axes[1, 2].transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=args.row_label_size,
        fontweight="normal",
    )

    make_legend(fig, legend_handles, args)

    fig.subplots_adjust(
        left=0.045,
        right=0.790,
        top=0.855,
        bottom=0.16,
        wspace=0.25,
        hspace=0.45,
    )

    stem = (
        f"figure_gsm8k_piqa_alpaca_2x3_B{args.batch_size}_"
        f"LRFB{args.lrf_batch_size}_scale{args.scale}_alpaca_var0p02"
    ).replace(".", "")
    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
