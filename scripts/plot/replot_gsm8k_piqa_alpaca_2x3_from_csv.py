#!/usr/bin/env python3
"""Restyle the paper 2x3 figure from its saved aggregate CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

import plot_gsm8k_piqa_alpaca_paper_grid as grid
import plot_gsm8k_piqa_paper_grid as pg


def parse_args() -> argparse.Namespace:
    default_dir = Path(
        "outputs/figure/new_figure/final/"
        "gsm8k_piqa_alpaca_2x3_latest_Gs_sysrs"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=default_dir)
    p.add_argument("--out-dir", type=Path, default=default_dir)
    p.add_argument("--stem", default="figure_gsm8k_piqa_alpaca_2x3_restyled")
    p.add_argument("--color-gittins-s", default="tab:orange")
    p.add_argument("--color-gittins-g", default="tab:green")
    p.add_argument("--color-sysrs", default="tab:pink")
    p.add_argument("--color-ucb", default="tab:blue")
    p.add_argument("--color-lrf", default="tab:purple")
    p.add_argument("--color-bo-pbgi", default="tab:olive")
    p.add_argument("--color-bo-logei", default="tab:brown")
    p.add_argument("--no-range", action="store_true")
    p.add_argument("--no-show-stopping", action="store_true")
    p.add_argument(
        "--stop-band", choices=["stderr", "std", "iqr", "none"], default="stderr"
    )
    p.add_argument("--stderr-k", type=float, default=1.0)
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
        pg.STYLE_BY_KIND[kind]["color"] = color
    pg.COLOR_GITTINS_S = args.color_gittins_s
    pg.COLOR_GITTINS_G = args.color_gittins_g


def draw_stops(
    ax: plt.Axes, stops: pd.DataFrame, args: argparse.Namespace
) -> None:
    if args.no_show_stopping or stops.empty:
        return
    for row in stops.itertuples(index=False):
        kind = str(row.kind)
        color = pg.STYLE_BY_KIND[kind]["color"]
        mean = float(row.mean_stop)
        if args.stop_band == "stderr":
            delta = float(row.stderr_stop) * args.stderr_k
            lo, hi = mean - delta, mean + delta
        elif args.stop_band == "std":
            lo, hi = mean - float(row.std_stop), mean + float(row.std_stop)
        elif args.stop_band == "iqr":
            lo, hi = float(row.q25_stop), float(row.q75_stop)
        else:
            lo = hi = mean
        if hi > lo:
            ax.axvspan(
                lo,
                hi,
                color=color,
                alpha=args.stop_alpha,
                linewidth=0,
                zorder=1,
            )
        extra = pg.GITTINS_LINE_EXTRA_MULT if kind.startswith("gittins_") else 1.0
        ax.axvline(
            mean,
            color=color,
            linestyle="--",
            linewidth=2.6 * pg.LINEWIDTH_MULT * extra,
            alpha=args.stop_line_alpha,
            zorder=2,
        )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.data_dir / "plot_data_curves.csv")
    stops_path = args.data_dir / "plot_data_stopping.csv"
    stops = pd.read_csv(stops_path) if stops_path.exists() else pd.DataFrame()

    set_colors(args)
    pg.LINEWIDTH_MULT = float(args.line_width_mult)
    pg.GITTINS_LINE_EXTRA_MULT = 1.2
    pg.STYLE_BY_KIND["gittins_data"]["linewidth"] = 3.0
    pg.STYLE_BY_KIND["gittins_default"]["linewidth"] = 3.0
    pg.setup_matplotlib(args)

    datasets = [
        ("gsm8k", "GSM8K"),
        ("piqa", "PIQA"),
        ("alpaca", "AlpacaEval"),
    ]
    panels = [
        ("unit", "cum_eval"),
        ("aware", "cum_original_cost"),
    ]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(args.fig_width, args.fig_height),
        sharey=False,
        constrained_layout=False,
    )
    handles: dict[str, plt.Line2D] = {}

    for col, (dataset, title) in enumerate(datasets):
        for row, (cost_mode, x_axis) in enumerate(panels):
            ax = axes[row, col]
            panel = curves[
                (curves["dataset"] == dataset)
                & (curves["cost_mode"] == cost_mode)
                & (curves["x_axis"] == x_axis)
            ]
            for kind in grid.CURVE_ORDER:
                kg = panel[panel["kind"] == kind].sort_values("x")
                if kg.empty:
                    continue
                style = pg.STYLE_BY_KIND[kind]
                extra = (
                    pg.GITTINS_LINE_EXTRA_MULT
                    if kind.startswith("gittins_")
                    else 1.0
                )
                line = ax.plot(
                    kg["x"],
                    kg["mean"],
                    color=style["color"],
                    linewidth=style["linewidth"] * pg.LINEWIDTH_MULT * extra,
                    label=style["label"],
                    zorder=style["zorder"],
                )[0]
                handles[kind] = line
                if not args.no_range:
                    ax.fill_between(
                        kg["x"].to_numpy(dtype=float),
                        kg["band_lo"].to_numpy(dtype=float),
                        kg["band_hi"].to_numpy(dtype=float),
                        color=style["color"],
                        alpha=0.15,
                        linewidth=0,
                        zorder=style["zorder"] - 0.5,
                    )

            panel_stops = (
                stops[
                    (stops["dataset"] == dataset)
                    & (stops["cost_mode"] == cost_mode)
                    & (stops["x_axis"] == x_axis)
                ]
                if not stops.empty
                else stops
            )
            draw_stops(ax, panel_stops, args)
            ax.grid(True, alpha=0.23, linewidth=0.9)
            ax.tick_params(
                axis="both", labelsize=args.tick_size, width=1.2, length=6
            )
            ax.xaxis.set_major_formatter(FuncFormatter(pg.clean_tick_label))
            ax.yaxis.set_major_formatter(FuncFormatter(pg.clean_tick_label))
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
        axes[0, col].set_title(
            title, fontsize=args.title_size, fontweight="normal", pad=6
        )

    axes[0, 0].set_ylabel("Simple regret", fontsize=args.label_size)
    axes[1, 0].set_ylabel("Simple regret", fontsize=args.label_size)
    for col in range(3):
        axes[0, col].set_xlabel(
            "Cumulative evaluations", fontsize=args.label_size
        )
        axes[1, col].set_xlabel("Cumulative cost", fontsize=args.label_size)
    for row, text in enumerate(["Unit-cost", "Cost-aware"]):
        axes[row, 2].text(
            1.04,
            0.5,
            text,
            transform=axes[row, 2].transAxes,
            rotation=-90,
            va="center",
            ha="left",
            fontsize=args.row_label_size,
            fontweight="normal",
        )

    args.show_stopping = not args.no_show_stopping
    args.range = "none" if args.no_range else "stderr"
    grid.make_legend(fig, handles, args)
    fig.subplots_adjust(
        left=0.045,
        right=0.790,
        top=0.855,
        bottom=0.16,
        wspace=0.25,
        hspace=0.45,
    )
    out_png = args.out_dir / f"{args.stem}.png"
    out_pdf = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
