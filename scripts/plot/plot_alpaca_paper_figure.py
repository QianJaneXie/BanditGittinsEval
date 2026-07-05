#!/usr/bin/env python3
"""Paper-style AlpacaEval figure: unit-cost (top) + cost-aware (bottom).

Reuses styling/aggregation from ``plot_gsm8k_piqa_paper_grid.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_gsm8k_piqa_paper_grid as pg  # noqa: E402


def alpaca_variant_names(
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--alpaca-dir",
        type=Path,
        default=Path(r"outputs/wandb_downloads_new/ucb_gittins/alpaca_840"),
    )
    p.add_argument(
        "--alpaca-lrf-dir",
        type=Path,
        default=Path(r"outputs/wandb_downloads_new/lrf/alpaca_lrf"),
    )
    p.add_argument(
        "--alpaca-bo-dir",
        type=Path,
        default=Path(r"outputs/wandb_downloads_new/bo_baseline_5pct/alpaca_bo"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"outputs/figure/new_figure/alpaca_final_current"),
    )
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
    p.add_argument("--fig-width", type=float, default=26.8)
    p.add_argument("--fig-height", type=float, default=15.8)
    p.add_argument("--title-size", type=float, default=81)
    p.add_argument("--label-size", type=float, default=81)
    p.add_argument("--tick-size", type=float, default=57)
    p.add_argument("--legend-size", type=float, default=58)
    p.add_argument("--row-label-size", type=float, default=81)
    p.add_argument("--y-limit-min", type=float, default=None)
    p.add_argument("--y-limit-max", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pg.variant_names = alpaca_variant_names  # type: ignore[assignment]

    pg.setup_matplotlib(args)

    history = pg.read_histories([args.alpaca_dir, args.alpaca_lrf_dir, args.alpaca_bo_dir])
    stopping = pg.read_stoppings([args.alpaca_dir, args.alpaca_bo_dir])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(args.fig_width, args.fig_height),
        sharey=False,
        constrained_layout=False,
    )

    legend_handles: dict[str, plt.Line2D] = {}
    legend_handles.update(
        pg.plot_panel(
            axes[0],
            history,
            stopping,
            dataset="alpaca",
            dataset_title="AlpacaEval",
            cost_mode="unit",
            x_axis="cum_eval",
            args=args,
        )
    )
    legend_handles.update(
        pg.plot_panel(
            axes[1],
            history,
            stopping,
            dataset="alpaca",
            dataset_title="AlpacaEval",
            cost_mode="aware",
            x_axis="cum_original_cost",
            args=args,
        )
    )

    panel_title_size = args.title_size * 0.82
    axis_label_size = args.title_size * 0.82
    axes[0].set_title("Unit-cost", fontsize=panel_title_size, fontweight="normal", pad=8)
    axes[1].set_title("Cost-aware", fontsize=panel_title_size, fontweight="normal", pad=8)
    axes[0].set_ylabel("Simple regret", fontsize=axis_label_size)
    axes[1].set_ylabel("")
    axes[0].set_xlabel("Cumulative evaluations", fontsize=axis_label_size)
    axes[1].set_xlabel("Cumulative cost", fontsize=axis_label_size)

    legend_order = [
        "gittins_data",
        "gittins_default",
        "ucb",
        "lrf",
        "bo_pbgi_unit",
        "bo_logei_unit",
    ]
    final_handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    final_labels = [pg.STYLE_BY_KIND[k]["label"] for k in legend_order if k in legend_handles]

    if args.show_stopping:
        if "gittins_data" in legend_handles:
            final_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=pg.COLOR_GITTINS_S,
                    linestyle="--",
                    linewidth=2.6 * float(pg.LINEWIDTH_MULT) * float(pg.GITTINS_LINE_EXTRA_MULT),
                    alpha=args.stop_line_alpha,
                )
            )
            final_labels.append("Gittins-S mean stop")

        if "gittins_default" in legend_handles:
            final_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=pg.COLOR_GITTINS_G,
                    linestyle="--",
                    linewidth=2.6 * float(pg.LINEWIDTH_MULT) * float(pg.GITTINS_LINE_EXTRA_MULT),
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
                    linewidth=2.6 * float(pg.LINEWIDTH_MULT),
                    alpha=args.stop_line_alpha,
                )
            )
            final_labels.append(f"{pg.STYLE_BY_KIND[kind]['label']} mean stop")

    if args.range != "none" or (args.show_stopping and args.stop_band != "none"):
        final_handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.18))
        k = float(args.stderr_k)
        if args.range == "stderr" or args.stop_band == "stderr":
            k_txt = str(int(k)) if abs(k - round(k)) < 1e-9 else f"{k:g}"
            final_labels.append(rf"$\pm${k_txt} SE band")
        else:
            final_labels.append(r"$\pm$1 std band")


    fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=args.legend_size,
        handlelength=2.0,
        handletextpad=0.35,
        columnspacing=1.15,
        borderpad=0.55,
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.51, wspace=0.22)

    stem = (
        f"figure_alpaca_B{args.batch_size}_LRFB{args.lrf_batch_size}_scale{args.scale}".replace(".", "")
    )
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
