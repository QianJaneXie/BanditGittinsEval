#!/usr/bin/env python3
"""Assemble fixed-size timing panels into one grid figure.

This script reads PNGs produced by `plot_timing_panel.py`.
It does not re-read W&B data.

Example:
python scripts/assemble_timing_panels.py --panel-root outputs\\wandb_plots\\timing\\panels\\unit --out outputs\\wandb_plots\\timing\\timing_unit_assembled.png --grid-cols 3
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--panels", nargs="+", default=None, help="Optional explicit panel PNG filenames or paths in order.")

    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--fig-width", type=float, default=11.5)
    p.add_argument("--fig-height", type=float, default=0.0, help="0 = auto height.")
    p.add_argument("--panel-aspect", type=float, default=2.25 / 1.65)
    p.add_argument("--dpi", type=int, default=260)

    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--shared-label-size", type=float, default=24)
    p.add_argument("--legend-size", type=float, default=20)
    p.add_argument("--legend-ncol", type=int, default=5)
    p.add_argument("--legend-y", type=float, default=0.02)
    p.add_argument("--legend-x", type=float, default=0.5)
    p.add_argument("--legend-columnspacing", type=float, default=1.15)
    p.add_argument("--legend-handlelength", type=float, default=2.1)
    p.add_argument("--legend-handletextpad", type=float, default=0.55)

    p.add_argument("--left", type=float, default=0.075)
    p.add_argument("--right", type=float, default=0.985)
    p.add_argument("--top", type=float, default=0.955)
    p.add_argument("--bottom", type=float, default=0.24)
    p.add_argument("--wspace", type=float, default=0.02)
    p.add_argument("--hspace", type=float, default=0.04)

    p.add_argument("--shared-y-label-x", type=float, default=0.030)
    p.add_argument("--shared-y-label-y-offset", type=float, default=0.00)

    p.add_argument("--range-label", default="IQR", help="Error-bar label in legend, e.g. IQR or ±2 SE.")
    p.add_argument("--no-lrf-legend", action="store_true")
    return p.parse_args()


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.family": str(args.font_family),
        "font.serif": [str(args.font_family), "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.weight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def collect_panel_paths(args: argparse.Namespace) -> list[Path]:
    if args.panels:
        paths = []
        for x in args.panels:
            p = Path(x)
            if not p.is_file():
                p = args.panel_root / x
            paths.append(p)
    else:
        paths = sorted(args.panel_root.glob("*_timing.png"))

    if not paths:
        raise SystemExit(f"No timing panel PNGs found under {args.panel_root}")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("Missing panel PNGs:\n" + "\n".join(missing))
    return paths


def legend_handles_labels(args: argparse.Namespace) -> tuple[list[object], list[str]]:
    methods = ["Gittins-S", "Gittins-G", "UCB-E"]
    if not bool(args.no_lrf_legend):
        methods.append("LRF")

    handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], linewidth=6.0, solid_capstyle="butt")
        for m in methods
    ]
    labels = list(methods)

    # Error bars in panels are black capped bars.
    handles.append(Line2D([0], [0], color="black", marker="_", markersize=16, linewidth=1.5))
    labels.append(args.range_label)
    return handles, labels


def main() -> int:
    args = parse_args()
    setup_matplotlib(args)

    paths = collect_panel_paths(args)
    n = len(paths)
    ncols = max(1, int(args.grid_cols))
    nrows = int(math.ceil(n / ncols))

    fig_height = float(args.fig_height)
    if fig_height <= 0:
        panel_area_width = (float(args.right) - float(args.left)) * float(args.fig_width)
        row_height = (panel_area_width / float(ncols)) / float(args.panel_aspect)
        panel_area_fraction = max(0.2, float(args.top) - float(args.bottom))
        fig_height = row_height * float(nrows) / panel_area_fraction

    fig, axes = plt.subplots(nrows, ncols, figsize=(float(args.fig_width), fig_height), squeeze=False)

    first_shape = None
    for idx, p in enumerate(paths):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        img = mpimg.imread(p)
        if first_shape is None:
            first_shape = img.shape
        elif img.shape != first_shape:
            print(f"WARNING: image size differs: {p} has {img.shape}, first image has {first_shape}")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.subplots_adjust(
        left=float(args.left),
        right=float(args.right),
        top=float(args.top),
        bottom=float(args.bottom),
        wspace=float(args.wspace),
        hspace=float(args.hspace),
    )

    panel_center_y = 0.5 * (float(args.bottom) + float(args.top))
    fig.text(
        float(args.shared_y_label_x),
        float(panel_center_y + float(args.shared_y_label_y_offset)),
        "Per-batch decision time (s)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=float(args.shared_label_size),
    )

    handles, labels = legend_handles_labels(args)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(int(args.legend_ncol), len(labels)),
        frameon=False,
        bbox_to_anchor=(float(args.legend_x), float(args.legend_y)),
        fontsize=float(args.legend_size),
        handlelength=float(args.legend_handlelength),
        handletextpad=float(args.legend_handletextpad),
        columnspacing=float(args.legend_columnspacing),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi))
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote timing grid: {args.out}")
    print(f"Wrote PDF: {args.out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
