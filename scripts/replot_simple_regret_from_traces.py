#!/usr/bin/env python3
"""Rebuild the simple-regret figure from a trace bundle written by plot_simple_regret_gsm8k.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Path to *_traces.npz from plot_simple_regret_gsm8k.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (default: same stem as traces, .png)",
    )
    args = parser.parse_args()

    if not args.traces.is_file():
        print(f"Traces not found: {args.traces}", file=sys.stderr)
        return 1

    z = np.load(args.traces)
    meta_path = args.traces.with_suffix(".meta.json")
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    title = meta.get("title")

    out = args.out
    if out is None:
        out = args.traces.with_name(args.traces.stem.replace("_traces", "_replot") + ".png")
        if out == args.traces:
            out = args.traces.with_name(args.traces.stem + "_replot.png")

    def _plot_if_nonempty(xkey: str, ykey: str, label: str) -> None:
        x, y = z[xkey], z[ykey]
        if x.size > 0:
            plt.plot(x, y, label=label, linewidth=1.5)

    plt.figure(figsize=(8, 5))
    # Order matches plot_simple_regret_gsm8k.py so default colors match.
    _plot_if_nonempty("rr_x", "rr_regret", "Round-robin (sample mean)")
    _plot_if_nonempty("ucb_x", "ucb_regret", "UCB-E")
    _plot_if_nonempty("lrf_x_plot", "lrf_regret_plot", "UCB-E-LRF")

    gittins_per_cell = bool(meta.get("gittins_per_cell_dp", False))
    gittins_label = (
        "Gittins (τ² = 1/(4B), per-cell DP)"
        if gittins_per_cell
        else "Gittins (τ² = 1/(4B), batch-mean DP)"
    )
    x_g, r_g = z["gittins_x"], z["gittins_regret"]
    line_gittins = None
    if x_g.size > 0:
        (line_gittins,) = plt.plot(x_g, r_g, label=gittins_label, linewidth=1.5)

    if "gittins_stop_cum_eval" in z.files:
        stop = int(z["gittins_stop_cum_eval"].reshape(()))
        if stop >= 0 and line_gittins is not None:
            plt.axvline(
                stop,
                color=line_gittins.get_color(),
                linestyle="--",
                alpha=0.85,
                linewidth=1.2,
                label=f"Gittins nominal stop ({stop} evals)",
            )

    plt.xlabel("Cumulative examples evaluated (matrix entries revealed)")
    plt.ylabel("Simple regret")
    plt.title(title or "Simple regret (from traces)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
