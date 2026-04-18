#!/usr/bin/env python3
"""Rebuild the simple-regret figure from a trace bundle written by plot_simple_regret_gsm8k.py.

**Gittins x-axis:** ``evals`` (cost-unaware style: cumulative matrix entries revealed) vs
``original_cost`` (cost-aware: cumulative monetary cost stored in ``gittins_x_original_cost``, same
units as the cost vector used at simulation time—e.g. **USD per 1M input tokens** for GSM8K pricing
JSON). The default cost-axis label matches the main script’s GSM8K figures.
"""

from __future__ import annotations

# Same label as plot_simple_regret_gsm8k for GSM8K cost-aware runs; change if your traces use other units.
_GITTINS_COST_AWARE_XLABEL = "Cumulative cost (USD per 1M input tokens)"

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
    parser.add_argument(
        "--gittins-x-axis",
        choices=["evals", "original_cost"],
        default="evals",
        help="Gittins only: evals = cumulative matrix entries (cost-unaware-style); original_cost = "
        "cumulative monetary cost when traces contain gittins_x_original_cost (e.g. USD per 1M in-tokens for GSM8K pricing)",
    )
    args = parser.parse_args()

    if not args.traces.is_file():
        print(f"Traces not found: {args.traces}", file=sys.stderr)
        return 1

    z = np.load(args.traces)
    meta_path = args.traces.with_suffix(".meta.json")
    title = None
    if meta_path.is_file():
        title = json.loads(meta_path.read_text(encoding="utf-8")).get("title")

    out = args.out
    if out is None:
        out = args.traces.with_name(args.traces.stem.replace("_traces", "_replot") + ".png")
        if out == args.traces:
            out = args.traces.with_name(args.traces.stem + "_replot.png")

    def _plot_if_nonempty(xkey: str, ykey: str, label: str) -> None:
        x, y = z[xkey], z[ykey]
        if x.size > 0:
            plt.plot(x, y, label=label, linewidth=1.5)

    gittins_cost_x = (
        args.gittins_x_axis == "original_cost"
        and "gittins_x_original_cost" in z.files
        and z["gittins_x_original_cost"].size > 0
        and z["gittins_regret"].size > 0
        and z["gittins_x_original_cost"].shape == z["gittins_regret"].shape
    )
    if args.gittins_x_axis == "original_cost" and not gittins_cost_x:
        print(
            "Warning: original-cost x-axis requested but traces lack gittins_x_original_cost; "
            "using cumulative evals for Gittins.",
            file=sys.stderr,
        )

    plt.figure(figsize=(8, 5))
    if not gittins_cost_x:
        _plot_if_nonempty("ucb_x", "ucb_regret", "UCB-E")
        _plot_if_nonempty("lrf_x_plot", "lrf_regret_plot", "UCB-E-LRF")
    gittins_x_key = "gittins_x_original_cost" if gittins_cost_x else "gittins_x"
    gittins_label = (
        "Gittins (τ² = 1/(4B), cost-aware, x = cum. cost)"
        if gittins_cost_x
        else "Gittins (τ² = 1/(4B))"
    )
    _plot_if_nonempty(gittins_x_key, "gittins_regret", gittins_label)
    if "gittins_stop_cum_eval" in z.files and not gittins_cost_x:
        stop = int(z["gittins_stop_cum_eval"].reshape(()))
        if stop >= 0:
            plt.axvline(
                stop,
                color="C2",
                linestyle="--",
                alpha=0.85,
                linewidth=1.2,
                label=f"Gittins nominal stop ({stop} evals)",
            )
    if gittins_cost_x:
        plt.xlabel(_GITTINS_COST_AWARE_XLABEL)
    else:
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
