#!/usr/bin/env python3
"""Plot simple-regret curves from a trace bundle produced by simulate_simple_regret.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Use a repo-local, writable matplotlib cache/config directory when possible.
_repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True, help="Path to .npz from simulation.")
    p.add_argument("--out", type=Path, required=True, help="Output figure path (e.g. .png).")
    p.add_argument("--title", type=str, default=None)
    p.add_argument(
        "--x-axis",
        choices=["evals", "original_cost"],
        default="evals",
        help="Plot vs cumulative evaluations (evals) or cumulative cost (original_cost).",
    )
    args = p.parse_args()

    z = np.load(args.traces)

    def _plot(x_key: str, y_key: str, label: str) -> None:
        if x_key in z.files and y_key in z.files:
            x = z[x_key]
            y = z[y_key]
            if x.size and y.size:
                plt.plot(x, y, label=label, linewidth=1.6)

    x_ucb = "ucb_x" if args.x_axis == "evals" else "ucb_x_original_cost"
    x_gittins = "gittins_x" if args.x_axis == "evals" else "gittins_x_original_cost"

    plt.figure(figsize=(8, 5))
    _plot(x_ucb, "ucb_regret", "UCB-E (recommend: empirical mean)")
    _plot(x_gittins, "gittins_regret", "Gittins (recommend: posterior mean)")
    if args.x_axis == "original_cost":
        plt.xlabel("Cumulative cost (original units of cost vector)")
    else:
        plt.xlabel("Cumulative examples evaluated (matrix entries revealed)")
    plt.ylabel("Simple regret")

    title = args.title
    if title is None:
        title = "Simple regret"
        if "matrix" in z.files:
            title = f"Simple regret ({Path(str(z['matrix'])).name})"
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()

    def _gittins_stop_vline(
        eval_key: str,
        cost_key: str,
        *,
        color: str,
        label: str,
    ) -> None:
        if eval_key not in z.files:
            return
        stop_eval = int(np.asarray(z[eval_key]).reshape(()))
        if stop_eval < 0:
            return
        if args.x_axis == "original_cost" and cost_key in z.files:
            stop_x = float(np.asarray(z[cost_key]).reshape(()))
            if stop_x < 0:
                stop_x = float(stop_eval)
        else:
            stop_x = float(stop_eval)
        plt.axvline(
            stop_x,
            color=color,
            linestyle="--",
            alpha=0.85,
            linewidth=1.2,
            label=label,
        )

    # Index-induced stop: argmax (Γ on incomplete, μ on complete) is a complete arm.
    _gittins_stop_vline(
        "gittins_stop_cum_eval",
        "gittins_stop_cum_original_cost",
        color="C1",
        label="Gittins index-induced stop",
    )
    # Recommendation-aware stop: max incomplete Γ < max μ.
    _gittins_stop_vline(
        "gittins_recommendation_aware_stop_cum_eval",
        "gittins_recommendation_aware_stop_cum_original_cost",
        color="C4",
        label="Gittins recommendation-aware stop",
    )
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

