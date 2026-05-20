#!/usr/bin/env python3
"""Plot bandit and BayesOpt PBGI simple-regret curves from trace files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


def _scalar(z: np.lib.npyio.NpzFile, key: str, default: float = -1.0) -> float:
    if key not in z.files:
        return default
    return float(np.asarray(z[key]).reshape(()))


def _plot_curve(
    z: np.lib.npyio.NpzFile,
    *,
    x_key: str,
    y_key: str,
    label: str,
    linewidth: float = 1.7,
) -> None:
    if x_key not in z.files or y_key not in z.files:
        return
    x = np.asarray(z[x_key])
    y = np.asarray(z[y_key])
    if x.size and y.size:
        plt.plot(x, y, linewidth=linewidth, label=label)


def _plot_stop(x: float, *, label: str, color: str) -> None:
    if x < 0:
        return
    plt.axvline(x, color=color, linestyle="--", alpha=0.8, linewidth=1.2, label=label)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bandit-trace", type=Path, required=True)
    p.add_argument("--pbgi-trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str, required=True)
    p.add_argument(
        "--x-axis",
        choices=["evals", "original_cost"],
        required=True,
        help="Use cumulative examples/evaluations or cumulative full-evaluation cost.",
    )
    args = p.parse_args()

    bandit = np.load(args.bandit_trace)
    pbgi = np.load(args.pbgi_trace)

    if args.x_axis == "evals":
        ucb_x = "ucb_x"
        gittins_x = "gittins_x"
        pbgi_x = "x"
        gittins_stop_key = "gittins_stop_cum_eval"
        pbgi_stop_key = "pbgi_stop_cum_eval"
        xlabel = "Cumulative examples evaluated"
    else:
        ucb_x = "ucb_x_original_cost"
        gittins_x = "gittins_x_original_cost"
        pbgi_x = "x_original_cost"
        gittins_stop_key = "gittins_stop_cum_original_cost"
        pbgi_stop_key = "pbgi_stop_cum_original_cost"
        xlabel = "Cumulative full-evaluation cost"

    plt.figure(figsize=(8, 5))
    _plot_curve(bandit, x_key=ucb_x, y_key="ucb_regret", label="bandit UCB-E")
    _plot_curve(bandit, x_key=gittins_x, y_key="gittins_regret", label="bandit Gittins")
    _plot_curve(pbgi, x_key=pbgi_x, y_key="regret", label="BayesOpt PBGI")

    _plot_stop(_scalar(bandit, gittins_stop_key), label="bandit Gittins stop", color="C1")
    _plot_stop(_scalar(pbgi, pbgi_stop_key), label="BayesOpt PBGI stop", color="C2")

    plt.xlabel(xlabel)
    plt.ylabel("Simple regret")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
