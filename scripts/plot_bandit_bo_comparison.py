#!/usr/bin/env python3
"""Plot bandit and BayesOpt simple-regret curves from trace files."""

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
    color: str | None = None,
) -> None:
    if x_key not in z.files or y_key not in z.files:
        return
    x = np.asarray(z[x_key])
    y = np.asarray(z[y_key])
    if x.size and y.size:
        plt.plot(x, y, linewidth=linewidth, label=label, color=color)


def _plot_stop(x: float, *, label: str, color: str) -> None:
    if x < 0:
        return
    plt.axvline(x, color=color, linestyle="--", alpha=0.8, linewidth=1.2, label=label)


def _plot_bo_with_stop(
    z: np.lib.npyio.NpzFile,
    *,
    x_key: str,
    stop_key: str,
    label: str,
    curve_color: str | None,
    stop_label: str,
    stop_color: str,
) -> None:
    _plot_curve(z, x_key=x_key, y_key="regret", label=label, color=curve_color)
    _plot_stop(_scalar(z, stop_key), label=stop_label, color=stop_color)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bandit-trace", type=Path, required=True)
    p.add_argument(
        "--bo-trace",
        "--pbgi-trace",
        dest="bo_trace",
        type=Path,
        required=True,
        help="Trace npz from scripts/run_bo_baseline.py (PBGI, LogEI, LogEIPC, etc.).",
    )
    p.add_argument(
        "--bo-label",
        type=str,
        default="BayesOpt",
        help="Legend label for the BO curve.",
    )
    p.add_argument(
        "--bo-stop-key-evals",
        type=str,
        default="pbgi_stop_cum_eval",
        help="Trace key for BO natural stop when x-axis is evals.",
    )
    p.add_argument(
        "--bo-stop-key-cost",
        type=str,
        default="pbgi_stop_cum_original_cost",
        help="Trace key for BO natural stop when x-axis is original_cost.",
    )
    p.add_argument(
        "--bo-stop-label",
        type=str,
        default="BayesOpt stop",
        help="Legend label for BO natural stop marker.",
    )
    p.add_argument(
        "--bo-stop-color",
        type=str,
        default="C2",
        help="Matplotlib color for BO stop marker.",
    )
    p.add_argument(
        "--bo-color",
        type=str,
        default=None,
        help="Optional matplotlib color for the BO curve.",
    )
    p.add_argument(
        "--bo2-trace",
        type=Path,
        default=None,
        help="Optional second BO trace to overlay in the same figure.",
    )
    p.add_argument(
        "--bo2-label",
        type=str,
        default="BayesOpt 2",
        help="Legend label for the second BO curve.",
    )
    p.add_argument(
        "--bo2-stop-key-evals",
        type=str,
        default="pbgi_stop_cum_eval",
        help="Trace key for second BO natural stop when x-axis is evals.",
    )
    p.add_argument(
        "--bo2-stop-key-cost",
        type=str,
        default="pbgi_stop_cum_original_cost",
        help="Trace key for second BO natural stop when x-axis is original_cost.",
    )
    p.add_argument(
        "--bo2-stop-label",
        type=str,
        default="BayesOpt 2 stop",
        help="Legend label for second BO natural stop marker.",
    )
    p.add_argument(
        "--bo2-stop-color",
        type=str,
        default="C3",
        help="Matplotlib color for second BO stop marker.",
    )
    p.add_argument(
        "--bo2-color",
        type=str,
        default=None,
        help="Optional matplotlib color for the second BO curve.",
    )
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
    bo = np.load(args.bo_trace)
    bo2 = np.load(args.bo2_trace) if args.bo2_trace is not None else None

    if args.x_axis == "evals":
        ucb_x = "ucb_x"
        gittins_x = "gittins_x"
        bo_x = "x"
        gittins_stop_key = "gittins_stop_cum_eval"
        bo_stop_key = str(args.bo_stop_key_evals)
        bo2_stop_key = str(args.bo2_stop_key_evals)
        xlabel = "Cumulative examples evaluated"
    else:
        ucb_x = "ucb_x_original_cost"
        gittins_x = "gittins_x_original_cost"
        bo_x = "x_original_cost"
        gittins_stop_key = "gittins_stop_cum_original_cost"
        bo_stop_key = str(args.bo_stop_key_cost)
        bo2_stop_key = str(args.bo2_stop_key_cost)
        xlabel = "Cumulative full-evaluation cost"

    plt.figure(figsize=(8, 5))
    _plot_curve(bandit, x_key=ucb_x, y_key="ucb_regret", label="bandit UCB-E")
    _plot_curve(bandit, x_key=gittins_x, y_key="gittins_regret", label="bandit Gittins")
    _plot_bo_with_stop(
        bo,
        x_key=bo_x,
        stop_key=bo_stop_key,
        label=str(args.bo_label),
        curve_color=args.bo_color,
        stop_label=str(args.bo_stop_label),
        stop_color=str(args.bo_stop_color),
    )
    if bo2 is not None:
        _plot_bo_with_stop(
            bo2,
            x_key=bo_x,
            stop_key=bo2_stop_key,
            label=str(args.bo2_label),
            curve_color=args.bo2_color,
            stop_label=str(args.bo2_stop_label),
            stop_color=str(args.bo2_stop_color),
        )

    _plot_stop(_scalar(bandit, gittins_stop_key), label="bandit Gittins stop", color="C1")

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
