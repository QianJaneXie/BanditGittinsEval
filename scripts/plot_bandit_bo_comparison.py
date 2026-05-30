#!/usr/bin/env python3
"""Plot bandit and BayesOpt simple-regret curves from trace files.

BayesOpt curves include only post-initialization evaluations (selection_phase
is not ``random_init``). The x-axis remains cumulative evals or cost from the
start of the run, so each BO curve begins where acquisition-driven evaluation starts.
"""

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


def _plot_stop(
    x: float,
    *,
    label: str,
    color: str,
    linestyle: str = "--",
) -> None:
    if x < 0:
        return
    plt.axvline(x, color=color, linestyle=linestyle, alpha=0.8, linewidth=1.2, label=label)


def _bo_post_init_mask(z: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "selection_phase" not in z.files:
        return None
    phase = np.asarray(z["selection_phase"]).astype(str)
    return phase != "random_init"


def _bo_post_init_xy(z: np.lib.npyio.NpzFile, *, x_key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(z[x_key], dtype=np.float64)
    y = np.asarray(z["regret"], dtype=np.float64)
    mask = _bo_post_init_mask(z)
    if mask is None:
        return x, y
    if not bool(mask.any()):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return x[mask], y[mask]


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
    x, y = _bo_post_init_xy(z, x_key=x_key)
    if x.size and y.size:
        plt.plot(x, y, linewidth=1.7, label=label, color=curve_color)
    stop_x = _scalar(z, stop_key)
    if x.size and stop_x >= 0 and stop_x >= float(x[0]):
        _plot_stop(stop_x, label=stop_label, color=stop_color)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bandit-trace", type=Path, required=True)
    p.add_argument(
        "--pbgi-trace",
        "--pbgi_trace",
        type=Path,
        required=True,
        help="PBGI trace from scripts/run_bo_baseline.py.",
    )
    p.add_argument(
        "--pbgi-label",
        type=str,
        default="BayesOpt PBGI",
        help="Legend label for the PBGI curve.",
    )
    p.add_argument(
        "--log-bo-trace",
        "--log_bo_trace",
        type=Path,
        default=None,
        help="LogEI trace (unit-cost plots) or LogEIPC trace (cost-aware plots).",
    )
    p.add_argument(
        "--log-bo-label",
        type=str,
        default="BayesOpt Log",
        help="Legend label for the LogEI / LogEIPC curve.",
    )
    p.add_argument(
        "--pbgi-color",
        type=str,
        default="C2",
        help="Matplotlib color for the PBGI curve and natural-stop marker.",
    )
    p.add_argument(
        "--log-bo-color",
        type=str,
        default="C3",
        help="Matplotlib color for the LogEI / LogEIPC curve and natural-stop marker.",
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
    pbgi = np.load(args.pbgi_trace)
    log_bo = np.load(args.log_bo_trace) if args.log_bo_trace is not None else None

    if args.x_axis == "evals":
        ucb_x = "ucb_x"
        gittins_x = "gittins_x"
        bo_x = "x"
        bo_stop_key = "pbgi_stop_cum_eval"
        gittins_stop_key = "gittins_stop_cum_eval"
        gittins_rec_stop_key = "gittins_recommendation_aware_stop_cum_eval"
        xlabel = "Cumulative examples evaluated"
    else:
        ucb_x = "ucb_x_original_cost"
        gittins_x = "gittins_x_original_cost"
        bo_x = "x_original_cost"
        bo_stop_key = "pbgi_stop_cum_original_cost"
        gittins_stop_key = "gittins_stop_cum_original_cost"
        gittins_rec_stop_key = "gittins_recommendation_aware_stop_cum_original_cost"
        xlabel = "Cumulative full-evaluation cost"

    plt.figure(figsize=(8, 5))
    _plot_curve(bandit, x_key=ucb_x, y_key="ucb_regret", label="bandit UCB-E")
    _plot_curve(bandit, x_key=gittins_x, y_key="gittins_regret", label="bandit Gittins")
    if log_bo is not None:
        _plot_bo_with_stop(
            log_bo,
            x_key=bo_x,
            stop_key=bo_stop_key,
            label=str(args.log_bo_label),
            curve_color=args.log_bo_color,
            stop_label=f"{args.log_bo_label} stop",
            stop_color=str(args.log_bo_color),
        )
    _plot_bo_with_stop(
        pbgi,
        x_key=bo_x,
        stop_key=bo_stop_key,
        label=str(args.pbgi_label),
        curve_color=args.pbgi_color,
        stop_label=f"{args.pbgi_label} stop",
        stop_color=str(args.pbgi_color),
    )

    _plot_stop(
        _scalar(bandit, gittins_stop_key),
        label="bandit Gittins stop (index)",
        color="C1",
        linestyle="--",
    )
    _plot_stop(
        _scalar(bandit, gittins_rec_stop_key),
        label="bandit Gittins stop (recommendation)",
        color="C1",
        linestyle="-.",
    )

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
