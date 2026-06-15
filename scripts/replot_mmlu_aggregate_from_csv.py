#!/usr/bin/env python3
"""Replot MMLU aggregate figure from an existing aggregated CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


STYLE = {
    "gittins_data": ("tab:orange", "Gittins-S", 3.2, 5),
    "gittins_default": ("tab:green", "Gittins-G", 3.2, 4),
    "ucb": ("tab:blue", "UCB-E", 3.0, 3),
    "lrf": ("tab:purple", "LRF", 3.0, 3),
    "bo_pbgi_unit": ("tab:red", "BO-PBGI", 3.0, 2),
    "bo_logei_unit": ("tab:brown", "BO-LogEI", 3.0, 2),
    "bo_pbgi_cost": ("tab:red", "BO-PBGI", 3.0, 2),
    "bo_logeipc_cost": ("tab:brown", "BO-LogEI(PC)", 3.0, 2),
}
ORDER = [
    "gittins_data",
    "gittins_default",
    "ucb",
    "lrf",
    "bo_pbgi_unit",
    "bo_logei_unit",
    "bo_pbgi_cost",
    "bo_logeipc_cost",
]
LINEWIDTH_MULT = 3.6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    stem = "mmlu_aggregate_2x2_unit_Bsmall4_Blarge16_scale1e-4_xfinal_yinitial_fast"
    base = Path(r"outputs\wandb_plots_new\paper_figures")
    p.add_argument("--aggregated", type=Path, default=base / f"{stem}_aggregated.csv")
    p.add_argument("--groups", type=Path, default=base / f"{stem}_groups.json")
    p.add_argument("--out-png", type=Path, default=base / f"{stem}.png")
    p.add_argument("--out-pdf", type=Path, default=base / f"{stem}.pdf")
    p.add_argument("--shared-x-label-y", type=float, default=0.205)
    p.add_argument("--shared-y-label-x", type=float, default=-0.023)
    p.add_argument("--small-summary", type=Path, default=Path(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_small\runs_summary.csv"))
    p.add_argument("--large-summary", type=Path, default=Path(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_large\runs_summary.csv"))
    p.add_argument("--small-bo-summary", type=Path, default=Path(r"outputs\wandb_downloads_new\bo_baseline\mmlu_small_bo\runs_summary.csv"))
    p.add_argument("--large-bo-summary", type=Path, default=Path(r"outputs\wandb_downloads_new\bo_baseline\mmlu_large_bo\runs_summary.csv"))
    p.add_argument("--cost-mode", choices=["unit", "aware"], default="unit")
    return p.parse_args()


def norm_task(v: object) -> str:
    return str(v).replace("\\", "/").rstrip("/").split("/")[-1]


def stop_values(summary: Path, tasks: list[str], variant: str, kind: str, cost_mode: str) -> np.ndarray:
    if not summary.is_file():
        return np.array([])
    header = pd.read_csv(summary, nrows=0)
    if cost_mode == "aware":
        x_stop = "bo_stop_cum_original_cost" if kind.startswith("bo_") else "gittins_stop_cum_original_cost"
        final_col = "final_cum_original_cost"
    else:
        x_stop = "bo_stop_cum_eval" if kind.startswith("bo_") else "gittins_stop_cum_eval"
        final_col = "final_cum_eval"
    usecols = [c for c in ["mmlu_task", "matrix_task", "experiment_variant", x_stop, final_col] if c in header.columns]
    if x_stop not in usecols or final_col not in usecols:
        return np.array([])
    df = pd.read_csv(summary, usecols=usecols)
    df = df[df["experiment_variant"].astype(str) == variant].copy()
    task_set = set(tasks)
    mask = pd.Series(False, index=df.index)
    for col in ["mmlu_task", "matrix_task"]:
        if col in df.columns:
            mask |= df[col].map(norm_task).isin(task_set)
    df = df[mask].copy()
    stop = pd.to_numeric(df[x_stop], errors="coerce")
    final = pd.to_numeric(df[final_col], errors="coerce")
    vals = (stop / final).to_numpy(dtype=float)
    return vals[np.isfinite(vals) & (vals >= 0.0) & (vals <= 1.0)]


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.aggregated)
    groups = json.loads(args.groups.read_text(encoding="utf-8"))["selected_tasks"]

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(26.8, 25.9), constrained_layout=False)
    rows = ["Easy", "Hard"]
    cols = ["Small", "Large"]
    legend_handles: dict[str, Line2D] = {}

    for r, difficulty in enumerate(rows):
        for c, size in enumerate(cols):
            ax = axes[r, c]
            group = f"{difficulty}-{size}"
            gdf = df[df["group"] == group]
            for kind in ORDER:
                sub = gdf[gdf["method_kind"] == kind].sort_values("x")
                if sub.empty:
                    continue
                color, label, lw, z = STYLE[kind]
                line = ax.plot(sub["x"], sub["mean"], color=color, linewidth=lw * LINEWIDTH_MULT, zorder=z, label=label)[0]
                legend_handles.setdefault(kind, line)
                band = 2.0 * sub["stderr"].to_numpy(dtype=float)
                x = sub["x"].to_numpy(dtype=float)
                mean = sub["mean"].to_numpy(dtype=float)
                ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.15, linewidth=0, zorder=z - 0.5)

                summary = args.small_summary if size == "Small" else args.large_summary
                if kind.startswith("bo_"):
                    summary = args.small_bo_summary if size == "Small" else args.large_bo_summary
                vals = stop_values(summary, groups[group], str(sub["variant"].iloc[0]), kind, args.cost_mode)
                if vals.size:
                    m = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / np.sqrt(vals.size) * 2.0) if vals.size > 1 else 0.0
                    if se > 0:
                        ax.axvspan(m - se, m + se, color=color, alpha=0.12, linewidth=0, zorder=1)
                    ax.axvline(m, color=color, linestyle="--", linewidth=2.6 * LINEWIDTH_MULT, alpha=0.72, zorder=2)

            if r == 0:
                ax.set_title(size, fontsize=81, pad=8)
            ax.grid(True, alpha=0.23, linewidth=0.9)
            ax.tick_params(axis="both", labelsize=57, width=1.2, length=6)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

    axes[0, 1].text(1.035, 0.5, "Easy", transform=axes[0, 1].transAxes, rotation=-90, va="center", ha="left", fontsize=81)
    axes[1, 1].text(1.035, 0.5, "Hard", transform=axes[1, 1].transAxes, rotation=-90, va="center", ha="left", fontsize=81)
    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.text(float(args.shared_y_label_x), 0.5 * (0.325 + 0.765), "Normalized simple regret", ha="center", va="center", rotation="vertical", fontsize=81)
    fig.text(
        0.5 * (0.045 + 0.995),
        float(args.shared_x_label_y),
        "Normalized cumulative cost" if args.cost_mode == "aware" else "Normalized cumulative evaluations",
        ha="center",
        va="center",
        fontsize=81,
    )

    handles = []
    labels = []
    seen_labels = set()
    for k in ORDER:
        if k not in legend_handles:
            continue
        label = STYLE[k][1]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(legend_handles[k])
        labels.append(label)
    for kind in ["gittins_data", "gittins_default", "bo_pbgi_unit", "bo_logei_unit", "bo_pbgi_cost", "bo_logeipc_cost"]:
        if kind in legend_handles:
            label = f"{STYLE[kind][1]} mean stop"
            if label in labels:
                continue
            color = STYLE[kind][0]
            handles.append(Line2D([0], [0], color=color, linestyle="--", linewidth=2.6 * LINEWIDTH_MULT, alpha=0.72))
            labels.append(label)
    handles.append(Patch(facecolor="0.75", edgecolor="none", alpha=0.18))
    labels.append("±2 SE band")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02), fontsize=58, handlelength=2.0, handletextpad=0.35, columnspacing=1.15)

    fig.subplots_adjust(left=0.045, right=0.995, top=0.765, bottom=0.325, wspace=0.24, hspace=0.34)
    fig.savefig(args.out_png, dpi=260, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.22)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
