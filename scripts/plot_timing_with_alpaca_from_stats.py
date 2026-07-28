#!/usr/bin/env python3
"""Replot total wall-clock timing bars and add AlpacaEval as a sixth panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


METHOD_ORDER = ["Gittins-S", "Gittins-G", "SySRs", "UCB-E", "LRF", "BO-PBGI", "BO-LogEI"]
METHOD_COLOR = {
    "Gittins-S": "tab:orange",
    "Gittins-G": "tab:green",
    "SySRs": "tab:pink",
    "UCB-E": "tab:blue",
    "LRF": "tab:purple",
    "BO-PBGI": "tab:olive",
    "BO-LogEI": "tab:brown",
    "BO-LogEIPC": "tab:brown",
}
GROUP_ORDER = ["GSM8K", "PIQA", "AlpacaEval", "MMLU-small", "MMLU-medium", "MMLU-large"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats-dir", type=Path, default=Path(r"outputs\figure\new_figure\final"))
    p.add_argument("--alpaca-bandit-root", type=Path, default=Path(r"outputs\wandb_downloads_new\ucb_gittins\alpaca_0.1_0.02"))
    p.add_argument("--alpaca-lrf-root", type=Path, default=Path(r"outputs\wandb_downloads_new\lrf\alpaca_lrf"))
    p.add_argument("--alpaca-bo-root", type=Path, default=Path(r"outputs\wandb_downloads_new\bo_baseline_5pct\alpaca_bo"))
    p.add_argument("--sysrs-root", type=Path, default=Path(r"outputs\wandb_downloads_new\sysrs"))
    p.add_argument("--out-dir", type=Path, default=Path(r"outputs\figure\new_figure\final\timing_with_alpaca_sysrs"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lrf-batch-size", type=int, default=32)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--se-mult", type=float, default=1.0)
    p.add_argument("--bottom-max", type=float, default=90.0)
    p.add_argument("--top-min", type=float, default=100.0)
    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.family": str(args.font_family),
        "font.serif": [str(args.font_family), "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_summary(root: Path) -> pd.DataFrame:
    path = root / "runs_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "state" in df.columns:
        df = df[df["state"].astype(str).str.lower().eq("finished")].copy()
    return df


def center_se(vals: pd.Series, se_mult: float) -> tuple[float, float, int]:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, 0
    center = float(np.median(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return center, float(se_mult) * se, int(arr.size)


def alpaca_final_rows(args: argparse.Namespace, mode: str) -> pd.DataFrame:
    bandit = read_summary(args.alpaca_bandit_root)
    lrf = read_summary(args.alpaca_lrf_root)
    bo = read_summary(args.alpaca_bo_root)
    b = int(args.batch_size)
    lb = int(args.lrf_batch_size)
    scale = str(args.scale)
    if mode == "unit":
        variants = {
            "Gittins-S": f"gittins_unit_B{b}_scale{scale}_dataset",
            "Gittins-G": f"gittins_unit_B{b}_scale{scale}_default",
            "UCB-E": f"ucb_B{b}",
            "LRF": f"lrf_B{lb}",
            "BO-PBGI": "pbgi_unit",
            "BO-LogEI": "logei_unit",
        }
    else:
        variants = {
            "Gittins-S": f"gittins_cost_B{b}_scale{scale}_dataset",
            "Gittins-G": f"gittins_cost_B{b}_scale{scale}_default",
            "UCB-E": f"ucb_cost_B{b}",
            "LRF": f"lrf_cost_B{lb}",
            "BO-PBGI": "pbgi_cost",
            "BO-LogEIPC": "logeipc_cost",
        }
    roots = {
        "Gittins-S": bandit,
        "Gittins-G": bandit,
        "UCB-E": bandit,
        "LRF": lrf,
        "BO-PBGI": bo,
        "BO-LogEI": bo,
        "BO-LogEIPC": bo,
    }
    rows = []
    for method, variant in variants.items():
        df = roots[method]
        sub = df[df["experiment_variant"].astype(str).eq(variant)]
        med, se, n = center_se(sub["total_wall_time_s"], float(args.se_mult))
        rows.append({
            "mode": mode,
            "group": "AlpacaEval",
            "batch_for_ucb_gittins": b,
            "method": method,
            "variant": variant,
            "median_s": med,
            "se_s": se,
            "n_runs": n,
        })
    return pd.DataFrame(rows)


def sysrs_final_rows(args: argparse.Namespace, mode: str) -> pd.DataFrame:
    variant = "sysrs" if mode == "unit" else "sysrs_cost"
    roots = {
        "GSM8K": args.sysrs_root / "gsm8k",
        "PIQA": args.sysrs_root / "piqa",
        "AlpacaEval": args.sysrs_root / "alpaca",
        "MMLU-small": args.sysrs_root / "mmlu_small",
        "MMLU-medium": args.sysrs_root / "mmlu_medium",
        "MMLU-large": args.sysrs_root / "mmlu_large",
    }
    rows = []
    for group, root in roots.items():
        df = read_summary(root)
        sub = df[df["experiment_variant"].astype(str).eq(variant)]
        med, se, n = center_se(sub["total_wall_time_s"], float(args.se_mult))
        rows.append({
            "mode": mode,
            "group": group,
            "batch_for_ucb_gittins": 0,
            "method": "SySRs",
            "variant": variant,
            "median_s": med,
            "se_s": se,
            "n_runs": n,
        })
    return pd.DataFrame(rows)


def stop_wall_rows(args: argparse.Namespace, mode: str, stop_family: str) -> pd.DataFrame:
    b = int(args.batch_size)
    scale = str(args.scale)
    if stop_family == "gittins":
        df = read_summary(args.alpaca_bandit_root)
        if mode == "unit":
            methods = {
                "Gittins-S": f"gittins_unit_B{b}_scale{scale}_dataset",
                "Gittins-G": f"gittins_unit_B{b}_scale{scale}_default",
            }
            stop_col, final_col = "gittins_stop_cum_eval", "final_cum_eval"
        else:
            methods = {
                "Gittins-S": f"gittins_cost_B{b}_scale{scale}_dataset",
                "Gittins-G": f"gittins_cost_B{b}_scale{scale}_default",
            }
            stop_col, final_col = "gittins_stop_cum_original_cost", "final_cum_original_cost"
    else:
        df = read_summary(args.alpaca_bo_root)
        if mode == "unit":
            methods = {"BO-PBGI": "pbgi_unit", "BO-LogEI": "logei_unit"}
            stop_col, final_col = "bo_stop_cum_eval", "final_cum_eval"
        else:
            methods = {"BO-PBGI": "pbgi_cost", "BO-LogEIPC": "logeipc_cost"}
            stop_col, final_col = "bo_stop_cum_original_cost", "final_cum_original_cost"

    rows = []
    for method, variant in methods.items():
        sub = df[df["experiment_variant"].astype(str).eq(variant)].copy()
        if stop_col not in sub or final_col not in sub:
            vals = pd.Series(dtype=float)
        else:
            stop = pd.to_numeric(sub[stop_col], errors="coerce")
            final = pd.to_numeric(sub[final_col], errors="coerce")
            wall = pd.to_numeric(sub["total_wall_time_s"], errors="coerce")
            frac = stop / final
            vals = wall * frac
            vals = vals[np.isfinite(vals) & (frac >= 0) & (frac <= 1)]
        med = float(np.median(vals)) if len(vals) else np.nan
        rows.append({
            "mode": mode,
            "group": "AlpacaEval",
            "method": method,
            "variant": variant,
            "median_stop_wall_time_s": med,
            "n_stop_runs": int(len(vals)),
        })
    return pd.DataFrame(rows)


def load_mode_stats(args: argparse.Namespace, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = args.stats_dir
    final = pd.read_csv(base / f"total_wall_time_{mode}_minbatch_5groups_1se_final_stats.csv")
    gs = pd.read_csv(base / f"total_wall_time_{mode}_minbatch_5groups_1se_gittins_stop_wall_stats.csv")
    bo = pd.read_csv(base / f"total_wall_time_{mode}_minbatch_5groups_1se_bo_stop_wall_stats.csv")
    final = pd.concat([final, alpaca_final_rows(args, mode)], ignore_index=True)
    final = pd.concat([final, sysrs_final_rows(args, mode)], ignore_index=True)
    gs = pd.concat([gs, stop_wall_rows(args, mode, "gittins")], ignore_index=True)
    bo = pd.concat([bo, stop_wall_rows(args, mode, "bo")], ignore_index=True)
    return final, gs, bo


def draw_break(ax_top: plt.Axes, ax_bot: plt.Axes, d: float = 0.014) -> None:
    kwargs = dict(color="k", clip_on=False, linewidth=1.1)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)


def plot_mode(args: argparse.Namespace, mode: str) -> None:
    final, gs_stop, bo_stop = load_mode_stats(args, mode)
    final["method_plot"] = final["method"].replace({"BO-LogEIPC": "BO-LogEI"})
    groups = GROUP_ORDER

    top_max = float(np.nanmax(final["median_s"] + final["se_s"]) * 1.16)
    top_max = max(top_max, 1000.0)

    fig, axes = plt.subplots(
        2,
        len(groups),
        figsize=(22.8, 6.35),
        gridspec_kw={"height_ratios": [0.62, 1.18]},
        constrained_layout=False,
    )
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.78

    for col, group in enumerate(groups):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        sub = final[final["group"].eq(group)].copy()
        vals = []
        errs = []
        for method in METHOD_ORDER:
            msub = sub[sub["method_plot"].eq(method)]
            vals.append(float(msub["median_s"].iloc[0]) if not msub.empty else np.nan)
            errs.append(float(msub["se_s"].iloc[0]) if not msub.empty else 0.0)
        colors = [METHOD_COLOR[m] for m in METHOD_ORDER]
        ax_top.bar(x, vals, yerr=errs, color=colors, edgecolor="black", linewidth=0.7, ecolor="black", capsize=3, width=width)
        ax_bot.bar(x, vals, yerr=errs, color=colors, edgecolor="black", linewidth=0.7, ecolor="black", capsize=3, width=width)

        gstop = gs_stop[gs_stop["group"].eq(group)]
        bstop = bo_stop[bo_stop["group"].eq(group)]
        for method, marker, face in [
            ("Gittins-S", "D", "white"),
            ("Gittins-G", "D", "white"),
            ("BO-PBGI", "o", "white"),
            ("BO-LogEI", "o", "white"),
            ("BO-LogEIPC", "o", "white"),
        ]:
            src = gstop if method.startswith("Gittins") else bstop
            lookup = method
            if method == "BO-LogEI" and mode == "aware":
                lookup = "BO-LogEIPC"
            row = src[src["method"].eq(lookup)]
            if row.empty:
                continue
            y = float(row["median_stop_wall_time_s"].iloc[0])
            if not np.isfinite(y):
                continue
            xpos = METHOD_ORDER.index("BO-LogEI" if lookup == "BO-LogEIPC" else method)
            for ax in [ax_top, ax_bot]:
                ax.scatter([xpos], [y], marker=marker, s=52, facecolor=face, edgecolor="white", linewidth=1.4, zorder=6)

        ax_bot.set_ylim(0, float(args.bottom_max))
        ax_top.set_ylim(float(args.top_min), top_max)
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_bot.set_xticks([])
        draw_break(ax_top, ax_bot)
        ax_top.set_title(group, fontsize=30, pad=4)
        for ax in [ax_top, ax_bot]:
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=18, width=0.9, length=3.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
        if col != 0:
            ax_top.set_yticklabels([])
            ax_bot.set_yticklabels([])

    fig.text(0.004, 0.55, "Time (s)", rotation="vertical", ha="center", va="center", fontsize=31)

    handles: list[object] = [
        Patch(facecolor=METHOD_COLOR["Gittins-S"], edgecolor="black", label="Gittins-S"),
        Patch(facecolor=METHOD_COLOR["Gittins-G"], edgecolor="black", label="Gittins-G"),
        Patch(facecolor=METHOD_COLOR["SySRs"], edgecolor="black", label="SySRs"),
        Patch(facecolor=METHOD_COLOR["UCB-E"], edgecolor="black", label="UCB-E"),
        Patch(facecolor=METHOD_COLOR["LRF"], edgecolor="black", label="LRF"),
        Patch(facecolor=METHOD_COLOR["BO-PBGI"], edgecolor="black", label="BO-PBGI"),
        Patch(facecolor=METHOD_COLOR["BO-LogEI"], edgecolor="black", label="BO-LogEIPC" if mode == "aware" else "BO-LogEI"),
        Line2D([0], [0], color="black", marker="D", markersize=8, linestyle="none", markerfacecolor="white", label="Gittins stop median"),
        Line2D([0], [0], color="black", marker="o", markersize=8, linestyle="none", markerfacecolor="white", label="BO stop median"),
        Line2D([0], [0], color="black", marker="_", markersize=22, linewidth=1.7, label=f"\u00b1{args.se_mult:g} SE"),
    ]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.028), fontsize=25, handlelength=2.2, columnspacing=1.32)
    fig.subplots_adjust(left=0.050, right=0.995, top=0.80, bottom=0.28, wspace=0.34, hspace=0.07)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"total_wall_time_{mode}_minbatch_6groups_1se_with_bo_stops_sysrs"
    png = args.out_dir / f"{stem}.png"
    pdf = args.out_dir / f"{stem}.pdf"
    csv = args.out_dir / f"{stem}_final_stats.csv"
    final.to_csv(csv, index=False)
    fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.16)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv}")


def main() -> int:
    args = parse_args()
    setup_matplotlib(args)
    plot_mode(args, "unit")
    plot_mode(args, "aware")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
