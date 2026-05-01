#!/usr/bin/env python3
"""Plot mean simple-regret curves from downloaded W&B history.

Aggregates repeated run_seeds by experiment_variant.
Each run is interpolated onto a common x-grid, then mean +/- std is plotted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-csv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. gsm8k or piqa")
    p.add_argument("--matrix-seed", default=None, help="Optional matrix/data seed filter, e.g. 1")
    p.add_argument("--run-seed", default=None, help="Optional algorithm run seed filter. Usually leave unset for mean/std aggregation.")
    p.add_argument("--x-axis", choices=["cum_eval", "cum_original_cost", "step_idx"], default="cum_eval")
    p.add_argument("--y-axis", default="simple_regret")
    p.add_argument("--group-by", default="experiment_variant")
    p.add_argument("--variant-regex", default=None, help="Optional regex filter on experiment_variant")
    p.add_argument("--max-variants", type=int, default=12, help="Limit number of plotted variants after filtering")
    p.add_argument("--grid-size", type=int, default=300)
    p.add_argument("--range", choices=["std", "stderr", "none"], default="std")
    p.add_argument("--title", default=None)
    return p.parse_args()


def aggregate_variant(df: pd.DataFrame, x_col: str, y_col: str, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runs = []
    max_x = 0.0
    for _, g in df.groupby("run_id"):
        gg = g[[x_col, y_col]].dropna().sort_values(x_col)
        if gg.empty:
            continue
        gg = gg.groupby(x_col, as_index=False)[y_col].last()
        x = gg[x_col].to_numpy(dtype=float)
        y = gg[y_col].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        runs.append((x, y))
        max_x = max(max_x, float(np.nanmax(x)))

    if not runs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_grid = np.linspace(0.0, max_x, grid_size)
    ys = []
    for x, y in runs:
        yi = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
        ys.append(yi)
    y_arr = np.vstack(ys)
    mean = np.nanmean(y_arr, axis=0)
    std = np.nanstd(y_arr, axis=0)
    n = np.sum(~np.isnan(y_arr), axis=0)
    stderr = std / np.sqrt(np.maximum(n, 1))
    return x_grid, mean, std, stderr


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.history_csv)
    if args.dataset is not None:
        dataset = args.dataset.lower()
        mask = (df.get("dataset_tag_resolved", "").astype(str).str.lower() == dataset) | (df.get("dataset_tag", "").astype(str).str.lower() == dataset)
        df = df[mask]
    if args.matrix_seed is not None:
        df = df[df["matrix_seed"].astype(str) == str(args.matrix_seed)]
    if args.run_seed is not None:
        df = df[df["run_seed"].astype(str) == str(args.run_seed)]
    if args.variant_regex is not None:
        rx = re.compile(args.variant_regex)
        df = df[df["experiment_variant"].astype(str).map(lambda x: bool(rx.search(x)))]

    df = df.dropna(subset=[args.x_axis, args.y_axis, args.group_by, "run_id"])
    variants = sorted(df[args.group_by].dropna().astype(str).unique().tolist())

    if len(variants) > args.max_variants:
        final_rows = []
        for variant, vg in df.groupby(args.group_by):
            vals = []
            for _, rg in vg.groupby("run_id"):
                rr = rg[[args.x_axis, args.y_axis]].dropna().sort_values(args.x_axis)
                if not rr.empty:
                    vals.append(float(rr[args.y_axis].iloc[-1]))
            if vals:
                final_rows.append((str(variant), float(np.mean(vals))))
        variants = [v for v, _ in sorted(final_rows, key=lambda t: t[1])[: args.max_variants]]
        print(f"Too many variants; plotting top {len(variants)} by mean final {args.y_axis}:")
        for v in variants:
            print("  ", v)

    if not variants:
        raise SystemExit("No variants left after filtering.")

    plt.figure(figsize=(10, 6))
    aggregate_rows = []
    for variant in variants:
        vg = df[df[args.group_by].astype(str) == variant]
        x_grid, mean, std, stderr = aggregate_variant(vg, args.x_axis, args.y_axis, args.grid_size)
        if x_grid.size == 0:
            continue
        plt.plot(x_grid, mean, label=variant, linewidth=1.8)
        if args.range != "none":
            band = std if args.range == "std" else stderr
            plt.fill_between(x_grid, mean - band, mean + band, alpha=0.15)
        for x, m, s, se in zip(x_grid, mean, std, stderr):
            aggregate_rows.append({"variant": variant, "x_axis": args.x_axis, "x": x, "mean": m, "std": s, "stderr": se})

    title = args.title or f"{args.dataset or 'all'} matrix_seed={args.matrix_seed or 'all'}: {args.y_axis} vs {args.x_axis}"
    plt.title(title)
    plt.xlabel(args.x_axis)
    plt.ylabel(args.y_axis)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    name_parts = [args.dataset or "all", f"seed{args.matrix_seed or 'all'}", args.y_axis, "vs", args.x_axis]
    if args.variant_regex:
        name_parts.append(safe_token(args.variant_regex)[:80])
    out_png = args.out_dir / ("_".join(name_parts) + ".png")
    out_csv = args.out_dir / ("_".join(name_parts) + "_aggregated.csv")

    plt.savefig(out_png, dpi=180)
    plt.close()
    pd.DataFrame(aggregate_rows).to_csv(out_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
