#!/usr/bin/env python3
"""Append currently available MMLU large LRF curves to an aggregate CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    base = Path(r"outputs\wandb_plots_new\paper_figures")
    stem = "mmlu_aggregate_2x2_unit_Bsmall4_Blarge16_scale1e-4_xfinal_yinitial_fast"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregated", type=Path, default=base / f"{stem}_aggregated.csv")
    p.add_argument("--groups", type=Path, default=base / f"{stem}_groups.json")
    p.add_argument("--large-lrf-history", type=Path, default=Path(r"outputs\wandb_downloads_new\lrf\mmlu_large_lrf\runs_history.csv.gz"))
    p.add_argument("--out", type=Path, default=base / f"{stem}_with_large_lrf_aggregated.csv")
    p.add_argument("--variant", default="lrf_B32")
    p.add_argument("--grid-size", type=int, default=350)
    return p.parse_args()


def normalize_task(v: object) -> str:
    return str(v).replace("\\", "/").rstrip("/").split("/")[-1]


def curve_from_run(g: pd.DataFrame, grid_size: int) -> np.ndarray | None:
    warmup = pd.to_numeric(g.get("warmup_percentage"), errors="coerce").dropna()
    n_cells = pd.to_numeric(g.get("n_cells"), errors="coerce").dropna()
    if not warmup.empty and not n_cells.empty:
        min_eval = math.ceil(float(warmup.iloc[0]) * float(n_cells.iloc[0]))
        g = g[pd.to_numeric(g["cum_eval"], errors="coerce") >= min_eval]

    gg = g[["cum_eval", "simple_regret"]].dropna().sort_values("cum_eval")
    gg = gg.groupby("cum_eval", as_index=False)["simple_regret"].last()
    if len(gg) < 2:
        return None

    x = pd.to_numeric(gg["cum_eval"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(gg["simple_regret"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 2 or x[-1] <= 0:
        return None

    # Match the current plotted aggregate: LRF begins at its post-warmup
    # cumulative-budget fraction rather than being shifted back to zero.
    x = x / float(x[-1])
    y = y / max(abs(float(y[0])), 1e-8)
    x_grid = np.linspace(0.0, 1.0, int(grid_size))
    return np.interp(x_grid, x, y, left=np.nan, right=np.nan)


def aggregate(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.linspace(0.0, 1.0, int(len(curves[0])))
    arr = np.vstack(curves)
    valid = np.isfinite(arr)
    n = valid.sum(axis=0)
    mean = np.full(arr.shape[1], np.nan)
    std = np.full(arr.shape[1], np.nan)
    stderr = np.full(arr.shape[1], np.nan)
    ok = n > 0
    mean[ok] = np.nanmean(arr[:, ok], axis=0)
    ok_std = n > 1
    std[ok_std] = np.nanstd(arr[:, ok_std], axis=0, ddof=1)
    stderr[ok_std] = std[ok_std] / np.sqrt(n[ok_std])
    std[ok & ~ok_std] = 0.0
    stderr[ok & ~ok_std] = 0.0
    return x_grid, mean, std, stderr, n.astype(float)


def main() -> int:
    args = parse_args()
    agg = pd.read_csv(args.aggregated)
    groups = json.loads(args.groups.read_text(encoding="utf-8"))["selected_tasks"]

    df = pd.read_csv(
        args.large_lrf_history,
        compression="gzip",
        usecols=lambda c: c
        in {
            "run_id",
            "experiment_variant",
            "mmlu_task",
            "matrix_task",
            "cum_eval",
            "simple_regret",
            "warmup_percentage",
            "n_cells",
        },
    )
    df = df[df["experiment_variant"].astype(str) == str(args.variant)].copy()
    task_col = df["mmlu_task"].map(normalize_task)
    rows = []
    for group in ["Easy-Large", "Hard-Large"]:
        tasks = set(groups[group])
        gdf = df[task_col.isin(tasks)].copy()
        curves = []
        used_tasks = set()
        for run_id, rg in gdf.groupby("run_id", sort=False):
            yi = curve_from_run(rg, int(args.grid_size))
            if yi is not None:
                curves.append(yi)
                used_tasks.update(rg["mmlu_task"].map(normalize_task).unique())
        if not curves:
            continue
        x_grid, mean, std, stderr, n = aggregate(curves)
        for x, m, s, se, nn in zip(x_grid, mean, std, stderr, n):
            rows.append(
                {
                    "group": group,
                    "method_kind": "lrf",
                    "method": "LRF",
                    "variant": args.variant,
                    "n_tasks_available": len(tasks),
                    "n_tasks_used": len(used_tasks),
                    "n_curves_total": len(curves),
                    "n_curves_at_x": int(nn),
                    "x": float(x),
                    "mean": float(m) if np.isfinite(m) else np.nan,
                    "std": float(s) if np.isfinite(s) else np.nan,
                    "stderr": float(se) if np.isfinite(se) else np.nan,
                }
            )
        print(f"{group}: added {len(curves)} LRF curves from {len(used_tasks)}/{len(tasks)} tasks")

    out = pd.concat([agg, pd.DataFrame(rows)], ignore_index=True, sort=False)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
