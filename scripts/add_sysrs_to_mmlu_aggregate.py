#!/usr/bin/env python3
"""Add SySRs curves to the cached final MMLU 2x2 paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_STEM = (
    "mmlu_aggregate_2x2_{mode}_Bsmall2_Blarge8_scale1e-4_"
    "xfinal_ynone_fast_lrf_bo_xoffset_bo_after_init"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/figure/new_figure/final/mmlu2_2"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figure/new_figure/final/mmlu2_2"),
    )
    p.add_argument(
        "--small-sysrs",
        type=Path,
        default=Path("outputs/wandb_downloads_new/sysrs/mmlu_small/runs_history.csv.gz"),
    )
    p.add_argument(
        "--large-sysrs",
        type=Path,
        default=Path("outputs/wandb_downloads_new/sysrs/mmlu_large/runs_history.csv.gz"),
    )
    p.add_argument("--grid-size", type=int, default=350)
    p.add_argument("--modes", nargs="+", choices=["unit", "aware"], default=["unit", "aware"])
    return p.parse_args()


def normalize_task(value: object) -> str:
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1]


def read_sysrs(path: Path, variant: str, tasks: set[str]) -> pd.DataFrame:
    wanted = {
        "run_id",
        "experiment_variant",
        "mmlu_task",
        "matrix_task",
        "cum_eval",
        "cum_original_cost",
        "simple_regret",
    }
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=lambda col: col in wanted,
        chunksize=50_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["experiment_variant"].astype(str) == variant].copy()
        if chunk.empty:
            continue
        task = pd.Series("", index=chunk.index, dtype=str)
        for col in ["mmlu_task", "matrix_task"]:
            if col in chunk.columns:
                values = chunk[col].map(normalize_task)
                task = task.mask((task == "") & values.notna(), values)
        chunk["_task"] = task
        chunk = chunk[chunk["_task"].isin(tasks)]
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True, sort=False)


def run_curve(run: pd.DataFrame, x_col: str, grid_size: int) -> np.ndarray | None:
    points = run[[x_col, "simple_regret"]].dropna().sort_values(x_col)
    points = points.groupby(x_col, as_index=False)["simple_regret"].last()
    if len(points) < 2:
        return None
    x = pd.to_numeric(points[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(points["simple_regret"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or x[-1] <= x[0]:
        return None
    x = (x - x[0]) / (x[-1] - x[0])
    return np.interp(np.linspace(0.0, 1.0, grid_size), x, y)


def aggregate_rows(
    history: pd.DataFrame,
    *,
    group: str,
    tasks: list[str],
    variant: str,
    x_col: str,
    grid_size: int,
) -> list[dict[str, object]]:
    selected = history[history["_task"].isin(set(tasks))].copy()
    curves: list[np.ndarray] = []
    used_tasks: set[str] = set()
    for _, run in selected.groupby("run_id", sort=False):
        curve = run_curve(run, x_col, grid_size)
        if curve is None:
            continue
        curves.append(curve)
        used_tasks.update(run["_task"].dropna().astype(str).unique())
    if not curves:
        raise RuntimeError(f"No SySRs curves found for {group} / {variant}")

    values = np.vstack(curves)
    n = np.sum(np.isfinite(values), axis=0)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    stderr = std / np.sqrt(np.maximum(n, 1))
    x_grid = np.linspace(0.0, 1.0, grid_size)
    rows: list[dict[str, object]] = []
    for x, avg, sd, se, count in zip(
        x_grid, mean, std, stderr, n, strict=True
    ):
        rows.append(
            {
                "group": group,
                "method_kind": "sysrs",
                "method": "SySRs",
                "variant": variant,
                "n_tasks_available": len(tasks),
                "n_tasks_used": len(used_tasks),
                "n_curves_total": len(curves),
                "n_curves_at_x": int(count),
                "x": float(x),
                "mean": float(avg),
                "std": float(sd),
                "stderr": float(se),
            }
        )
    print(
        f"{group} {variant}: {len(curves)} curves, "
        f"{len(used_tasks)}/{len(tasks)} tasks"
    )
    return rows


def process_mode(args: argparse.Namespace, mode: str) -> None:
    stem = BASE_STEM.format(mode=mode)
    aggregate_path = args.base_dir / f"{stem}_aggregated.csv"
    groups_path = args.base_dir / f"{stem}_groups.json"
    if not aggregate_path.is_file() or not groups_path.is_file():
        raise FileNotFoundError(f"Missing cached final MMLU data for {mode}")

    metadata = json.loads(groups_path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = metadata["selected_tasks"]
    all_small = {
        task for group, tasks in groups.items() if group.endswith("-Small") for task in tasks
    }
    all_large = {
        task for group, tasks in groups.items() if group.endswith("-Large") for task in tasks
    }
    variant = "sysrs" if mode == "unit" else "sysrs_cost"
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"
    small = read_sysrs(args.small_sysrs, variant, all_small)
    large = read_sysrs(args.large_sysrs, variant, all_large)

    rows: list[dict[str, object]] = []
    for group in ["Easy-Small", "Easy-Large", "Hard-Small", "Hard-Large"]:
        history = small if group.endswith("-Small") else large
        rows.extend(
            aggregate_rows(
                history,
                group=group,
                tasks=groups[group],
                variant=variant,
                x_col=x_col,
                grid_size=int(args.grid_size),
            )
        )

    sysrs = pd.DataFrame(rows)
    original = pd.read_csv(aggregate_path)
    original = original[original["method_kind"].astype(str) != "sysrs"]
    combined = pd.concat([original, sysrs], ignore_index=True, sort=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        args.out_dir / f"{stem}_with_sysrs_aggregated.csv", index=False
    )
    sysrs.to_csv(args.out_dir / f"{stem}_sysrs_only.csv", index=False)

    metadata["sysrs"] = {
        "variant": variant,
        "color": "tab:pink",
        "small_history": str(args.small_sysrs),
        "large_history": str(args.large_sysrs),
        "aggregation": "same run-level x-final normalization and 350-point grid as cached paper curves",
    }
    metadata["paper_priors"] = {
        "Gittins-G": {"mean": 0.5, "variance": 0.04},
        "Gittins-S Easy/high": {"mean": 0.75, "variance": 0.01},
        "Gittins-S Hard/low": {"mean": 0.4, "variance": 0.02},
    }
    metadata["base_cache"] = str(aggregate_path)
    (args.out_dir / f"{stem}_with_sysrs_groups.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote combined and SySRs-only caches for {mode}")


def main() -> int:
    args = parse_args()
    for mode in args.modes:
        process_mode(args, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
