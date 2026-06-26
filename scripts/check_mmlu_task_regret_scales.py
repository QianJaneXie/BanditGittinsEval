#!/usr/bin/env python3
"""Check per-task raw simple-regret scales used by MMLU normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_mmlu_aggregate_2x2_normalized_fast_shared_labels import (
    compute_bandit_initial_mean_denominators,
    compute_task_initial_denominators,
    normalize_task_name,
    read_filtered_history,
    safe_token,
    task_column,
    tasks_for_group,
    tasks_for_size,
    variant_names,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small-root", type=Path, default=Path(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_small"))
    p.add_argument("--large-root", type=Path, default=Path(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_large"))
    p.add_argument("--small-lrf-root", type=Path, default=Path(r"outputs\wandb_downloads_new\lrf\mmlu_small_lrf"))
    p.add_argument("--large-lrf-root", type=Path, default=Path(r"outputs\wandb_downloads_new\lrf\mmlu_large_lrf"))
    p.add_argument("--small-bo-root", type=Path, default=Path(r"outputs\wandb_downloads_new\bo_baseline_5pct\mmlu_small_bo"))
    p.add_argument("--large-bo-root", type=Path, default=Path(r"outputs\wandb_downloads_new\bo_baseline_5pct\mmlu_large_bo"))
    p.add_argument("--task-metadata", type=Path, default=Path(r"data\MMLU_matrices\task_metadata.json"))
    p.add_argument("--out-csv", type=Path, default=Path(r"outputs\wandb_plots_new\paper_figures\mmlu_task_initial_denominators_Bsmall2_Blarge8.csv"))
    p.add_argument("--cost-mode", choices=["unit", "aware"], default="unit")
    p.add_argument("--denominator", choices=["task_initial", "bandit_initial_mean"], default="task_initial")
    p.add_argument("--small-batch-size", type=int, default=2)
    p.add_argument("--large-batch-size", type=int, default=8)
    p.add_argument("--small-lrf-batch-size", type=int, default=32)
    p.add_argument("--large-lrf-batch-size", type=int, default=32)
    p.add_argument("--scale", default="1e-4")
    p.add_argument("--bo-pbgi-unit-variant", default="pbgi_unit")
    p.add_argument("--bo-logei-unit-variant", default="logei_unit")
    p.add_argument("--bo-pbgi-cost-variant", default="pbgi_cost")
    p.add_argument("--bo-logei-cost-variant", default="logeipc_cost")
    p.add_argument("--y-eps", type=float, default=1e-8)
    return p.parse_args()


def load_metadata(path: Path) -> dict[str, dict]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["task"]): row for row in data.get("tasks", []) if isinstance(row, dict) and "task" in row}


def selected_tasks(args: argparse.Namespace) -> dict[str, list[str]]:
    metadata = load_metadata(args.task_metadata)
    by_size = {
        "small": tasks_for_size(args.small_root),
        "large": tasks_for_size(args.large_root),
    }
    out: dict[str, list[str]] = {}
    for difficulty in ["Easy", "Hard"]:
        for size, root in [("Small", args.small_root), ("Large", args.large_root)]:
            size_key = size.lower()
            candidates = tasks_for_group(metadata, size_bucket=size_key, difficulty=difficulty)
            available = by_size[size_key]
            out[f"{difficulty}-{size}"] = sorted([t for t in candidates if not available or t in available])
    return out


def read_source(root: Path, variants: set[str], tasks: set[str], x_col: str) -> pd.DataFrame:
    path = root / "runs_history.csv.gz"
    df = read_filtered_history(path, variants, x_col=x_col, tasks=tasks)
    if not df.empty:
        df["_mmlu_task_for_plot"] = task_column(df, sorted(tasks))
    return df


def main() -> int:
    args = parse_args()
    x_col = "cum_eval" if args.cost_mode == "unit" else "cum_original_cost"
    tasks_by_group = selected_tasks(args)
    rows: list[dict] = []

    for group, tasks in tasks_by_group.items():
        size = group.split("-")[1]
        is_small = size == "Small"
        batch = args.small_batch_size if is_small else args.large_batch_size
        lrf_batch = args.small_lrf_batch_size if is_small else args.large_lrf_batch_size
        variants = variant_names(batch, lrf_batch, str(args.scale), args.cost_mode, args)
        roots = {
            "gittins_data": args.small_root if is_small else args.large_root,
            "gittins_default": args.small_root if is_small else args.large_root,
            "ucb": args.small_root if is_small else args.large_root,
            "lrf": args.small_lrf_root if is_small else args.large_lrf_root,
            "bo_pbgi_unit": args.small_bo_root if is_small else args.large_bo_root,
            "bo_logei_unit": args.small_bo_root if is_small else args.large_bo_root,
            "bo_pbgi_cost": args.small_bo_root if is_small else args.large_bo_root,
            "bo_logeipc_cost": args.small_bo_root if is_small else args.large_bo_root,
        }

        loaded = []
        for root in sorted({roots[k] for k in variants}):
            kinds = [k for k in variants if roots[k] == root]
            wanted = {variants[k] for k in kinds}
            df = read_source(root, wanted, set(tasks), x_col)
            loaded.append({"df": df})

        if args.denominator == "bandit_initial_mean":
            denoms = compute_bandit_initial_mean_denominators(loaded, x_col=x_col, y_eps=float(args.y_eps))
        else:
            denoms = compute_task_initial_denominators(loaded, x_col=x_col, y_eps=float(args.y_eps))
        for task in tasks:
            rows.append(
                {
                    "cost_mode": args.cost_mode,
                    "group": group,
                    "task": normalize_task_name(task),
                    "denominator": denoms.get(task, np.nan),
                }
            )
        print(f"{group}: {len(denoms)}/{len(tasks)} denominators")

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(out.groupby("group")["denominator"].describe().to_string())
    finite = out["denominator"].dropna()
    if not finite.empty:
        print("\nOverall:")
        print(f"min={finite.min():.6g}, median={finite.median():.6g}, max={finite.max():.6g}, max/min={finite.max()/finite.min():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
