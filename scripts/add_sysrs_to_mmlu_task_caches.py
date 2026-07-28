#!/usr/bin/env python3
"""Append SySRs to the saved 57-task MMLU plotting caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent / "plot"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

import plot_mmlu_easy_task_panels_bo5pct as panels  # noqa: E402


PRIOR_BY_DIFFICULTY = {
    "easy": ("high", 0.75, 0.01),
    "medium": ("medium", 0.6, 0.02),
    "hard": ("low", 0.4, 0.02),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-root",
        type=Path,
        default=Path("outputs/figure/new_figure/final"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/figure/new_figure/final/mmlu_task_panels_sysrs"),
    )
    p.add_argument(
        "--sysrs-root",
        type=Path,
        default=Path("outputs/wandb_downloads_new/sysrs"),
    )
    p.add_argument("--grid-size", type=int, default=320)
    p.add_argument("--se-mult", type=float, default=1.0)
    return p.parse_args()


def all_tasks() -> list[str]:
    return [
        task
        for difficulty in ["easy", "medium", "hard"]
        for task in panels.TASKS_BY_BUCKET[difficulty]
    ]


def task_difficulty() -> dict[str, str]:
    return {
        task: difficulty
        for difficulty, tasks in panels.TASKS_BY_BUCKET.items()
        for task in tasks
    }


def read_group_history(
    args: argparse.Namespace, group: str, mode: str, tasks: list[str]
) -> pd.DataFrame:
    path = args.sysrs_root / f"mmlu_{group}" / "runs_history.csv.gz"
    variant = "sysrs" if mode == "unit" else "sysrs_cost"
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"
    df = panels.read_filtered_history(
        path, {variant}, x_col=x_col, tasks=set(tasks)
    )
    if df.empty:
        raise RuntimeError(f"No {variant} rows found under {path}")
    df["_mmlu_task_for_plot"] = panels.task_column(df, tasks)
    return df


def sysrs_curve(
    df: pd.DataFrame,
    task: str,
    mode: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    x_col = "cum_eval" if mode == "unit" else "cum_original_cost"
    task_df = df[df["_mmlu_task_for_plot"].astype(str) == task]
    x, y_arr = panels.aggregate_runs_to_grid(
        task_df,
        x_col,
        "simple_regret",
        int(args.grid_size),
    )
    if x.size == 0:
        raise RuntimeError(f"No plottable SySRs curve for {task}/{mode}")
    mean, lo, hi = panels.band_from_yarr(
        y_arr, float(args.se_mult), "stderr"
    )
    return pd.DataFrame(
        {
            "mmlu_task": task,
            "mode": mode,
            "method_kind": "sysrs",
            "method_label": "SySRs",
            "x": x,
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "n_runs": np.sum(~np.isnan(y_arr), axis=0),
            "x_label": x_col,
        }
    )


def recompute_limits(curves: pd.DataFrame, meta: dict[str, object]) -> None:
    old_xlim = meta.get("xlim")
    preserve_zero_left = (
        isinstance(old_xlim, list)
        and len(old_xlim) == 2
        and abs(float(old_xlim[0])) < 1e-12
    )
    xs = pd.to_numeric(curves["x"], errors="coerce").dropna().to_numpy(float)
    ys = np.concatenate(
        [
            pd.to_numeric(curves[col], errors="coerce").dropna().to_numpy(float)
            for col in ["lo", "hi", "mean"]
        ]
    )
    for stop in meta.get("stop_lines", []):
        for key in ["lo", "hi", "x"]:
            if key in stop and np.isfinite(float(stop[key])):
                xs = np.append(xs, float(stop[key]))
    if xs.size:
        left = 0.0 if preserve_zero_left else float(np.min(xs))
        right = float(np.max(xs))
        pad = (
            0.08 * (right - left)
            if right > left
            else max(1.0, abs(right) * 0.08)
        )
        meta["xlim"] = [left, right + pad]
    if ys.size:
        low, high = float(np.min(ys)), float(np.max(ys))
        pad = (
            0.05 * (high - low)
            if high > low
            else max(0.01, abs(high) * 0.05)
        )
        meta["ylim"] = [low - pad, high + pad]


def write_prior_table(args: argparse.Namespace) -> None:
    difficulty_by_task = task_difficulty()
    rows = []
    for task in all_tasks():
        difficulty = difficulty_by_task[task]
        bucket, mean, variance = PRIOR_BY_DIFFICULTY[difficulty]
        rows.append(
            {
                "mmlu_task": task,
                "difficulty": difficulty,
                "prior_bucket": bucket,
                "prior_mean": mean,
                "prior_variance": variance,
                "size_bucket": panels.TASK_GROUP[task],
                "paper_table5_match": True,
            }
        )
    pd.DataFrame(rows).to_csv(args.out_root / "mmlu_task_priors.csv", index=False)


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    difficulty_by_task = task_difficulty()

    for mode in ["unit", "aware"]:
        for group in ["small", "medium", "large"]:
            tasks = [task for task in all_tasks() if panels.TASK_GROUP[task] == group]
            history = read_group_history(args, group, mode, tasks)
            for task in tasks:
                base_curve = (
                    args.base_root
                    / "processed_curves"
                    / mode
                    / f"{task}_curves.csv"
                )
                base_meta = (
                    args.base_root
                    / "processed_curves"
                    / mode
                    / f"{task}_meta.json"
                )
                if not base_curve.is_file() or not base_meta.is_file():
                    raise FileNotFoundError(f"Missing paper cache for {task}/{mode}")

                curves = pd.read_csv(base_curve)
                curves = curves[curves["method_kind"].astype(str) != "sysrs"]
                addition = sysrs_curve(history, task, mode, args)
                combined = pd.concat(
                    [curves, addition], ignore_index=True, sort=False
                )
                meta = json.loads(base_meta.read_text(encoding="utf-8"))
                difficulty = difficulty_by_task[task]
                bucket, mean, variance = PRIOR_BY_DIFFICULTY[difficulty]
                meta["style_keys_present"] = sorted(
                    combined["method_kind"].dropna().astype(str).unique()
                )
                meta["sysrs"] = {
                    "variant": "sysrs" if mode == "unit" else "sysrs_cost",
                    "color": "tab:pink",
                    "n_runs": int(addition["n_runs"].max()),
                    "source": str(
                        args.sysrs_root
                        / f"mmlu_{group}"
                        / "runs_history.csv.gz"
                    ),
                }
                meta["paper_prior"] = {
                    "difficulty": difficulty,
                    "bucket": bucket,
                    "Gittins-S_mean": mean,
                    "Gittins-S_variance": variance,
                    "Gittins-G_mean": 0.5,
                    "Gittins-G_variance": 0.04,
                }
                recompute_limits(combined, meta)

                out_dir = args.out_root / "processed_curves" / mode
                out_dir.mkdir(parents=True, exist_ok=True)
                out_curve = out_dir / f"{task}_curves.csv"
                original_text = base_curve.read_text(encoding="utf-8").rstrip(
                    "\r\n"
                )
                addition_text = addition.to_csv(
                    index=False,
                    header=False,
                    columns=curves.columns,
                    lineterminator="\n",
                ).rstrip("\r\n")
                out_curve.write_text(
                    f"{original_text}\n{addition_text}\n", encoding="utf-8"
                )
                (out_dir / f"{task}_meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
            print(f"{mode}/{group}: added SySRs to {len(tasks)} task caches")

    write_prior_table(args)
    print(f"Wrote augmented caches and priors under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
