#!/usr/bin/env python3
"""Print starting y values for bandit curves in the latest MMLU figure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def norm_task(value: object) -> str:
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1]


def load_selected_tasks() -> dict[str, list[str]]:
    rows = json.loads(Path(r"data\MMLU_matrices\task_metadata.json").read_text(encoding="utf-8"))["tasks"]
    meta = {r["task"]: r for r in rows}

    def available(root: str) -> set[str]:
        df = pd.read_csv(Path(root) / "runs_summary.csv", usecols=lambda c: c in {"mmlu_task", "matrix_task"})
        out: set[str] = set()
        for col in df.columns:
            out.update(norm_task(v) for v in df[col].dropna().unique())
        return out

    by_size = {
        "Small": available(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_small"),
        "Large": available(r"outputs\wandb_downloads_new\ucb_gittins\mmlu_large"),
    }
    prior = {"Easy": "high", "Hard": "low"}
    out: dict[str, list[str]] = {}
    for difficulty in ["Easy", "Hard"]:
        for size in ["Small", "Large"]:
            tasks = [
                task
                for task, row in meta.items()
                if row.get("dataset_prior_bucket") == prior[difficulty] and task in by_size[size]
            ]
            out[f"{difficulty}-{size}"] = sorted(tasks)
    return out


def first_rows(path: str, variants: dict[str, str]) -> pd.DataFrame:
    inv = {v: k for k, v in variants.items()}
    usecols = ["run_id", "mmlu_task", "matrix_task", "experiment_variant", "cum_eval", "simple_regret"]
    best: dict[tuple[str, str, str], tuple[float, float]] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000):
        chunk = chunk[chunk["experiment_variant"].isin(inv)].copy()
        if chunk.empty:
            continue
        raw_task = chunk["mmlu_task"].where(chunk["mmlu_task"].notna(), chunk["matrix_task"])
        chunk["task"] = raw_task.map(norm_task)
        chunk["method"] = chunk["experiment_variant"].map(inv)
        chunk = chunk.sort_values("cum_eval").groupby(["task", "method", "run_id"], as_index=False).first()
        for row in chunk.itertuples(index=False):
            key = (row.task, row.method, row.run_id)
            old = best.get(key)
            if old is None or float(row.cum_eval) < old[0]:
                best[key] = (float(row.cum_eval), float(row.simple_regret))
    return pd.DataFrame(
        {"task": k[0], "method": k[1], "run_id": k[2], "first_regret": v[1]}
        for k, v in best.items()
    )


def main() -> int:
    selected = load_selected_tasks()
    specs = [
        (
            "Small",
            r"outputs\wandb_downloads_new\ucb_gittins\mmlu_small\runs_history.csv.gz",
            {
                "Gittins-S": "gittins_unit_B2_scale1e-4_dataset",
                "Gittins-G": "gittins_unit_B2_scale1e-4_default",
                "UCB-E": "ucb_B2",
            },
        ),
        (
            "Large",
            r"outputs\wandb_downloads_new\ucb_gittins\mmlu_large\runs_history.csv.gz",
            {
                "Gittins-S": "gittins_unit_B8_scale1e-4_dataset",
                "Gittins-G": "gittins_unit_B8_scale1e-4_default",
                "UCB-E": "ucb_B8",
            },
        ),
    ]
    first = pd.concat([first_rows(path, variants) for _, path, variants in specs], ignore_index=True)
    denom = first.groupby("task")["first_regret"].mean().rename("denom")
    first = first.join(denom, on="task")
    first["start_y"] = first["first_regret"] / first["denom"]

    rows = []
    for group, tasks in selected.items():
        sub = first[first["task"].isin(tasks)]
        for method in ["Gittins-S", "Gittins-G", "UCB-E"]:
            vals = sub[sub["method"] == method]["start_y"].dropna()
            rows.append(
                {
                    "group": group,
                    "method": method,
                    "start_y_mean": vals.mean(),
                    "start_y_std": vals.std(),
                    "n": len(vals),
                }
            )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
