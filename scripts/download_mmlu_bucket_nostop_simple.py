#!/usr/bin/env python3
"""Simple no-stopping downloader for one MMLU size bucket.

This script is intentionally simple:
  - no checkpoint/resume
  - no stopping artifacts
  - no *_traces.npz download
  - downloads only runs_summary.csv and runs_history.csv.gz
  - writes both a raw merged folder and per-task folders while downloading

For MMLU-large:
python scripts/download_mmlu_bucket_nostop_simple.py --entity EfficientLLMEval --project GittinsBanditEval --sweep-id o5z7qtml --size-bucket large --raw-dir outputs/wandb_downloads/mmlu_large_raw_finished_nostop --out-root outputs/wandb_downloads/mmlu_large_merged_finished_nostop --task-metadata data/MMLU_matrices/task_metadata.json --states finished
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


HISTORY_KEYS = [
    "cum_eval",
    "cum_original_cost",
    "simple_regret",
    "recommended_arm",
    "recommended_mean",
    "iter_step_s",
    "iter_total_s",
    "batch_cells",
    "step_idx",
]

BASE_COLUMNS = [
    "run_id",
    "run_name",
    "state",
    "url",
    "dataset_tag_resolved",
    "dataset_tag",
    "matrix",
    "matrix_task",
    "matrix_seed",
    "run_seed",
    "experiment_variant",
    "policy_variant",
    "policy_family",
    "cost_mode",
    "batch_size",
    "gittins_batch_size",
    "cost_scaling_factor",
    "prior_type",
    "prior_mean_resolved",
    "prior_variance_resolved",
    "n_arms",
    "n_examples",
    "n_cells",
    "budget_max_evals",
    "eval_budget_fraction",
    "warmup_percentage",
]

SUMMARY_COLUMNS = BASE_COLUMNS + [
    "final_simple_regret",
    "best_seen_regret",
    "final_cum_eval",
    "final_cum_original_cost",
    "num_batches",
    "lookup_table_s",
    "iter_step_mean_s",
    "iter_step_median_s",
    "iter_step_p90_s",
    "iter_total_mean_s",
    "iter_total_median_s",
    "iter_total_p90_s",
]

HISTORY_COLUMNS = BASE_COLUMNS + HISTORY_KEYS


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_variant_fallback(experiment_variant: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not experiment_variant:
        return out

    v = str(experiment_variant)
    m = re.fullmatch(r"(ucb|lrf)_B(\d+)", v)
    if m:
        policy = m.group(1)
        b = int(m.group(2))
        out.update(
            policy_variant=policy,
            policy_family=policy,
            cost_mode="baseline",
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=1e-4,
            prior_type="default",
        )
        return out

    m = re.fullmatch(
        r"gittins_(unit|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)",
        v,
    )
    if m:
        cost_mode = m.group(1)
        b = int(m.group(2))
        out.update(
            policy_variant=f"gittins_{cost_mode}",
            policy_family="gittins",
            cost_mode=cost_mode,
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=float(m.group(3)),
            prior_type=m.group(4),
        )
        return out

    return out


def matrix_task_from_matrix(matrix_value: Any) -> str | None:
    if matrix_value is None:
        return None
    stem = Path(str(matrix_value)).stem
    return stem if stem else None


def get_field(cfg: dict[str, Any], summary: dict[str, Any], key: str) -> Any:
    if key in summary and summary[key] is not None:
        return summary[key]
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return None


def base_row(run: wandb.apis.public.Run) -> dict[str, Any]:
    cfg = dict(run.config)
    summary = dict(run.summary)

    exp = cfg.get("experiment_variant") or cfg.get("raw") or cfg.get("policy_variant")
    fallback = parse_variant_fallback(exp)
    matrix = cfg.get("matrix") or summary.get("matrix")
    matrix_task = (
        cfg.get("matrix_task")
        or cfg.get("task")
        or summary.get("matrix_task")
        or matrix_task_from_matrix(matrix)
    )

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "dataset_tag_resolved": cfg.get("dataset_tag_resolved"),
        "dataset_tag": cfg.get("dataset_tag"),
        "matrix": matrix,
        "matrix_task": matrix_task,
        "matrix_seed": get_field(cfg, summary, "matrix_seed"),
        "run_seed": cfg.get("run_seed"),
        "experiment_variant": exp,
        "policy_variant": cfg.get("policy_variant", fallback.get("policy_variant")),
        "policy_family": cfg.get("policy_family", fallback.get("policy_family")),
        "cost_mode": cfg.get("cost_mode", fallback.get("cost_mode")),
        "batch_size": cfg.get("batch_size", fallback.get("batch_size")),
        "gittins_batch_size": cfg.get("gittins_batch_size", fallback.get("gittins_batch_size")),
        "cost_scaling_factor": cfg.get("cost_scaling_factor", fallback.get("cost_scaling_factor")),
        "prior_type": cfg.get("prior_type", fallback.get("prior_type")),
        "prior_mean_resolved": get_field(cfg, summary, "prior_mean_resolved"),
        "prior_variance_resolved": get_field(cfg, summary, "prior_variance_resolved"),
        "n_arms": get_field(cfg, summary, "n_arms"),
        "n_examples": get_field(cfg, summary, "n_examples"),
        "n_cells": get_field(cfg, summary, "n_cells"),
        "budget_max_evals": get_field(cfg, summary, "budget_max_evals"),
        "eval_budget_fraction": get_field(cfg, summary, "eval_budget_fraction"),
        "warmup_percentage": get_field(cfg, summary, "warmup_percentage"),
    }


def load_bucket_tasks(task_metadata: Path, size_bucket: str) -> set[str]:
    data = json.loads(task_metadata.read_text(encoding="utf-8"))
    tasks = {
        str(t["task"])
        for t in data.get("tasks", [])
        if str(t.get("size_bucket", "")).lower() == str(size_bucket).lower()
    }
    if not tasks:
        raise ValueError(f"No tasks found for size_bucket={size_bucket!r} in {task_metadata}")
    return tasks


def keep_state(run_state: str, states_arg: str) -> bool:
    if states_arg.lower() == "all":
        return True
    allowed = {s.strip() for s in states_arg.split(",") if s.strip()}
    return str(run_state) in allowed


def write_header(path: Path, columns: list[str], gzip_file: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_file:
        f = gzip.open(path, "wt", newline="", encoding="utf-8")
    else:
        f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    return f, writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--sweep-id", required=True)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--task-metadata", required=True, type=Path)
    p.add_argument("--size-bucket", choices=["small", "medium", "large"], default="large")
    p.add_argument("--states", default="finished")
    p.add_argument("--page-size", type=int, default=10000)
    p.add_argument("--print-every", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bucket_tasks = load_bucket_tasks(args.task_metadata, args.size_bucket)

    print(f"Keeping {len(bucket_tasks)} {args.size_bucket} tasks:")
    print(", ".join(sorted(bucket_tasks)))

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    raw_summary_f, raw_summary_writer = write_header(args.raw_dir / "runs_summary.csv", SUMMARY_COLUMNS, gzip_file=False)
    raw_history_f, raw_history_writer = write_header(args.raw_dir / "runs_history.csv.gz", HISTORY_COLUMNS, gzip_file=True)

    task_summary_files: dict[str, Any] = {}
    task_history_files: dict[str, Any] = {}
    task_summary_writers: dict[str, csv.DictWriter] = {}
    task_history_writers: dict[str, csv.DictWriter] = {}

    def get_task_writers(task: str):
        task = safe_token(task)
        if task not in task_summary_writers:
            task_dir = args.out_root / task
            sf, sw = write_header(task_dir / "runs_summary.csv", SUMMARY_COLUMNS, gzip_file=False)
            hf, hw = write_header(task_dir / "runs_history.csv.gz", HISTORY_COLUMNS, gzip_file=True)
            task_summary_files[task] = sf
            task_history_files[task] = hf
            task_summary_writers[task] = sw
            task_history_writers[task] = hw
        return task_summary_writers[task], task_history_writers[task]

    api = wandb.Api()
    runs = api.runs(path=f"{args.entity}/{args.project}", filters={"sweep": args.sweep_id})

    n_seen = 0
    n_kept = 0
    n_history_rows = 0
    by_task: dict[str, int] = {t: 0 for t in sorted(bucket_tasks)}

    try:
        for run in runs:
            n_seen += 1
            if not keep_state(run.state, args.states):
                continue

            row = base_row(run)
            task = str(row.get("matrix_task") or "")
            if task not in bucket_tasks:
                continue

            summary = dict(run.summary)

            # Summary row
            summary_row = dict(row)
            for k in SUMMARY_COLUMNS:
                if k not in summary_row:
                    summary_row[k] = summary.get(k)

            raw_summary_writer.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})
            task_summary_writer, task_history_writer = get_task_writers(task)
            task_summary_writer.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})

            # History rows
            run_history_rows = 0
            for h in run.scan_history(keys=HISTORY_KEYS, page_size=int(args.page_size)):
                hrow = dict(row)
                for k in HISTORY_KEYS:
                    hrow[k] = h.get(k)
                raw_history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
                task_history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
                n_history_rows += 1
                run_history_rows += 1

            n_kept += 1
            by_task[task] = by_task.get(task, 0) + 1

            raw_summary_f.flush()
            raw_history_f.flush()
            task_summary_files[safe_token(task)].flush()
            task_history_files[safe_token(task)].flush()

            if n_kept % int(args.print_every) == 0:
                print(f"Downloaded {n_kept} kept runs; W&B seen={n_seen}; history rows={n_history_rows}")

    finally:
        raw_summary_f.close()
        raw_history_f.close()
        for f in task_summary_files.values():
            f.close()
        for f in task_history_files.values():
            f.close()

    metadata = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "states": args.states,
        "size_bucket": args.size_bucket,
        "n_runs_seen": n_seen,
        "n_runs_kept": n_kept,
        "n_history_rows": n_history_rows,
        "tasks": sorted(bucket_tasks),
        "runs_by_task": by_task,
        "raw_dir": str(args.raw_dir),
        "out_root": str(args.out_root),
        "note": "No stopping artifacts or traces were downloaded.",
    }
    (args.raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
