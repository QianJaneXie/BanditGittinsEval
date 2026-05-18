#!/usr/bin/env python3
"""Checkpointed MMLU-medium W&B downloader without stopping artifacts.

Why this script exists
----------------------
For dense MMLU task-panel figures, we no longer need Gittins stopping lines.
Therefore this downloader intentionally DOES NOT download W&B artifacts or
*_traces.npz files.  It downloads only:

  1. runs_summary.csv
  2. runs_history.csv.gz

It also writes per-task folders while downloading, so you do not need a second
large split pass over runs_history.csv.gz.

Checkpoint / resume behavior
----------------------------
On the first run, the script builds a fixed W&B run manifest:

  RAW_DIR/wandb_run_manifest.csv

After each run is fully downloaded and written, it appends one JSON line to:

  RAW_DIR/completed_runs.jsonl

and updates:

  RAW_DIR/download_checkpoint.json

If the process is interrupted, re-run the exact same command.  The script will
load the manifest and completed_runs.jsonl, then continue from the first
incomplete manifest row.  It does not scan the large runs_history.csv.gz file.

Recommended command
-------------------
python scripts/download_mmlu_medium_checkpointed_nostop.py \
  --entity EfficientLLMEval \
  --project GittinsBanditEval \
  --sweep-id e1gh776a \
  --raw-dir outputs/wandb_downloads/mmlu_medium_raw_finished_nostop_checkpointed \
  --out-root outputs/wandb_downloads/mmlu_medium_merged_finished_nostop_checkpointed \
  --task-metadata data/MMLU_matrices/task_metadata.json \
  --states finished
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
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


def get_field(cfg: dict[str, Any], summary: dict[str, Any], key: str) -> Any:
    if key in summary and summary[key] is not None:
        return summary[key]
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return None


def matrix_task_from_matrix(matrix_value: Any) -> str | None:
    if matrix_value is None:
        return None
    stem = Path(str(matrix_value)).stem
    return stem if stem else None


def base_row(run: wandb.apis.public.Run) -> dict[str, Any]:
    cfg = dict(run.config)
    summary = dict(run.summary)
    exp = cfg.get("experiment_variant") or cfg.get("raw") or cfg.get("policy_variant")
    fallback = parse_variant_fallback(exp)

    matrix = cfg.get("matrix") or summary.get("matrix")
    task = cfg.get("matrix_task") or cfg.get("task") or summary.get("matrix_task") or matrix_task_from_matrix(matrix)

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "dataset_tag_resolved": cfg.get("dataset_tag_resolved"),
        "dataset_tag": cfg.get("dataset_tag"),
        "matrix": matrix,
        "matrix_task": task,
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


def load_medium_tasks(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"task metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = {
        str(t["task"])
        for t in data.get("tasks", [])
        if str(t.get("size_bucket", "")).lower() == "medium"
    }
    if not tasks:
        raise ValueError(f"No medium tasks found in {path}")
    return tasks


def keep_state(run_state: str, states_arg: str) -> bool:
    if states_arg.lower() == "all":
        return True
    allowed = {s.strip() for s in states_arg.split(",") if s.strip()}
    return str(run_state) in allowed


def ensure_csv_header(path: Path, columns: list[str], *, gzip_file: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    if gzip_file:
        with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def build_manifest(args: argparse.Namespace, manifest_path: Path, medium_tasks: set[str] | None) -> list[dict[str, Any]]:
    print("Building W&B run manifest. This happens only on the first run or with --refresh-manifest.")
    api = wandb.Api()
    runs = api.runs(path=f"{args.entity}/{args.project}", filters={"sweep": args.sweep_id})

    rows: list[dict[str, Any]] = []
    n_seen = 0
    n_state = 0
    n_medium = 0

    for run in runs:
        n_seen += 1
        if not keep_state(run.state, args.states):
            continue
        n_state += 1

        row = base_row(run)
        task = row.get("matrix_task")
        if medium_tasks is not None and task not in medium_tasks:
            continue
        n_medium += 1

        rows.append({k: row.get(k) for k in BASE_COLUMNS})

        if len(rows) % int(args.print_every) == 0:
            print(f"  manifest rows kept={len(rows)}; W&B runs seen={n_seen}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Manifest written: {manifest_path}\n"
        f"  W&B runs seen={n_seen}; state-kept={n_state}; medium-kept={n_medium}"
    )
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def load_completed_runs(path: Path) -> tuple[set[str], dict[str, int]]:
    completed: set[str] = set()
    index_by_run: dict[str, int] = {}
    if not path.is_file():
        return completed, index_by_run

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            run_id = str(obj.get("run_id", ""))
            if not run_id:
                continue
            completed.add(run_id)
            try:
                index_by_run[run_id] = int(obj.get("manifest_index"))
            except Exception:
                pass
    return completed, index_by_run


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class DownloadWriters:
    """Keep raw and per-task CSV/GZ handles open while appending rows."""

    def __init__(self, raw_dir: Path, out_root: Path | None):
        self.raw_dir = raw_dir
        self.out_root = out_root

        self.raw_summary_path = raw_dir / "runs_summary.csv"
        self.raw_history_path = raw_dir / "runs_history.csv.gz"

        ensure_csv_header(self.raw_summary_path, SUMMARY_COLUMNS, gzip_file=False)
        ensure_csv_header(self.raw_history_path, HISTORY_COLUMNS, gzip_file=True)

        self.raw_summary_f = self.raw_summary_path.open("a", newline="", encoding="utf-8")
        self.raw_history_f = gzip.open(self.raw_history_path, "at", newline="", encoding="utf-8")
        self.raw_summary_writer = csv.DictWriter(self.raw_summary_f, fieldnames=SUMMARY_COLUMNS)
        self.raw_history_writer = csv.DictWriter(self.raw_history_f, fieldnames=HISTORY_COLUMNS)

        self.task_summary_handles: dict[str, Any] = {}
        self.task_history_handles: dict[str, Any] = {}
        self.task_summary_writers: dict[str, csv.DictWriter] = {}
        self.task_history_writers: dict[str, csv.DictWriter] = {}

    def _task_writers(self, task: str) -> tuple[csv.DictWriter, csv.DictWriter]:
        if self.out_root is None:
            return None, None  # type: ignore[return-value]

        task = safe_token(task)
        if task not in self.task_summary_writers:
            task_dir = self.out_root / task
            task_dir.mkdir(parents=True, exist_ok=True)
            summary_path = task_dir / "runs_summary.csv"
            history_path = task_dir / "runs_history.csv.gz"
            ensure_csv_header(summary_path, SUMMARY_COLUMNS, gzip_file=False)
            ensure_csv_header(history_path, HISTORY_COLUMNS, gzip_file=True)

            sf = summary_path.open("a", newline="", encoding="utf-8")
            hf = gzip.open(history_path, "at", newline="", encoding="utf-8")
            self.task_summary_handles[task] = sf
            self.task_history_handles[task] = hf
            self.task_summary_writers[task] = csv.DictWriter(sf, fieldnames=SUMMARY_COLUMNS)
            self.task_history_writers[task] = csv.DictWriter(hf, fieldnames=HISTORY_COLUMNS)

        return self.task_summary_writers[task], self.task_history_writers[task]

    def write_run(
        self,
        *,
        task: str,
        summary_row: dict[str, Any],
        history_rows: list[dict[str, Any]],
    ) -> None:
        self.raw_summary_writer.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})
        for hrow in history_rows:
            self.raw_history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})

        if self.out_root is not None:
            sw, hw = self._task_writers(task)
            sw.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})
            for hrow in history_rows:
                hw.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})

        self.flush()

    def flush(self) -> None:
        self.raw_summary_f.flush()
        self.raw_history_f.flush()
        for f in self.task_summary_handles.values():
            f.flush()
        for f in self.task_history_handles.values():
            f.flush()

    def close(self) -> None:
        self.raw_summary_f.close()
        self.raw_history_f.close()
        for f in self.task_summary_handles.values():
            f.close()
        for f in self.task_history_handles.values():
            f.close()


def scan_history_with_retry(
    run: wandb.apis.public.Run,
    *,
    page_size: int,
    retries: int,
    sleep_s: float,
) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            return [dict(h) for h in run.scan_history(keys=HISTORY_KEYS, page_size=page_size)]
        except Exception as e:  # W&B network/API failures are common in long downloads.
            last_err = e
            wait = float(sleep_s) * attempt
            print(
                f"  scan_history failed for run={run.id}, attempt={attempt}/{retries}: "
                f"{type(e).__name__}: {e}; sleeping {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"scan_history failed for run={run.id} after {retries} retries: {last_err!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--sweep-id", required=True)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--out-root", type=Path, default=None, help="Optional per-task output root written incrementally.")
    p.add_argument("--task-metadata", type=Path, default=None, help="JSON with MMLU size_bucket annotations; keeps size_bucket=medium.")
    p.add_argument("--states", default="finished", help="Comma-separated states to keep, default: finished.")
    p.add_argument("--refresh-manifest", action="store_true", help="Rebuild manifest from W&B, keeping existing completed checkpoint.")
    p.add_argument("--page-size", type=int, default=10000)
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--sleep", type=float, default=8.0)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--max-new-runs", type=int, default=None, help="Debug option: stop after appending this many new runs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    if args.out_root is not None:
        args.out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.raw_dir / "wandb_run_manifest.csv"
    completed_path = args.raw_dir / "completed_runs.jsonl"
    checkpoint_path = args.raw_dir / "download_checkpoint.json"
    metadata_path = args.raw_dir / "download_metadata.json"

    medium_tasks = load_medium_tasks(args.task_metadata)

    if args.refresh_manifest or not manifest_path.is_file():
        manifest_rows = build_manifest(args, manifest_path, medium_tasks)
    else:
        manifest_rows = load_manifest(manifest_path)
        print(f"Loaded existing manifest: {manifest_path} ({len(manifest_rows)} runs)")

    completed_ids, completed_index_by_run = load_completed_runs(completed_path)
    print(f"Loaded checkpoint: completed runs={len(completed_ids)}")

    # Find the first incomplete manifest row. This is fast because it reads only completed_runs.jsonl.
    first_incomplete = None
    for i, row in enumerate(manifest_rows):
        if str(row["run_id"]) not in completed_ids:
            first_incomplete = i
            break

    if first_incomplete is None:
        print("All manifest runs are already completed. Nothing to download.")
        return 0

    print(
        f"Resuming from manifest index {first_incomplete} / {len(manifest_rows)} "
        f"(run_id={manifest_rows[first_incomplete]['run_id']})"
    )

    api = wandb.Api()
    writers = DownloadWriters(args.raw_dir, args.out_root)

    counts_by_task: Counter[str] = Counter()
    history_rows_by_task: Counter[str] = Counter()
    n_appended = 0
    n_history_rows = 0

    try:
        with completed_path.open("a", encoding="utf-8") as completed_f:
            for i in range(first_incomplete, len(manifest_rows)):
                manifest_row = manifest_rows[i]
                run_id = str(manifest_row["run_id"])

                if run_id in completed_ids:
                    continue

                # Fetch the run directly by id using the fixed manifest order.
                run = api.run(f"{args.entity}/{args.project}/{run_id}")
                if not keep_state(run.state, args.states):
                    print(f"Skipping run changed state: index={i}, run_id={run_id}, state={run.state}")
                    continue

                row = base_row(run)
                task = str(row.get("matrix_task") or manifest_row.get("matrix_task") or "unknown_task")
                if medium_tasks is not None and task not in medium_tasks:
                    print(f"Skipping non-medium task from manifest: index={i}, run_id={run_id}, task={task}")
                    continue

                summary = dict(run.summary)

                # Important: collect history completely before writing anything for this run.
                hist = scan_history_with_retry(
                    run,
                    page_size=int(args.page_size),
                    retries=int(args.retries),
                    sleep_s=float(args.sleep),
                )

                summary_row = dict(row)
                for k in SUMMARY_COLUMNS:
                    if k not in summary_row:
                        summary_row[k] = summary.get(k)

                history_rows = []
                for h in hist:
                    hrow = dict(row)
                    for k in HISTORY_KEYS:
                        hrow[k] = h.get(k)
                    history_rows.append(hrow)

                writers.write_run(task=task, summary_row=summary_row, history_rows=history_rows)

                completed_record = {
                    "manifest_index": i,
                    "run_id": run_id,
                    "matrix_task": task,
                    "experiment_variant": row.get("experiment_variant"),
                    "history_rows": len(history_rows),
                    "completed_at_unix": time.time(),
                }
                completed_f.write(json.dumps(completed_record, ensure_ascii=False) + "\n")
                completed_f.flush()

                completed_ids.add(run_id)
                n_appended += 1
                n_history_rows += len(history_rows)
                counts_by_task[task] += 1
                history_rows_by_task[task] += len(history_rows)

                checkpoint = {
                    "entity": args.entity,
                    "project": args.project,
                    "sweep_id": args.sweep_id,
                    "manifest_path": str(manifest_path),
                    "completed_path": str(completed_path),
                    "last_completed_manifest_index": i,
                    "last_completed_run_id": run_id,
                    "total_manifest_runs": len(manifest_rows),
                    "total_completed_runs": len(completed_ids),
                    "new_runs_this_process": n_appended,
                    "new_history_rows_this_process": n_history_rows,
                    "raw_summary_path": str(args.raw_dir / "runs_summary.csv"),
                    "raw_history_path": str(args.raw_dir / "runs_history.csv.gz"),
                    "out_root": None if args.out_root is None else str(args.out_root),
                }
                write_checkpoint(checkpoint_path, checkpoint)

                if n_appended % int(args.print_every) == 0:
                    print(
                        f"Appended {n_appended} new runs; "
                        f"new history rows={n_history_rows}; "
                        f"checkpoint index={i}/{len(manifest_rows)}; "
                        f"completed total={len(completed_ids)}",
                        flush=True,
                    )

                if args.max_new_runs is not None and n_appended >= int(args.max_new_runs):
                    print(f"Stopping early because --max-new-runs={args.max_new_runs}")
                    break

    finally:
        writers.close()

    metadata = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "raw_dir": str(args.raw_dir),
        "out_root": None if args.out_root is None else str(args.out_root),
        "states": args.states,
        "task_metadata": None if args.task_metadata is None else str(args.task_metadata),
        "manifest_path": str(manifest_path),
        "completed_path": str(completed_path),
        "checkpoint_path": str(checkpoint_path),
        "manifest_runs": len(manifest_rows),
        "completed_runs_after_process": len(completed_ids),
        "new_runs_this_process": n_appended,
        "new_history_rows_this_process": n_history_rows,
        "new_runs_by_task": dict(counts_by_task),
        "new_history_rows_by_task": dict(history_rows_by_task),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("Done. Re-run the same command to resume if the download was interrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
