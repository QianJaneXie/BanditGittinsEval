#!/usr/bin/env python3
"""Resume-safe downloader for MMLU-medium W&B sweep data with Gittins stopping artifacts.

Use this after `download_mmlu_medium_with_stopping_and_split.py` was interrupted by a W&B/network error.
It keeps completed runs from the existing RAW_DIR, downloads only missing/incomplete runs, then splits by task.

Place this file in the same `scripts/` folder as `download_mmlu_medium_with_stopping_and_split.py`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

# Reuse constants and helper functions from the original script.
import download_mmlu_medium_with_stopping_and_split as base


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read {path}: {e!r}", flush=True)
        return pd.DataFrame()


def _history_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.is_file():
        return counts
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get("run_id")
                if rid:
                    counts[str(rid)] += 1
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not fully read {path}: {e!r}", flush=True)
    return counts


def _completed_run_ids(raw_dir: Path) -> set[str]:
    summary = _read_csv_or_empty(raw_dir / "runs_summary.csv")
    if summary.empty or "run_id" not in summary.columns:
        return set()

    counts = _history_counts(raw_dir / "runs_history.csv.gz")
    completed: set[str] = set()

    summary = summary.drop_duplicates(subset=["run_id"], keep="last")
    for _, row in summary.iterrows():
        rid = str(row.get("run_id"))
        if not rid or rid == "nan":
            continue
        hist_n = int(counts.get(rid, 0))
        nb = pd.to_numeric(row.get("num_batches"), errors="coerce")
        if pd.notna(nb) and float(nb) > 0:
            if hist_n >= int(float(nb)):
                completed.add(rid)
        elif hist_n > 0:
            completed.add(rid)
    return completed


def _copy_completed_existing(raw_dir: Path, tmp_dir: Path, completed: set[str]) -> dict[str, int]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    copied_summary = copied_stopping = copied_history = 0

    old_summary = _read_csv_or_empty(raw_dir / "runs_summary.csv")
    old_stopping = _read_csv_or_empty(raw_dir / "runs_stopping.csv")

    summary_out = tmp_dir / "runs_summary.csv"
    stopping_out = tmp_dir / "runs_stopping.csv"
    history_out = tmp_dir / "runs_history.csv.gz"

    with summary_out.open("w", newline="", encoding="utf-8") as sf, stopping_out.open("w", newline="", encoding="utf-8") as stf:
        sw = csv.DictWriter(sf, fieldnames=base.SUMMARY_COLUMNS)
        tw = csv.DictWriter(stf, fieldnames=base.STOPPING_CSV_COLUMNS)
        sw.writeheader()
        tw.writeheader()

        if not old_summary.empty and "run_id" in old_summary.columns:
            old_summary = old_summary[old_summary["run_id"].astype(str).isin(completed)]
            old_summary = old_summary.drop_duplicates(subset=["run_id"], keep="last")
            for _, row in old_summary.iterrows():
                sw.writerow({k: row.get(k, None) for k in base.SUMMARY_COLUMNS})
                copied_summary += 1

        if not old_stopping.empty and "run_id" in old_stopping.columns:
            old_stopping = old_stopping[old_stopping["run_id"].astype(str).isin(completed)]
            old_stopping = old_stopping.drop_duplicates(subset=["run_id"], keep="last")
            for _, row in old_stopping.iterrows():
                tw.writerow({k: row.get(k, None) for k in base.STOPPING_CSV_COLUMNS})
                copied_stopping += 1

    with gzip.open(history_out, "wt", newline="", encoding="utf-8") as hf:
        hw = csv.DictWriter(hf, fieldnames=base.HISTORY_COLUMNS)
        hw.writeheader()
        old_history = raw_dir / "runs_history.csv.gz"
        if old_history.is_file():
            try:
                with gzip.open(old_history, "rt", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if str(row.get("run_id")) in completed:
                            hw.writerow({k: row.get(k) for k in base.HISTORY_COLUMNS})
                            copied_history += 1
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: could not fully copy old history: {e!r}", flush=True)

    return {"summary": copied_summary, "stopping": copied_stopping, "history": copied_history}


def _scan_history_with_retry(run: wandb.apis.public.Run, *, page_size: int, max_retries: int, sleep_s: float) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return list(run.scan_history(keys=base.HISTORY_KEYS, page_size=page_size))
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = sleep_s * attempt
            print(f"WARNING: scan_history failed for run {run.id} attempt {attempt}/{max_retries}: {e!r}. Sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"scan_history failed after {max_retries} attempts for run {run.id}: {last_err!r}")


def resume_download_raw(args: argparse.Namespace) -> dict[str, Any]:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.raw_dir / "_resume_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    completed = _completed_run_ids(args.raw_dir)
    copied = _copy_completed_existing(args.raw_dir, tmp_dir, completed)
    print(f"Resume mode: found {len(completed)} completed existing runs; copied {copied['history']} history rows.", flush=True)

    keep_states = None if args.states.lower() == "all" else {s.strip() for s in args.states.split(",") if s.strip()}
    api = wandb.Api()
    runs = api.runs(path=f"{args.entity}/{args.project}", filters={"sweep": args.sweep_id})

    summary_path = tmp_dir / "runs_summary.csv"
    history_path = tmp_dir / "runs_history.csv.gz"
    stopping_path = tmp_dir / "runs_stopping.csv"
    metadata_path = tmp_dir / "metadata.json"
    artifact_root = args.raw_dir / "artifacts"  # keep artifact cache in the final raw dir

    n_runs_seen = 0
    n_runs_kept = len(completed)
    n_history_rows = int(copied["history"])
    n_artifact_attempts = 0
    n_trace_found = 0
    n_stop_found = 0
    n_skipped_completed = 0
    failed_runs: list[dict[str, str]] = []

    # Count existing traces/stops in copied stopping rows.
    copied_stopping_df = _read_csv_or_empty(stopping_path)
    if not copied_stopping_df.empty:
        if "trace_local_path" in copied_stopping_df.columns:
            n_trace_found += int(copied_stopping_df["trace_local_path"].notna().sum())
        if "gittins_stop_found" in copied_stopping_df.columns:
            n_stop_found += int(copied_stopping_df["gittins_stop_found"].astype(str).str.lower().isin(["true", "1"]).sum())

    with summary_path.open("a", newline="", encoding="utf-8") as sf, stopping_path.open("a", newline="", encoding="utf-8") as stf, gzip.open(history_path, "at", newline="", encoding="utf-8") as hf:
        summary_writer = csv.DictWriter(sf, fieldnames=base.SUMMARY_COLUMNS)
        stopping_writer = csv.DictWriter(stf, fieldnames=base.STOPPING_CSV_COLUMNS)
        history_writer = csv.DictWriter(hf, fieldnames=base.HISTORY_COLUMNS)

        for run in runs:
            n_runs_seen += 1
            if keep_states is not None and run.state not in keep_states:
                continue
            if str(run.id) in completed:
                n_skipped_completed += 1
                if n_skipped_completed % args.print_every == 0:
                    print(f"Skipped {n_skipped_completed} already-completed runs...", flush=True)
                continue

            try:
                summary = dict(run.summary)
                row = base.base_row(run)

                stopping_info = {
                    "gittins_stop_cum_eval": base.get_field(dict(run.config), summary, "gittins_stop_cum_eval") or -1,
                    "gittins_stop_cum_original_cost": base.get_field(dict(run.config), summary, "gittins_stop_cum_original_cost") or -1.0,
                    "gittins_stop_found": False,
                    "trace_artifact_name": None,
                    "trace_artifact_path": None,
                    "trace_local_path": None,
                    "trace_extract_error": None,
                }
                try:
                    stopping_info["gittins_stop_found"] = int(stopping_info["gittins_stop_cum_eval"]) >= 0
                except Exception:
                    stopping_info["gittins_stop_found"] = False

                if args.download_artifacts:
                    if args.artifact_policy == "all" or str(row.get("policy_family")) == "gittins":
                        n_artifact_attempts += 1
                    stopping_info.update(base.maybe_download_trace_artifact(run, row, artifact_root, args.artifact_policy))

                history_rows = _scan_history_with_retry(run, page_size=args.page_size, max_retries=args.max_run_retries, sleep_s=args.retry_sleep)

                if stopping_info.get("trace_local_path"):
                    n_trace_found += 1
                if bool(stopping_info.get("gittins_stop_found")):
                    n_stop_found += 1

                summary_row = dict(row)
                for k in base.SUMMARY_COLUMNS:
                    if k in stopping_info:
                        summary_row[k] = stopping_info.get(k)
                    elif k not in summary_row:
                        summary_row[k] = summary.get(k)
                summary_writer.writerow({k: summary_row.get(k) for k in base.SUMMARY_COLUMNS})

                stopping_row = dict(row)
                stopping_row.update(stopping_info)
                stopping_writer.writerow({k: stopping_row.get(k) for k in base.STOPPING_CSV_COLUMNS})

                for h in history_rows:
                    hrow = dict(row)
                    for k in base.HISTORY_KEYS:
                        hrow[k] = h.get(k)
                    history_writer.writerow({k: hrow.get(k) for k in base.HISTORY_COLUMNS})
                    n_history_rows += 1

                n_runs_kept += 1
                if n_runs_kept % args.print_every == 0:
                    print(f"Downloaded/resumed {n_runs_kept} finished runs; history rows={n_history_rows}; traces={n_trace_found}; stops={n_stop_found}", flush=True)
            except Exception as e:  # noqa: BLE001
                failed_runs.append({"run_id": str(getattr(run, "id", "")), "run_name": str(getattr(run, "name", "")), "error": repr(e)})
                print(f"ERROR: failed run {getattr(run, 'id', '')} {getattr(run, 'name', '')}: {e!r}", flush=True)
                if not args.continue_on_failed_run:
                    raise

    metadata = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "states": args.states,
        "download_artifacts": bool(args.download_artifacts),
        "artifact_policy": args.artifact_policy,
        "resume_completed_existing_runs": len(completed),
        "n_runs_seen": n_runs_seen,
        "n_runs_kept": n_runs_kept,
        "n_history_rows": n_history_rows,
        "n_artifact_attempts": n_artifact_attempts,
        "n_trace_found": n_trace_found,
        "n_stop_found": n_stop_found,
        "n_skipped_completed": n_skipped_completed,
        "n_failed_runs": len(failed_runs),
        "failed_runs": failed_runs,
        "summary_path": str(args.raw_dir / "runs_summary.csv"),
        "history_path": str(args.raw_dir / "runs_history.csv.gz"),
        "stopping_path": str(args.raw_dir / "runs_stopping.csv"),
        "artifact_root": str(artifact_root) if args.download_artifacts else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Replace raw CSVs only after the temp download is complete.
    for name in ["runs_summary.csv", "runs_stopping.csv", "runs_history.csv.gz", "metadata.json"]:
        src = tmp_dir / name
        dst = args.raw_dir / name
        if dst.exists():
            backup = args.raw_dir / (name + ".bak")
            if backup.exists():
                backup.unlink()
            dst.replace(backup)
        src.replace(dst)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="EfficientLLMEval")
    p.add_argument("--project", default="GittinsBanditEval")
    p.add_argument("--sweep-id", default="e1gh776a")
    p.add_argument("--states", default="finished")
    p.add_argument("--raw-dir", type=Path, default=Path("outputs/wandb_downloads/mmlu_medium_raw_finished_with_stopping"))
    p.add_argument("--out-root", type=Path, default=Path("outputs/wandb_downloads/mmlu_medium_merged_finished_with_stopping"))
    p.add_argument("--task-metadata", type=Path, default=Path("data/MMLU_matrices/task_metadata.json"))
    p.add_argument("--page-size", type=int, default=5000)
    p.add_argument("--download-artifacts", action="store_true", default=True)
    p.add_argument("--artifact-policy", choices=["gittins", "all"], default="gittins")
    p.add_argument("--skip-download", action="store_true", help="Only redo the task split from existing raw CSVs.")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--max-run-retries", type=int, default=6)
    p.add_argument("--retry-sleep", type=float, default=10.0)
    p.add_argument("--continue-on-failed-run", action="store_true", help="Continue after a run fails even after retries; metadata records failed runs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    medium_tasks = base.medium_tasks_from_metadata(args.task_metadata)
    print(f"Medium tasks ({len(medium_tasks)}): {', '.join(medium_tasks)}", flush=True)
    if not args.skip_download:
        resume_download_raw(args)
    base.split_by_task(args, medium_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
