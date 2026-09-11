#!/usr/bin/env python3
"""Checkpointed W&B downloader for simple-regret sweeps.

Downloads run summary + history only. It does NOT download W&B artifacts or
*_traces.npz trace files.

Features
--------
1. Supports GSM8K, PIQA, and MMLU.
2. For MMLU, supports small / medium / large / all size buckets.
3. Supports checkpoint/resume through:
   - RAW_DIR/wandb_run_manifest.csv
   - RAW_DIR/completed_runs.jsonl
   - RAW_DIR/download_checkpoint.json
4. Deduplicates W&B API duplicate run_id rows while building the manifest.
5. Optional expected-grid check + targeted recovery:
   - build/download one deduplicated sweep manifest once;
   - compare downloaded summary against expected configs;
   - for missing configs, query ONLY inside the selected sweep;
   - append any found missing runs to the same output.

Recommended GSM8K full download + one targeted recovery pass:
  python scripts/download_wandb_simple_regret.py \
    --entity EfficientLLMEval \
    --project GittinsBanditEval \
    --sweep-id nmsy4qou \
    --dataset gsm8k \
    --raw-dir outputs/wandb_downloads_new/gsm8k \
    --states finished \
    --expected-grid-yaml scripts/config/GSM8KSimpleRegretSweep.yml \
    --recover-missing-targeted

For MMLU small, split by task:
  python scripts/download_wandb_simple_regret.py \
    --entity EfficientLLMEval \
    --project GittinsBanditEval \
    --sweep-id e04jhnd9 \
    --dataset mmlu \
    --mmlu-size-bucket small \
    --task-metadata data/MMLU_matrices/task_metadata.json \
    --raw-dir outputs/wandb_downloads_new/mmlu_small \
    --split-mmlu-by-task \
    --out-root outputs/wandb_downloads_new/mmlu_small_by_task \
    --states finished \
    --expected-grid-yaml scripts/config/MMLUSimpleRegretPilot_small.yml \
    --recover-missing-targeted
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
from typing import Any, Iterable

import wandb


HISTORY_KEYS = [
    "cum_eval",
    "cum_original_cost",
    "simple_regret",
    "recommended_arm",
    "recommended_mean",
    "pulled_arm",
    "pulled_arms_json",
    "pulled_rows_json",
    "pulled_cols_json",
    "n_pulled_arms",
    "n_pulled_cols",
    "iter_step_s",
    "iter_total_s",
    "batch_cells",
    "step_idx",
    "gittins_index_pulled",
    "posterior_mean_pulled",
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
    "mmlu_task",
    "mmlu_size_bucket",
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
    "prior_bucket",
    "prior_source",
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
    "total_wall_time_s",
    "git_commit",
    "git_dirty",
    "matrix_sha256",
    "cost_vector_sha256",
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "hostname",
    "python_version",
    "python_executable",
    "gittins_stop_cum_eval",
    "gittins_stop_cum_original_cost",
    "gittins_recommendation_aware_stop_cum_eval",
    "gittins_recommendation_aware_stop_cum_original_cost",
    "lookup_table_s",
    "lookup_memory_rss_before_mb",
    "lookup_memory_rss_after_mb",
    "lookup_memory_rss_delta_mb",
    "lookup_memory_peak_before_mb",
    "lookup_memory_peak_after_mb",
    "lookup_memory_peak_delta_mb",
    "peak_rss_gb",
    "extra_peak_memory_gb",
    "lookup_roots_table_mb",
    "iter_step_mean_s",
    "iter_step_median_s",
    "iter_step_p90_s",
    "iter_total_mean_s",
    "iter_total_median_s",
    "iter_total_p90_s",
]

HISTORY_COLUMNS = BASE_COLUMNS + HISTORY_KEYS

CONFIG_KEY_COLUMNS = [
    "dataset_tag",
    "matrix",
    "matrix_seed",
    "run_seed",
    "experiment_variant",
]

FULL_CONFIG_DIAGNOSTIC_COLUMNS = [
    "dataset_tag",
    "matrix",
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
    "eval_budget_fraction",
]


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "unknown"


def normalize_dataset_name(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s or s in {"none", "nan"}:
        return None
    if "gsm8k" in s:
        return "gsm8k"
    if "piqa" in s:
        return "piqa"
    if "mmlu" in s:
        return "mmlu"
    return s


def normalize_scalar(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"none", "nan"}:
        return ""
    if re.fullmatch(r"-?\d+\.0", s):
        return s[:-2]
    return s


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
    m = re.fullmatch(r"(ucb|lrf)_cost_B(\d+)", v)
    if m:
        policy = m.group(1)
        b = int(m.group(2))
        out.update(
            policy_variant=f"{policy}_cost",
            policy_family=policy,
            cost_mode="cost",
            batch_size=b,
            gittins_batch_size=b,
            cost_scaling_factor=1e-4,
            prior_type="default",
        )
        return out
    if v in {"sysrs", "sysrs_cost", "sysrs_aware"}:
        cost_mode = "cost" if v != "sysrs" else "baseline"
        out.update(
            policy_variant="sysrs_cost" if cost_mode == "cost" else "sysrs",
            policy_family="sysrs",
            cost_mode=cost_mode if cost_mode == "cost" else "baseline",
            batch_size=0,
            gittins_batch_size=0,
            cost_scaling_factor=1e-4,
            prior_type="default",
        )
        return out
    m = re.fullmatch(r"gittins_(unit|cost|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)", v)
    if m:
        cost_mode = m.group(1)
        if cost_mode == "aware":
            cost_mode = "cost"
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


def infer_matrix_seed(matrix_value: Any) -> str:
    m = re.search(r"seed(\d+)", str(matrix_value))
    return m.group(1) if m else ""


def infer_dataset_from_row(row: dict[str, Any]) -> str | None:
    for key in ("dataset_tag_resolved", "dataset_tag"):
        ds = normalize_dataset_name(row.get(key))
        if ds:
            return ds
    matrix = str(row.get("matrix") or "").lower()
    if "gsm8k" in matrix:
        return "gsm8k"
    if "piqa" in matrix:
        return "piqa"
    if "mmlu" in matrix:
        return "mmlu"
    if row.get("mmlu_task") or row.get("mmlu_size_bucket"):
        return "mmlu"
    return None


def base_row(run: wandb.apis.public.Run) -> dict[str, Any]:
    cfg = dict(run.config)
    summary = dict(run.summary)
    exp = cfg.get("experiment_variant") or cfg.get("raw") or cfg.get("policy_variant")
    fallback = parse_variant_fallback(exp)

    matrix = get_field(cfg, summary, "matrix")
    matrix_task = get_field(cfg, summary, "matrix_task") or get_field(cfg, summary, "task") or matrix_task_from_matrix(matrix)
    mmlu_task = get_field(cfg, summary, "mmlu_task") or matrix_task

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "dataset_tag_resolved": get_field(cfg, summary, "dataset_tag_resolved"),
        "dataset_tag": get_field(cfg, summary, "dataset_tag"),
        "matrix": matrix,
        "matrix_task": matrix_task,
        "mmlu_task": mmlu_task,
        "mmlu_size_bucket": get_field(cfg, summary, "mmlu_size_bucket"),
        "matrix_seed": get_field(cfg, summary, "matrix_seed") or infer_matrix_seed(matrix),
        "run_seed": get_field(cfg, summary, "run_seed"),
        "experiment_variant": exp,
        "policy_variant": get_field(cfg, summary, "policy_variant") or fallback.get("policy_variant"),
        "policy_family": get_field(cfg, summary, "policy_family") or fallback.get("policy_family"),
        "cost_mode": get_field(cfg, summary, "cost_mode") or fallback.get("cost_mode"),
        "batch_size": get_field(cfg, summary, "batch_size") or fallback.get("batch_size"),
        "gittins_batch_size": get_field(cfg, summary, "gittins_batch_size") or fallback.get("gittins_batch_size"),
        "cost_scaling_factor": get_field(cfg, summary, "cost_scaling_factor") or fallback.get("cost_scaling_factor"),
        "prior_type": get_field(cfg, summary, "prior_type") or fallback.get("prior_type"),
        "prior_mean_resolved": get_field(cfg, summary, "prior_mean_resolved"),
        "prior_variance_resolved": get_field(cfg, summary, "prior_variance_resolved"),
        "prior_bucket": get_field(cfg, summary, "prior_bucket"),
        "prior_source": get_field(cfg, summary, "prior_source"),
        "n_arms": get_field(cfg, summary, "n_arms"),
        "n_examples": get_field(cfg, summary, "n_examples"),
        "n_cells": get_field(cfg, summary, "n_cells"),
        "budget_max_evals": get_field(cfg, summary, "budget_max_evals"),
        "eval_budget_fraction": get_field(cfg, summary, "eval_budget_fraction"),
        "warmup_percentage": get_field(cfg, summary, "warmup_percentage"),
    }


def load_mmlu_task_buckets(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"task metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for t in data.get("tasks", []):
        task = str(t.get("task", "")).strip()
        bucket = str(t.get("size_bucket", "")).strip().lower()
        if task and bucket:
            out[task] = bucket
    if not out:
        raise ValueError(f"No task size-bucket records found in {path}")
    return out


def get_mmlu_task(row: dict[str, Any]) -> str | None:
    for key in ("mmlu_task", "matrix_task"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return matrix_task_from_matrix(row.get("matrix"))


def infer_mmlu_size_bucket(row: dict[str, Any], task_buckets: dict[str, str]) -> str | None:
    value = row.get("mmlu_size_bucket")
    if value is not None and str(value).strip():
        return str(value).strip().lower()
    task = get_mmlu_task(row)
    if task is not None and task in task_buckets:
        return task_buckets[task]
    return None


def keep_state(run_state: str, states_arg: str) -> bool:
    if states_arg.lower() == "all":
        return True
    allowed = {s.strip() for s in states_arg.split(",") if s.strip()}
    return str(run_state) in allowed


def keep_dataset(row: dict[str, Any], dataset_arg: str) -> bool:
    dataset_arg = dataset_arg.lower()
    if dataset_arg == "auto":
        return True
    return infer_dataset_from_row(row) == dataset_arg


def keep_mmlu_size(row: dict[str, Any], size_arg: str, task_buckets: dict[str, str]) -> bool:
    size_arg = size_arg.lower()
    if size_arg == "all":
        return True
    return infer_mmlu_size_bucket(row, task_buckets) == size_arg


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


def write_csv_rows(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def config_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        normalize_scalar(row.get("matrix")),
        normalize_scalar(row.get("matrix_seed") or infer_matrix_seed(row.get("matrix"))),
        normalize_scalar(row.get("run_seed")),
        normalize_scalar(row.get("experiment_variant")),
    )


def full_diagnostic_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(normalize_scalar(row.get(c)) for c in FULL_CONFIG_DIAGNOSTIC_COLUMNS)


def write_manifest_diagnostics(raw_dir: Path, candidate_rows: list[dict[str, Any]], unique_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_counts = Counter(str(r.get("run_id", "")) for r in candidate_rows)
    duplicate_run_ids = {rid: count for rid, count in run_counts.items() if rid and count > 1}
    write_csv_rows(
        raw_dir / "duplicate_run_ids_in_manifest.csv",
        ["run_id", "count"],
        [{"run_id": rid, "count": count} for rid, count in sorted(duplicate_run_ids.items(), key=lambda x: (-x[1], x[0]))],
    )

    detail_columns = ["duplicate_occurrence_index"] + BASE_COLUMNS
    detail_rows: list[dict[str, Any]] = []
    occurrence_by_run: Counter[str] = Counter()
    for row in candidate_rows:
        rid = str(row.get("run_id", ""))
        if rid not in duplicate_run_ids:
            continue
        occurrence_by_run[rid] += 1
        out = dict(row)
        out["duplicate_occurrence_index"] = occurrence_by_run[rid]
        detail_rows.append(out)
    write_csv_rows(raw_dir / "duplicate_manifest_rows_detail.csv", detail_columns, detail_rows)

    key_to_rows: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        key_to_rows[full_diagnostic_key(row)].append(row)

    dup_config_columns = FULL_CONFIG_DIAGNOSTIC_COLUMNS + ["count", "run_ids", "run_names", "urls"]
    dup_config_rows: list[dict[str, Any]] = []
    for _key, rows in key_to_rows.items():
        unique_run_ids = sorted({str(r.get("run_id", "")) for r in rows if r.get("run_id")})
        if len(unique_run_ids) <= 1:
            continue
        first = rows[0]
        out = {c: first.get(c) for c in FULL_CONFIG_DIAGNOSTIC_COLUMNS}
        out.update(
            count=len(unique_run_ids),
            run_ids=",".join(unique_run_ids),
            run_names=",".join(str(r.get("run_name", "")) for r in rows),
            urls=",".join(str(r.get("url", "")) for r in rows),
        )
        dup_config_rows.append(out)
    write_csv_rows(raw_dir / "duplicate_config_keys_in_manifest.csv", dup_config_columns, dup_config_rows)

    return {
        "candidate_rows_before_run_id_dedupe": len(candidate_rows),
        "unique_run_ids_after_dedupe": len(unique_rows),
        "duplicate_run_id_count": len(duplicate_run_ids),
        "duplicate_run_id_extra_rows": len(candidate_rows) - len(unique_rows),
        "duplicate_config_key_count": len(dup_config_rows),
        "duplicate_run_ids_path": str(raw_dir / "duplicate_run_ids_in_manifest.csv"),
        "duplicate_manifest_rows_detail_path": str(raw_dir / "duplicate_manifest_rows_detail.csv"),
        "duplicate_config_keys_path": str(raw_dir / "duplicate_config_keys_in_manifest.csv"),
    }


def build_manifest(args: argparse.Namespace, manifest_path: Path, task_buckets: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("Building W&B run manifest. This happens only on the first run or with --refresh-manifest.")
    api = wandb.Api()
    filters: dict[str, Any] = {"sweep": args.sweep_id}
    if args.variant:
        filters["config.experiment_variant"] = {"$in": list(args.variant)}
    if args.matrix:
        filters["config.matrix"] = {"$in": list(args.matrix)}
    runs = api.runs(path=f"{args.entity}/{args.project}", filters=filters)

    candidate_rows: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    n_seen = n_state = n_dataset = n_size = n_duplicate_api_rows = 0

    for run in runs:
        n_seen += 1
        if not keep_state(run.state, args.states):
            continue
        n_state += 1
        row = base_row(run)
        if not keep_dataset(row, args.dataset):
            continue
        n_dataset += 1
        if args.dataset.lower() == "mmlu" or infer_dataset_from_row(row) == "mmlu":
            if not keep_mmlu_size(row, args.mmlu_size_bucket, task_buckets):
                continue
        n_size += 1
        if not row.get("mmlu_size_bucket"):
            bucket = infer_mmlu_size_bucket(row, task_buckets)
            if bucket:
                row["mmlu_size_bucket"] = bucket

        row_for_manifest = {k: row.get(k) for k in BASE_COLUMNS}
        candidate_rows.append(row_for_manifest)
        run_id = str(row_for_manifest.get("run_id", ""))
        if run_id in seen_run_ids:
            n_duplicate_api_rows += 1
            continue
        seen_run_ids.add(run_id)
        unique_rows.append(row_for_manifest)
        if len(unique_rows) % int(args.print_every) == 0:
            print(
                f"  manifest unique rows kept={len(unique_rows)}; "
                f"raw kept rows={len(candidate_rows)}; W&B runs seen={n_seen}; "
                f"duplicate API rows skipped={n_duplicate_api_rows}",
                flush=True,
            )

    diagnostics = write_manifest_diagnostics(manifest_path.parent, candidate_rows, unique_rows)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(
        f"Manifest written: {manifest_path}\n"
        f"  W&B rows seen={n_seen}; state-kept={n_state}; "
        f"dataset-kept={n_dataset}; size-kept={n_size}; "
        f"raw-kept-before-dedupe={len(candidate_rows)}; "
        f"manifest-unique-kept={len(unique_rows)}; "
        f"duplicate-run-id-extra-rows={n_duplicate_api_rows}"
    )
    if diagnostics["duplicate_run_id_count"]:
        print(
            "  Warning: W&B API returned duplicate run_id rows. "
            f"Deduped manifest by run_id. Details: {diagnostics['duplicate_manifest_rows_detail_path']}"
        )
    if diagnostics["duplicate_config_key_count"]:
        print(
            "  Warning: distinct run_ids share the same config key. "
            f"Details: {diagnostics['duplicate_config_keys_path']}"
        )
    diagnostics.update(wandb_rows_seen=n_seen, state_kept=n_state, dataset_kept=n_dataset, size_kept=n_size)
    return unique_rows, diagnostics


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return read_csv_rows(path)


def dedupe_loaded_manifest(rows: list[dict[str, Any]], raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row.get("run_id", ""))
        if rid in seen:
            continue
        seen.add(rid)
        unique_rows.append(row)
    diagnostics = write_manifest_diagnostics(raw_dir, rows, unique_rows)
    if len(unique_rows) != len(rows):
        print(f"Loaded manifest contained duplicate run_id rows: raw rows={len(rows)}, unique rows={len(unique_rows)}. Using unique rows in memory.")
    return unique_rows, diagnostics


def load_completed_runs(path: Path) -> tuple[set[str], dict[str, str]]:
    completed: set[str] = set()
    index_by_run: dict[str, str] = {}
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
            if obj.get("manifest_index") is not None:
                index_by_run[run_id] = str(obj.get("manifest_index"))
    return completed, index_by_run


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class DownloadWriters:
    def __init__(self, raw_dir: Path, out_root: Path | None, *, no_history: bool, split_mmlu_by_task: bool):
        self.raw_dir = raw_dir
        self.out_root = out_root
        self.no_history = bool(no_history)
        self.split_mmlu_by_task = bool(split_mmlu_by_task)
        self.raw_summary_path = raw_dir / "runs_summary.csv"
        self.raw_history_path = raw_dir / "runs_history.csv.gz"
        ensure_csv_header(self.raw_summary_path, SUMMARY_COLUMNS, gzip_file=False)
        if not self.no_history:
            ensure_csv_header(self.raw_history_path, HISTORY_COLUMNS, gzip_file=True)
        self.raw_summary_f = self.raw_summary_path.open("a", newline="", encoding="utf-8")
        self.raw_summary_writer = csv.DictWriter(self.raw_summary_f, fieldnames=SUMMARY_COLUMNS)
        self.raw_history_f = None
        self.raw_history_writer = None
        if not self.no_history:
            self.raw_history_f = gzip.open(self.raw_history_path, "at", newline="", encoding="utf-8")
            self.raw_history_writer = csv.DictWriter(self.raw_history_f, fieldnames=HISTORY_COLUMNS)
        self.task_summary_handles: dict[str, Any] = {}
        self.task_history_handles: dict[str, Any] = {}
        self.task_summary_writers: dict[str, csv.DictWriter] = {}
        self.task_history_writers: dict[str, csv.DictWriter] = {}

    def _task_writers(self, task: str) -> tuple[csv.DictWriter | None, csv.DictWriter | None]:
        if not self.split_mmlu_by_task or self.out_root is None:
            return None, None
        task = safe_token(task)
        if task not in self.task_summary_writers:
            task_dir = self.out_root / task
            task_dir.mkdir(parents=True, exist_ok=True)
            summary_path = task_dir / "runs_summary.csv"
            history_path = task_dir / "runs_history.csv.gz"
            ensure_csv_header(summary_path, SUMMARY_COLUMNS, gzip_file=False)
            if not self.no_history:
                ensure_csv_header(history_path, HISTORY_COLUMNS, gzip_file=True)
            sf = summary_path.open("a", newline="", encoding="utf-8")
            self.task_summary_handles[task] = sf
            self.task_summary_writers[task] = csv.DictWriter(sf, fieldnames=SUMMARY_COLUMNS)
            if not self.no_history:
                hf = gzip.open(history_path, "at", newline="", encoding="utf-8")
                self.task_history_handles[task] = hf
                self.task_history_writers[task] = csv.DictWriter(hf, fieldnames=HISTORY_COLUMNS)
        return self.task_summary_writers[task], self.task_history_writers.get(task)

    def write_run(self, *, task: str | None, summary_row: dict[str, Any], history_rows: list[dict[str, Any]]) -> None:
        self.raw_summary_writer.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})
        if not self.no_history and self.raw_history_writer is not None:
            for hrow in history_rows:
                self.raw_history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
        if self.split_mmlu_by_task and self.out_root is not None and task:
            sw, hw = self._task_writers(task)
            if sw is not None:
                sw.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})
            if not self.no_history and hw is not None:
                for hrow in history_rows:
                    hw.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
        self.flush()

    def flush(self) -> None:
        self.raw_summary_f.flush()
        if self.raw_history_f is not None:
            self.raw_history_f.flush()
        for f in self.task_summary_handles.values():
            f.flush()
        for f in self.task_history_handles.values():
            f.flush()

    def close(self) -> None:
        self.raw_summary_f.close()
        if self.raw_history_f is not None:
            self.raw_history_f.close()
        for f in self.task_summary_handles.values():
            f.close()
        for f in self.task_history_handles.values():
            f.close()


def scan_history_with_retry(run: wandb.apis.public.Run, *, page_size: int, retries: int, sleep_s: float) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            return [dict(h) for h in run.scan_history(page_size=page_size)]
        except Exception as e:
            last_err = e
            wait = float(sleep_s) * attempt
            print(f"  scan_history failed for run={run.id}, attempt={attempt}/{retries}: {type(e).__name__}: {e}; sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"scan_history failed for run={run.id} after {retries} retries: {last_err!r}")


def summary_existing_keys(summary_path: Path) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for row in read_csv_rows(summary_path):
        keys.add(config_key(row))
    return keys


def load_expected_rows_from_csv(path: Path, dataset: str) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        matrix = row.get("matrix") or ""
        matrix_seed = row.get("matrix_seed") or infer_matrix_seed(matrix)
        out.append(
            {
                "dataset_tag": dataset if dataset != "auto" else normalize_dataset_name(row.get("dataset_tag")) or "",
                "matrix": matrix,
                "matrix_seed": normalize_scalar(matrix_seed),
                "run_seed": normalize_scalar(row.get("run_seed")),
                "experiment_variant": normalize_scalar(row.get("experiment_variant")),
            }
        )
    return out


def _yaml_param_values(params: dict[str, Any], name: str) -> list[Any]:
    if name not in params:
        return []
    obj = params[name]
    if isinstance(obj, dict):
        if "values" in obj:
            return list(obj["values"])
        if "value" in obj:
            return [obj["value"]]
    return []


def load_expected_rows_from_sweep_yaml(path: Path, dataset: str) -> list[dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for --expected-grid-yaml. Run: pip install pyyaml") from e
    if not path.is_file():
        raise FileNotFoundError(f"expected grid YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = data.get("parameters", {}) if isinstance(data, dict) else {}
    matrices = _yaml_param_values(params, "matrix")
    run_seeds = _yaml_param_values(params, "run_seed") or _yaml_param_values(params, "seed")
    variants = _yaml_param_values(params, "experiment_variant") or _yaml_param_values(params, "policy_variant")
    if not matrices or not run_seeds or not variants:
        raise ValueError("Expected YAML must contain parameters.matrix.values, parameters.run_seed.values, and parameters.experiment_variant.values")
    rows: list[dict[str, Any]] = []
    for matrix in matrices:
        matrix = str(matrix)
        matrix_seed = infer_matrix_seed(matrix)
        for run_seed in run_seeds:
            for variant in variants:
                rows.append(
                    {
                        "dataset_tag": dataset,
                        "matrix": matrix,
                        "matrix_seed": matrix_seed,
                        "run_seed": normalize_scalar(run_seed),
                        "experiment_variant": str(variant),
                    }
                )
    return rows


def expected_run_name(dataset: str, row: dict[str, Any]) -> str:
    ds = normalize_dataset_name(dataset) or normalize_dataset_name(row.get("dataset_tag")) or str(dataset)
    seed = normalize_scalar(row.get("matrix_seed") or infer_matrix_seed(row.get("matrix")))
    variant = normalize_scalar(row.get("experiment_variant"))
    run_seed = normalize_scalar(row.get("run_seed"))
    return f"{ds}_seed{seed}_{variant}_runseed{run_seed}"


def load_expected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.expected_configs_csv is not None:
        return load_expected_rows_from_csv(args.expected_configs_csv, args.dataset)
    if args.expected_grid_yaml is not None:
        return load_expected_rows_from_sweep_yaml(args.expected_grid_yaml, args.dataset)
    return []


def write_missing_expected_configs(args: argparse.Namespace, expected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = summary_existing_keys(args.raw_dir / "runs_summary.csv")
    missing: list[dict[str, Any]] = []
    seen_expected: set[tuple[str, ...]] = set()
    for row in expected_rows:
        key = config_key(row)
        if key in seen_expected:
            continue
        seen_expected.add(key)
        if key in existing:
            continue
        out = dict(row)
        out["expected_name"] = expected_run_name(args.dataset, row)
        missing.append(out)
    write_csv_rows(
        args.raw_dir / "missing_expected_configs.csv",
        ["dataset_tag", "matrix", "matrix_seed", "run_seed", "experiment_variant", "expected_name"],
        missing,
    )
    print(
        f"Completeness check: expected unique configs={len(seen_expected)}, "
        f"downloaded config keys={len(existing)}, missing configs={len(missing)}"
    )
    if missing:
        print(f"  Missing config list written: {args.raw_dir / 'missing_expected_configs.csv'}")
    return missing


def _as_int_if_possible(x: Any) -> Any:
    s = normalize_scalar(x)
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return s


def _query_runs(api: wandb.Api, entity: str, project: str, filters: dict[str, Any]) -> list[wandb.apis.public.Run]:
    return list(api.runs(path=f"{entity}/{project}", filters=filters))


def find_target_runs_within_sweep(args: argparse.Namespace, missing_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find missing configs with targeted queries restricted to this sweep only."""
    api = wandb.Api()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    match_detail: list[dict[str, Any]] = []
    still_missing: list[dict[str, Any]] = []
    multiple: list[dict[str, Any]] = []

    for idx, target in enumerate(missing_rows):
        matrix = normalize_scalar(target.get("matrix"))
        run_seed = _as_int_if_possible(target.get("run_seed"))
        variant = normalize_scalar(target.get("experiment_variant"))
        expected_name = target.get("expected_name") or expected_run_name(args.dataset, target)

        candidates_by_id: dict[str, wandb.apis.public.Run] = {}
        query_errors: list[str] = []

        if args.target_match_by in {"config", "both"}:
            filters = {
                "sweep": args.sweep_id,
                "config.matrix": matrix,
                "config.run_seed": run_seed,
                "config.experiment_variant": variant,
            }
            try:
                for run in _query_runs(api, args.entity, args.project, filters):
                    candidates_by_id[run.id] = run
            except Exception as e:
                query_errors.append(f"config query failed: {type(e).__name__}: {e}")

        if args.target_match_by in {"name", "both"}:
            # Keep these restricted to the same sweep. W&B deployments have varied
            # accepted name fields, so try both common spellings and tolerate failures.
            for name_key in ("display_name", "name"):
                filters = {"sweep": args.sweep_id, name_key: expected_name}
                try:
                    for run in _query_runs(api, args.entity, args.project, filters):
                        candidates_by_id[run.id] = run
                except Exception as e:
                    query_errors.append(f"{name_key} query failed: {type(e).__name__}: {e}")

        # State/dataset/size validation, still only for candidates from this sweep.
        valid: list[tuple[wandb.apis.public.Run, dict[str, Any]]] = []
        for run in candidates_by_id.values():
            if not keep_state(run.state, args.states):
                continue
            row = base_row(run)
            if not keep_dataset(row, args.dataset):
                continue
            valid.append((run, row))

        if not valid:
            row_out = dict(target)
            row_out.update(query_errors=" | ".join(query_errors), matched_run_ids="", matched_states="")
            still_missing.append(row_out)
            match_detail.append({**row_out, "target_index": idx, "match_status": "missing"})
            continue

        if len(valid) > 1:
            multiple.append(
                {
                    **target,
                    "matched_run_ids": ",".join(run.id for run, _ in valid),
                    "matched_names": ",".join(str(run.name) for run, _ in valid),
                    "matched_urls": ",".join(str(run.url) for run, _ in valid),
                }
            )

        # Choose one deterministic candidate for this config. Prefer exact run name;
        # otherwise use the first W&B API candidate. Do not duplicate already chosen run_ids.
        chosen_run, chosen_row = valid[0]
        for run, row in valid:
            if str(run.name).strip() == str(expected_name).strip():
                chosen_run, chosen_row = run, row
                break

        match_detail.append(
            {
                **target,
                "target_index": idx,
                "match_status": "matched",
                "matched_run_id": chosen_run.id,
                "matched_run_name": chosen_run.name,
                "matched_state": chosen_run.state,
                "matched_url": chosen_run.url,
                "query_errors": " | ".join(query_errors),
            }
        )
        if chosen_run.id not in selected_ids:
            selected_ids.add(chosen_run.id)
            selected.append({k: chosen_row.get(k) for k in BASE_COLUMNS})

    write_csv_rows(
        args.raw_dir / "target_match_detail.csv",
        [
            "target_index", "match_status", "dataset_tag", "matrix", "matrix_seed", "run_seed",
            "experiment_variant", "expected_name", "matched_run_id", "matched_run_name",
            "matched_state", "matched_url", "query_errors", "matched_run_ids", "matched_states",
        ],
        match_detail,
    )
    write_csv_rows(
        args.raw_dir / "target_missing_configs.csv",
        ["dataset_tag", "matrix", "matrix_seed", "run_seed", "experiment_variant", "expected_name", "query_errors", "matched_run_ids", "matched_states"],
        still_missing,
    )
    write_csv_rows(
        args.raw_dir / "target_multiple_matches.csv",
        ["dataset_tag", "matrix", "matrix_seed", "run_seed", "experiment_variant", "expected_name", "matched_run_ids", "matched_names", "matched_urls"],
        multiple,
    )

    diag = {
        "target_configs": len(missing_rows),
        "matched_targets": len(match_detail) - len(still_missing),
        "still_missing_targets": len(still_missing),
        "multiple_match_targets": len(multiple),
        "unique_runs_selected": len(selected),
        "target_match_detail_path": str(args.raw_dir / "target_match_detail.csv"),
        "target_missing_configs_path": str(args.raw_dir / "target_missing_configs.csv"),
        "target_multiple_matches_path": str(args.raw_dir / "target_multiple_matches.csv"),
        "target_search_scope": "selected_sweep_only",
    }
    print(
        "Targeted recovery search within selected sweep only: "
        f"targets={diag['target_configs']}, matched={diag['matched_targets']}, "
        f"still_missing={diag['still_missing_targets']}, unique_runs_selected={diag['unique_runs_selected']}"
    )
    return selected, diag


def download_rows(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    completed_path: Path,
    checkpoint_path: Path,
    completed_ids: set[str],
    task_buckets: dict[str, str],
    label: str,
    manifest_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        return {"label": label, "new_runs": 0, "new_history_rows": 0, "new_runs_by_task": {}, "new_history_rows_by_task": {}}

    api = wandb.Api()
    writers = DownloadWriters(args.raw_dir, args.out_root, no_history=bool(args.no_history), split_mmlu_by_task=bool(args.split_mmlu_by_task))
    counts_by_task: Counter[str] = Counter()
    history_rows_by_task: Counter[str] = Counter()
    n_appended = 0
    n_history_rows = 0

    try:
        with completed_path.open("a", encoding="utf-8") as completed_f:
            for i, manifest_row in enumerate(rows):
                run_id = str(manifest_row["run_id"])
                if run_id in completed_ids:
                    continue

                run = api.run(f"{args.entity}/{args.project}/{run_id}")
                if not keep_state(run.state, args.states):
                    print(f"Skipping run changed state: label={label}, index={i}, run_id={run_id}, state={run.state}")
                    continue
                row = base_row(run)
                if not keep_dataset(row, args.dataset):
                    print(f"Skipping run changed dataset: label={label}, index={i}, run_id={run_id}, dataset={infer_dataset_from_row(row)}")
                    continue
                if args.dataset == "mmlu" and not keep_mmlu_size(row, args.mmlu_size_bucket, task_buckets):
                    print(f"Skipping run changed MMLU size: label={label}, index={i}, run_id={run_id}, bucket={infer_mmlu_size_bucket(row, task_buckets)}")
                    continue
                if args.dataset == "mmlu" and not row.get("mmlu_size_bucket"):
                    bucket = infer_mmlu_size_bucket(row, task_buckets)
                    if bucket:
                        row["mmlu_size_bucket"] = bucket

                task = get_mmlu_task(row) if args.dataset == "mmlu" else None
                summary = dict(run.summary)
                hist: list[dict[str, Any]] = []
                if not args.no_history:
                    hist = scan_history_with_retry(run, page_size=int(args.page_size), retries=int(args.retries), sleep_s=float(args.sleep))

                summary_row = dict(row)
                for k in SUMMARY_COLUMNS:
                    if k not in summary_row:
                        summary_row[k] = summary.get(k)

                history_rows: list[dict[str, Any]] = []
                for h in hist:
                    hrow = dict(row)
                    for k in HISTORY_KEYS:
                        hrow[k] = h.get(k)
                    history_rows.append(hrow)

                writers.write_run(task=task, summary_row=summary_row, history_rows=history_rows)

                completed_record = {
                    "manifest_index": i,
                    "download_label": label,
                    "run_id": run_id,
                    "dataset": infer_dataset_from_row(row),
                    "mmlu_task": task,
                    "mmlu_size_bucket": row.get("mmlu_size_bucket"),
                    "experiment_variant": row.get("experiment_variant"),
                    "history_rows": len(history_rows),
                    "completed_at_unix": time.time(),
                }
                completed_f.write(json.dumps(completed_record, ensure_ascii=False) + "\n")
                completed_f.flush()

                completed_ids.add(run_id)
                n_appended += 1
                n_history_rows += len(history_rows)
                if task:
                    counts_by_task[task] += 1
                    history_rows_by_task[task] += len(history_rows)

                checkpoint = {
                    "entity": args.entity,
                    "project": args.project,
                    "sweep_id": args.sweep_id,
                    "dataset": args.dataset,
                    "mmlu_size_bucket": args.mmlu_size_bucket,
                    "download_label": label,
                    "total_rows_in_current_label": len(rows),
                    "last_completed_label_index": i,
                    "last_completed_run_id": run_id,
                    "total_completed_runs": len(completed_ids),
                    "new_runs_this_label": n_appended,
                    "new_history_rows_this_label": n_history_rows,
                    "raw_summary_path": str(args.raw_dir / "runs_summary.csv"),
                    "raw_history_path": None if args.no_history else str(args.raw_dir / "runs_history.csv.gz"),
                    "out_root": None if args.out_root is None else str(args.out_root),
                    "no_history": bool(args.no_history),
                    "split_mmlu_by_task": bool(args.split_mmlu_by_task),
                    "manifest_diagnostics": manifest_diagnostics,
                }
                write_checkpoint(checkpoint_path, checkpoint)

                if n_appended % int(args.print_every) == 0:
                    print(f"[{label}] Appended {n_appended} new runs; new history rows={n_history_rows}; completed total={len(completed_ids)}", flush=True)
                if args.max_new_runs is not None and n_appended >= int(args.max_new_runs):
                    print(f"Stopping early because --max-new-runs={args.max_new_runs}")
                    break
    finally:
        writers.close()

    return {
        "label": label,
        "new_runs": n_appended,
        "new_history_rows": n_history_rows,
        "new_runs_by_task": dict(counts_by_task),
        "new_history_rows_by_task": dict(history_rows_by_task),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--sweep-id", required=True)
    p.add_argument(
        "--dataset",
        choices=["auto", "gsm8k", "piqa", "alpaca", "mmlu"],
        default="auto",
    )
    p.add_argument("--mmlu-size-bucket", choices=["all", "small", "medium", "large"], default="all")
    p.add_argument("--task-metadata", type=Path, default=None)
    p.add_argument("--raw-dir", "--out-dir", dest="raw_dir", required=True, type=Path)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--split-mmlu-by-task", action="store_true")
    p.add_argument("--states", default="finished")
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--refresh-manifest", action="store_true")
    p.add_argument("--page-size", type=int, default=10000)
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--sleep", type=float, default=8.0)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--max-new-runs", type=int, default=None)
    p.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Exact experiment_variant to keep server-side; repeat as needed.",
    )
    p.add_argument(
        "--matrix",
        action="append",
        default=[],
        help="Exact matrix config value to keep server-side; repeat as needed.",
    )

    # Expected-grid completeness and targeted recovery.
    p.add_argument("--expected-grid-yaml", type=Path, default=None, help="Sweep YAML used to generate expected matrix x run_seed x experiment_variant configs.")
    p.add_argument("--expected-configs-csv", type=Path, default=None, help="CSV of expected or missing configs with matrix, run_seed, experiment_variant columns.")
    p.add_argument("--recover-missing-targeted", action="store_true", help="After normal download, compare against expected configs and query missing configs only inside the selected sweep.")
    p.add_argument("--target-match-by", choices=["config", "name", "both"], default="both", help="Targeted recovery matching strategy. All queries are restricted to --sweep-id.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    if args.split_mmlu_by_task:
        if args.dataset != "mmlu":
            raise SystemExit("--split-mmlu-by-task requires --dataset mmlu")
        if args.out_root is None:
            raise SystemExit("--split-mmlu-by-task requires --out-root")
        args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mmlu_size_bucket != "all" and args.dataset != "mmlu":
        raise SystemExit("--mmlu-size-bucket other than all requires --dataset mmlu")
    if args.recover_missing_targeted and not (args.expected_grid_yaml or args.expected_configs_csv):
        raise SystemExit("--recover-missing-targeted requires --expected-grid-yaml or --expected-configs-csv")

    manifest_path = args.raw_dir / "wandb_run_manifest.csv"
    completed_path = args.raw_dir / "completed_runs.jsonl"
    checkpoint_path = args.raw_dir / "download_checkpoint.json"
    metadata_path = args.raw_dir / "download_metadata.json"

    task_buckets = load_mmlu_task_buckets(args.task_metadata)
    if args.refresh_manifest or not manifest_path.is_file():
        manifest_rows, manifest_diagnostics = build_manifest(args, manifest_path, task_buckets)
    else:
        loaded_rows = load_manifest(manifest_path)
        manifest_rows, manifest_diagnostics = dedupe_loaded_manifest(loaded_rows, args.raw_dir)
        print(f"Loaded existing manifest: {manifest_path} (raw rows={len(loaded_rows)}, unique rows={len(manifest_rows)})")

    completed_ids, _completed_index_by_run = load_completed_runs(completed_path)
    print(f"Loaded checkpoint: completed unique runs={len(completed_ids)}")

    incomplete_rows = [row for row in manifest_rows if str(row.get("run_id", "")) not in completed_ids]
    if incomplete_rows:
        print(f"Downloading incomplete manifest rows: {len(incomplete_rows)} / {len(manifest_rows)}")
        manifest_result = download_rows(
            args=args,
            rows=incomplete_rows,
            completed_path=completed_path,
            checkpoint_path=checkpoint_path,
            completed_ids=completed_ids,
            task_buckets=task_buckets,
            label="manifest",
            manifest_diagnostics=manifest_diagnostics,
        )
    else:
        print("All unique manifest runs are already completed. Nothing to download from manifest.")
        manifest_result = {"label": "manifest", "new_runs": 0, "new_history_rows": 0, "new_runs_by_task": {}, "new_history_rows_by_task": {}}

    recovery_diag: dict[str, Any] | None = None
    recovery_result: dict[str, Any] | None = None
    final_missing_count: int | None = None

    if args.expected_grid_yaml is not None or args.expected_configs_csv is not None:
        expected_rows = load_expected_rows(args)
        missing_rows = write_missing_expected_configs(args, expected_rows)
        final_missing_count = len(missing_rows)

        if args.recover_missing_targeted and missing_rows:
            target_rows, recovery_diag = find_target_runs_within_sweep(args, missing_rows)
            # Do not redownload runs already completed. The target query can find rows already in manifest.
            target_rows = [r for r in target_rows if str(r.get("run_id", "")) not in completed_ids]
            if target_rows:
                recovery_result = download_rows(
                    args=args,
                    rows=target_rows,
                    completed_path=completed_path,
                    checkpoint_path=checkpoint_path,
                    completed_ids=completed_ids,
                    task_buckets=task_buckets,
                    label="targeted_recovery",
                    manifest_diagnostics=manifest_diagnostics,
                )
            else:
                recovery_result = {"label": "targeted_recovery", "new_runs": 0, "new_history_rows": 0, "new_runs_by_task": {}, "new_history_rows_by_task": {}}
            # Recompute after recovery.
            final_missing_count = len(write_missing_expected_configs(args, expected_rows))

    metadata = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "dataset": args.dataset,
        "mmlu_size_bucket": args.mmlu_size_bucket,
        "raw_dir": str(args.raw_dir),
        "out_root": None if args.out_root is None else str(args.out_root),
        "states": args.states,
        "variants": list(args.variant),
        "matrices": list(args.matrix),
        "task_metadata": None if args.task_metadata is None else str(args.task_metadata),
        "manifest_path": str(manifest_path),
        "completed_path": str(completed_path),
        "checkpoint_path": str(checkpoint_path),
        "manifest_runs": len(manifest_rows),
        "manifest_unique_run_ids": len({str(r.get("run_id", "")) for r in manifest_rows}),
        "completed_runs_after_process": len(completed_ids),
        "manifest_result": manifest_result,
        "targeted_recovery_result": recovery_result,
        "targeted_recovery_diagnostics": recovery_diag,
        "final_missing_expected_configs": final_missing_count,
        "no_history": bool(args.no_history),
        "split_mmlu_by_task": bool(args.split_mmlu_by_task),
        "manifest_diagnostics": manifest_diagnostics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("Done. Re-run the same command to resume if interrupted. Targeted recovery is restricted to the selected sweep only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
