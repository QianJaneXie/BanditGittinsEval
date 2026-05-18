#!/usr/bin/env python3
"""Download MMLU-medium W&B sweep data with Gittins stopping artifacts, then split by task.

Outputs:
  RAW_DIR/
    runs_summary.csv
    runs_history.csv.gz
    runs_stopping.csv
    metadata.json
    artifacts/                 # downloaded only for Gittins runs by default

  OUT_ROOT/<task>/
    runs_summary.csv
    runs_history.csv.gz
    runs_stopping.csv
    metadata.json

Default medium sweep:
  EfficientLLMEval/GittinsBanditEval/e1gh776a
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
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
    "matrix",                 # IMPORTANT for MMLU task splitting
    "matrix_task",            # normalized task name, e.g. high_school_physics
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

STOPPING_COLUMNS = [
    "gittins_stop_cum_eval",
    "gittins_stop_cum_original_cost",
    "gittins_stop_found",
    "trace_artifact_name",
    "trace_artifact_path",
    "trace_local_path",
    "trace_extract_error",
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
] + STOPPING_COLUMNS

HISTORY_COLUMNS = BASE_COLUMNS + HISTORY_KEYS
STOPPING_CSV_COLUMNS = BASE_COLUMNS + STOPPING_COLUMNS

FALLBACK_MEDIUM_TASKS = [
    "high_school_physics",
    "astronomy",
    "logical_fallacies",
    "high_school_european_history",
    "virology",
    "world_religions",
    "college_medicine",
    "high_school_government_and_politics",
    "high_school_geography",
    "sociology",
    "high_school_chemistry",
    "high_school_us_history",
    "high_school_statistics",
    "human_aging",
    "marketing",
    "conceptual_physics",
    "high_school_world_history",
    "high_school_microeconomics",
    "security_studies",
    "clinical_knowledge",
    "high_school_mathematics",
    "professional_medicine",
    "professional_accounting",
    "nutrition",
    "high_school_biology",
    "philosophy",
    "prehistory",
    "moral_disputes",
    "elementary_mathematics",
    "high_school_macroeconomics",
]


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def matrix_to_task(matrix: Any) -> str | None:
    if matrix is None:
        return None
    s = str(matrix).strip()
    if not s or s.lower() == "nan":
        return None
    return Path(s).stem


def parse_variant_fallback(experiment_variant: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not experiment_variant:
        return out

    m = re.fullmatch(r"(ucb|lrf)_B(\d+)", experiment_variant)
    if m:
        out["policy_variant"] = m.group(1)
        out["policy_family"] = m.group(1)
        out["cost_mode"] = "baseline"
        out["batch_size"] = int(m.group(2))
        out["gittins_batch_size"] = int(m.group(2))
        out["cost_scaling_factor"] = 1e-4
        out["prior_type"] = "default"
        return out

    m = re.fullmatch(r"gittins_(unit|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)", experiment_variant)
    if m:
        cost_mode = m.group(1)
        out["policy_variant"] = f"gittins_{cost_mode}"
        out["policy_family"] = "gittins"
        out["cost_mode"] = cost_mode
        out["batch_size"] = int(m.group(2))
        out["gittins_batch_size"] = int(m.group(2))
        out["cost_scaling_factor"] = float(m.group(3))
        out["prior_type"] = m.group(4)
        return out

    return out


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
    task = matrix_to_task(matrix)

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


def _scalar_or_default(z: np.lib.npyio.NpzFile, key: str, default: float | int | None = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    try:
        return arr.reshape(())[()]
    except Exception:
        return arr.reshape(-1)[0]


def _infer_stop_cost_from_trace(z: np.lib.npyio.NpzFile, stop_eval: int) -> float:
    if stop_eval < 0:
        return -1.0
    x_key = "x" if "x" in z.files else "gittins_x" if "gittins_x" in z.files else None
    cost_key = "x_original_cost" if "x_original_cost" in z.files else "gittins_x_original_cost" if "gittins_x_original_cost" in z.files else None
    if x_key is None or cost_key is None:
        return -1.0
    x = np.asarray(z[x_key], dtype=float).reshape(-1)
    c = np.asarray(z[cost_key], dtype=float).reshape(-1)
    if x.size == 0 or c.size == 0 or x.size != c.size:
        return -1.0
    order = np.argsort(x)
    x = x[order]
    c = c[order]
    idx = int(np.searchsorted(x, float(stop_eval), side="right") - 1)
    if idx < 0:
        return 0.0
    return float(c[idx])


def extract_stopping_from_trace(trace_path: Path) -> dict[str, Any]:
    out = {
        "gittins_stop_cum_eval": -1,
        "gittins_stop_cum_original_cost": -1.0,
        "gittins_stop_found": False,
        "trace_extract_error": None,
    }
    try:
        with np.load(trace_path, allow_pickle=False) as z:
            stop_eval = int(_scalar_or_default(z, "gittins_stop_cum_eval", -1))
            raw_stop_cost = _scalar_or_default(z, "gittins_stop_cum_original_cost", None)
            stop_cost = _infer_stop_cost_from_trace(z, stop_eval) if raw_stop_cost is None else float(raw_stop_cost)
            out["gittins_stop_cum_eval"] = int(stop_eval)
            out["gittins_stop_cum_original_cost"] = float(stop_cost)
            out["gittins_stop_found"] = bool(stop_eval >= 0)
    except Exception as e:
        out["trace_extract_error"] = repr(e)
    return out


def find_trace_npz(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*_traces.npz"))
    if not candidates:
        candidates = sorted(root.rglob("*.npz"))
    return candidates[0] if candidates else None


def maybe_download_trace_artifact(run: wandb.apis.public.Run, row: dict[str, Any], artifact_root: Path, artifact_policy: str) -> dict[str, Any]:
    result = {
        "trace_artifact_name": None,
        "trace_artifact_path": None,
        "trace_local_path": None,
        "trace_extract_error": None,
        "gittins_stop_cum_eval": -1,
        "gittins_stop_cum_original_cost": -1.0,
        "gittins_stop_found": False,
    }
    if artifact_policy == "gittins" and str(row.get("policy_family")) != "gittins":
        return result
    try:
        artifacts = list(run.logged_artifacts())
    except Exception as e:
        result["trace_extract_error"] = f"logged_artifacts failed: {e!r}"
        return result
    artifacts = [a for a in artifacts if getattr(a, "type", None) == "experiment_outputs"] or artifacts
    if not artifacts:
        result["trace_extract_error"] = "no logged artifact found"
        return result
    artifact = artifacts[-1]
    result["trace_artifact_name"] = getattr(artifact, "name", None)
    try:
        local_root = artifact_root / safe_token(run.id) / safe_token(str(artifact.name))
        local_root.mkdir(parents=True, exist_ok=True)
        downloaded_dir = Path(artifact.download(root=str(local_root)))
        trace_path = find_trace_npz(downloaded_dir)
        result["trace_artifact_path"] = str(downloaded_dir)
        if trace_path is None:
            result["trace_extract_error"] = f"no *_traces.npz under {downloaded_dir}"
            return result
        result["trace_local_path"] = str(trace_path)
        result.update(extract_stopping_from_trace(trace_path))
    except Exception as e:
        result["trace_extract_error"] = f"artifact download/extract failed: {e!r}"
    return result


def medium_tasks_from_metadata(path: Path | None) -> list[str]:
    if path is not None and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = [t["task"] for t in data.get("tasks", []) if t.get("size_bucket") == "medium"]
        if tasks:
            return tasks
    return FALLBACK_MEDIUM_TASKS


def download_raw(args: argparse.Namespace) -> dict[str, Any]:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    keep_states = None if args.states.lower() == "all" else {s.strip() for s in args.states.split(",") if s.strip()}
    api = wandb.Api()
    runs = api.runs(path=f"{args.entity}/{args.project}", filters={"sweep": args.sweep_id})

    summary_path = args.raw_dir / "runs_summary.csv"
    history_path = args.raw_dir / "runs_history.csv.gz"
    stopping_path = args.raw_dir / "runs_stopping.csv"
    metadata_path = args.raw_dir / "metadata.json"
    artifact_root = args.raw_dir / "artifacts"

    n_runs_seen = n_runs_kept = n_history_rows = n_artifact_attempts = n_trace_found = n_stop_found = 0

    with summary_path.open("w", newline="", encoding="utf-8") as sf, stopping_path.open("w", newline="", encoding="utf-8") as stf:
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_COLUMNS)
        stopping_writer = csv.DictWriter(stf, fieldnames=STOPPING_CSV_COLUMNS)
        summary_writer.writeheader()
        stopping_writer.writeheader()

        history_handle = gzip.open(history_path, "wt", newline="", encoding="utf-8")
        history_writer = csv.DictWriter(history_handle, fieldnames=HISTORY_COLUMNS)
        history_writer.writeheader()
        try:
            for run in runs:
                n_runs_seen += 1
                if keep_states is not None and run.state not in keep_states:
                    continue
                n_runs_kept += 1
                summary = dict(run.summary)
                row = base_row(run)
                stopping_info = {
                    "gittins_stop_cum_eval": get_field(dict(run.config), summary, "gittins_stop_cum_eval") or -1,
                    "gittins_stop_cum_original_cost": get_field(dict(run.config), summary, "gittins_stop_cum_original_cost") or -1.0,
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
                    stopping_info.update(maybe_download_trace_artifact(run, row, artifact_root, args.artifact_policy))

                if stopping_info.get("trace_local_path"):
                    n_trace_found += 1
                if bool(stopping_info.get("gittins_stop_found")):
                    n_stop_found += 1

                summary_row = dict(row)
                for k in SUMMARY_COLUMNS:
                    if k in stopping_info:
                        summary_row[k] = stopping_info.get(k)
                    elif k not in summary_row:
                        summary_row[k] = summary.get(k)
                summary_writer.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})

                stopping_row = dict(row)
                stopping_row.update(stopping_info)
                stopping_writer.writerow({k: stopping_row.get(k) for k in STOPPING_CSV_COLUMNS})

                for h in run.scan_history(keys=HISTORY_KEYS, page_size=args.page_size):
                    hrow = dict(row)
                    for k in HISTORY_KEYS:
                        hrow[k] = h.get(k)
                    history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
                    n_history_rows += 1

                if n_runs_kept % args.print_every == 0:
                    print(f"Downloaded {n_runs_kept} finished runs; history rows={n_history_rows}; traces={n_trace_found}; stops={n_stop_found}", flush=True)
        finally:
            history_handle.close()

    metadata = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "states": args.states,
        "download_artifacts": bool(args.download_artifacts),
        "artifact_policy": args.artifact_policy,
        "n_runs_seen": n_runs_seen,
        "n_runs_kept": n_runs_kept,
        "n_history_rows": n_history_rows,
        "n_artifact_attempts": n_artifact_attempts,
        "n_trace_found": n_trace_found,
        "n_stop_found": n_stop_found,
        "summary_path": str(summary_path),
        "history_path": str(history_path),
        "stopping_path": str(stopping_path),
        "artifact_root": str(artifact_root) if args.download_artifacts else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def write_filtered_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def split_by_task(args: argparse.Namespace, medium_tasks: list[str]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    task_set = set(medium_tasks)

    summary = pd.read_csv(args.raw_dir / "runs_summary.csv")
    stopping = pd.read_csv(args.raw_dir / "runs_stopping.csv")

    if "matrix_task" not in summary.columns:
        if "matrix" not in summary.columns:
            raise SystemExit("runs_summary.csv has neither matrix_task nor matrix. Please use this script's downloader, not the older downloader.")
        summary["matrix_task"] = summary["matrix"].map(matrix_to_task)
    if "matrix_task" not in stopping.columns:
        if "matrix" not in stopping.columns:
            raise SystemExit("runs_stopping.csv has neither matrix_task nor matrix. Please use this script's downloader, not the older downloader.")
        stopping["matrix_task"] = stopping["matrix"].map(matrix_to_task)

    summary = summary[summary["matrix_task"].isin(task_set)].copy()
    stopping = stopping[stopping["matrix_task"].isin(task_set)].copy()

    print(f"Splitting summary/stopping for {len(medium_tasks)} medium tasks ...", flush=True)
    for task in medium_tasks:
        task_dir = args.out_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        s = summary[summary["matrix_task"] == task]
        st = stopping[stopping["matrix_task"] == task]
        write_filtered_csv(s, task_dir / "runs_summary.csv")
        write_filtered_csv(st, task_dir / "runs_stopping.csv")

    print("Splitting history csv.gz by task ...", flush=True)
    handles: dict[str, Any] = {}
    writers: dict[str, csv.DictWriter] = {}
    counts = {task: 0 for task in medium_tasks}
    try:
        for task in medium_tasks:
            h = gzip.open(args.out_root / task / "runs_history.csv.gz", "wt", newline="", encoding="utf-8")
            handles[task] = h
            w = csv.DictWriter(h, fieldnames=HISTORY_COLUMNS)
            writers[task] = w
            w.writeheader()

        with gzip.open(args.raw_dir / "runs_history.csv.gz", "rt", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row.get("matrix_task") or matrix_to_task(row.get("matrix"))
                if task in task_set:
                    writers[task].writerow({k: row.get(k) for k in HISTORY_COLUMNS})
                    counts[task] += 1
    finally:
        for h in handles.values():
            h.close()

    for task in medium_tasks:
        task_dir = args.out_root / task
        s = summary[summary["matrix_task"] == task]
        st = stopping[stopping["matrix_task"] == task]
        meta = {
            "task": task,
            "size_bucket": "medium",
            "recommended_batches": [8, 16, 32],
            "source_raw_dir": str(args.raw_dir),
            "n_summary_runs": int(len(s)),
            "n_stopping_rows": int(len(st)),
            "n_history_rows": int(counts[task]),
            "n_trace_found": int(st["trace_local_path"].notna().sum()) if "trace_local_path" in st.columns else 0,
            "n_stop_found": int(st["gittins_stop_found"].astype(str).str.lower().isin(["true", "1"]).sum()) if "gittins_stop_found" in st.columns else 0,
        }
        (task_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done. Per-task folders written under:", args.out_root, flush=True)
    print("Quick counts:", flush=True)
    for task in medium_tasks:
        print(f"  {task}: runs={int((summary['matrix_task'] == task).sum())}, history_rows={counts[task]}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="EfficientLLMEval")
    p.add_argument("--project", default="GittinsBanditEval")
    p.add_argument("--sweep-id", default="e1gh776a")
    p.add_argument("--states", default="finished")
    p.add_argument("--raw-dir", type=Path, default=Path("outputs/wandb_downloads/mmlu_medium_raw_finished_with_stopping"))
    p.add_argument("--out-root", type=Path, default=Path("outputs/wandb_downloads/mmlu_medium_merged_finished_with_stopping"))
    p.add_argument("--task-metadata", type=Path, default=Path("data/MMLU_matrices/task_metadata.json"))
    p.add_argument("--page-size", type=int, default=10000)
    p.add_argument("--download-artifacts", action="store_true", default=True)
    p.add_argument("--artifact-policy", choices=["gittins", "all"], default="gittins")
    p.add_argument("--skip-download", action="store_true", help="Use existing raw csv files and only redo the task split.")
    p.add_argument("--print-every", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    medium_tasks = medium_tasks_from_metadata(args.task_metadata)
    print(f"Medium tasks ({len(medium_tasks)}): {', '.join(medium_tasks)}", flush=True)
    if not args.skip_download:
        download_raw(args)
    split_by_task(args, medium_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
