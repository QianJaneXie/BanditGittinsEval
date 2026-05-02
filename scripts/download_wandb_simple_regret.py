#!/usr/bin/env python3
"""Download W&B simple-regret sweep data, including Gittins stopping info from artifacts.

Outputs:
1. runs_summary.csv      : run-level config + summary metrics + stopping columns
2. runs_history.csv.gz   : step-level history points
3. runs_stopping.csv     : one row per run with stopping info extracted from *_traces.npz
4. metadata.json         : download metadata
5. artifacts/            : downloaded W&B artifacts, when --download-artifacts is used

Important:
- Existing runs did not log gittins_stop_cum_eval to run.summary.
- For existing runs, use --download-artifacts so this script can open each *_traces.npz.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
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
    # These are useful for plotting LRF warmup cropping.
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


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


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

    m = re.fullmatch(
        r"gittins_(unit|aware)_B(\d+)_scale([0-9.eE+-]+)_(default|dataset)",
        experiment_variant,
    )
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

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "dataset_tag_resolved": cfg.get("dataset_tag_resolved"),
        "dataset_tag": cfg.get("dataset_tag"),
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
    """Infer stop cost for old traces that only stored gittins_stop_cum_eval.

    Current W&B traces store x and x_original_cost for the single variant.  The natural stop is
    recorded at a decision boundary before the next batch, so the best fallback is the last logged
    cumulative cost whose cumulative eval is <= stop_eval.  If stop_eval is before the first logged
    point, the cost is 0.
    """
    if stop_eval < 0:
        return -1.0

    x_key = "x" if "x" in z.files else "gittins_x" if "gittins_x" in z.files else None
    cost_key = (
        "x_original_cost"
        if "x_original_cost" in z.files
        else "gittins_x_original_cost"
        if "gittins_x_original_cost" in z.files
        else None
    )
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
            if raw_stop_cost is None:
                stop_cost = _infer_stop_cost_from_trace(z, stop_eval)
            else:
                stop_cost = float(raw_stop_cost)

            out["gittins_stop_cum_eval"] = int(stop_eval)
            out["gittins_stop_cum_original_cost"] = float(stop_cost)
            out["gittins_stop_found"] = bool(stop_eval >= 0)
    except Exception as e:  # noqa: BLE001 - we want this downloader to keep going.
        out["trace_extract_error"] = repr(e)
    return out


def find_trace_npz(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*_traces.npz"))
    if not candidates:
        candidates = sorted(root.rglob("*.npz"))
    return candidates[0] if candidates else None


def maybe_download_trace_artifact(
    *,
    run: wandb.apis.public.Run,
    row: dict[str, Any],
    artifact_root: Path,
    artifact_policy: str,
) -> dict[str, Any]:
    result = {
        "trace_artifact_name": None,
        "trace_artifact_path": None,
        "trace_local_path": None,
        "trace_extract_error": None,
        "gittins_stop_cum_eval": -1,
        "gittins_stop_cum_original_cost": -1.0,
        "gittins_stop_found": False,
    }

    # Current stopping is only meaningful for Gittins variants.
    if artifact_policy == "gittins" and str(row.get("policy_family")) != "gittins":
        return result

    try:
        artifacts = list(run.logged_artifacts())
    except Exception as e:  # noqa: BLE001
        result["trace_extract_error"] = f"logged_artifacts failed: {e!r}"
        return result

    # Prefer the experiment_outputs artifact created by run_simple_regret_wandb.py.
    artifacts = [a for a in artifacts if getattr(a, "type", None) == "experiment_outputs"] or artifacts
    if not artifacts:
        result["trace_extract_error"] = "no logged artifact found"
        return result

    # Use the latest-looking artifact if there are multiple.
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
    except Exception as e:  # noqa: BLE001
        result["trace_extract_error"] = f"artifact download/extract failed: {e!r}"

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--sweep-id", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--states",
        default="finished",
        help="Comma-separated W&B run states to keep, e.g. finished,running,failed or all",
    )
    p.add_argument("--no-history", action="store_true", help="Only download run config + summary, not step-level history")
    p.add_argument("--page-size", type=int, default=10000)
    p.add_argument(
        "--download-artifacts",
        action="store_true",
        help="Download W&B artifacts and extract gittins_stop_cum_eval from *_traces.npz.",
    )
    p.add_argument(
        "--artifact-policy",
        choices=["gittins", "all"],
        default="gittins",
        help="When --download-artifacts is used, download artifacts for only Gittins runs or all runs.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keep_states = None if args.states.lower() == "all" else {s.strip() for s in args.states.split(",") if s.strip()}

    api = wandb.Api()
    runs = api.runs(path=f"{args.entity}/{args.project}", filters={"sweep": args.sweep_id})

    summary_path = args.out_dir / "runs_summary.csv"
    history_path = args.out_dir / "runs_history.csv.gz"
    stopping_path = args.out_dir / "runs_stopping.csv"
    metadata_path = args.out_dir / "metadata.json"
    artifact_root = args.out_dir / "artifacts"

    n_runs_seen = 0
    n_runs_kept = 0
    n_history_rows = 0
    n_artifact_attempts = 0
    n_trace_found = 0
    n_stop_found = 0

    with summary_path.open("w", newline="", encoding="utf-8") as sf, stopping_path.open("w", newline="", encoding="utf-8") as stf:
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_COLUMNS)
        stopping_writer = csv.DictWriter(stf, fieldnames=STOPPING_CSV_COLUMNS)
        summary_writer.writeheader()
        stopping_writer.writeheader()

        if args.no_history:
            history_handle = None
            history_writer = None
        else:
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
                    artifact_info = maybe_download_trace_artifact(
                        run=run,
                        row=row,
                        artifact_root=artifact_root,
                        artifact_policy=args.artifact_policy,
                    )
                    # Artifact info should overwrite empty/default summary stopping values.
                    stopping_info.update(artifact_info)

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

                if history_writer is not None:
                    for h in run.scan_history(keys=HISTORY_KEYS, page_size=args.page_size):
                        hrow = dict(row)
                        for k in HISTORY_KEYS:
                            hrow[k] = h.get(k)
                        history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
                        n_history_rows += 1

                if n_runs_kept % 50 == 0:
                    print(
                        f"Downloaded {n_runs_kept} kept runs; history rows={n_history_rows}; "
                        f"traces={n_trace_found}; stops={n_stop_found}"
                    )
        finally:
            if history_handle is not None:
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
        "history_path": None if args.no_history else str(history_path),
        "stopping_path": str(stopping_path),
        "artifact_root": str(artifact_root) if args.download_artifacts else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
