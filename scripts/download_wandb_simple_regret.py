#!/usr/bin/env python3
"""Download W&B simple-regret sweep data.

Downloads:
1. run-level config + summary metrics -> runs_summary.csv
2. step-level history points -> runs_history.csv.gz

Example:
python scripts/download_wandb_simple_regret.py --entity EfficientLLMEval --project GittinsBanditEval --sweep-id 406hah4y --out-dir outputs/wandb_downloads/gsm8k_406hah4y
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

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
]

SUMMARY_COLUMNS = BASE_COLUMNS + [
    "final_simple_regret",
    "best_seen_regret",
    "final_cum_eval",
    "final_cum_original_cost",
    "num_batches",
    "gittins_stop_cum_eval",
    "gittins_stop_cum_original_cost",
    "gittins_recommendation_aware_stop_cum_eval",
    "gittins_recommendation_aware_stop_cum_original_cost",
    "lookup_table_s",
    "iter_step_mean_s",
    "iter_step_median_s",
    "iter_step_p90_s",
    "iter_total_mean_s",
    "iter_total_median_s",
    "iter_total_p90_s",
]

HISTORY_COLUMNS = BASE_COLUMNS + HISTORY_KEYS


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

    row = {
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
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--sweep-id", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--states", default="finished", help="Comma-separated W&B run states to keep, e.g. finished,running,failed or all")
    p.add_argument("--no-history", action="store_true", help="Only download run config + summary, not step-level history")
    p.add_argument("--page-size", type=int, default=10000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keep_states = None if args.states.lower() == "all" else {s.strip() for s in args.states.split(",") if s.strip()}

    api = wandb.Api()
    runs = api.runs(path=f"{args.entity}/{args.project}", filters={"sweep": args.sweep_id})

    summary_path = args.out_dir / "runs_summary.csv"
    history_path = args.out_dir / "runs_history.csv.gz"
    metadata_path = args.out_dir / "metadata.json"

    n_runs_seen = 0
    n_runs_kept = 0
    n_history_rows = 0

    with summary_path.open("w", newline="", encoding="utf-8") as sf:
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_COLUMNS)
        summary_writer.writeheader()

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

                summary_row = dict(row)
                for k in SUMMARY_COLUMNS:
                    if k not in summary_row:
                        summary_row[k] = summary.get(k)
                summary_writer.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})

                if history_writer is not None:
                    for h in run.scan_history(keys=HISTORY_KEYS, page_size=args.page_size):
                        hrow = dict(row)
                        for k in HISTORY_KEYS:
                            hrow[k] = h.get(k)
                        history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
                        n_history_rows += 1

                if n_runs_kept % 50 == 0:
                    print(f"Downloaded {n_runs_kept} kept runs; history rows={n_history_rows}")
        finally:
            if history_handle is not None:
                history_handle.close()

    metadata = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "states": args.states,
        "n_runs_seen": n_runs_seen,
        "n_runs_kept": n_runs_kept,
        "n_history_rows": n_history_rows,
        "summary_path": str(summary_path),
        "history_path": None if args.no_history else str(history_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
