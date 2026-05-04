#!/usr/bin/env python3
"""Download multiple matrices/subjects from one W&B simple-regret sweep in a single scan.

Place this file in the same folder as download_wandb_one_matrix_local.py, usually:
  GittinsBanditEval/scripts/

It scans the sweep once, then writes one output folder per matrix:
  <out-dir>/<matrix_name>/runs_summary.csv
  <out-dir>/<matrix_name>/runs_history.csv.gz
  <out-dir>/<matrix_name>/runs_stopping.csv
  <out-dir>/<matrix_name>/metadata.json

Example:
  python .\scripts\download_wandb_multi_matrix_local.py `
    --entity EfficientLLMEval `
    --project GittinsBanditEval `
    --sweep-id 4dcyp8dx `
    --matrix-contains medical_genetics us_foreign_policy college_physics management jurisprudence public_relations machine_learning econometrics international_law `
    --states finished `
    --out-dir .\outputs\wandb_downloads\mmlu_small_selected_matrices_finished
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import wandb

# This script is intentionally small by reusing the single-matrix downloader helpers.
# Keep download_wandb_one_matrix_local.py in the same directory as this file.
from download_wandb_one_matrix_local import (  # type: ignore
    HISTORY_KEYS,
    SUMMARY_COLUMNS,
    HISTORY_COLUMNS,
    STOPPING_CSV_COLUMNS,
    base_row,
    get_field,
    maybe_download_trace_artifact,
    safe_token,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--sweep-id", required=True)
    p.add_argument(
        "--matrix-contains",
        required=True,
        nargs="+",
        help="Matrix/task substrings. You may pass space-separated values or comma-separated values.",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--states", default="finished", help="Comma-separated states to keep, or 'all'.")
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--page-size", type=int, default=10000)
    p.add_argument("--download-artifacts", dest="download_artifacts", action="store_true", default=True)
    p.add_argument("--no-download-artifacts", dest="download_artifacts", action="store_false")
    p.add_argument("--artifact-policy", choices=["gittins", "all"], default="gittins")
    return p.parse_args()


def normalize_targets(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        for part in str(item).split(","):
            t = part.strip()
            if t and t not in out:
                out.append(t)
    return out


class MatrixWriter:
    def __init__(self, root: Path, target: str, no_history: bool):
        self.target = target
        self.out_dir = root / safe_token(target)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.out_dir / "runs_summary.csv"
        self.history_path = self.out_dir / "runs_history.csv.gz"
        self.stopping_path = self.out_dir / "runs_stopping.csv"
        self.metadata_path = self.out_dir / "metadata.json"
        self.artifact_root = self.out_dir / "artifacts"

        self.summary_handle = self.summary_path.open("w", newline="", encoding="utf-8")
        self.stopping_handle = self.stopping_path.open("w", newline="", encoding="utf-8")
        self.summary_writer = csv.DictWriter(self.summary_handle, fieldnames=SUMMARY_COLUMNS)
        self.stopping_writer = csv.DictWriter(self.stopping_handle, fieldnames=STOPPING_CSV_COLUMNS)
        self.summary_writer.writeheader()
        self.stopping_writer.writeheader()

        self.no_history = bool(no_history)
        if self.no_history:
            self.history_handle = None
            self.history_writer = None
        else:
            self.history_handle = gzip.open(self.history_path, "wt", newline="", encoding="utf-8")
            self.history_writer = csv.DictWriter(self.history_handle, fieldnames=HISTORY_COLUMNS)
            self.history_writer.writeheader()

        self.n_runs_kept = 0
        self.n_history_rows = 0
        self.n_artifact_attempts = 0
        self.n_trace_found = 0
        self.n_stop_found = 0

    def close(self) -> None:
        self.summary_handle.close()
        self.stopping_handle.close()
        if self.history_handle is not None:
            self.history_handle.close()

    def write_metadata(self, base: dict[str, Any]) -> None:
        metadata = dict(base)
        metadata.update(
            {
                "matrix_contains": self.target,
                "n_runs_kept": self.n_runs_kept,
                "n_history_rows": self.n_history_rows,
                "n_artifact_attempts": self.n_artifact_attempts,
                "n_trace_found": self.n_trace_found,
                "n_stop_found": self.n_stop_found,
                "summary_path": str(self.summary_path),
                "history_path": None if self.no_history else str(self.history_path),
                "stopping_path": str(self.stopping_path),
                "artifact_root": str(self.artifact_root) if base.get("download_artifacts") else None,
            }
        )
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_matches_targets(run: wandb.apis.public.Run, targets: list[str]) -> list[str]:
    cfg = dict(run.config)
    matrix = str(cfg.get("matrix", "")).replace("\\", "/").lower()
    name = str(run.name or "").lower()
    matches: list[str] = []
    for target in targets:
        t = target.lower()
        if t in matrix or t in name:
            matches.append(target)
    return matches


def stopping_info_for_run(
    *,
    run: wandb.apis.public.Run,
    row: dict[str, Any],
    cfg: dict[str, Any],
    summary: dict[str, Any],
    writer: MatrixWriter,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stopping_info = {
        "gittins_stop_cum_eval": get_field(cfg, summary, "gittins_stop_cum_eval") or -1,
        "gittins_stop_cum_original_cost": get_field(cfg, summary, "gittins_stop_cum_original_cost") or -1.0,
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
            writer.n_artifact_attempts += 1
        artifact_info = maybe_download_trace_artifact(
            run=run,
            row=row,
            artifact_root=writer.artifact_root,
            artifact_policy=args.artifact_policy,
        )
        stopping_info.update(artifact_info)

    if stopping_info.get("trace_local_path"):
        writer.n_trace_found += 1
    if bool(stopping_info.get("gittins_stop_found")):
        writer.n_stop_found += 1

    return stopping_info


def write_run_to_matrix(
    *,
    run: wandb.apis.public.Run,
    writer: MatrixWriter,
    args: argparse.Namespace,
) -> None:
    cfg = dict(run.config)
    summary = dict(run.summary)
    row = base_row(run)

    stopping_info = stopping_info_for_run(
        run=run,
        row=row,
        cfg=cfg,
        summary=summary,
        writer=writer,
        args=args,
    )

    summary_row = dict(row)
    for k in SUMMARY_COLUMNS:
        if k in stopping_info:
            summary_row[k] = stopping_info.get(k)
        elif k not in summary_row:
            summary_row[k] = summary.get(k)
    writer.summary_writer.writerow({k: summary_row.get(k) for k in SUMMARY_COLUMNS})

    stopping_row = dict(row)
    stopping_row.update(stopping_info)
    writer.stopping_writer.writerow({k: stopping_row.get(k) for k in STOPPING_CSV_COLUMNS})

    if writer.history_writer is not None:
        for h in run.scan_history(keys=HISTORY_KEYS, page_size=args.page_size):
            hrow = dict(row)
            for k in HISTORY_KEYS:
                hrow[k] = h.get(k)
            writer.history_writer.writerow({k: hrow.get(k) for k in HISTORY_COLUMNS})
            writer.n_history_rows += 1

    writer.n_runs_kept += 1


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = normalize_targets(args.matrix_contains)
    if not targets:
        raise SystemExit("No valid --matrix-contains targets.")

    keep_states = None if args.states.lower() == "all" else {s.strip() for s in args.states.split(",") if s.strip()}

    api = wandb.Api(timeout=120)
    filters: dict[str, Any] = {"sweep": args.sweep_id}
    if keep_states is not None and len(keep_states) == 1:
        filters["state"] = next(iter(keep_states))

    print("Targets:", ", ".join(targets), flush=True)
    print("Scanning sweep once...", flush=True)
    runs = api.runs(path=f"{args.entity}/{args.project}", filters=filters)

    writers = {target: MatrixWriter(args.out_dir, target, args.no_history) for target in targets}
    n_runs_seen = 0
    n_matching_seen = 0

    try:
        for run in runs:
            n_runs_seen += 1
            if n_runs_seen % 500 == 0:
                counts = ", ".join(f"{t}={writers[t].n_runs_kept}" for t in targets)
                total_history = sum(w.n_history_rows for w in writers.values())
                print(f"scanned={n_runs_seen}, matched={n_matching_seen}, history_rows={total_history}, kept: {counts}", flush=True)

            if keep_states is not None and run.state not in keep_states:
                continue

            matched_targets = run_matches_targets(run, targets)
            if not matched_targets:
                continue
            n_matching_seen += 1

            for target in matched_targets:
                write_run_to_matrix(run=run, writer=writers[target], args=args)
    finally:
        for w in writers.values():
            w.close()

    base_metadata = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "states": args.states,
        "download_artifacts": bool(args.download_artifacts),
        "artifact_policy": args.artifact_policy,
        "n_runs_seen": n_runs_seen,
        "n_matching_seen": n_matching_seen,
    }
    for w in writers.values():
        w.write_metadata(base_metadata)

    summary = {
        "entity": args.entity,
        "project": args.project,
        "sweep_id": args.sweep_id,
        "states": args.states,
        "targets": targets,
        "n_runs_seen": n_runs_seen,
        "n_matching_seen": n_matching_seen,
        "outputs": {t: {"out_dir": str(writers[t].out_dir), "n_runs_kept": writers[t].n_runs_kept, "n_history_rows": writers[t].n_history_rows} for t in targets},
    }
    (args.out_dir / "metadata_all_matrices.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
