#!/usr/bin/env python3
"""Checkpointed W&B downloader for BayesOpt baseline sweeps.

Downloads run summary + step history from ``run_bo_baseline_wandb.py`` sweeps.
Output schema is aligned with ``download_wandb_simple_regret.py`` where columns
overlap, so histories can be merged at plot time.

Example:
  python scripts/download_wandb_bo_baseline.py \\
    --entity EfficientLLMEval \\
    --project GittinsBanditEval \\
    --sweep-id <bo_sweep_id> \\
    --dataset gsm8k \\
    --raw-dir outputs/wandb_downloads/gsm8k_bo \\
    --states finished \\
    --expected-grid-yaml scripts/config/GSM8KBOBaselineSweep.yml \\
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
    "selected_config_arm_id",
    "selection_phase",
    "acquisition_value",
    "step_idx",
    "iter_fit_s",
    "iter_score_s",
    "iter_total_s",
]

BASE_COLUMNS = [
    "run_id",
    "run_name",
    "state",
    "url",
    "dataset_tag_resolved",
    "dataset_tag",
    "bo_inputs",
    "benchmark_key",
    "mmlu_task",
    "mmlu_size_bucket",
    "matrix_seed",
    "run_seed",
    "experiment_variant",
    "policy_variant",
    "policy_family",
    "method_label",
    "acquisition",
    "cost_mode",
    "cost_scaling_factor",
    "n_configs",
    "n_examples",
    "eval_budget_fraction",
    "n_init",
    "n_steps",
    "dominant_dim",
    "n_init_budget_cap",
    "observation_noise",
]

SUMMARY_COLUMNS = BASE_COLUMNS + [
    "final_simple_regret",
    "best_seen_regret",
    "final_cum_eval",
    "final_cum_original_cost",
    "num_evaluated_configs",
    "num_batches",
    "total_wall_time_s",
    "total_fit_s",
    "total_score_s",
    "total_iter_logged_s",
    "bo_stop_cum_eval",
    "bo_stop_cum_original_cost",
    "bo_stop_index_value",
    "nominal_total_configs",
    "budget_original_cost",
    "total_brute_force_original_cost",
    "cost_aware_run",
]

HISTORY_COLUMNS = BASE_COLUMNS + HISTORY_KEYS

CONFIG_KEY_COLUMNS = [
    "bo_inputs",
    "run_seed",
    "experiment_variant",
]

FULL_CONFIG_DIAGNOSTIC_COLUMNS = [
    "dataset_tag",
    "bo_inputs",
    "benchmark_key",
    "run_seed",
    "experiment_variant",
    "policy_variant",
    "acquisition",
    "cost_mode",
    "eval_budget_fraction",
    "n_init",
    "n_steps",
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


def infer_matrix_seed(path_value: Any) -> str:
    m = re.search(r"seed(\d+)", str(path_value))
    return m.group(1) if m else ""


def infer_mmlu_task_from_bo_inputs(path_value: Any) -> str | None:
    if path_value is None:
        return None
    stem = Path(str(path_value)).stem
    if not stem:
        return None
    return stem.removesuffix("_bo_inputs").removesuffix("_bo") or None


def infer_benchmark_key(
    *,
    dataset: str | None,
    bo_inputs: Any,
    mmlu_task: str | None,
    matrix_seed: Any,
) -> str:
    ds = normalize_dataset_name(dataset) or "unknown"
    if ds == "mmlu":
        task = mmlu_task or infer_mmlu_task_from_bo_inputs(bo_inputs)
        return f"mmlu_{safe_token(task)}" if task else f"mmlu_{safe_token(Path(str(bo_inputs)).stem)}"
    seed = normalize_scalar(matrix_seed) or infer_matrix_seed(bo_inputs)
    return f"{ds}_seed{seed or 'NA'}"


def parse_bo_variant(experiment_variant: str | None, acquisition: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"policy_family": "bo", "policy_variant": experiment_variant}
    if not experiment_variant:
        return out
    v = str(experiment_variant).strip()
    acq = v
    cost_mode = "unit"
    if v.endswith("_cost_aware"):
        acq = v[: -len("_cost_aware")]
        cost_mode = "cost"
    elif v.endswith("_cost"):
        acq = v[: -len("_cost")]
        cost_mode = "cost"
    elif v.endswith("_unit"):
        acq = v[: -len("_unit")]
        cost_mode = "unit"
    if acq == "logeipc":
        cost_mode = "cost"
    if acquisition:
        acq = str(acquisition)
    out.update(acquisition=acq, cost_mode=cost_mode, policy_variant=v)
    return out


def method_label_from_variant(experiment_variant: str | None, acquisition: str | None = None) -> str:
    parsed = parse_bo_variant(experiment_variant, acquisition)
    acq = str(parsed.get("acquisition") or "bo").lower()
    cost_mode = str(parsed.get("cost_mode") or "unit")
    labels = {
        ("pbgi", "unit"): "BO PBGI",
        ("pbgi", "cost"): "BO PBGI (cost)",
        ("logei", "unit"): "BO LogEI",
        ("logeipc", "cost"): "BO LogEIPC",
    }
    return labels.get((acq, cost_mode), f"BO {acq}" + (" (cost)" if cost_mode == "cost" else ""))


def get_field(cfg: dict[str, Any], summary: dict[str, Any], key: str) -> Any:
    if key in summary and summary[key] is not None:
        return summary[key]
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return None


def infer_dataset_from_row(row: dict[str, Any]) -> str | None:
    for key in ("dataset_tag_resolved", "dataset_tag"):
        ds = normalize_dataset_name(row.get(key))
        if ds:
            return ds
    bo_inputs = str(row.get("bo_inputs") or "").lower()
    if "gsm8k" in bo_inputs:
        return "gsm8k"
    if "piqa" in bo_inputs:
        return "piqa"
    if "mmlu" in bo_inputs or row.get("mmlu_task"):
        return "mmlu"
    return None


def base_row(run: wandb.apis.public.Run) -> dict[str, Any]:
    cfg = dict(run.config)
    summary = dict(run.summary)
    exp = get_field(cfg, summary, "experiment_variant") or get_field(cfg, summary, "policy_variant")
    acquisition = get_field(cfg, summary, "acquisition")
    parsed = parse_bo_variant(exp, acquisition)
    bo_inputs = get_field(cfg, summary, "bo_inputs")
    dataset = normalize_dataset_name(get_field(cfg, summary, "dataset_tag_resolved")) or normalize_dataset_name(
        get_field(cfg, summary, "dataset_tag")
    )
    mmlu_task = get_field(cfg, summary, "mmlu_task") or infer_mmlu_task_from_bo_inputs(bo_inputs)
    matrix_seed = get_field(cfg, summary, "matrix_seed") or infer_matrix_seed(bo_inputs)
    benchmark_key = get_field(cfg, summary, "benchmark_key") or infer_benchmark_key(
        dataset=dataset,
        bo_inputs=bo_inputs,
        mmlu_task=mmlu_task,
        matrix_seed=matrix_seed,
    )
    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "dataset_tag_resolved": get_field(cfg, summary, "dataset_tag_resolved"),
        "dataset_tag": get_field(cfg, summary, "dataset_tag"),
        "bo_inputs": bo_inputs,
        "benchmark_key": benchmark_key,
        "mmlu_task": mmlu_task,
        "mmlu_size_bucket": get_field(cfg, summary, "mmlu_size_bucket"),
        "matrix_seed": matrix_seed,
        "run_seed": get_field(cfg, summary, "run_seed"),
        "experiment_variant": exp,
        "policy_variant": get_field(cfg, summary, "policy_variant") or parsed.get("policy_variant"),
        "policy_family": get_field(cfg, summary, "policy_family") or "bo",
        "method_label": get_field(cfg, summary, "method_label") or method_label_from_variant(exp, acquisition),
        "acquisition": parsed.get("acquisition"),
        "cost_mode": get_field(cfg, summary, "cost_mode") or parsed.get("cost_mode"),
        "cost_scaling_factor": get_field(cfg, summary, "cost_scaling_factor"),
        "n_configs": get_field(cfg, summary, "n_configs"),
        "n_examples": get_field(cfg, summary, "n_examples"),
        "eval_budget_fraction": get_field(cfg, summary, "eval_budget_fraction"),
        "n_init": get_field(cfg, summary, "n_init"),
        "n_steps": get_field(cfg, summary, "n_steps"),
        "dominant_dim": get_field(cfg, summary, "dominant_dim"),
        "n_init_budget_cap": get_field(cfg, summary, "n_init_budget_cap"),
        "observation_noise": get_field(cfg, summary, "observation_noise"),
    }


def load_mmlu_task_buckets(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for t in data.get("tasks", []):
        task = str(t.get("task", "")).strip()
        bucket = str(t.get("size_bucket", "")).strip().lower()
        if task and bucket:
            out[task] = bucket
    return out


def get_mmlu_task(row: dict[str, Any]) -> str | None:
    value = row.get("mmlu_task")
    if value is not None and str(value).strip():
        return str(value).strip()
    return infer_mmlu_task_from_bo_inputs(row.get("bo_inputs"))


def infer_mmlu_size_bucket(row: dict[str, Any], task_buckets: dict[str, str]) -> str | None:
    value = row.get("mmlu_size_bucket")
    if value is not None and str(value).strip():
        return str(value).strip().lower()
    task = get_mmlu_task(row)
    if task and task in task_buckets:
        return task_buckets[task]
    return None


def keep_state(run_state: str, states_arg: str) -> bool:
    if states_arg.lower() == "all":
        return True
    allowed = {s.strip() for s in states_arg.split(",") if s.strip()}
    return str(run_state) in allowed


def keep_dataset(row: dict[str, Any], dataset_arg: str) -> bool:
    if dataset_arg.lower() == "auto":
        return True
    return infer_dataset_from_row(row) == dataset_arg.lower()


def keep_mmlu_size(row: dict[str, Any], size_arg: str, task_buckets: dict[str, str]) -> bool:
    if size_arg.lower() == "all":
        return True
    return infer_mmlu_size_bucket(row, task_buckets) == size_arg.lower()


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
        normalize_scalar(row.get("bo_inputs")),
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
    return {
        "candidate_rows_before_run_id_dedupe": len(candidate_rows),
        "unique_run_ids_after_dedupe": len(unique_rows),
        "duplicate_run_id_count": len(duplicate_run_ids),
    }


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
            return [dict(h) for h in run.scan_history(keys=HISTORY_KEYS, page_size=page_size)]
        except Exception as e:
            last_err = e
            time.sleep(float(sleep_s) * attempt)
    raise RuntimeError(f"scan_history failed for run={run.id} after {retries} retries: {last_err!r}")


def build_manifest(args: argparse.Namespace, manifest_path: Path, task_buckets: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api = wandb.Api()
    runs = api.runs(path=f"{args.entity}/{args.project}", filters={"sweep": args.sweep_id})
    candidate_rows: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for run in runs:
        if not keep_state(run.state, args.states):
            continue
        row = base_row(run)
        if not keep_dataset(row, args.dataset):
            continue
        if infer_dataset_from_row(row) == "mmlu" and not keep_mmlu_size(row, args.mmlu_size_bucket, task_buckets):
            continue
        if not row.get("mmlu_size_bucket"):
            bucket = infer_mmlu_size_bucket(row, task_buckets)
            if bucket:
                row["mmlu_size_bucket"] = bucket
        row_for_manifest = {k: row.get(k) for k in BASE_COLUMNS}
        candidate_rows.append(row_for_manifest)
        run_id = str(row_for_manifest.get("run_id", ""))
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        unique_rows.append(row_for_manifest)
    diagnostics = write_manifest_diagnostics(manifest_path.parent, candidate_rows, unique_rows)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        writer.writerows(unique_rows)
    print(f"Manifest written: {manifest_path} ({len(unique_rows)} unique runs)")
    return unique_rows, diagnostics


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return read_csv_rows(path)


def load_completed_runs(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
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
            if run_id:
                completed.add(run_id)
    return completed


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
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = data.get("parameters", {}) if isinstance(data, dict) else {}
    bo_inputs_list = _yaml_param_values(params, "bo_inputs")
    run_seeds = _yaml_param_values(params, "run_seed") or _yaml_param_values(params, "seed")
    variants = _yaml_param_values(params, "experiment_variant") or _yaml_param_values(params, "policy_variant")
    if not bo_inputs_list or not run_seeds or not variants:
        raise ValueError(
            "Expected YAML must contain parameters.bo_inputs.values, "
            "parameters.run_seed.values, and parameters.experiment_variant.values"
        )
    rows: list[dict[str, Any]] = []
    for bo_inputs in bo_inputs_list:
        bo_inputs = str(bo_inputs)
        ds = normalize_dataset_name(dataset) or infer_dataset_from_row({"bo_inputs": bo_inputs})
        mmlu_task = infer_mmlu_task_from_bo_inputs(bo_inputs)
        matrix_seed = infer_matrix_seed(bo_inputs)
        benchmark_key = infer_benchmark_key(
            dataset=ds,
            bo_inputs=bo_inputs,
            mmlu_task=mmlu_task,
            matrix_seed=matrix_seed,
        )
        for run_seed in run_seeds:
            for variant in variants:
                rows.append(
                    {
                        "dataset_tag": ds,
                        "bo_inputs": bo_inputs,
                        "benchmark_key": benchmark_key,
                        "mmlu_task": mmlu_task,
                        "matrix_seed": matrix_seed,
                        "run_seed": normalize_scalar(run_seed),
                        "experiment_variant": str(variant),
                    }
                )
    return rows


def expected_run_name(dataset: str, row: dict[str, Any]) -> str:
    ds = normalize_dataset_name(dataset) or normalize_dataset_name(row.get("dataset_tag")) or str(dataset)
    variant = normalize_scalar(row.get("experiment_variant"))
    run_seed = normalize_scalar(row.get("run_seed"))
    if ds == "mmlu":
        task = get_mmlu_task(row) or "unknown"
        label = f"task{safe_token(task)}"
    else:
        label = f"seed{normalize_scalar(row.get('matrix_seed')) or infer_matrix_seed(row.get('bo_inputs')) or 'NA'}"
    return f"{ds}_{label}_{variant}_runseed{run_seed}"


def summary_existing_keys(summary_path: Path) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for row in read_csv_rows(summary_path):
        keys.add(config_key(row))
    return keys


def write_missing_expected_configs(args: argparse.Namespace, expected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = summary_existing_keys(args.raw_dir / "runs_summary.csv")
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in expected_rows:
        key = config_key(row)
        if key in seen:
            continue
        seen.add(key)
        if key in existing:
            continue
        out = dict(row)
        out["expected_name"] = expected_run_name(args.dataset, row)
        missing.append(out)
    write_csv_rows(
        args.raw_dir / "missing_expected_configs.csv",
        ["dataset_tag", "bo_inputs", "benchmark_key", "run_seed", "experiment_variant", "expected_name"],
        missing,
    )
    print(f"Completeness check: expected={len(seen)}, downloaded={len(existing)}, missing={len(missing)}")
    return missing


def _query_runs(api: wandb.Api, entity: str, project: str, filters: dict[str, Any]) -> list[wandb.apis.public.Run]:
    return list(api.runs(path=f"{entity}/{project}", filters=filters))


def find_target_runs_within_sweep(args: argparse.Namespace, missing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    api = wandb.Api()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for target in missing_rows:
        bo_inputs = normalize_scalar(target.get("bo_inputs"))
        run_seed = int(target["run_seed"]) if str(target.get("run_seed", "")).isdigit() else target.get("run_seed")
        variant = normalize_scalar(target.get("experiment_variant"))
        expected_name = target.get("expected_name") or expected_run_name(args.dataset, target)
        candidates: dict[str, wandb.apis.public.Run] = {}
        for filters in (
            {"sweep": args.sweep_id, "config.bo_inputs": bo_inputs, "config.run_seed": run_seed, "config.experiment_variant": variant},
            {"sweep": args.sweep_id, "display_name": expected_name},
            {"sweep": args.sweep_id, "name": expected_name},
        ):
            try:
                for run in _query_runs(api, args.entity, args.project, filters):
                    candidates[run.id] = run
            except Exception:
                continue
        valid = []
        for run in candidates.values():
            if not keep_state(run.state, args.states):
                continue
            row = base_row(run)
            if not keep_dataset(row, args.dataset):
                continue
            valid.append((run, row))
        if not valid:
            continue
        chosen_run, chosen_row = valid[0]
        for run, row in valid:
            if str(run.name).strip() == str(expected_name).strip():
                chosen_run, chosen_row = run, row
                break
        if chosen_run.id not in selected_ids:
            selected_ids.add(chosen_run.id)
            selected.append({k: chosen_row.get(k) for k in BASE_COLUMNS})
    return selected


def download_rows(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    completed_path: Path,
    completed_ids: set[str],
    task_buckets: dict[str, str],
    label: str,
) -> dict[str, Any]:
    if not rows:
        return {"label": label, "new_runs": 0, "new_history_rows": 0}

    api = wandb.Api()
    writers = DownloadWriters(args.raw_dir, args.out_root, no_history=bool(args.no_history), split_mmlu_by_task=bool(args.split_mmlu_by_task))
    n_appended = 0
    n_history_rows = 0
    try:
        with completed_path.open("a", encoding="utf-8") as completed_f:
            for i, manifest_row in enumerate(rows, start=1):
                run_id = str(manifest_row.get("run_id", ""))
                if not run_id or run_id in completed_ids:
                    continue
                run = api.run(f"{args.entity}/{args.project}/{run_id}")
                row = base_row(run)
                if not keep_dataset(row, args.dataset):
                    continue
                if infer_dataset_from_row(row) == "mmlu" and not keep_mmlu_size(row, args.mmlu_size_bucket, task_buckets):
                    continue
                if not row.get("mmlu_size_bucket"):
                    bucket = infer_mmlu_size_bucket(row, task_buckets)
                    if bucket:
                        row["mmlu_size_bucket"] = bucket
                summary = dict(run.summary)
                hist = [] if args.no_history else scan_history_with_retry(
                    run, page_size=int(args.page_size), retries=int(args.retries), sleep_s=float(args.sleep)
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
                task = get_mmlu_task(row) if infer_dataset_from_row(row) == "mmlu" else None
                writers.write_run(task=task, summary_row=summary_row, history_rows=history_rows)
                completed_f.write(json.dumps({"run_id": run_id, "label": label, "history_rows": len(history_rows)}) + "\n")
                completed_f.flush()
                completed_ids.add(run_id)
                n_appended += 1
                n_history_rows += len(history_rows)
                if n_appended % int(args.print_every) == 0:
                    print(f"[{label}] downloaded {n_appended} runs", flush=True)
    finally:
        writers.close()
    return {"label": label, "new_runs": n_appended, "new_history_rows": n_history_rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--sweep-id", required=True)
    p.add_argument("--dataset", choices=["auto", "gsm8k", "piqa", "mmlu"], default="auto")
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
    p.add_argument("--expected-grid-yaml", type=Path, default=None)
    p.add_argument("--recover-missing-targeted", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    if args.split_mmlu_by_task and args.dataset != "mmlu":
        raise SystemExit("--split-mmlu-by-task requires --dataset mmlu")
    if args.mmlu_size_bucket != "all" and args.dataset != "mmlu":
        raise SystemExit("--mmlu-size-bucket requires --dataset mmlu")

    manifest_path = args.raw_dir / "wandb_run_manifest.csv"
    completed_path = args.raw_dir / "completed_runs.jsonl"
    task_buckets = load_mmlu_task_buckets(args.task_metadata)

    if args.refresh_manifest or not manifest_path.is_file():
        manifest_rows, _ = build_manifest(args, manifest_path, task_buckets)
    else:
        manifest_rows = load_manifest(manifest_path)
        print(f"Loaded manifest: {manifest_path} ({len(manifest_rows)} rows)")

    completed_ids = load_completed_runs(completed_path)
    incomplete = [r for r in manifest_rows if str(r.get("run_id", "")) not in completed_ids]
    if incomplete:
        result = download_rows(
            args=args,
            rows=incomplete,
            completed_path=completed_path,
            completed_ids=completed_ids,
            task_buckets=task_buckets,
            label="manifest",
        )
        print(json.dumps(result, indent=2))
    else:
        print("All manifest runs already downloaded.")

    if args.expected_grid_yaml is not None:
        expected_rows = load_expected_rows_from_sweep_yaml(args.expected_grid_yaml, args.dataset)
        missing_rows = write_missing_expected_configs(args, expected_rows)
        if args.recover_missing_targeted and missing_rows:
            target_rows = find_target_runs_within_sweep(args, missing_rows)
            target_rows = [r for r in target_rows if str(r.get("run_id", "")) not in completed_ids]
            if target_rows:
                recovery = download_rows(
                    args=args,
                    rows=target_rows,
                    completed_path=completed_path,
                    completed_ids=completed_ids,
                    task_buckets=task_buckets,
                    label="targeted_recovery",
                )
                print(json.dumps(recovery, indent=2))
            write_missing_expected_configs(args, expected_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
