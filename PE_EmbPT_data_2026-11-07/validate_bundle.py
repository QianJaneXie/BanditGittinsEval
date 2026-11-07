"""Check bundle integrity and independently recompute the BAI checkpoint tables.

Uses only the Python standard library. Run from anywhere: python validate_bundle.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ("logistic_39", "rasch39_random", "rasch39_balanced")
CONTRASTS = {
    "rasch_random_minus_logistic": ("rasch39_random", "logistic_39"),
    "rasch_balanced_minus_random": ("rasch39_balanced", "rasch39_random"),
    "rasch_balanced_minus_logistic": ("rasch39_balanced", "logistic_39"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), f"{a} != {b}")


def stats(values):
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def check_stat(actual, expected):
    if expected is None:
        require(actual == "", f"Expected blank, got {actual}")
    else:
        close(actual, expected)


def main():
    manifest = rows("FILE_MANIFEST.csv")
    listed = set()
    for entry in manifest:
        relative = entry["path"]
        target = (ROOT / relative).resolve()
        require(target.is_relative_to(ROOT), f"Path outside bundle: {relative}")
        require(relative not in listed, f"Duplicate manifest entry: {relative}")
        listed.add(relative)
        data = target.read_bytes()
        require(len(data) == int(entry["bytes"]), f"Size mismatch: {relative}")
        require(hashlib.sha256(data).hexdigest() == entry["sha256"], f"Hash mismatch: {relative}")

    metadata = {r["subject"]: r for r in rows("supplementary/subject_metadata.csv")}
    require(len(metadata) == 57, "Expected 57 subjects")
    require(sum(int(r["n_examples"]) for r in metadata.values()) == 14042, "Question count")
    traces = defaultdict(list)
    raw = rows("supplementary/pe_seed_traces.csv")
    require(len(raw) == 17100, "Expected 17,100 phase records")
    active = (1500, 750, 375, 187, 93)
    for row in raw:
        subject, variant = row["subject"], row["variant"]
        seed, phase = int(row["seed"]), int(row["phase"])
        require(variant in VARIANTS and 0 <= seed < 20 and 0 <= phase < 5, "Invalid run index")
        meta = metadata[subject]
        close(row["exhaustive_obs"], int(meta["n_examples"]) * 1500)
        close(row["exhaustive_cost"], meta["exhaustive_cost"])
        close(row["budget_obs_percent"], 2 * (phase + 1))
        for field in ("obs", "cost"):
            close(row[f"budget_{field}_percent"], 100 * float(row[f"budget_{field}"]) / float(row[f"exhaustive_{field}"]))
        require(int(row["n_active"]) == active[phase], "Active arm count")
        require(int(row["chosen_arm"]) == 100 * int(row["model_index"]) + int(row["prompt_index"]), "Arm index")
        require(0 <= int(row["chosen_arm"]) < 1500 and 0 <= int(row["prompt_index"]) < 100, "Arm range")
        close(row["simple_regret"], float(row["oracle_mean"]) - float(row["chosen_mean"]))
        require(0 <= float(row["simple_regret"]) <= 1, "Regret range")
        traces[subject, variant, seed].append(row)
    require(len(traces) == 57 * 3 * 20, "Trajectory count")
    for trajectory in traces.values():
        trajectory.sort(key=lambda r: int(r["phase"]))
        require([int(r["phase"]) for r in trajectory] == list(range(5)), "Missing/duplicate phases")
        require(all(float(a["budget_cost"]) < float(b["budget_cost"]) for a, b in zip(trajectory, trajectory[1:])), "Cost must increase")

    def samples(subject, variant, mode, budget):
        field = "budget_obs_percent" if mode == "unit" else "budget_cost_percent"
        result = {}
        for seed in range(20):
            available = [r for r in traces[subject, variant, seed] if float(r[field]) <= budget + 1e-9]
            if available:
                result[seed] = float(available[-1]["simple_regret"])
        return result

    def check_tables(subject_path, aggregate_path, expected_subject_rows, expected_aggregate_rows):
        subject_rows = rows(subject_path)
        require(len(subject_rows) == expected_subject_rows, f"Row count: {subject_path}")
        grouped = defaultdict(list)
        for row in subject_rows:
            key = row["mode"], row["group"], float(row["x_percent_exhaustive"])
            by_variant = {v: samples(row["subject"], v, key[0], key[2]) for v in VARIANTS}
            recomputed = {}
            for variant, values in by_variant.items():
                mean, se = stats(list(values.values()))
                require(int(row[f"{variant}_contributing_seeds"]) == len(values), "Seed coverage")
                check_stat(row[f"{variant}_mean_regret"], mean)
                check_stat(row[f"{variant}_seed_se"], se)
                recomputed[variant] = mean, len(values)
            for contrast, (a, b) in CONTRASTS.items():
                shared = sorted(set(by_variant[a]) & set(by_variant[b]))
                delta = [by_variant[a][seed] - by_variant[b][seed] for seed in shared]
                mean, se = stats(delta)
                require(int(row[f"{contrast}_paired_seeds"]) == len(shared), "Paired coverage")
                check_stat(row[contrast], mean)
                check_stat(row[f"{contrast}_seed_se"], se)
                recomputed[contrast] = mean, len(shared)
            grouped[key].append(recomputed)
        aggregate_rows = rows(aggregate_path)
        require(len(aggregate_rows) == expected_aggregate_rows, f"Row count: {aggregate_path}")
        for row in aggregate_rows:
            values = grouped[row["mode"], row["group"], float(row["x_percent_exhaustive"])]
            require(len(values) == int(row["cohort_subjects"]), "Subject coverage")
            for method in (*VARIANTS, *CONTRASTS):
                complete = [v[method][0] for v in values if v[method][1] == 20]
                require(len(complete) == int(row[f"{method}_full_coverage_subjects"]), "Full coverage")
                mean, se = stats(complete) if len(complete) == len(values) else (None, None)
                check_stat(row[f"{method}_mean"], mean)
                check_stat(row[f"{method}_subject_se"], se)

    check_tables("data/subject_checkpoints.csv", "data/checkpoints.csv", 190, 40)
    check_tables("data/all57_bai_subject_checkpoints.csv", "data/all57_bai_checkpoints.csv", 570, 10)
    curves = rows("data/plotted_curves.csv")
    require(len(curves) == 14088, "Plotted row count")
    gittins_paths = sorted((ROOT / "supplementary/gittins/per_task").glob("*_curves.csv"))
    require(len(gittins_paths) == 38, "Gittins subject/mode count")
    for path in gittins_paths:
        values = rows(path.relative_to(ROOT))
        require(len(values) == 640, f"Gittins row count: {path.name}")
        require({r["method_kind"] for r in values} == {"gittins_s", "gittins_g"}, "Gittins method filter")
    require(len(rows("supplementary/gittins/runs_summary.csv")) == 1520, "Gittins run count")
    require(len(rows("supplementary/gittins/runs_stopping.csv")) == 1520, "Gittins stopping count")
    print(json.dumps({"status": "passed", "manifest_files_verified": len(manifest),
                      "bai_subjects": len(metadata), "bai_trajectories": len(traces),
                      "phase_records": len(raw), "subject_checkpoint_rows_recomputed": 760,
                      "aggregate_checkpoint_rows_recomputed": 50, "plotted_points": len(curves),
                      "gittins_cache_files": len(gittins_paths), "gittins_runs": 1520}, indent=2))


if __name__ == "__main__":
    main()
