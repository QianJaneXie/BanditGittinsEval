"""Independently check the portable interaction bundle and its report tables."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics


BAI = ("logistic_39", "rasch39_random", "rasch39_balanced", "rasch389_random", "rasch389_balanced")
GITTINS = ("gittins_s", "gittins_g")
MODES = ("unit", "aware")
SEEDS = tuple(range(20))
ACTIVE = (1500, 750, 375, 187, 93)
COHORT_COUNTS = {"Easy-Small": 6, "Easy-Large": 2, "Hard-Small": 9, "Hard-Large": 2}
CONTRASTS = {
    "interaction_random_minus_concat_random": ("rasch389_random", "rasch39_random"),
    "interaction_balanced_minus_concat_balanced": ("rasch389_balanced", "rasch39_balanced"),
    "interaction_balanced_minus_interaction_random": ("rasch389_balanced", "rasch389_random"),
    "concat_balanced_minus_concat_random": ("rasch39_balanced", "rasch39_random"),
    "interaction_random_minus_logistic": ("rasch389_random", "logistic_39"),
    "interaction_balanced_minus_logistic": ("rasch389_balanced", "logistic_39"),
}
REFERENCE = Path("supplementary/reference")
GITTINS_ROOT = REFERENCE / "gittins"
TOLERANCE = 1e-9  # Percentage points at BAI budget boundaries.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    if expected is None:
        require(actual == "", f"{label}: expected a blank, got {actual!r}")
    else:
        require(actual != "" and math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12),
                f"{label}: {actual!r} differs from {expected}")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def keyed(records, fields, label):
    result = {tuple(row[field] for field in fields): row for row in records}
    require(len(result) == len(records), f"Duplicate keys in {label}")
    return result


def stats(values):
    values = list(values)
    if not values:
        return None, None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return mean, se


def check_manifest(root):
    manifest = rows(root / "FILE_MANIFEST.csv")
    listed = set()
    for row in manifest:
        relative = row["path"]
        path = (root / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(root), f"Manifest path outside bundle: {relative}")
        require(relative not in listed and relative != "FILE_MANIFEST.csv", f"Invalid manifest entry: {relative}")
        listed.add(relative)
        require(path.stat().st_size == int(row["bytes"]), f"Size mismatch: {relative}")
        require(digest(path) == row["sha256"], f"Hash mismatch: {relative}")
    require(listed, "Empty file manifest")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "FILE_MANIFEST.csv"
        and path.relative_to(root).parts[0] != "reproduced"
        and "__pycache__" not in path.relative_to(root).parts
    }
    require(listed == actual, f"Manifest/file set differs: missing {sorted(listed - actual)}, unlisted {sorted(actual - listed)}")
    return len(listed)


def load_subjects(root):
    groups = defaultdict(list)
    for row in rows(root / "data/selected_tasks.csv"):
        groups[row["group"]].append(row["task"])
    require({group: len(members) for group, members in groups.items()} == COHORT_COUNTS, "Figure 3 cohort sizes")
    membership = {task: group for group, tasks in groups.items() for task in tasks}
    require(len(membership) == 19, "Duplicate Figure 3 subjects")

    metadata = {subject: row for (subject,), row in keyed(
        rows(root / "supplementary/subject_metadata.csv"), ("subject",), "subject metadata"
    ).items()}
    reference = {subject: row for (subject,), row in keyed(
        rows(root / REFERENCE / "subject_metadata.csv"), ("subject",), "reference metadata"
    ).items()}
    require(set(metadata) == set(reference) and len(metadata) == 57, "Expected 57 matched MMLU subjects")
    require(set(membership) <= metadata.keys(), "Missing Figure 3 subject metadata")
    require(sum(int(row["n_examples"]) for row in metadata.values()) == 14042, "Total question count")

    weights = read_json(root / REFERENCE / "mmlu_input_price_weights.json")
    require(set(weights) == {str(arm) for arm in range(1500)}, "Fixed price vector arm IDs")
    prices = [float(weights[str(arm)]["estimated_cost_per_1m_input_tokens"]) for arm in range(1500)]
    require(all(math.isfinite(price) and price > 0 for price in prices), "Fixed price vector values")
    for subject, row in metadata.items():
        questions = int(row["n_examples"])
        require(all(row[field] == value for field, value in reference[subject].items()), f"{subject}: reference metadata differs")
        require(row["group"] == membership.get(subject, ""), f"{subject}: wrong group")
        require(int(row["n_models"]) == 15 and int(row["prompts_per_model"]) == 100, f"{subject}: model/prompt dimensions")
        require(int(row["n_arms"]) == 1500 and int(row["X389_dimensions"]) == 389, f"{subject}: interaction dimensions")
        require(int(row["rasch389_fitted_coefficients"]) == 389 + questions, f"{subject}: question effect count")
        require(int(row["phase_count"]) == 5 and int(row["seed_count"]) == 20, f"{subject}: phase or seed count")
        close(row["exhaustive_obs"], 1500 * questions, subject)
        close(row["observation_budget"], 150 * questions, subject)
        close(row["exhaustive_cost"], math.fsum(prices) * questions, subject)
    return metadata, groups, min(prices), max(prices)


def check_report_provenance(root, metadata, groups):
    provenance = read_json(root / "supplementary/report_inputs.json")
    require(provenance["cohorts"] == groups, "Report cohort provenance")
    require(set(provenance["bai_methods"]) == set(BAI) and provenance["seeds_per_subject_method"] == 20,
            "Report method/seed provenance")
    expected = {
        path.relative_to(root).as_posix(): digest(path)
        for path in (root / REFERENCE).rglob("*")
        if path.is_file()
    }
    require(expected == provenance["reference_sha256"], "Report reference file hashes")
    archives = provenance["interaction_archives"]
    require(archives.keys() == metadata.keys(), "Interaction archive provenance coverage")
    for subject, row in metadata.items():
        archive = archives[subject]
        for field, key in (("matrix_sha256", "matrix_sha256"), ("X39_sha256", "shared_X39_sha256"), ("X389_sha256", "X389_sha256")):
            require(archive[field] == row[key], f"{subject}: {field} provenance")
        require(archive["question_effects"] == int(row["n_examples"])
                and archive["fitted_coefficients"] == int(row["rasch389_fitted_coefficients"]), f"{subject}: fit dimensions")


def load_traces(root, metadata, min_price, max_price):
    grouped = defaultdict(list)
    true_means, oracles = {}, {}
    inputs = (
        (REFERENCE / "pe_seed_traces.csv", BAI[:3], 17100),
        (Path("supplementary/interaction_seed_traces.csv"), BAI[3:], 11400),
    )
    for path, allowed, count in inputs:
        records = rows(root / path)
        require(len(records) == count, f"{path}: expected {count} phase records")
        for row in records:
            subject, method = row["subject"], row["variant"]
            seed, phase = int(row["seed"]), int(row["phase"])
            label = f"{subject}/{method}/{seed}/{phase}"
            require(subject in metadata and method in allowed and seed in SEEDS and phase in range(5), label)
            meta = metadata[subject]
            n = int(meta["n_examples"])
            require(row["group"] == meta["group"] and int(row["n_active"]) == ACTIVE[phase], label)
            close(row["exhaustive_obs"], 1500 * n, label)
            close(row["exhaustive_cost"], float(meta["exhaustive_cost"]), label)
            obs, cost = float(row["budget_obs"]), float(row["budget_cost"])
            close(obs, 30 * n * (phase + 1), label)
            close(row["budget_obs_percent"], 100 * obs / float(meta["exhaustive_obs"]), label)
            close(row["budget_cost_percent"], 100 * cost / float(meta["exhaustive_cost"]), label)
            require(min_price * obs - 1e-6 <= cost <= max_price * obs + 1e-6, f"{label}: impossible spend")
            arm, model, prompt = (int(row[field]) for field in ("chosen_arm", "model_index", "prompt_index"))
            require(0 <= model < 15 and 0 <= prompt < 100 and arm == model * 100 + prompt, f"{label}: arm index")
            chosen, oracle, regret = (float(row[field]) for field in ("chosen_mean", "oracle_mean", "simple_regret"))
            require(0 <= chosen <= oracle <= 1 and 0 <= regret <= 1, f"{label}: regret or mean range")
            close(regret, oracle - chosen, label)
            close(chosen * n, round(chosen * n), label)
            close(oracle * n, round(oracle * n), label)
            close(chosen, true_means.setdefault((subject, arm), chosen), label)
            close(oracle, oracles.setdefault(subject, oracle), label)
            grouped[subject, method, seed].append((phase, obs, cost, regret))

    expected = {(subject, method, seed) for subject in metadata for method in BAI for seed in SEEDS}
    require(set(grouped) == expected, "Missing BAI trajectories")
    traces = {}
    for key, values in grouped.items():
        values.sort()
        require([point[0] for point in values] == list(range(5)), f"{key}: missing or duplicate phases")
        for previous, current in zip(values, values[1:]):
            delta = current[2] - previous[2]
            cells = current[1] - previous[1]
            require(min_price * cells - 1e-6 <= delta <= max_price * cells + 1e-6, f"{key}: invalid spend increment")
        meta = metadata[key[0]]
        x = tuple(tuple(100 * point[i] / float(meta[field]) for point in values)
                  for i, field in ((1, "exhaustive_obs"), (2, "exhaustive_cost")))
        traces[key] = x, tuple(point[3] for point in values)
    for subject in metadata:
        for seed in SEEDS:
            for left, right in (("logistic_39", "rasch39_random"), *zip(BAI[1:3], BAI[3:], strict=True)):
                require(grouped[subject, left, seed][0][:3] == grouped[subject, right, seed][0][:3],
                        f"{subject}/{seed}: matched first-phase spend differs")
    return traces


def load_gittins(root, metadata, groups):
    expected = {(subject, mode, method) for subjects in groups.values() for subject in subjects
                for mode in MODES for method in GITTINS}
    run_rows = keyed(rows(root / GITTINS_ROOT / "runs_summary.csv"), ("run_id",), "Gittins summaries")
    stop_rows = keyed(rows(root / GITTINS_ROOT / "runs_stopping.csv"), ("run_id",), "Gittins stopping")
    require(len(run_rows) == 1520 and run_rows.keys() == stop_rows.keys(), "Gittins run/stop coverage")
    summaries = defaultdict(list)
    for run_id, row in run_rows.items():
        subject, mode, prior = row["mmlu_task"], row["cost_mode"], row["prior_type"]
        require(prior in {"dataset", "default"}, f"{run_id}: Gittins prior")
        method = "gittins_s" if prior == "dataset" else "gittins_g"
        label = subject, mode, method
        require(label in expected and row["matrix_sha256"] == metadata[subject]["matrix_sha256"], str(label))
        require(row["recommendation_rule"] == "lcb" and row["experiment_variant"].endswith("_lcb_new"), str(label))
        for field in ("mmlu_task", "cost_mode", "run_seed", "prior_type", "gittins_stop_cum_eval", "gittins_stop_cum_original_cost"):
            require(row[field] == stop_rows[run_id][field], f"{run_id}: stop metadata differs ({field})")
        summaries[label].append(row)
    require(set(summaries) == expected, "Gittins per-subject run coverage")
    curves = {}
    for subject, mode, method in sorted(expected):
        label = subject, mode, method
        runs = summaries[label]
        require(sorted(int(row["run_seed"]) for row in runs) == list(SEEDS), f"{label}: Gittins seed coverage")
        path = root / GITTINS_ROOT / "per_task" / f"{mode}_{subject}_curves.csv"
        source = [row for row in rows(path) if row["method_kind"] == method]
        x_label = "cum_eval" if mode == "unit" else "cum_original_cost"
        denom = "exhaustive_obs" if mode == "unit" else "exhaustive_cost"
        require(len(source) == 320, f"{label}: expected 320 cached rows")
        require(all((row["mmlu_task"], row["mode"], row["x_label"]) == (subject, mode, x_label) for row in source), str(label))
        x = [float(row["x"]) for row in source]
        y = [float(row["mean"]) for row in source]
        require(all(math.isfinite(v) for v in x) and all(a < b for a, b in zip(x, x[1:])), f"{label}: raw x grid")
        require(all(0 <= v <= 1 for v in y), f"{label}: regret range")
        indices = [i for i, row in enumerate(source) if int(row["n_runs"]) == 20]
        require(len(indices) >= 2 and indices == list(range(indices[0], indices[-1] + 1)), f"{label}: full-run support")
        final_x = max(float(run[f"final_{x_label}"]) for run in runs)
        ending = [float(run["final_simple_regret"]) for run in runs
                  if math.isclose(float(run[f"final_{x_label}"]), final_x, rel_tol=1e-12, abs_tol=1e-10)]
        close(x[-1], final_x, label)
        close(y[-1], statistics.fmean(ending), label)
        require(int(source[-1]["n_runs"]) == len(ending), f"{label}: terminal run count")
        curves[mode, subject, method] = (
            tuple(x[i] / float(metadata[subject][denom]) * 100 for i in indices),
            tuple(y[i] for i in indices),
        )
    return curves


def interpolate(curve, budget):
    x, y = curve
    if budget < x[0] or budget > x[-1]:
        return None
    i = bisect_right(x, budget) - 1
    if i == len(x) - 1:
        return y[i]
    return y[i] + (y[i + 1] - y[i]) * (budget - x[i]) / (x[i + 1] - x[i])


def sample_lookup(traces):
    @lru_cache(maxsize=None)
    def at(subject, method, mode, budget):
        values = {}
        for seed in SEEDS:
            x, y = traces[subject, method, seed]
            i = bisect_right(x[MODES.index(mode)], budget + TOLERANCE) - 1
            if i >= 0:
                values[seed] = y[i]
        return values

    return at


def check_checkpoints(root, basename, groups, budgets, at, gittins):
    prefix = "all57_bai_" if basename == "all57" else ""
    subject_path = root / "data" / f"{prefix}subject_checkpoints.csv"
    aggregate_path = root / "data" / f"{prefix}checkpoints.csv"
    subject_rows = keyed(rows(subject_path), ("mode", "group", "subject", "x_percent_exhaustive"), subject_path)
    aggregate_rows = keyed(rows(aggregate_path), ("mode", "group", "x_percent_exhaustive"), aggregate_path)
    for mode in MODES:
        for group, members in groups.items():
            for budget in budgets:
                key = mode, group, str(float(budget))
                aggregate = aggregate_rows.pop(key)
                require(int(aggregate["cohort_subjects"]) == len(members), str(key))
                complete = defaultdict(list)
                for subject in members:
                    row = subject_rows.pop((mode, group, subject, key[2]))
                    samples = {method: at(subject, method, mode, budget) for method in BAI}
                    for method, values in samples.items():
                        mean, se = stats(values.values())
                        close(row[f"{method}_mean_regret"], mean, key)
                        close(row[f"{method}_seed_se"], se, key)
                        require(int(row[f"{method}_contributing_seeds"]) == len(values), f"{key}: seed coverage")
                        if len(values) == 20:
                            complete[method].append(mean)
                    for label, (left, right) in CONTRASTS.items():
                        a, b = samples[left], samples[right]
                        shared = sorted(a.keys() & b.keys())
                        mean, se = stats(a[seed] - b[seed] for seed in shared)
                        close(row[label], mean, key)
                        close(row[f"{label}_seed_se"], se, key)
                        require(int(row[f"{label}_paired_seeds"]) == len(shared), f"{key}: paired seed coverage")
                        if len(shared) == 20:
                            complete[label].append(mean)
                    if gittins:
                        for method in GITTINS:
                            value = interpolate(gittins[mode, subject, method], budget)
                            close(row[f"{method}_mean_regret"], value, key)
                            require(int(row[f"{method}_contributing_seeds"]) == (20 if value is not None else 0), str(key))
                            if value is not None:
                                complete[method].append(value)
                for method in (*BAI, *CONTRASTS, *(GITTINS if gittins else ())):
                    values = complete[method]
                    require(int(aggregate[f"{method}_full_coverage_subjects"]) == len(values), f"{key}: full subject coverage")
                    mean, se = stats(values) if len(values) == len(members) else (None, None)
                    close(aggregate[f"{method}_mean"], mean, key)
                    close(aggregate[f"{method}_subject_se"], se, key)
    require(not subject_rows and not aggregate_rows, "Unexpected checkpoint rows")
    return 2 * sum(len(members) for members in groups.values()) * len(budgets), 2 * len(groups) * len(budgets)


def check_plot(root, groups, at, gittins):
    actual = defaultdict(list)
    for row in rows(root / "data/plotted_curves.csv"):
        actual[row["mode"], row["group"], row["method"]].append(row)
    count = 0
    grid = [i * (10.0 / 400) for i in range(401)]
    for mode in MODES:
        for group, subjects in groups.items():
            for method in (*GITTINS, *BAI):
                key = mode, group, method
                points = actual.pop(key)
                if method in GITTINS:
                    inputs = [gittins[mode, subject, method] for subject in subjects]
                    start = max(curve[0][0] for curve in inputs)
                    end = min(10.0, min(curve[0][-1] for curve in inputs))
                    budgets = [start, *(x for x in grid if start < x < end), end]
                    expected = [(x, [interpolate(curve, x) for curve in inputs]) for x in budgets]
                else:
                    expected = []
                    for x in grid:
                        samples = [at(subject, method, mode, x) for subject in subjects]
                        if all(len(sample) == 20 for sample in samples):
                            expected.append((x, [statistics.fmean(sample.values()) for sample in samples]))
                require(len(points) == len(expected), f"{key}: plotted support/count")
                for row, (x, values) in zip(points, expected, strict=True):
                    require(int(row["subjects"]) == len(subjects) and int(row["seeds_per_subject"]) == 20, str(key))
                    mean, se = stats(values)
                    close(row["x_percent_exhaustive"], x, key)
                    close(row["mean_simple_regret"], mean, key)
                    close(row["subject_se"], se, key)
                count += len(points)
    require(not actual, "Unexpected plotted curves")
    return count


def validate(root):
    manifest_count = check_manifest(root)
    metadata, groups, min_price, max_price = load_subjects(root)
    check_report_provenance(root, metadata, groups)
    traces = load_traces(root, metadata, min_price, max_price)
    gittins = load_gittins(root, metadata, groups)
    at = sample_lookup(traces)
    figure_subject_rows, figure_aggregate_rows = check_checkpoints(root, "figure", groups, (2, 4, 6, 8, 9.9), at, gittins)
    all_subject_rows, all_aggregate_rows = check_checkpoints(root, "all57", {"All 57 subjects": sorted(metadata)}, (2, 4, 6, 8, 10), at, {})
    plot_points = check_plot(root, groups, at, gittins)
    return {
        "status": "passed", "manifest_files_verified": manifest_count, "bai_subjects": len(metadata),
        "reference_phase_records": 17100, "interaction_phase_records": 11400, "bai_trajectories": len(traces),
        "subject_checkpoint_rows_recomputed": figure_subject_rows + all_subject_rows,
        "aggregate_checkpoint_rows_recomputed": figure_aggregate_rows + all_aggregate_rows,
        "plotted_points_recomputed": plot_points, "gittins_task_curves_verified": len(gittins), "gittins_runs": 1520,
    }


def main():
    here = Path(__file__).resolve().parent
    default = here if (here / "FILE_MANIFEST.csv").is_file() else here.parents[1] / "PE_EmbPT_interaction_data_2026-11-07"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=default)
    parser.add_argument("--report", type=Path, help="Optional JSON validation report outside the checked bundle.")
    args = parser.parse_args()
    root = args.bundle_dir.resolve()
    if args.report and args.report.resolve().is_relative_to(root):
        parser.error("--report must be outside the bundle so validation does not alter it.")
    output = json.dumps(validate(root), indent=2) + "\n"
    if args.report:
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
