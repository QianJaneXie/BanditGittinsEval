"""Build the PE-EmbPT interaction report from the first bundle and completed Rasch389 runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from replot_interaction import METHODS, MODES, load_groups, replot


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEEDS = tuple(range(20))
INTERACTION = ("rasch389_random", "rasch389_balanced")
REFERENCE = ("logistic_39", "rasch39_random", "rasch39_balanced")
BAI_METHODS = (*REFERENCE, *INTERACTION)
GITTINS = ("gittins_s", "gittins_g")
PHASE_ARMS = (1500, 750, 375, 187, 93)
GRID = np.linspace(0, 10, 401)
BUDGET_TOLERANCE = 1e-9
PAIRS = (
    ("interaction_random_minus_concat_random", "rasch389_random", "rasch39_random"),
    ("interaction_balanced_minus_concat_balanced", "rasch389_balanced", "rasch39_balanced"),
    ("interaction_balanced_minus_interaction_random", "rasch389_balanced", "rasch389_random"),
    ("concat_balanced_minus_concat_random", "rasch39_balanced", "rasch39_random"),
    ("interaction_random_minus_logistic", "rasch389_random", "logistic_39"),
    ("interaction_balanced_minus_logistic", "rasch389_balanced", "logistic_39"),
)
SPEND_FIELDS = ("budget_obs", "budget_cost")
INTEGER_FIELDS = ("seed", "phase", "exhaustive_obs", "chosen_arm", "model_index", "prompt_index", "n_active")
FLOAT_FIELDS = (
    *SPEND_FIELDS, "exhaustive_cost", "budget_obs_percent", "budget_cost_percent",
    "chosen_mean", "oracle_mean", "simple_regret",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def subject_metadata(first_bundle: Path, groups: dict[str, list[str]]) -> dict[str, dict]:
    rows = read_csv(first_bundle / "supplementary" / "subject_metadata.csv")
    if len(rows) != 57 or len({row["subject"] for row in rows}) != 57:
        raise ValueError("Expected metadata for 57 distinct MMLU subjects.")
    membership = {subject: group for group, subjects in groups.items() for subject in subjects}
    metadata = {row["subject"]: row for row in rows}
    if not membership.keys() <= metadata.keys():
        raise ValueError("The first bundle is missing a Figure 3 subject.")
    for subject, row in metadata.items():
        questions = int(row["n_examples"])
        if (
            row["group"] != membership.get(subject, "")
            or int(row["n_arms"]) != 1500 or int(row["seed_count"]) != len(SEEDS)
            or int(row["exhaustive_obs"]) != 1500 * questions
            or int(row["X39_dimensions"]) != 39
            or not np.isfinite(float(row["exhaustive_cost"])) or float(row["exhaustive_cost"]) <= 0
        ):
            raise ValueError(f"Invalid subject metadata for {subject}.")
    return metadata


def load_reference(first_bundle: Path) -> list[dict]:
    raw = read_csv(first_bundle / "supplementary" / "pe_seed_traces.csv")
    return [
        row | {field: int(row[field]) for field in INTEGER_FIELDS}
        | {field: float(row[field]) for field in FLOAT_FIELDS}
        for row in raw
    ]


def load_interaction(results: Path, metadata: dict[str, dict]) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    sources = {}
    for subject, meta in sorted(metadata.items()):
        archive = results / f"bai_results_MMLU_{subject}_combined.npy"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing completed interaction archive for {subject}: {archive}")
        data = np.load(archive, allow_pickle=True).item()
        provenance = data["reproduction"]
        questions = int(meta["n_examples"])
        if (
            data["tasks"] != [subject]
            or data["jobs"] != [(subject, seed) for seed in SEEDS]
            or tuple(data["seeds"]) != SEEDS or len(data["out"]) != len(SEEDS)
            or tuple(data["feature_names"]) != INTERACTION
            or not data["combine_models"] or data["n_models_stacked"] != 15 or not data["costs_recorded"]
            or provenance["matrix_sha256"] != meta["matrix_sha256"]
            or provenance["X39_sha256"] != meta["shared_X39_sha256"]
            or provenance["arm_feature_width"] != 389 or provenance["question_feature_width"] != questions
            or provenance["fitted_coefficients"] != 389 + questions
        ):
            raise ValueError(f"The completed archive does not match the reference experiment: {archive}")
        sources[subject] = {
            "archive_sha256": digest(archive),
            "fingerprint": provenance["fingerprint"],
            "matrix_sha256": provenance["matrix_sha256"],
            "X39_sha256": provenance["X39_sha256"],
            "X389_sha256": provenance["X389_sha256"],
            "question_effects": questions,
            "fitted_coefficients": provenance["fitted_coefficients"],
        }
        cells, price = int(meta["exhaustive_obs"]), float(meta["exhaustive_cost"])
        for seed, run in zip(SEEDS, data["out"], strict=True):
            if len(run) != 1 or len(run[0]) != len(INTERACTION):
                raise ValueError(f"Expected both interaction variants for {subject}, seed {seed}.")
            for variant, phases in zip(INTERACTION, run[0], strict=True):
                for record in phases:
                    arm = record["chosen_arm"]
                    rows.append({
                        "subject": subject, "group": meta["group"], "variant": variant,
                        "seed": seed, "phase": record["phase"],
                        "budget_obs": record["budget_obs"], "budget_cost": record["budget_cost"],
                        "exhaustive_obs": cells, "exhaustive_cost": price,
                        "budget_obs_percent": 100 * record["budget_obs"] / cells,
                        "budget_cost_percent": 100 * record["budget_cost"] / price,
                        "chosen_arm": arm, "model_index": arm // 100, "prompt_index": arm % 100,
                        "chosen_mean": record["chosen_mean"], "oracle_mean": record["oracle_mean"],
                        "simple_regret": record["simple_regret"], "n_active": record["n_active"],
                    })
    return rows, sources


def index_traces(rows: list[dict], metadata: dict[str, dict]) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        key = row["subject"], row["variant"], row["seed"]
        buckets[key].append(row)
    expected = {(subject, method, seed) for subject in metadata for method in BAI_METHODS for seed in SEEDS}
    if set(buckets) != expected:
        raise ValueError("Expected all five BAI variants and seeds 0–19 for each of 57 subjects.")
    traces = {}
    for key, phases in buckets.items():
        subject, _, _ = key
        phases = sorted(phases, key=lambda record: record["phase"])
        meta = metadata[subject]
        cells, price = int(meta["exhaustive_obs"]), float(meta["exhaustive_cost"])
        observed = np.array([[row[field] for row in phases] for field in SPEND_FIELDS], dtype=float)
        regret = np.array([row["simple_regret"] for row in phases])
        denominators = np.array([cells, price])[:, None]
        if (
            [row["phase"] for row in phases] != list(range(5))
            or tuple(row["n_active"] for row in phases) != PHASE_ARMS
            or list(observed[0]) != [cells * phase // 50 for phase in range(1, 6)]
            or not np.isfinite(observed).all() or np.any(np.diff(observed, axis=1) <= 0)
            or np.any(observed <= 0)
            or not np.isfinite(regret).all() or np.any(regret < 0) or np.any(regret > 1)
            or any(
                row["group"] != meta["group"] or row["exhaustive_obs"] != cells or row["exhaustive_cost"] != price
                or row["simple_regret"] != row["oracle_mean"] - row["chosen_mean"]
                or not 0 <= row["chosen_mean"] <= row["oracle_mean"] <= 1
                or not 0 <= row["chosen_arm"] < 1500
                or (row["model_index"], row["prompt_index"]) != divmod(row["chosen_arm"], 100)
                for row in phases
            )
        ):
            raise ValueError(f"Invalid observation budget or regret records for {key}.")
        traces[key] = 100 * observed / denominators, regret
    for subject in metadata:
        for seed in SEEDS:
            for old, new in zip(REFERENCE[1:], INTERACTION, strict=True):
                if not np.array_equal(traces[subject, old, seed][0][:, 0], traces[subject, new, seed][0][:, 0]):
                    raise ValueError(f"The first-phase spend differs under the matched sampler for {subject}, seed {seed}.")
    return traces


def load_gittins(first_bundle: Path, groups: dict[str, list[str]], metadata: dict[str, dict]) -> dict:
    source = first_bundle / "supplementary" / "gittins" / "per_task"
    curves = {}
    for mode, _ in MODES:
        field = "exhaustive_obs" if mode == "unit" else "exhaustive_cost"
        x_label = "cum_eval" if mode == "unit" else "cum_original_cost"
        for subject in (task for tasks in groups.values() for task in tasks):
            rows = read_csv(source / f"{mode}_{subject}_curves.csv")
            for method in GITTINS:
                method_rows = [row for row in rows if row["method_kind"] == method]
                keep = np.flatnonzero([int(row["n_runs"]) == len(SEEDS) for row in method_rows])
                if (
                    len(method_rows) != 320 or len(keep) < 2 or np.any(np.diff(keep) != 1)
                    or any(row["mmlu_task"] != subject or row["mode"] != mode or row["x_label"] != x_label for row in method_rows)
                ):
                    raise ValueError(f"Inconsistent full-coverage Gittins curve for {mode}, {subject}, {method}.")
                x = np.array([float(method_rows[i]["x"]) for i in keep]) / float(metadata[subject][field]) * 100
                y = np.array([float(method_rows[i]["mean"]) for i in keep])
                if not np.isfinite([x, y]).all() or np.any(np.diff(x) <= 0) or np.any(y < 0) or np.any(y > 1):
                    raise ValueError(f"Invalid Gittins curve for {mode}, {subject}, {method}.")
                curves[mode, subject, method] = x, y
    return curves


def bai_at(traces, subject, method, mode_index, grid) -> np.ndarray:
    values = np.full((len(SEEDS), len(grid)), np.nan)
    for seed in SEEDS:
        spend, regret = traces[subject, method, seed]
        index = np.searchsorted(spend[mode_index], grid + BUDGET_TOLERANCE, side="right") - 1
        available = index >= 0
        values[seed, available] = regret[index[available]]
    return values


def mean_and_se(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, ...]:
    present = np.isfinite(values)
    count = present.sum(axis=axis)
    mean = np.full(count.shape, np.nan)
    np.divide(np.nansum(values, axis=axis), count, where=count > 0, out=mean)
    residual = np.where(present, values - np.expand_dims(mean, axis), 0.0)
    se = np.full(count.shape, np.nan)
    np.divide((residual * residual).sum(axis=axis), count * (count - 1), where=count > 1, out=se)
    return mean, np.sqrt(se), count


def interpolate(curve: tuple[np.ndarray, np.ndarray], grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, *curve, left=np.nan, right=np.nan)


def plotted_rows(groups, traces, gittins) -> list[dict]:
    rows = []
    for mode_index, (mode, _) in enumerate(MODES):
        for group, subjects in groups.items():
            for method in METHODS:
                if method in GITTINS:
                    source = [gittins[mode, subject, method] for subject in subjects]
                    start = max(x[0] for x, _ in source)
                    end = min(10.0, min(x[-1] for x, _ in source))
                    grid = np.r_[start, GRID[(GRID > start) & (GRID < end)], end]
                    values = np.array([interpolate(curve, grid) for curve in source])
                else:
                    seeds = np.array([bai_at(traces, subject, method, mode_index, GRID) for subject in subjects])
                    full = np.isfinite(seeds).all(axis=(0, 1))
                    grid = GRID[full]
                    values = seeds[:, :, full].mean(axis=1)
                if len(grid) < 2 or not np.isfinite(values).all():
                    raise ValueError(f"Missing common budget support for {mode}, {group}, {method}.")
                mean, se, _ = mean_and_se(values)
                rows.extend({
                    "mode": mode, "group": group, "method": method,
                    "x_percent_exhaustive": float(x), "mean_simple_regret": float(y),
                    "subject_se": float(error), "subjects": len(subjects), "seeds_per_subject": len(SEEDS),
                } for x, y, error in zip(grid, mean, se, strict=True))
    return rows


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def add_group_estimate(row: dict, method: str, values: np.ndarray, complete: np.ndarray) -> None:
    row[f"{method}_full_coverage_subjects"] = int(complete.sum())
    mean, se, _ = mean_and_se(values) if complete.all() else (np.nan, np.nan, 0)
    row[f"{method}_mean"] = finite(mean)
    row[f"{method}_subject_se"] = finite(se)


def checkpoints(groups, traces, gittins, grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    aggregate_rows, subject_rows = [], []
    for mode_index, (mode, _) in enumerate(MODES):
        for group, subjects in groups.items():
            raw = {
                method: np.array([bai_at(traces, subject, method, mode_index, grid) for subject in subjects])
                for method in BAI_METHODS
            }
            stats = {method: mean_and_se(array, axis=1) for method, array in raw.items()}
            contrasts = {label: mean_and_se(raw[left] - raw[right], axis=1) for label, left, right in PAIRS}
            from_cache = {
                method: np.array([interpolate(gittins[mode, subject, method], grid) for subject in subjects])
                for method in GITTINS
            } if gittins else {}
            for i, budget in enumerate(grid):
                base = {"mode": mode, "group": group, "x_percent_exhaustive": float(budget)}
                aggregate = base | {"cohort_subjects": len(subjects)}
                for method, array in from_cache.items():
                    add_group_estimate(aggregate, method, array[:, i], np.isfinite(array[:, i]))
                for method, (mean, _, count) in (stats | contrasts).items():
                    add_group_estimate(aggregate, method, mean[:, i], count[:, i] == len(SEEDS))
                aggregate_rows.append(aggregate)
                for j, subject in enumerate(subjects):
                    row = base | {"subject": subject}
                    for method, array in from_cache.items():
                        row[f"{method}_mean_regret"] = finite(array[j, i])
                        row[f"{method}_contributing_seeds"] = len(SEEDS) if np.isfinite(array[j, i]) else 0
                    for method, (mean, se, count) in stats.items():
                        row[f"{method}_mean_regret"] = finite(mean[j, i])
                        row[f"{method}_seed_se"] = finite(se[j, i])
                        row[f"{method}_contributing_seeds"] = int(count[j, i])
                    for label, (mean, se, count) in contrasts.items():
                        row[label] = finite(mean[j, i])
                        row[f"{label}_seed_se"] = finite(se[j, i])
                        row[f"{label}_paired_seeds"] = int(count[j, i])
                    subject_rows.append(row)
    return aggregate_rows, subject_rows


def copy_report_dependencies(first_bundle: Path, output: Path) -> None:
    copies = {
        "data/selected_tasks.csv": first_bundle / "data" / "selected_tasks.csv",
        "code/piecewise_regret_axis.py": first_bundle / "code" / "piecewise_regret_axis.py",
        "requirements.txt": first_bundle / "requirements.txt",
        "replot.py": HERE / "replot_interaction.py",
    }
    for name in ("pe_seed_traces.csv", "subject_metadata.csv", "model_index.csv"):
        copies[f"supplementary/reference/{name}"] = first_bundle / "supplementary" / name
    copies["supplementary/reference/mmlu_input_price_weights.json"] = first_bundle / "provenance" / "mmlu_input_price_weights.json"
    for relative, source in copies.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copytree(first_bundle / "supplementary" / "gittins", output / "supplementary" / "reference" / "gittins", dirs_exist_ok=True)


def build(first_bundle: Path, results: Path, output: Path) -> None:
    first_bundle, output = first_bundle.resolve(), output.resolve()
    if output.is_relative_to(first_bundle) or first_bundle.is_relative_to(output):
        raise ValueError("Write the interaction report to a separate location from the first bundle.")
    groups = load_groups(first_bundle / "data" / "selected_tasks.csv")
    metadata = subject_metadata(first_bundle, groups)
    new_rows, archive_sources = load_interaction(results, metadata)
    traces = index_traces(load_reference(first_bundle) + new_rows, metadata)
    gittins = load_gittins(first_bundle, groups, metadata)
    curves = plotted_rows(groups, traces, gittins)
    cohort, cohort_subjects = checkpoints(groups, traces, gittins, np.array([2, 4, 6, 8, 9.9]))
    full, full_subjects = checkpoints({"All 57 subjects": sorted(metadata)}, traces, {}, np.array([2, 4, 6, 8, 10]))

    copy_report_dependencies(first_bundle, output)
    data = output / "data"
    write_csv(data / "plotted_curves.csv", curves)
    write_csv(data / "checkpoints.csv", cohort)
    write_csv(data / "subject_checkpoints.csv", cohort_subjects)
    write_csv(data / "all57_bai_checkpoints.csv", full)
    write_csv(data / "all57_bai_subject_checkpoints.csv", full_subjects)
    write_csv(output / "supplementary" / "interaction_seed_traces.csv", new_rows)
    write_csv(output / "supplementary" / "subject_metadata.csv", [
        metadata[subject] | {
            "X389_dimensions": 389,
            "rasch389_fitted_coefficients": source["fitted_coefficients"],
            "X389_sha256": source["X389_sha256"],
        }
        for subject, source in archive_sources.items()
    ])
    reference = output / "supplementary" / "reference"
    provenance = {
        "comparison": "Joint MMLU prompt-model Rasch comparison between concatenated and interaction designs",
        "cohorts": groups, "seeds_per_subject_method": len(SEEDS),
        "bai_methods": BAI_METHODS, "interaction_phase_rows": len(new_rows),
        "unit_axis": "Actual observation count divided by exhaustive observation count",
        "price_axis": "Actual price-weighted spend divided by exhaustive price-weighted spend",
        "bai_aggregation": "Last observed phase at or before spend, 1e-9 percentage point tolerance, carry final recommendation forward",
        "gittins_aggregation": "Interpolate per-subject means within full-20-run support only",
        "uncertainty": "One standard error across equally weighted subject means; BAI contrasts pair seeds within each subject",
        "coverage": "Aggregate rows require all 20 seeds in every subject; subject rows retain partial counts",
        "reference_sha256": {path.relative_to(output).as_posix(): digest(path) for path in sorted(reference.rglob("*")) if path.is_file()},
        "interaction_archives": archive_sources,
    }
    (output / "supplementary" / "report_inputs.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    replot(output, output / "figure")
    print(f"Wrote {len(curves)} plotted points and {len(new_rows)} interaction phase records to {output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-bundle", type=Path, default=REPO / "PE_EmbPT_data_2026-11-07")
    parser.add_argument("--results-dir", type=Path, default=REPO / "outputs" / "mmlu_rasch389" / "results")
    parser.add_argument("--out-dir", type=Path, default=REPO / "PE_EmbPT_interaction_data_2026-11-07")
    args = parser.parse_args()
    build(args.first_bundle, args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
