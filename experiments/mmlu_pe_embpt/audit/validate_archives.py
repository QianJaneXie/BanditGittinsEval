"""Independently audit complete interaction sweep archives and phase records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np


REPO = Path(__file__).resolve().parents[3]
MATRICES = REPO / "data" / "MMLU_matrices"
PROMPT_DATA = REPO / "prompteval" / "data"
CONCAT_PACKAGE = REPO / "PE_EmbPT_data_2026-11-07"
PRICE_PATH = REPO / "data_analysis" / "pricing" / "mmlu_prompt_eval_configurations_input_price.json"
DEFAULT_RESULTS = REPO / "outputs" / "mmlu_rasch389" / "results"
VARIANTS = ("rasch389_random", "rasch389_balanced")
REFERENCE_VARIANT = dict(zip(VARIANTS, ("rasch39_random", "rasch39_balanced"), strict=True))
SEEDS = tuple(range(20))
ACTIVE_COUNTS = (1500, 750, 375, 187, 93)
FIELDS = {
    "phase", "budget_obs", "budget_cost", "budget", "chosen_arm",
    "chosen_mean", "oracle_mean", "simple_regret", "n_active",
}
EXPECTED_SOURCES = {
    "experiments/mmlu_pe_embpt/interaction.py", "experiments/mmlu_pe_embpt/run_interaction.py",
    "prompteval/bai_evaluation.py", "prompteval/methods.py", "prompteval/methods_gpu.py", "prompteval/utils.py",
}
EXPECTED_INPUTS = {
    "prompteval/data/Ys.pickle", "prompteval/data/Xs.pickle", PRICE_PATH.relative_to(REPO).as_posix(),
}


def require(valid: bool, label: str) -> None:
    if not valid:
        raise ValueError(label)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def feature_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype.str}:{value.shape}".encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_responses(name: str) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated", category=DeprecationWarning)
        with (PROMPT_DATA / name).open("rb") as handle:
            return pickle.load(handle)["MMLU"]


def concat_phase_one() -> dict:
    path = CONCAT_PACKAGE / "supplementary" / "pe_seed_traces.csv"
    expected = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] == "0" and row["variant"] in REFERENCE_VARIANT.values():
                key = row["subject"], int(row["seed"]), row["variant"]
                require(key not in expected, f"Duplicate reference first-phase record: {key}")
                expected[key] = (float(row["budget_obs"]), float(row["budget_cost"]))
    return expected


def reconstruct_hashes(xs: list) -> tuple[str, str]:
    require(len(xs) == 15 and all(np.shape(model[1]) == (100, 25) for model in xs), "Unexpected cached prompt features.")
    embeddings = np.vstack([model[1] for model in xs])
    require(all(np.array_equal(xs[0][1], model[1]) for model in xs[1:]), "Prompt features differ across models.")
    model_ids = np.repeat(np.eye(15, dtype=np.float64), 100, axis=0)
    reference_offsets = model_ids[:, 1:]
    x39 = np.column_stack((embeddings, reference_offsets))
    interaction = np.where(model_ids[:, :, None] != 0, embeddings[:, None, :], 0.0).reshape(1500, 375)
    x389 = np.column_stack((interaction, reference_offsets))
    return feature_hash(x39), feature_hash(x389)


def verify_trajectory(updates, means, oracle, budget, min_price, max_price, label):
    require(len(updates) == len(ACTIVE_COUNTS), f"{label}: wrong phase count")
    previous_cost = 0.0
    previous_observations = 0
    for phase, (row, active) in enumerate(zip(updates, ACTIVE_COUNTS, strict=True)):
        require(set(row) == FIELDS and row["phase"] == phase, f"{label}: phase schema or order")
        observations = budget * (phase + 1) // len(ACTIVE_COUNTS)
        require(row["budget_obs"] == row["budget"] == observations and row["n_active"] == active,
                f"{label}: phase budget or active-arm count")
        cost = row["budget_cost"]
        cost_change = cost - previous_cost
        observation_change = observations - previous_observations
        require(np.isfinite(cost) and cost_change > 0, f"{label}: non-increasing spend")
        require(observation_change * min_price <= cost_change + 1e-8
                and cost_change <= observation_change * max_price + 1e-8,
                f"{label}: incremental spend lies outside per-arm price bounds")
        previous_cost, previous_observations = cost, observations
        chosen = row["chosen_arm"]
        require(type(chosen) is int and 0 <= chosen < len(means), f"{label}: invalid chosen arm")
        require(row["chosen_mean"] == means[chosen] and row["oracle_mean"] == oracle, f"{label}: incorrect arm means")
        require(row["simple_regret"] == oracle - means[chosen], f"{label}: incorrect simple regret")


def validate_subject(subject, results, xs, ys, reference, first_phase, source_hashes, input_hashes, prices):
    archive_path = results / f"bai_results_MMLU_{subject}_combined.npy"
    archive = np.load(archive_path, allow_pickle=True).item()
    checkpoint_root = results / "checkpoints" / subject
    manifest = load_json(checkpoint_root / "manifest.json")
    provenance = archive["reproduction"]
    fingerprint = manifest["fingerprint"]
    require(archive["jobs"] == [(subject, seed) for seed in SEEDS] and archive["seeds"] == list(SEEDS),
            f"{subject}: incomplete or unordered jobs")
    require(archive["tasks"] == [subject] and tuple(archive["feature_names"]) == VARIANTS,
            f"{subject}: wrong subject or variant labels")
    require(archive["bench"] == "MMLU" and archive["combine_models"] and archive["n_models_stacked"] == 15,
            f"{subject}: wrong benchmark or arm configuration")
    require(archive["costs_recorded"] and archive["cost_file"] == PRICE_PATH.relative_to(REPO).as_posix()
            and set(archive["update_fields"]) == FIELDS, f"{subject}: cost or phase schema")
    require(len(archive["out"]) == len(SEEDS) and archive["out_by_task"] == [archive["out"]],
            f"{subject}: incomplete or misnested archive")
    require(json_hash({k: v for k, v in manifest.items() if k != "fingerprint"}) == fingerprint,
            f"{subject}: wrong manifest fingerprint")
    embedded = {k: v for k, v in provenance.items() if k not in {"checkpoint_sha256", "convergence_warning_counts"}}
    require(json.loads(json.dumps(embedded)) == manifest, f"{subject}: archive and manifest differ")
    require(manifest["source_sha256"] == source_hashes and manifest["input_sha256"] == input_hashes,
            f"{subject}: current source or input file hashes differ from the run")

    matrix_path = MATRICES / f"{subject}.npy"
    matrix = np.vstack(ys)
    matrix_sha = file_hash(matrix_path)
    require(np.array_equal(matrix, np.load(matrix_path, allow_pickle=False)), f"{subject}: response rows differ from saved matrix")
    require(matrix_sha == manifest["matrix_sha256"] == reference["matrix_sha256"], f"{subject}: matrix provenance differs")
    questions = matrix.shape[1]
    budget = matrix.size // 10
    require(matrix.shape[0] == 1500 and list(matrix.shape) == manifest["matrix_shape"]
            and manifest["observation_budget"] == budget and manifest["budget_fraction"] == 0.1,
            f"{subject}: matrix or budget metadata")
    require(manifest["fitted_coefficients"] == 389 + questions and manifest["question_feature_width"] == questions
            and manifest["arm_feature_width"] == 389 and manifest["rank_min_explained_variance_ratio"] > 1e-6,
            f"{subject}: wrong dimensions or native rank check")
    require(manifest["phase_cap"] == len(ACTIVE_COUNTS) and manifest["first_version_input_match"],
            f"{subject}: interaction protocol does not match reference inputs")
    x39, x389 = reconstruct_hashes(xs)
    require(x39 == manifest["X39_sha256"] == reference["shared_X39_sha256"], f"{subject}: shared X39 hash differs")
    require(x389 == manifest["X389_sha256"], f"{subject}: reconstructed unscaled X389 hash differs")

    checkpoint_ids = {f"{seed}:{variant}" for seed in SEEDS for variant in VARIANTS}
    expected_names = {f"seed_{seed:02d}_{variant}.json" for seed in SEEDS for variant in VARIANTS}
    require({path.name for path in checkpoint_root.glob("seed_*.json")} == expected_names,
            f"{subject}: checkpoint file coverage")
    require(set(provenance["checkpoint_sha256"]) == set(provenance["convergence_warning_counts"]) == checkpoint_ids,
            f"{subject}: checkpoint provenance coverage")
    means, min_price, max_price = matrix.mean(axis=1), float(prices.min()), float(prices.max())
    oracle = means.max()
    convergence_count = 0
    for seed, record in enumerate(archive["out"]):
        require(len(record) == 1 and len(record[0]) == len(VARIANTS), f"{subject}, seed {seed}: wrong nesting")
        for variant, updates in zip(VARIANTS, record[0], strict=True):
            label = f"{subject}, seed {seed}, {variant}"
            path = checkpoint_root / f"seed_{seed:02d}_{variant}.json"
            checkpoint = load_json(path)
            checkpoint_id = f"{seed}:{variant}"
            require(file_hash(path) == provenance["checkpoint_sha256"][checkpoint_id], f"{label}: checkpoint file hash")
            require((checkpoint["subject"], checkpoint["seed"], checkpoint["variant"], checkpoint["fingerprint"])
                    == (subject, seed, variant, fingerprint), f"{label}: checkpoint identity")
            require(checkpoint["updates"] == updates and checkpoint["updates_sha256"] == json_hash({"updates": updates}),
                    f"{label}: phase records disagree with archive or hash")
            require(np.isfinite(checkpoint["elapsed_seconds"]) and checkpoint["elapsed_seconds"] > 0,
                    f"{label}: runtime record")
            messages = checkpoint["convergence_warnings"]
            require(isinstance(messages, list) and all(isinstance(message, str) for message in messages)
                    and len(messages) == provenance["convergence_warning_counts"][checkpoint_id],
                    f"{label}: convergence warning records disagree")
            convergence_count += len(messages)
            verify_trajectory(updates, means, oracle, budget, min_price, max_price, label)
            expected_observations, expected_cost = first_phase[subject, seed, REFERENCE_VARIANT[variant]]
            require(updates[0]["budget_obs"] == expected_observations and updates[0]["budget_cost"] == expected_cost,
                    f"{label}: first-phase spend differs from the same-sampler concatenation run")
    return file_hash(archive_path), convergence_count


def validate(results: Path) -> dict:
    subjects = sorted(path.stem for path in MATRICES.glob("*.npy"))
    observed = {path.name.removeprefix("bai_results_MMLU_").removesuffix("_combined.npy")
                for path in results.glob("bai_results_MMLU_*_combined.npy")}
    require(len(subjects) == 57 and observed == set(subjects),
            f"Expected the exact 57 archived subjects. Missing {sorted(set(subjects) - observed)}. Extra {sorted(observed - set(subjects))}.")
    xs, ys = load_responses("Xs.pickle"), load_responses("Ys.pickle")
    reference = load_json(CONCAT_PACKAGE / "provenance" / "subject_sources.json")["subjects"]
    first_phase = concat_phase_one()
    require(len(first_phase) == len(subjects) * len(SEEDS) * len(VARIANTS), "Incomplete reference first-phase records")
    price_data = load_json(PRICE_PATH)
    prices = np.array([price_data[str(arm)]["estimated_cost_per_1m_input_tokens"] for arm in range(1500)])
    require(len(price_data) == 1500 and np.isfinite(prices).all() and np.all(prices > 0), "Invalid input price vector")
    source_hashes = {name: file_hash(REPO / name) for name in sorted(EXPECTED_SOURCES)}
    input_hashes = {name: file_hash(REPO / name) for name in sorted(EXPECTED_INPUTS)}

    archive_hashes = {}
    warning_count = 0
    for subject in subjects:
        digest, warnings_found = validate_subject(subject, results, xs[subject], ys[subject], reference[subject],
                                                  first_phase, source_hashes, input_hashes, prices)
        archive_hashes[subject] = digest
        warning_count += warnings_found
    trajectories = len(subjects) * len(SEEDS) * len(VARIANTS)
    return {
        "status": "passed",
        "subjects": len(subjects),
        "trajectories": trajectories,
        "phase_records": trajectories * len(ACTIVE_COUNTS),
        "first_phase_observation_and_price_matches_to_concat": trajectories,
        "convergence_warnings": warning_count,
        "archive_set_sha256": json_hash(archive_hashes),
        "archive_sha256_by_subject": archive_hashes,
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, help="Optional JSON evidence file.")
    args = parser.parse_args()
    report = json.dumps(validate(args.results_dir), indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
