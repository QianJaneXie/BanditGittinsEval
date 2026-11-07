"""Independently audit the Rasch39 sampling and archived MMLU results."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np


AUDIT_DIR = Path(__file__).resolve().parent
EXPERIMENT = AUDIT_DIR.parent
REPO = EXPERIMENT.parents[1]
MATRICES = REPO / "data" / "MMLU_matrices"
PROMPT_DATA = REPO / "prompteval" / "data"
BASELINE = EXPERIMENT.parent / "mmlu_feature_comparison"
VARIANTS = ("rasch39_random", "rasch39_balanced")
SEEDS = tuple(range(20))
FIELDS = {
    "phase", "budget_obs", "budget_cost", "budget", "chosen_arm",
    "chosen_mean", "oracle_mean", "simple_regret", "n_active",
}


def require(valid: bool, label: str) -> None:
    if not valid:
        raise ValueError(label)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}:{array.shape}".encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def load_data(name: str) -> dict:
    with (PROMPT_DATA / name).open("rb") as handle:
        return pickle.load(handle)["MMLU"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_unseen_labels(results_dir: Path) -> dict:
    spec = importlib.util.spec_from_file_location("independent_rasch_runner", EXPERIMENT / "run_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.initialize_worker()
    bai = runner.legacy.STATE["bai"]
    bai.MAX_BAI_PHASES = 1
    subject, seed = "college_chemistry", 0
    matrix, features, _, _ = runner.prepare_subject(subject, BASELINE / "results")
    budget = matrix.size // 50

    source = REPO / "prompteval" / "methods.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "StratSample")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    sample_native = namespace["StratSample"]
    original_fit = runner.RaschScore.fit
    masks, scores = [], []

    def capture_fit(self, seen, outcomes, design):
        original_fit(self, seen, outcomes, design)
        masks.append(seen.copy())
        scores.append(self.thetas.copy())

    runner.RaschScore.fit = capture_fit
    results = {}
    for variant in VARIANTS:
        expected = sample_native(
            np.zeros(matrix.shape, dtype=bool), budget, seed,
            list(range(matrix.shape[0])), variant == VARIANTS[0],
        )
        [original] = runner.replay_case(matrix, features, budget, seed, variant)
        mutated = matrix.copy()
        mutated[~expected] = 1 - mutated[~expected]
        [changed] = runner.replay_case(mutated, features, budget, seed, variant)
        require(len(masks) == 2 and all(np.array_equal(mask, expected) for mask in masks), f"{variant}: sampler mask differs")
        require(np.array_equal(scores[0], scores[1]), f"{variant}: unseen outcomes changed the fitted arm scores")
        require(all(original[key] == changed[key] for key in ("chosen_arm", "budget_obs", "budget_cost")), f"{variant}: unseen outcomes changed the recommendation or spend")
        require(original["oracle_mean"] != changed["oracle_mean"], f"{variant}: unseen-outcome intervention did not affect evaluation metrics")
        checkpoint = results_dir / "checkpoints" / subject / f"seed_{seed:02d}_{variant}.json"
        require(load_json(checkpoint)["updates"][0] == original, f"{variant}: first phase differs from checkpoint")
        results[variant] = {
            "native_first_phase_mask_exact": True,
            "first_phase_matches_saved_checkpoint": True,
            "unseen_outcomes_flipped": int((~expected).sum()),
            "all_arm_scores_unchanged": True,
            "chosen_arm_unchanged": original["chosen_arm"],
            "original_oracle_mean": original["oracle_mean"],
            "mutated_oracle_mean": changed["oracle_mean"],
        }
        masks.clear()
        scores.clear()
    return {"subject": subject, "seed": seed, "observations": budget, "variants": results}


def validate_subject(subject: str, results: Path, xs: dict, ys: dict, min_price: float, max_price: float) -> str:
    filename = f"bai_results_MMLU_{subject}_combined.npy"
    archive_path = results / filename
    archive = np.load(archive_path, allow_pickle=True).item()
    reference_path = BASELINE / "results" / filename
    reference = np.load(reference_path, allow_pickle=True).item()
    root = results / "checkpoints" / subject
    manifest = load_json(root / "manifest.json")
    provenance = archive["reproduction"]
    fingerprint = manifest["fingerprint"]
    jobs = [(subject, seed) for seed in SEEDS]
    require(archive["jobs"] == reference["jobs"] == jobs and archive["seeds"] == list(SEEDS), f"{subject}: incomplete seeds")
    require(archive["tasks"] == [subject] and archive["feature_names"] == VARIANTS, f"{subject}: variant metadata")
    require(archive["bench"] == "MMLU" and archive["combine_models"] and archive["n_models_stacked"] == 15, f"{subject}: arm metadata")
    require(archive["costs_recorded"] and set(archive["update_fields"]) == FIELDS, f"{subject}: phase schema")
    require(archive["out_by_task"] == [archive["out"]], f"{subject}: archive nesting")
    require(json_hash({k: v for k, v in manifest.items() if k != "fingerprint"}) == fingerprint, f"{subject}: manifest fingerprint")
    archived_manifest = {k: v for k, v in provenance.items() if k != "checkpoint_sha256"}
    require(json.loads(json.dumps(archived_manifest)) == manifest, f"{subject}: manifest and archive differ")
    for category in ("source_sha256", "input_sha256"):
        for path, digest in manifest[category].items():
            require(file_hash(REPO / path) == digest, f"{subject}: changed source or input {path}")

    matrix_path = MATRICES / f"{subject}.npy"
    matrix = np.load(matrix_path, allow_pickle=False)
    require(file_hash(matrix_path) == manifest["matrix_sha256"], f"{subject}: matrix SHA")
    require(list(matrix.shape) == manifest["matrix_shape"] and np.array_equal(matrix, np.vstack(ys[subject])), f"{subject}: matrix alignment")
    question_count = matrix.shape[1]
    budget = matrix.size // 10
    require(matrix.shape[0] == 1500 and manifest["observation_budget"] == budget, f"{subject}: arm or budget metadata")
    require(manifest["fitted_coefficients"] == 39 + question_count, f"{subject}: coefficient count")
    require(manifest["question_feature_width"] == question_count and manifest["prompt_feature_width"] == 39, f"{subject}: design width")
    dummies = np.repeat(np.eye(15, dtype=np.float64)[:, 1:], 100, axis=0)
    embeddings = np.vstack([model[1] for model in xs[subject]])
    design_hash = array_hash(np.column_stack((embeddings, dummies)))
    require(manifest["feature_sha256"] == dict.fromkeys(VARIANTS, design_hash), f"{subject}: X39 hash")
    require(manifest["prompt_feature_sha256"] == reference["reproduction"]["feature_sha256"]["pretrained_embedding"] == design_hash, f"{subject}: reference X39 hash")
    require(manifest["reference_logistic_archive_sha256"] == file_hash(reference_path), f"{subject}: reference archive SHA")
    require(manifest["reference_logistic_fingerprint"] == reference["reproduction"]["fingerprint"], f"{subject}: reference fingerprint")

    expected_names = {f"seed_{seed:02d}_{variant}.json" for seed in SEEDS for variant in VARIANTS}
    require({path.name for path in root.glob("seed_*.json")} == expected_names, f"{subject}: incomplete checkpoints")
    means = matrix.mean(axis=1)
    oracle = means.max()
    for seed, (record, old) in enumerate(zip(archive["out"], reference["out"], strict=True)):
        require(len(record) == 1 and len(record[0]) == 2, f"{subject}, seed {seed}: wrong nesting")
        require(record[0][0][0]["budget_cost"] == old[0][2][0]["budget_cost"], f"{subject}, seed {seed}: random first-phase spend differs")
        for variant, updates in zip(VARIANTS, record[0], strict=True):
            label = f"{subject}, seed {seed}, {variant}"
            path = root / f"seed_{seed:02d}_{variant}.json"
            checkpoint = load_json(path)
            require(file_hash(path) == provenance["checkpoint_sha256"][f"{seed}:{variant}"], f"{label}: checkpoint file SHA")
            require((checkpoint["subject"], checkpoint["seed"], checkpoint["variant"], checkpoint["fingerprint"]) == (subject, seed, variant, fingerprint), f"{label}: checkpoint identity")
            require(checkpoint["updates"] == updates and checkpoint["updates_sha256"] == json_hash({"updates": updates}), f"{label}: phase updates differ")
            require(len(updates) == 5, f"{label}: phase count")
            previous_cost = 0.0
            for phase, update in enumerate(updates):
                require(set(update) == FIELDS and update["phase"] == phase, f"{label}: phase schema")
                require(update["n_active"] == (1500, 750, 375, 187, 93)[phase], f"{label}: active arms")
                observations = budget * (phase + 1) // 5
                require(update["budget"] == update["budget_obs"] == observations, f"{label}: observation budget")
                cost = update["budget_cost"]
                require(np.isfinite(cost) and cost > previous_cost, f"{label}: non-increasing spend")
                require(observations * min_price <= cost + 1e-8 and cost <= observations * max_price + 1e-8, f"{label}: price bounds")
                previous_cost = cost
                arm = update["chosen_arm"]
                require(isinstance(arm, int) and 0 <= arm < len(means), f"{label}: chosen arm")
                require(update["chosen_mean"] == means[arm] and update["oracle_mean"] == oracle, f"{label}: benchmark means")
                require(update["simple_regret"] == oracle - means[arm], f"{label}: simple regret")
    require(len(archive["out"]) == len(SEEDS), f"{subject}: incomplete trajectories")
    return file_hash(archive_path)


def validate_archives(results: Path, all_subjects: bool) -> dict:
    selected_tasks = REPO / "Figure3_data_2026-11-07" / "data" / "selected_tasks.csv"
    with selected_tasks.open(encoding="utf-8-sig", newline="") as handle:
        figure_tasks = {row["task"] for row in csv.DictReader(handle)}
    require(len(figure_tasks) == 19, "Invalid Figure 3 subject set")
    subjects = {path.stem for path in MATRICES.glob("*.npy")} if all_subjects else figure_tasks
    actual = {path.stem.removeprefix("bai_results_MMLU_").removesuffix("_combined") for path in results.glob("bai_results_MMLU_*_combined.npy")}
    require(actual == subjects, f"Expected {len(subjects)} exact archives, found {len(actual)}")
    xs, ys = load_data("Xs.pickle"), load_data("Ys.pickle")
    price_path = REPO / "data_analysis" / "pricing" / "mmlu_prompt_eval_configurations_input_price.json"
    prices = [record["estimated_cost_per_1m_input_tokens"] for record in load_json(price_path).values()]
    archive_hashes = {
        subject: validate_subject(subject, results, xs, ys, min(prices), max(prices))
        for subject in sorted(subjects)
    }
    return {
        "subjects": len(subjects),
        "trajectories": len(subjects) * len(SEEDS) * len(VARIANTS),
        "phases": len(subjects) * len(SEEDS) * len(VARIANTS) * 5,
        "archive_set_sha256": json_hash(archive_hashes),
        "verified": [
            "Source, input, matrix, reconstructed X39, checkpoint, and reference archive hashes",
            "Complete seeds and variants, exact archive/checkpoint records and five budgets",
            "Canonical chosen and oracle means, regrets, price bounds, and random first-phase spend",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results")
    args = parser.parse_args()
    result = {"status": "passed", "unseen_label_check": validate_unseen_labels(args.results_dir)}
    if not args.pilot_only:
        result["archives"] = validate_archives(args.results_dir, args.all_subjects)
    report = json.dumps(result, indent=2, allow_nan=False)
    (AUDIT_DIR / "independent_checks.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
