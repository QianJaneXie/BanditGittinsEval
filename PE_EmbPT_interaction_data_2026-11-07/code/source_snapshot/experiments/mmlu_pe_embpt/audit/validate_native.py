"""Validate one full interaction replay against dense PromptEval on college chemistry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch
import warnings

import numpy as np
from scipy.sparse import isspmatrix_csr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
SUBJECT = "college_chemistry"
SEED = 0
PHASES = 5
ABS_TOLERANCE = 1e-12
sys.path.insert(0, str(EXPERIMENT))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_methods(path: Path):
    """Load the unmodified dense estimator and sampler without loading an encoder."""
    def check_multicolinearity(features, tol=1e-6):
        pca = PCA().fit(features)
        require(np.all(pca.explained_variance_ratio_ > tol), "Native PCA rank check failed.")

    utils = ModuleType("utils")
    utils.check_multicolinearity = check_multicolinearity
    spec = importlib.util.spec_from_file_location("independent_native_pe_methods", path)
    methods = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"utils": utils}):
        spec.loader.exec_module(methods)
    return methods


def fit(methods, seen, y, features):
    model = methods.ExtendedRaschModel()
    model.fit(seen, y, features, Z=None)
    return model.rasch_model.mu, model.get_Y_hat().mean(axis=1)


def reconstruct_features() -> tuple[np.ndarray, np.ndarray]:
    import pickle

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated", category=DeprecationWarning)
        with (REPO / "prompteval" / "data" / "Xs.pickle").open("rb") as handle:
            models = pickle.load(handle)["MMLU"][SUBJECT]
    prompt = np.vstack([model[1] for model in models])
    dummies = np.repeat(np.eye(15, dtype=np.float64)[:, 1:], 100, axis=0)
    x39 = np.column_stack((prompt, dummies))
    # Build the tensor-product row independently, without any runner helper.
    blocks = np.array([np.kron(np.eye(15)[arm // 100], row) for arm, row in enumerate(prompt)])
    return x39, np.column_stack((blocks, dummies))


def update_record(phase, matrix_means, chosen, counts, prices, n_active):
    observations = float(counts.sum())
    mean = float(matrix_means[chosen])
    oracle = float(matrix_means.max())
    return {
        "phase": phase,
        "budget_obs": observations,
        "budget_cost": float((counts * prices).sum()),
        "budget": observations,
        "chosen_arm": int(chosen),
        "chosen_mean": mean,
        "oracle_mean": oracle,
        "simple_regret": oracle - mean,
        "n_active": n_active,
    }


def compare_records(reference, candidate):
    require(len(reference) == len(candidate) == PHASES, "Expected two complete five-phase records.")
    numeric = ("budget_obs", "budget_cost", "budget", "chosen_mean", "oracle_mean", "simple_regret")
    exact = reference == candidate
    max_diff = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        require(expected.keys() == actual.keys(), "Phase keys differ from native reference.")
        for name in ("phase", "chosen_arm", "n_active"):
            require(expected[name] == actual[name], f"Native {name} differs in phase {expected['phase']}.")
        for name in numeric:
            diff = abs(expected[name] - actual[name])
            require(diff <= ABS_TOLERANCE, f"Native {name} differs by {diff} in phase {expected['phase']}.")
            max_diff = max(max_diff, diff)
    return exact, max_diff


def validate_concat_anchor(interaction, dense, matrix, x39, prices, budget):
    trace_path = REPO / "PE_EmbPT_data_2026-11-07" / "supplementary" / "pe_seed_traces.csv"
    with trace_path.open(encoding="utf-8-sig", newline="") as handle:
        archived = {row["variant"]: row for row in csv.DictReader(handle)
                    if row["subject"] == SUBJECT and row["seed"] == "0" and row["phase"] == "0"}
    interaction.register_design(x39)
    active = np.arange(matrix.shape[0])
    summary = {}
    for variant, random_column in (("rasch39_random", True), ("rasch39_balanced", False)):
        mask = dense.StratSample(np.zeros(matrix.shape, dtype=bool), budget // PHASES, SEED, active, random_column)
        dense_coefs, dense_scores = fit(dense, mask, matrix, x39)
        with patch.object(dense, "GenXY", interaction.sparse_rasch_design):
            sparse_coefs, sparse_scores = fit(dense, mask, matrix, x39)
        require(dense_coefs.size == 39 + matrix.shape[1], "Wrong fitted coefficient count for X39 anchor.")
        np.testing.assert_allclose(sparse_coefs, dense_coefs, rtol=0, atol=ABS_TOLERANCE)
        np.testing.assert_allclose(sparse_scores, dense_scores, rtol=0, atol=ABS_TOLERANCE)
        chosen = int(np.argmax(dense_scores))
        expected = archived[variant]
        require(chosen == int(np.argmax(sparse_scores)) == int(expected["chosen_arm"]), f"{variant}: first recommendation differs.")
        reference = update_record(0, matrix.mean(axis=1), chosen, mask.sum(axis=1), prices, len(active))
        for field in ("budget_obs", "budget_cost", "chosen_mean", "oracle_mean", "simple_regret"):
            require(abs(reference[field] - float(expected[field])) <= ABS_TOLERANCE, f"{variant}: archived first-phase {field} differs.")
        summary[variant] = {
            "chosen_arm": chosen,
            "fitted_coefficients": int(dense_coefs.size),
            "native_csr_max_absolute_coefficient_difference": float(np.max(abs(sparse_coefs - dense_coefs))),
            "native_csr_max_absolute_score_difference": float(np.max(abs(sparse_scores - dense_scores))),
            "archive_phase_one_match": True,
            "first_phase_mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
        }
    return summary


def validate_variant(interaction, dense, matrix, features, prices, budget, variant, random_column):
    native_mask = np.zeros(matrix.shape, dtype=bool)
    csr_mask = np.zeros(matrix.shape, dtype=bool)
    n_arms, n_questions = matrix.shape
    active = np.arange(n_arms)
    means = matrix.mean(axis=1)
    reference = []
    max_coefficient_difference = 0.0
    max_score_difference = 0.0
    first_mask_sha256 = None
    intervention = None
    start = perf_counter()

    for phase in range(PHASES):
        target = budget * (phase + 1) // PHASES
        dense.StratSample(native_mask, target, SEED, active, random_column)
        interaction.stratified_sample(csr_mask, target, SEED, active, random_column)
        np.testing.assert_array_equal(csr_mask, native_mask)
        require(int(native_mask.sum()) == target, "Pilot did not reach its scheduled budget.")

        coefficients, scores = fit(dense, native_mask, matrix, features)
        if phase == 0:
            identity = np.eye(n_questions)
            dense_design, dense_labels = dense.GenXY(native_mask, matrix, features, identity)
            csr_design, csr_labels = interaction.sparse_rasch_design(csr_mask, matrix, features, identity)
            require(isspmatrix_csr(csr_design), "Production design is not CSR.")
            np.testing.assert_array_equal(csr_design.toarray(), dense_design)
            np.testing.assert_array_equal(csr_labels, dense_labels)
            first_mask_sha256 = hashlib.sha256(np.ascontiguousarray(native_mask).tobytes()).hexdigest()

        with patch.object(dense, "GenXY", interaction.sparse_rasch_design):
            csr_coefficients, csr_scores = fit(dense, csr_mask, matrix, features)
            if phase == 0:
                changed = matrix.copy()
                changed[~csr_mask] = 1 - changed[~csr_mask]
                changed_coefficients, changed_scores = fit(dense, csr_mask, changed, features)
                np.testing.assert_array_equal(changed_coefficients, csr_coefficients)
                np.testing.assert_array_equal(changed_scores, csr_scores)
                require(changed.mean(axis=1).max() != means.max(), "The unseen-label intervention did not change the oracle.")
                intervention = {
                    "unseen_labels_inverted": int((~csr_mask).sum()),
                    "all_coefficients_and_scores_unchanged": True,
                    "chosen_arm_unchanged": int(active[np.argmax(csr_scores[active])]),
                    "oracle_changed": True,
                }

        require(coefficients.shape == csr_coefficients.shape == (389 + n_questions,), "Wrong fitted coefficient count.")
        np.testing.assert_allclose(csr_coefficients, coefficients, rtol=0, atol=ABS_TOLERANCE)
        np.testing.assert_allclose(csr_scores, scores, rtol=0, atol=ABS_TOLERANCE)
        max_coefficient_difference = max(max_coefficient_difference, float(np.max(abs(csr_coefficients - coefficients))))
        max_score_difference = max(max_score_difference, float(np.max(abs(csr_scores - scores))))
        chosen = active[np.argmax(scores[active])]
        require(chosen == active[np.argmax(csr_scores[active])], f"Recommendations differ at phase {phase}.")
        reference.append(update_record(phase, means, chosen, native_mask.sum(axis=1), prices, len(active)))
        remove = (len(active) + 1) // 2
        survivors = active[np.argsort(scores[active])[remove:]]
        np.testing.assert_array_equal(survivors, active[np.argsort(csr_scores[active])[remove:]])
        active = survivors

    native_comparison_seconds = perf_counter() - start
    start = perf_counter()
    actual = interaction.replay_case(matrix, features, budget, SEED, variant)
    replay_seconds = perf_counter() - start
    exact, max_difference = compare_records(reference, actual)
    return {
        "variant": variant,
        "native_masks_exact_at_all_five_phases": True,
        "native_survivor_order_exact_at_all_five_phases": True,
        "first_phase_dense_and_csr_design_and_labels_exact": True,
        "first_phase_mask_sha256": first_mask_sha256,
        "fitted_coefficients": int(coefficients.size),
        "max_absolute_coefficient_difference": max_coefficient_difference,
        "max_absolute_score_difference": max_score_difference,
        "unseen_outcome_intervention": intervention,
        "production_replay_exact": exact,
        "production_replay_max_absolute_numeric_difference": max_difference,
        "chosen_arms": [row["chosen_arm"] for row in actual],
        "simple_regrets": [row["simple_regret"] for row in actual],
        "dense_and_csr_comparison_seconds": native_comparison_seconds,
        "production_replay_seconds": replay_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-methods", type=Path, default=REPO / "prompteval" / "methods.py",
                        help="Path to the original dense PromptEval methods.py. The vendored copy is the default.")
    parser.add_argument("--out", type=Path, help="Optional JSON evidence file.")
    args = parser.parse_args()

    import interaction

    dense = native_methods(args.native_methods)
    with threadpool_limits(limits=1):
        interaction.initialize_worker()
        matrix, features, provenance, _ = interaction.prepare_subject(SUBJECT)
        x39, expected = reconstruct_features()
        np.testing.assert_array_equal(interaction.build_interaction_features(x39), expected)
        np.testing.assert_array_equal(features, expected)
        require(features.shape == (1500, 389), "Unexpected interaction feature shape.")
        price_file = REPO / "data_analysis" / "pricing" / "mmlu_prompt_eval_configurations_input_price.json"
        records = json.loads(price_file.read_text(encoding="utf-8"))
        prices = np.array([records[str(arm)]["estimated_cost_per_1m_input_tokens"] for arm in range(1500)])
        budget = matrix.size // 10
        require(budget % PHASES == 0, "Expected equal pilot phase increments.")
        concat_anchor = validate_concat_anchor(interaction, dense, matrix, x39, prices, budget)
        interaction.register_design(features)
        variants = [validate_variant(interaction, dense, matrix, features, prices, budget, variant, random_column)
                    for variant, random_column in zip(interaction.VARIANTS, (True, False), strict=True)]
        for actual, old in zip(variants, concat_anchor.values(), strict=True):
            require(actual["first_phase_mask_sha256"] == old["first_phase_mask_sha256"], "The X389 first-phase sampler differs from X39.")
        report = {
            "status": "passed",
            "subject": SUBJECT,
            "seed": SEED,
            "matrix_shape": list(matrix.shape),
            "matrix_sha256": provenance["matrix_sha256"],
            "feature_shape": list(features.shape),
            "unscaled_kronecker_and_reference_indicators_exact": True,
            "native_methods_path": Path(os.path.relpath(args.native_methods.resolve(), REPO)).as_posix(),
            "native_methods_sha256": file_sha(args.native_methods),
            "validation_script_sha256": file_sha(Path(__file__)),
            "runner_source_sha256": provenance["source_sha256"],
            "X39_sha256": provenance["X39_sha256"],
            "X389_sha256": provenance["X389_sha256"],
            "coefficient_and_score_absolute_tolerance": ABS_TOLERANCE,
            "concat_anchor": concat_anchor,
            "variants": variants,
        }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
