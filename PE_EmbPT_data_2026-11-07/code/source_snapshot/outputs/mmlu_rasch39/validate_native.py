"""Check the CSR Rasch replay against upstream's native dense implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np

RUNNER_SPEC = importlib.util.spec_from_file_location("rasch39_runner", Path(__file__).with_name("run_experiment.py"))
experiment = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(experiment)


SUBJECT = "college_chemistry"
SEED = 0


def import_upstream_methods():
    source = experiment.REPLAY.REPO.parent / "prompteval" / "prompteval" / "methods.py"
    spec = importlib.util.spec_from_file_location("upstream_prompteval_methods", source)
    methods = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(methods)
    methods.check_multicolinearity = experiment.check_multicollinearity
    return methods, source


def fit_row_scores(methods, seen, matrix, features):
    model = methods.ExtendedRaschModel()
    model.fit(seen, matrix, features, Z=None)
    return model.rasch_model.mu, model.get_Y_hat().mean(axis=1)


def replay_native(methods, matrix, features, budget, random_column):
    bai = experiment.legacy.STATE["bai"]
    old_model, old_sampler = bai.LogReg, bai.StratSample

    class NativeScore:
        def fit(self, seen, matrix, features):
            _, self.thetas = fit_row_scores(methods, seen, matrix, features)

    def sample(seen, max_seen, seed, active_arms, _random_column):
        return methods.StratSample(seen, max_seen, seed, active_arms, random_column)

    bai.LogReg, bai.StratSample = NativeScore, sample
    try:
        return bai.compute_regrets(
            budget, matrix, features, random_seed=SEED, backend="sklearn",
            costs=experiment.legacy.STATE["prices"],
        )
    finally:
        bai.LogReg, bai.StratSample = old_model, old_sampler


def validate_variant(upstream, matrix, features, budget, variant):
    bai = experiment.legacy.STATE["bai"]
    sparse_methods = experiment.STATE["methods"]
    random_column = variant == experiment.VARIANTS[0]
    optimized_sampler = experiment.STATE["random_sampler"] if random_column else experiment.sample_balanced
    active = list(range(matrix.shape[0]))
    empty = np.zeros(matrix.shape, dtype=bool)
    first_budget = budget - (bai.MAX_BAI_PHASES - 1) * (budget // bai.MAX_BAI_PHASES)
    native_mask = upstream.StratSample(empty.copy(), first_budget, SEED, active, random_column)
    sparse_mask = optimized_sampler(empty.copy(), first_budget, SEED, active, random_column)
    assert np.array_equal(native_mask, sparse_mask)

    # Keep some observations on eliminated rows and preserve the supplied
    # active-arm order to test the native global column counts.
    survivors = active[::-3]
    second_budget = first_budget + budget // bai.MAX_BAI_PHASES
    native_second = upstream.StratSample(native_mask.copy(), second_budget, 7, survivors, random_column)
    sparse_second = optimized_sampler(sparse_mask.copy(), second_budget, 7, survivors, random_column)
    assert np.array_equal(native_second, sparse_second)

    native_coefficients, native_scores = fit_row_scores(upstream, native_mask, matrix, features)
    sparse_coefficients, sparse_scores = fit_row_scores(sparse_methods, sparse_mask, matrix, features)
    np.testing.assert_allclose(sparse_coefficients, native_coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sparse_scores, native_scores, rtol=0, atol=1e-12)
    assert len(sparse_coefficients) == features.shape[1] + matrix.shape[1]

    counterfactual = matrix.copy()
    counterfactual[~sparse_mask] = 1 - counterfactual[~sparse_mask]
    flipped_coefficients, flipped_scores = fit_row_scores(sparse_methods, sparse_mask, counterfactual, features)
    np.testing.assert_array_equal(flipped_coefficients, sparse_coefficients)
    np.testing.assert_array_equal(flipped_scores, sparse_scores)

    begin = time.perf_counter()
    native = replay_native(upstream, matrix, features, budget, random_column)
    native_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    optimized = experiment.replay_case(matrix, features, budget, SEED, variant)
    optimized_seconds = time.perf_counter() - begin
    comparison = experiment.REPLAY.compare_updates(native, optimized, bai.UPDATE_FIELDS)
    assert comparison["exact_match"], comparison
    return {
        "variant": variant,
        "first_and_second_masks_exact": True,
        "first_phase_max_absolute_coefficient_difference": float(np.max(np.abs(native_coefficients - sparse_coefficients))),
        "first_phase_max_absolute_score_difference": float(np.max(np.abs(native_scores - sparse_scores))),
        "unseen_labels_inverted": int(np.count_nonzero(matrix != counterfactual)),
        "chosen_arm_unchanged_after_inversion": int(np.argmax(sparse_scores)),
        "inversion_coefficients_and_scores_exact": True,
        "native_optimized_complete_replay": comparison,
        "native_seconds": native_seconds,
        "optimized_seconds": optimized_seconds,
    }


def main():
    experiment.initialize_worker()
    upstream, source = import_upstream_methods()
    matrix, features, provenance, _ = experiment.prepare_subject(SUBJECT, experiment.LEGACY_DIR / "results")
    report = {
        "subject": SUBJECT,
        "seed": SEED,
        "native_methods_source_sha256": experiment.REPLAY.sha256(source),
        "prompt_feature_sha256": provenance["prompt_feature_sha256"],
        "matrix_sha256": provenance["matrix_sha256"],
        "variants": [
            validate_variant(upstream, matrix, features, provenance["observation_budget"], variant)
            for variant in experiment.VARIANTS
        ],
    }
    experiment.REPLAY.write_json(Path(__file__).resolve().parent / "audit" / "native_validation.json", report)
    for case in report["variants"]:
        print(case, flush=True)


if __name__ == "__main__":
    main()
