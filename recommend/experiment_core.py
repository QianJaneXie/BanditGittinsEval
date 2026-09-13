"""Generic replay engine, extending the frozen GSM8K experiment to explicit priors.

The original run_gsm8k.py is retained so its saved source hashes remain valid.
The same index equations use the portable numerical backend; NumPy samples cells.
"""
from __future__ import annotations
import numpy as np
from portable_gittins import posterior, next_arm
from methods import RecommendationRules, method_specs


def run_path(
    matrix: np.ndarray, roots: np.ndarray, *, prior_tag: str, prior_pair: tuple[float, float], run_seed: int,
    matrix_seed: int, batch_size: int, budget_fraction: float,
    tau_sq_cell: float,
) -> dict[str, np.ndarray]:
    prior_mean, prior_variance = prior_pair
    n_arms, n_examples = matrix.shape
    budget = max(1, round(budget_fraction * matrix.size))
    rules = RecommendationRules(n_arms, n_examples, prior_mean, prior_variance, tau_sq_cell)
    specs = method_specs()
    keys = [spec.key for spec in specs]
    counts = np.zeros(n_arms, dtype=np.int64)
    sums = np.zeros(n_arms, dtype=np.float64)
    observed = np.zeros(matrix.shape, dtype=bool)
    generator = np.random.default_rng(run_seed)
    true_means = matrix.mean(axis=1)
    best_mean = float(true_means.max())
    history = {key: [] for key in [
        "x", "recommended_arm", "regret", "selected_n", "selected_variance", "scores",
        "eligible_count", "pulled_arm", "pulled_columns", "counts", "sums",
        "confidence_lower", "confidence_upper", "certified_regret_bound",
    ]}
    evaluated = 0
    step = 0
    while evaluated < budget:
        arm = next_arm(counts, sums, roots, n_examples, prior_mean, prior_variance, tau_sq_cell)
        remaining = np.flatnonzero(~observed[arm])
        # Exact cell budget: only the last batch can be smaller than native runs.
        size = min(batch_size, len(remaining), budget - evaluated)
        if size == 0:
            break
        perm = generator.permutation(len(remaining))[:size]
        columns = remaining[perm]
        observed[arm, columns] = True
        counts[arm] += size
        sums[arm] += float(matrix[arm, columns].sum())
        evaluated += size
        recommendations = rules.evaluate(counts, sums)
        _, variance = posterior(counts, sums, prior_mean, prior_variance, tau_sq_cell)
        arms = np.array([recommendations[key].arm for key in keys], dtype=np.int32)
        valid = arms >= 0
        regret = np.full(len(keys), np.nan)
        selected_n = np.full(len(keys), -1, dtype=np.int32)
        selected_var = np.full(len(keys), np.nan)
        regret[valid] = best_mean - true_means[arms[valid]]
        selected_n[valid] = counts[arms[valid]]
        selected_var[valid] = variance[arms[valid]]
        lower, upper = rules.confidence_bounds(counts, sums)
        # Simultaneous intervals give an honest bound on regret of any chosen arm.
        bound = np.full(len(keys), np.nan)
        top = int(np.argmax(upper))
        second = float(np.max(np.delete(upper, top), initial=0.0))
        competing_upper = np.where(arms[valid] == top, second, upper[top])
        bound[valid] = np.maximum(0.0, competing_upper - lower[arms[valid]])
        padded_columns = np.full(batch_size, -1, dtype=np.int32)
        padded_columns[:size] = columns
        history["x"].append(evaluated)
        history["recommended_arm"].append(arms)
        history["regret"].append(regret)
        history["selected_n"].append(selected_n)
        history["selected_variance"].append(selected_var)
        history["scores"].append([recommendations[key].score for key in keys])
        history["eligible_count"].append([recommendations[key].eligible_count for key in keys])
        history["pulled_arm"].append(arm)
        history["pulled_columns"].append(padded_columns)
        history["counts"].append(counts.astype(np.uint32).copy())
        history["sums"].append(sums.astype(np.float32).copy())
        history["confidence_lower"].append(lower)
        history["confidence_upper"].append(upper)
        history["certified_regret_bound"].append(bound)
        step += 1
    result = {key: np.asarray(value) for key, value in history.items()}
    result.update(
        prior_tag=np.asarray(prior_tag), prior_mean=np.asarray(prior_mean),
        prior_variance=np.asarray(prior_variance), run_seed=np.asarray(run_seed),
        matrix_seed=np.asarray(matrix_seed), method_keys=np.asarray(keys),
        n_arms=np.asarray(n_arms), n_examples=np.asarray(n_examples),
        true_means=true_means, native_steps_verified=np.asarray(0),
        sampling_generator=np.asarray("numpy.random.PCG64"),
    )
    # Invariants catch duplicate evaluations and incorrect regret/abstention handling.
    assert evaluated == budget == int(counts.sum())
    assert np.all(counts <= n_examples)
    assert observed.sum() == evaluated
    assert np.all(np.isnan(result["regret"]) == (result["recommended_arm"] < 0))
    assert np.nanmin(result["regret"]) >= -1e-12
    return result

