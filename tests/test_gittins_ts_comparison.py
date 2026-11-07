"""Reference parity for the paper-matched paired Gittins/TS simulator."""

from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_ts_policy import gaussian_thompson_sampling
from gittins_policy import gittins_index_exploration
from gittins_shrinking_posterior import compute_finite_population_roots_lookup_table
from simple_regret_recommend import posterior_incumbent


@pytest.fixture(scope="module")
def comparison():
    spec = importlib.util.spec_from_file_location(
        "paired_gittins_ts_under_test", ROOT / "scripts/compare_gittins_ts.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def native_roots(n_examples: int, prior_variance: float, costs: tuple[float, ...]):
    scaled = torch.tensor(costs, dtype=torch.float32) * 1e-4
    return np.asarray(compute_finite_population_roots_lookup_table(
        prior_variance=prior_variance, tau_sq_cell=.25, n_examples=n_examples,
        costs_per_arm=scaled.numpy(), n_points=257,
    ), dtype=np.float32)


def native_simulation(matrix, *, policy, seed, prior_mean, prior_variance,
                      batch_size, per_arm_cost, cost_mode, roots_lookup=None,
                      budget_fraction=.1, max_steps=None):
    """Original public policies and recommendation helpers, without fast state."""
    truth = torch.tensor(matrix, dtype=torch.float32)
    obs = torch.full_like(truth, float("nan"))
    prices = np.asarray(per_arm_cost, dtype=np.float64)
    decision_prices = np.ones(len(prices)) if cost_mode == "unit" else prices
    full_resource = matrix.size if cost_mode == "unit" else matrix.shape[1] * prices.sum()
    budget = budget_fraction * full_resource
    torch.manual_seed(seed)
    history = {key: [] for key in (
        "x", "x_original_cost", "recommended_arm", "recommended_mean",
        "pulled_arm", "n_new_samples", "sampled_columns",
    )}
    evaluated, spent = 0, 0.0
    cached_scores = torch.full((truth.shape[0],), float("inf"), dtype=torch.float32)
    previous_arm = None
    while (evaluated if cost_mode == "unit" else spent) < budget:
        if max_steps is not None and len(history["x"]) >= max_steps:
            break
        if policy == "gittins":
            batch = gittins_index_exploration(
                obs, prior_mean=prior_mean, prior_variance=prior_variance,
                obs_noise_variance=.25 / batch_size, batch_size=batch_size,
                cost_per_transition=torch.tensor(decision_prices, dtype=torch.float64),
                cost_scaling_factor=1e-4, roots_lookup_table=torch.tensor(roots_lookup),
                batch_observation_model=True, allow_early_stop=False,
                cached_scores=cached_scores,
                recompute_arms=None if previous_arm is None else [previous_arm],
            )
        else:
            batch = gaussian_thompson_sampling(
                obs, prior_mean=prior_mean, prior_variance=prior_variance,
                obs_noise_variance=.25 / batch_size, batch_size=batch_size,
            )
        if batch is None:
            break
        rows, columns = batch
        arm = int(rows[0])
        previous_arm = arm
        obs[rows, columns] = truth[rows, columns]
        n_new = len(columns)
        evaluated += n_new
        spent += n_new * prices[arm]
        recommended, means = posterior_incumbent(
            obs, prior_mean=prior_mean, prior_variance=prior_variance,
            tau_sq_cell=.25, std_penalty=1,
        )
        values = dict(
            x=evaluated, x_original_cost=spent, recommended_arm=recommended,
            recommended_mean=float(means[recommended]), pulled_arm=arm,
            n_new_samples=n_new,
            sampled_columns=np.pad(columns.numpy(), (0, batch_size - n_new), constant_values=-1),
        )
        for key, value in values.items():
            history[key].append(value)
    return {key: np.asarray(value) for key, value in history.items()}


@pytest.mark.parametrize("policy", ["gittins", "ts"])
@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
@pytest.mark.parametrize("cost_mode", ["unit", "cost"])
def test_sufficient_statistics_simulation_matches_native_policies(
    comparison, policy, prior_mean, prior_variance, cost_mode,
):
    matrix = (np.random.default_rng(37).random((3, 24)) < np.array([.2, .6, .8])[:, None]).astype(np.float32)
    prices = np.array([.25, .45, .25])
    decision_prices = np.ones(3) if cost_mode == "unit" else prices
    roots = native_roots(24, prior_variance, tuple(decision_prices)) if policy == "gittins" else None
    kwargs = dict(
        policy=policy, seed=8, prior_mean=prior_mean, prior_variance=prior_variance,
        batch_size=4, per_arm_cost=prices, cost_mode=cost_mode,
        roots_lookup=roots, budget_fraction=.5,
    )
    expected = native_simulation(matrix, **kwargs)
    actual = comparison.simulate_policy(matrix, **kwargs, record_cells=True)
    for key in ("x", "recommended_arm", "pulled_arm", "n_new_samples", "sampled_columns"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    np.testing.assert_allclose(actual["x_original_cost"], expected["x_original_cost"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual["recommended_mean"], expected["recommended_mean"])


@pytest.mark.parametrize("policy", ["gittins", "ts"])
@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
@pytest.mark.parametrize("cost_mode", ["unit", "cost"])
def test_gsm8k_first_fifty_rounds_match_native_policies(
    comparison, policy, prior_mean, prior_variance, cost_mode,
):
    matrix_path = ROOT / "data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy"
    archive_path = ROOT / "data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz"
    if not matrix_path.is_file() or not archive_path.is_file():
        pytest.skip("GSM8K supplied-data parity check requires the bundled matrices")
    matrix = np.load(matrix_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        prices = archive["cost"] / matrix.shape[1]
    # A supplied test root table deliberately varies with arm price and sample
    # count. Native policy comparison then isolates online simulation
    # arithmetic and RNG order from an expensive repeated solver invocation.
    decision_prices = np.ones(len(prices)) if cost_mode == "unit" else prices
    progress = np.linspace(1, 0, matrix.shape[1] + 1, dtype=np.float32)
    roots = (-.2 * progress[None, :] / (1 + decision_prices[:, None])).astype(np.float32)
    kwargs = dict(
        policy=policy, seed=11, prior_mean=prior_mean, prior_variance=prior_variance,
        batch_size=8, per_arm_cost=prices, cost_mode=cost_mode,
        roots_lookup=roots if policy == "gittins" else None,
        budget_fraction=.1, max_steps=50,
    )
    expected = native_simulation(matrix, **kwargs)
    actual = comparison.simulate_policy(matrix, **kwargs, record_cells=True)
    for key in ("x", "recommended_arm", "pulled_arm", "n_new_samples", "sampled_columns"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    np.testing.assert_array_equal(actual["recommended_mean"], expected["recommended_mean"])


@pytest.mark.parametrize("prior_variance", [.04, .01])
def test_price_deduplication_preserves_native_dp_roots(comparison, prior_variance):
    prices = np.array([.25, .45, .25], dtype=np.float64)
    expected = native_roots(24, prior_variance, tuple(prices))
    actual = comparison.build_roots_lookup(
        prior_variance=prior_variance, tau_sq_cell=.25, n_examples=24,
        costs_per_arm=prices, cost_scaling_factor=1e-4, n_points=257,
    )
    np.testing.assert_array_equal(actual, expected)
