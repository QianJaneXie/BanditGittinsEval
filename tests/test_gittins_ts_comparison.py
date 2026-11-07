"""Independent paper-batch Gittins reference and unchanged native TS parity.

For B > 1 the historical per-cell public Gittins policy is not the reference:
one manuscript DP transition observes a batch and costs lambda * b * price.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gaussian_ts_policy import gaussian_thompson_sampling
from gittins_index_computation import compute_gittins_for_random_walk
from gittins_shrinking_posterior import compute_finite_population_roots_lookup_table
from simple_regret_recommend import posterior_incumbent


@pytest.fixture(scope="module")
def comparison():
    spec = importlib.util.spec_from_file_location(
        "paired_gittins_ts_under_test", ROOT / "scripts/compare_gittins_ts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paper_schedule(n_examples, batch_size, prior_variance):
    """Appendix equations derived in NumPy, without a production helper."""
    batches = np.diff(np.r_[np.arange(0, n_examples, batch_size), n_examples])
    before = np.r_[0, batches.cumsum()[:-1]]
    v = 1 / (1 / prior_variance + before / .25)
    a = 1 + .25 / (n_examples * prior_variance)
    sigmas = a * v / np.sqrt(v + .25 / batches)
    return batches, sigmas


@lru_cache(maxsize=None)
def paper_root_solver(n_examples, batch_size, prior_variance, n_points):
    batches, sigmas = paper_schedule(n_examples, batch_size, prior_variance)
    sigmas = jnp.asarray(sigmas, dtype=jnp.float32)
    costs = jnp.asarray(1e-4 * batches, dtype=jnp.float32)

    @jax.jit
    def solve(stage, price):
        # Per-state solve on the full schedules retains the same grid at each
        # stage. A sliced suffix solve would change the discretization grid.
        return -compute_gittins_for_random_walk(
            stage, jnp.float32(0), sigmas, costs * price, n_points)
    return solve


@lru_cache(maxsize=None)
def paper_root(n_examples, batch_size, prior_variance, price, stage, n_points=257):
    if stage == (n_examples + batch_size - 1) // batch_size:
        return np.float32(0)
    return np.float32(paper_root_solver(n_examples, batch_size, prior_variance, n_points)(
        jnp.uint32(stage), jnp.float32(price)))


def paper_posterior(observed, prior_mean, prior_variance):
    """Recompute empirical-target moments from the masked matrix."""
    n_examples = observed.shape[1]
    counts = np.isfinite(observed).sum(axis=1)
    sums = np.nansum(observed.astype(np.float64), axis=1)
    v = 1 / (1 / prior_variance + counts / .25)
    m = v * (prior_mean / prior_variance + sums / .25)
    remaining = n_examples - counts
    means = ((sums + remaining * m) / n_examples).astype(np.float32)
    variances = (remaining ** 2 * v + remaining * .25) / n_examples ** 2
    return counts, sums, means, variances


def reference_simulation(matrix, *, policy, seed, prior_mean, prior_variance,
                         batch_size, per_arm_cost, cost_mode,
                         budget_fraction=.1, max_steps=None, n_points=257):
    """Slow paper Gittins reference; TS calls the unchanged native policy."""
    truth = torch.tensor(matrix, dtype=torch.float32)
    obs = torch.full_like(truth, float("nan"))
    n_arms, n_examples = truth.shape
    prices = np.asarray(per_arm_cost, dtype=np.float64)
    dp_prices = np.ones(n_arms) if cost_mode == "unit" else prices
    full_resource = matrix.size if cost_mode == "unit" else n_examples * prices.sum()
    budget = (max(1, round(budget_fraction * full_resource)) if cost_mode == "unit"
              else budget_fraction * full_resource)
    torch.manual_seed(seed)
    stages = np.zeros(n_arms, dtype=np.int64)
    history = {key: [] for key in (
        "x", "x_original_cost", "recommended_arm", "recommended_mean",
        "pulled_arm", "n_new_samples", "sampled_columns")}
    evaluated, spent, natural_stop = 0, 0., -1

    def paper_indices(counts, means):
        roots = np.array([paper_root(n_examples, batch_size, prior_variance,
                                    float(dp_prices[k]), int(stages[k]), n_points)
                          for k in range(n_arms)], dtype=np.float32)
        scores = means - roots
        scores[counts == n_examples] = means[counts == n_examples]
        return scores

    while (evaluated if cost_mode == "unit" else spent) < budget:
        if max_steps is not None and len(history["x"]) >= max_steps:
            break
        if policy == "gittins":
            counts, _, means, _ = paper_posterior(obs.numpy(), prior_mean, prior_variance)
            if (counts == n_examples).all():
                break
            scores = paper_indices(counts, means)
            scores[counts == n_examples] = -np.inf
            arm = int(scores.argmax())
            unseen = torch.nonzero(torch.isnan(obs[arm]), as_tuple=True)[0]
            columns = unseen[torch.randperm(len(unseen))[:batch_size]]
            rows = torch.full_like(columns, arm)
        else:
            batch = gaussian_thompson_sampling(
                obs, prior_mean=prior_mean, prior_variance=prior_variance,
                obs_noise_variance=.25 / batch_size, batch_size=batch_size)
            if batch is None:
                break
            rows, columns = batch
            arm = int(rows[0])
        obs[rows, columns] = truth[rows, columns]
        stages[arm] += 1
        n_new = len(columns)
        evaluated += n_new
        spent += float(torch.full((n_new,), prices[arm], dtype=torch.float64).sum())
        counts, sums, means, variances = paper_posterior(obs.numpy(), prior_mean, prior_variance)
        if policy == "gittins":
            rec = int((means.astype(np.float64) - np.sqrt(variances)).argmax())
            scores = paper_indices(counts, means)
            if natural_stop < 0 and counts[int(scores.argmax())] == n_examples:
                natural_stop = evaluated
        else:
            rec, native_means = posterior_incumbent(
                obs, prior_mean=prior_mean, prior_variance=prior_variance,
                tau_sq_cell=.25, std_penalty=1)
            means = native_means.numpy()
        values = dict(x=evaluated, x_original_cost=spent, recommended_arm=rec,
                      recommended_mean=float(means[rec]), pulled_arm=arm,
                      n_new_samples=n_new, sampled_columns=np.pad(
                          columns.numpy(), (0, batch_size - n_new), constant_values=-1))
        for key, value in values.items():
            history[key].append(value)
    result = {key: np.asarray(value) for key, value in history.items()}
    result.update(counts=counts, observed_sums=sums, posterior_means=means,
                  posterior_variances=variances, natural_stop_cum_eval=natural_stop)
    if policy == "gittins":
        result["indices"] = scores
    return result


def assert_same_trajectory(actual, expected):
    for key in ("x", "recommended_arm", "pulled_arm", "n_new_samples", "sampled_columns"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    np.testing.assert_allclose(actual["x_original_cost"], expected["x_original_cost"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(actual["recommended_mean"], expected["recommended_mean"])


@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
@pytest.mark.parametrize("cost_mode", ["unit", "cost"])
def test_batch_gittins_matches_paper_reference_including_partial_completion(
    comparison, prior_mean, prior_variance, cost_mode,
):
    matrix = (np.random.default_rng(37).random((3, 11)) < np.array([.2, .6, .8])[:, None]).astype(np.float32)
    prices = np.array([.25, .45, .25])
    dp_prices = np.ones(3) if cost_mode == "unit" else prices
    roots = comparison.build_roots_lookup(
        prior_variance=prior_variance, tau_sq_cell=.25, n_examples=11,
        batch_size=4, costs_per_arm=dp_prices, n_points=257)
    assert roots.shape == (3, 4)  # H=ceil(11/4), not 11 scalar DP transitions.
    kwargs = dict(policy="gittins", seed=8, prior_mean=prior_mean, prior_variance=prior_variance,
                  batch_size=4, per_arm_cost=prices, cost_mode=cost_mode, budget_fraction=1.)
    expected = reference_simulation(matrix, **kwargs)
    actual = comparison.simulate_policy(matrix, roots_lookup=roots, record_cells=True, **kwargs)
    assert_same_trajectory(actual, expected)
    np.testing.assert_array_equal(actual["counts"], [11, 11, 11])
    np.testing.assert_array_equal(actual["observed_sums"], matrix.sum(axis=1))
    np.testing.assert_array_equal(expected["posterior_variances"], [0., 0., 0.])
    np.testing.assert_array_equal(expected["indices"], matrix.mean(axis=1))
    assert sorted(actual["n_new_samples"].tolist()) == [3] * 3 + [4] * 6
    assert actual["natural_stop_cum_eval"] == expected["natural_stop_cum_eval"]


@pytest.mark.parametrize("prior_variance", [.04, .01])
def test_batch_lookup_matches_separate_state_solves(comparison, prior_variance):
    prices = np.array([.25, .45, .25])
    actual = comparison.build_roots_lookup(
        prior_variance=prior_variance, tau_sq_cell=.25, n_examples=11,
        batch_size=4, costs_per_arm=prices, n_points=257)
    expected = np.array([[paper_root(11, 4, prior_variance, float(price), stage)
                          for stage in range(4)] for price in prices])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-7)
    np.testing.assert_array_equal(actual[0], actual[2])
    np.testing.assert_array_equal(actual[:, -1], 0.)
    batches, sigmas = paper_schedule(11, 4, prior_variance)
    np.testing.assert_array_equal(batches, [4, 4, 3])
    final_v = 1 / (1 / prior_variance + 8 / .25)
    a = 1 + .25 / (11 * prior_variance)
    assert sigmas[-1] == pytest.approx(a * final_v / np.sqrt(final_v + .25 / 3))


@pytest.mark.parametrize("prior_variance", [.04, .01])
def test_batch_size_one_alone_matches_historical_per_cell_dp(comparison, prior_variance):
    prices = np.array([.25, .45, .25], dtype=np.float32)
    expected = np.asarray(compute_finite_population_roots_lookup_table(
        prior_variance=prior_variance, tau_sq_cell=.25, n_examples=11,
        costs_per_arm=prices * np.float32(1e-4), n_points=257), dtype=np.float32)
    actual = comparison.build_roots_lookup(
        prior_variance=prior_variance, tau_sq_cell=.25, n_examples=11,
        batch_size=1, costs_per_arm=prices, n_points=257)
    np.testing.assert_array_equal(actual, expected)


def test_gittins_rejects_per_cell_roots_for_batch_actions(comparison):
    with pytest.raises(ValueError, match=r"H\+1|batch|ceil"):
        comparison.simulate_policy(
            np.ones((3, 11)), policy="gittins", seed=0,
            prior_mean=.5, prior_variance=.04, batch_size=4,
            per_arm_cost=np.ones(3), cost_mode="unit", roots_lookup=np.zeros((3, 12)))


@pytest.mark.parametrize("bad_cache", ["per_cell_shape", "missing_batch_version"])
def test_prepare_rejects_incompatible_cached_model(comparison, tmp_path, monkeypatch, bad_cache):
    gsm8k_case()

    def cheap_roots(**kwargs):
        horizon = (kwargs["n_examples"] + kwargs["batch_size"] - 1) // kwargs["batch_size"]
        return np.zeros((len(kwargs["costs_per_arm"]), horizon + 1), dtype=np.float32)

    monkeypatch.setattr(comparison, "build_roots_lookup", cheap_roots)
    kwargs = dict(batch_size=8, n_points=65, matrix_seeds=[1], run_seeds=[0])
    settings = comparison.prepare(tmp_path, **kwargs)
    assert settings["gittins_dp_transitions"] == "batch"
    assert settings["gittins_dp_horizon"] == 125
    assert settings["gittins_batch_sizes"] == [8] * 125
    path = tmp_path / "roots.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if bad_cache == "per_cell_shape":
        values["G_unit_roots"] = np.zeros((122, 1001), dtype=np.float32)
    else:
        values.pop("gittins_model_version")
    np.savez(path, **values)
    with pytest.raises(ValueError, match="batch-level"):
        comparison.prepare(tmp_path, **kwargs)


@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
@pytest.mark.parametrize("cost_mode", ["unit", "cost"])
def test_ts_matches_unchanged_native_policy(comparison, prior_mean, prior_variance, cost_mode):
    matrix = (np.random.default_rng(37).random((3, 24)) < np.array([.2, .6, .8])[:, None]).astype(np.float32)
    kwargs = dict(policy="ts", seed=8, prior_mean=prior_mean, prior_variance=prior_variance,
                  batch_size=4, per_arm_cost=np.array([.25, .45, .25]),
                  cost_mode=cost_mode, budget_fraction=.5)
    expected = reference_simulation(matrix, **kwargs)
    actual = comparison.simulate_policy(matrix, record_cells=True, **kwargs)
    assert_same_trajectory(actual, expected)


@pytest.mark.parametrize("policy", ["gittins", "ts"])
def test_both_methods_use_same_empirical_lcb_when_mean_only_would_disagree(comparison, policy):
    # Seed0 chooses arm0 under TS; Gittins starts from its fixed-order tie.
    # Every cell on that arm is identical, so their different permutation RNG
    # histories give precisely the same posterior after the first batch.
    matrix = np.array([[.45] * 10, [.2] * 10], dtype=np.float32)
    actual = comparison.simulate_policy(
        matrix, policy=policy, seed=0, prior_mean=.5, prior_variance=.04,
        batch_size=4, per_arm_cost=np.ones(2), cost_mode="unit",
        roots_lookup=np.zeros((2, 4)) if policy == "gittins" else None,
        budget_fraction=1., max_steps=1)
    observed = np.full_like(matrix, np.nan)
    observed[0, :4] = matrix[0, :4]
    _, _, means, variances = paper_posterior(observed, .5, .04)
    assert means.argmax() == 1
    assert (means - np.sqrt(variances)).argmax() == 0
    assert actual["pulled_arm"].tolist() == [0]
    assert actual["recommended_arm"].tolist() == [0]
    assert actual["recommended_mean"].tolist() == [float(means[0])]


def gsm8k_case():
    matrix_path = ROOT / "data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy"
    archive_path = ROOT / "data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz"
    if not matrix_path.is_file() or not archive_path.is_file():
        pytest.skip("GSM8K supplied-data check requires the bundled matrices")
    matrix = np.load(matrix_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        prices = archive["cost"] / matrix.shape[1]
    return matrix, prices


@pytest.mark.parametrize("policy", ["gittins", "ts"])
@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
@pytest.mark.parametrize("cost_mode", ["unit", "cost"])
def test_gsm8k_full_budget_paper_gittins_or_first_fifty_native_ts(
    comparison, policy, prior_mean, prior_variance, cost_mode,
):
    matrix, prices = gsm8k_case()
    roots = None
    if policy == "gittins":
        dp_prices = np.ones(len(prices)) if cost_mode == "unit" else prices
        roots = comparison.build_roots_lookup(
            prior_variance=prior_variance, tau_sq_cell=.25, n_examples=1000,
            batch_size=8, costs_per_arm=dp_prices, n_points=1025)
        assert roots.shape == (122, 126)
    kwargs = dict(policy=policy, seed=11, prior_mean=prior_mean, prior_variance=prior_variance,
                  batch_size=8, per_arm_cost=prices, cost_mode=cost_mode,
                  budget_fraction=.1, max_steps=50 if policy == "ts" else None)
    expected = reference_simulation(matrix, n_points=1025, **kwargs)
    actual = comparison.simulate_policy(matrix, roots_lookup=roots, record_cells=True, **kwargs)
    assert_same_trajectory(actual, expected)
