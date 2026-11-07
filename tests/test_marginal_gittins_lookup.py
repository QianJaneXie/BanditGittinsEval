"""Independent finite-target checks for the marginal-index root cache."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import marginal_gittins_lookup as lookup_module
from marginal_gittins_lookup import (
    MarginalGittinsRootTable,
    build_marginal_gittins_root_table,
    compute_marginal_stopping_root,
    marginal_transition_stds,
)
from gittins_shrinking_posterior import compute_finite_population_batch_roots_lookup_table


def _independent_future_schedule(variance, remaining, *, n_examples, batch_size, noise=.25):
    """Condition the empirical target on each future batch using Cov(F,Y)."""
    sigmas, sizes = [], []
    v, m = float(variance), int(remaining)
    while m:
        b = min(batch_size, m)
        # F contains all m remaining cells, including this observed batch.
        cov_target_observation = (m * v + noise) / n_examples
        observation_variance = v + noise / b
        sigmas.append(cov_target_observation / np.sqrt(observation_variance))
        sizes.append(b)
        v = v * (noise / b) / observation_variance
        m -= b
    return np.asarray(sigmas), np.asarray(sizes)


def _normal_pdf(x):
    return np.exp(-x*x/2) / np.sqrt(2*np.pi)


def _last_batch_q(state, sigma, cost):
    return state * ndtr(state / sigma) + sigma * _normal_pdf(state / sigma) - cost


def _two_batch_root(sigmas, costs):
    """Actual Gaussian integration, independent of the project's FFT solver."""
    r1 = brentq(lambda s: _last_batch_q(s, sigmas[1], costs[1]), -3., 3.)

    def first_q(state):
        integral, _ = quad(
            lambda z: _last_batch_q(state + sigmas[0]*z, sigmas[1], costs[1]) * _normal_pdf(z),
            (r1-state) / sigmas[0], np.inf, epsabs=1e-11,
        )
        return integral - costs[0]

    return brentq(first_q, -3., 3., xtol=1e-11), r1


@lru_cache(maxsize=None)
def _table():
    return build_marginal_gittins_root_table(
        prior_variance=.04, tau_sq_cell=.25, n_examples=11, batch_size=4,
        costs_per_arm=np.array([.5, 1.5]), n_arms=3,
        effective_prior_variances=np.array([0., .004, .02, .04]),
        n_points=513, cost_scaling_factor=.01,
    )


@pytest.mark.parametrize("variance", [.04, .002, 0.])
@pytest.mark.parametrize("remaining", [11, 9, 3, 0])
def test_future_transition_variance_matches_independent_gaussian_conditioning(variance, remaining):
    expected, batches = _independent_future_schedule(
        variance, remaining, n_examples=11, batch_size=4,
    )
    actual = np.asarray(marginal_transition_stds(
        variance, remaining, n_examples=11, batch_size=4, tau_sq_cell=.25,
    ))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-9)
    np.testing.assert_allclose(np.square(actual).sum(),
        (remaining**2 * variance + remaining * .25) / 11**2, rtol=3e-6, atol=1e-9)
    assert int(batches.sum()) == remaining


def test_known_latent_mean_still_has_unrevealed_item_uncertainty():
    actual = np.asarray(marginal_transition_stds(
        0., 9, n_examples=11, batch_size=4, tau_sq_cell=.25,
    ))
    np.testing.assert_allclose(actual, np.sqrt(np.array([4, 4, 1])*.25)/11,
                               rtol=2e-6, atol=1e-9)
    assert np.all(actual > 0)


def test_two_batch_root_matches_independent_bellman_quadrature():
    # N=9, n=4, m=5: an irregular boundary whose future batches are [3,2].
    n_examples, remaining, batch_size = 9, 5, 3
    effective_variance, noise, raw_cost, scale = .02, .25, 1.3, .015
    variance = 1/(1/effective_variance + (n_examples-remaining)/noise)
    sigmas, batches = _independent_future_schedule(
        variance, remaining, n_examples=n_examples, batch_size=batch_size,
    )
    expected, _ = _two_batch_root(sigmas, scale * raw_cost * batches)
    actual = compute_marginal_stopping_root(
        variance, remaining, raw_cost, n_examples=n_examples,
        batch_size=batch_size, tau_sq_cell=noise,
        cost_scaling_factor=scale, n_points=4097,
    )
    grid_step = (1.01*scale*raw_cost*n_examples
                 + 10*np.sqrt(effective_variance+noise/n_examples)) / 4096
    assert float(actual) == pytest.approx(expected, abs=3*grid_step)


def test_partial_final_batch_is_charged_only_for_its_actual_items():
    n_examples, remaining, batch_size = 9, 2, 3
    effective_variance, noise, raw_cost, scale = .02, .25, 1.3, .03
    variance = 1/(1/effective_variance + (n_examples-remaining)/noise)
    sigma = _independent_future_schedule(
        variance, remaining, n_examples=n_examples, batch_size=batch_size,
    )[0][0]
    expected = brentq(lambda s: _last_batch_q(s, sigma, scale*remaining*raw_cost), -3., 3.)
    wrong_full_batch = brentq(lambda s: _last_batch_q(s, sigma, scale*batch_size*raw_cost), -3., 3.)
    actual = compute_marginal_stopping_root(
        variance, remaining, raw_cost, n_examples=n_examples,
        batch_size=batch_size, tau_sq_cell=noise,
        cost_scaling_factor=scale, n_points=4097,
    )
    step = (1.01*scale*raw_cost*n_examples+10*np.sqrt(effective_variance+noise/n_examples))/4096
    assert float(actual) == pytest.approx(expected, abs=3*step)
    assert abs(float(actual)-wrong_full_batch) > 10*step


def test_direct_solver_supports_variances_outside_a_fixed_prior_cache():
    # A stand-alone marginal stopping problem is meaningful even if it cannot
    # arise under the cache's particular fixed-diagonal prior and own history.
    variance, remaining, raw_cost, scale = .1, 3, .5, .01
    sigma = _independent_future_schedule(
        variance, remaining, n_examples=11, batch_size=4,
    )[0][0]
    cost = scale*remaining*raw_cost
    expected = brentq(lambda s: _last_batch_q(s, sigma, cost), -3., 3.)
    actual = compute_marginal_stopping_root(
        variance, remaining, raw_cost, n_examples=11, batch_size=4,
        tau_sq_cell=.25, cost_scaling_factor=scale, n_points=4097,
    )
    step = (1.01*cost+10*sigma)/4096
    assert float(actual) == pytest.approx(expected, abs=3*step)


def test_zero_cross_covariance_anchor_reproduces_original_batch_roots():
    table = _table()
    prices = np.array([.5, 1.5])
    original = np.asarray(compute_finite_population_batch_roots_lookup_table(
        prior_variance=.04, tau_sq_cell=.25, n_examples=11, batch_size=4,
        costs_per_arm=prices, cost_scaling_factor=.01, n_points=513,
    ))
    for stage, observed in enumerate([0, 4, 8, 11]):
        remaining = 11-observed
        variance = 1/(1/.04 + observed/.25)
        actual = table.lookup(variance, remaining, prices)
        np.testing.assert_allclose(actual, original[:, stage], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(table.lookup([0., .002, .04], 0, .5), 0.)


def test_all_remaining_counts_include_irregular_batch_residues():
    table = _table()
    # This includes n=2,m=9, requiring future batches [4,4,1].
    for remaining in range(1, 12):
        variance = 1/(1/.02 + (11-remaining)/.25)
        actual = float(table.lookup(variance, remaining, .5))
        expected = compute_marginal_stopping_root(
            variance, remaining, .5, n_examples=11, batch_size=4,
            tau_sq_cell=.25, cost_scaling_factor=.01, n_points=513,
        )
        step = (1.01*.01*.5*11+10*np.sqrt(.02+.25/11))/512
        assert actual == pytest.approx(float(expected), abs=2*step)


def test_interpolation_is_in_current_variance_and_uses_only_cached_roots(monkeypatch):
    table = _table()
    remaining, observed = 7, 4
    v0 = 1/(1/.004+observed/.25)
    v1 = 1/(1/.02+observed/.25)
    expected = .5 * (table.roots[0, 1, remaining] + table.roots[0, 2, remaining])

    def no_solver(*args, **kwargs):
        raise AssertionError("Online lookup called a stopping-problem solver")

    monkeypatch.setattr(lookup_module, "compute_marginal_stopping_root", no_solver)
    if hasattr(lookup_module, "_compute_roots_for_random_walk"):
        monkeypatch.setattr(lookup_module, "_compute_roots_for_random_walk", no_solver)
    actual = table.lookup(.5*(v0+v1), remaining, .5)
    assert float(actual) == pytest.approx(float(expected), abs=2e-7)


@pytest.mark.parametrize("remaining,effective_variance", [(11, .0006), (9, .003), (6, .012), (1, .027)])
def test_held_out_variance_interpolation_agrees_with_finer_direct_fft(remaining, effective_variance):
    table = _interpolation_table()
    variance = 1/(1/effective_variance + (11-remaining)/.25)
    actual = float(table.lookup(variance, remaining, 1.5))
    direct = compute_marginal_stopping_root(
        variance, remaining, 1.5, n_examples=11, batch_size=4,
        tau_sq_cell=.25, cost_scaling_factor=.01, n_points=8193,
    )
    step = (1.01*.01*1.5*11+10*np.sqrt(.04+.25/11))/1024
    assert actual == pytest.approx(float(direct), abs=3*step)


@lru_cache(maxsize=None)
def _interpolation_table():
    return build_marginal_gittins_root_table(
        prior_variance=.04, tau_sq_cell=.25, n_examples=11, batch_size=4,
        costs_per_arm=np.array([1.5]), n_arms=3, n_points=1025,
        cost_scaling_factor=.01,
        effective_prior_variances=np.r_[0., np.geomspace(.0001, .04, 17)],
    )


def test_lookup_rejects_unphysical_variances_and_unsupported_costs():
    table = _table()
    own_only = 1/(1/.04+4/.25)
    with pytest.raises(ValueError):
        table.lookup(own_only*1.1, 7, .5)
    with pytest.raises(ValueError):
        table.lookup(.002, 7, 7.0)
    with pytest.raises(ValueError):
        table.lookup(.002, 7.5, .5)


def test_saved_table_preserves_roots_and_rejects_incompatible_metadata(tmp_path):
    table = _table()
    path = tmp_path/"roots.npz"
    table.save(path)
    restored = MarginalGittinsRootTable.load(path, n_examples=11, batch_size=4, prior_variance=.04)
    np.testing.assert_array_equal(restored.roots, table.roots)
    np.testing.assert_array_equal(restored.lookup([.001, .003], [9, 4], [.5, 1.5]),
                                  table.lookup([.001, .003], [9, 4], [.5, 1.5]))
    with pytest.raises(ValueError):
        MarginalGittinsRootTable.load(path, batch_size=8)
    with np.load(path, allow_pickle=False) as original:
        changed = {key: original[key].copy() for key in original.files}
    changed["roots"][0, 1, 7] += .1
    np.savez_compressed(path, **changed)
    with pytest.raises(ValueError):
        MarginalGittinsRootTable.load(path)
