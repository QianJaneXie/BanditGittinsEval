"""Batch-boundary empirical-target DP checks against the paper's equations."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

from gittins_shrinking_posterior import (
    compute_finite_population_batch_roots_lookup_table,
    compute_finite_population_roots_lookup_table,
)


def _normal_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _last_batch_q(state, sigma, cost):
    # Q_H(s)=s, hence Q_{H-1}(s)=E[(s+sigma*Z)^+]-cost.
    return state * ndtr(state / sigma) + sigma * _normal_pdf(state / sigma) - cost


def test_batch_size_one_preserves_the_per_cell_dp_and_absolute_costs():
    raw_prices = np.array([0.75, 2.5], dtype=np.float32)
    scale = 0.02
    actual = np.asarray(compute_finite_population_batch_roots_lookup_table(
        prior_variance=0.04, tau_sq_cell=0.25, n_examples=7, batch_size=1,
        costs_per_arm=raw_prices, cost_scaling_factor=scale, n_points=513,
    ))
    expected = np.asarray(compute_finite_population_roots_lookup_table(
        prior_variance=0.04, tau_sq_cell=0.25, n_examples=7,
        costs_per_arm=scale * raw_prices, n_points=513,
    ))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_partial_final_batch_matches_independent_bellman_quadrature():
    """Use actual Gaussian integration, without any project DP/lookup routine."""
    prior_variance, tau_sq, n_cells, batch_size = 0.04, 0.25, 5, 3
    raw_price, scale, n_points = 1.4, 0.015, 4097
    # The two observations are means of three and then two fresh cells.
    a = 1.0 + tau_sq / (n_cells * prior_variance)
    v0 = prior_variance
    v1 = 1.0 / (1.0 / prior_variance + batch_size / tau_sq)
    sigma0 = a * v0 / np.sqrt(v0 + tau_sq / 3)
    sigma1 = a * v1 / np.sqrt(v1 + tau_sq / 2)
    cost0, cost1 = scale * 3 * raw_price, scale * 2 * raw_price
    np.testing.assert_allclose(sigma0**2 + sigma1**2, prior_variance + tau_sq / n_cells)

    root1 = brentq(lambda s: _last_batch_q(s, sigma1, cost1), -2.0, 2.0)

    def first_batch_q(state):
        # Q_1 is increasing and positive above root1, so integrate only that
        # region in E[max(Q_1(state + sigma0*Z), 0)] - cost0.
        value, _ = quad(
            lambda z: _last_batch_q(state + sigma0 * z, sigma1, cost1) * _normal_pdf(z),
            (root1 - state) / sigma0, np.inf, epsabs=1e-11,
        )
        return value - cost0

    root0 = brentq(first_batch_q, -2.0, 2.0, xtol=1e-11)
    actual = np.asarray(compute_finite_population_batch_roots_lookup_table(
        prior_variance=prior_variance, tau_sq_cell=tau_sq, n_examples=n_cells,
        batch_size=batch_size, costs_per_arm=np.array([raw_price]),
        cost_scaling_factor=scale, n_points=n_points,
    ))
    assert actual.shape == (1, 3)
    # The implementation selects the first nonnegative grid value of Q.
    grid_step = (1.01 * (cost0 + cost1) + 10 * np.hypot(sigma0, sigma1)) / (n_points - 1)
    np.testing.assert_allclose(actual[0, :2], [root0, root1], atol=2 * grid_step, rtol=0)
    assert actual[0, 2] == 0.0

    # Charging a full batch for the final two cells would give a different
    # one-step Bellman root, even though the terminal payoff is unchanged.
    wrong_full_batch_cost_root = brentq(
        lambda s: _last_batch_q(s, sigma1, cost0), -2.0, 2.0
    )
    assert abs(float(actual[0, 1]) - wrong_full_batch_cost_root) > 10 * grid_step


def test_batch_roots_share_identical_prices_and_preserve_other_rows():
    options = dict(
        prior_variance=0.04, tau_sq_cell=0.25, n_examples=7,
        batch_size=3, cost_scaling_factor=0.02, n_points=513,
    )
    roots = np.asarray(compute_finite_population_batch_roots_lookup_table(
        costs_per_arm=np.array([1.0, 3.0, 1.0]), **options,
    ))
    shared = np.asarray(compute_finite_population_batch_roots_lookup_table(
        costs_per_arm=np.array([1.0, 1.0]), **options,
    ))
    assert roots.shape == (3, 4)  # Local batches have sizes 3, 3, 1.
    assert shared.shape == (1, 4)
    np.testing.assert_array_equal(roots[0], roots[2])
    np.testing.assert_array_equal(shared[0], roots[0])
    assert np.all(roots[1, :-1] > roots[0, :-1])
    np.testing.assert_array_equal(roots[:, -1], 0.0)
