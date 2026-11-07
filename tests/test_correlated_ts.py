"""Independent Gaussian-conditioning and joint-sampling checks for correlated TS."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from botorch.models import MixedSingleTaskGP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from correlated_ts_policy import CorrelatedGaussianTS
from simple_regret_recommend import finite_population_posterior_moments


FEATURES = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def default_botorch_correlation(features):
    """Read the actual all-categorical BO covariance, without fitting labels."""
    features = torch.as_tensor(features, dtype=torch.float64)
    # Observations are immaterial: this accesses the prior covariance module,
    # never a posterior conditioned on these dummy values.
    dummy_targets = torch.linspace(-1, 1, len(features), dtype=features.dtype)
    dummy_targets = dummy_targets / dummy_targets.std()
    model = MixedSingleTaskGP(
        features, dummy_targets[:, None],
        cat_dims=list(range(features.shape[1])),
        train_Yvar=torch.full((len(features), 1), .25, dtype=features.dtype),
    )
    with torch.no_grad():
        covariance = model.covar_module(features).to_dense().numpy()
    scale = np.sqrt(np.diag(covariance))
    return covariance / scale[:, None] / scale[None, :]


def design_matrix_posterior(prior_covariance, observations, mean, noise):
    """Condition all individual observations together using the MVN formula."""
    dimension = len(prior_covariance)
    if not observations:
        return np.full(dimension, mean), prior_covariance.copy()
    arms, values = zip(*observations)
    design = np.eye(dimension)[list(arms)]
    joint_cross = prior_covariance @ design.T
    observation_covariance = design @ joint_cross + noise * np.eye(len(values))
    posterior_mean = mean + joint_cross @ np.linalg.solve(
        observation_covariance, np.asarray(values) - mean,
    )
    posterior_covariance = prior_covariance - joint_cross @ np.linalg.solve(
        observation_covariance, joint_cross.T,
    )
    return posterior_mean, posterior_covariance


@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
def test_default_kernel_matches_actual_configuration_kernel_and_prior_marginals(
    prior_mean, prior_variance,
):
    policy = CorrelatedGaussianTS(
        FEATURES, n_examples=7, prior_mean=prior_mean,
        prior_variance=prior_variance, tau_sq_cell=.25, kernel_mode="fixed",
    )
    expected = prior_variance * default_botorch_correlation(FEATURES)
    np.testing.assert_allclose(as_numpy(policy.latent_covariance), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(as_numpy(policy.latent_mean), prior_mean, atol=1e-14)
    means, covariance = map(as_numpy, policy.empirical_target_moments())
    np.testing.assert_allclose(means, prior_mean, atol=1e-14)
    np.testing.assert_allclose(covariance, expected + (.25 / 7) * np.eye(4), atol=1e-12)
    assert policy.fit_count == 0


@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
def test_repeated_batches_match_full_multivariate_observation_design(
    prior_mean, prior_variance,
):
    n_examples, noise = 5, .25
    policy = CorrelatedGaussianTS(
        FEATURES, n_examples=n_examples, prior_mean=prior_mean,
        prior_variance=prior_variance, tau_sq_cell=noise, kernel_mode="fixed",
    )
    prior_covariance = prior_variance * default_botorch_correlation(FEATURES)
    observations = []
    counts = np.zeros(4, dtype=int)
    sums = np.zeros(4)
    # Repeated batches, unequal counts, and a final batch completing one arm.
    for arm, values in [(0, [.7, .2]), (2, [.1, .9, .4]), (0, [.3]),
                        (1, [.6, .8, .2]), (1, [.5, .9])]:
        policy.observe(arm, torch.tensor(values, dtype=torch.float64))
        observations.extend((arm, value) for value in values)
        counts[arm] += len(values)
        sums[arm] += sum(values)
        latent_mean, latent_covariance = design_matrix_posterior(
            prior_covariance, observations, prior_mean, noise,
        )
        np.testing.assert_array_equal(as_numpy(policy.counts), counts)
        np.testing.assert_allclose(as_numpy(policy.sums), sums, atol=1e-13)
        np.testing.assert_allclose(as_numpy(policy.latent_mean), latent_mean, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(as_numpy(policy.latent_covariance), latent_covariance, atol=1e-11, rtol=1e-10)
        remaining = n_examples - counts
        expected_mean = (sums + remaining * latent_mean) / n_examples
        expected_covariance = (
            remaining[:, None] * latent_covariance * remaining[None, :]
            + np.diag(remaining * noise)
        ) / n_examples**2
        means, covariance = map(as_numpy, policy.empirical_target_moments())
        np.testing.assert_allclose(means, expected_mean, atol=1e-11)
        np.testing.assert_allclose(covariance, expected_covariance, atol=1e-11)
    # Full empirical targets collapse, although the latent mean remains noisy.
    assert means[1] == pytest.approx(sums[1] / n_examples)
    np.testing.assert_array_equal(covariance[1], 0)
    np.testing.assert_array_equal(covariance[:, 1], 0)
    assert as_numpy(policy.latent_covariance)[1, 1] > 0


def test_one_arm_observation_changes_unseen_correlated_arms():
    policy = CorrelatedGaussianTS(FEATURES, n_examples=10, kernel_mode="fixed")
    before_mean = as_numpy(policy.latent_mean).copy()
    before_covariance = as_numpy(policy.latent_covariance).copy()
    policy.observe(0, torch.ones(4, dtype=torch.float64))
    after_mean, after_covariance = map(as_numpy, [policy.latent_mean, policy.latent_covariance])
    assert np.all(after_mean[1:] > before_mean[1:])
    assert np.all(np.diag(after_covariance)[1:] < np.diag(before_covariance)[1:])
    assert as_numpy(policy.counts)[1:].sum() == 0


def test_identity_kernel_limit_recovers_independent_empirical_target_ts():
    policy = CorrelatedGaussianTS(
        FEATURES, n_examples=5, prior_mean=.2, prior_variance=.01,
        tau_sq_cell=.25, kernel_mode="fixed",
        correlation_matrix=torch.eye(4, dtype=torch.float64),
    )
    observed = torch.full((4, 5), float("nan"), dtype=torch.float64)
    for arm, values in [(0, [1., 0.]), (2, [0., 1., 1.]), (1, [1.] * 5)]:
        policy.observe(arm, torch.tensor(values, dtype=torch.float64))
        observed[arm, :len(values)] = torch.tensor(values, dtype=torch.float64)
    expected_mean, expected_variance = finite_population_posterior_moments(
        observed, prior_mean=.2, prior_variance=.01, tau_sq_cell=.25,
    )
    mean, covariance = policy.empirical_target_moments()
    torch.testing.assert_close(mean, expected_mean.to(torch.float64), rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(covariance, torch.diag(expected_variance), rtol=1e-10, atol=1e-12)
    seed = 73
    actual = policy.sample_target(generator=torch.Generator().manual_seed(seed))
    expected = expected_mean.to(torch.float64) + expected_variance.sqrt() * torch.randn(
        (4,), dtype=torch.float64, generator=torch.Generator().manual_seed(seed),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)


def test_category_relabeling_preserves_model_and_joint_draws():
    relabeled = torch.tensor([[101, -20], [101, 400], [-8, -20], [-8, 400]], dtype=torch.float64)
    policies = [CorrelatedGaussianTS(x, n_examples=9, kernel_mode="fixed")
                for x in (FEATURES, relabeled)]
    for policy in policies:
        policy.observe(1, torch.tensor([1., 0., 1.], dtype=torch.float64))
        policy.observe(2, torch.tensor([0., 1.], dtype=torch.float64))
    for left, right in zip(policies[0].empirical_target_moments(), policies[1].empirical_target_moments()):
        np.testing.assert_allclose(as_numpy(left), as_numpy(right), atol=1e-12)
    draws = [as_numpy(policy.sample_target(generator=torch.Generator().manual_seed(79)))
             for policy in policies]
    np.testing.assert_allclose(*draws, atol=1e-12)


def test_joint_draw_covariance_contains_cross_arm_dependence_and_residual_noise():
    features = FEATURES[:3]
    policy = CorrelatedGaussianTS(features, n_examples=11, kernel_mode="fixed")
    policy.observe(0, torch.tensor([1., 0.], dtype=torch.float64))
    policy.observe(2, torch.tensor([1.], dtype=torch.float64))
    expected_mean, expected_covariance = map(as_numpy, policy.empirical_target_moments())
    generator = torch.Generator().manual_seed(514)
    draws = np.stack([as_numpy(policy.sample_target(generator=generator)) for _ in range(12000)])
    # This catches an implementation that samples only independent marginals.
    assert np.min(expected_covariance[np.triu_indices(3, 1)]) > .004
    np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=.006, rtol=0)
    np.testing.assert_allclose(np.cov(draws.T), expected_covariance, atol=.0015, rtol=.08)


def test_joint_samples_are_seeded_and_completed_targets_are_exact():
    policy = CorrelatedGaussianTS(FEATURES, n_examples=3, kernel_mode="fixed")
    policy.observe(0, torch.tensor([0., 1., 0.], dtype=torch.float64))
    first = as_numpy(policy.sample_target(generator=torch.Generator().manual_seed(123)))
    second = as_numpy(policy.sample_target(generator=torch.Generator().manual_seed(123)))
    different = as_numpy(policy.sample_target(generator=torch.Generator().manual_seed(124)))
    np.testing.assert_array_equal(first, second)
    assert np.any(first[1:] != different[1:])
    assert first[0] == pytest.approx(1 / 3, abs=1e-15)
    np.testing.assert_array_equal(as_numpy(policy.counts), [3, 0, 0, 0])


def test_fit_cadence_uses_observed_data_and_preserves_marginal_prior():
    policy = CorrelatedGaussianTS(
        FEATURES, n_examples=20, prior_mean=.2, prior_variance=.01,
        tau_sq_cell=.25, kernel_mode="fit", fit_every=2, fit_maxiter=3,
    )
    assert policy.fit_count == 0
    policy.observe(0, torch.tensor([0., 0.], dtype=torch.float64))
    assert policy.fit_count == 0
    policy.observe(1, torch.tensor([1., 1., 1., 0.], dtype=torch.float64))
    assert policy.fit_count == 1
    policy.observe(0, torch.tensor([0.], dtype=torch.float64))
    assert policy.fit_count == 1
    policy.observe(2, torch.tensor([1., 0., 1.], dtype=torch.float64))
    assert policy.fit_count == 2
    metadata = policy.kernel_metadata
    assert isinstance(metadata, dict)
    assert np.isfinite(as_numpy(policy.latent_mean)).all()
    assert np.isfinite(as_numpy(policy.latent_covariance)).all()
    # Kernel fitting may change correlation but may not change μ0, v0, or τ².
    assert policy.prior_mean == .2
    assert policy.prior_variance == .01
    assert policy.tau_sq_cell == .25
    np.testing.assert_allclose(as_numpy(policy.prior_covariance).diagonal(), .01, atol=1e-12)


def test_kernel_fit_likelihood_uses_unique_seen_arm_means_and_correct_noise():
    policy = CorrelatedGaussianTS(
        FEATURES, n_examples=10, prior_mean=.2, prior_variance=.04,
        tau_sq_cell=.25, kernel_mode="fit", fit_every=100, fit_maxiter=4,
    )
    batches = [(0, [0., 1.]), (2, [1., 1., 0., 1.]), (0, [0.])]
    observations = []
    for arm, values in batches:
        policy.observe(arm, torch.tensor(values, dtype=torch.float64))
        observations.extend((arm, value) for value in values)
    # Repeated cumulative means are not separate independent observations.
    indices = [0, 2]
    counts = np.array([3, 4])
    sums = np.array([1., 3.])
    prior = as_numpy(policy.prior_covariance).copy()
    covariance = prior[np.ix_(indices, indices)] + np.diag(.25 / counts)
    residual = sums / counts - .2
    expected_nll = .5 * (residual @ np.linalg.solve(covariance, residual)
                          + np.linalg.slogdet(covariance)[1])
    policy.fit_kernel()
    diagnostic = policy.kernel_metadata["fit_diagnostics"][-1]
    assert diagnostic["observed_arms"] == 2
    assert diagnostic["negative_log_likelihood_before"] == pytest.approx(expected_nll, abs=1e-11)
    assert diagnostic["negative_log_likelihood_after"] <= expected_nll + 1e-8
    # The changed kernel must recondition the same real observations, never
    # double-counting nested prefixes or conditioning unseen arms on zeros.
    expected_mean, expected_covariance = design_matrix_posterior(
        as_numpy(policy.prior_covariance), observations, .2, .25,
    )
    np.testing.assert_allclose(as_numpy(policy.latent_mean), expected_mean, atol=1e-11)
    np.testing.assert_allclose(as_numpy(policy.latent_covariance), expected_covariance, atol=1e-11)
    np.testing.assert_allclose(as_numpy(policy.prior_covariance).diagonal(), .04, atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"n_examples": 0}, {"prior_variance": 0}, {"tau_sq_cell": 0},
    {"prior_mean": float("nan")}, {"kernel_mode": "unknown"},
    {"fit_every": 0}, {"fit_maxiter": 0},
])
def test_invalid_model_settings_are_rejected(kwargs):
    parameters = {"n_examples": 5, **kwargs}
    with pytest.raises(ValueError):
        CorrelatedGaussianTS(FEATURES, **parameters)


def test_observe_rejects_nonfinite_values_and_overfull_rows():
    policy = CorrelatedGaussianTS(FEATURES, n_examples=3, kernel_mode="fixed")
    with pytest.raises(ValueError):
        policy.observe(0, torch.tensor([float("nan")], dtype=torch.float64))
    with pytest.raises(ValueError):
        policy.observe(0, torch.zeros(4, dtype=torch.float64))
    np.testing.assert_array_equal(as_numpy(policy.counts), 0)
