"""Joint-belief and stopping integration for the marginal-index approximation."""

from pathlib import Path
from functools import lru_cache
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from correlated_gittins_policy import MarginalIndexGittins
from correlated_ts_policy import CorrelatedGaussianTS


FEATURES = torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.float64)
PRICES = np.array([.25, 1., 2.])


class RecordingRoots:
    """A known lookup function exposes which marginal state drives indices."""

    def __init__(self, zero=False):
        self.n_examples = 7
        self.batch_size = 3
        self.prior_variance = .04
        self.tau_sq_cell = .25
        self.cost_scaling_factor = 1e-4
        self.n_arms = 3
        self.costs_per_arm = PRICES
        self.calls = []
        self.zero = zero

    def lookup(self, variances, remaining_items, costs_per_arm):
        v, m, prices = map(np.asarray, (variances, remaining_items, costs_per_arm))
        if not np.isin(prices, self.costs_per_arm).all():
            raise ValueError("raw price is absent from root table")
        self.calls.append((v.copy(), m.copy(), prices.copy()))
        if self.zero:
            return np.zeros_like(v, dtype=np.float32)
        return (-np.sqrt(v) * m / self.n_examples + .01 * prices).astype(np.float32)


def make_policy(table=None, **kwargs):
    return MarginalIndexGittins(
        FEATURES, root_table=table if table is not None else RecordingRoots(),
        n_examples=7, batch_size=3, per_arm_cost=PRICES, **kwargs,
    )


@pytest.mark.parametrize("kernel_mode", ["fixed", "fit"])
def test_policy_reuses_the_correlated_ts_belief_and_updates_unpulled_indices(kernel_mode):
    table = RecordingRoots()
    policy = make_policy(table, kernel_mode=kernel_mode, fit_every=2, fit_maxiter=4)
    reference = CorrelatedGaussianTS(
        FEATURES, n_examples=7, kernel_mode=kernel_mode, fit_every=2, fit_maxiter=4,
    )
    before = policy.indices()
    for arm, values in ((0, [1., 1., 1.]), (1, [0., 1., 0.])):
        batch = torch.tensor(values, dtype=torch.float64)
        policy.observe(arm, batch)
        reference.observe(arm, batch)
        torch.testing.assert_close(policy.belief.latent_mean, reference.latent_mean)
        torch.testing.assert_close(policy.belief.latent_covariance, reference.latent_covariance)
        torch.testing.assert_close(policy.counts, reference.counts)
        for observed, expected in zip(policy.empirical_target_moments(), reference.empirical_target_moments()):
            torch.testing.assert_close(observed, expected)
        v, m, prices = table.calls[-1]
        np.testing.assert_allclose(v, policy.belief.latent_covariance.diagonal().numpy())
        np.testing.assert_array_equal(m, 7 - policy.counts.numpy())
        np.testing.assert_array_equal(prices, PRICES)
        means, _ = policy.empirical_target_moments()
        expected = means.float() - torch.as_tensor(
            -np.sqrt(v) * m / 7 + .01 * prices, dtype=torch.float32,
        )
        torch.testing.assert_close(policy.indices(), expected, rtol=0, atol=0)
    assert policy.counts[2] == 0
    assert policy.indices()[2] != before[2]
    assert policy.kernel_metadata == reference.kernel_metadata


def test_identity_correlation_keeps_unobserved_arms_unchanged():
    policy = make_policy(kernel_mode="fixed", correlation_matrix=torch.eye(3))
    before = policy.indices()
    policy.observe(0, torch.tensor([1., 0., 1.]))
    torch.testing.assert_close(policy.indices()[1:], before[1:], rtol=0, atol=0)


def test_completed_scores_and_stop_signal_use_exact_terminal_empirical_target():
    table = RecordingRoots(zero=True)
    policy = make_policy(table, kernel_mode="fixed", correlation_matrix=torch.eye(3))
    policy.observe(0, torch.ones(7))
    assert policy.indices()[0].item() == 1.
    assert np.all(table.calls[-1][1] > 0)  # Completed arms never query a future root.
    assert policy.select_arm(allow_early_stop=True) is None
    assert policy.select_arm(allow_early_stop=False) in (1, 2)
    natural, recommendation = [None], [None]
    policy.stopping_diagnostics(7, natural, recommendation)
    assert natural == [7]
    assert recommendation == [7]
    policy.stopping_diagnostics(99, natural, recommendation)
    assert natural == [7]
    assert recommendation == [7]
    policy.observe(1, torch.zeros(7))
    policy.observe(2, torch.ones(7) * .5)
    torch.testing.assert_close(policy.indices(), torch.tensor([1., 0., .5]))
    assert policy.select_arm() is None
    assert policy.select_arm(allow_early_stop=True) is None


def test_recommendation_penalty_cannot_change_allocation_indices():
    policy = make_policy(kernel_mode="fixed")
    policy.observe(0, torch.tensor([1., 0., 0.]))
    indices = policy.indices()
    selected = policy.select_arm()
    for penalty in (0., 1., 5.):
        recommended, means, covariance = policy.recommend(penalty)
        expected_scores = means.float().double() - penalty * covariance.diagonal().sqrt()
        assert recommended == int(expected_scores.argmax())
        torch.testing.assert_close(policy.indices(), indices, rtol=0, atol=0)
        assert policy.select_arm() == selected


@pytest.mark.parametrize("name,value", [
    ("n_examples", 8), ("batch_size", 2), ("prior_variance", .01),
    ("tau_sq_cell", .5), ("cost_scaling_factor", 1e-3), ("n_arms", 2),
])
def test_incompatible_precomputed_table_is_rejected_before_observation(name, value):
    table = RecordingRoots()
    setattr(table, name, value)
    with pytest.raises(ValueError, match="root_table"):
        make_policy(table, kernel_mode="fixed")


def test_missing_arm_price_is_rejected_before_spending_a_batch():
    table = RecordingRoots()
    table.costs_per_arm = np.array([.25, 1.])
    with pytest.raises(ValueError, match="price"):
        make_policy(table, kernel_mode="fixed")


@lru_cache(maxsize=None)
def _small_real_table(n_examples, batch_size, prior_variance, prices):
    from marginal_gittins_lookup import build_marginal_gittins_root_table

    return build_marginal_gittins_root_table(
        n_examples=n_examples, batch_size=batch_size, prior_variance=prior_variance,
        tau_sq_cell=.25, costs_per_arm=np.asarray(prices), cost_scaling_factor=1e-4,
        n_points=257, n_arms=3,
        effective_prior_variances=np.array([0., prior_variance * .1, prior_variance]),
    )


@pytest.mark.parametrize("n_examples,batch_size", [(12, 4), (11, 4)])
@pytest.mark.parametrize("prior_mean,prior_variance", [(.5, .04), (.2, .01)])
@pytest.mark.parametrize("prices", [(1., 1., 1.), (.25, 1., 2.)])
def test_zero_covariance_reproduces_native_indices_choices_recommendations_and_stops(
    n_examples, batch_size, prior_mean, prior_variance, prices,
):
    from gittins_policy import gittins_index_exploration, gittins_post_pull_update
    from gittins_shrinking_posterior import compute_finite_population_batch_roots_lookup_table
    from simple_regret_recommend import posterior_incumbent

    table = _small_real_table(n_examples, batch_size, prior_variance, prices)
    policy = MarginalIndexGittins(
        FEATURES, root_table=table, n_examples=n_examples, batch_size=batch_size,
        per_arm_cost=np.asarray(prices), prior_mean=prior_mean,
        prior_variance=prior_variance, kernel_mode="fixed", correlation_matrix=torch.eye(3),
    )
    original_roots = torch.tensor(np.asarray(compute_finite_population_batch_roots_lookup_table(
        n_examples=n_examples, batch_size=batch_size, prior_variance=prior_variance,
        tau_sq_cell=.25, costs_per_arm=np.asarray(prices), cost_scaling_factor=1e-4,
        n_points=257,
    )), dtype=torch.float32)
    truth = torch.as_tensor(
        np.random.default_rng(27).random((3, n_examples)) < np.array([.2, .8, .4])[:, None],
        dtype=torch.float32,
    )
    observed = torch.full_like(truth, torch.nan)
    cache = torch.full((3,), torch.inf)
    options = dict(
        prior_mean=prior_mean, prior_variance=prior_variance,
        obs_noise_variance=.25 / batch_size, batch_observation_model=True,
        batch_size=batch_size, cost_per_transition=prices, cost_scaling_factor=1e-4,
        n_gittins_grid_points=257, roots_lookup_table=original_roots,
    )
    torch.manual_seed(29)
    generator = torch.Generator().manual_seed(29)
    previous, evaluated = None, 0
    original_natural, original_aware, marginal_natural, marginal_aware = [None], [None], [None], [None]
    while True:
        native_batch = gittins_index_exploration(
            observed, cached_scores=cache, recompute_arms=None if previous is None else [previous],
            allow_early_stop=False, **options,
        )
        if native_batch is None:
            assert policy.select_arm() is None
            break
        torch.testing.assert_close(policy.indices(), cache, rtol=0, atol=2e-7)
        arm = policy.select_arm()
        assert arm == int(native_batch[0, 0])
        unseen = torch.where(observed[arm].isnan())[0]
        columns = unseen[torch.randperm(len(unseen), generator=generator)[:batch_size]]
        torch.testing.assert_close(columns, native_batch[1], rtol=0, atol=0)
        observed[arm, columns] = truth[arm, columns]
        policy.observe(arm, truth[arm, columns])
        evaluated += len(columns)
        previous = arm
        gittins_post_pull_update(
            observed, cached_scores=cache, recompute_arms=[arm], sim_cum_eval=evaluated,
            natural_stop_cum_eval_holder=original_natural,
            recommendation_aware_stop_cum_eval_holder=original_aware,
            recommendation_std_penalty=1., **options,
        )
        policy.stopping_diagnostics(evaluated, marginal_natural, marginal_aware)
        original_recommendation, _ = posterior_incumbent(
            observed, prior_mean=prior_mean, prior_variance=prior_variance,
            tau_sq_cell=.25, std_penalty=1.,
        )
        assert policy.recommend()[0] == original_recommendation
        assert original_natural == marginal_natural
        assert original_aware == marginal_aware
    assert evaluated == truth.numel()
    torch.testing.assert_close(policy.indices(), truth.mean(1))
