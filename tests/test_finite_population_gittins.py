"""Finite-payoff acquisition, cost scaling, and terminal-state regressions."""

import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import torch

from gittins_lookup import compute_roots_lookup_table
from gittins_policy import gittins_index_exploration, gittins_post_pull_update
from gittins_shrinking_posterior import (
    compute_finite_population_roots_lookup_table,
    compute_gittins_shrinking_posterior_walk_batch_mean,
    finite_population_mean_scale,
    transition_stds_batch_mean_sequence,
    transition_stds_finite_population_posterior,
    transition_stds_shrinking_gaussian_posterior,
)
from simple_regret_recommend import finite_population_posterior_moments, posterior_moments


class FinitePopulationGittinsTests(unittest.TestCase):
    def test_remaining_transition_variance_equals_finite_posterior_variance(self):
        n_cells, prior_variance, noise = 7, 0.04, 0.25
        transitions = np.asarray(transition_stds_finite_population_posterior(
            jnp.float32(prior_variance), jnp.float32(noise), n_cells
        ))
        for count in (0, 2, 6, 7):
            with self.subTest(count=count):
                obs = torch.full((1, n_cells), float("nan"), dtype=torch.float64)
                obs[0, :count] = 0.3
                _, variance = finite_population_posterior_moments(
                    obs, prior_mean=0.5, prior_variance=prior_variance, tau_sq_cell=noise
                )
                np.testing.assert_allclose(
                    np.square(transitions[count:]).sum(), float(variance[0]),
                    rtol=2e-6, atol=1e-9,
                )

    def test_partial_batch_transitions_partition_remaining_uncertainty(self):
        n_cells, count, batch_size = 7, 2, 3
        prior_variance, noise = 0.04, 0.25
        latent_variance = 1.0 / (1.0 / prior_variance + count / noise)
        scale = finite_population_mean_scale(prior_variance, noise, n_cells)
        batches = np.asarray(transition_stds_batch_mean_sequence(
            jnp.float32(latent_variance), jnp.float32(noise), batch_size, n_cells - count
        )) * scale
        self.assertEqual(batches.shape, (2,))  # Three cells, then a final batch of two.
        cells = np.asarray(transition_stds_finite_population_posterior(
            jnp.float32(prior_variance), jnp.float32(noise), n_cells
        ))
        np.testing.assert_allclose(
            batches**2, [np.square(cells[2:5]).sum(), np.square(cells[5:7]).sum()],
            rtol=2e-6, atol=1e-9,
        )
        obs = torch.tensor([[0.3, 0.9] + [float("nan")] * 5])
        _, variance = finite_population_posterior_moments(
            obs, prior_mean=0.5, prior_variance=prior_variance, tau_sq_cell=noise
        )
        np.testing.assert_allclose(np.square(batches).sum(), float(variance[0]), rtol=2e-6)

    def test_roots_match_affine_payoff_with_divided_latent_cost(self):
        n_cells, prior_variance, noise = 4, 0.02, 0.25
        costs = jnp.array([0.02, 0.08], dtype=jnp.float32)
        scale = finite_population_mean_scale(prior_variance, noise, n_cells)
        finite = np.asarray(compute_finite_population_roots_lookup_table(
            prior_variance=prior_variance, tau_sq_cell=noise, n_examples=n_cells,
            costs_per_arm=costs, n_points=513,
        ))
        latent_transitions = transition_stds_shrinking_gaussian_posterior(
            jnp.float32(prior_variance), jnp.float32(noise), n_cells
        )
        divided_cost = np.asarray(compute_roots_lookup_table(
            transition_stds=latent_transitions, costs_per_arm=costs / scale, n_points=513
        ))
        unchanged_latent_cost = np.asarray(compute_roots_lookup_table(
            transition_stds=latent_transitions, costs_per_arm=costs, n_points=513
        ))
        np.testing.assert_allclose(finite[:, :-1], scale * divided_cost[:, :-1], atol=2e-6, rtol=2e-5)
        self.assertGreater(np.max(np.abs(finite[:, :-1] - scale * unchanged_latent_cost[:, :-1])), 0.01)
        np.testing.assert_array_equal(finite[:, -1], 0.0)
        shared = compute_finite_population_roots_lookup_table(
            prior_variance=prior_variance, tau_sq_cell=noise, n_examples=n_cells,
            costs_per_arm=jnp.array([0.02, 0.02]), n_points=513,
        )
        self.assertEqual(shared.shape, (1, n_cells + 1))
        np.testing.assert_allclose(np.asarray(shared[0]), finite[0], atol=2e-6)

    def test_lookup_and_forced_cell_dp_agree_with_nonuniform_costs(self):
        obs = torch.tensor(
            [[float("nan")] * 5, [0.7, 0.2, float("nan"), float("nan"), float("nan")],
             [0.8, 0.6, 0.9, 0.4, float("nan")], [0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        costs = torch.tensor([1.0, 1.1, 0.9, 1.4], dtype=torch.float64)
        options = dict(
            prior_mean=0.5, prior_variance=0.04, obs_noise_variance=0.25 / 3,
            batch_observation_model=True, batch_size=3, cost_per_transition=costs,
            cost_scaling_factor=0.01, n_gittins_grid_points=513, sim_cum_eval=11,
        )
        lookup_means, lookup_scores = gittins_post_pull_update(
            obs, cached_scores=torch.zeros(4), recompute_arms=range(4), **options
        )
        with patch("gittins_policy._finite_roots_for_policy", side_effect=AssertionError("unused roots")):
            direct_means, direct_scores = gittins_post_pull_update(
                obs, cached_scores=torch.zeros(4), recompute_arms=range(4),
                force_per_observation_dp=True, **options
            )
        torch.testing.assert_close(lookup_means, direct_means)
        torch.testing.assert_close(lookup_scores, direct_scores, rtol=2e-5, atol=2e-6)
        self.assertAlmostEqual(float(lookup_scores[-1]), float(obs[-1].mean()), places=6)

    def test_batch_dp_uses_cell_noise_and_fixed_cost_with_partial_final_batch(self):
        obs = torch.tensor([[0.3, 0.9] + [float("nan")] * 5])
        model = dict(prior_mean=0.5, prior_variance=0.04, tau_sq_cell=0.25)
        latent_means, latent_variances = posterior_moments(obs, **model)
        finite_means, _ = finite_population_posterior_moments(obs, **model)
        scale = finite_population_mean_scale(0.04, 0.25, 7)
        latent_index = compute_gittins_shrinking_posterior_walk_batch_mean(
            jnp.float32(float(latent_means[0])), jnp.float32(float(latent_variances[0])),
            jnp.float32(0.25), 3, 5, jnp.full((2,), 0.01 / scale), jnp.uint32(513),
        )
        expected = scale * float(latent_index) + (1.0 - scale) * 0.5
        with patch("gittins_policy._finite_roots_for_policy", side_effect=AssertionError("unused roots")):
            for batch_model, supplied_noise in ((False, 0.25), (True, 0.25 / 3)):
                with self.subTest(batch_model=batch_model):
                    means, scores = gittins_post_pull_update(
                        obs, cached_scores=torch.zeros(1), recompute_arms=[0],
                        prior_mean=0.5, prior_variance=0.04, obs_noise_variance=supplied_noise,
                        batch_observation_model=batch_model, batch_size=3,
                        use_batch_mean_gittins_dp=True, cost_scaling_factor=0.01,
                        n_gittins_grid_points=513, sim_cum_eval=2,
                    )
                    torch.testing.assert_close(means, finite_means)
                    self.assertAlmostEqual(float(scores[0]), expected, places=5)

    def test_completed_index_controls_early_stop_and_partial_batch_fallback(self):
        obs = torch.tensor([[0.7, 0.7, 0.7], [0.8, float("nan"), float("nan")]])
        for early_stop in (True, False):
            with self.subTest(early_stop=early_stop):
                cache = torch.tensor([0.0, 0.69])
                batch, means = gittins_index_exploration(
                    obs, prior_mean=0.0, prior_variance=0.01, obs_noise_variance=1.0,
                    cached_scores=cache, recompute_arms=[], roots_lookup_table=torch.zeros((1, 4)),
                    batch_size=4, allow_early_stop=early_stop, return_mus=True,
                )
                self.assertAlmostEqual(float(cache[0]), 0.7, places=6)
                self.assertAlmostEqual(float(means[0]), 0.7, places=6)
                if early_stop:
                    self.assertIsNone(batch)
                else:
                    self.assertEqual(tuple(batch.shape), (2, 2))
                    self.assertTrue(bool((batch[0] == 1).all()))
                    self.assertEqual(set(batch[1].tolist()), {1, 2})

    def test_complete_matrix_returns_empirical_means_without_building_roots(self):
        obs = torch.tensor([[0.1, 0.3], [0.8, 0.6]])
        with patch("gittins_policy._finite_roots_for_policy", side_effect=AssertionError("unused roots")):
            batch, means = gittins_index_exploration(obs, return_mus=True)
            natural_stop = [None]
            updated_means, scores = gittins_post_pull_update(
                obs, cached_scores=torch.zeros(2), recompute_arms=[0, 1], sim_cum_eval=4,
                natural_stop_cum_eval_holder=natural_stop,
            )
        self.assertIsNone(batch)
        torch.testing.assert_close(means, obs.mean(dim=1))
        torch.testing.assert_close(updated_means, means)
        torch.testing.assert_close(scores, means)
        self.assertEqual(natural_stop, [4])

    def test_recommendation_penalty_does_not_change_acquisition_or_natural_stop(self):
        obs = torch.tensor([[1.88, float("nan"), float("nan"), float("nan")],
                            [1.0, 1.0, 1.0, float("nan")]])
        outputs = []
        for penalty in (0.0, 1.0):
            natural_stop = [None]
            outputs.append(gittins_post_pull_update(
                obs, cached_scores=torch.zeros(2), recompute_arms=range(2),
                prior_mean=0.0, prior_variance=1.0, obs_noise_variance=1.0,
                roots_lookup_table=torch.zeros((1, 5)), sim_cum_eval=4,
                recommendation_std_penalty=penalty, natural_stop_cum_eval_holder=natural_stop,
            ))
            self.assertEqual(natural_stop, [None])
        torch.testing.assert_close(outputs[0][0], outputs[1][0])
        torch.testing.assert_close(outputs[0][1], outputs[1][1])


if __name__ == "__main__":
    unittest.main()
