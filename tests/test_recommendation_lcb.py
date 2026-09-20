"""Regression tests for finite-row posterior recommendations and optional LCBs."""

import unittest

import torch

from gittins_policy import evaluate_gittins_stopping_rules, gittins_post_pull_update
from simple_regret_recommend import (
    finite_population_posterior_moments,
    posterior_incumbent,
    posterior_means,
    posterior_moments,
    recommend_from_means,
    recommend_from_posterior,
)


class PosteriorLCBRecommendationTests(unittest.TestCase):
    def test_moments_use_observed_cell_counts_and_cell_noise(self):
        obs = torch.tensor(
            [[float("nan")] * 3, [2.0, float("nan"), float("nan")], [3.0, 6.0, 9.0]],
            dtype=torch.float64,
        )
        means, variances = posterior_moments(
            obs, prior_mean=0.5, prior_variance=2.0, tau_sq_cell=2.0
        )
        torch.testing.assert_close(means, torch.tensor([0.5, 1.25, 4.625]))
        torch.testing.assert_close(
            variances, torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)
        )
        torch.testing.assert_close(
            means,
            posterior_means(obs, prior_mean=0.5, prior_variance=2.0, tau_sq_cell=2.0),
        )

    def test_finite_moments_cover_unobserved_partial_and_complete_rows(self):
        obs = torch.tensor(
            [[float("nan")] * 3, [2.0, float("nan"), float("nan")], [3.0, 6.0, 9.0]],
            dtype=torch.float64,
        )
        means, variances = finite_population_posterior_moments(
            obs, prior_mean=0.5, prior_variance=2.0, tau_sq_cell=2.0
        )
        # The partial row's two unknown cells have latent variance 1 and
        # individual noise variance 2: Var(full-row mean) = (4 + 4) / 9.
        torch.testing.assert_close(means, torch.tensor([0.5, 1.5, 6.0]))
        torch.testing.assert_close(
            variances, torch.tensor([8.0 / 3.0, 8.0 / 9.0, 0.0], dtype=torch.float64)
        )

    def test_completed_rows_use_empirical_means_without_uncertainty_penalty(self):
        obs = torch.tensor([[0.6, 0.8], [0.9, 0.7]])
        for penalty in (0.0, 1.0, 100.0):
            with self.subTest(penalty=penalty):
                selected, means = posterior_incumbent(
                    obs, prior_mean=0.0, prior_variance=0.01, tau_sq_cell=1.0,
                    std_penalty=penalty,
                )
                self.assertEqual(selected, 1)
                torch.testing.assert_close(means, obs.mean(dim=1))

    def test_shared_prior_preserves_mean_ranking_when_target_changes(self):
        obs = torch.tensor(
            [[float("nan")] * 4, [0.7, float("nan"), float("nan"), float("nan")],
             [0.4, 0.8, float("nan"), float("nan")], [0.4, 0.3, 0.5, 0.7]]
        )
        model = dict(prior_mean=0.5, prior_variance=0.01, tau_sq_cell=0.25)
        latent_means, _ = posterior_moments(obs, **model)
        selected, finite_means = posterior_incumbent(obs, **model)
        # With common prior/noise/N, full-row means are a positive affine
        # transformation of latent means, even though their values differ.
        torch.testing.assert_close(
            finite_means, 0.5 + (1.0 + 25.0 / 4) * (latent_means - 0.5)
        )
        self.assertEqual(selected, int(latent_means.argmax().item()))

    def test_different_counts_preserve_equal_posterior_means(self):
        obs = torch.tensor(
            [[0.604, float("nan"), float("nan"), float("nan")],
             [0.55, 0.558, float("nan"), float("nan")],
             [0.52, 0.54, 0.552, float("nan")], [0.5, 0.52, 0.53, 0.566]],
            dtype=torch.float64,
        )
        # Each sum is (n + 25)*.504 - 25*.5, so every latent posterior
        # mean is .504 and every full-row posterior mean is .529.
        selected, means = posterior_incumbent(
            obs, prior_mean=0.5, prior_variance=0.01, tau_sq_cell=0.25
        )
        self.assertTrue(torch.equal(means, torch.full((4,), 0.529)))
        self.assertEqual(selected, 0)

    def test_penalty_uses_standard_deviation_not_variance(self):
        # mu - sigma = [0, -0.05], but mu - variance = [0, 0.1375].
        selected, _ = recommend_from_posterior(
            torch.tensor([1.0, 0.2]), torch.tensor([1.0, 0.0625]), std_penalty=1.0
        )
        self.assertEqual(selected, 0)

    def test_penalty_can_select_lower_mean_and_returns_unmodified_means(self):
        means = torch.tensor([0.94, 0.75], requires_grad=True)
        variances = torch.tensor([0.5, 0.25])
        baseline, _ = recommend_from_posterior(means, variances, std_penalty=0.0)
        selected, returned_means = recommend_from_posterior(
            means, variances, std_penalty=1.0
        )
        self.assertEqual(baseline, 0)
        self.assertEqual(selected, 1)
        torch.testing.assert_close(returned_means, means)
        self.assertFalse(returned_means.requires_grad)

    def test_penalty_magnitude_controls_ranking(self):
        means = torch.tensor([0.94, 0.75])
        variances = torch.tensor([0.5, 0.25])
        for penalty, expected in ((0.5, 0), (1.0, 1), (2.0, 1)):
            with self.subTest(penalty=penalty):
                selected, _ = recommend_from_posterior(
                    means, variances, std_penalty=penalty
                )
                self.assertEqual(selected, expected)

    def test_zero_and_default_penalties_preserve_posterior_mean_rule(self):
        for values in (
            [0.9, 0.6, 0.7],
            [float("nan"), 0.6, 0.7],
            [0.7, 0.7, 0.6],
            [float("nan")] * 3,
        ):
            for options in ({}, {"std_penalty": 0.0}):
                with self.subTest(values=values, options=options):
                    means = torch.tensor(values)
                    expected_arm, expected_means = recommend_from_means(means)
                    selected, returned_means = recommend_from_posterior(
                        means, torch.tensor([float("nan"), 0.25, 0.125]), **options
                    )
                    self.assertEqual(selected, expected_arm)
                    torch.testing.assert_close(returned_means, expected_means, equal_nan=True)

    def test_unknown_scores_and_exact_ties_are_deterministic(self):
        cases = (
            ([float("nan"), 0.6, 0.9], [0.1, float("nan"), 0.25], 2),
            ([float("nan"), float("nan")], [0.25, 0.25], 0),
            ([1.0, 0.75], [1.0, 0.5625], 0),
        )
        for means, variances, expected in cases:
            with self.subTest(means=means, variances=variances):
                selected, _ = recommend_from_posterior(
                    torch.tensor(means), torch.tensor(variances), std_penalty=1.0
                )
                self.assertEqual(selected, expected)

    def test_all_unobserved_arms_use_prior_and_lowest_index(self):
        selected, means = posterior_incumbent(
            torch.full((3, 5), float("nan")),
            prior_mean=0.5, prior_variance=1.0, tau_sq_cell=1.0,
            std_penalty=1.0,
        )
        self.assertEqual(selected, 0)
        torch.testing.assert_close(means, torch.full((3,), 0.5))

    def test_posterior_incumbent_applies_std_penalty(self):
        obs = torch.tensor(
            [[1.88, float("nan"), float("nan"), float("nan")],
             [1.0, 1.0, 1.0, float("nan")]]
        )
        for options, expected_arm in (({}, 0), ({"std_penalty": 0.0}, 0), ({"std_penalty": 1.0}, 1)):
            with self.subTest(options=options):
                selected, means = posterior_incumbent(
                    obs, prior_mean=0.0, prior_variance=1.0, tau_sq_cell=1.0,
                    **options,
                )
                self.assertEqual(selected, expected_arm)
                torch.testing.assert_close(means, torch.tensor([1.175, 0.9375]))

    def test_invalid_std_penalties_are_rejected(self):
        for penalty in (-0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(penalty=penalty):
                with self.assertRaisesRegex(ValueError, "std_penalty"):
                    recommend_from_posterior(
                        torch.tensor([0.5]), torch.tensor([0.25]), std_penalty=penalty
                    )

    def test_invalid_model_variances_are_rejected(self):
        for name in ("prior_variance", "tau_sq_cell"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    kwargs = {"prior_variance": 1.0, "tau_sq_cell": 1.0, name: value}
                    for moments in (posterior_moments, finite_population_posterior_moments):
                        with self.assertRaisesRegex(ValueError, name):
                            moments(torch.tensor([[0.5]]), prior_mean=0.5, **kwargs)

    def test_finite_row_target_requires_nonempty_rows(self):
        with self.assertRaisesRegex(ValueError, "at least one cell"):
            finite_population_posterior_moments(
                torch.empty((2, 0)), prior_mean=0.5, prior_variance=1.0, tau_sq_cell=1.0
            )


class RecommendationAwareStoppingTests(unittest.TestCase):
    def test_lcb_aligned_stop_uses_strict_crossing_and_preserves_first_hit(self):
        means = torch.tensor([1.0, 0.75])
        variances = torch.tensor([0.25, 0.0625])
        completely_sensed = torch.tensor([False, False])
        stop = [None]
        common = dict(
            mus_posterior=means,
            completely_sensed_mask=completely_sensed,
            recommended_arm=1,
            recommendation_variances=variances,
            recommendation_std_penalty=1.0,
            lcb_aligned_stop_cum_eval_holder=stop,
        )
        # Selected LCB is .75 - .25 = .50; equality is not a crossing.
        evaluate_gittins_stopping_rules(torch.tensor([0.50, 0.4]), sim_cum_eval=10, **common)
        self.assertEqual(stop, [None])
        evaluate_gittins_stopping_rules(torch.tensor([0.49, 0.4]), sim_cum_eval=12, **common)
        evaluate_gittins_stopping_rules(torch.tensor([0.30, 0.2]), sim_cum_eval=15, **common)
        self.assertEqual(stop, [12])

    def test_stop_uses_selected_mean_and_preserves_first_crossing(self):
        means = torch.tensor([1.0, 0.75])
        incomplete = torch.tensor([False, False])
        baseline_stop = [None]
        selected_stop = [None]
        scores = torch.tensor([0.75, 0.7])
        evaluate_gittins_stopping_rules(
            scores, means, incomplete, sim_cum_eval=10,
            recommendation_aware_stop_cum_eval_holder=baseline_stop,
        )
        evaluate_gittins_stopping_rules(
            scores, means, incomplete, sim_cum_eval=10, recommended_arm=1,
            recommendation_aware_stop_cum_eval_holder=selected_stop,
        )
        self.assertEqual(baseline_stop, [10])
        self.assertEqual(selected_stop, [None])  # Equality is not a strict crossing.
        for current_eval in (12, 15):
            evaluate_gittins_stopping_rules(
                torch.tensor([0.74, 0.7]), means, incomplete,
                sim_cum_eval=current_eval, recommended_arm=1,
                recommendation_aware_stop_cum_eval_holder=selected_stop,
            )
        self.assertEqual(selected_stop, [12])

    def test_post_pull_converts_batch_noise_and_stops_against_selected_raw_mean(self):
        obs = torch.tensor(
            [[1.88, float("nan"), float("nan"), float("nan")],
             [1.0, 1.0, 1.0, float("nan")]]
        )
        # Cell variance 1 gives full-row moments (1.175, .46875), (.9375, .078125).
        # LCB selects arm 1; forgetting batch-to-cell conversion selects arm 0.
        for batch_model, noise in ((True, 0.25), (False, 1.0)):
            for penalty, maximum_score, expected_raw_stop, expected_lcb_stop in (
                (0.0, 1.1, 4, 4),
                (1.0, 1.1, None, None),
                (1.0, 0.9, 4, None),  # .9 < selected raw mean .9375, but > its LCB .658.
                (1.0, 0.65, 4, 4),
            ):
                with self.subTest(batch_model=batch_model, penalty=penalty, score=maximum_score):
                    raw_stop = [None]
                    lcb_stop = [None]
                    cached_scores = torch.tensor([maximum_score, 0.6])
                    means, scores = gittins_post_pull_update(
                        obs, cached_scores=cached_scores, recompute_arms=[],
                        prior_mean=0.0, prior_variance=1.0,
                        obs_noise_variance=noise, batch_size=4,
                        batch_observation_model=batch_model,
                        roots_lookup_table=torch.zeros((1, 5)), sim_cum_eval=4,
                        recommendation_aware_stop_cum_eval_holder=raw_stop,
                        lcb_aligned_stop_cum_eval_holder=lcb_stop,
                        recommendation_std_penalty=penalty,
                    )
                    torch.testing.assert_close(means, torch.tensor([1.175, 0.9375]))
                    self.assertIs(scores, cached_scores)
                    self.assertEqual(raw_stop, [expected_raw_stop])
                    self.assertEqual(lcb_stop, [expected_lcb_stop])

    def test_finite_indices_and_stopping_share_the_completed_empirical_payoff(self):
        obs = torch.tensor([[1.0] * 4, [1.0, 1.0, 1.0, float("nan")]])
        for unfinished_score, expected_stop in ((1.1, None), (0.9, 7)):
            with self.subTest(unfinished_score=unfinished_score):
                natural_stop = [None]
                recommendation_stop = [None]
                lcb_stop = [None]
                means, scores = gittins_post_pull_update(
                    obs, cached_scores=torch.tensor([0.0, unfinished_score]), recompute_arms=[],
                    prior_mean=0.0, prior_variance=1.0, obs_noise_variance=1.0,
                    roots_lookup_table=torch.zeros((1, 5)), sim_cum_eval=7,
                    natural_stop_cum_eval_holder=natural_stop,
                    recommendation_aware_stop_cum_eval_holder=recommendation_stop,
                    lcb_aligned_stop_cum_eval_holder=lcb_stop,
                )
                torch.testing.assert_close(means, torch.tensor([1.0, 0.9375]))
                torch.testing.assert_close(scores, torch.tensor([1.0, unfinished_score]))
                self.assertEqual(recommendation_stop, [expected_stop])
                self.assertEqual(lcb_stop, [expected_stop])
                self.assertEqual(natural_stop, [expected_stop])


if __name__ == "__main__":
    unittest.main()
