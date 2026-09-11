"""Regression tests for posterior mean minus standard-deviation recommendations."""

import unittest

import torch

from gittins_policy import evaluate_gittins_stopping_rules, gittins_post_pull_update
from simple_regret_recommend import (
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
        selected, means = posterior_incumbent(
            obs, prior_mean=0.0, prior_variance=1.0, tau_sq_cell=1.0,
            std_penalty=1.0,
        )
        self.assertEqual(selected, 1)
        torch.testing.assert_close(means, torch.tensor([0.94, 0.75]))

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
                    with self.assertRaisesRegex(ValueError, name):
                        posterior_moments(torch.tensor([[0.5]]), prior_mean=0.5, **kwargs)


class RecommendationAwareStoppingTests(unittest.TestCase):
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
        # Cell variance 1 gives posteriors (mu, var) = (.94, .5), (.75, .25).
        # mu - sigma selects arm 1; forgetting batch-to-cell conversion in
        # the variance calculation would instead select arm 0.
        for batch_model, noise in ((True, 0.25), (False, 1.0)):
            for penalty, maximum_score, expected_stop in (
                (0.0, 0.9, 4),
                (1.0, 0.9, None),
                (1.0, 0.7, 4),  # .7 < selected raw mean .75, but > its LCB .25.
            ):
                with self.subTest(batch_model=batch_model, penalty=penalty, score=maximum_score):
                    stop = [None]
                    cached_scores = torch.tensor([maximum_score, 0.6])
                    means, scores = gittins_post_pull_update(
                        obs, cached_scores=cached_scores, recompute_arms=[],
                        prior_mean=0.0, prior_variance=1.0,
                        obs_noise_variance=noise, batch_size=4,
                        batch_observation_model=batch_model,
                        roots_lookup_table=torch.zeros((1, 5)), sim_cum_eval=4,
                        recommendation_aware_stop_cum_eval_holder=stop,
                        recommendation_std_penalty=penalty,
                    )
                    torch.testing.assert_close(means, torch.tensor([0.94, 0.75]))
                    self.assertIs(scores, cached_scores)
                    self.assertEqual(stop, [expected_stop])


if __name__ == "__main__":
    unittest.main()
