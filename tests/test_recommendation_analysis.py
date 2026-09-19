"""Checks that paired analysis preserves raw events and budget boundaries."""

import unittest

import numpy as np

from scripts.analyze_recommendation_oscillation import per_seed_metrics
from scripts.compare_recommendation_rules import align_histories


class RecommendationAnalysisTests(unittest.TestCase):
    def test_partial_batches_hold_past_observations_without_lookahead(self):
        histories = [
            {"evaluations": np.array([16, 32, 40, 56]), "regret": np.array([[.2], [.1], [.2], [0.]])},
            {"evaluations": np.array([16, 32, 48]), "regret": np.array([[.3], [.2], [.1]])},
        ]
        grid, aligned = align_histories(histories, budget=48)
        np.testing.assert_array_equal(grid, [16, 32, 40, 48])
        np.testing.assert_array_equal(aligned["source_evaluations"],
                                      [[16, 32, 40, 40], [16, 32, 32, 48]])
        np.testing.assert_allclose(aligned["regret"][:, :, 0],
                                   [[.2, .1, .2, .2], [.3, .2, .2, .1]])

    def test_raw_switches_and_budget_weights_exclude_overshooting_batch(self):
        history = {
            "evaluations": np.array([16, 32, 40, 56]),
            "regret": np.array([[.2], [.1], [.2], [0.]]),
            "recommendation": np.array([[0], [1], [0], [2]]),
            "recommended_count": np.array([[0], [1], [0], [5]]),
        }
        metrics = per_seed_metrics(history, rule=0, start=16, end=48)
        self.assertEqual(metrics["raw_post_pull_states"], 3)
        self.assertEqual(metrics["recommendation_switches"], 2)
        self.assertEqual(metrics["upward_regret_jumps"], 1)
        self.assertAlmostEqual(metrics["regret_total_variation"], .2)
        self.assertAlmostEqual(metrics["budget_weighted_mean_regret"], .175)
        self.assertAlmostEqual(metrics["final_regret"], .2)
        self.assertIsNone(metrics["first_zero_evaluations"])
        self.assertIsNone(metrics["settles_zero_evaluations"])
        self.assertAlmostEqual(metrics["unobserved_recommendation_fraction_raw_states"], 2 / 3)
        self.assertAlmostEqual(metrics["unobserved_recommendation_budget_fraction"], .75)


if __name__ == "__main__":
    unittest.main()
