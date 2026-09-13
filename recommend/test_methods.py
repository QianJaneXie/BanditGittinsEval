"""Run with: python -m unittest recommend.test_methods -v."""

import itertools
import math
import unittest

import numpy as np

from recommend.methods import RecommendationRules, method_specs


class RecommendationRulesTests(unittest.TestCase):
    def rules(self, n_arms=3, n_examples=1000, **kwargs):
        return RecommendationRules(n_arms, n_examples, 0.5, 0.04, **kwargs)

    def test_specs_are_unique(self):
        specs = method_specs()
        self.assertEqual(len(specs), 18)
        self.assertEqual(len({spec.key for spec in specs}), len(specs))

    def test_posterior_formula_and_native_float32_parity(self):
        rules = self.rules()
        n = np.array([4, 100, 0])
        sums = np.array([4, 75, 0])
        mean, variance = rules.posterior(n, sums)
        expected = (sums + 6.25 * 0.5) / (n + 6.25)
        np.testing.assert_allclose(mean, expected)
        np.testing.assert_allclose(variance, 0.25 / (n + 6.25))
        got = rules.evaluate(n, sums)["posterior_mean"]
        self.assertEqual(got.arm, 1)
        self.assertEqual(got.score, float(expected.astype(np.float32)[1]))

    def test_float32_tie_matches_current_recommender(self):
        rules = self.rules(n_arms=2)
        n, sums = [100, 100], [50, 50.000001]
        means, _ = rules.posterior(n, sums)
        self.assertGreater(means[1], means[0])
        self.assertEqual(rules.evaluate(n, sums)["posterior_mean"].arm, 0)

    def test_no_observations_and_abstention(self):
        results = self.rules().evaluate([0, 0, 0], [0, 0, 0])
        for spec in method_specs():
            recommendation = results[spec.key]
            if spec.family == "variance_gate" or spec.key == "empirical_mean":
                self.assertEqual(recommendation.arm, -1)
                self.assertEqual(recommendation.eligible_count, 0)
                self.assertTrue(math.isnan(recommendation.score))
            else:
                self.assertEqual(recommendation.arm, 0)
                self.assertEqual(recommendation.eligible_count, 3)
        self.assertEqual(results["finite_population_lcb"].score, 0)

    def test_gate_thresholds_and_exclusion(self):
        rules = self.rules(n_arms=2)
        # v/v0=6.25/(6.25+n): rho .5 first admits n=7.
        r = rules.evaluate([6, 7], [6, 0])
        self.assertEqual(r["variance_gate_0.5"].arm, 1)
        self.assertEqual(r["variance_gate_0.5"].eligible_count, 1)
        self.assertEqual(r["variance_gate_0.2"].arm, -1)
        # rho=.2 exactly admits n=25.
        r = rules.evaluate([24, 25], [24, 0])
        self.assertEqual(r["variance_gate_0.2"].arm, 1)
        self.assertEqual(r["variance_gate_0.2"].eligible_count, 1)

    def test_penalty_and_lower_credible_formulas(self):
        rules = self.rules(n_arms=2)
        n, sums = [1, 100], [1, 70]
        mean, variance = rules.posterior(n, sums)
        r = rules.evaluate(n, sums)
        for spec in method_specs():
            if spec.family == "mean_variance":
                score = mean - spec.parameter * variance
            elif spec.family == "gaussian_lcb":
                score = mean - spec.parameter * np.sqrt(variance)
            else:
                continue
            self.assertEqual(r[spec.key].arm, int(np.argmax(score)))
            self.assertAlmostEqual(r[spec.key].score, float(np.max(score)))

    def test_unobserved_gaussian_arm_can_be_recommended(self):
        r = self.rules(n_arms=2).evaluate([100, 0], [0, 0])
        self.assertEqual(r["posterior_mean"].arm, 1)
        self.assertEqual(r["empirical_mean"].arm, 0)

    def test_finite_gaussian_predictive_boundaries_and_formula(self):
        rules = self.rules(n_examples=10)
        n, sums = np.array([0, 4, 10]), np.array([0, 3, 7])
        mean, variance = rules.posterior(n, sums)
        row_mean, row_variance = rules.finite_posterior(n, sums)
        np.testing.assert_allclose(row_mean, (sums + (10 - n) * mean) / 10)
        np.testing.assert_allclose(row_variance, ((10 - n)**2 * variance + (10 - n) * 0.25) / 100)
        self.assertEqual(row_mean[0], 0.5)
        self.assertAlmostEqual(row_variance[0], 0.04 + 0.25 / 10)
        self.assertEqual(row_mean[2], 0.7)
        self.assertEqual(row_variance[2], 0)
        r = rules.evaluate(n, sums)
        for z in (1, 1.645, 1.96):
            scores = row_mean - z * np.sqrt(row_variance)
            self.assertEqual(r[f"finite_gaussian_lcb_{z:g}"].arm, int(np.argmax(scores)))
            self.assertAlmostEqual(r[f"finite_gaussian_lcb_{z:g}"].score, np.max(scores))

    def test_confidence_bound_formula(self):
        rules = self.rules(n_arms=2, n_examples=100)
        n, sums = np.array([10, 90]), np.array([7, 63])
        lo, hi = rules.confidence_bounds(n, sums)
        f = np.minimum(1 - (n - 1) / 100, (1 - n / 100) * (1 + 1 / n))
        radius = np.sqrt(f * np.log(2 * 2 * n * (n + 1) / 0.05) / (2 * n))
        np.testing.assert_allclose(lo, np.maximum(sums / 100, sums / n - radius))
        np.testing.assert_allclose(hi, np.minimum((sums + 100 - n) / 100, sums / n + radius))
        self.assertEqual(rules.evaluate(n, sums)["finite_population_lcb"].arm, 1)

    def test_confidence_zero_complete_and_single_example(self):
        rules = self.rules(n_examples=1)
        lo, hi = rules.confidence_bounds([0, 1, 1], [0, 0, 1])
        np.testing.assert_array_equal(lo, [0, 0, 1])
        np.testing.assert_array_equal(hi, [1, 0, 1])
        r = rules.evaluate([0, 1, 1], [0, 0, 1])
        self.assertEqual(r["finite_population_lcb"].arm, 2)
        rules = self.rules(n_examples=10)
        lo, hi = rules.confidence_bounds([10, 10, 10], [2, 5, 9])
        np.testing.assert_array_equal(lo, [0.2, 0.5, 0.9])
        np.testing.assert_array_equal(hi, lo)

    def test_small_binary_populations_simultaneous_coverage(self):
        # Exhaust every order of each N=8 binary population. Check the entire
        # confidence sequence, not only one fixed count or a Monte Carlo sample.
        size, delta = 8, 0.2
        rules = self.rules(n_arms=1, n_examples=size)
        for successes in range(size + 1):
            failures = 0
            total = math.comb(size, successes)
            for success_positions in itertools.combinations(range(size), successes):
                outcomes = np.zeros(size)
                outcomes[list(success_positions)] = 1
                sums = np.concatenate([[0], np.cumsum(outcomes)])
                bad_path = False
                for count in range(size + 1):
                    lo, hi = rules.confidence_bounds([count], [sums[count]], delta)
                    truth = successes / size
                    bad_path |= bool(truth < lo[0] - 1e-14 or truth > hi[0] + 1e-14)
                failures += bad_path
            self.assertLessEqual(failures / total, delta)

    def test_invalid_statistics(self):
        rules = self.rules(n_arms=2, n_examples=10)
        for counts, sums in [
            ([1], [0]), ([1, 2], [0, np.nan]), ([-1, 0], [0, 0]),
            ([11, 0], [0, 0]), ([0.5, 0], [0, 0]), ([0, 0], [1, 0]),
            ([1, 1], [-1, 0]),
        ]:
            with self.subTest(counts=counts, sums=sums), self.assertRaises(ValueError):
                rules.evaluate(counts, sums)
        for delta in [0, 1, -1, np.nan]:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                rules.confidence_bounds([0, 0], [0, 0], delta)


if __name__ == "__main__":
    unittest.main()
