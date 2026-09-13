"""Small fixed replay checks, including failures caused by corrupted artifacts."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommend.validate_full_suite import audit_trace


class FullSuiteAuditTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0.5, 0, 1]], dtype=float)
        self.roots = np.array([-0.15, -0.12, -0.09, -0.06, -0.03, 0], dtype=np.float32)
        self.config = {"tau_sq_cell": 0.25, "batch_size": 2, "budget_fraction": 0.4}
        self.priors = {"default": [0.5, 0.04]}
        # The three sampled pairs are the fixed first three PCG64(7)
        # permutations of five columns. Fractional rewards exercise the same
        # float64 accumulation required by Alpaca, without a Bernoulli shortcut.
        self.trace = {
            "prior_tag": np.array("default"), "prior_mean": np.array(0.5),
            "prior_variance": np.array(0.04), "matrix_seed": np.array(1),
            "run_seed": np.array(7), "n_arms": np.array(3), "n_examples": np.array(5),
            "method_keys": np.array(["posterior_mean", "gaussian_lcb_1.645"]),
            "true_means": np.array([0.4, 0.2, 0.3]), "x": np.array([2, 4, 6]),
            "pulled_arm": np.array([0, 1, 2]),
            "pulled_columns": np.array([[2, 0], [0, 1], [4, 2]]),
            "recommended_arm": np.array([[0, 0], [0, 0], [2, 2]]),
            "scores": np.array([[0.5, 0.5 - 1.645 * np.sqrt(1 / 33)],
                                [0.5, 0.5 - 1.645 * np.sqrt(1 / 33)],
                                [np.float32(18.5 / 33), 18.5 / 33 - 1.645 * np.sqrt(1 / 33)]]),
            "selected_n": np.full((3, 2), 2, dtype=int),
            "selected_variance": np.full((3, 2), 1 / 33),
            "regret": np.array([[0, 0], [0, 0], [0.1, 0.1]]),
            "final_counts": np.array([2, 2, 2]), "final_sums": np.array([1, 0, 1.5]),
        }

    def check(self, trace=None, chunk_size=2):
        return audit_trace(self.trace if trace is None else trace, self.matrix, self.roots,
                           self.config, self.priors, chunk_size=chunk_size)

    def test_fixed_continuous_reward_trace_across_chunks(self):
        for size in (1, 2, 256):
            with self.subTest(chunk_size=size):
                self.assertEqual(self.check(chunk_size=size), (("default", 1, 7), 3, 6))

    def test_rejects_corrupted_recommendation(self):
        trace = copy.deepcopy(self.trace)
        trace["recommended_arm"][1, 1] = 1
        trace["regret"][1, 1] = 0.2
        with self.assertRaises(AssertionError):
            self.check(trace)

    def test_rejects_changed_sample_order(self):
        trace = copy.deepcopy(self.trace)
        trace["pulled_columns"][0] = [0, 2]
        with self.assertRaisesRegex(AssertionError, "seeded sampling"):
            self.check(trace)

    def test_rejects_allocation_with_same_observed_rewards(self):
        trace = copy.deepcopy(self.trace)
        trace["pulled_arm"][0] = 2
        with self.assertRaisesRegex(AssertionError, "allocation"):
            self.check(trace)

    def test_rejects_changed_score(self):
        trace = copy.deepcopy(self.trace)
        trace["scores"][0, 1] += 0.001
        with self.assertRaises(AssertionError):
            self.check(trace)

    def test_rejects_corrupted_final_statistics(self):
        for name in ("final_counts", "final_sums"):
            trace = copy.deepcopy(self.trace)
            trace[name][0] += 1
            with self.subTest(field=name), self.assertRaises(AssertionError):
                self.check(trace)


if __name__ == "__main__":
    unittest.main()
