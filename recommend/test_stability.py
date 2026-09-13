"""Analytic examples covering temporal metrics, boundaries, and missing data."""
import unittest
import numpy as np
from recommend.analyze_stability import path_metrics


class StabilityTests(unittest.TestCase):
    def test_repeated_backtracking_accumulates(self):
        m=path_metrics([.01,.02,.03,.1], [0,.1,0,.1], [1,2,1,2])
        self.assertAlmostEqual(m["total_downward_pp"],20)
        self.assertAlmostEqual(m["maximum_drawdown_pp"],10)
        self.assertAlmostEqual(m["largest_drop_pp"],10)
        self.assertEqual(m["switches"],3)
        self.assertEqual(m["downward_jumps"],2)

    def test_gradual_decline_and_inclusive_threshold(self):
        m=path_metrics([.01,.02,.1], [0,.02,.04], [1,2,3])
        self.assertAlmostEqual(m["largest_drop_pp"],2)
        self.assertAlmostEqual(m["maximum_drawdown_pp"],4)
        self.assertEqual(m["jumps_ge_2pp"],2)

    def test_anchor_and_budget_weighting(self):
        m=path_metrics([.008,.012,.02,.1], [.05,.10,.06,.02], [1,2,3,4])
        self.assertAlmostEqual(m["mean_regret_pp"], 100*(.002*.05+.008*.1+.08*.06)/.09)
        self.assertAlmostEqual(m["mean_drawdown_pp"], 100*(.008*.05+.08*.01)/.09)
        self.assertAlmostEqual(m["final_regret_pp"],2)
        self.assertEqual(m["switches"],3)

    def test_stable_wrong_arm_keeps_quality_penalty(self):
        m=path_metrics([.01,.05,.1], [.5,.5,.5], [1,1,1])
        self.assertEqual(m["switches"],0)
        self.assertEqual(m["total_downward_pp"],0)
        self.assertAlmostEqual(m["mean_regret_pp"],50)

    def test_harmless_switch_and_abstention(self):
        m=path_metrics([.01,.1], [.1,.1], [1,2])
        self.assertEqual(m["switches"],1)
        self.assertEqual(m["total_downward_pp"],0)
        with self.assertRaises(ValueError):
            path_metrics([.01,.1], [.1,np.nan], [1,-1])


if __name__ == "__main__":
    unittest.main()
