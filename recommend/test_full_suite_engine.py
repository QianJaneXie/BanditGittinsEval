"""Check compact cached replay against the frozen full-rule replay engine."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from full_suite_engine import run_path, METHOD_KEYS
from experiment_core import run_path as reference_path
from portable_gittins import compute_roots


class CompactReplayTests(unittest.TestCase):
    def compare(self,matrix,prior,batch,budget,seed):
        roots=compute_roots(prior[1],matrix.shape[1])
        args=dict(prior_tag="dataset",prior_pair=prior,run_seed=seed,matrix_seed=1,
                  batch_size=batch,budget_fraction=budget,tau_sq_cell=.25)
        compact=run_path(matrix,roots,**args)
        reference=reference_path(matrix,roots,**args)
        for field in ["x","pulled_arm","pulled_columns"]:
            np.testing.assert_array_equal(compact[field],reference[field])
        cols=[list(reference["method_keys"]).index(k) for k in METHOD_KEYS]
        for field in ["recommended_arm","regret","scores","selected_n","selected_variance"]:
            np.testing.assert_array_equal(compact[field],reference[field][:,cols])
        np.testing.assert_array_equal(compact["final_counts"],reference["counts"][-1])
        np.testing.assert_allclose(compact["final_sums"],reference["sums"][-1],rtol=1e-7,atol=1e-7)

    def test_binary_with_full_row_exhaustion(self):
        matrix=np.random.default_rng(6).integers(0,2,(9,13)).astype(float)
        self.compare(matrix,(.6,.02),4,1.,0)

    def test_continuous_with_truncated_budget(self):
        matrix=np.random.default_rng(4).random((12,29))
        self.compare(matrix,(.2,.01),16,.3,3)

    def test_unobserved_arm_ties(self):
        self.compare(np.zeros((15,40)),(.5,.04),4,.1,0)


if __name__=="__main__":
    unittest.main()
