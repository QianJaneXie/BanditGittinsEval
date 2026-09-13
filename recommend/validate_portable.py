"""Check the portable DP against analytic cases and archived native GSM8K runs."""
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from portable_gittins import compute_roots, gaussian_convolve, next_arm

ROOT = Path(__file__).resolve().parents[1]


def main():
    grid = np.linspace(-2, 2, 1025)
    sigma = 0.3
    expected = grid * ndtr(grid / sigma) + sigma * np.exp(-0.5 * (grid / sigma)**2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(gaussian_convolve(np.maximum(grid, 0), sigma, grid[1]-grid[0]), expected, atol=1e-12)
    checks = {"analytic_relu_convolution": "passed", "native_reference_comparisons": {}}
    for tag, variance in [("default", 0.04), ("dataset", 0.01)]:
        portable = compute_roots(variance, 1000)
        native = np.load(ROOT / f"recommend/results/gsm8k/roots/{tag}_N1000.npy")
        stage = np.arange(1000)
        v = 1 / (1 / variance + stage / 0.25)
        grid_spacing = (10 * np.sqrt(np.sum(v**2 / (v + 0.25))) + 1.01 * 0.1) / 1024
        difference = np.abs(portable - native)
        assert difference.max() <= grid_spacing + 1e-6
        trace_path = ROOT / f"recommend/results/gsm8k/traces/{tag}_matrix1_run00.npz"
        with np.load(trace_path) as trace:
            counts = np.r_[np.zeros_like(trace["counts"][:1]), trace["counts"][:-1]].astype(int)
            sums = np.r_[np.zeros_like(trace["sums"][:1]), trace["sums"][:-1]]
            native_predictions = [next_arm(n, s, native, 1000, float(trace["prior_mean"]), variance, .25) for n, s in zip(counts, sums)]
            np.testing.assert_array_equal(native_predictions, trace["pulled_arm"])
            portable_predictions = [next_arm(n, s, portable, 1000, float(trace["prior_mean"]), variance, .25) for n, s in zip(counts, sums)]
            checks["native_reference_comparisons"][tag] = {
                "max_root_difference": float(difference.max()), "one_grid_spacing": float(grid_spacing),
                "roots_differing_above_1e_6": int((difference > 1e-6).sum()), "n_roots": len(native),
                "archived_native_allocation_decisions_reproduced_with_saved_roots": len(counts),
                "portable_root_decision_disagreements_on_archived_states": int(np.sum(portable_predictions != trace["pulled_arm"])),
            }
    checks["limitation"] = "Numerical agreement within one grid cell, not bitwise equivalence. New runs use PCG64, not Torch randomness; no new native-policy execution was possible."
    target = ROOT / "recommend/results/portable_validation.json"
    target.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
