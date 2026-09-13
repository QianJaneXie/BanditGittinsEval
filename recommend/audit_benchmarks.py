"""Independently audit every recommendation, allocation, and sampled cell.

Uses vectorized formulas rather than calling the experiment's rule evaluator.
Run after validate_results.py, which reconstructs sufficient statistics and
checks source hashes, budgets, oracle regrets, and the complete seed grid.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    assert manifest["status"] == "complete"
    assert config["sampling_generator"] == "numpy.random.PCG64"
    total = 0
    runs = 0
    for path in sorted((directory / "traces").glob("*.npz")):
        with np.load(path) as z:
            counts, sums = z["counts"].astype(int), z["sums"].astype(float)
            rec, scores, eligible = z["recommended_arm"], z["scores"], z["eligible_count"]
            lower_saved, upper_saved = z["confidence_lower"], z["confidence_upper"]
            bounds, selected_variance = z["certified_regret_bound"], z["selected_variance"]
            pulled, columns = z["pulled_arm"], z["pulled_columns"]
            k, n = int(z["n_arms"]), int(z["n_examples"])
            mean0, variance0 = float(z["prior_mean"]), float(z["prior_variance"])
            tau = config["tau_sq_cell"]
            roots = np.load(directory / "roots" / f"{str(z['prior_tag'])}_N{n}.npy")
            generator = np.random.default_rng(int(z["run_seed"]))
            observed = np.zeros((k, n), dtype=bool)
            for arm, cols in zip(pulled, columns):
                remaining = np.flatnonzero(~observed[arm])
                cols = cols[cols >= 0]
                expected = remaining[generator.permutation(len(remaining))[:len(cols)]]
                np.testing.assert_array_equal(cols, expected)
                observed[arm, cols] = True
            for start in range(0, len(counts), 128):
                end = min(len(counts), start + 128)
                nn, ss = counts[start:end], sums[start:end]
                v = 1 / (1 / variance0 + nn / tau)
                mu = v * (mean0 / variance0 + ss / tau)
                mu[nn == 0] = mean0
                empirical = np.divide(ss, nn, out=np.zeros_like(ss), where=nn > 0)
                remaining = n - nn
                finite_mu = (ss + remaining * mu) / n
                finite_v = (remaining**2 * v + remaining * tau) / n**2
                radius = np.full_like(ss, np.inf)
                partial = (nn > 0) & (nn < n)
                count = nn[partial]
                correction = np.minimum(1 - (count - 1) / n, (1 - count / n) * (1 + 1 / count))
                radius[partial] = np.sqrt(correction * np.log(2 * k * count * (count + 1) / .05) / (2 * count))
                radius[nn == n] = 0
                lower = np.clip(np.maximum(ss / n, empirical - radius), 0, 1)
                upper = np.clip(np.minimum((ss + remaining) / n, empirical + radius), 0, 1)
                np.testing.assert_allclose(lower, lower_saved[start:end], atol=1e-12)
                np.testing.assert_allclose(upper, upper_saved[start:end], atol=1e-12)
                rows = np.arange(end-start)
                for j, spec in enumerate(manifest["methods"]):
                    key, family, parameter = spec["key"], spec["family"], spec["parameter"]
                    valid = np.ones_like(nn, dtype=bool)
                    if key == "posterior_mean":
                        values = mu.astype(np.float32)
                    elif key == "empirical_mean":
                        values, valid = empirical, nn > 0
                    elif family == "variance_gate":
                        values, valid = mu, v / variance0 <= parameter
                    elif family == "mean_variance":
                        values = mu - parameter * v
                    elif family == "gaussian_lcb":
                        values = mu - parameter * np.sqrt(v)
                    elif family == "finite_gaussian_lcb":
                        values = finite_mu - parameter * np.sqrt(finite_v)
                    elif family == "confidence":
                        values = lower
                    else:
                        raise AssertionError(key)
                    ne = valid.sum(1)
                    chosen = np.argmax(np.where(valid, values, -np.inf), axis=1)
                    chosen[ne == 0] = -1
                    np.testing.assert_array_equal(chosen, rec[start:end, j])
                    np.testing.assert_array_equal(ne, eligible[start:end, j])
                    vals = np.where(ne > 0, values[rows, chosen], np.nan)
                    np.testing.assert_allclose(vals, scores[start:end, j], atol=1e-12)
                    vars_ = np.where(ne > 0, v[rows, chosen], np.nan)
                    np.testing.assert_allclose(vars_, selected_variance[start:end, j], atol=1e-12)
                    competing = upper.copy()
                    competing[rows, chosen] = -np.inf
                    certificate = np.where(ne > 0, np.maximum(0, competing.max(1) - lower[rows, chosen]), np.nan)
                    np.testing.assert_allclose(certificate, bounds[start:end, j], atol=1e-12)
                previous_n = np.vstack([np.zeros(k, dtype=int) if start == 0 else counts[start-1], counts[start:end-1]])
                previous_s = np.vstack([np.zeros(k) if start == 0 else sums[start-1], sums[start:end-1]])
                previous_v = 1 / (1 / variance0 + previous_n / tau)
                index = (previous_v * (mean0 / variance0 + previous_s / tau)).astype(np.float32) - roots[previous_n]
                index[previous_n == n] = -np.inf
                np.testing.assert_array_equal(index.argmax(1), pulled[start:end])
            total += len(counts)
            runs += 1
    result = {"runs": runs, "states": total, "method_decisions": total * len(manifest["methods"]),
              "checks": ["all 18 recommendation choices, scores and eligibility counts", "latent selected variances",
                         "confidence bounds and regret certificates", "Gittins allocation from prior observations",
                         "exact PCG64 sampled columns"], "status": "passed"}
    (directory / "decision_audit.json").write_text(json.dumps(result, indent=2))
    print(directory.name, json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    for directory in parser.parse_args().directories:
        audit(directory)
