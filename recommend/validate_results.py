"""Independently reconstruct saved observations and audit experiment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "recommend/results/gsm8k")
    args = parser.parse_args()
    manifest = json.loads((args.results_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    expected = {(p, m, s) for p in config["priors"] for m in config["matrix_seeds"] for s in config["run_seeds"]}
    assert manifest["status"] == "complete"
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    matrices = {}
    for seed, info in manifest["matrices"].items():
        path = ROOT / info["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        matrices[int(seed)] = np.load(path, allow_pickle=False)
    seen = set()
    native_steps = 0
    pulls = 0
    violations = []
    final_certificate_counts = np.zeros(len(manifest["methods"]), dtype=int)
    for path in sorted((args.results_dir / "traces").glob("*.npz")):
        with np.load(path, allow_pickle=False) as trace:
            identity = (str(trace["prior_tag"]), int(trace["matrix_seed"]), int(trace["run_seed"]))
            assert identity not in seen
            seen.add(identity)
            matrix = matrices[identity[1]]
            n_arms, n_examples = matrix.shape
            x = trace["x"]
            cells = trace["pulled_columns"]
            arms = trace["pulled_arm"]
            valid_cells = cells >= 0
            n_each = valid_cells.sum(axis=1)
            np.testing.assert_array_equal(x, np.cumsum(n_each))
            assert x[-1] == round(config["budget_fraction"] * matrix.size)
            flat_rows = np.broadcast_to(arms[:, None], cells.shape)[valid_cells]
            flat_cols = cells[valid_cells]
            assert np.all((flat_cols >= 0) & (flat_cols < n_examples))
            assert np.unique(flat_rows * n_examples + flat_cols).size == x[-1]
            counts = np.zeros((len(x), n_arms), dtype=int)
            sums = np.zeros((len(x), n_arms), dtype=float)
            counts[np.arange(len(x)), arms] = n_each
            revealed = np.zeros(cells.shape)
            revealed[valid_cells] = matrix[flat_rows, flat_cols]
            sums[np.arange(len(x)), arms] = revealed.sum(axis=1)
            np.testing.assert_array_equal(counts.cumsum(axis=0), trace["counts"])
            np.testing.assert_array_equal(sums.cumsum(axis=0), trace["sums"])
            rec = trace["recommended_arm"]
            eligible = rec >= 0
            true_mean = matrix.mean(axis=1)
            oracle_regret = np.full(rec.shape, np.nan)
            oracle_regret[eligible] = true_mean.max() - true_mean[rec[eligible]]
            np.testing.assert_allclose(oracle_regret, trace["regret"], atol=1e-12)
            np.testing.assert_array_equal(trace["method_keys"], [m["key"] for m in manifest["methods"]])
            selected_n = trace["selected_n"]
            observed_counts = trace["counts"]
            step_rows = np.broadcast_to(np.arange(len(x))[:, None], rec.shape)
            np.testing.assert_array_equal(selected_n[eligible], observed_counts[step_rows[eligible], rec[eligible]])
            assert np.all(selected_n[~eligible] == -1)
            lower, upper = trace["confidence_lower"], trace["confidence_upper"]
            assert np.all(lower <= upper + 1e-12)
            if np.any(lower > true_mean[None, :] + 1e-12) or np.any(upper < true_mean[None, :] - 1e-12):
                violations.append(path.name)
            final_certificate_counts += trace["certified_regret_bound"][-1] <= 0
            native_steps += int(trace["native_steps_verified"])
            pulls += len(x)
    assert seen == expected, ("missing", expected - seen, "unexpected", seen - expected)
    result = {
        "validated_runs": len(seen), "validated_post_pull_states": pulls,
        "native_policy_pulls_verified": native_steps,
        "checks": ["input and source hashes", "complete seed grid", "exact budgets",
                   "unique observed cells", "reconstructed counts and sums",
                   "regret against full matrix oracle", "abstention encoding",
                   "selected-arm observation counts", "ordered confidence intervals"],
        "confidence_coverage_violation_runs": violations,
        "final_zero_regret_certificates": {m["key"]: int(c) for m, c in zip(manifest["methods"], final_certificate_counts)},
        "note": "Observed confidence coverage is a diagnostic, not proof of nominal coverage; the guarantee follows the stated sampling model and union bound.",
    }
    (args.results_dir / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
