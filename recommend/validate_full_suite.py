"""Independently replay compact two-rule Gittins experiment artifacts.

Every saved pull is checked. This auditor imports no experiment-engine or
recommendation helpers: it rebuilds cumulative statistics in chunks, evaluates
the posterior over every arm, and reproduces the exact seeded cell samples.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHOD_KEYS = ["posterior_mean", "gaussian_lcb_1.645"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_trace(trace, matrix, roots, config, priors, chunk_size=256):
    """Check one loaded trace against raw rewards, returning its identity/size."""
    identity = (str(trace["prior_tag"]), int(trace["matrix_seed"]), int(trace["run_seed"]))
    prior_mean, prior_variance = map(float, priors[identity[0]])
    require(float(trace["prior_mean"]) == prior_mean, "prior mean metadata")
    require(float(trace["prior_variance"]) == prior_variance, "prior variance metadata")
    require(prior_variance > 0, "positive prior variance")
    tau = float(config["tau_sq_cell"])
    require(tau > 0, "positive observation variance")
    n_arms, n_examples = matrix.shape
    require(int(trace["n_arms"]) == n_arms, "arm count metadata")
    require(int(trace["n_examples"]) == n_examples, "example count metadata")
    np.testing.assert_array_equal(trace["method_keys"], METHOD_KEYS)
    true_means = matrix.mean(axis=1, dtype=np.float64)
    np.testing.assert_allclose(trace["true_means"], true_means, rtol=0, atol=1e-12)
    best = float(true_means.max())
    require(roots.shape == (n_examples + 1,), "root-table shape")
    require(roots.dtype == np.float32, "float32 root table")
    require(np.isfinite(roots).all(), "finite root table")

    x = trace["x"]
    arms = trace["pulled_arm"]
    cells = trace["pulled_columns"]
    rec = trace["recommended_arm"]
    scores = trace["scores"]
    variance_selected = trace["selected_variance"]
    n_selected = trace["selected_n"]
    n_steps = len(x)
    batch_size = int(config["batch_size"])
    budget = max(1, round(float(config["budget_fraction"]) * matrix.size))
    require(n_steps > 0 and x.ndim == 1, "nonempty one-dimensional budget trace")
    require(np.issubdtype(x.dtype, np.integer), "integer cell budgets")
    require(arms.shape == (n_steps,), "pulled-arm shape")
    require(np.issubdtype(arms.dtype, np.integer), "integer pulled arms")
    require(cells.shape == (n_steps, batch_size), "pulled-column shape")
    require(np.issubdtype(cells.dtype, np.integer), "integer pulled columns")
    require(np.all((arms >= 0) & (arms < n_arms)), "valid pulled arms")
    require(np.all((cells >= -1) & (cells < n_examples)), "valid columns or -1 padding")
    require(rec.shape == (n_steps, 2), "recommendation shape")
    require(np.issubdtype(rec.dtype, np.integer), "integer recommendations")
    require(np.all((rec >= 0) & (rec < n_arms)), "valid non-abstaining recommendations")
    for key in ("regret", "scores", "selected_n", "selected_variance"):
        require(trace[key].shape == (n_steps, 2), f"{key} shape")
        require(np.isfinite(trace[key]).all(), f"finite {key}")
    require(np.issubdtype(n_selected.dtype, np.integer), "integer selected counts")
    n_each = np.sum(cells >= 0, axis=1)
    require(np.all(n_each > 0), "nonempty pulls")
    np.testing.assert_array_equal(x, np.cumsum(n_each))
    require(int(x[-1]) == budget, "exact final budget")
    np.testing.assert_allclose(trace["regret"], best - true_means[rec], rtol=0, atol=1e-12)

    # Raw-cell replay is independent of both stored sufficient statistics and
    # the production engine's cached arm scores.
    counts = np.zeros(n_arms, dtype=np.int64)
    sums = np.zeros(n_arms, dtype=np.float64)
    observed = np.zeros((n_arms, n_examples), dtype=bool)
    generator = np.random.Generator(np.random.PCG64(identity[2]))
    initial_mu = np.float32((1.0 / (1.0 / prior_variance)) * (prior_mean / prior_variance))
    next_allocation = int(np.argmax(np.full(n_arms, initial_mu, dtype=np.float32) - roots[0]))
    evaluated = 0
    for begin in range(0, n_steps, chunk_size):
        end = min(n_steps, begin + chunk_size)
        rows = end - begin
        count_delta = np.zeros((rows, n_arms), dtype=np.int64)
        sum_delta = np.zeros((rows, n_arms), dtype=np.float64)
        require(int(arms[begin]) == next_allocation, f"allocation at step {begin}")
        for local, step in enumerate(range(begin, end)):
            arm = int(arms[step])
            remaining = np.flatnonzero(~observed[arm])
            size = min(batch_size, len(remaining), budget - evaluated)
            require(size == int(n_each[step]), f"batch size at step {step}")
            require(np.all(cells[step, :size] >= 0) and np.all(cells[step, size:] == -1),
                    f"contiguous column padding at step {step}")
            actual = cells[step, :size]
            expected = remaining[generator.permutation(len(remaining))[:size]]
            require(np.array_equal(actual, expected), f"seeded sampling at step {step}")
            # Matching the sample also establishes no duplicate revealed cells.
            observed[arm, actual] = True
            count_delta[local, arm] = size
            sum_delta[local, arm] = matrix[arm, actual].sum(dtype=np.float64)
            evaluated += size

        reconstructed_counts = np.cumsum(count_delta, axis=0) + counts
        reconstructed_sums = np.cumsum(sum_delta, axis=0) + sums
        variance = 1.0 / (1.0 / prior_variance + reconstructed_counts / tau)
        mean = variance * (prior_mean / prior_variance + reconstructed_sums / tau)
        # The allocation implementation uses the arithmetic posterior, whereas
        # the recommendation reference explicitly preserves the unobserved prior.
        allocation_scores = mean.astype(np.float32) - roots[reconstructed_counts]
        allocation_scores[reconstructed_counts == n_examples] = -np.inf
        allocations = np.argmax(allocation_scores, axis=1)
        np.testing.assert_array_equal(arms[begin + 1:end], allocations[:-1],
                                      err_msg=f"allocation in steps {begin + 1}:{end}")
        next_allocation = int(allocations[-1])
        mean[reconstructed_counts == 0] = prior_mean
        baseline_scores = mean.astype(np.float32)
        lcb_scores = mean - 1.645 * np.sqrt(variance)
        expected_rec = np.column_stack((np.argmax(baseline_scores, axis=1),
                                        np.argmax(lcb_scores, axis=1)))
        np.testing.assert_array_equal(rec[begin:end], expected_rec,
                                      err_msg=f"recommendations in steps {begin}:{end}")
        row_ids = np.arange(rows)
        expected_scores = np.column_stack((baseline_scores[row_ids, expected_rec[:, 0]],
                                           lcb_scores[row_ids, expected_rec[:, 1]]))
        np.testing.assert_allclose(scores[begin:end], expected_scores, rtol=0, atol=1e-12,
                                   err_msg=f"scores in steps {begin}:{end}")
        np.testing.assert_array_equal(n_selected[begin:end],
                                      reconstructed_counts[row_ids[:, None], expected_rec])
        np.testing.assert_allclose(variance_selected[begin:end],
                                   variance[row_ids[:, None], expected_rec], rtol=0, atol=1e-14)
        counts = reconstructed_counts[-1].copy()
        sums = reconstructed_sums[-1].copy()
    require(evaluated == budget == int(counts.sum()) == int(observed.sum()), "reconstructed final budget")
    require(np.all(counts <= n_examples), "no arm exhausted beyond its matrix")
    np.testing.assert_array_equal(trace["final_counts"], counts)
    np.testing.assert_allclose(trace["final_sums"], sums, rtol=0, atol=1e-12)
    return identity, n_steps, budget


def validate_directory(directory, chunk_size=256):
    directory = Path(directory).resolve()
    started = time.perf_counter()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "complete", f"incomplete experiment: {directory}")
    config = manifest["config"]
    require(config.get("sampling_generator") == "numpy.random.PCG64", "recorded PCG64 sampling")
    require([entry["key"] for entry in manifest["methods"]] == METHOD_KEYS, "two-rule method manifest")
    for name, expected in manifest["source_sha256"].items():
        require(digest(ROOT / name) == expected, f"source hash: {name}")
    matrices = {}
    for seed, info in manifest["matrices"].items():
        path = ROOT / info["path"]
        require(digest(path) == info["sha256"], f"input hash: {path}")
        # Alpaca's canonical float32 rewards are summed in float64, exactly as
        # binary MMLU rewards. No thresholding or Bernoulli conversion occurs.
        matrix = np.load(path, allow_pickle=False).astype(np.float64)
        require(matrix.ndim == 2 and min(matrix.shape) > 0, "matrix dimensions")
        require(np.isfinite(matrix).all() and np.all((matrix >= 0) & (matrix <= 1)),
                "complete bounded reward matrix")
        require(list(matrix.shape) == info["shape"], "matrix shape metadata")
        if "best_mean" in info:
            require(abs(float(matrix.mean(1).max()) - float(info["best_mean"])) <= 1e-12,
                    "matrix best mean metadata")
        matrices[int(seed)] = matrix
    for name, expected in manifest.get("roots_sha256", {}).items():
        require(digest(directory / name) == expected, f"root-table hash: {name}")
    expected_runs = {(p, m, s) for p in config["priors"]
                     for m in config["matrix_seeds"] for s in config["run_seeds"]}
    require(len(expected_runs) == len(config["priors"]) * len(config["matrix_seeds"]) * len(config["run_seeds"]),
            "no duplicate seed-grid entries")
    seen = set()
    pulls = cells = 0
    roots_cache = {}
    for path in sorted((directory / "traces").glob("*.npz")):
        with np.load(path, allow_pickle=False) as trace:
            identity = (str(trace["prior_tag"]), int(trace["matrix_seed"]), int(trace["run_seed"]))
            require(identity in expected_runs and identity not in seen, f"unexpected/duplicate trace: {path}")
            matrix = matrices[identity[1]]
            root_key = (identity[0], matrix.shape[1])
            if root_key not in roots_cache:
                roots_cache[root_key] = np.load(directory / "roots" / f"{root_key[0]}_N{root_key[1]}.npy",
                                                allow_pickle=False)
            try:
                checked_identity, steps, budget = audit_trace(trace, matrix, roots_cache[root_key],
                                                               config, manifest["priors"], chunk_size)
            except Exception as error:
                raise AssertionError(f"{path}: {error}") from error
            seen.add(checked_identity)
            pulls += steps
            cells += budget
    require(seen == expected_runs, f"missing seed-grid entries: {sorted(expected_runs - seen)}")
    require(manifest.get("completed_runs", len(seen)) == len(seen), "manifest completed run count")
    result = {
        "status": "passed", "validated_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": config["benchmark"], "validated_runs": len(seen),
        "validated_post_pull_states": pulls, "validated_observed_cells": cells,
        "elapsed_seconds": time.perf_counter() - started,
        "manifest_sha256": digest(manifest_path), "auditor_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "checks": ["complete unique prior/matrix/run seed grid", "input and source hashes",
                   "bounded raw reward matrix and exact oracle means", "every seeded PCG64 cell sample",
                   "unique observed cells and exact cell budgets", "every Gittins allocation from rebuilt statistics",
                   "every float32 posterior-mean recommendation", "every float64 Gaussian LCB recommendation",
                   "every recommended score, observation count and variance", "every oracle regret",
                   "final counts and float64 reward sums"],
        "note": "Independent chunked posterior reconstruction; no sampling of audit steps. Index DP root generation is validated separately.",
    }
    if manifest.get("roots_sha256"):
        result["checks"].append("saved root-table hashes")
    target = directory / "validation.json"
    temporary = directory / "validation.json.tmp"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="*", help="Specific result directories to audit")
    parser.add_argument("--root", type=Path, default=ROOT / "recommend/results/full_suite")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("workers and chunk size must be positive")
    directories = args.directories or sorted(path.parent for path in args.root.glob("*/manifest.json"))
    if not directories:
        parser.error("no result directories found")
    if args.workers == 1:
        for directory in directories:
            print(json.dumps(validate_directory(directory, args.chunk_size)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(validate_directory, path, args.chunk_size): path for path in directories}
            for future in as_completed(jobs):
                print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
