#!/usr/bin/env python3
"""Paired recommendation experiments on unchanged Gaussian Gittins allocations.

Uses the project's JAX root computation and an incremental implementation of
the same index rule. A native-policy parity check verifies arm and cell choices.
All recommendation rules see every identical post-pull observation history.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "recommend"))

from gittins_lookup import compute_roots_lookup_table
from gittins_policy import gittins_index_exploration
from gittins_shrinking_posterior import transition_stds_shrinking_gaussian_posterior
from methods import RecommendationRules, method_specs


PRIORS = {"default": (0.5, 0.04), "dataset": (0.2, 0.01)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posterior(counts, sums, prior_mean, prior_variance, tau_sq_cell):
    variance = 1.0 / (1.0 / prior_variance + counts.astype(np.float64) / tau_sq_cell)
    means = variance * (prior_mean / prior_variance + sums / tau_sq_cell)
    return means.astype(np.float32), variance


def next_arm(counts, sums, roots, n_examples, prior_mean, prior_variance, tau_sq_cell):
    means, _ = posterior(counts, sums, prior_mean, prior_variance, tau_sq_cell)
    scores = means - roots[counts]
    # The experiment runners continue beyond the diagnostic natural stop.
    scores[counts == n_examples] = -np.inf
    return int(np.argmax(scores))


def run_path(
    matrix: np.ndarray, roots: np.ndarray, *, prior_tag: str, run_seed: int,
    matrix_seed: int, batch_size: int, budget_fraction: float,
    cost_scale: float, tau_sq_cell: float, grid_points: int, native_steps: int,
) -> dict[str, np.ndarray]:
    prior_mean, prior_variance = PRIORS[prior_tag]
    n_arms, n_examples = matrix.shape
    budget = max(1, round(budget_fraction * matrix.size))
    rules = RecommendationRules(n_arms, n_examples, prior_mean, prior_variance, tau_sq_cell)
    specs = method_specs()
    keys = [spec.key for spec in specs]
    counts = np.zeros(n_arms, dtype=np.int64)
    sums = np.zeros(n_arms, dtype=np.float64)
    observed = np.zeros(matrix.shape, dtype=bool)
    generator = torch.Generator(device="cpu").manual_seed(run_seed)
    true_means = matrix.mean(axis=1)
    best_mean = float(true_means.max())
    native_obs = torch.full(matrix.shape, float("nan"), dtype=torch.float32) if native_steps else None
    native_scores = torch.full((n_arms,), float("inf"), dtype=torch.float32)
    roots_tensor = torch.tensor(roots[None, :], dtype=torch.float32)
    previous_arm = None
    history = {key: [] for key in [
        "x", "recommended_arm", "regret", "selected_n", "selected_variance", "scores",
        "eligible_count", "pulled_arm", "pulled_columns", "counts", "sums",
        "confidence_lower", "confidence_upper", "certified_regret_bound",
    ]}
    evaluated = 0
    step = 0
    while evaluated < budget:
        arm = next_arm(counts, sums, roots, n_examples, prior_mean, prior_variance, tau_sq_cell)
        remaining = np.flatnonzero(~observed[arm])
        # Exact cell budget: only the last batch can be smaller than native runs.
        size = min(batch_size, len(remaining), budget - evaluated)
        if size == 0:
            break
        if step < native_steps:
            torch.set_rng_state(generator.get_state())
            native_batch = gittins_index_exploration(
                native_obs, prior_mean=prior_mean, prior_variance=prior_variance,
                obs_noise_variance=tau_sq_cell / batch_size,
                cost_per_transition=1.0, cost_scaling_factor=cost_scale,
                batch_size=batch_size, n_gittins_grid_points=grid_points,
                cached_scores=native_scores,
                recompute_arms=None if previous_arm is None else [previous_arm],
                use_batch_mean_gittins_dp=False, batch_observation_model=True,
                roots_lookup_table=roots_tensor, allow_early_stop=False,
            )
        perm = torch.randperm(len(remaining), generator=generator)[:size].numpy()
        columns = remaining[perm]
        if step < native_steps:
            if native_batch is None or int(native_batch[0, 0]) != arm:
                raise AssertionError(f"Native Gittins arm mismatch at step {step}, {prior_tag}")
            np.testing.assert_array_equal(native_batch[1, :size].numpy(), columns)
            native_obs[arm, columns] = torch.tensor(matrix[arm, columns], dtype=torch.float32)
        observed[arm, columns] = True
        counts[arm] += size
        sums[arm] += float(matrix[arm, columns].sum())
        evaluated += size
        recommendations = rules.evaluate(counts, sums)
        _, variance = posterior(counts, sums, prior_mean, prior_variance, tau_sq_cell)
        arms = np.array([recommendations[key].arm for key in keys], dtype=np.int32)
        valid = arms >= 0
        regret = np.full(len(keys), np.nan)
        selected_n = np.full(len(keys), -1, dtype=np.int32)
        selected_var = np.full(len(keys), np.nan)
        regret[valid] = best_mean - true_means[arms[valid]]
        selected_n[valid] = counts[arms[valid]]
        selected_var[valid] = variance[arms[valid]]
        lower, upper = rules.confidence_bounds(counts, sums)
        # Simultaneous intervals give an honest bound on regret of any chosen arm.
        bound = np.full(len(keys), np.nan)
        for m in np.flatnonzero(valid):
            others = np.arange(n_arms) != arms[m]
            bound[m] = max(0.0, float(np.max(upper[others], initial=0.0) - lower[arms[m]]))
        padded_columns = np.full(batch_size, -1, dtype=np.int32)
        padded_columns[:size] = columns
        history["x"].append(evaluated)
        history["recommended_arm"].append(arms)
        history["regret"].append(regret)
        history["selected_n"].append(selected_n)
        history["selected_variance"].append(selected_var)
        history["scores"].append([recommendations[key].score for key in keys])
        history["eligible_count"].append([recommendations[key].eligible_count for key in keys])
        history["pulled_arm"].append(arm)
        history["pulled_columns"].append(padded_columns)
        history["counts"].append(counts.astype(np.uint16).copy())
        history["sums"].append(sums.astype(np.float32).copy())
        history["confidence_lower"].append(lower)
        history["confidence_upper"].append(upper)
        history["certified_regret_bound"].append(bound)
        previous_arm = arm
        step += 1
    result = {key: np.asarray(value) for key, value in history.items()}
    result.update(
        prior_tag=np.asarray(prior_tag), prior_mean=np.asarray(prior_mean),
        prior_variance=np.asarray(prior_variance), run_seed=np.asarray(run_seed),
        matrix_seed=np.asarray(matrix_seed), method_keys=np.asarray(keys),
        n_arms=np.asarray(n_arms), n_examples=np.asarray(n_examples),
        true_means=true_means, native_steps_verified=np.asarray(min(native_steps, step)),
    )
    # Invariants catch duplicate evaluations and incorrect regret/abstention handling.
    assert evaluated == budget == int(counts.sum())
    assert np.all(counts <= n_examples)
    assert observed.sum() == evaluated
    assert np.all(np.isnan(result["regret"]) == (result["recommended_arm"] < 0))
    assert np.nanmin(result["regret"]) >= -1e-12
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "recommend/results/gsm8k")
    parser.add_argument("--matrix-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--priors", nargs="+", choices=PRIORS, default=list(PRIORS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--cost-scale", type=float, default=1e-4)
    parser.add_argument("--tau-sq-cell", type=float, default=0.25)
    parser.add_argument("--grid-points", type=int, default=1025)
    parser.add_argument("--validate-native-steps", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.budget_fraction <= 1 or args.batch_size < 1:
        parser.error("budget fraction must be in (0,1] and batch size positive")
    torch.set_num_threads(1)
    out = args.out_dir.resolve()
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "roots").mkdir(exist_ok=True)
    matrices = {}
    for seed in args.matrix_seeds:
        path = ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{seed}.npy"
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or not np.isin(matrix, [0, 1]).all():
            raise ValueError(f"Expected a complete binary GSM8K matrix: {path}")
        matrices[seed] = (matrix, path)
    config = {key: value for key, value in vars(args).items() if key not in ["out_dir", "resume"]}
    source_paths = [Path(__file__), ROOT / "recommend/methods.py", ROOT / "src/gittins_lookup.py",
                    ROOT / "src/gittins_policy.py", ROOT / "src/q_estimation.py",
                    ROOT / "src/gittins_shrinking_posterior.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    inputs = {str(seed): {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
              "shape": list(matrix.shape), "best_mean": float(matrix.mean(axis=1).max())}
              for seed, (matrix, path) in matrices.items()}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "methods": [asdict(spec) for spec in method_specs()], "priors": PRIORS,
        "matrices": inputs, "source_sha256": source_hashes,
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "matplotlib", "torch", "jax", "jaxlib", "jaxtyping"]},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "design": "Identical Gittins observations for every recommendation; 10% exact unit-cost budget; final batch truncated; no diagnostic early stop.",
        "status": "running",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Results already exist; choose a new directory or --resume: {out}")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ["config", "methods", "source_sha256", "matrices", "versions"]:
            if old[key] != manifest[key]:
                raise ValueError(f"Cannot resume results after {key} changed")
        manifest["created_utc"] = old["created_utc"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    completed = 0
    total = len(args.priors) * len(matrices) * len(args.run_seeds)
    for prior_tag in args.priors:
        prior_mean, prior_variance = PRIORS[prior_tag]
        roots_by_n = {}
        for matrix_seed, (matrix, _path) in matrices.items():
            n_examples = matrix.shape[1]
            if n_examples not in roots_by_n:
                transitions = transition_stds_shrinking_gaussian_posterior(
                    np.float32(prior_variance), np.float32(args.tau_sq_cell), n_examples)
                roots = np.asarray(compute_roots_lookup_table(
                    transition_stds=transitions, costs_per_arm=np.float32(args.cost_scale),
                    n_points=args.grid_points), dtype=np.float32)[0]
                roots_by_n[n_examples] = roots
                np.save(out / "roots" / f"{prior_tag}_N{n_examples}.npy", roots)
                print(f"Precomputed {prior_tag} roots, {n_examples} stages", flush=True)
            for run_seed in args.run_seeds:
                path = out / "traces" / f"{prior_tag}_matrix{matrix_seed}_run{run_seed:02d}.npz"
                if args.resume and path.is_file():
                    with np.load(path, allow_pickle=False) as old_trace:
                        assert old_trace["x"][-1] == round(args.budget_fraction * matrix.size)
                        assert old_trace["regret"].shape[1] == len(method_specs())
                    completed += 1
                    continue
                validate = args.validate_native_steps if matrix_seed == args.matrix_seeds[0] and run_seed == args.run_seeds[0] else 0
                result = run_path(matrix, roots_by_n[n_examples], prior_tag=prior_tag,
                    run_seed=run_seed, matrix_seed=matrix_seed, batch_size=args.batch_size,
                    budget_fraction=args.budget_fraction, cost_scale=args.cost_scale,
                    tau_sq_cell=args.tau_sq_cell, grid_points=args.grid_points, native_steps=validate)
                np.savez_compressed(path, **result)
                completed += 1
                elapsed = time.perf_counter() - started
                print(f"[{completed}/{total}] {path.stem}: {len(result['x'])} batches, "
                      f"baseline regret={result['regret'][-1, 0]:.4f}, {elapsed:.1f}s elapsed", flush=True)
    manifest["status"] = "complete"
    manifest["completed_runs"] = completed
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
