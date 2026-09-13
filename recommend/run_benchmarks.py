#!/usr/bin/env python3
"""Run unchanged recommendation grids on PIQA or selected MMLU subjects."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "recommend"))
from experiment_core import run_path
from portable_gittins import compute_roots
from methods import method_specs

DEFAULT_PRIOR = (0.5, 0.04)
MMLU_PRIORS = {"low": (0.4, 0.02), "medium": (0.6, 0.02), "high": (0.75, 0.01)}
DEFAULT_TASKS = ["computer_security", "anatomy", "business_ethics"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_specs(args):
    if args.benchmark == "piqa":
        yield ("piqa", "PIQA", {
            seed: ROOT / f"data/BanditEval_matrices/piqa_1_samples_various_models_seed{seed}.npy"
            for seed in args.matrix_seeds
        }, (0.4, 0.02), 16)
        return
    metadata = json.loads((ROOT / "data/MMLU_matrices/task_metadata.json").read_text())
    buckets = {row["task"]: row["dataset_prior_bucket"] for row in metadata["tasks"]}
    for subject in args.tasks:
        if subject not in buckets:
            raise ValueError(f"MMLU subject missing from prior metadata: {subject}")
        yield (f"mmlu_{subject}", f"MMLU: {subject.replace('_', ' ').title()}",
               {1: ROOT / f"data/MMLU_matrices/{subject}.npy"},
               MMLU_PRIORS[buckets[subject]], 4)


def run_benchmark(args, slug, label, paths, dataset_prior, batch_default):
    out = args.out_root.resolve() / slug
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "roots").mkdir(exist_ok=True)
    batch_size = args.batch_size or batch_default
    priors = {"default": DEFAULT_PRIOR, "dataset": dataset_prior}
    matrices = {}
    inputs = {}
    for seed, path in paths.items():
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or not np.isin(matrix, [0, 1]).all():
            raise ValueError(f"Expected complete binary matrix: {path}")
        matrices[seed] = matrix
        inputs[str(seed)] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                            "shape": list(matrix.shape), "best_mean": float(matrix.mean(1).max())}
    config = {
        "benchmark": slug, "benchmark_label": label, "priors": args.priors,
        "matrix_seeds": list(paths), "run_seeds": args.run_seeds,
        "batch_size": batch_size, "budget_fraction": args.budget_fraction,
        "cost_scale": args.cost_scale, "tau_sq_cell": 0.25,
        "grid_points": 1025, "dp_backend": "NumPy/SciPy float64 m-diff FFT",
        "sampling_generator": "numpy.random.PCG64",
        "primary_method": "gaussian_lcb_1.645",
    }
    sources = [Path(__file__), ROOT / "recommend/experiment_core.py",
               ROOT / "recommend/portable_gittins.py",
               ROOT / "recommend/run_gsm8k.py", ROOT / "recommend/methods.py",
               ROOT / "src/gittins_lookup.py", ROOT / "src/gittins_policy.py",
               ROOT / "src/gittins_shrinking_posterior.py", ROOT / "src/q_estimation.py"]
    if args.benchmark == "mmlu":
        sources.append(ROOT / "data/MMLU_matrices/task_metadata.json")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "methods": [asdict(spec) for spec in method_specs()], "priors": priors,
        "matrices": inputs, "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy"]},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "design": "Same fixed Gittins observations for all 18 rules; exact unit-cost budget; explicit abstention; no diagnostic early stop.",
        "status": "running",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Results exist; use a new --out-root or --resume: {out}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = json.loads(json.dumps(manifest))
        for key in ["config", "methods", "priors", "source_sha256", "matrices", "versions"]:
            if previous[key] != current[key]:
                raise ValueError(f"Cannot resume changed {key}: {out}")
        manifest["created_utc"] = previous["created_utc"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    completed = 0
    total = len(args.priors) * len(paths) * len(args.run_seeds)
    for prior_tag in args.priors:
        prior_pair = priors[prior_tag]
        roots_cache = {}
        for matrix_seed, matrix in matrices.items():
            n_examples = matrix.shape[1]
            if n_examples not in roots_cache:
                roots = compute_roots(prior_pair[1], n_examples, args.cost_scale)
                roots_cache[n_examples] = roots
                np.save(out / "roots" / f"{prior_tag}_N{n_examples}.npy", roots)
                print(f"{slug}: {prior_tag} prior={prior_pair}, B={batch_size}, N={n_examples}", flush=True)
            for run_seed in args.run_seeds:
                target = out / "traces" / f"{prior_tag}_matrix{matrix_seed}_run{run_seed:02d}.npz"
                if args.resume and target.is_file():
                    with np.load(target, allow_pickle=False) as saved:
                        assert saved["x"][-1] == round(args.budget_fraction * matrix.size)
                        assert saved["regret"].shape[1] == len(method_specs())
                    completed += 1
                    continue
                result = run_path(
                    matrix, roots_cache[n_examples], prior_tag=prior_tag, prior_pair=prior_pair,
                    run_seed=run_seed, matrix_seed=matrix_seed, batch_size=batch_size,
                    budget_fraction=args.budget_fraction, tau_sq_cell=0.25)
                result["benchmark"] = np.asarray(slug)
                np.savez_compressed(target, **result)
                completed += 1
                print(f"{slug} [{completed}/{total}] {target.stem}: regret={result['regret'][-1, 0]:.4f}; "
                      f"{time.perf_counter()-started:.1f}s elapsed", flush=True)
    manifest.update(status="complete", completed_runs=completed,
                    elapsed_seconds=time.perf_counter() - started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["piqa", "mmlu"], required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--matrix-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--priors", choices=["default", "dataset"], nargs="+", default=["default", "dataset"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--cost-scale", type=float, default=1e-4)
    parser.add_argument("--out-root", type=Path, default=ROOT / "recommend/results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.budget_fraction <= 1 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("budget fraction must lie in (0,1], and batch size must be positive")
    for field in ["tasks", "matrix_seeds", "run_seeds", "priors"]:
        values = getattr(args, field)
        if len(values) != len(set(values)):
            parser.error(f"duplicate {field} are not permitted")
    for specification in benchmark_specs(args):
        run_benchmark(args, *specification)


if __name__ == "__main__":
    main()
