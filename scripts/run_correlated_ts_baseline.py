#!/usr/bin/env python3
"""Run configuration-correlated empirical-target Gaussian TS on one matrix."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compare_correlated_ts import MODEL_VERSION, simulate_correlated_ts
from compare_gittins_ts import _atomic_npz, exact_normalized_auc
from run_mfkg_baseline import load_data


def _experiment_hash(metadata: dict) -> str:
    # Paths are provenance, not experiment settings. Hash the actual inputs.
    settings = {key: value for key, value in metadata.items()
                if key not in ("matrix", "bo_inputs", "feature_source", "recommendation_rule")}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


@contextmanager
def _output_lock(output: Path):
    lock = output.with_suffix(output.suffix + ".lock")
    try:
        stream = lock.open("x")
    except FileExistsError as error:
        raise FileExistsError(
            f"Output is locked: {lock}. If a previous process crashed, remove the "
            "lock only after confirming no process is still writing this output."
        ) from error
    try:
        with stream:
            yield
    finally:
        lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bo-inputs", type=Path, help="Aligned categorical X, arm IDs and costs; archive Y is ignored.")
    parser.add_argument("--cost-vector", type=Path, help="Override per-cell evaluation prices.")
    parser.add_argument("--cost-aware", action="store_true", help="Use a cost budget; acquisition still samples accuracy.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prior-mean", type=float, default=0.5)
    parser.add_argument("--prior-variance", type=float, default=0.04)
    parser.add_argument("--tau-sq-cell", type=float, default=0.25)
    parser.add_argument(
        "--recommendation-std-penalty", "--recommendation_std_penalty",
        type=float, default=1.0,
        help="Recommend M - beta*sqrt(V): 0 = posterior mean, 1 = LCB (default). "
             "Only recommendation changes; TS allocation is unaffected.",
    )
    parser.add_argument("--eval-budget-fraction", type=float, default=0.1)
    parser.add_argument("--kernel-mode", choices=("fixed", "fit"), default="fit")
    parser.add_argument("--fit-every", type=int, default=32)
    parser.add_argument("--fit-maxiter", type=int, default=25)
    parser.add_argument("--max-steps", type=int, help="Optional smoke-test cap; omit for full experiments.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not math.isfinite(args.recommendation_std_penalty) or args.recommendation_std_penalty < 0:
        parser.error("--recommendation-std-penalty must be finite and nonnegative")
    data = load_data(args.matrix, args.bo_inputs, args.cost_vector)
    mode = "cost" if args.cost_aware or args.cost_vector is not None else "unit"
    if args.out is not None and args.out.suffix != ".npz":
        parser.error("out must end in .npz")
    metadata = {
        "model_version": MODEL_VERSION, "policy_family": "correlated_ts",
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "bo_inputs": str(args.bo_inputs.resolve()) if args.bo_inputs else None,
        "feature_source": data.feature_source,
        "ordered_feature_sha256": hashlib.sha256(data.features.numpy().astype("<f8").tobytes()).hexdigest(),
        "feature_shape": list(data.features.shape),
        "ordered_cost_sha256": hashlib.sha256(np.asarray(data.per_cell_cost, dtype="<f8").tobytes()).hexdigest(),
        "n_arms": len(data.matrix), "n_examples": data.matrix.shape[1],
        "seed": args.seed, "batch_size": args.batch_size,
        "prior_mean": args.prior_mean, "prior_variance": args.prior_variance,
        "tau_sq_cell": args.tau_sq_cell, "budget_fraction": args.eval_budget_fraction,
        "target": "complete_fixed_benchmark_empirical_mean",
        "recommendation_std_penalty": args.recommendation_std_penalty,
        "recommendation_rule": (
            "finite_population_posterior_mean_minus_std"
            if args.recommendation_std_penalty > 0 else "finite_population_posterior_mean"
        ),
        "cost_mode": mode, "allocation_uses_prices": False,
        "kernel_mode": args.kernel_mode, "fit_every": args.fit_every,
        "fit_maxiter": args.fit_maxiter, "max_steps": args.max_steps,
    }
    metadata["experiment_sha256"] = _experiment_hash(metadata)
    output = args.out or Path("outputs/correlated_ts") / (
        f"{args.matrix.stem}_{mode}_{args.kernel_mode}_seed{args.seed}_"
        f"{metadata['experiment_sha256'][:16]}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        if output.exists():
            with np.load(output, allow_pickle=False) as saved:
                previous = json.loads(str(saved["metadata_json"].item()))
            if previous.get("experiment_sha256") != metadata["experiment_sha256"]:
                raise FileExistsError(f"Refusing to overwrite a different or legacy experiment: {output}")
            # The NPZ is authoritative; repair a missing sidecar after interruption.
            output.with_suffix(".json").write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
            print(f"Reusing completed experiment: {output}")
            return 0
        if output.with_suffix(".json").exists():
            raise FileExistsError(f"Refusing to overwrite an existing metadata sidecar: {output.with_suffix('.json')}")
        trace = simulate_correlated_ts(
            data.matrix, features=data.features, seed=args.seed,
            prior_mean=args.prior_mean, prior_variance=args.prior_variance,
            tau_sq_cell=args.tau_sq_cell, std_penalty=args.recommendation_std_penalty,
            batch_size=args.batch_size, per_arm_cost=data.per_cell_cost, cost_mode=mode,
            budget_fraction=args.eval_budget_fraction, kernel_mode=args.kernel_mode,
            fit_every=args.fit_every, fit_maxiter=args.fit_maxiter,
            max_steps=args.max_steps, record_cells=True,
        )
        metadata.update(
            kernel=json.loads(trace["kernel_final_json"]),
            budget_average_regret=exact_normalized_auc(trace, args.eval_budget_fraction),
            elapsed_s=float(trace["elapsed_s"]),
        )
        _atomic_npz(output, **trace, features=data.features.numpy(),
                    per_arm_cost=data.per_cell_cost, metadata_json=json.dumps(metadata))
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(trace['x'])} batches to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
