#!/usr/bin/env python3
"""Run configuration-correlated empirical-target Gaussian TS on one matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_correlated_ts import MODEL_VERSION, simulate_correlated_ts
from compare_gittins_ts import _atomic_npz, exact_normalized_auc
from run_mfkg_baseline import load_data


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
    parser.add_argument("--recommendation-std-penalty", type=float, default=1.0)
    parser.add_argument("--eval-budget-fraction", type=float, default=0.1)
    parser.add_argument("--kernel-mode", choices=("fixed", "fit"), default="fit")
    parser.add_argument("--fit-every", type=int, default=32)
    parser.add_argument("--fit-maxiter", type=int, default=25)
    parser.add_argument("--max-steps", type=int, help="Optional smoke-test cap; omit for full experiments.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    data = load_data(args.matrix, args.bo_inputs, args.cost_vector)
    mode = "cost" if args.cost_aware or args.cost_vector is not None else "unit"
    output = args.out or Path("outputs/correlated_ts") / f"{args.matrix.stem}_{mode}_{args.kernel_mode}_seed{args.seed}.npz"
    if output.suffix != ".npz":
        parser.error("out must end in .npz")
    trace = simulate_correlated_ts(
        data.matrix, features=data.features, seed=args.seed,
        prior_mean=args.prior_mean, prior_variance=args.prior_variance,
        tau_sq_cell=args.tau_sq_cell, std_penalty=args.recommendation_std_penalty,
        batch_size=args.batch_size, per_arm_cost=data.per_cell_cost, cost_mode=mode,
        budget_fraction=args.eval_budget_fraction, kernel_mode=args.kernel_mode,
        fit_every=args.fit_every, fit_maxiter=args.fit_maxiter,
        max_steps=args.max_steps, record_cells=True,
    )
    metadata = {
        "model_version": MODEL_VERSION, "policy_family": "correlated_ts",
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "bo_inputs": str(args.bo_inputs.resolve()) if args.bo_inputs else None,
        "feature_source": data.feature_source,
        "ordered_feature_sha256": hashlib.sha256(data.features.numpy().astype("<f8").tobytes()).hexdigest(),
        "n_arms": len(data.matrix), "n_examples": data.matrix.shape[1],
        "seed": args.seed, "batch_size": args.batch_size,
        "prior_mean": args.prior_mean, "prior_variance": args.prior_variance,
        "tau_sq_cell": args.tau_sq_cell, "budget_fraction": args.eval_budget_fraction,
        "target": "complete_fixed_benchmark_empirical_mean",
        "recommendation_std_penalty": args.recommendation_std_penalty,
        "cost_mode": mode, "allocation_uses_prices": False,
        "kernel_mode": args.kernel_mode, "fit_every": args.fit_every,
        "fit_maxiter": args.fit_maxiter, "max_steps": args.max_steps,
        "kernel": json.loads(trace["kernel_final_json"]),
        "budget_average_regret": exact_normalized_auc(trace, args.eval_budget_fraction),
        "elapsed_s": float(trace["elapsed_s"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output, **trace, features=data.features.numpy(),
                per_arm_cost=data.per_cell_cost, metadata_json=json.dumps(metadata))
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(trace['x'])} batches to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
