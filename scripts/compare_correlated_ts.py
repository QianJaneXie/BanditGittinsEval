#!/usr/bin/env python3
"""Paired GSM8K independent and configuration-correlated Gaussian TS.

Both policies use fixed-benchmark empirical targets, B=8 and LCB coefficient
one by default. Configuration correlations come from the same categorical
kernel family used by the existing mixed-input BO surrogate. Prices affect
the stopping budget, never the sampled accuracy objective. Outputs stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compare_gittins_ts import (  # noqa: E402
    PRIORS, _atomic_npz, exact_normalized_auc, load_paper_costs, simulate_policy,
)
from run_mfkg_baseline import load_data  # noqa: E402

MODEL_VERSION = "categorical_empirical_correlated_ts_v1"
METHODS = ("ts", "correlated_ts")


def simulate_correlated_ts(
    matrix: np.ndarray, *, features: torch.Tensor, seed: int,
    prior_mean: float, prior_variance: float, batch_size: int,
    per_arm_cost: np.ndarray, cost_mode: str, budget_fraction: float = 0.1,
    tau_sq_cell: float = 0.25, std_penalty: float = 1.0,
    kernel_mode: str = "fit", fit_every: int = 32, fit_maxiter: int = 25,
    max_steps: int | None = None, record_cells: bool = True,
) -> dict[str, Any]:
    """Supply the policy with revealed batches only, and score after feedback."""
    from correlated_ts_policy import CorrelatedGaussianTS

    raw = np.asarray(matrix)
    if raw.ndim != 2 or min(raw.shape) < 1 or not np.isfinite(raw).all():
        raise ValueError("matrix must be finite and nonempty")
    if cost_mode not in {"unit", "cost"}:
        raise ValueError("cost_mode must be unit or cost")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(budget_fraction) or not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must lie in (0,1]")
    if not math.isfinite(std_penalty) or std_penalty < 0:
        raise ValueError("std_penalty must be finite and nonnegative")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    n_arms, n_examples = raw.shape
    costs = np.asarray(per_arm_cost, dtype=np.float64)
    if costs.shape != (n_arms,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("per_arm_cost must be positive, finite and row-aligned")
    if tuple(features.shape[:1]) != (n_arms,):
        raise ValueError("features must align with matrix rows")
    model = CorrelatedGaussianTS(
        features, n_examples=n_examples, prior_mean=prior_mean,
        prior_variance=prior_variance, tau_sq_cell=tau_sq_cell,
        kernel_mode=kernel_mode, fit_every=fit_every, fit_maxiter=fit_maxiter,
    )
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observations = np.asarray(raw, dtype=np.float32)
    true_means = np.asarray(raw, dtype=np.float64).mean(axis=1)
    best = float(true_means.max())
    initial_regret = best - float(true_means[0])
    seen = np.zeros((n_arms, n_examples), dtype=bool)
    if cost_mode == "unit":
        full_budget = float(n_arms * n_examples)
        budget = float(max(1, round(budget_fraction * full_budget)))
    else:
        full_budget = float(n_examples * costs.sum())
        budget = budget_fraction * full_budget
    history = {name: [] for name in (
        "x", "x_original_cost", "recommended_arm", "recommended_mean", "regret",
        "pulled_arm", "n_new_samples",
    )}
    columns_history, fit_history = [], []
    evaluated, spent, stop_reason = 0, 0.0, "budget"
    initial_kernel = json.loads(json.dumps(model.kernel_metadata))
    while (evaluated if cost_mode == "unit" else spent) < budget:
        if max_steps is not None and len(history["x"]) >= max_steps:
            stop_reason = "max_steps"
            break
        complete = model.counts == n_examples
        if bool(complete.all()):
            stop_reason = "all_arms_complete"
            break
        draws = model.sample_target(generator=generator).masked_fill(complete, -torch.inf)
        arm = int(draws.argmax())
        unseen = np.flatnonzero(~seen[arm])
        size = min(batch_size, len(unseen))
        if cost_mode == "unit":
            size = min(size, int(budget - evaluated))
        order = torch.randperm(len(unseen), generator=generator).numpy()
        columns = unseen[order[:size]]
        seen[arm, columns] = True
        old_fit_count = model.fit_count
        model.observe(arm, torch.as_tensor(observations[arm, columns], dtype=torch.float64))
        evaluated += size
        spent += float(torch.full((size,), costs[arm], dtype=torch.float64).sum())
        if model.fit_count != old_fit_count:
            kernel = model.kernel_metadata
            fit_history.append({"batch": len(history["x"]) + 1, "evaluations": evaluated,
                                "kernel": {k: v for k, v in kernel.items() if k != "fit_diagnostics"},
                                "diagnostic": kernel.get("fit_diagnostics", [None])[-1]})
        means, covariance = model.empirical_target_moments()
        recommendation = int((means - std_penalty * covariance.diagonal().clamp_min(0).sqrt()).argmax())
        values = dict(
            x=evaluated, x_original_cost=spent, recommended_arm=recommendation,
            recommended_mean=float(means[recommendation]),
            regret=best - float(true_means[recommendation]), pulled_arm=arm,
            n_new_samples=size,
        )
        for name, value in values.items():
            history[name].append(value)
        if record_cells:
            padded = np.full(batch_size, -1, dtype=np.int64)
            padded[:size] = columns
            columns_history.append(padded)
    integer = {"x", "recommended_arm", "pulled_arm", "n_new_samples"}
    out = {k: np.asarray(v, dtype=np.int64 if k in integer else np.float64)
           for k, v in history.items()}
    out.update(
        policy="correlated_ts", cost_mode=cost_mode, budget=budget,
        full_budget=full_budget, counts=model.counts.numpy().copy(),
        observed_sums=model.sums.numpy().copy(), elapsed_s=time.perf_counter() - started,
        stop_reason=stop_reason, initial_recommended_arm=0, initial_regret=initial_regret,
        fit_count=model.fit_count, kernel_initial_json=json.dumps(initial_kernel),
        kernel_final_json=json.dumps(model.kernel_metadata),
        kernel_fit_history_json=json.dumps(fit_history),
    )
    if record_cells:
        out["sampled_columns"] = np.asarray(columns_history, dtype=np.int64).reshape(-1, batch_size)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    """Fingerprint protocol, alignment and numerical source before resuming."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_hashes, feature_hashes, shapes = {}, {}, set()
    costs = load_paper_costs()
    for seed in args.matrix_seeds:
        matrix = ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{seed}.npy"
        bo = ROOT / f"data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed{seed}_bo_inputs.npz"
        data = load_data(matrix, bo)
        shapes.add(data.matrix.shape)
        np.testing.assert_allclose(data.per_cell_cost, costs, rtol=0, atol=1e-12)
        matrix_hashes[str(seed)] = _sha256(matrix)
        feature_hashes[str(seed)] = hashlib.sha256(data.features.numpy().astype("<f8").tobytes()).hexdigest()
    if len(shapes) != 1:
        raise ValueError("All paired response matrices must have the same shape")
    n_arms, n_examples = next(iter(shapes))
    sources = ["src/correlated_ts_policy.py", "src/gaussian_ts_policy.py",
               "src/simple_regret_recommend.py", "scripts/compare_correlated_ts.py",
               "scripts/compare_gittins_ts.py", "scripts/run_mfkg_baseline.py"]
    import botorch
    settings = dict(
        model_version=MODEL_VERSION, methods=list(METHODS), matrix_seeds=args.matrix_seeds,
        run_seeds=args.run_seeds, n_arms=n_arms, n_examples=n_examples,
        batch_size=args.batch_size, budget_fraction=args.eval_budget_fraction,
        priors={k: list(v) for k, v in PRIORS.items()}, per_cell_noise_variance=0.25,
        recommendation_std_penalty=1.0, target="complete_fixed_benchmark_empirical_mean",
        kernel_mode=args.kernel_mode, fit_every=args.fit_every, fit_maxiter=args.fit_maxiter,
        kernel="MixedSingleTaskGP_all_categorical_configuration_covariance",
        kernel_marginal_variance="normalized_to_same_prior_variance_as_independent_TS",
        fit_data="observed_arm_sample_means_only_with_noise_0.25_divided_by_observed_count",
        prior_mean_and_variance="fixed_per_panel_not_fit", warmup_batches=0,
        cost_modes=["unit", "cost"], allocation_cost_usage="none_for_both_methods",
        budget_convention="TS_unit_final_batch_cap; cost_post_batch_stop; metrics_exclude_over_budget_feedback",
        unit_exhaustive_cost=n_arms * n_examples,
        heterogeneous_exhaustive_cost=float(n_examples * costs.sum()),
        max_steps=args.max_steps, matrix_sha256=matrix_hashes,
        ordered_feature_sha256=feature_hashes,
        ordered_per_cell_cost_sha256=hashlib.sha256(costs.astype("<f8").tobytes()).hexdigest(),
        source_sha256={name: _sha256(ROOT / name) for name in sources},
        botorch_version=botorch.__version__, torch_version=torch.__version__,
    )
    settings["settings_hash"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    path = args.out_dir / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError("Output directory has different protocol/data/source; choose a new --out-dir")
    else:
        temp = path.with_name(f"manifest.{args.worker_index}.tmp")
        temp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    return settings


def run_worker(args: argparse.Namespace, settings: dict) -> None:
    costs = load_paper_costs()
    pairs = [(matrix, run) for matrix in args.matrix_seeds for run in args.run_seeds]
    cache, completed = {}, 0
    started = time.perf_counter()
    for index, (matrix_seed, seed) in enumerate(pairs):
        if index % args.workers != args.worker_index:
            continue
        if matrix_seed not in cache:
            cache[matrix_seed] = load_data(
                ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{matrix_seed}.npy",
                ROOT / f"data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed{matrix_seed}_bo_inputs.npz",
            )
        data = cache[matrix_seed]
        for prior, (mean, variance) in PRIORS.items():
            for mode in ("unit", "cost"):
                path = args.out_dir / "runs" / f"matrix{matrix_seed}_seed{seed}_{prior}_{mode}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as archive:
                        if str(archive["settings_hash"].item()) != settings["settings_hash"]:
                            raise ValueError(f"Incompatible existing case: {path}")
                else:
                    case = dict(matrix_seed=matrix_seed, run_seed=seed, prior_name=prior,
                                cost_mode=mode, settings_hash=settings["settings_hash"])
                    common = dict(seed=seed, prior_mean=mean, prior_variance=variance,
                                  batch_size=args.batch_size, per_arm_cost=costs,
                                  cost_mode=mode, budget_fraction=args.eval_budget_fraction,
                                  max_steps=args.max_steps, record_cells=True)
                    traces = {
                        "ts": simulate_policy(data.matrix, policy="ts", **common),
                        "correlated_ts": simulate_correlated_ts(
                            data.matrix, features=data.features, kernel_mode=args.kernel_mode,
                            fit_every=args.fit_every, fit_maxiter=args.fit_maxiter, **common),
                    }
                    for method, trace in traces.items():
                        case.update({f"{method}_{key}": value for key, value in trace.items()})
                        case[f"{method}_auc_normalized"] = exact_normalized_auc(trace, args.eval_budget_fraction)
                    _atomic_npz(path, **case)
                completed += 1
                progress = dict(worker_index=args.worker_index, cases_complete=completed,
                                matrix_seed=matrix_seed, run_seed=seed, prior_name=prior,
                                cost_mode=mode, elapsed_s=time.perf_counter() - started)
                (args.out_dir / f"worker_{args.worker_index}_progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    print(json.dumps({"worker_index": args.worker_index, "complete": completed,
                      "elapsed_s": time.perf_counter() - started}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/gsm8k_correlated_ts_lcb100")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix-seeds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-budget-fraction", type=float, default=0.1)
    parser.add_argument("--kernel-mode", choices=("fixed", "fit"), default="fit")
    parser.add_argument("--fit-every", type=int, default=32)
    parser.add_argument("--fit-maxiter", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=None, help="Explicit smoke cap; omitted for complete trials.")
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("worker-index must lie in [0,workers)")
    if args.batch_size < 1 or not 0 < args.eval_budget_fraction <= 1:
        parser.error("Invalid batch size or budget fraction")
    if args.fit_every < 1 or args.fit_maxiter < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Fit controls and any max-steps cap must be positive")
    if (not args.matrix_seeds or not args.run_seeds or
            len(set(args.matrix_seeds)) != len(args.matrix_seeds) or
            len(set(args.run_seeds)) != len(args.run_seeds) or
            not set(args.matrix_seeds) <= set(range(1, 6)) or min(args.run_seeds) < 0):
        parser.error("Use distinct matrix seeds1..5 and nonnegative run seeds")
    torch.set_num_threads(1)
    settings = prepare(args)
    if not args.prepare_only:
        run_worker(args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
