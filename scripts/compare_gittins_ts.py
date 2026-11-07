#!/usr/bin/env python3
"""Paper-parameter GSM8K comparison: empirical-target Gittins and Gaussian TS.

Both methods recommend full-test-set posterior mean minus one posterior std.
This runner keeps the existing per-cell Gittins DP, batch likelihood, fixed arm
order, and batch-end stopping convention. Incremental sufficient statistics
avoid rescanning the entire masked accuracy matrix after every observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PRIORS = {"G": (0.5, 0.04), "S": (0.2, 0.01)}
METHODS = ("gittins", "ts")
PRICE_PATH = ROOT / "data_analysis/pricing/gsm8k_various_models_configurations_price_ratio_1to2_rounded.json"


def load_paper_costs(path: Path = PRICE_PATH, n_arms: int = 122) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cost_per_arm" in data:
        data = data["cost_per_arm"]
    if isinstance(data, dict):
        data = [data[str(i)] for i in range(n_arms)]
    values = np.asarray([
        row["estimated_cost_per_1m_input_tokens"] if isinstance(row, dict) else row
        for row in data
    ], dtype=np.float64)
    if values.shape != (n_arms,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be a finite positive vector aligned with matrix rows")
    return values


def build_roots_lookup(*, prior_variance: float, tau_sq_cell: float,
                       n_examples: int, costs_per_arm: np.ndarray,
                       cost_scaling_factor: float = 1e-4, n_points: int = 1025) -> np.ndarray:
    """Use the existing finite-population DP once per distinct float32 price."""
    from gittins_shrinking_posterior import compute_finite_population_roots_lookup_table
    scaled = np.asarray(costs_per_arm, dtype=np.float32) * float(cost_scaling_factor)
    unique, inverse = np.unique(scaled.astype(np.float32), return_inverse=True)
    roots = np.asarray(compute_finite_population_roots_lookup_table(
        prior_variance=prior_variance, tau_sq_cell=tau_sq_cell,
        n_examples=n_examples, costs_per_arm=unique, n_points=n_points,
    ), dtype=np.float32)
    return roots[inverse]


def simulate_policy(
    matrix: np.ndarray, *, policy: str, seed: int,
    prior_mean: float, prior_variance: float, batch_size: int,
    per_arm_cost: np.ndarray, cost_mode: str,
    roots_lookup: np.ndarray | None = None, budget_fraction: float = 0.1,
    obs_noise_variance: float | None = None, std_penalty: float = 1.0,
    max_steps: int | None = None, record_cells: bool = False,
) -> dict[str, Any]:
    """Simulate one policy, preserving native selection and torch random draws.

    ``obs_noise_variance`` is nominal batch-mean noise, as in the existing
    policies. The default corresponds to per-cell variance 0.25. Costs affect
    TS stopping only. Gittins receives unit or actual-price DP roots from the
    caller. Post-batch observations may cross the budget; fixed-budget summaries
    below use only completed observations at or before each checkpoint.
    """
    raw = np.asarray(matrix)
    if raw.ndim != 2 or min(raw.shape) < 1 or not np.isfinite(raw).all():
        raise ValueError("matrix must be finite and nonempty")
    if policy not in METHODS or cost_mode not in {"unit", "cost"}:
        raise ValueError("Unknown policy or cost_mode")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(prior_mean) or not math.isfinite(prior_variance) or prior_variance <= 0:
        raise ValueError("Invalid prior")
    if not math.isfinite(std_penalty) or std_penalty < 0:
        raise ValueError("std_penalty must be finite and nonnegative")
    if not math.isfinite(budget_fraction) or not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must lie in (0, 1]")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    n_arms, n_examples = raw.shape
    costs = np.asarray(per_arm_cost, dtype=np.float64)
    if costs.shape != (n_arms,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Invalid per_arm_cost")
    noise_batch = 1.0 / (4.0 * batch_size) if obs_noise_variance is None else float(obs_noise_variance)
    tau_sq_cell = noise_batch * batch_size
    if not math.isfinite(tau_sq_cell) or tau_sq_cell <= 0:
        raise ValueError("Observation noise must be finite and positive")
    roots = None
    if policy == "gittins":
        if roots_lookup is None:
            raise ValueError("Gittins requires roots_lookup")
        roots = np.asarray(roots_lookup, dtype=np.float32)
        if roots.ndim == 1:
            roots = roots[None]
        if roots.shape not in {(1, n_examples + 1), (n_arms, n_examples + 1)}:
            raise ValueError("roots_lookup must have one or n_arms rows and N+1 columns")
        if not np.isfinite(roots).all():
            raise ValueError("roots_lookup must be finite")
        roots = np.broadcast_to(roots, (n_arms, n_examples + 1))

    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # Native runners store observations in float32; preserve that quantization.
    observations = np.asarray(raw, dtype=np.float32)
    true_means = np.asarray(raw, dtype=np.float64).mean(axis=1)
    mu_star = float(true_means.max())
    seen = np.zeros((n_arms, n_examples), dtype=bool)
    counts = np.zeros(n_arms, dtype=np.int64)
    sums = np.zeros(n_arms, dtype=np.float64)
    ns = np.arange(n_examples + 1, dtype=np.float64)
    remaining = n_examples - ns
    latent_var = 1.0 / (1.0 / prior_variance + ns / tau_sq_cell)
    variances = (remaining ** 2 * latent_var + remaining * tau_sq_cell) / n_examples ** 2
    stds = np.sqrt(variances)
    means = np.full(n_arms, prior_mean, dtype=np.float32)
    scores = means - roots[:, 0] if roots is not None else None
    if cost_mode == "unit":
        budget = float(max(1, round(budget_fraction * n_arms * n_examples)))
        full_budget = float(n_arms * n_examples)
    else:
        full_budget = float(n_examples * costs.sum())
        budget = budget_fraction * full_budget
    hist: dict[str, list] = {k: [] for k in (
        "x", "x_original_cost", "recommended_arm", "recommended_mean", "regret",
        "pulled_arm", "n_new_samples",
    )}
    columns_history = []
    evaluated, spent = 0, 0.0
    natural_stop, recommendation_stop = -1, -1
    stop_reason = "budget"
    while (evaluated if cost_mode == "unit" else spent) < budget:
        if max_steps is not None and len(hist["x"]) >= max_steps:
            stop_reason = "max_steps"
            break
        complete = counts == n_examples
        if complete.all():
            stop_reason = "all_arms_complete"
            break
        if policy == "gittins":
            selection_scores = scores.copy()
            selection_scores[complete] = -np.inf
            arm = int(selection_scores.argmax())
        else:
            z = torch.randn((n_arms,), dtype=torch.float64, generator=generator).numpy()
            draws = means.astype(np.float64) + stds[counts] * z
            draws[complete] = -np.inf
            arm = int(draws.argmax())
        unseen = np.flatnonzero(~seen[arm])
        size = min(batch_size, len(unseen))
        # Native TS caps its unit-budget final batch; B=8 divides this study's
        # GSM8K budget, so this is also identical to batch-end stopping here.
        if policy == "ts" and cost_mode == "unit":
            size = min(size, int(budget - evaluated))
        order = torch.randperm(len(unseen), generator=generator).numpy()
        columns = unseen[order[:size]]
        seen[arm, columns] = True
        sums[arm] += observations[arm, columns].astype(np.float64).sum()
        counts[arm] += size
        evaluated += size
        # Sum identical prices as native torch does, without a large matrix.
        spent += float(torch.full((size,), costs[arm], dtype=torch.float64).sum().item())
        n = int(counts[arm])
        r = n_examples - n
        latent_mean = latent_var[n] * (prior_mean / prior_variance + sums[arm] / tau_sq_cell)
        means[arm] = np.float32((sums[arm] + r * latent_mean) / n_examples)
        if scores is not None:
            scores[arm] = means[arm] if n == n_examples else np.float32(means[arm] - roots[arm, n])
        rec = int((means.astype(np.float64) - std_penalty * stds[counts]).argmax())
        if scores is not None:
            complete = counts == n_examples
            if natural_stop < 0 and complete[int(scores.argmax())]:
                natural_stop = evaluated
            incomplete = ~complete
            if recommendation_stop < 0 and incomplete.any() and scores[incomplete].max() < means[rec]:
                recommendation_stop = evaluated
        values = dict(x=evaluated, x_original_cost=spent, recommended_arm=rec,
                      recommended_mean=float(means[rec]), regret=mu_star - float(true_means[rec]),
                      pulled_arm=arm, n_new_samples=size)
        for name, value in values.items():
            hist[name].append(value)
        if record_cells:
            padded = np.full(batch_size, -1, dtype=np.int64)
            padded[:size] = columns
            columns_history.append(padded)
    out = {k: np.asarray(v, dtype=np.int64 if k in {"x", "recommended_arm", "pulled_arm", "n_new_samples"} else np.float64)
           for k, v in hist.items()}
    out.update(policy=policy, cost_mode=cost_mode, budget=budget, full_budget=full_budget,
               counts=counts, observed_sums=sums, elapsed_s=time.perf_counter() - started,
               stop_reason=stop_reason, natural_stop_cum_eval=natural_stop,
               recommendation_aware_stop_cum_eval=recommendation_stop,
               initial_recommended_arm=0, initial_regret=mu_star - float(true_means[0]))
    if record_cells:
        out["sampled_columns"] = np.asarray(columns_history, dtype=np.int64).reshape(-1, batch_size)
    return out


def grid_trace(trace: dict, grid_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Previous completed recommendation at each common budget, including t=0."""
    axis = trace["x"] if trace["cost_mode"] == "unit" else trace["x_original_cost"]
    targets = np.asarray(grid_fraction, dtype=np.float64) * float(trace["full_budget"])
    tolerance = 32 * np.finfo(np.float64).eps * max(1, float(trace["full_budget"]))
    indices = np.searchsorted(axis, targets + tolerance, side="right") - 1
    regret = np.full(len(targets), trace["initial_regret"], dtype=np.float64)
    arms = np.full(len(targets), trace["initial_recommended_arm"], dtype=np.int64)
    valid = indices >= 0
    regret[valid], arms[valid] = trace["regret"][indices[valid]], trace["recommended_arm"][indices[valid]]
    return regret, arms


def exact_normalized_auc(trace: dict, budget_fraction: float) -> float:
    axis = trace["x"] if trace["cost_mode"] == "unit" else trace["x_original_cost"]
    axis = axis.astype(np.float64) / trace["full_budget"]
    valid = axis <= budget_fraction + 32 * np.finfo(float).eps
    boundaries = np.r_[0.0, np.minimum(axis[valid], budget_fraction), budget_fraction]
    values = np.r_[trace["initial_regret"], trace["regret"][valid]]
    return float(np.dot(np.diff(boundaries), values) / budget_fraction)


def _atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def prepare(out_dir: Path, *, batch_size: int = 8, budget_fraction: float = .1,
            cost_scaling_factor: float = 1e-4, n_points: int = 1025,
            matrix_seeds: list[int] = list(range(1, 6)), run_seeds: list[int] = list(range(20))) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = np.load(ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{matrix_seeds[0]}.npy", mmap_mode="r")
    n_arms, n_examples = first.shape
    matrix_hashes = {}
    for seed in matrix_seeds:
        path = ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{seed}.npy"
        values = np.load(path, mmap_mode="r")
        if values.shape != (n_arms, n_examples):
            raise ValueError(f"Matrix shape differs across seeds: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        matrix_hashes[str(seed)] = digest.hexdigest()
    costs = load_paper_costs(n_arms=n_arms)
    settings = dict(n_models=n_arms, n_examples=n_examples, full_evals=n_arms * n_examples,
                    full_original_cost=float(n_examples * costs.sum()), matrix_seeds=matrix_seeds,
                    run_seeds=run_seeds, priors={k: list(v) for k, v in PRIORS.items()}, batch_size=batch_size,
                    budget_fraction=budget_fraction, recommendation_std_penalty=1.0,
                    per_cell_noise_variance=.25, batch_noise_variance=.25 / batch_size,
                    cost_scaling_factor=cost_scaling_factor, gittins_grid_points=n_points,
                    gittins_target="finite_population_mean", ts_target="finite_population_mean",
                    gittins_dp_transitions="per_cell", fixed_arm_order=True,
                    budget_convention="post_batch_stop; checkpoints exclude over-budget feedback",
                    grid_points=1001, methods=list(METHODS), cost_modes=["unit", "cost"],
                    matrix_sha256=matrix_hashes,
                    ordered_cost_vector_sha256=hashlib.sha256(costs.astype("<f8").tobytes()).hexdigest())
    canonical = json.dumps(settings, sort_keys=True)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    settings["settings_hash"] = fingerprint
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("settings_hash") != fingerprint:
            raise ValueError("Output directory has different experiment settings; choose a new --out-dir")
    cache = out_dir / "roots.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if str(z["settings_hash"].item()) != fingerprint:
                raise ValueError("Root cache settings do not match this study")
    else:
        values = {"settings_hash": fingerprint, "per_arm_cost": costs}
        started = time.perf_counter()
        for name, (_, variance) in PRIORS.items():
            for mode in ("unit", "cost"):
                prices = np.ones(n_arms) if mode == "unit" else costs
                tick = time.perf_counter()
                values[f"{name}_{mode}_roots"] = build_roots_lookup(
                    prior_variance=variance, tau_sq_cell=.25, n_examples=n_examples,
                    costs_per_arm=prices, cost_scaling_factor=cost_scaling_factor, n_points=n_points)
                print(json.dumps({"prepare_prior": name, "mode": mode, "elapsed_s": time.perf_counter() - tick}), flush=True)
        values["precomputation_s"] = time.perf_counter() - started
        _atomic_npz(cache, **values)
    if not manifest_path.exists():
        temporary_manifest = manifest_path.with_name("manifest.json.tmp")
        temporary_manifest.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
    return settings


def run_worker(args: argparse.Namespace, settings: dict) -> None:
    out = args.out_dir
    grid = np.linspace(0.0, args.eval_budget_fraction, 1001)
    with np.load(out / "roots.npz", allow_pickle=False) as z:
        costs = z["per_arm_cost"].copy()
        roots = {f"{name}_{mode}": z[f"{name}_{mode}_roots"].copy()
                 for name in PRIORS for mode in ("unit", "cost")}
    records = []
    pairs = [(m, s) for m in args.matrix_seeds for s in args.run_seeds]
    matrices = {}
    worker_started = time.perf_counter()
    for pair_index, (matrix_seed, seed) in enumerate(pairs):
        if pair_index % args.workers != args.worker_index:
            continue
        if matrix_seed not in matrices:
            matrices[matrix_seed] = np.load(ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{matrix_seed}.npy")
        matrix = matrices[matrix_seed]
        for name, (mean, variance) in PRIORS.items():
            for mode in ("unit", "cost"):
                path = out / "runs" / f"matrix{matrix_seed}_seed{seed}_{name}_{mode}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        if str(z["settings_hash"].item()) != settings["settings_hash"]:
                            raise ValueError(f"Settings mismatch: {path}")
                        case = {k: z[k].copy() for k in z.files}
                else:
                    case = dict(matrix_seed=matrix_seed, run_seed=seed, prior_name=name,
                                cost_mode=mode, settings_hash=settings["settings_hash"], grid_fraction=grid)
                    for policy in METHODS:
                        trace = simulate_policy(matrix, policy=policy, seed=seed,
                            prior_mean=mean, prior_variance=variance, batch_size=args.batch_size,
                            per_arm_cost=costs, cost_mode=mode,
                            roots_lookup=roots[f"{name}_{mode}"] if policy == "gittins" else None,
                            budget_fraction=args.eval_budget_fraction)
                        for key in ("x", "x_original_cost", "recommended_arm", "recommended_mean", "regret",
                                    "pulled_arm", "n_new_samples", "elapsed_s", "budget", "full_budget",
                                    "natural_stop_cum_eval", "recommendation_aware_stop_cum_eval"):
                            case[f"{policy}_{key}"] = trace[key]
                        gr, ga = grid_trace(trace, grid)
                        case[f"{policy}_grid_regret"], case[f"{policy}_grid_arm"] = gr, ga
                        case[f"{policy}_auc_normalized"] = exact_normalized_auc(trace, args.eval_budget_fraction)
                    _atomic_npz(path, **case)
                records.append(case)
                progress = dict(worker_index=args.worker_index, cases_complete=len(records),
                                matrix_seed=matrix_seed, run_seed=seed, prior_name=name, cost_mode=mode,
                                gittins_final_regret=float(case["gittins_grid_regret"][-1]),
                                ts_final_regret=float(case["ts_grid_regret"][-1]),
                                elapsed_s=time.perf_counter() - worker_started)
                (out / f"worker_{args.worker_index}_progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    _atomic_npz(out / f"worker_{args.worker_index}.npz",
        settings_hash=settings["settings_hash"], grid_fraction=grid,
        methods=np.asarray(METHODS),
        matrix_seed=np.asarray([int(c["matrix_seed"]) for c in records]),
        run_seed=np.asarray([int(c["run_seed"]) for c in records]),
        prior_name=np.asarray([str(np.asarray(c["prior_name"]).item()) for c in records]),
        cost_mode=np.asarray([str(np.asarray(c["cost_mode"]).item()) for c in records]),
        grid_regret=np.asarray([[c[f"{p}_grid_regret"] for p in METHODS] for c in records]),
        grid_arm=np.asarray([[c[f"{p}_grid_arm"] for p in METHODS] for c in records]),
        auc_normalized=np.asarray([[c[f"{p}_auc_normalized"] for p in METHODS] for c in records]),
        elapsed_s=time.perf_counter() - worker_started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/gsm8k_gittins_ts_lcb100")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix-seeds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-budget-fraction", type=float, default=.1)
    parser.add_argument("--cost-scaling-factor", type=float, default=1e-4)
    parser.add_argument("--gittins-grid-points", type=int, default=1025)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("worker-index must lie in [0, workers)")
    if args.batch_size < 1 or not 0 < args.eval_budget_fraction <= 1:
        parser.error("Invalid batch size or budget fraction")
    if len(set(args.matrix_seeds)) != len(args.matrix_seeds) or len(set(args.run_seeds)) != len(args.run_seeds):
        parser.error("Seed lists must not contain duplicates")
    settings = prepare(args.out_dir, batch_size=args.batch_size, budget_fraction=args.eval_budget_fraction,
                       cost_scaling_factor=args.cost_scaling_factor, n_points=args.gittins_grid_points,
                       matrix_seeds=args.matrix_seeds, run_seeds=args.run_seeds)
    if not args.prepare_only:
        run_worker(args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
