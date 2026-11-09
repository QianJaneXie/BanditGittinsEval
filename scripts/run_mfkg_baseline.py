#!/usr/bin/env python3
"""Discrete, nested item-subsampling MF-KG on a fixed accuracy matrix.

The objective f(arm, s) is the deterministic mean of a seeded prefix containing
ceil(s * N) items. A GP models configuration and fidelity jointly. Prefixes are
reused on promotion, so only newly revealed matrix cells are charged. Training
uses prefix means, never the complete-score Y arrays in BO input archives.
By default MF-KG chooses one configuration and its next fidelity jointly
(q=1), allowing variable jumps on the fidelity grid. The fixed-batch protocol
remains available explicitly for compatibility with earlier experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mfkg_policy import fit_mfkg_model, score_mfkg_candidates  # noqa: E402

DEFAULT_FIDELITIES = [0.016, 0.032, 0.064, 0.128, 0.25, 0.5, 1.0]


@dataclass(frozen=True)
class MFData:
    matrix: np.ndarray
    features: torch.Tensor
    per_cell_cost: np.ndarray
    feature_source: str
    dataset: str = "custom"


def load_cost_vector(path: Path, n_arms: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            if "cost_per_arm" in values:
                values = values["cost_per_arm"]
            else:
                values = [values[str(i)] for i in range(n_arms)]
                values = [v["estimated_cost_per_1m_input_tokens"] if isinstance(v, dict) else v for v in values]
    else:
        raise ValueError("--cost-vector must be a .json or .npy per-cell cost vector")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (n_arms,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"cost vector must contain {n_arms} finite, positive per-cell costs")
    return values


def load_data(matrix_path: Path, bo_inputs: Path | None = None, cost_vector: Path | None = None) -> MFData:
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a nonempty, finite (n_arms, n_examples) array")
    n_arms, n_examples = matrix.shape
    features = np.arange(n_arms, dtype=np.float64)[:, None]
    costs = np.ones(n_arms, dtype=np.float64)
    source, dataset = "arm_identity", "custom"
    if bo_inputs is not None:
        with np.load(bo_inputs, allow_pickle=False) as z:
            # Deliberately do not load Y or raw_scores_mean: they reveal the answer.
            features = np.asarray(z["X"], dtype=np.float64)
            arm_ids = np.asarray(z["arm_ids"])
            if arm_ids.shape != (n_arms,) or not np.issubdtype(arm_ids.dtype, np.integer):
                raise ValueError("BO arm_ids must be an integer permutation of matrix row IDs")
            if not np.array_equal(np.sort(arm_ids), np.arange(n_arms)):
                raise ValueError("BO arm_ids must be a permutation of matrix row IDs")
            if features.ndim != 2 or features.shape[0] != n_arms or features.shape[1] < 1:
                raise ValueError("BO X must have one nonempty feature row per matrix arm")
            order = np.argsort(arm_ids)
            features = features[order]
            full_cost = np.asarray(z["cost"], dtype=np.float64)
            if full_cost.shape != (n_arms,):
                raise ValueError("BO cost must have one full-evaluation cost per arm")
            costs = full_cost[order] / n_examples
            metadata = json.loads(str(z["metadata_json"].item())) if "metadata_json" in z.files else {}
            if int(metadata.get("n_examples", n_examples)) != n_examples:
                raise ValueError("BO n_examples does not match the supplied matrix")
            if "dataset" in z.files:
                dataset = str(z["dataset"].item())
        source = "bo_categorical_features"
    if cost_vector is not None:
        costs = load_cost_vector(cost_vector, n_arms)
    if not np.isfinite(features).all() or len(np.unique(features, axis=0)) != n_arms:
        raise ValueError("configuration features must be finite and uniquely identify each arm")
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("per-cell costs must be finite and positive")
    return MFData(matrix, torch.tensor(features, dtype=torch.float64), costs, source, dataset)


def fidelity_counts(n_examples: int, fractions: list[float], samples: list[int] | None = None) -> np.ndarray:
    if samples is not None:
        if not samples or any(n < 1 or n > n_examples for n in samples):
            raise ValueError("--fidelity-samples must be between 1 and n_examples")
        counts = samples
    else:
        if not fractions or any(not math.isfinite(s) or not 0 < s <= 1 for s in fractions):
            raise ValueError("--fidelities must be finite fractions in (0, 1]")
        counts = [int(math.ceil(s * n_examples)) for s in fractions]
    return np.unique([*counts, n_examples]).astype(np.int64)


def select_coverage_arm(features: np.ndarray, available: np.ndarray, selected: list[int], rng: np.random.Generator) -> int:
    """Greedily cover unseen category levels, using seeded random ties."""
    gains = np.zeros(len(available), dtype=np.int64)
    for dim in range(features.shape[1]):
        seen = features[selected, dim] if selected else np.asarray([])
        gains += ~np.isin(features[available, dim], seen)
    return int(rng.choice(available[gains == gains.max()]))


def target_posterior_means(model: Any, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([model.posterior(chunk).mean.squeeze(-1) for chunk in targets.split(batch_size)])


def run_mfkg(
    data: MFData,
    *,
    seed: int = 0,
    eval_budget_fraction: float = 0.1,
    cost_aware: bool = False,
    sample_counts: np.ndarray | None = None,
    allocation_mode: str = "adaptive_fidelity",
    batch_size: int = 16,
    n_init: int | None = None,
    warmup_budget_fraction: float = 0.1,
    warmup_anchors: int = 1,
    warmup_strategy: str = "coverage",
    num_fantasies: int = 64,
    observation_noise: float = 1e-6,
    fit_maxiter: int = 200,
    candidate_batch_size: int = 8,
    target_batch_size: int = 128,
    max_steps: int | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a hard-budget MF-KG experiment, including every warmup observation.

    ``observation_noise`` is a numerical nugget for the deterministic prefix
    objective, not an independent noise model for overlapping sample means.
    Correlation between those means is modeled by the GP fidelity kernel.
    By default, adaptive_fidelity can choose any higher affordable grid level
    for each unfinished arm. If sample_counts is omitted, DEFAULT_FIDELITIES
    defines the grid. Explicit fixed_batch allocation offers one next-prefix
    query per unfinished arm, adding batch_size items up to arm/budget limits.
    """
    if not math.isfinite(eval_budget_fraction) or not 0 < eval_budget_fraction <= 1:
        raise ValueError("eval_budget_fraction must be in (0, 1]")
    if not math.isfinite(warmup_budget_fraction) or not 0 < warmup_budget_fraction <= 1:
        raise ValueError("warmup_budget_fraction must be in (0, 1]")
    if n_init is not None and n_init < 1:
        raise ValueError("n_init must be positive")
    if warmup_anchors < 0 or warmup_strategy not in {"coverage", "random"}:
        raise ValueError("invalid warmup settings")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    for name, value in (("num_fantasies", num_fantasies), ("fit_maxiter", fit_maxiter), ("candidate_batch_size", candidate_batch_size), ("target_batch_size", target_batch_size)):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(observation_noise) or observation_noise <= 0:
        raise ValueError("observation_noise must be positive and finite")
    if allocation_mode not in {"fixed_batch", "adaptive_fidelity"}:
        raise ValueError("allocation_mode must be fixed_batch or adaptive_fidelity")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    n_arms, n_examples = data.matrix.shape
    if allocation_mode == "fixed_batch":
        if sample_counts is not None:
            raise ValueError("sample_counts requires allocation_mode='adaptive_fidelity'; fixed_batch uses batch_size")
        levels = np.unique(np.append(np.arange(batch_size, n_examples + 1, batch_size), n_examples))
    else:
        if sample_counts is None:
            sample_counts = fidelity_counts(n_examples, DEFAULT_FIDELITIES)
        levels = np.asarray(sample_counts, dtype=np.int64)
        if levels.ndim != 1 or not len(levels) or (levels < 1).any() or (levels > n_examples).any():
            raise ValueError("sample_counts must be nonempty counts in [1, n_examples]")
        levels = np.unique(np.append(levels, n_examples))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Prefix selection is independent of scores and fixed before any observation.
    permutations = np.stack([rng.permutation(n_examples) for _ in range(n_arms)])
    counts = np.zeros(n_arms, dtype=np.int64)
    observed_sums = np.zeros(n_arms)
    features = data.features
    feature_np = features.numpy()
    target_X = torch.cat([features, torch.ones((n_arms, 1), dtype=features.dtype)], dim=-1)
    action_unit_cost = data.per_cell_cost if cost_aware else np.ones(n_arms)
    full_budget = float(n_examples * action_unit_cost.sum())
    budget = eval_budget_fraction * full_budget if cost_aware else float(max(1, math.floor(eval_budget_fraction * full_budget)))
    warmup_cap = warmup_budget_fraction * budget
    if levels[0] * action_unit_cost.min() > warmup_cap + 1e-9:
        raise ValueError("warmup cap cannot afford one lowest-fidelity observation; increase --warmup-budget-fraction or use smaller --fidelity-samples")
    if n_init is None:
        n_init = min(n_arms, 8) if data.feature_source == "arm_identity" else max(len(np.unique(feature_np[:, j])) for j in range(feature_np.shape[1]))
    n_init = min(n_init, n_arms)
    train_X: list[torch.Tensor] = []
    train_Y: list[float] = []
    true_means = data.matrix.mean(axis=1)  # Metrics only; never passed to the GP.
    history: dict[str, list[Any]] = {k: [] for k in (
        "x", "x_original_cost", "regret", "recommended_arm", "recommended_mean",
        "pulled_arm", "fidelity", "n_observed_arm", "n_new_samples", "observed_y",
        "acquisition_value", "selection_phase", "iter_fit_s", "iter_score_s",
    )}
    spent, evaluated, original_cost = 0.0, 0, 0.0
    last_fit_s, last_score_s = 0.0, 0.0

    def fit_model():
        nonlocal last_fit_s
        started = time.perf_counter()
        result = fit_mfkg_model(torch.stack(train_X), torch.tensor(train_Y, dtype=features.dtype).unsqueeze(-1), cat_dims=list(range(features.shape[1])), observation_noise=observation_noise, fit_maxiter=fit_maxiter)
        last_fit_s = time.perf_counter() - started
        return result

    def reveal(arm: int, count: int, phase: str, score: float = float("nan")) -> Any:
        nonlocal spent, evaluated, original_cost
        old = int(counts[arm])
        additional = count - old
        if additional <= 0:
            raise RuntimeError("MF-KG must reveal at least one new cell")
        if allocation_mode == "fixed_batch" and additional > batch_size:
            raise RuntimeError("fixed-batch MF-KG cannot reveal more than batch_size new cells per query")
        observed_sums[arm] += data.matrix[arm, permutations[arm, old:count]].sum()
        counts[arm] = count
        evaluated += additional
        original_cost += additional * data.per_cell_cost[arm]
        spent += additional * action_unit_cost[arm]
        if spent > budget + 1e-8 * max(1, budget):
            raise RuntimeError("MF-KG exceeded its evaluation budget")
        prefix_mean = float(observed_sums[arm] / count)
        train_X.append(torch.cat([features[arm], features.new_tensor([count / n_examples])]))
        train_Y.append(prefix_mean)
        # Exact target observations are already known when an arm is completed.
        model = fit_model()
        means = target_posterior_means(model, target_X, target_batch_size)
        complete = np.flatnonzero(counts == n_examples)
        means[complete] = means.new_tensor(observed_sums[complete] / n_examples)
        rec = int(means.argmax())
        values = dict(x=evaluated, x_original_cost=original_cost,
            regret=float(true_means.max() - true_means[rec]), recommended_arm=rec,
            recommended_mean=float(means[rec]), pulled_arm=arm, fidelity=count / n_examples,
            n_observed_arm=count, n_new_samples=additional, observed_y=prefix_mean,
            acquisition_value=score, selection_phase=phase, iter_fit_s=last_fit_s,
            iter_score_s=last_score_s)
        for key, value in values.items():
            history[key].append(value)
        if on_step is not None:
            on_step(values)
        return model

    selected: list[int] = []
    model = None
    # Every initialization observation costs only the lowest-fidelity prefix.
    for _ in range(n_init):
        available = np.flatnonzero((counts == 0) & (levels[0] * action_unit_cost <= warmup_cap - spent + 1e-9))
        if not len(available):
            break
        arm = select_coverage_arm(feature_np, available, selected, rng) if warmup_strategy == "coverage" else int(rng.choice(available))
        selected.append(arm)
        model = reveal(arm, int(levels[0]), "random_init")
    # Choose anchors independently of scores. In fixed mode, split promotions
    # into B-item queries so warmup also respects the per-round item count.
    anchor_order = rng.permutation(selected)[:warmup_anchors]
    for arm in anchor_order:
        if allocation_mode == "fixed_batch":
            affordable_count = int(math.floor((warmup_cap - spent) / action_unit_cost[arm] + 1e-9))
            anchor_target = min(n_examples, int(counts[arm]) + max(0, affordable_count))
            while counts[arm] < anchor_target:
                next_count = min(int(counts[arm]) + batch_size, anchor_target)
                model = reveal(int(arm), next_count, "random_init_anchor")
        else:
            affordable = levels[(levels > counts[arm]) & ((levels - counts[arm]) * action_unit_cost[arm] <= warmup_cap - spent + 1e-9)]
            if len(affordable):
                model = reveal(int(arm), int(affordable[-1]), "random_init_anchor")
    warmup_spent = spent
    warmup_steps = len(history["x"])
    if model is None:
        raise RuntimeError("no warmup observations were collected")
    stop_reason = "no_affordable_action"
    steps = 0
    while max_steps is None or steps < max_steps:
        if allocation_mode == "fixed_batch":
            arm_ids = np.flatnonzero(counts < n_examples)
            additional = np.minimum(batch_size, n_examples - counts[arm_ids])
            budget_cells = np.floor((budget - spent) / action_unit_cost[arm_ids] + 1e-9).astype(np.int64)
            additional = np.minimum(additional, np.maximum(0, budget_cells))
            feasible = additional > 0
            arm_ids, additional = arm_ids[feasible], additional[feasible]
            new_counts = counts[arm_ids] + additional
            marginal = additional * action_unit_cost[arm_ids]
        else:
            arm_ids, target_counts = np.where(levels[None, :] > counts[:, None])
            new_counts = levels[target_counts]
            marginal = (new_counts - counts[arm_ids]) * action_unit_cost[arm_ids]
            affordable = marginal <= budget - spent + 1e-9
            arm_ids, new_counts, marginal = arm_ids[affordable], new_counts[affordable], marginal[affordable]
        if not len(arm_ids):
            break
        candidate_X = torch.cat([features[arm_ids], features.new_tensor(new_counts / n_examples).unsqueeze(-1)], dim=-1)
        started = time.perf_counter()
        scores = score_mfkg_candidates(model, candidate_X, target_X, features.new_tensor(marginal), num_fantasies=num_fantasies, seed=seed + steps + 1, candidate_batch_size=candidate_batch_size, target_batch_size=target_batch_size)
        last_score_s = time.perf_counter() - started
        choice = int(scores.argmax())
        model = reveal(int(arm_ids[choice]), int(new_counts[choice]), "mfkg", float(scores[choice]))
        steps += 1
    else:
        stop_reason = "max_steps"
    return {**history, "budget": budget, "budget_spent": spent, "warmup_budget": warmup_cap,
        "warmup_spent": warmup_spent, "warmup_steps": warmup_steps,
        "n_init_requested": n_init, "n_init_actual": len(selected), "stop_reason": stop_reason,
        "sample_counts": levels, "allocation_mode": allocation_mode, "batch_size": batch_size if allocation_mode == "fixed_batch" else -1,
        "q": 1, "final_counts": counts, "permutations": permutations,
        "train_X": torch.stack(train_X).numpy(), "train_Y": np.asarray(train_Y)[:, None]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--bo-inputs", "--bo_inputs", type=Path, help="Optional archive providing categorical X, arm_ids, and full-evaluation costs; Y is ignored.")
    p.add_argument("--out", type=Path, help="Output trace .npz; default outputs/mfkg/{matrix}__{variant}__{allocation}__seed{seed}.npz; metadata saved beside it.")
    p.add_argument("--experiment-variant", "--experiment_variant", choices=["mfkg_unit", "mfkg_cost"], default=None)
    p.add_argument("--seed", "--run-seed", "--run_seed", type=int, default=0)
    p.add_argument("--eval-budget-fraction", "--eval_budget_fraction", type=float, default=0.1)
    p.add_argument("--cost-aware", "--cost_aware", action="store_true")
    p.add_argument("--cost-vector", "--cost_vector", type=Path, help="Per-cell costs; supplying this also enables cost-aware allocation and stopping.")
    p.add_argument("--allocation-mode", "--allocation_mode", choices=["fixed_batch", "adaptive_fidelity"], default="adaptive_fidelity",
                   help="Default adaptive_fidelity: jointly choose one configuration and any higher affordable fidelity (q=1). fixed_batch is the optional earlier protocol.")
    p.add_argument("--batch-size", "--batch_size", type=int, default=16,
                   help="Only for explicit fixed_batch mode: new benchmark items per query, up to arm or budget limits. Not used by the default adaptive_fidelity mode.")
    fidelity = p.add_mutually_exclusive_group()
    fidelity.add_argument("--fidelities", type=float, nargs="+", help="Fidelity fractions for adaptive_fidelity mode only.")
    fidelity.add_argument("--fidelity-samples", "--fidelity_samples", type=int, nargs="+", help="Absolute counts for adaptive_fidelity mode; full fidelity is always added.")
    p.add_argument("--n-init", "--n_init", type=int, default=None, help="Default: largest categorical level count (or 8 with arm identity features), limited by warmup cost cap.")
    p.add_argument("--warmup-budget-fraction", "--warmup_budget_fraction", type=float, default=0.1, help="Maximum fraction of the total experiment budget spent on initialization.")
    p.add_argument("--warmup-anchors", "--warmup_anchors", type=int, default=1)
    p.add_argument("--warmup-strategy", "--warmup_strategy", choices=["coverage", "random"], default="coverage")
    p.add_argument("--num-fantasies", "--num_fantasies", type=int, default=64)
    p.add_argument("--observation-noise", "--observation_noise", type=float, default=1e-6, help="Numerical variance nugget for the deterministic prefix objective.")
    p.add_argument("--fit-maxiter", "--fit_maxiter", type=int, default=200)
    p.add_argument("--candidate-batch-size", "--candidate_batch_size", type=int, default=8)
    p.add_argument("--target-batch-size", "--target_batch_size", type=int, default=128)
    p.add_argument("--max-steps", "--max_steps", type=int, default=None, help="Optional acquisition-step cap after warmup, for smoke tests.")
    p.add_argument("--wandb-mode", "--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb-project", "--wandb_project", default="GittinsBanditEval")
    p.add_argument("--wandb-entity", "--wandb_entity", default=None)
    p.add_argument("--wandb-group", "--wandb_group", default=None)
    p.add_argument("--dataset-tag", "--dataset_tag", default=None)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if args.allocation_mode == "fixed_batch" and (args.fidelities is not None or args.fidelity_samples is not None):
        p.error("--fidelities/--fidelity-samples require --allocation-mode adaptive_fidelity; fixed_batch uses --batch-size")
    return args


def main() -> int:
    args = parse_args()
    wall_start = time.perf_counter()
    data = load_data(args.matrix, args.bo_inputs, args.cost_vector)
    cost_aware = args.cost_aware or args.cost_vector is not None or args.experiment_variant == "mfkg_cost"
    if args.experiment_variant == "mfkg_unit" and cost_aware:
        raise ValueError("mfkg_unit conflicts with --cost-aware or --cost-vector")
    if cost_aware and args.bo_inputs is None and args.cost_vector is None:
        raise ValueError("--cost-aware requires --bo-inputs or --cost-vector")
    allocation = f"fixed_B{args.batch_size}" if args.allocation_mode == "fixed_batch" else "adaptive_fidelity"
    if args.out is None:
        variant = "mfkg_cost" if cost_aware else "mfkg_unit"
        args.out = REPO_ROOT / "outputs" / "mfkg" / f"{args.matrix.stem}__{variant}__{allocation}__seed{args.seed}.npz"
    if args.out.suffix.lower() != ".npz":
        raise ValueError("--out must end in .npz")
    if args.allocation_mode == "fixed_batch":
        n_examples = data.matrix.shape[1]
        levels = np.unique(np.append(np.arange(args.batch_size, n_examples + 1, args.batch_size), n_examples))
    else:
        levels = fidelity_counts(data.matrix.shape[1], args.fidelities or DEFAULT_FIDELITIES, args.fidelity_samples)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_tag": args.dataset_tag or data.dataset, "policy_family": "mfkg", "acquisition": "mfkg",
        "run_seed": args.seed, "cost_mode": "cost" if cost_aware else "unit",
        "experiment_variant": "mfkg_cost" if cost_aware else "mfkg_unit",
        "cost_aware": cost_aware, "feature_source": data.feature_source,
        "allocation_mode": args.allocation_mode, "batch_size": args.batch_size if args.allocation_mode == "fixed_batch" else None, "q": 1,
        "fidelity_samples": levels.tolist(), "target_fidelity": 1.0,
        "recommendation_rule": "full_fidelity_posterior_mean",
        "observation_model": "deterministic_nested_prefix_gp_with_numerical_nugget",
        "n_arms": data.matrix.shape[0], "n_examples": data.matrix.shape[1]}
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
            mode=args.wandb_mode, name=f"{args.matrix.stem}_{config['experiment_variant']}_{allocation}_seed{args.seed}", config=config)
        run.define_metric("cum_eval")
        run.define_metric("simple_regret", step_metric="cum_eval")

    def on_step(values):
        if run is not None:
            run.log({**values, "cum_eval": values["x"], "cum_original_cost": values["x_original_cost"], "simple_regret": values["regret"], "bo_phase": values["selection_phase"]})

    try:
        result = run_mfkg(data, seed=args.seed, eval_budget_fraction=args.eval_budget_fraction,
            cost_aware=cost_aware, sample_counts=levels if args.allocation_mode == "adaptive_fidelity" else None,
            allocation_mode=args.allocation_mode, batch_size=args.batch_size, n_init=args.n_init,
            warmup_budget_fraction=args.warmup_budget_fraction, warmup_anchors=args.warmup_anchors,
            warmup_strategy=args.warmup_strategy, num_fantasies=args.num_fantasies,
            observation_noise=args.observation_noise, fit_maxiter=args.fit_maxiter,
            candidate_batch_size=args.candidate_batch_size, target_batch_size=args.target_batch_size,
            max_steps=args.max_steps, on_step=on_step)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **result, **{k: v for k, v in config.items() if v is not None and k not in result},
            mfkg_x=np.asarray(result["x"]), mfkg_x_original_cost=np.asarray(result["x_original_cost"]),
            mfkg_regret=np.asarray(result["regret"]), mfkg_recommended_arm=np.asarray(result["recommended_arm"]),
            mfkg_recommended_mean=np.asarray(result["recommended_mean"]), cost_per_arm_original=data.per_cell_cost)
        summary = {"final_simple_regret": result["regret"][-1], "final_cum_eval": result["x"][-1],
            "final_cum_original_cost": result["x_original_cost"][-1], "budget_spent": result["budget_spent"],
            "budget": result["budget"], "warmup_spent": result["warmup_spent"], "warmup_budget": result["warmup_budget"],
            "n_init_actual": result["n_init_actual"], "stop_reason": result["stop_reason"],
            "num_batches": len(result["x"]), "total_wall_time_s": time.perf_counter() - wall_start}
        args.out.with_suffix(".json").write_text(json.dumps({**config, **summary}, indent=2), encoding="utf-8")
        if run is not None:
            run.summary.update(summary)
        print(f"Wrote {args.out}\n{json.dumps(summary, indent=2)}")
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
