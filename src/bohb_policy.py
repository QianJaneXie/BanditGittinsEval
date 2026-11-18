"""Author BOHB components on finite categorical, nested-prefix objectives.

HpBandSter supplies both its KDE configuration generator and Hyperband/SH
scheduler. This synchronous driver replaces only distributed worker dispatch.
The original defaults, promotion decisions, and duplicate proposals are kept.
Matrix entries are cached, so repeated/promotion jobs pay only for new cells.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
import inspect
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BOHBData:
    matrix: np.ndarray
    features: np.ndarray
    per_cell_cost: np.ndarray
    feature_names: tuple[str, ...]
    dataset: str = "custom"


def load_data(matrix_path: Path, bo_inputs: Path) -> BOHBData:
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a nonempty finite two-dimensional array")
    K, N = matrix.shape
    with np.load(bo_inputs, allow_pickle=False) as archive:
        # Y and raw_scores_mean are deliberately never read.
        ids = np.asarray(archive["arm_ids"])
        if ids.shape != (K,) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("arm_ids must be an integer permutation of matrix rows")
        if not np.array_equal(np.sort(ids), np.arange(K)):
            raise ValueError("arm_ids must be an integer permutation of matrix rows")
        order = np.argsort(ids)
        X = np.asarray(archive["X"], dtype=float)
        cost = np.asarray(archive["cost"], dtype=float)
        metadata = json.loads(str(archive["metadata_json"].item()))
        if X.ndim != 2 or X.shape[0] != K or X.shape[1] < 1:
            raise ValueError("X must have one feature vector per matrix row")
        if cost.shape != (K,) or int(metadata.get("n_examples", N)) != N:
            raise ValueError("BO costs or example count do not match the matrix")
        names = tuple(metadata.get("feature_names", [f"feature_{j}" for j in range(X.shape[1])]))
        dataset = str(metadata.get("dataset", "custom"))
    if len(names) != X.shape[1] or not np.isfinite(X).all() or len(np.unique(X, axis=0)) != K:
        raise ValueError("features must be finite and uniquely identify each configuration")
    if not np.isfinite(cost).all() or (cost <= 0).any():
        raise ValueError("costs must be positive and finite")
    return BOHBData(matrix, X[order], cost[order] / N, names, dataset)


def default_parameters() -> dict:
    from hpbandster.optimizers.bohb import BOHB
    names = ("eta", "min_budget", "max_budget", "min_points_in_model", "top_n_percent",
             "num_samples", "random_fraction", "bandwidth_factor", "min_bandwidth")
    signature = inspect.signature(BOHB.__init__)
    return {name: signature.parameters[name].default for name in names}


def fidelity_schedule(min_fidelity: float = 0.01, eta: float = 3) -> np.ndarray:
    if not math.isfinite(min_fidelity) or not 0 < min_fidelity < 1:
        raise ValueError("min_fidelity must lie strictly between zero and one")
    if not math.isfinite(eta) or eta < 2:
        raise ValueError("eta must be finite and at least two")
    # Exactly the author's schedule: min_budget is a lower bound, not
    # necessarily a realized rung. With defaults the first rung is 1/81.
    n = -int(np.log(min_fidelity) / np.log(eta)) + 1
    return np.power(eta, -np.linspace(n - 1, 0, n))


def make_configspace(features: np.ndarray, seed: int):
    import ConfigSpace as CS
    X = np.asarray(features)
    if X.ndim != 2 or X.shape[0] < 2 or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("BOHB requires at least two finite categorical configurations")
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError("configuration features must be unique")
    active = [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) > 1]
    encoded = np.column_stack([np.unique(X[:, j], return_inverse=True)[1] for j in active])
    cs = CS.ConfigurationSpace(seed=seed)
    hps = [CS.CategoricalHyperparameter(f"feature_{j:03d}", choices=list(range(int(encoded[:, i].max()) + 1)),
                                       default_value=int(encoded[0, i]))
           for i, j in enumerate(active)]
    cs.add(hps)
    lookup = {tuple(row): arm for arm, row in enumerate(encoded)}
    # GSM8K/PIQA are sparse grids. Forbid combinations absent from the oracle;
    # numeric category IDs carry no metric or ordering information.
    forbidden = []
    for combination in itertools.product(*(hp.choices for hp in hps)):
        if combination not in lookup:
            clauses = [CS.ForbiddenEqualsClause(hp, value) for hp, value in zip(hps, combination)]
            forbidden.append(CS.ForbiddenAndConjunction(*clauses) if len(clauses) > 1 else clauses[0])
    if forbidden:
        cs.add(forbidden)
    return cs, hps, lookup, active


@contextmanager
def _legacy_random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def run_bohb(data: BOHBData, *, seed: int = 0, eval_budget_fraction: float = 0.1,
             cost_budgeted: bool = False, min_fidelity: float = 0.01,
             eta: float = 3, max_jobs: int | None = None,
             on_step: Callable | None = None) -> dict:
    """Run native BOHB sequentially, stopping before an unaffordable job.

    Costs change accounting/termination only, never loss, KDE selection or SH
    rankings. An unaffordable requested job is not resized or skipped. This
    preserves native allocation and can leave an unused budget remainder.
    """
    from hpbandster.optimizers.bohb import BOHB as NativeBOHB
    from hpbandster.optimizers.config_generators.bohb import BOHB as Generator
    if not math.isfinite(eval_budget_fraction) or not 0 < eval_budget_fraction <= 1:
        raise ValueError("eval_budget_fraction must be in (0, 1]")
    if max_jobs is not None and (isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1):
        raise ValueError("max_jobs must be a positive integer")
    K, N = data.matrix.shape
    if data.per_cell_cost.shape != (K,) or not np.isfinite(data.per_cell_cost).all() or (data.per_cell_cost <= 0).any():
        raise ValueError("per-cell costs must be finite and positive")
    levels = fidelity_schedule(min_fidelity, eta)
    sample_counts = np.ceil(levels * N - 1e-12).clip(1, N).astype(int)
    cs, hps, lookup, active = make_configspace(data.features, seed)
    rng = np.random.default_rng(seed)
    # Match the MF-KG oracle's seeded per-arm example orders, independent of
    # the BOHB proposal stream and of all response values.
    permutations = np.stack([rng.permutation(N) for _ in range(K)])
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K)
    action_cost = data.per_cell_cost if cost_budgeted else np.ones(K)
    budget = float(eval_budget_fraction * N * action_cost.sum())
    if not cost_budgeted:
        budget = float(math.floor(budget))
    truth = data.matrix.mean(axis=1)  # Reporting only; not used in decisions.
    fields = ("x", "x_original_cost", "regret", "recommended_arm", "recommended_mean",
              "pulled_arm", "fidelity", "n_requested_samples", "n_observed_arm",
              "n_new_samples", "observed_y", "bracket", "rung", "model_based_pick",
              "largest_model_fidelity", "incumbent_fidelity", "budget_spent")
    history = {name: [] for name in fields}
    spent = original_cost = 0.0
    rec, rec_mean, rec_fidelity = -1, -math.inf, -math.inf
    full_rec, full_mean = -1, -math.inf
    bracket = jobs = 0
    stop_reason = "next_job_unaffordable"
    next_job_cost = 0.0
    with _legacy_random_seed(seed):
        generator = Generator(configspace=cs)
        provider = SimpleNamespace(max_SH_iter=len(levels), eta=eta, budgets=levels,
                                   config_generator=generator)
        done = False
        while not done:
            iteration = NativeBOHB.get_next_iteration(provider, bracket)
            while not iteration.is_finished:
                if max_jobs is not None and jobs >= max_jobs:
                    stop_reason, done = "max_jobs", True
                    break
                if np.all(counts == N):
                    stop_reason, done = "all_cells_observed", True
                    break
                requested = iteration.get_next_run()
                if requested is None:
                    if not iteration.is_finished:
                        raise RuntimeError("native scheduler stalled with no pending workers")
                    break
                config_id, config, fidelity = requested
                arm = lookup[tuple(int(config[hp.name]) for hp in hps)]
                target = int(np.ceil(fidelity * N - 1e-12).clip(1, N))
                old = int(counts[arm])
                added = max(0, target - old)
                marginal = float(added * action_cost[arm])
                if marginal > budget - spent + 1e-9 * max(1.0, budget):
                    next_job_cost, done = marginal, True
                    break
                if added:
                    sums[arm] += data.matrix[arm, permutations[arm, old:target]].sum()
                    counts[arm] = target
                # For a repeated lower-fidelity request, only already observed
                # entries from that shorter prefix are returned.
                mean = float(data.matrix[arm, permutations[arm, :target]].mean())
                spent += marginal
                original_cost += added * data.per_cell_cost[arm]
                loss = 1.0 - mean
                job = SimpleNamespace(id=config_id, kwargs={"config": config, "budget": fidelity},
                                      timestamps={"submitted": jobs, "started": jobs, "finished": jobs + 1},
                                      result={"loss": loss, "info": {}}, exception=None)
                iteration.register_result(job)
                generator.new_result(job)
                # Native get_incumbent_trajectory defaults: a larger nominal
                # fidelity wins; at equal fidelity require strictly better loss.
                if fidelity > rec_fidelity or (fidelity == rec_fidelity and mean > rec_mean):
                    rec, rec_mean, rec_fidelity = arm, mean, fidelity
                if fidelity == 1 and mean > full_mean:
                    full_rec, full_mean = arm, mean
                info = iteration.data[config_id].config_info
                row = dict(x=int(counts.sum()), x_original_cost=float(original_cost),
                           regret=float(truth.max() - truth[rec]), recommended_arm=rec,
                           recommended_mean=rec_mean, pulled_arm=arm, fidelity=float(fidelity),
                           n_requested_samples=target, n_observed_arm=int(counts[arm]),
                           n_new_samples=added, observed_y=mean, bracket=bracket,
                           rung=int(np.argmin(abs(levels - fidelity))),
                           model_based_pick=bool(info.get("model_based_pick", False)),
                           largest_model_fidelity=max(0.0, float(generator.largest_budget_with_model())),
                           incumbent_fidelity=float(rec_fidelity), budget_spent=float(spent))
                for name in fields:
                    history[name].append(row[name])
                if on_step is not None:
                    on_step(row)
                jobs += 1
            bracket += 1
    return {**{k: np.asarray(v) for k, v in history.items()}, "budget": budget,
            "budget_spent_total": spent, "unused_budget": max(0.0, budget - spent),
            "next_job_cost": next_job_cost, "stop_reason": stop_reason,
            "final_counts": counts, "final_observed_sums": sums, "permutations": permutations,
            "fidelity_levels": levels, "fidelity_samples": sample_counts,
            "full_fidelity_recommended_arm": full_rec, "active_feature_columns": np.asarray(active),
            "recommendation_rule": "hpbandster_default_incumbent_trajectory",
            "allocation_cost_aware": False,
            "hpbandster_version": version("hpbandster"), "configspace_version": version("ConfigSpace"),
            "statsmodels_version": version("statsmodels"), "numpy_version": version("numpy")}
