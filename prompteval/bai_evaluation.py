import argparse
import json
import pickle
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning

try:
    from .methods import LogReg, StratSample
except ImportError:
    from methods import LogReg, StratSample

try:
    from .methods_gpu import LogRegTorch
except ImportError:
    try:
        from methods_gpu import LogRegTorch
    except ImportError:
        LogRegTorch = None  # type: ignore[misc, assignment]

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# BAI budgets are derived per-Y in evaluate_bai (see budgets_from_y), not fixed counts.
# Paper-style fixed list kept for reference / other scripts:
LEGACY_BUDGETS = [200, 400, 800, 1600]
# Single BAI budget: this fraction of |Y| = n_formats * n_examples (rounded, at least 1 observation).
OBSERVATION_BUDGET_MAX_FRACTION = 0.1
# Cap successive-halving rounds. With a 10% budget, survivors typically exhaust their
# examples by ~round 5 (ceil(log2(K)) may be larger, e.g. 7 for K~100), so later
# rounds produce no new fits.
MAX_BAI_PHASES = 5
# PromptEval paper, section 6.1: fit the binary correctness model after thresholding
# AlpacaEval 2.0 instance scores at 1/2, while retaining raw scores at evaluation time.
ALPACA_FIT_BINARIZE_THRESHOLD = 0.5


def budgets_from_y(Y: np.ndarray, max_fraction: float = OBSERVATION_BUDGET_MAX_FRACTION) -> List[int]:
    """
    One total observation budget for BAI: ``max_fraction * n_formats * n_examples`` (rounded).

    The budget always counts observations (each sample costs 1 for sampling purposes);
    per-arm costs are only *recorded* alongside regrets, never used to drive sampling.
    """
    n_cells = int(Y.shape[0]) * int(Y.shape[1])
    cap = max(1, int(round(float(max_fraction) * n_cells)))
    return [cap]


# Default per-benchmark cost files (arm index -> estimated_cost_per_1m_input_tokens).
DEFAULT_COST_FILES = {
    "MMLU": "data_analysis/pricing/mmlu_prompt_eval_configurations_input_price.json",
    "GSM8K": "data_analysis/pricing/gsm8k_various_models_configurations_price_ratio_1to2_rounded.json",
    "PIQA": "data_analysis/pricing/piqa_various_models_configurations_input_price.json",
    "ALPACA": "data_analysis/pricing/alpaca_153_models_no_rounding_debias_price_1to8.json",
}


def load_arm_costs(cost_file: str) -> np.ndarray:
    """
    Load a per-arm cost vector from a pricing-configurations JSON.

    The JSON is keyed by the arm/configuration index ("0", "1", ...) in matrix row order
    (for MMLU combined: ``model_idx * n_prompts + prompt_idx``); the cost of one observation
    is ``estimated_cost_per_1m_input_tokens``.
    """
    with open(cost_file, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    idx = sorted((key for key in cfg if str(key).isdigit()), key=int)
    if not idx:
        raise ValueError(f"{cost_file}: no numeric arm keys found.")
    if [int(k) for k in idx] != list(range(len(idx))):
        raise ValueError(f"{cost_file}: configuration keys are not contiguous 0..{len(idx) - 1}.")
    return np.array([float(cfg[k]["estimated_cost_per_1m_input_tokens"]) for k in idx])


def slice_costs_for_llm(costs: Optional[np.ndarray], n_arms: int, llm: int) -> Optional[np.ndarray]:
    """
    Return the per-arm cost vector for one LLM's Y matrix (``n_arms`` rows).

    If ``costs`` already has ``n_arms`` entries (GSM8K/PIQA: rows are models), use it directly;
    if it covers ``n_llm * n_arms`` stacked arms (MMLU layout), take the ``llm``-th slice.
    """
    if costs is None:
        return None
    if costs.shape[0] == n_arms:
        return costs
    if costs.shape[0] % n_arms == 0:
        return costs[llm * n_arms : (llm + 1) * n_arms]
    raise ValueError(f"Cost vector length {costs.shape[0]} incompatible with {n_arms} arms.")

# Used by run_bai_evaluation() when no tasks= / tasks_csv / only_task / max_tasks / all_tasks / default_task_subset.
DEFAULT_TASK_SUBSET = ("abstract_algebra", "professional_law")

# BanditEval GSM8K/PIQA/ALPACA pickles from build_banditeval_pickle.py
# (no prompt-template covariates).
DEFAULT_BANDITEVAL_PICKLE_DIR = "prompteval/banditeval_pickle/"

# CLI: python prompteval/bai_evaluation.py --bench {MMLU,GSM8K,PIQA,ALPACA} [--tasks ...]
# (see parse_args / --help).

UPDATE_FIELDS = [
    "phase",
    "budget_obs",
    "budget_cost",
    "budget",  # legacy alias of budget_obs
    "chosen_arm",
    "chosen_mean",
    "oracle_mean",
    "simple_regret",
    "n_active",
]


def results_file_stem(bench: str, tag: str, task: Optional[str] = None) -> str:
    """Filename stem after ``bai_results_`` / ``bai_processed_results_`` (no extension)."""
    if task is not None:
        return f"{bench}_{task}{tag}"
    return f"{bench}{tag}"


def raw_results_tag(*, combine_models: bool) -> str:
    """Tag for the shared raw file (no unitcost/costaware — both spends are stored)."""
    return "_combined" if combine_models else ""


def processed_results_tag(
    *,
    cost_aware: bool,
    combine_models: bool,
    banditeval: bool,
) -> str:
    """
    Tag for a processed file.

    GSM8K/PIQA: ``_unitcost`` or ``_costaware``.
    MMLU: ``_combined`` (unit) or ``_costaware_combined``.
    """
    parts: List[str] = []
    if cost_aware:
        parts.append("_costaware")
    elif banditeval:
        parts.append("_unitcost")
    if combine_models:
        parts.append("_combined")
    return "".join(parts)


def summarize_wall_times(vals: Sequence[float]) -> Dict[str, Any]:
    """Summary stats for wall-clock times (seconds), matching bandit timing plots."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean_s": None,
            "median_s": None,
            "std_s": None,
            "se_s": None,
            "min_s": None,
            "max_s": None,
        }
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "std_s": std,
        "se_s": se,
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
    }


def save_bai_raw(
    path: str,
    *,
    out: list,
    jobs: list,
    tasks: Sequence[str],
    seeds: Sequence[int],
    bench: str,
    combine_models: bool,
    costs_recorded: bool,
    n_models_stacked: int,
    out_by_task: Optional[list] = None,
    cost_file: Optional[str] = None,
    total_wall_time_s: Optional[Sequence[float]] = None,
    fit_binarize_threshold: Optional[float] = None,
) -> None:
    payload: Dict[str, Any] = {
        "out": out,
        "jobs": jobs,
        "combine_models": combine_models,
        "n_models_stacked": n_models_stacked,
        "tasks": list(tasks),
        "seeds": list(seeds),
        "bench": bench,
        "costs_recorded": costs_recorded,
        "cost_file": cost_file,
        "update_fields": UPDATE_FIELDS,
        "fit_binarize_threshold": fit_binarize_threshold,
        "fit_target_transform": (
            "identity"
            if fit_binarize_threshold is None
            else f"indicator(score >= {float(fit_binarize_threshold):g})"
        ),
        "evaluation_target_transform": "raw",
        "note": (
            "Each phase update stores budget_obs (observation count) and budget_cost "
            "(cost-weighted spend when costs were loaded). Sampling is always unit-cost; "
            "processed unitcost/costaware files are derived from the same raw out. "
            "total_wall_time_s is aligned with out/jobs (one wall-clock seconds value per run)."
        ),
    }
    if out_by_task is not None:
        payload["out_by_task"] = out_by_task
    if total_wall_time_s is not None:
        wall = [float(x) for x in total_wall_time_s]
        payload["total_wall_time_s"] = wall
        payload["total_wall_time_stats"] = summarize_wall_times(wall)
    np.save(path, payload)


def save_bai_processed(
    path: str,
    *,
    curves: np.ndarray,
    tasks: Sequence[str],
    seeds: Sequence[int],
    bench: str,
    combine_models: bool,
    cost_aware: bool,
    n_models_stacked: int,
    raw_path: str,
    total_wall_time_s: Optional[Sequence[float]] = None,
    fit_binarize_threshold: Optional[float] = None,
) -> None:
    if cost_aware:
        note = (
            "cost-aware view of a shared BAI run: x-axis is budget_cost. "
            "At each cost on the shared grid, each seed contributes its last evaluated "
            "simple regret (hold-last / step). Same sampling as the unit-cost processed file "
            f"(raw: {raw_path})."
        )
        aggregation = "cost_grid_hold_last"
        spend_field = "budget_cost"
    else:
        note = (
            "unit-cost view of a shared BAI run: x-axis is budget_obs (observation counts). "
            "Curves average simple_regret and spend over seeds at each phase. "
            f"Same sampling as the cost-aware processed file when present (raw: {raw_path})."
        )
        aggregation = "phase_mean"
        spend_field = "budget_obs"
    payload: Dict[str, Any] = {
        "curves": curves,  # (n_tasks, n_budgets, n_blocks, 2, n_points)
        "channels": ["simple_regret", "budget"],
        "spend_field": spend_field,
        "tasks": list(tasks),
        "seeds": list(seeds),
        "bench": bench,
        "combine_models": combine_models,
        "cost_aware": cost_aware,
        "aggregation": aggregation,
        "n_models_stacked": n_models_stacked,
        "update_fields": UPDATE_FIELDS,
        "fit_binarize_threshold": fit_binarize_threshold,
        "fit_target_transform": (
            "identity"
            if fit_binarize_threshold is None
            else f"indicator(score >= {float(fit_binarize_threshold):g})"
        ),
        "evaluation_target_transform": "raw",
        "note": note,
    }
    if total_wall_time_s is not None:
        wall = [float(x) for x in total_wall_time_s]
        payload["total_wall_time_s"] = wall
        payload["total_wall_time_stats"] = summarize_wall_times(wall)
    np.save(path, payload)


def evaluate_bai_combined_one_seed(
    Y_cat,
    xs_cat,
    task: str,
    random_seed: int,
    backend,
    torch_device,
    torch_fit_log_interval: int = 0,
    fit_binarize_threshold: Optional[float] = None,
    costs=None,
) -> Tuple[list, float]:
    """One parallel worker job: BAI on an already-stacked (LLM×template) matrix for a single seed.

    Returns ``(eval_out, total_wall_time_s)`` where wall time covers the full ``evaluate_bai`` call.
    """
    print(f"[BAI] start task={task!r} seed={random_seed}", flush=True)
    wall_t0 = time.perf_counter()
    out = evaluate_bai(
        Y_cat,
        xs_cat,
        random_seed,
        backend=backend,
        torch_device=torch_device,
        torch_fit_log_interval=torch_fit_log_interval,
        fit_binarize_threshold=fit_binarize_threshold,
        costs=costs,
    )
    total_wall_time_s = float(time.perf_counter() - wall_t0)
    print(
        f"[BAI] done  task={task!r} seed={random_seed} total_wall_time_s={total_wall_time_s:.3f}",
        flush=True,
    )
    return out, total_wall_time_s


def evaluate_bai_combined_one_task(
    Ys,
    Xs,
    bench: str,
    task: str,
    random_seeds,
    backend,
    torch_device,
    torch_fit_log_interval: int = 0,
    fit_binarize_threshold: Optional[float] = None,
    costs=None,
):
    """Stack all models for one benchmark task, then run evaluate_bai for each seed (serial helper).

    Returns ``(eval_outs, total_wall_time_s_list)``.
    """
    Y_cat, xs_cat = combine_models_y_xs(Ys, Xs, bench, task)
    if costs is not None and costs.shape[0] != Y_cat.shape[0]:
        raise ValueError(
            f"{bench}/{task}: cost vector has {costs.shape[0]} entries but stacked Y has {Y_cat.shape[0]} arms."
        )
    outs: List = []
    walls: List[float] = []
    for random_seed in random_seeds:
        out, wt = evaluate_bai_combined_one_seed(
            Y_cat,
            xs_cat,
            task,
            random_seed,
            backend,
            torch_device,
            torch_fit_log_interval,
            fit_binarize_threshold,
            costs=costs,
        )
        outs.append(out)
        walls.append(wt)
    return outs, walls


def combine_models_y_xs(Ys, Xs, bench: str, task: str):
    """
    Vertically stack each model's Y and each covariate view in Xs (same layout as MMLU_matrix rows).

    If an Xs view has different column widths across models (e.g. odd vs even finetuned PCA), the
    narrower matrices are right-padded with zeros to the max width before vstack.

    Then each stacked view is augmented with **(n_models - 1)** dummy columns (reference = model 0),
    so each row encodes which LLM it came from alongside prompt features.

    Arms become (n_models * n_formats) rows; one BAI run per (task, seed) instead of per (task, llm, seed).
    """
    n_llm = len(Ys[bench][task])
    y_mats = [np.asarray(Ys[bench][task][m]) for m in range(n_llm)]
    y_shapes = {y.shape for y in y_mats}
    if len(y_shapes) != 1:
        raise ValueError(f"{bench}/{task}: Y shapes differ across models: {y_shapes}")
    Y_cat = np.vstack(y_mats)

    n_views = len(Xs[bench][task][0])
    xs_cat = []
    for vi in range(n_views):
        mats = [np.asarray(Xs[bench][task][m][vi], dtype=np.float64) for m in range(n_llm)]
        n_rows = {m.shape[0] for m in mats}
        if len(n_rows) != 1:
            raise ValueError(f"{bench}/{task}: Xs view {vi} row counts differ across models: {n_rows}")
        widths = [m.shape[1] for m in mats]
        if len(set(widths)) > 1:
            max_w = max(widths)
            padded = []
            for m in mats:
                if m.shape[1] < max_w:
                    padded.append(
                        np.pad(m, ((0, 0), (0, max_w - m.shape[1])), mode="constant", constant_values=0.0)
                    )
                else:
                    padded.append(m)
            warnings.warn(
                f"{bench}/{task}: Xs view {vi} column widths {widths} → right-padded with zeros to {max_w} for combine_models.",
                stacklevel=2,
            )
            mats = padded
        xs_cat.append(np.vstack(mats))
    if not xs_cat:
        raise ValueError(f"{bench}/{task}: no Xs views with consistent shapes across models for combine_models.")

    n_formats = y_mats[0].shape[0]
    n_total = n_llm * n_formats
    if Y_cat.shape[0] != n_total:
        raise ValueError(f"{bench}/{task}: stacked Y rows {Y_cat.shape[0]} != n_models * n_formats {n_total}.")
    if n_llm > 1:
        llm_oh = np.zeros((n_total, n_llm - 1), dtype=np.float64)
        for m in range(1, n_llm):
            lo, hi = m * n_formats, (m + 1) * n_formats
            llm_oh[lo:hi, m - 1] = 1.0
        xs_cat = [np.hstack([x, llm_oh]) for x in xs_cat]

    return Y_cat, xs_cat


### Functions
def compute_regrets(
    budget,
    Y,
    X,
    Z=None,
    random_seed=None,
    backend="sklearn",
    torch_device="auto",
    torch_fit_log_interval: int = 0,
    fit_binarize_threshold: Optional[float] = None,
    costs=None,
):
    """
    Computes the regret (best-format identification) using **logistic regression only**
    (``LogReg`` or ``LogRegTorch``) to score arms between elimination rounds.

    ``Z`` is accepted for backwards compatibility and ignored.

    Parameters:
    budget (int): The total budget for evaluation.
    Y (numpy.ndarray): Original labels or responses for each format-example pair. These values
        are always used for arm means and simple regret.
    X (numpy.ndarray): The format covariates. If not provided, an identity matrix is used.
    Z (numpy.ndarray, optional): Unused.
    random_seed (int, optional): The seed for random operations to ensure reproducibility.
    backend (str): "sklearn" (default) or "torch" for PyTorch logistic regression.
    torch_device (str): "auto", "cpu", or "cuda" when backend is "torch".
    torch_fit_log_interval (int): When ``backend=="torch"``, print training loss every this many
        epochs inside ``TorchLogisticRegression`` (0 = silent).
    fit_binarize_threshold (float, optional): If set, fit the logistic correctness model with
        ``1[Y >= threshold]`` while retaining original ``Y`` for evaluation. This is the
        PromptEval paper's AlpacaEval 2.0 adaptation (threshold 0.5).
    costs (numpy.ndarray, optional): Per-arm cost of one observation, used for **accounting
        only**. Sampling and the observation budget are identical with or without costs.
        When provided, each phase records both observation count and cost-weighted spend.

    Returns:
    list[dict]: One record per phase (length ``n_phases = min(ceil(log2(n_formats)), MAX_BAI_PHASES)``).
        Each record has:
        - ``phase`` (int)
        - ``budget_obs`` (float): cumulative observation count after this phase
        - ``budget_cost`` (float): cost-weighted cumulative spend (equals ``budget_obs`` if
          ``costs`` is None)
        - ``budget`` (float): legacy alias of ``budget_obs``
        - ``chosen_arm`` (int | None): arm index returned by logreg (None if no fit this phase)
        - ``chosen_mean`` (float | None): true mean reward of ``chosen_arm``
        - ``oracle_mean`` (float): best arm's true mean
        - ``simple_regret`` (float | None): ``oracle_mean - chosen_mean``
        - ``n_active`` (int): number of arms still active before elimination this phase
    """
    del Z  # unused; kept in signature for call-site compatibility
    use_torch = backend == "torch"
    if use_torch and LogRegTorch is None:
        raise ImportError("methods_gpu / torch backend unavailable (import failed).")

    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D matrix, got shape {Y.shape}.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains non-finite values.")
    if fit_binarize_threshold is None:
        Y_fit = Y
    else:
        threshold = float(fit_binarize_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"fit_binarize_threshold must be in [0, 1], got {threshold}.")
        y_min = float(np.min(Y))
        y_max = float(np.max(Y))
        if y_min < 0.0 or y_max > 1.0:
            raise ValueError(
                "Fit-time binarization requires bounded scores in [0, 1], "
                f"got min={y_min}, max={y_max}."
            )
        Y_fit = (Y >= threshold).astype(np.int8)

    n_formats, n_examples = Y.shape
    if costs is not None:
        costs = np.asarray(costs, dtype=float)
        if costs.shape[0] != n_formats:
            raise ValueError(f"costs has length {costs.shape[0]} but Y has {n_formats} arms.")
    n_phases = min(int(np.ceil(np.log2(n_formats))), MAX_BAI_PHASES)
    budget_phase = int(np.floor(budget / n_phases))
    extra_first_phase = budget - n_phases * budget_phase

    active_arms = list(range(n_formats))
    random_column = True
    seen_examples = np.zeros(Y.shape).astype(bool)
    oracle = float(Y.mean(-1).max())
    true_means = Y.mean(-1)
    phase_updates: List[Dict[str, Any]] = []

    for phase in range(n_phases):
        if phase == 0:
            seen_examples = StratSample(
                seen_examples, budget_phase + extra_first_phase, random_seed, active_arms, random_column
            )
        else:
            seen_examples = StratSample(
                seen_examples, seen_examples.sum() + budget_phase, random_seed, active_arms, random_column
            )

        budget_obs = float(seen_examples.sum())
        if costs is None:
            budget_cost = budget_obs
        else:
            budget_cost = float((seen_examples.sum(1) * costs).sum())

        update: Dict[str, Any] = {
            "phase": phase,
            "budget_obs": budget_obs,
            "budget_cost": budget_cost,
            "budget": budget_obs,  # legacy alias
            "chosen_arm": None,
            "chosen_mean": None,
            "oracle_mean": oracle,
            "simple_regret": None,
            "n_active": len(active_arms),
        }

        if seen_examples[active_arms].mean() == 1:
            # No unseen labels left on active arms: cannot run a new fit. Keep the
            # last identified arm's regret unchanged (do not clear it). If this is
            # the first phase and everything is already observed, fall back to the
            # empirical best among active arms.
            if phase_updates and phase_updates[-1].get("simple_regret") is not None:
                prev = phase_updates[-1]
                update["chosen_arm"] = prev["chosen_arm"]
                update["chosen_mean"] = prev["chosen_mean"]
                update["simple_regret"] = prev["simple_regret"]
            else:
                ba = int(active_arms[int(np.argmax(true_means[active_arms]))])
                update["chosen_arm"] = ba
                update["chosen_mean"] = float(true_means[ba])
                update["simple_regret"] = float(oracle - true_means[ba])
            phase_updates.append(update)
            break

        if use_torch:
            rasch_model = LogRegTorch(device=torch_device, fit_log_interval=torch_fit_log_interval)
        else:
            rasch_model = LogReg()
        rasch_model.fit(seen_examples, Y_fit, X)
        mu = np.asarray(rasch_model.thetas)[active_arms]
        best_local = int(np.argmax(mu))
        ba_logreg = int(active_arms[best_local])
        chosen_mean = float(true_means[ba_logreg])
        update["chosen_arm"] = ba_logreg
        update["chosen_mean"] = chosen_mean
        update["simple_regret"] = float(oracle - chosen_mean)
        phase_updates.append(update)

        n_active = len(active_arms)
        n_eliminate = int(np.ceil(n_active / 2))
        if n_active - n_eliminate <= 0:
            break
        active_arms = [active_arms[i] for i in np.argsort(mu)[n_eliminate:].tolist()]

    # Run stopped before using all successive-halving rounds (exhausted labels,
    # single arm left, etc.): later "phases" never reached a new evaluation budget,
    # so budget and simple regret stay at the last evaluated values.
    while len(phase_updates) < n_phases:
        last = phase_updates[-1]
        phase_updates.append(
            {
                "phase": len(phase_updates),
                "budget_obs": last["budget_obs"],
                "budget_cost": last["budget_cost"],
                "budget": last["budget"],
                "chosen_arm": last["chosen_arm"],
                "chosen_mean": last["chosen_mean"],
                "oracle_mean": last["oracle_mean"],
                "simple_regret": last["simple_regret"],
                "n_active": last["n_active"],
            }
        )

    return phase_updates


def updates_to_regret_spend(
    updates: List[Dict[str, Any]],
    spend_field: str = "budget_obs",
) -> np.ndarray:
    """Convert phase updates to ``(2, n_phases)``: simple regret, spend (obs or cost)."""
    n = len(updates)
    out = np.full((2, n), np.nan, dtype=float)
    for i, u in enumerate(updates):
        r = u.get("simple_regret")
        out[0, i] = float(r) if r is not None else np.nan
        b = u.get(spend_field)
        if b is None and spend_field != "budget":
            # Legacy raw files only stored ``budget`` (obs or cost, depending on the run).
            b = u.get("budget")
        if b is None or (isinstance(b, float) and np.isnan(b)):
            out[1, i] = np.nan
        else:
            out[1, i] = float(b)
    return out


def evaluate_bai_to_curves(eval_out: list, spend_field: str = "budget_obs") -> np.ndarray:
    """
    ``evaluate_bai`` output → ``(n_budgets, n_blocks, 2, n_phases)`` float array
    (channel 0 = simple regret, 1 = cumulative spend from ``spend_field``).
    """
    n_budgets = len(eval_out)
    n_blocks = len(eval_out[0]) if n_budgets else 0
    n_phases = len(eval_out[0][0]) if n_blocks else 0
    curves = np.full((n_budgets, n_blocks, 2, n_phases), np.nan, dtype=float)
    for bi in range(n_budgets):
        for bj in range(n_blocks):
            curves[bi, bj] = updates_to_regret_spend(eval_out[bi][bj], spend_field=spend_field)
    return curves


def aggregate_eval_outs(eval_outs: Sequence, *, cost_aware: bool) -> np.ndarray:
    """Seed-average evaluate_bai outputs using obs or cost spend on the x-axis."""
    spend_field = "budget_cost" if cost_aware else "budget_obs"
    return aggregate_curves(
        [evaluate_bai_to_curves(r, spend_field=spend_field) for r in eval_outs],
        cost_aware=cost_aware,
    )


# Cost-aware aggregation: if n_grid is None, use every distinct observed spend
# (≈ n_seeds × n_phases). Pass an int to force a uniform linspace of that length.
COST_AWARE_GRID_POINTS = None


def _step_regret_on_grid(spend: np.ndarray, regret: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Map a single run's phase evaluations onto ``grid`` as a right-constant step.

    Until the next evaluation, the chosen arm (and thus simple regret) is unchanged.
    That includes: (1) costs between two evaluations of this seed, and (2) costs
    beyond this seed's last evaluation when other seeds reached a higher spend —
    this seed still contributes its last regret (does not drop out as NaN).

    Points on ``grid`` before this seed's first evaluation are NaN.
    """
    order = np.argsort(spend, kind="mergesort")  # stable
    xs = spend[order]
    ys = regret[order]
    if xs.size > 1:
        # Identical spends (early-stop pads): keep the last copy.
        keep = np.concatenate([np.diff(xs) > 0, [True]])
        xs, ys = xs[keep], ys[keep]
    # idx[j] = last phase with xs[idx] <= grid[j]; -1 if grid[j] < xs[0]
    idx = np.searchsorted(xs, grid, side="right") - 1
    out = np.full(grid.shape, np.nan, dtype=float)
    valid = idx >= 0
    out[valid] = ys[idx[valid]]
    return out


def _collect_eval_spends(curves_list: Sequence[np.ndarray]) -> np.ndarray:
    """Finite spend values that accompany a finite regret, across all curves."""
    spends: List[np.ndarray] = []
    for c in curves_list:
        arr = np.asarray(c, dtype=float)
        # (n_budgets, n_blocks, 2, n_points)
        x = arr[:, :, 1, :]
        y = arr[:, :, 0, :]
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            spends.append(x[m])
    if not spends:
        return np.array([], dtype=float)
    return np.concatenate(spends)


def _regrid_curve(curve: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Hold-last remap of one ``(n_budgets, n_blocks, 2, n_points)`` curve onto ``grid``."""
    arr = np.asarray(curve, dtype=float)
    n_budgets, n_blocks = arr.shape[0], arr.shape[1]
    out = np.full((n_budgets, n_blocks, 2, grid.shape[0]), np.nan, dtype=float)
    for bi in range(n_budgets):
        for bl in range(n_blocks):
            x = arr[bi, bl, 1, :]
            y = arr[bi, bl, 0, :]
            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                continue
            out[bi, bl, 0, :] = _step_regret_on_grid(x[m], y[m], grid)
            out[bi, bl, 1, :] = grid
    return out


def stack_task_curves(curves_list: Sequence[np.ndarray], *, cost_aware: bool) -> np.ndarray:
    """
    Stack per-task curves into ``(n_tasks, n_budgets, n_blocks, 2, n_points)``.

    Cost-aware tasks can have different spend grids; align them onto the union of
    evaluation spends with hold-last before stacking.
    """
    if not curves_list:
        raise ValueError("stack_task_curves: empty curves_list.")
    arrays = [np.asarray(c, dtype=float) for c in curves_list]
    if not cost_aware:
        return np.stack(arrays, axis=0)
    shapes = {a.shape for a in arrays}
    if len(shapes) == 1:
        return np.stack(arrays, axis=0)
    spends = _collect_eval_spends(arrays)
    if spends.size == 0:
        raise ValueError("stack_task_curves: no finite evaluation spends to build a shared grid.")
    grid = np.unique(spends.astype(float))
    return np.stack([_regrid_curve(a, grid) for a in arrays], axis=0)


def aggregate_curves(
    curves_list: Sequence[np.ndarray],
    *,
    cost_aware: bool,
    n_grid: Optional[int] = COST_AWARE_GRID_POINTS,
) -> np.ndarray:
    """
    Combine several ``(n_budgets, n_blocks, 2, n_points)`` curves.

    - Unit cost: phases share the same observation counts → ``nanmean`` over the list
      (still one point per successive-halving phase). Early-stopped runs hold their
      last regret in later phase slots, so they stay in the average.
    - Cost-aware: x-grid = sorted unique evaluation spends across runs. At each cost,
      each run contributes its last evaluated simple regret (hold-last). A run that
      never reached a higher spend still contributes its last regret there.
      Curves may have different ``n_points`` (different spend grids per task/seed).
    """
    if not curves_list:
        raise ValueError("aggregate_curves: empty curves_list.")
    arrays = [np.asarray(c, dtype=float) for c in curves_list]

    if not cost_aware:
        stacked = np.stack(arrays, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            return np.nanmean(stacked, axis=0)

    spends = _collect_eval_spends(arrays)
    if spends.size == 0:
        # Fall back only if shapes already match.
        stacked = np.stack(arrays, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            return np.nanmean(stacked, axis=0)

    b_min = float(spends.min())
    b_max = float(spends.max())
    if b_min == b_max:
        grid = np.array([b_min], dtype=float)
    elif n_grid is None:
        grid = np.unique(spends.astype(float))
    else:
        grid = np.linspace(b_min, b_max, int(n_grid))

    n_budgets, n_blocks = arrays[0].shape[0], arrays[0].shape[1]
    out = np.full((n_budgets, n_blocks, 2, grid.shape[0]), np.nan, dtype=float)
    for bi in range(n_budgets):
        for bl in range(n_blocks):
            stepped = []
            for arr in arrays:
                x = arr[bi, bl, 1, :]
                y = arr[bi, bl, 0, :]
                m = np.isfinite(x) & np.isfinite(y)
                if not np.any(m):
                    continue
                stepped.append(_step_regret_on_grid(x[m], y[m], grid))
            if not stepped:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                out[bi, bl, 0, :] = np.nanmean(np.stack(stepped, axis=0), axis=0)
            out[bi, bl, 1, :] = grid
    return out


def evaluate_bai(
    Y,
    Xs,
    random_seed,
    backend="sklearn",
    torch_device="auto",
    torch_fit_log_interval: int = 0,
    fit_binarize_threshold: Optional[float] = None,
    costs=None,
):
    """
    Evaluates the Best Arm Identification (BAI) performance across multiple budgets and contexts.

    BAI is a process in finding the best 'arm' or decision option. This function applies BAI
    to different evaluation contexts and budgets using a specified set of inputs and comparison models.

    Budget is ``budgets_from_y(Y)``: a single value,
    ``OBSERVATION_BUDGET_MAX_FRACTION * Y.shape[0] * Y.shape[1]`` (rounded).

    Parameters:
    Y (numpy.ndarray): A matrix of true outcomes or labels for various test examples across different test formats.
    Xs (list of numpy.ndarray): A list of feature matrices representing the formats covariates used when fitting models.
    random_seed (int): An integer seed for ensuring deterministic behavior in randomized processes.
    torch_fit_log_interval (int): Passed to ``compute_regrets`` when using the torch backend (see there).
    fit_binarize_threshold (float, optional): Fit logistic models on ``1[Y >= threshold]``;
        retain raw ``Y`` for arm means/regret.
    costs (numpy.ndarray, optional): Per-arm costs, recorded per phase for accounting only
        (sampling is unaffected; see ``compute_regrets``).

    Returns:
    list: Nested ``[budget][block]`` → list of phase update dicts from ``compute_regrets``
          (each with ``budget``, ``chosen_arm``, ``simple_regret``, ...).
    """

    regrets = []

    ### running
    for budget in budgets_from_y(Y):
        regrets.append([])

        # n_formats = Y.shape[0]
        # X = np.eye(n_formats)[:,1:]
        regrets[-1].append(
            compute_regrets(
                budget,
                Y,
                None,
                None,
                random_seed,
                backend=backend,
                torch_device=torch_device,
                torch_fit_log_interval=torch_fit_log_interval,
                fit_binarize_threshold=fit_binarize_threshold,
                costs=costs,
            )
        )

        for X in Xs:
            regrets[-1].append(
                compute_regrets(
                    budget,
                    Y,
                    X,
                    None,
                    random_seed,
                    backend=backend,
                    torch_device=torch_device,
                    torch_fit_log_interval=torch_fit_log_interval,
                    fit_binarize_threshold=fit_binarize_threshold,
                    costs=costs,
                )
            )

    return regrets


def run_bai_evaluation(
    data_path: str = "prompteval/data/",
    results_path: str = "prompteval/results/",
    bench: str = "MMLU",
    random_seeds: int = 20,
    seed: Optional[int] = None,
    backend: str = "sklearn",
    torch_device: str = "auto",
    tasks: Optional[Sequence[str]] = None,
    tasks_csv: Optional[str] = None,
    only_task: Optional[str] = None,
    max_tasks: Optional[int] = None,
    all_tasks: bool = False,
    default_task_subset: Optional[Sequence[str]] = None,
    combine_models: bool = False,
    results_tag: str = "",
    torch_fit_log_interval: int = 0,
    cost_aware: bool = False,
    cost_file: Optional[str] = None,
    fit_binarize_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run the BAI evaluation pipeline (same behavior as the CLI).

    From another module, e.g.::

        from bai_evaluation import run_bai_evaluation
        run_bai_evaluation(bench="MMLU", random_seeds=5, tasks=["abstract_algebra"], combine_models=True)

    ``seed``: if set, run exactly one repeat with that RNG seed (ignores ``random_seeds``).
    Otherwise use seeds ``0 .. random_seeds-1``.

    Task selection priority: ``tasks`` → ``tasks_csv`` → ``only_task`` → ``max_tasks`` →
    ``all_tasks`` → ``default_task_subset`` (if not None) → module ``DEFAULT_TASK_SUBSET`` (if not None) → all tasks.

    ``torch_fit_log_interval``: if >0 and ``backend=="torch"``, prints logistic training loss every
    that many epochs inside each ``TorchLogisticRegression`` fit.

    ``record_costs`` / ``cost_file``: if a cost file is available (explicit ``cost_file`` or
    ``DEFAULT_COST_FILES[bench]``), per-arm costs are loaded and **both** observation spend
    and cost-weighted spend are recorded on every phase. Sampling is always unit-cost.
    One raw file is written; unit-cost and cost-aware **processed** files are derived from it.
    ``cost_aware`` is kept for API compatibility and is ignored (both views are written when
    costs are available).

    ``fit_binarize_threshold`` separates model fitting from evaluation: logistic models see
    ``1[Y >= threshold]`` while chosen/oracle means and regret use raw ``Y``. For ALPACA,
    the default is 0.5, matching PromptEval section 6.1.
    """
    if seed is not None:
        random_seed_list = [int(seed)]
    else:
        random_seed_list = list(range(int(random_seeds)))
    del cost_aware  # unused; both processed views are written when costs exist

    if fit_binarize_threshold is None and bench == "ALPACA":
        fit_binarize_threshold = ALPACA_FIT_BINARIZE_THRESHOLD

    cost_path = cost_file or DEFAULT_COST_FILES.get(bench)
    costs_full: Optional[np.ndarray] = None
    if cost_path is not None:
        costs_full = load_arm_costs(cost_path)
        print(
            f"recording costs: {len(costs_full)} arms from {cost_path} "
            f"(min={costs_full.min():.4g}, max={costs_full.max():.4g}); "
            "will write both unit-cost and cost-aware processed files",
            flush=True,
        )
    else:
        print("no cost file for this benchmark; writing unit-cost processed only", flush=True)
    if backend == "torch" and torch_device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("device=cuda requested but torch.cuda.is_available() is False.")

    with open(data_path + "Ys.pickle", "rb") as handle:
        Ys = pickle.load(handle)
    with open(data_path + "Xs.pickle", "rb") as handle:
        Xs = pickle.load(handle)

    all_task_keys = list(Ys[bench].keys())
    if tasks is not None:
        task_list = [str(t).strip() for t in tasks if str(t).strip()]
        if not task_list:
            raise ValueError("tasks= must contain at least one non-empty name.")
        unknown = [t for t in task_list if t not in Ys[bench]]
        if unknown:
            raise ValueError(f"Unknown task(s) for benchmark {bench!r}: {unknown!r}.")
    elif tasks_csv is not None:
        task_list = [t.strip() for t in tasks_csv.split(",") if t.strip()]
        if not task_list:
            raise ValueError("tasks_csv must list at least one non-empty task name.")
        unknown = [t for t in task_list if t not in Ys[bench]]
        if unknown:
            raise ValueError(f"Unknown task(s) for benchmark {bench!r}: {unknown!r}.")
    elif only_task is not None:
        if only_task not in Ys[bench]:
            raise ValueError(f"Unknown only_task {only_task!r} for benchmark {bench!r}.")
        task_list = [only_task]
    elif max_tasks is not None:
        task_list = all_task_keys[: max(0, int(max_tasks))]
    elif all_tasks:
        task_list = all_task_keys
    elif default_task_subset is not None:
        task_list = [t for t in default_task_subset if t in Ys[bench]]
        missing = [t for t in default_task_subset if t not in Ys[bench]]
        if missing:
            print("Warning: default_task_subset has unknown keys for this benchmark (skipped):", missing)
        if not task_list:
            raise ValueError("default_task_subset left no valid tasks after filtering.")
    elif DEFAULT_TASK_SUBSET is not None:
        task_list = [t for t in DEFAULT_TASK_SUBSET if t in Ys[bench]]
        missing = [t for t in DEFAULT_TASK_SUBSET if t not in Ys[bench]]
        if missing:
            print("Warning: DEFAULT_TASK_SUBSET has unknown keys for this benchmark (skipped):", missing)
        if not task_list:
            raise ValueError("DEFAULT_TASK_SUBSET left no valid tasks after filtering.")
    else:
        task_list = all_task_keys

    raw_tag = (results_tag or "").strip() + raw_results_tag(combine_models=combine_models)
    # Strip accidental mode tags from a custom --results-tag; modes go only on processed files.
    for mode in ("_unitcost", "_costaware"):
        raw_tag = raw_tag.replace(mode, "")
    banditeval = bench in ("GSM8K", "PIQA", "ALPACA")
    costs_recorded = costs_full is not None

    n_llm_ref = len(Ys[bench][task_list[0]]) if task_list else 0
    # MMLU subjects are independent datasets → one result file per task so plotting
    # can load a subject directly. GSM8K/PIQA keep a single multi-task file (tasks
    # are repeated measurements of the same questions).
    per_task_files = bench == "MMLU"

    def save_processed_views(
        outs_by_task: Dict[str, list],
        *,
        tasks_in_file: Sequence[str],
        raw_path: str,
        task_stem: Optional[str] = None,
        walls_by_task: Optional[Dict[str, Sequence[float]]] = None,
    ) -> List[str]:
        """Write unit-cost and (if costs recorded) cost-aware processed files from shared outs."""
        written: List[str] = []
        wall_flat: Optional[List[float]] = None
        if walls_by_task is not None:
            wall_flat = [float(w) for t in tasks_in_file for w in walls_by_task.get(t, [])]
        modes = [False, True] if costs_recorded else [False]
        for use_cost in modes:
            curves_list = [
                aggregate_eval_outs(outs_by_task[t], cost_aware=use_cost) for t in tasks_in_file
            ]
            stacked = stack_task_curves(curves_list, cost_aware=use_cost)
            tag = processed_results_tag(
                cost_aware=use_cost,
                combine_models=combine_models,
                banditeval=banditeval,
            )
            # Keep any custom results_tag prefix (without mode/combined) on processed files too.
            custom = (results_tag or "").strip()
            for mode in ("_unitcost", "_costaware", "_combined"):
                custom = custom.replace(mode, "")
            stem = results_file_stem(bench, custom + tag, task_stem)
            proc_path = results_path + f"bai_processed_results_{stem}.npy"
            save_bai_processed(
                proc_path,
                curves=stacked,
                tasks=list(tasks_in_file),
                seeds=random_seed_list,
                bench=bench,
                combine_models=combine_models,
                cost_aware=use_cost,
                n_models_stacked=n_llm_ref if combine_models else 1,
                raw_path=raw_path,
                total_wall_time_s=wall_flat,
                fit_binarize_threshold=fit_binarize_threshold,
            )
            written.append(proc_path)
            print(
                f"saved processed ({'cost-aware' if use_cost else 'unit-cost'}): {proc_path}",
                flush=True,
            )
        return written

    # Pre-stack once per combined-models task (ms), then run jobs serially.
    combined_by_task: Dict[str, Any] = {}
    if combine_models:
        for task in task_list:
            Y_cat, xs_cat = combine_models_y_xs(Ys, Xs, bench, task)
            if costs_full is not None and costs_full.shape[0] != Y_cat.shape[0]:
                raise ValueError(
                    f"{bench}/{task}: cost vector has {costs_full.shape[0]} entries "
                    f"but stacked Y has {Y_cat.shape[0]} arms."
                )
            combined_by_task[task] = (Y_cat, xs_cat)
            print(
                f"stacked {bench}/{task}: Y={Y_cat.shape}, budget={budgets_from_y(Y_cat)[0]}, "
                f"n_views={len(xs_cat)}",
                flush=True,
            )
        n_work = len(task_list) * len(random_seed_list)
        job_desc = (
            f"n_work_items (task×seed): {n_work} = {len(task_list)}×{len(random_seed_list)} "
            f"({n_llm_ref} models stacked / task)"
        )
    else:
        n_work = sum(len(Ys[bench][t]) for t in task_list) * len(random_seed_list)
        job_desc = (
            f"n_work_items (llm×bench_task×seed): {n_work} = "
            f"{len(task_list)}×{n_llm_ref}×{len(random_seed_list)}"
        )
    print(
        "benchmark:",
        bench,
        "combine_models:",
        combine_models,
        "backend:",
        backend,
        "torch_device:",
        torch_device,
        "random_seeds:",
        len(random_seed_list),
        "per_task_files:",
        per_task_files,
        job_desc,
        flush=True,
    )

    raw_paths: List[str] = []
    proc_paths: List[str] = []
    final_curves: List[np.ndarray] = []

    if combine_models:
        retained_out: Dict[str, list] = {}
        retained_jobs: Dict[str, list] = {}
        retained_walls: Dict[str, List[float]] = {}
        for task in task_list:
            Y_cat, xs_cat = combined_by_task[task]
            jobs = [(task, seed) for seed in random_seed_list]
            results: List = []
            walls: List[float] = []
            for _, seed in jobs:
                out, wt = evaluate_bai_combined_one_seed(
                    Y_cat,
                    xs_cat,
                    task,
                    seed,
                    backend,
                    torch_device,
                    torch_fit_log_interval,
                    fit_binarize_threshold,
                    costs=costs_full,
                )
                results.append(out)
                walls.append(wt)
            retained_out[task] = results
            retained_jobs[task] = jobs
            retained_walls[task] = walls
            stats = summarize_wall_times(walls)
            print(
                f"[BAI] timing task={task!r}: median={stats['median_s']:.3f}s "
                f"mean={stats['mean_s']:.3f}s se={stats['se_s']:.3f}s n={stats['n']}",
                flush=True,
            )
            # Unit-cost curves kept for the return value.
            final_curves.append(aggregate_eval_outs(results, cost_aware=False))

            if per_task_files:
                stem = results_file_stem(bench, raw_tag, task)
                raw_path = results_path + f"bai_results_{stem}.npy"
                save_bai_raw(
                    raw_path,
                    out=results,
                    jobs=jobs,
                    tasks=[task],
                    seeds=random_seed_list,
                    bench=bench,
                    combine_models=True,
                    costs_recorded=costs_recorded,
                    n_models_stacked=n_llm_ref,
                    out_by_task=[results],
                    cost_file=cost_path,
                    total_wall_time_s=walls,
                    fit_binarize_threshold=fit_binarize_threshold,
                )
                raw_paths.append(raw_path)
                proc_paths.extend(
                    save_processed_views(
                        {task: results},
                        tasks_in_file=[task],
                        raw_path=raw_path,
                        task_stem=task,
                        walls_by_task={task: walls},
                    )
                )

        if not per_task_files:
            all_out = [r for task in task_list for r in retained_out[task]]
            all_jobs = [j for task in task_list for j in retained_jobs[task]]
            all_walls = [w for task in task_list for w in retained_walls[task]]
            out_by_task = [retained_out[task] for task in task_list]
            stem = results_file_stem(bench, raw_tag)
            raw_path = results_path + f"bai_results_{stem}.npy"
            save_bai_raw(
                raw_path,
                out=all_out,
                jobs=all_jobs,
                tasks=task_list,
                seeds=random_seed_list,
                bench=bench,
                combine_models=True,
                costs_recorded=costs_recorded,
                n_models_stacked=n_llm_ref,
                out_by_task=out_by_task,
                cost_file=cost_path,
                total_wall_time_s=all_walls,
                fit_binarize_threshold=fit_binarize_threshold,
            )
            raw_paths = [raw_path]
            proc_paths = save_processed_views(
                retained_out,
                tasks_in_file=task_list,
                raw_path=raw_path,
                walls_by_task=retained_walls,
            )
    else:
        results_by_task: Dict[str, List] = {task: [] for task in task_list}
        jobs_by_task: Dict[str, List] = {task: [] for task in task_list}
        walls_by_task: Dict[str, List[float]] = {task: [] for task in task_list}
        for task in task_list:
            for random_seed in random_seed_list:
                for llm in range(len(Ys[bench][task])):
                    job = (llm, task, random_seed)
                    jobs_by_task[task].append(job)
                    print(
                        f"[BAI] start llm={llm} task={task!r} seed={random_seed}",
                        flush=True,
                    )
                    wall_t0 = time.perf_counter()
                    results_by_task[task].append(
                        evaluate_bai(
                            Ys[bench][task][llm],
                            Xs[bench][task][llm],
                            random_seed,
                            backend=backend,
                            torch_device=torch_device,
                            torch_fit_log_interval=torch_fit_log_interval,
                            fit_binarize_threshold=fit_binarize_threshold,
                            costs=slice_costs_for_llm(
                                costs_full, np.asarray(Ys[bench][task][llm]).shape[0], llm
                            ),
                        )
                    )
                    wt = float(time.perf_counter() - wall_t0)
                    walls_by_task[task].append(wt)
                    print(
                        f"[BAI] done  llm={llm} task={task!r} seed={random_seed} "
                        f"total_wall_time_s={wt:.3f}",
                        flush=True,
                    )

            stats = summarize_wall_times(walls_by_task[task])
            print(
                f"[BAI] timing task={task!r}: median={stats['median_s']:.3f}s "
                f"mean={stats['mean_s']:.3f}s se={stats['se_s']:.3f}s n={stats['n']}",
                flush=True,
            )
            final_curves.append(
                aggregate_eval_outs(results_by_task[task], cost_aware=False)
            )

            if per_task_files:
                stem = results_file_stem(bench, raw_tag, task)
                raw_path = results_path + f"bai_results_{stem}.npy"
                save_bai_raw(
                    raw_path,
                    out=results_by_task[task],
                    jobs=jobs_by_task[task],
                    tasks=[task],
                    seeds=random_seed_list,
                    bench=bench,
                    combine_models=False,
                    costs_recorded=costs_recorded,
                    n_models_stacked=1,
                    cost_file=cost_path,
                    total_wall_time_s=walls_by_task[task],
                    fit_binarize_threshold=fit_binarize_threshold,
                )
                raw_paths.append(raw_path)
                proc_paths.extend(
                    save_processed_views(
                        {task: results_by_task[task]},
                        tasks_in_file=[task],
                        raw_path=raw_path,
                        task_stem=task,
                        walls_by_task={task: walls_by_task[task]},
                    )
                )

        if not per_task_files:
            all_out = [r for task in task_list for r in results_by_task[task]]
            all_jobs = [j for task in task_list for j in jobs_by_task[task]]
            all_walls = [w for task in task_list for w in walls_by_task[task]]
            stem = results_file_stem(bench, raw_tag)
            raw_path = results_path + f"bai_results_{stem}.npy"
            save_bai_raw(
                raw_path,
                out=all_out,
                jobs=all_jobs,
                tasks=task_list,
                seeds=random_seed_list,
                bench=bench,
                combine_models=False,
                costs_recorded=costs_recorded,
                n_models_stacked=1,
                cost_file=cost_path,
                total_wall_time_s=all_walls,
                fit_binarize_threshold=fit_binarize_threshold,
            )
            raw_paths = [raw_path]
            proc_paths = save_processed_views(
                results_by_task,
                tasks_in_file=task_list,
                raw_path=raw_path,
                walls_by_task=walls_by_task,
            )

    final_results = stack_task_curves(final_curves, cost_aware=False)
    return {
        "final_results": final_results,
        "tasks": task_list,
        "raw_results_path": raw_paths[0] if len(raw_paths) == 1 else raw_paths,
        "processed_results_path": proc_paths[0] if len(proc_paths) == 1 else proc_paths,
        "combine_models": combine_models,
        "n_work_items": n_work,
        "costs_recorded": costs_recorded,
        "fit_binarize_threshold": fit_binarize_threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BAI evaluation on PromptEval (MMLU) or BanditEval "
            "(GSM8K/PIQA/AlpacaEval) pickles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bench",
        default="MMLU",
        choices=["MMLU", "GSM8K", "PIQA", "ALPACA"],
        help="Dataset/benchmark name. MMLU uses prompteval/data/ with combine_models; "
        "GSM8K/PIQA/ALPACA use prompteval/banditeval_pickle/ "
        "(build_banditeval_pickle.py) without.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Directory with Ys.pickle/Xs.pickle. Default depends on --bench.",
    )
    parser.add_argument("--results-path", default="prompteval/results/", help="Output directory.")
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=20,
        help="Number of independent BAI sampling repeats (seeds 0..N-1). Averaged in processed results. "
        "Ignored if --seed is set. Example: --random-seeds 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single BAI repeat with this RNG seed (overrides --random-seeds). Example: --seed 0",
    )
    parser.add_argument("--backend", default="sklearn", choices=["sklearn", "torch"], help="Regression backend.")
    parser.add_argument("--torch-device", default="auto", help="Torch device (auto/cpu/cuda).")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names (e.g. 'abstract_algebra,professional_law'). "
        "Default: all tasks for GSM8K/PIQA/ALPACA; "
        "abstract_algebra + professional_law for MMLU.",
    )
    parser.add_argument("--all-tasks", action="store_true", help="Run every task in the benchmark.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit to the first N tasks.")
    parser.add_argument(
        "--combine-models",
        dest="combine_models",
        action="store_true",
        default=None,
        help="Stack all LLM matrices per task (MMLU default).",
    )
    parser.add_argument(
        "--no-combine-models",
        dest="combine_models",
        action="store_false",
        help="One matrix per task (GSM8K/PIQA/ALPACA default).",
    )
    parser.add_argument("--results-tag", default=None, help="Optional custom suffix for result filenames.")
    parser.add_argument(
        "--cost-aware",
        action="store_true",
        help="Deprecated/no-op: when a cost file is available, one run records both observation "
        "and cost spend and writes unit-cost + cost-aware processed files.",
    )
    parser.add_argument(
        "--cost-file",
        default=None,
        help="JSON with per-arm costs (configuration index -> estimated_cost_per_1m_input_tokens). "
        "Default depends on --bench (see DEFAULT_COST_FILES). If present, both processed views are written.",
    )
    parser.add_argument(
        "--fit-binarize-threshold",
        type=float,
        default=None,
        help=(
            "Fit logistic models on indicator(score >= threshold) while evaluating regret on raw scores. "
            "Defaults to 0.5 for ALPACA and no transform for other benchmarks."
        ),
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    """
    CLI entry. Examples (from repo root)::

        python prompteval/bai_evaluation.py --bench MMLU
        python prompteval/bai_evaluation.py --bench GSM8K
        python prompteval/bai_evaluation.py --bench PIQA --tasks various_models_seed1,various_models_seed2
        python prompteval/bai_evaluation.py --bench ALPACA --random-seeds 20
        python prompteval/bai_evaluation.py --bench MMLU --all-tasks --random-seeds 20
        python prompteval/bai_evaluation.py --bench MMLU --tasks anatomy --seed 0
    """
    args = parse_args()
    is_banditeval = args.bench in ("GSM8K", "PIQA", "ALPACA")

    data_path = args.data_path or (DEFAULT_BANDITEVAL_PICKLE_DIR if is_banditeval else "prompteval/data/")
    combine_models = args.combine_models if args.combine_models is not None else not is_banditeval
    results_tag = args.results_tag if args.results_tag is not None else ""

    tasks_csv = args.tasks
    default_subset = None
    all_tasks = args.all_tasks
    if tasks_csv is None and not all_tasks and args.max_tasks is None:
        if is_banditeval:
            # The various_models_seed1-5 tasks are repeated measurements of the same
            # benchmark (different LLM-query seeds, same questions): run all by default.
            all_tasks = True
        else:
            default_subset = DEFAULT_TASK_SUBSET

    return run_bai_evaluation(
        data_path=data_path,
        results_path=args.results_path,
        bench=args.bench,
        random_seeds=args.random_seeds,
        seed=args.seed,
        backend=args.backend,
        torch_device=args.torch_device,
        tasks=None,
        tasks_csv=tasks_csv,
        only_task=None,
        max_tasks=args.max_tasks,
        all_tasks=all_tasks,
        default_task_subset=default_subset,
        combine_models=combine_models,
        results_tag=results_tag,
        cost_aware=args.cost_aware,
        cost_file=args.cost_file,
        fit_binarize_threshold=args.fit_binarize_threshold,
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        raise SystemExit(str(e)) from e
