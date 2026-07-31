import argparse
import json
import pickle
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from methods import LogReg, StratSample
from utils import flatten

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
    idx = sorted(cfg.keys(), key=int)
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

# BanditEval GSM8K/PIQA pickles from build_banditeval_pickle.py (no prompt-template covariates).
DEFAULT_BANDITEVAL_PICKLE_DIR = "prompteval/pickle/"

# CLI: python prompteval/bai_evaluation.py --bench {MMLU,GSM8K,PIQA} [--tasks ...] (see parse_args / --help).


def evaluate_bai_combined_one_task(
    Ys, Xs, bench: str, task: str, random_seeds, backend, torch_device, torch_fit_log_interval: int = 0, costs=None
):
    """Stack all models for one benchmark task, then run evaluate_bai for each seed (one worker job)."""
    Y_cat, xs_cat = combine_models_y_xs(Ys, Xs, bench, task)
    if costs is not None and costs.shape[0] != Y_cat.shape[0]:
        raise ValueError(
            f"{bench}/{task}: cost vector has {costs.shape[0]} entries but stacked Y has {Y_cat.shape[0]} arms."
        )
    return [
        evaluate_bai(
            Y_cat,
            xs_cat,
            random_seed,
            backend=backend,
            torch_device=torch_device,
            torch_fit_log_interval=torch_fit_log_interval,
            costs=costs,
        )
        for random_seed in random_seeds
    ]


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
    costs=None,
):
    """
    Computes the regret (best-format identification) using **logistic regression only**
    (``LogReg`` or ``LogRegTorch``) to score arms between elimination rounds.

    ``Z`` is accepted for backwards compatibility and ignored.

    Parameters:
    budget (int): The total budget for evaluation.
    Y (numpy.ndarray): The labels or responses for each format-example pair.
    X (numpy.ndarray): The format covariates. If not provided, an identity matrix is used.
    Z (numpy.ndarray, optional): Unused.
    random_seed (int, optional): The seed for random operations to ensure reproducibility.
    backend (str): "sklearn" (default) or "torch" for PyTorch logistic regression.
    torch_device (str): "auto", "cpu", or "cuda" when backend is "torch".
    torch_fit_log_interval (int): When ``backend=="torch"``, print training loss every this many
        epochs inside ``TorchLogisticRegression`` (0 = silent). Use e.g. 100 with ``n_jobs=1`` to
        avoid interleaved logs from parallel workers.
    costs (numpy.ndarray, optional): Per-arm cost of one observation, used for **accounting
        only**. Sampling and the observation budget are identical with or without costs; the
        recorded per-phase spend is the cost-weighted sum of observations when costs are given,
        otherwise the plain observation count.

    Returns:
    list: ``[phase_regrets, phase_spends]``, each of length
          ``n_phases = min(ceil(log2(n_formats)), MAX_BAI_PHASES)``.
          ``phase_regrets[p]`` is the simple regret of the arm logreg ranks highest among
          ``active_arms`` after sampling in phase ``p`` (``nan`` if that phase stopped before
          fitting, e.g. all examples already seen for active arms). ``phase_spends[p]`` is the
          cumulative budget spent through phase ``p`` (observations, or cost units with ``costs``).
    """
    del Z  # unused; kept in signature for call-site compatibility
    use_torch = backend == "torch"
    if use_torch and LogRegTorch is None:
        raise ImportError("methods_gpu / torch backend unavailable (import failed).")

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
    phase_regrets: List[float] = [float("nan")] * n_phases
    phase_spends: List[float] = [float("nan")] * n_phases
    oracle = float(Y.mean(-1).max())
    true_means = Y.mean(-1)

    for phase in range(n_phases):
        if phase == 0:
            seen_examples = StratSample(
                seen_examples, budget_phase + extra_first_phase, random_seed, active_arms, random_column
            )
        else:
            seen_examples = StratSample(
                seen_examples, seen_examples.sum() + budget_phase, random_seed, active_arms, random_column
            )

        if costs is None:
            phase_spends[phase] = float(seen_examples.sum())
        else:
            phase_spends[phase] = float((seen_examples.sum(1) * costs).sum())

        if seen_examples[active_arms].mean() == 1:
            break

        if use_torch:
            rasch_model = LogRegTorch(device=torch_device, fit_log_interval=torch_fit_log_interval)
        else:
            rasch_model = LogReg()
        rasch_model.fit(seen_examples, Y, X)
        mu = np.asarray(rasch_model.thetas)[active_arms]
        best_local = int(np.argmax(mu))
        ba_logreg = active_arms[best_local]
        phase_regrets[phase] = float(oracle - float(true_means[ba_logreg]))

        n_active = len(active_arms)
        n_eliminate = int(np.ceil(n_active / 2))
        if n_active - n_eliminate <= 0:
            break
        active_arms = [active_arms[i] for i in np.argsort(mu)[n_eliminate:].tolist()]

    return [phase_regrets, phase_spends]


def evaluate_bai(
    Y, Xs, random_seed, backend="sklearn", torch_device="auto", torch_fit_log_interval: int = 0, costs=None
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
    costs (numpy.ndarray, optional): Per-arm costs, recorded per phase for accounting only
        (sampling is unaffected; see ``compute_regrets``).

    Returns:
    list: A nested list for each budget and covariate block. Each ``compute_regrets`` entry is
          ``[phase_regrets, phase_spends]`` (see ``compute_regrets``).
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
                    costs=costs,
                )
            )

    return regrets


def run_bai_evaluation(
    data_path: str = "prompteval/data/",
    results_path: str = "prompteval/results/",
    bench: str = "MMLU",
    random_seeds: int = 20,
    backend: str = "sklearn",
    torch_device: str = "auto",
    n_jobs: Optional[int] = None,
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
) -> Dict[str, Any]:
    """
    Run the BAI evaluation pipeline (same behavior as the CLI).

    From another module, e.g.::

        from bai_evaluation import run_bai_evaluation
        run_bai_evaluation(bench="MMLU", random_seeds=5, tasks=["abstract_algebra"], combine_models=True)

    Task selection priority: ``tasks`` → ``tasks_csv`` → ``only_task`` → ``max_tasks`` →
    ``all_tasks`` → ``default_task_subset`` (if not None) → module ``DEFAULT_TASK_SUBSET`` (if not None) → all tasks.

    ``torch_fit_log_interval``: if >0 and ``backend=="torch"``, prints logistic training loss every
    that many epochs inside each ``TorchLogisticRegression`` fit. Prefer ``n_jobs=1`` when nonzero
    so worker logs do not interleave.

    ``cost_aware``: if True, per-arm costs (from ``cost_file``, default
    ``DEFAULT_COST_FILES[bench]``) are **recorded** alongside regrets: each phase's cumulative
    spend is the cost-weighted number of observations instead of the plain count. Sampling and
    the observation budget (10% of cells) are identical either way — costs never change the
    algorithm, only the recorded spend (e.g. for plotting regret vs dollars).
    """
    random_seed_list = list(range(int(random_seeds)))
    costs_full: Optional[np.ndarray] = None
    if cost_aware:
        cost_path = cost_file or DEFAULT_COST_FILES.get(bench)
        if cost_path is None:
            raise ValueError(f"cost_aware=True but no cost file known for benchmark {bench!r}.")
        costs_full = load_arm_costs(cost_path)
        print(f"cost-aware: {len(costs_full)} arm costs from {cost_path} "
              f"(min={costs_full.min():.4g}, max={costs_full.max():.4g})")
    n_jobs_eff = n_jobs if n_jobs is not None else (1 if backend == "torch" else 4)
    if backend == "torch" and torch_device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("device=cuda requested but torch.cuda.is_available() is False.")
    if backend == "torch" and torch_device == "cuda" and n_jobs_eff != 1:
        print("Warning: backend=torch with device=cuda and n_jobs>1 can overload the GPU; prefer n_jobs=1.")

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

    tag = (results_tag or "").strip()
    if cost_aware:
        tag = tag + "_costaware"
    if combine_models:
        tag = tag + "_combined"

    n_llm_ref = len(Ys[bench][task_list[0]]) if task_list else 0

    if combine_models:
        jobs = list(task_list)
    else:
        jobs = flatten(
            flatten(
                [
                    [[(llm, task, random_seed) for llm in range(len(Ys[bench][task]))] for task in task_list]
                    for random_seed in random_seed_list
                ]
            )
        )

    if combine_models:
        job_desc = (
            f"n_parallel_jobs (bench_task only): {len(jobs)} = {len(task_list)} tasks "
            f"(each runs {len(random_seed_list)} seeds; {n_llm_ref} models stacked / task)"
        )
    else:
        job_desc = (
            f"n_parallel_jobs (llm×bench_task×seed): {len(jobs)} = {len(task_list)}×{n_llm_ref}×{len(random_seed_list)}"
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
        "parallel_n_jobs:",
        n_jobs_eff,
        job_desc,
    )

    if combine_models:
        results = Parallel(n_jobs=n_jobs_eff, verbose=10)(
            delayed(evaluate_bai_combined_one_task)(
                Ys,
                Xs,
                bench,
                task,
                random_seed_list,
                backend,
                torch_device,
                torch_fit_log_interval,
                costs=costs_full,
            )
            for task in jobs
        )
    else:
        results = Parallel(n_jobs=n_jobs_eff, verbose=10)(
            delayed(evaluate_bai)(
                Ys[bench][job[1]][job[0]],
                Xs[bench][job[1]][job[0]],
                job[2],
                backend=backend,
                torch_device=torch_device,
                torch_fit_log_interval=torch_fit_log_interval,
                costs=slice_costs_for_llm(
                    costs_full, np.asarray(Ys[bench][job[1]][job[0]]).shape[0], job[0]
                ),
            )
            for job in jobs
        )

    raw_path = results_path + f"bai_results_{bench}{tag}.npy"
    proc_path = results_path + f"bai_processed_results_{bench}{tag}.npy"

    if combine_models:
        results_flat = [r for per_task in results for r in per_task]
        np.save(
            raw_path,
            {
                "out": results_flat,
                "out_by_task": results,
                "combine_models": True,
                "n_models_stacked": n_llm_ref,
            },
        )
    else:
        np.save(raw_path, {"out": results, "combine_models": False, "n_models_stacked": 1})

    if combine_models:
        results_dic = {task: results[ti] for ti, task in enumerate(task_list)}
        final_results = np.stack(
            [
                np.mean(np.stack([np.asarray(r, dtype=float) for r in results_dic[task]]), axis=0)
                for task in task_list
            ]
        )
    else:
        results_dic = {}
        for task in task_list:
            results_dic[task] = []
            for llm in range(len(Ys[bench][task])):
                results_dic[task].append([])
        for i, job in enumerate(jobs):
            task = job[1]
            llm = job[0]
            results_dic[task][llm].append(results[i])

        final_results = np.stack([np.stack(results_dic[task]).mean(0).mean(0) for task in task_list])

    np.save(proc_path, final_results)

    return {
        "final_results": final_results,
        "tasks": task_list,
        "raw_results_path": raw_path,
        "processed_results_path": proc_path,
        "combine_models": combine_models,
        "n_parallel_jobs": len(jobs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BAI evaluation on PromptEval (MMLU) or BanditEval (GSM8K/PIQA) pickles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bench",
        default="MMLU",
        choices=["MMLU", "GSM8K", "PIQA"],
        help="Dataset/benchmark name. MMLU uses prompteval/data/ with combine_models; "
        "GSM8K/PIQA use prompteval/pickle/ (build_banditeval_pickle.py) without.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Directory with Ys.pickle/Xs.pickle. Default depends on --bench.",
    )
    parser.add_argument("--results-path", default="prompteval/results/", help="Output directory.")
    parser.add_argument("--random-seeds", type=int, default=20, help="Number of random seeds.")
    parser.add_argument("--backend", default="sklearn", choices=["sklearn", "torch"], help="Regression backend.")
    parser.add_argument("--torch-device", default="auto", help="Torch device (auto/cpu/cuda).")
    parser.add_argument("--n-jobs", type=int, default=2, help="Parallel jobs.")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names (e.g. 'abstract_algebra,professional_law'). "
        "Default: all various_models_seed1-5 for GSM8K/PIQA; "
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
        help="One matrix per task (GSM8K/PIQA default).",
    )
    parser.add_argument("--results-tag", default=None, help="Suffix for result filenames. Default depends on --bench.")
    parser.add_argument(
        "--cost-aware",
        action="store_true",
        help="Record each phase's cumulative spend in cost units (per-arm costs from --cost-file) "
        "instead of observation counts. Sampling itself is unchanged (all costs 1, 10%% budget).",
    )
    parser.add_argument(
        "--cost-file",
        default=None,
        help="JSON with per-arm costs (configuration index -> estimated_cost_per_1m_input_tokens). "
        "Default depends on --bench (see DEFAULT_COST_FILES).",
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    """
    CLI entry. Examples (from repo root)::

        python prompteval/bai_evaluation.py --bench MMLU
        python prompteval/bai_evaluation.py --bench GSM8K
        python prompteval/bai_evaluation.py --bench PIQA --tasks various_models_seed1,various_models_seed2
        python prompteval/bai_evaluation.py --bench MMLU --all-tasks --random-seeds 20
    """
    args = parse_args()
    is_banditeval = args.bench in ("GSM8K", "PIQA")

    data_path = args.data_path or (DEFAULT_BANDITEVAL_PICKLE_DIR if is_banditeval else "prompteval/data/")
    combine_models = args.combine_models if args.combine_models is not None else not is_banditeval
    results_tag = args.results_tag if args.results_tag is not None else ("_banditeval" if is_banditeval else "")

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
        backend=args.backend,
        torch_device=args.torch_device,
        n_jobs=args.n_jobs,
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
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        raise SystemExit(str(e)) from e
