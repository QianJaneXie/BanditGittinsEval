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


def budgets_from_y(Y: np.ndarray, max_fraction: float = OBSERVATION_BUDGET_MAX_FRACTION) -> List[int]:
    """
    One total observation budget for BAI: ``max_fraction * n_formats * n_examples`` (rounded).
    """
    n_cells = int(Y.shape[0]) * int(Y.shape[1])
    cap = max(1, int(round(float(max_fraction) * n_cells)))
    return [cap]

# Used by run_bai_evaluation() when no tasks= / tasks_csv / only_task / max_tasks / all_tasks / default_task_subset.
DEFAULT_TASK_SUBSET = ("abstract_algebra", "professional_law")

# python bai_evaluation.py  # runs main(): torch+GPU, combine_models, two MMLU tasks (edit main() to change).


def evaluate_bai_combined_one_task(
    Ys, Xs, bench: str, task: str, random_seeds, backend, torch_device, torch_fit_log_interval: int = 0
):
    """Stack all models for one benchmark task, then run evaluate_bai for each seed (one worker job)."""
    Y_cat, xs_cat = combine_models_y_xs(Ys, Xs, bench, task)
    return [
        evaluate_bai(
            Y_cat,
            xs_cat,
            random_seed,
            backend=backend,
            torch_device=torch_device,
            torch_fit_log_interval=torch_fit_log_interval,
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

    Returns:
    list: Length ``n_phases = ceil(log2(n_formats))``. Slot ``p`` is the simple regret of the arm
          logreg ranks highest among ``active_arms`` after sampling in phase ``p`` (``nan`` if that
          phase stopped before fitting, e.g. all examples already seen for active arms).
    """
    del Z  # unused; kept in signature for call-site compatibility
    use_torch = backend == "torch"
    if use_torch and LogRegTorch is None:
        raise ImportError("methods_gpu / torch backend unavailable (import failed).")

    n_formats, n_examples = Y.shape
    n_phases = int(np.ceil(np.log2(n_formats)))
    budget_phase = int(np.floor(budget / n_phases))
    extra_first_phase = budget - n_phases * budget_phase

    active_arms = list(range(n_formats))
    random_column = True
    seen_examples = np.zeros(Y.shape).astype(bool)
    phase_regrets: List[float] = [float("nan")] * n_phases
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

    return phase_regrets


def evaluate_bai(Y, Xs, random_seed, backend="sklearn", torch_device="auto", torch_fit_log_interval: int = 0):
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

    Returns:
    list: A nested list for each budget and covariate block. Each ``compute_regrets`` entry is a
          per-phase list (see ``compute_regrets``): simple regret of logreg's top arm after each phase.
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
    """
    random_seed_list = list(range(int(random_seeds)))
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


def main() -> Dict[str, Any]:
    """
    Local test entry: set all ``run_bai_evaluation`` arguments here.
    CLI parsing can be added later (e.g. argparse → same kwargs).
    """
    return run_bai_evaluation(
        data_path="prompteval/data/",
        results_path="prompteval/results/",
        bench="MMLU",
        random_seeds=10,
        backend="sklearn",
        torch_device="auto",
        n_jobs=2,
        tasks=("abstract_algebra", "professional_law"),
        tasks_csv=None,
        only_task=None,
        max_tasks=None,
        all_tasks=False,
        default_task_subset=None,
        combine_models=True,
        results_tag="",
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        raise SystemExit(str(e)) from e
