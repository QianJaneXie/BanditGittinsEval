#!/usr/bin/env python3
"""
Plot BAI simple regret vs budget from bai_evaluation.py results.

The processed results file is a dict with ``curves`` of shape
``(n_tasks, n_budgets, n_blocks, 2, n_phases)`` (regret, budget). Full per-phase
updates (chosen arm, mean, regret, budget) are in the raw results file.

File layout:
    - MMLU: one processed file per subject, e.g.
      ``bai_processed_results_MMLU_abstract_algebra_combined.npy``
    - GSM8K / PIQA: one multi-task file, e.g.
      ``bai_processed_results_GSM8K_unitcost.npy``

Plotting policy:
    - GSM8K / PIQA: ``various_models_seed1``…``seed5`` are repeated measurements of the
      same questions (different LLM-query seeds) → one panel, mean over those tasks.
    - MMLU: each subject file → one panel (pass ``--tasks`` to select subjects).

Examples (from repo root)::

    python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra
    python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra,professional_law
    python prompteval/plot_bai_regret.py --bench PIQA
    python prompteval/plot_bai_regret.py --bench GSM8K --cost-aware
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "prompteval") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "prompteval"))

from bai_evaluation import aggregate_curves  # noqa: E402

# Covariate-block labels, in evaluate_bai order (baseline first, then Xs views).
MMLU_BLOCK_LABELS = [
    "one-hot (baseline)",
    "discrete template features",
    "sentence-transformer (PCA)",
    "fine-tuned BERT (PCA)",
]
BANDITEVAL_BLOCK_LABELS = ["one-hot (baseline)"]

BLOCK_COLORS = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#8C564B", "#E377C2"]

DEFAULT_MMLU_TASKS = ("abstract_algebra", "professional_law")


def load_ys(data_path: Path) -> dict:
    with open(data_path / "Ys.pickle", "rb") as handle:
        return pickle.load(handle)


def load_processed_dict(path: Path) -> tuple[np.ndarray, list[str] | None]:
    """Load curves and optional task list from a processed results .npy."""
    if not path.is_file():
        raise FileNotFoundError(f"Processed results not found: {path}")
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        loaded = loaded.item()
    if isinstance(loaded, dict) and "curves" in loaded:
        curves = np.asarray(loaded["curves"], dtype=float)
        tasks = list(loaded["tasks"]) if loaded.get("tasks") else None
        return curves, tasks
    return np.asarray(loaded, dtype=float), None


def load_mmlu_curves(
    results_path: Path,
    bench: str,
    tag: str,
    tasks: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Load one processed file per MMLU subject and stack into (n_tasks, ...).

    Falls back to a legacy multi-task file ``bai_processed_results_{bench}{tag}.npy``
    when a per-subject file is missing (older runs).
    """
    by_name: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for task in tasks:
        per_task = results_path / f"bai_processed_results_{bench}_{task}{tag}.npy"
        if per_task.is_file():
            curves, file_tasks = load_processed_dict(per_task)
            if curves.ndim == 4:
                curves = curves[None, ...]
            if curves.shape[0] != 1:
                raise ValueError(
                    f"{per_task.name}: expected a single-task file (n_tasks=1), got shape {curves.shape}."
                )
            by_name[task] = curves[0]
            continue
        missing.append(task)

    if missing:
        legacy = results_path / f"bai_processed_results_{bench}{tag}.npy"
        if not legacy.is_file():
            raise FileNotFoundError(
                "Missing per-subject MMLU results for "
                + ", ".join(missing)
                + f". Expected e.g. bai_processed_results_{bench}_{missing[0]}{tag}.npy"
            )
        curves, file_tasks = load_processed_dict(legacy)
        if file_tasks is None:
            raise ValueError(
                f"{legacy.name} has no 'tasks' metadata; cannot map --tasks {missing}."
            )
        for task in missing:
            if task not in file_tasks:
                raise FileNotFoundError(
                    f"No per-subject file for {task!r} and it is not in legacy {legacy.name} "
                    f"(tasks={file_tasks})."
                )
            by_name[task] = np.asarray(curves[file_tasks.index(task)], dtype=float)

    stacked = np.stack([by_name[t] for t in tasks], axis=0)
    return stacked, list(tasks)


def load_multitask_curves(
    results_path: Path,
    bench: str,
    tag: str,
    tasks: list[str] | None,
) -> tuple[np.ndarray, list[str] | None]:
    """Load a single multi-task processed file (GSM8K / PIQA / legacy MMLU)."""
    proc_file = results_path / f"bai_processed_results_{bench}{tag}.npy"
    curves, file_tasks = load_processed_dict(proc_file)
    if tasks is not None and file_tasks is not None and list(tasks) != list(file_tasks):
        unknown = [t for t in tasks if t not in file_tasks]
        if unknown:
            raise ValueError(f"Unknown task(s) in {proc_file.name}: {unknown!r}.")
        idx = [file_tasks.index(t) for t in tasks]
        curves = curves[idx]
        return curves, list(tasks)
    return curves, file_tasks if file_tasks is not None else tasks


def infer_task_names(ys: dict, bench: str, n_tasks: int, tasks: list[str] | None) -> list[str]:
    """
    Resolve task names for a processed-results array of length ``n_tasks``.

    Prefer an explicit ``tasks`` list when it matches ``n_tasks``; otherwise use the
    pickle's task-key order (what ``--all-tasks`` / BanditEval defaults write).
    """
    pickle_tasks = list(ys[bench].keys())
    if tasks is not None and len(tasks) == n_tasks:
        unknown = [t for t in tasks if t not in ys[bench]]
        if unknown:
            raise ValueError(f"Unknown task(s) for {bench!r}: {unknown!r}.")
        return tasks
    if len(pickle_tasks) == n_tasks:
        if tasks is not None and len(tasks) != n_tasks:
            print(
                f"Warning: --tasks has {len(tasks)} name(s) but results have {n_tasks} task(s); "
                f"using pickle order {pickle_tasks}."
            )
        return pickle_tasks
    if tasks is None:
        raise ValueError(
            f"Results have {n_tasks} task(s) but {bench} pickle has {len(pickle_tasks)}. "
            "Pass --tasks with names in the same order as the bai_evaluation run."
        )
    raise ValueError(
        f"Results have {n_tasks} task(s) but {len(tasks)} names were given: {tasks}. "
        "Pass --tasks in the same order as the bai_evaluation run."
    )


def load_task_shapes(ys: dict, bench: str, tasks: list[str], combined: bool):
    """Return {task: (n_arms, n_examples)} (arms include stacked LLMs if combined)."""
    shapes = {}
    for task in tasks:
        mats = ys[bench][task]
        n_formats, n_examples = np.asarray(mats[0]).shape
        n_arms = len(mats) * n_formats if combined else n_formats
        shapes[task] = (n_arms, n_examples)
    return shapes


def plot_simple_regret(
    bench: str,
    results_tag: str,
    tasks: list[str] | None,
    data_path: Path,
    results_path: Path,
    out_path: Path | None = None,
    combined: bool | None = None,
    cost_aware: bool = False,
    aggregate_tasks: bool | None = None,
) -> Path:
    """
    Draw simple regret (y) vs the recorded cumulative budget (x: observations, or cost
    units for cost-aware runs), one line per covariate block.

    ``aggregate_tasks``: average results over the task axis into a single panel. Default:
    True for GSM8K/PIQA (tasks are repeated measurements of the same benchmark, one per
    LLM-query seed) and False for MMLU (tasks are different datasets, one panel each).
    Returns the saved figure path.
    """
    if combined is None:
        combined = bench == "MMLU"
    if aggregate_tasks is None:
        aggregate_tasks = bench != "MMLU"
    tag = results_tag + ("_combined" if combined else "")

    if bench == "MMLU":
        task_names = list(tasks) if tasks else list(DEFAULT_MMLU_TASKS)
        results, task_names = load_mmlu_curves(results_path, bench, tag, task_names)
    else:
        results, file_tasks = load_multitask_curves(results_path, bench, tag, tasks)
        if file_tasks is not None:
            tasks = file_tasks
        ys_tmp = load_ys(data_path)
        n_tasks_tmp = results.shape[0]
        task_names = infer_task_names(ys_tmp, bench, n_tasks_tmp, tasks)

    if results.ndim != 5 or results.shape[3] != 2:
        raise ValueError(
            f"curves have shape {results.shape}; expected (tasks, budgets, blocks, 2, phases). "
            "Re-run bai_evaluation.py to regenerate results in the current format."
        )
    n_tasks, n_budgets, n_blocks, _, n_phases = results.shape

    ys = load_ys(data_path)

    labels = MMLU_BLOCK_LABELS if bench == "MMLU" else BANDITEVAL_BLOCK_LABELS
    if len(labels) < n_blocks:
        labels = labels + [f"covariate block {i}" for i in range(len(labels), n_blocks)]

    shapes = load_task_shapes(ys, bench, task_names, combined)

    if aggregate_tasks:
        # Tasks are repeated measurements (same questions, different LLM-query seeds).
        # Unit-cost: phase-aligned mean. Cost-aware: interpolate onto a shared spend grid
        # (each data-seed's mean curve can sit on different cost points).
        panel_results = aggregate_curves(
            [results[ti] for ti in range(n_tasks)],
            cost_aware=cost_aware,
        )[None, ...]
        panel_titles = [f"{bench} — mean over {n_tasks} data seed(s)"]
        panel_shape_keys = [task_names[0]]
    else:
        panel_results = results
        panel_titles = [f"{bench} — {task}" for task in task_names]
        panel_shape_keys = task_names
    n_panels = panel_results.shape[0]

    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.2 * n_panels, 4.2), squeeze=False, sharey=False
    )
    for pi in range(n_panels):
        ax = axes[0][pi]
        n_arms, n_examples = shapes[panel_shape_keys[pi]]
        n_pts = panel_results.shape[-1]
        markevery = max(1, n_pts // 12) if n_pts > 12 else 1
        for bi in range(n_blocks):
            y = panel_results[pi, 0, bi, 0, :]
            x = panel_results[pi, 0, bi, 1, :]
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size == 0:
                continue
            # Hold regret constant between evaluations (chosen arm unchanged until next phase).
            ax.plot(
                x,
                y,
                drawstyle="steps-post",
                marker="o",
                markersize=4,
                markevery=markevery if x.size > 12 else 1,
                linewidth=1.8,
                color=BLOCK_COLORS[bi % len(BLOCK_COLORS)],
                label=labels[bi],
            )
        ax.set_title(f"{panel_titles[pi]}\n({n_arms} arms × {n_examples} examples)")
        ax.set_xlabel(
            "Budget (cumulative cost units)" if cost_aware else "Observation budget (cumulative samples)"
        )
        ax.set_ylabel("Simple regret")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    if out_path is None:
        out_dir = REPO_ROOT / "prompteval" / "outputs" / "bai_regret_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        if bench == "MMLU" and len(task_names) == 1:
            out_path = out_dir / f"bai_simple_regret_{bench}_{task_names[0]}{tag}.png"
        elif bench == "MMLU":
            out_path = out_dir / f"bai_simple_regret_{bench}_{'-'.join(task_names)}{tag}.png"
        else:
            out_path = out_dir / f"bai_simple_regret_{bench}{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot BAI simple regret vs budget from bai_evaluation.py results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bench", default="MMLU", choices=["MMLU", "GSM8K", "PIQA"])
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names. MMLU: loads one processed file per subject "
        f"(default: {','.join(DEFAULT_MMLU_TASKS)}). "
        "GSM8K/PIQA: optional subset of the multi-task file (default: all, then average).",
    )
    p.add_argument("--data-path", type=Path, default=None, help="Dir with Ys.pickle (default per bench).")
    p.add_argument("--results-path", type=Path, default=REPO_ROOT / "prompteval" / "results")
    p.add_argument("--results-tag", default=None, help="Tag used in the run (default per bench).")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    p.add_argument(
        "--no-combined",
        action="store_true",
        help="Results were produced with --no-combine-models (MMLU only; GSM8K/PIQA never combine).",
    )
    p.add_argument(
        "--cost-aware",
        action="store_true",
        help="Results were produced with --cost-aware: x axis in cost units, '_costaware' tag.",
    )
    p.add_argument(
        "--per-task",
        action="store_true",
        help="One panel per task instead of averaging over tasks "
        "(GSM8K/PIQA aggregate by default; MMLU is always per task).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    is_banditeval = args.bench in ("GSM8K", "PIQA")
    combined = False if is_banditeval else not args.no_combined
    data_path = args.data_path or (
        REPO_ROOT / "prompteval" / ("banditeval_pickle" if is_banditeval else "data")
    )
    # GSM8K/PIQA: `_unitcost` XOR `_costaware`. MMLU cost-aware: `_costaware` (+ `_combined` later).
    if args.results_tag is not None:
        results_tag = args.results_tag
    elif is_banditeval:
        results_tag = "_costaware" if args.cost_aware else "_unitcost"
    elif args.cost_aware:
        results_tag = "_costaware"
    else:
        results_tag = ""
    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks is not None
        else None
    )

    out = plot_simple_regret(
        bench=args.bench,
        results_tag=results_tag,
        tasks=tasks,
        data_path=data_path,
        results_path=args.results_path,
        out_path=args.out,
        combined=combined,
        cost_aware=args.cost_aware,
        aggregate_tasks=False if (args.per_task or args.bench == "MMLU") else True,
    )
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
