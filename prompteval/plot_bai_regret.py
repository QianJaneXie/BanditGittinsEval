#!/usr/bin/env python3
"""
Plot BAI simple regret vs budget from bai_evaluation.py results.

The processed results array has shape (n_tasks, n_budgets, n_blocks, 2, n_phases):
channel 0 is the per-phase simple regret and channel 1 is the cumulative budget
spent through that phase (observation counts, or cost units for --cost-aware runs),
both recorded during the run and averaged over sampling seeds.

Plotting policy:
- GSM8K / PIQA: ``various_models_seed1``…``seed5`` are repeated measurements of the
  same questions (different LLM-query seeds) → one panel, mean over those tasks.
- MMLU: each task is a different subject → one panel per task.

Examples (from repo root)::

    python prompteval/plot_bai_regret.py --bench PIQA
    python prompteval/plot_bai_regret.py --bench GSM8K --cost-aware
    python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra,professional_law
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Covariate-block labels, in evaluate_bai order (baseline first, then Xs views).
MMLU_BLOCK_LABELS = [
    "one-hot (baseline)",
    "discrete template features",
    "sentence-transformer (PCA)",
    "fine-tuned BERT (PCA)",
]
BANDITEVAL_BLOCK_LABELS = ["one-hot (baseline)"]

BLOCK_COLORS = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#8C564B", "#E377C2"]


def load_ys(data_path: Path) -> dict:
    with open(data_path / "Ys.pickle", "rb") as handle:
        return pickle.load(handle)


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

    proc_file = results_path / f"bai_processed_results_{bench}{tag}.npy"
    if not proc_file.is_file():
        raise FileNotFoundError(f"Processed results not found: {proc_file}")
    results = np.load(proc_file, allow_pickle=True)
    results = np.asarray(results, dtype=float)  # (n_tasks, n_budgets, n_blocks, 2, n_phases)
    if results.ndim != 5 or results.shape[3] != 2:
        raise ValueError(
            f"{proc_file.name} has shape {results.shape}; expected (tasks, budgets, blocks, 2, phases). "
            "Re-run bai_evaluation.py to regenerate results in the current format."
        )
    n_tasks, n_budgets, n_blocks, _, n_phases = results.shape

    ys = load_ys(data_path)
    task_names = infer_task_names(ys, bench, n_tasks, tasks)

    labels = MMLU_BLOCK_LABELS if bench == "MMLU" else BANDITEVAL_BLOCK_LABELS
    if len(labels) < n_blocks:
        labels = labels + [f"covariate block {i}" for i in range(len(labels), n_blocks)]

    shapes = load_task_shapes(ys, bench, task_names, combined)

    if aggregate_tasks:
        # Tasks are repeated measurements (same questions, different LLM-query seeds):
        # average regret and spend over the task axis into a single panel.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            panel_results = np.nanmean(results, axis=0, keepdims=True)  # (1, budgets, blocks, 2, phases)
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
        for bi in range(n_blocks):
            y = panel_results[pi, 0, bi, 0, :]
            x = panel_results[pi, 0, bi, 1, :]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
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
        help="Comma-separated task names in the same order as the bai_evaluation run. "
        "If omitted, names are inferred from the pickle when the task counts match. "
        "GSM8K/PIQA plots average over tasks by default; MMLU draws one panel per task.",
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
        REPO_ROOT / "prompteval" / ("pickle" if is_banditeval else "data")
    )
    results_tag = args.results_tag if args.results_tag is not None else ("_banditeval" if is_banditeval else "")
    if args.cost_aware:
        results_tag = results_tag + "_costaware"
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
