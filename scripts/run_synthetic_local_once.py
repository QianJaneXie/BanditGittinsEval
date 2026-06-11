#!/usr/bin/env python3
"""
Run local synthetic simple-regret experiments and plot three-way comparisons.

No W&B upload. This script runs the simple-regret implementation in-process and
writes local trace .npz files itself, so it does NOT require run_simple_regret_wandb.py
to support --out-dir.

For each synthetic matrix, plot three curves:
  1. UCB-E baseline
  2. Gittins with matched prior
  3. Gittins with mismatched prior

Conventions used here:
  - UCB and Gittins use the SAME batch size for a fair comparison.
  - GSM8K / PIQA synthetic:
      UCB B8 and Gittins B8.
      Unit mode uses ucb_B8 and gittins_unit_B8_...
      Cost mode uses ucb_cost_B8 and gittins_cost_B8_... with a cost vector.
      If the synthetic folder name ends with _default, default prior is matched.
      If the synthetic folder name ends with _dataset, dataset prior is matched.
  - MMLU synthetic:
      UCB and Gittins batch size depends on subject size:
        small  (n_examples <= 150): B=2
        medium (151..400):         B=4
        large  (>400):             B=8
      For mmlu_high synthetic, dataset/high prior is matched and default prior is mismatched.
  - LRF is not included in this three-way synthetic script; in the main experiments LRF stays B32.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def load_runner_module(runner_path: Path) -> ModuleType:
    runner_path = runner_path.resolve()
    repo_root = runner_path.parents[1]
    for p in (repo_root, repo_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("synthetic_simple_regret_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def mmlu_batch_size(n_examples: int) -> int:
    """
    Current synthetic MMLU convention requested:
      small:  n_examples <= 150       -> B=2
      medium: 151 <= n_examples <=400 -> B=4
      large:  n_examples > 400        -> B=8
    """
    if int(n_examples) <= 150:
        return 2
    if int(n_examples) <= 400:
        return 4
    return 8


def mmlu_size_bucket(n_examples: int) -> str:
    if int(n_examples) <= 150:
        return "small"
    if int(n_examples) <= 400:
        return "medium"
    return "large"


def dataset_gittins_batch_size(dataset_tag: str, n_examples: int | None = None) -> int:
    """Requested synthetic convention for BOTH UCB and Gittins.

    UCB and Gittins use the same batch size:
      - GSM8K / PIQA: B=8
      - MMLU small / medium / large: B=2 / 4 / 8
    LRF is not included here; LRF stays B32 in the main experiments.
    """
    ds = str(dataset_tag).lower()
    if ds in {"gsm8k", "piqa"}:
        return 8
    if ds == "mmlu":
        if n_examples is None:
            raise ValueError("n_examples is required for MMLU batch-size selection")
        return mmlu_batch_size(int(n_examples))
    raise ValueError(f"Unsupported dataset_tag for synthetic batch size: {dataset_tag!r}")


def matched_and_mismatched_priors_from_setting(setting_name: str) -> tuple[str, str]:
    """Infer matched and mismatched prior types from synthetic folder name.

    Examples:
      gsm8k_default -> matched=default, mismatched=dataset
      piqa_dataset  -> matched=dataset, mismatched=default
    """
    s = str(setting_name).lower()
    if s.endswith("_dataset") or "dataset" in s:
        return "dataset", "default"
    # Current generated GSM8K folder is gsm8k_default. Treat ambiguous folders as default-generated.
    return "default", "dataset"


def trace_path_for(*, out_dir: Path, dataset_tag: str, variant: str, matrix: Path, run_seed: int) -> Path:
    return (
        out_dir
        / dataset_tag
        / safe_token(variant)
        / f"{matrix.stem}__runseed{run_seed}__{safe_token(variant)}_traces.npz"
    )


def run_variant_local(
    *,
    runner_mod: ModuleType,
    dataset_tag: str,
    matrix: Path,
    variant_raw: str,
    out_dir: Path,
    run_seed: int,
    eval_budget_fraction: float,
    mmlu_task_metadata: Path | None,
    cost_vector: Path | None,
    reuse_existing: bool,
) -> Path:
    trace_path = trace_path_for(
        out_dir=out_dir,
        dataset_tag=dataset_tag,
        variant=variant_raw,
        matrix=matrix,
        run_seed=run_seed,
    )
    if reuse_existing and trace_path.is_file():
        print(f"Reusing existing trace: {trace_path}")
        return trace_path

    print()
    print("Running local variant:")
    print(f"  dataset={dataset_tag}")
    print(f"  matrix={matrix}")
    print(f"  variant={variant_raw}")
    print(f"  run_seed={run_seed}")

    if not matrix.is_file():
        raise FileNotFoundError(f"Matrix not found: {matrix}")

    variant = runner_mod.parse_experiment_variant(variant_raw)
    ground_truth = torch.tensor(np.load(matrix), dtype=torch.float32)
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {tuple(ground_truth.shape)}: {matrix}")

    n_arms, n_examples = map(int, ground_truth.shape)
    n_cells = int(ground_truth.numel())
    if cost_vector is not None:
        actual_cost_per_arm = runner_mod.load_cost_vector(cost_vector, n_arms)
    else:
        actual_cost_per_arm = torch.ones((n_arms,), dtype=torch.float64)

    total_bf_cost = float(n_examples * actual_cost_per_arm.sum().item())
    budget_evals = int(max(1, round(float(eval_budget_fraction) * n_cells)))
    cost_aware = variant.cost_mode in {"cost", "aware"}
    if cost_aware:
        if cost_vector is None:
            raise ValueError(f"Cost-aware variant requires a cost vector: {variant_raw}")
        max_evaluations = int(n_cells)
        max_original_cost = float(eval_budget_fraction) * total_bf_cost
    else:
        max_evaluations = budget_evals
        max_original_cost = None

    args = SimpleNamespace(
        dataset_tag=dataset_tag,
        matrix=matrix,
        experiment_variant=variant_raw,
        run_seed=int(run_seed),
        eval_budget_fraction=float(eval_budget_fraction),
        ucb_a=1.0,
        warmup_percentage=0.05,
        lrf_device="cpu",
        cost_vector=cost_vector,
        gittins_grid_points=2**10 + 1,
        gittins_obs_noise_variance=None,
        gittins_prior_mean=None,
        gittins_prior_variance=None,
        mmlu_task_metadata=mmlu_task_metadata,
        log_step_metrics=False,
        wandb_entity=None,
        wandb_project="GittinsBanditEval",
        wandb_group=None,
        wandb_name=None,
        wandb_mode="disabled",
    )

    task_prior_buckets = (
        runner_mod.load_mmlu_task_prior_buckets(mmlu_task_metadata)
        if dataset_tag == "mmlu" and mmlu_task_metadata is not None
        else {}
    )
    prior_mean, prior_variance, mmlu_task, prior_bucket, prior_source = runner_mod.resolve_prior(
        dataset_tag=dataset_tag,
        prior_type=variant.prior_type,
        matrix=matrix,
        mmlu_task_prior_buckets=task_prior_buckets,
    )

    t0 = time.perf_counter()
    result = runner_mod.run_simple_regret_experiment(
        ground_truth=ground_truth,
        original_cost_per_arm=actual_cost_per_arm,
        variant=variant,
        args=args,
        prior_mean=float(prior_mean),
        prior_variance=float(prior_variance),
        max_evaluations=int(max_evaluations),
        max_original_cost=max_original_cost,
        run=None,
        log_step_metrics=False,
    )
    wall_s = float(time.perf_counter() - t0)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        trace_path,
        dataset_tag=dataset_tag,
        matrix=str(matrix),
        matrix_stem=matrix.stem,
        run_seed=int(run_seed),
        experiment_variant=variant_raw,
        policy_family=variant.policy_family,
        cost_mode=variant.cost_mode,
        batch_size=int(variant.batch_size),
        gittins_batch_size=int(variant.gittins_batch_size),
        cost_scaling_factor=float(variant.cost_scaling_factor),
        prior_type=variant.prior_type,
        prior_mean_resolved=float(prior_mean),
        prior_variance_resolved=float(prior_variance),
        prior_bucket="" if prior_bucket is None else str(prior_bucket),
        prior_source=str(prior_source),
        mmlu_task="" if mmlu_task is None else str(mmlu_task),
        mmlu_size_bucket="" if dataset_tag != "mmlu" else mmlu_size_bucket(n_examples),
        n_arms=int(n_arms),
        n_examples=int(n_examples),
        n_cells=int(n_cells),
        eval_budget_fraction=float(eval_budget_fraction),
        budget_max_evals=int(budget_evals),
        budget_original_cost=(-1.0 if max_original_cost is None else float(max_original_cost)),
        total_brute_force_original_cost=float(total_bf_cost),
        cost_vector=("" if cost_vector is None else str(cost_vector)),
        cost_per_arm_original=np.asarray(actual_cost_per_arm.numpy(), dtype=np.float64),
        total_wall_time_s=wall_s,
        x=np.asarray(result["x"], dtype=np.int64),
        x_original_cost=np.asarray(result["x_original_cost"], dtype=np.float64),
        regret=np.asarray(result["regret"], dtype=np.float64),
        recommended_arm=np.asarray(result["recommended_arm"], dtype=np.int64),
        recommended_mean=np.asarray(result["recommended_mean"], dtype=np.float64),
        pulled_arm=np.asarray(result["pulled_arm"], dtype=np.int64),
        gittins_stop_cum_eval=-1 if result.get("natural_stop_cum_eval") is None else int(result["natural_stop_cum_eval"]),
        gittins_stop_cum_original_cost=-1.0 if result.get("natural_stop_cum_original_cost") is None else float(result["natural_stop_cum_original_cost"]),
        gittins_recommendation_aware_stop_cum_eval=-1 if result.get("recommendation_aware_stop_cum_eval") is None else int(result["recommendation_aware_stop_cum_eval"]),
        gittins_recommendation_aware_stop_cum_original_cost=(
            -1.0
            if result.get("recommendation_aware_stop_cum_original_cost") is None
            else float(result["recommendation_aware_stop_cum_original_cost"])
        ),
    )
    print(f"Wrote trace: {trace_path}")
    return trace_path


def load_curve(trace: Path, *, x_axis: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(trace)
    x_key = "x_original_cost" if x_axis == "cost" else "x"
    return np.asarray(z[x_key], dtype=float), np.asarray(z["regret"], dtype=float)


def plot_comparison(
    *,
    traces: list[dict[str, Any]],
    out_png: Path,
    title: str,
    x_axis: str,
) -> list[dict[str, Any]]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    plt.figure(figsize=(8, 5))
    for item in traces:
        label = str(item["label"])
        trace = Path(item["trace"])
        x, y = load_curve(trace, x_axis=x_axis)
        if x.size == 0 or y.size == 0:
            print(f"Warning: empty curve for {label}: {trace}")
            continue
        plt.plot(x, y, linewidth=1.8, label=label)
        rows.append(
            {
                "label": label,
                "variant": item["variant"],
                "prior_role": item["prior_role"],
                "trace": str(trace),
                "final_simple_regret": float(y[-1]),
                "best_seen_regret": float(np.min(y)),
                "x_axis": x_axis,
                "final_x": float(x[-1]),
            }
        )

    if x_axis == "cost":
        plt.xlabel("Cumulative full-evaluation cost")
    else:
        plt.xlabel("Cumulative examples evaluated")
    plt.ylabel("Simple regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote comparison plot: {out_png}")
    return rows


def write_summary(summary_csv: Path, rows: list[dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matrix",
        "comparison",
        "dataset_tag",
        "mmlu_size_bucket",
        "label",
        "variant",
        "prior_role",
        "final_simple_regret",
        "best_seen_regret",
        "x_axis",
        "final_x",
        "plot",
        "trace",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"Wrote summary: {summary_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-root", type=Path, default=Path("data/synthetic_matrices"))
    p.add_argument("--runner", type=Path, default=Path("scripts/run_simple_regret_wandb.py"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/synthetic_local_runs_three_way"))
    p.add_argument("--plots-dir", type=Path, default=Path("outputs/synthetic_local_plots_three_way"))
    p.add_argument("--matrix-seed-dir", default=None, help="Optional seed directory filter, e.g. seed1")
    p.add_argument("--run-seed", type=int, default=0)
    p.add_argument("--eval-budget-fraction", type=float, default=0.1)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["unit", "cost"],
        default=["unit"],
        help="Which synthetic comparisons to run. Use: --modes unit cost",
    )
    p.add_argument(
        "--gsm8k-cost-vector",
        type=Path,
        default=Path("data_analysis/pricing/gsm8k_various_models_configurations_price_ratio_1to2_rounded.json"),
        help="Cost vector/config JSON for GSM8K cost-aware synthetic runs.",
    )
    p.add_argument(
        "--piqa-cost-vector",
        type=Path,
        default=Path("data_analysis/pricing/piqa_various_models_configurations_input_price.json"),
        help="Cost vector/config JSON for PIQA cost-aware synthetic runs.",
    )
    p.add_argument(
        "--mmlu-cost-vector",
        type=Path,
        default=Path("data_analysis/pricing/mmlu_prompt_eval_configurations_input_price.json"),
        help="Cost vector/config JSON for MMLU cost-aware synthetic runs, if needed.",
    )
    p.add_argument("--mmlu-task-metadata", type=Path, default=Path("data/MMLU_matrices/task_metadata.json"))
    p.add_argument("--skip-gsm8k", action="store_true")
    p.add_argument("--skip-piqa", action="store_true")
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--reuse-existing", action="store_true")
    return p.parse_args()


def filter_seed_dir(files: list[Path], seed_dir: str | None) -> list[Path]:
    if not seed_dir:
        return files
    return [p for p in files if p.parent.name == seed_dir]


def main() -> int:
    args = parse_args()
    if not args.runner.is_file():
        raise FileNotFoundError(f"Runner not found: {args.runner}")
    if not args.skip_mmlu and not args.mmlu_task_metadata.is_file():
        raise FileNotFoundError(f"MMLU task metadata not found: {args.mmlu_task_metadata}")

    runner_mod = load_runner_module(args.runner)
    all_rows: list[dict[str, Any]] = []

    def run_default_or_dataset_generated_setting(*, dataset_tag: str, setting_name: str) -> None:
        setting_root = args.synthetic_root / setting_name
        files = filter_seed_dir(sorted(setting_root.rglob("*.npy")), args.matrix_seed_dir)
        if not files:
            print(f"No {dataset_tag.upper()} synthetic matrices found under: {setting_root}")
            return

        matched_prior, mismatched_prior = matched_and_mismatched_priors_from_setting(setting_name)
        B_gittins = dataset_gittins_batch_size(dataset_tag)

        cost_vector_by_dataset = {
            "gsm8k": args.gsm8k_cost_vector,
            "piqa": args.piqa_cost_vector,
            "mmlu": args.mmlu_cost_vector,
        }

        for matrix in files:
            for mode in args.modes:
                is_cost = mode == "cost"
                cost_vector = cost_vector_by_dataset.get(dataset_tag) if is_cost else None
                if is_cost and (cost_vector is None or not cost_vector.is_file()):
                    raise FileNotFoundError(f"Cost vector for {dataset_tag} not found: {cost_vector}")

                prefix = "cost" if is_cost else "unit"
                x_axis = "cost" if is_cost else "evals"
                ucb_variant = f"ucb_cost_B{B_gittins}" if is_cost else f"ucb_B{B_gittins}"
                variants = [
                    {
                        "label": f"UCB-E {prefix} B{B_gittins}",
                        "variant": ucb_variant,
                        "prior_role": "prior_free_baseline",
                    },
                    {
                        "label": f"Gittins matched prior {prefix} B{B_gittins} ({matched_prior})",
                        "variant": f"gittins_{prefix}_B{B_gittins}_scale1e-4_{matched_prior}",
                        "prior_role": "matched",
                    },
                    {
                        "label": f"Gittins mismatched prior {prefix} B{B_gittins} ({mismatched_prior})",
                        "variant": f"gittins_{prefix}_B{B_gittins}_scale1e-4_{mismatched_prior}",
                        "prior_role": "mismatched",
                    },
                ]
                trace_items = []
                for item in variants:
                    trace = run_variant_local(
                        runner_mod=runner_mod,
                        dataset_tag=dataset_tag,
                        matrix=matrix,
                        variant_raw=item["variant"],
                        out_dir=args.out_dir,
                        run_seed=args.run_seed,
                        eval_budget_fraction=args.eval_budget_fraction,
                        mmlu_task_metadata=None,
                        cost_vector=cost_vector,
                        reuse_existing=args.reuse_existing,
                    )
                    trace_items.append({**item, "trace": trace})

                plot_path = args.plots_dir / f"{matrix.stem}__{prefix}__ucbB{B_gittins}_gittinsB{B_gittins}_matched_mismatched.png"
                rows = plot_comparison(
                    traces=trace_items,
                    out_png=plot_path,
                    x_axis=x_axis,
                    title=(
                        f"{dataset_tag.upper()} synthetic ({setting_name}, {prefix}): "
                        f"UCB B{B_gittins} vs Gittins B{B_gittins} matched/mismatched\n{matrix.name}"
                    ),
                )
                for row in rows:
                    row.update(
                        {
                            "matrix": str(matrix),
                            "comparison": f"{setting_name}_{prefix}_three_way",
                            "dataset_tag": dataset_tag,
                            "mmlu_size_bucket": "",
                            "plot": str(plot_path),
                        }
                    )
                    all_rows.append(row)

    if not args.skip_gsm8k:
        # Current generated folder is usually gsm8k_default. If a gsm8k_dataset folder
        # exists later, this script will also handle it.
        for setting_name in ["gsm8k_default", "gsm8k_dataset"]:
            if (args.synthetic_root / setting_name).exists():
                run_default_or_dataset_generated_setting(dataset_tag="gsm8k", setting_name=setting_name)
        if not (args.synthetic_root / "gsm8k_default").exists() and not (args.synthetic_root / "gsm8k_dataset").exists():
            print(f"No GSM8K synthetic setting folders found under: {args.synthetic_root}")

    if not args.skip_piqa:
        # PIQA support is included for future/optional synthetic folders. If you have not
        # generated PIQA synthetic matrices yet, this will simply print a message and skip.
        any_piqa = False
        for setting_name in ["piqa_default", "piqa_dataset", "piqa"]:
            if (args.synthetic_root / setting_name).exists():
                any_piqa = True
                run_default_or_dataset_generated_setting(dataset_tag="piqa", setting_name=setting_name)
        if not any_piqa:
            print(f"No PIQA synthetic setting folders found under: {args.synthetic_root}")

    if not args.skip_mmlu:
        mmlu_root = args.synthetic_root / "mmlu_high"
        mmlu_files = filter_seed_dir(sorted(mmlu_root.rglob("*.npy")), args.matrix_seed_dir)
        if not mmlu_files:
            print(f"No MMLU high synthetic matrices found under: {mmlu_root}")
        for matrix in mmlu_files:
            arr = np.load(matrix, mmap_mode="r")
            n_examples = int(arr.shape[1])
            B_gittins = dataset_gittins_batch_size("mmlu", n_examples=n_examples)
            bucket = mmlu_size_bucket(n_examples)
            variants = [
                {
                    "label": f"UCB-E B{B_gittins}",
                    "variant": f"ucb_B{B_gittins}",
                    "prior_role": "prior_free_baseline",
                },
                {
                    "label": f"Gittins matched prior B{B_gittins} (dataset/high)",
                    "variant": f"gittins_unit_B{B_gittins}_scale1e-4_dataset",
                    "prior_role": "matched",
                },
                {
                    "label": f"Gittins mismatched prior B{B_gittins} (default)",
                    "variant": f"gittins_unit_B{B_gittins}_scale1e-4_default",
                    "prior_role": "mismatched",
                },
            ]
            trace_items = []
            for item in variants:
                trace = run_variant_local(
                    runner_mod=runner_mod,
                    dataset_tag="mmlu",
                    matrix=matrix,
                    variant_raw=item["variant"],
                    out_dir=args.out_dir,
                    run_seed=args.run_seed,
                    eval_budget_fraction=args.eval_budget_fraction,
                    mmlu_task_metadata=args.mmlu_task_metadata,
                    cost_vector=None,
                    reuse_existing=args.reuse_existing,
                )
                trace_items.append({**item, "trace": trace})

            plot_path = args.plots_dir / f"{matrix.stem}__ucbB{B_gittins}_gittinsB{B_gittins}_matched_mismatched.png"
            rows = plot_comparison(
                traces=trace_items,
                out_png=plot_path,
                x_axis="evals",
                title=(
                    f"MMLU high synthetic ({bucket}): "
                    f"UCB B{B_gittins} vs Gittins B{B_gittins} matched/mismatched\n{matrix.name}"
                ),
            )
            for row in rows:
                row.update(
                    {
                        "matrix": str(matrix),
                        "comparison": "mmlu_high_generated_three_way",
                        "dataset_tag": "mmlu",
                        "mmlu_size_bucket": bucket,
                        "plot": str(plot_path),
                    }
                )
                all_rows.append(row)

    write_summary(args.plots_dir / "summary.csv", all_rows)
    print()
    print("Done.")
    print(f"Comparison plots directory: {args.plots_dir}")
    print(f"Summary CSV: {args.plots_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
