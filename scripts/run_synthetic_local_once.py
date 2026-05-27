#!/usr/bin/env python3
"""
Run local synthetic simple-regret experiments once per matrix and plot comparisons.

No W&B upload:
    every call uses --wandb-mode disabled

Comparisons:
1. GSM8K synthetic:
   UCB_B16 vs Gittins default prior

2. MMLU high/easy synthetic:
   Gittins default prior vs Gittins dataset/easy prior

This is a lightweight local sanity run:
    one run_seed per matrix
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def mmlu_batch_size(n_examples: int) -> int:
    """
    Match our MMLU size-bucket batch-size convention:
      small:  n_examples <= 150       -> B=4
      medium: 151 <= n_examples <=400 -> B=8
      large:  n_examples > 400        -> B=16
    """
    if int(n_examples) <= 150:
        return 4
    if int(n_examples) <= 400:
        return 8
    return 16


def run_variant(
    *,
    runner: Path,
    dataset_tag: str,
    matrix: Path,
    variant: str,
    out_dir: Path,
    run_seed: int,
    eval_budget_fraction: float,
    mmlu_task_metadata: Path | None = None,
) -> Path:
    cmd = [
        sys.executable,
        str(runner),
        "--dataset-tag",
        dataset_tag,
        "--matrix",
        str(matrix),
        "--experiment-variant",
        variant,
        "--run-seed",
        str(run_seed),
        "--eval-budget-fraction",
        str(eval_budget_fraction),
        "--wandb-mode",
        "disabled",
        "--out-dir",
        str(out_dir),
    ]

    if dataset_tag == "mmlu" and mmlu_task_metadata is not None:
        cmd += ["--mmlu-task-metadata", str(mmlu_task_metadata)]

    print()
    print("Running:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    trace = (
        out_dir
        / dataset_tag
        / safe_token(variant)
        / f"{matrix.stem}__runseed{run_seed}__{safe_token(variant)}_traces.npz"
    )

    if not trace.is_file():
        raise FileNotFoundError(f"Expected trace not found: {trace}")

    return trace


def load_curve(trace: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(trace)
    return np.asarray(z["x"], dtype=float), np.asarray(z["regret"], dtype=float)


def plot_comparison(
    *,
    traces: list[tuple[str, Path]],
    out_png: Path,
    title: str,
) -> list[dict]:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    plt.figure(figsize=(8, 5))

    for label, trace in traces:
        x, y = load_curve(trace)

        if x.size == 0 or y.size == 0:
            print(f"Warning: empty curve for {label}: {trace}")
            continue

        plt.plot(x, y, linewidth=1.8, label=label)

        rows.append(
            {
                "label": label,
                "trace": str(trace),
                "final_simple_regret": float(y[-1]),
                "best_seen_regret": float(np.min(y)),
                "final_cum_eval": int(x[-1]),
            }
        )

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


def write_summary(summary_csv: Path, rows: list[dict]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "matrix",
        "comparison",
        "label",
        "final_simple_regret",
        "best_seen_regret",
        "final_cum_eval",
        "plot",
        "trace",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote summary: {summary_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path("data/synthetic_matrices"),
        help="Root directory containing generated synthetic matrices.",
    )
    p.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_simple_regret_wandb.py"),
        help="Path to local runner script.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/synthetic_local_runs"),
        help="Directory for traces/meta/single-run figures.",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("outputs/synthetic_local_plots"),
        help="Directory for comparison plots and summary.csv.",
    )
    p.add_argument(
        "--run-seed",
        type=int,
        default=0,
        help="Algorithm run seed.",
    )
    p.add_argument(
        "--eval-budget-fraction",
        type=float,
        default=0.1,
        help="Evaluation budget fraction.",
    )
    p.add_argument(
        "--mmlu-task-metadata",
        type=Path,
        default=Path("data/MMLU_matrices/task_metadata.json"),
        help="MMLU task metadata JSON for resolving dataset priors.",
    )

    p.add_argument(
        "--skip-gsm8k",
        action="store_true",
        help="Skip GSM8K synthetic comparison.",
    )
    p.add_argument(
        "--skip-mmlu",
        action="store_true",
        help="Skip MMLU high/easy synthetic comparisons.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.runner.is_file():
        raise FileNotFoundError(f"Runner not found: {args.runner}")

    if not args.skip_mmlu and not args.mmlu_task_metadata.is_file():
        raise FileNotFoundError(
            f"MMLU task metadata not found: {args.mmlu_task_metadata}\n"
            "Expected path: data/MMLU_matrices/task_metadata.json"
        )

    all_summary_rows: list[dict] = []

    # ------------------------------------------------------------------
    # 1. GSM8K synthetic: UCB-E vs Gittins default prior
    # ------------------------------------------------------------------
    if not args.skip_gsm8k:
        gsm8k_root = args.synthetic_root / "gsm8k_default"
        gsm8k_files = sorted(gsm8k_root.rglob("*.npy"))

        if not gsm8k_files:
            print(f"No GSM8K synthetic matrices found under: {gsm8k_root}")

        for matrix in gsm8k_files:
            variant_ucb = "ucb_B16"
            variant_gittins = "gittins_unit_B16_scale1e-4_default"

            trace_ucb = run_variant(
                runner=args.runner,
                dataset_tag="gsm8k",
                matrix=matrix,
                variant=variant_ucb,
                out_dir=args.out_dir,
                run_seed=args.run_seed,
                eval_budget_fraction=args.eval_budget_fraction,
            )

            trace_gittins = run_variant(
                runner=args.runner,
                dataset_tag="gsm8k",
                matrix=matrix,
                variant=variant_gittins,
                out_dir=args.out_dir,
                run_seed=args.run_seed,
                eval_budget_fraction=args.eval_budget_fraction,
            )

            plot_path = args.plots_dir / f"{matrix.stem}__ucb_vs_gittins.png"
            rows = plot_comparison(
                traces=[
                    ("UCB-E B16", trace_ucb),
                    ("Gittins default B16", trace_gittins),
                ],
                out_png=plot_path,
                title=f"GSM8K synthetic: UCB-E vs Gittins\n{matrix.name}",
            )

            for row in rows:
                row.update(
                    {
                        "matrix": str(matrix),
                        "comparison": "gsm8k_ucb_vs_gittins",
                        "plot": str(plot_path),
                    }
                )
                all_summary_rows.append(row)

    # ------------------------------------------------------------------
    # 2. MMLU high/easy synthetic: default prior vs dataset/easy prior
    # ------------------------------------------------------------------
    if not args.skip_mmlu:
        mmlu_root = args.synthetic_root / "mmlu_high"
        mmlu_files = sorted(mmlu_root.rglob("*.npy"))

        if not mmlu_files:
            print(f"No MMLU high synthetic matrices found under: {mmlu_root}")

        for matrix in mmlu_files:
            arr = np.load(matrix, mmap_mode="r")
            n_examples = int(arr.shape[1])
            B = mmlu_batch_size(n_examples)

            variant_default = f"gittins_unit_B{B}_scale1e-4_default"
            variant_dataset = f"gittins_unit_B{B}_scale1e-4_dataset"

            trace_default = run_variant(
                runner=args.runner,
                dataset_tag="mmlu",
                matrix=matrix,
                variant=variant_default,
                out_dir=args.out_dir,
                run_seed=args.run_seed,
                eval_budget_fraction=args.eval_budget_fraction,
                mmlu_task_metadata=args.mmlu_task_metadata,
            )

            trace_dataset = run_variant(
                runner=args.runner,
                dataset_tag="mmlu",
                matrix=matrix,
                variant=variant_dataset,
                out_dir=args.out_dir,
                run_seed=args.run_seed,
                eval_budget_fraction=args.eval_budget_fraction,
                mmlu_task_metadata=args.mmlu_task_metadata,
            )

            plot_path = args.plots_dir / f"{matrix.stem}__gittins_default_vs_dataset.png"
            rows = plot_comparison(
                traces=[
                    (f"Gittins default prior B{B}", trace_default),
                    (f"Gittins dataset/easy prior B{B}", trace_dataset),
                ],
                out_png=plot_path,
                title=f"MMLU high synthetic: default prior vs dataset/easy prior\n{matrix.name}",
            )

            for row in rows:
                row.update(
                    {
                        "matrix": str(matrix),
                        "comparison": "mmlu_default_vs_dataset_prior",
                        "plot": str(plot_path),
                    }
                )
                all_summary_rows.append(row)

    write_summary(args.plots_dir / "summary.csv", all_summary_rows)

    print()
    print("Done.")
    print(f"Comparison plots directory: {args.plots_dir}")
    print(f"Summary CSV: {args.plots_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())