#!/usr/bin/env python3
"""Merge Gittins-G/S W&B exports with local Gittins-EB JSON and plot the pilot."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Gittins-G", "Gittins-S", "Gittins-EB")
STYLE = {
    "Gittins-G": {"color": "tab:green", "lw": 2.0},
    "Gittins-S": {"color": "tab:orange", "lw": 2.0},
    "Gittins-EB": {"color": "tab:blue", "lw": 2.2},
}
MMLU_TASKS = (
    "abstract_algebra",
    "virology",
    "marketing",
    "security_studies",
    "computer_security",
    "high_school_government_and_politics",
    "prehistory",
    "management",
)
COST_FILES = {
    "gsm8k": REPO_ROOT
    / "data_analysis/pricing/gsm8k_various_models_configurations_price_ratio_1to2_rounded.json",
    "piqa": REPO_ROOT
    / "data_analysis/pricing/piqa_various_models_configurations_input_price.json",
    "mmlu": REPO_ROOT
    / "data_analysis/pricing/mmlu_prompt_eval_configurations_input_price.json",
}


@dataclass
class RunCurve:
    run_id: str
    dataset: str
    benchmark: str
    mode: str
    method: str
    x: np.ndarray
    y: np.ndarray


def load_cost_sum(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        values = data
    elif "cost_per_arm" in data:
        values = data["cost_per_arm"]
    else:
        values = [
            row.get("estimated_cost_per_1m_input_tokens", row.get("cost"))
            for _, row in sorted(data.items(), key=lambda item: int(item[0]))
        ]
    return float(np.asarray(values, dtype=float).sum())


COST_SUMS = {dataset: load_cost_sum(path) for dataset, path in COST_FILES.items()}
MMLU_BUCKET_PRIOR_MEAN = {"low": 0.4, "medium": 0.6, "high": 0.75}
MMLU_METADATA = json.loads(
    (REPO_ROOT / "data/MMLU_matrices/task_metadata.json").read_text(encoding="utf-8")
)
MMLU_PRIOR_MEANS = {
    str(row["task"]): MMLU_BUCKET_PRIOR_MEAN[str(row["dataset_prior_bucket"])]
    for row in MMLU_METADATA["tasks"]
    if str(row.get("dataset_prior_bucket", "")) in MMLU_BUCKET_PRIOR_MEAN
}


def clean_series(x: np.ndarray, y: np.ndarray, budget: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float) / float(budget)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 1.0 + 1e-12)
    x, y = x[keep], y[keep]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, indices = np.unique(x, return_index=True)
    last_indices = np.r_[indices[1:] - 1, x.size - 1]
    return unique_x, y[last_indices]


def method_from_variant(variant: str) -> str:
    if variant.endswith("_dataset"):
        return "Gittins-S"
    if variant.endswith("_default"):
        return "Gittins-G"
    if variant.endswith("_ebwarm"):
        return "Gittins-EB"
    raise ValueError(f"unrecognized Gittins variant: {variant}")


def benchmark_from_matrix(dataset: str, matrix: str) -> str:
    stem = Path(matrix).stem
    if dataset == "mmlu":
        return stem
    match = re.search(r"seed(\d+)", stem)
    return f"{dataset}_seed{match.group(1) if match else 'NA'}"


def load_baseline_curves(root: Path) -> list[RunCurve]:
    curves: list[RunCurve] = []
    for key in ("gsm8k", "piqa", "mmlu_small", "mmlu_medium"):
        path = root / key / "runs_history.csv.gz"
        frame = pd.read_csv(
            path,
            usecols=lambda col: col
            in {
                "run_id",
                "dataset_tag_resolved",
                "dataset_tag",
                "matrix",
                "mmlu_task",
                "n_examples",
                "budget_max_evals",
                "experiment_variant",
                "cum_eval",
                "cum_original_cost",
                "simple_regret",
            },
        )
        dataset_series = frame.get("dataset_tag_resolved", frame.get("dataset_tag"))
        frame["_dataset"] = dataset_series.astype(str).str.lower()
        for run_id, group in frame.groupby("run_id", sort=False):
            first = group.iloc[0]
            dataset = str(first["_dataset"])
            variant = str(first["experiment_variant"])
            mode = "cost" if "_cost_" in variant else "unit"
            matrix = str(first["matrix"])
            benchmark = (
                str(first.get("mmlu_task"))
                if dataset == "mmlu" and pd.notna(first.get("mmlu_task"))
                else benchmark_from_matrix(dataset, matrix)
            )
            if mode == "unit":
                budget = float(first["budget_max_evals"])
                x_raw = group["cum_eval"].to_numpy(dtype=float)
            else:
                budget = (
                    0.1 * float(first["n_examples"]) * float(COST_SUMS[dataset])
                )
                x_raw = group["cum_original_cost"].to_numpy(dtype=float)
            x, y = clean_series(x_raw, group["simple_regret"].to_numpy(dtype=float), budget)
            if x.size:
                curves.append(
                    RunCurve(
                        str(run_id),
                        dataset,
                        benchmark,
                        mode,
                        method_from_variant(variant),
                        x,
                        y,
                    )
                )
    return curves


def load_eb_curves(root: Path) -> tuple[list[RunCurve], list[dict[str, object]]]:
    curves: list[RunCurve] = []
    priors: list[dict[str, object]] = []
    seen_prior_keys: set[tuple[str, int]] = set()
    for path in sorted(root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        summary = payload["summary"]
        history = payload["history"]
        dataset = str(config["dataset_tag_resolved"])
        matrix = str(config["matrix"])
        benchmark = (
            str(config["mmlu_task"])
            if dataset == "mmlu"
            else benchmark_from_matrix(dataset, matrix)
        )
        mode = "cost" if bool(config["cost_aware_run"]) else "unit"
        budget = (
            float(config["budget_original_cost"])
            if mode == "cost"
            else float(config["budget_max_evals"])
        )
        x_raw = history["x_original_cost"] if mode == "cost" else history["x"]
        x, y = clean_series(x_raw, history["regret"], budget)
        curves.append(
            RunCurve(
                path.stem,
                dataset,
                benchmark,
                mode,
                "Gittins-EB",
                x,
                y,
            )
        )

        prior_key = (matrix, int(summary["run_seed"]))
        if prior_key in seen_prior_keys:
            continue
        seen_prior_keys.add(prior_key)
        dataset_prior = (
            0.2
            if dataset == "gsm8k"
            else 0.4
            if dataset == "piqa"
            else MMLU_PRIOR_MEANS[benchmark]
        )
        for arm, true_mean in enumerate(payload["true_arm_means"]):
            priors.append(
                {
                    "dataset": dataset,
                    "benchmark": benchmark,
                    "matrix": matrix,
                    "run_seed": int(summary["run_seed"]),
                    "arm": arm,
                    "fixed_prior_mean": 0.5,
                    "gittins_g_fixed_prior_mean": 0.5,
                    "gittins_s_fixed_prior_mean": dataset_prior,
                    "estimated_prior_mean": float(summary["prior_mean_resolved"]),
                    "gittins_eb_estimated_prior_mean": float(
                        summary["prior_mean_resolved"]
                    ),
                    "true_matrix_arm_mean": float(true_mean),
                }
            )
    return curves, priors


def load_stop_records(baseline_root: Path, eb_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for key in ("gsm8k", "piqa", "mmlu_small", "mmlu_medium"):
        frame = pd.read_csv(baseline_root / key / "runs_summary.csv")
        for _, row in frame.iterrows():
            dataset = str(
                row.get("dataset_tag_resolved")
                if pd.notna(row.get("dataset_tag_resolved"))
                else row.get("dataset_tag")
            ).lower()
            variant = str(row["experiment_variant"])
            mode = "cost" if "_cost_" in variant else "unit"
            matrix = str(row["matrix"])
            benchmark = (
                str(row.get("mmlu_task"))
                if dataset == "mmlu" and pd.notna(row.get("mmlu_task"))
                else benchmark_from_matrix(dataset, matrix)
            )
            if mode == "unit":
                budget = float(row["budget_max_evals"])
                stop = row.get("gittins_stop_cum_eval")
            else:
                budget = 0.1 * float(row["n_examples"]) * COST_SUMS[dataset]
                stop = row.get("gittins_stop_cum_original_cost")
            stop_fraction = (
                float(np.clip(float(stop) / budget, 0.0, 1.0))
                if pd.notna(stop)
                else float("nan")
            )
            records.append(
                {
                    "dataset": dataset,
                    "benchmark": benchmark,
                    "mode": mode,
                    "method": method_from_variant(variant),
                    "stop_fraction": stop_fraction,
                }
            )

    for path in sorted(eb_root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        summary = payload["summary"]
        dataset = str(config["dataset_tag_resolved"])
        matrix = str(config["matrix"])
        benchmark = (
            str(config["mmlu_task"])
            if dataset == "mmlu"
            else benchmark_from_matrix(dataset, matrix)
        )
        mode = "cost" if bool(config["cost_aware_run"]) else "unit"
        if mode == "unit":
            budget = float(config["budget_max_evals"])
            stop = summary["gittins_stop_cum_eval"]
        else:
            budget = float(config["budget_original_cost"])
            stop = summary["gittins_stop_cum_original_cost"]
        stop_fraction = (
            float(np.clip(float(stop) / budget, 0.0, 1.0))
            if stop is not None
            else float("nan")
        )
        records.append(
            {
                "dataset": dataset,
                "benchmark": benchmark,
                "mode": mode,
                "method": "Gittins-EB",
                "stop_fraction": stop_fraction,
            }
        )
    return records


def aggregate_stops(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    overall = frame[frame["dataset"] != "mmlu"].copy()
    overall["benchmark"] = overall["dataset"].astype(str) + "_all"
    frame = pd.concat([frame, overall], ignore_index=True)
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(
        ["dataset", "benchmark", "mode", "method"], sort=True
    ):
        values = pd.to_numeric(group["stop_fraction"], errors="coerce").dropna()
        n_stops = int(values.size)
        mean = float(values.mean()) if n_stops else float("nan")
        se = (
            float(values.std(ddof=1) / math.sqrt(n_stops))
            if n_stops > 1
            else 0.0
            if n_stops == 1
            else float("nan")
        )
        rows.append(
            {
                "dataset": key[0],
                "benchmark": key[1],
                "mode": key[2],
                "method": key[3],
                "n_runs": int(len(group)),
                "n_stops": n_stops,
                "mean_stop_budget_fraction": mean,
                "stop_SE": se,
            }
        )
    return pd.DataFrame(rows)


def run_auc(curve: RunCurve, grid: np.ndarray) -> float:
    values = np.interp(grid, curve.x, curve.y, left=curve.y[0], right=curve.y[-1])
    return float(np.trapezoid(values, grid))


def aggregate(
    curves: list[RunCurve], grid_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = np.linspace(0.0, 1.0, grid_size)
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, str], list[RunCurve]] = {}
    for curve in curves:
        key = (curve.dataset, curve.benchmark, curve.mode, curve.method)
        grouped.setdefault(key, []).append(curve)
        if curve.dataset != "mmlu":
            overall_key = (
                curve.dataset,
                f"{curve.dataset}_all",
                curve.mode,
                curve.method,
            )
            grouped.setdefault(overall_key, []).append(curve)

    for (dataset, benchmark, mode, method), runs in sorted(grouped.items()):
        values = np.vstack(
            [
                np.interp(grid, run.x, run.y, left=np.nan, right=run.y[-1])
                for run in runs
            ]
        )
        n = np.sum(np.isfinite(values), axis=0)
        mean = np.divide(
            np.nansum(values, axis=0),
            n,
            out=np.full(grid.shape, np.nan, dtype=float),
            where=n > 0,
        )
        centered = np.where(np.isfinite(values), values - mean, 0.0)
        variance = np.divide(
            np.sum(centered * centered, axis=0),
            np.maximum(n - 1, 1),
        )
        se = np.sqrt(variance) / np.sqrt(np.maximum(n, 1))
        se[n <= 1] = 0.0
        for x_value, mean_value, se_value, n_value in zip(
            grid, mean, se, n, strict=True
        ):
            if n_value == 0:
                continue
            curve_rows.append(
                {
                    "dataset": dataset,
                    "benchmark": benchmark,
                    "mode": mode,
                    "method": method,
                    "budget_fraction": x_value,
                    "mean": mean_value,
                    "SE": se_value,
                    "n": int(n_value),
                }
            )

        finals = np.asarray([run.y[-1] for run in runs], dtype=float)
        aucs = np.asarray([run_auc(run, grid) for run in runs], dtype=float)
        final_se = (
            float(np.std(finals, ddof=1) / math.sqrt(finals.size))
            if finals.size > 1
            else 0.0
        )
        auc_se = (
            float(np.std(aucs, ddof=1) / math.sqrt(aucs.size))
            if aucs.size > 1
            else 0.0
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "benchmark": benchmark,
                "mode": mode,
                "method": method,
                "n_runs": int(finals.size),
                "mean": float(np.mean(finals)),
                "SE": final_se,
                "final_regret": float(np.mean(finals)),
                "final_regret_SE": final_se,
                "normalized_AUC": float(np.mean(aucs)),
                "normalized_AUC_SE": auc_se,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def draw_panel(
    ax: plt.Axes, frame: pd.DataFrame, stops: pd.DataFrame, title: str
) -> None:
    for method in METHODS:
        method_frame = frame[frame["method"] == method].sort_values("budget_fraction")
        if method_frame.empty:
            continue
        x = method_frame["budget_fraction"].to_numpy(dtype=float) * 100.0
        mean = method_frame["mean"].to_numpy(dtype=float)
        se = method_frame["SE"].to_numpy(dtype=float)
        style = STYLE[method]
        ax.plot(x, mean, label=method, color=style["color"], lw=style["lw"])
        ax.fill_between(x, mean - se, mean + se, color=style["color"], alpha=0.18)
        method_stops = stops[stops["method"] == method]
        if not method_stops.empty:
            stop = method_stops.iloc[0]
            center = float(stop["mean_stop_budget_fraction"]) * 100.0
            stop_se = float(stop["stop_SE"]) * 100.0
            if np.isfinite(center):
                ax.axvspan(
                    max(0.0, center - stop_se),
                    min(100.0, center + stop_se),
                    color=style["color"],
                    alpha=0.08,
                    linewidth=0,
                )
                ax.axvline(
                    center,
                    color=style["color"],
                    linestyle="--",
                    lw=1.35,
                    alpha=0.9,
                )
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Budget used (%)")
    ax.set_ylabel("Simple regret")
    ax.grid(alpha=0.25)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gsm_piqa(
    curves: pd.DataFrame, stops: pd.DataFrame, out_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, dataset in enumerate(("gsm8k", "piqa")):
        for col, mode in enumerate(("unit", "cost")):
            frame = curves[
                (curves["dataset"] == dataset)
                & (curves["benchmark"] == f"{dataset}_all")
                & (curves["mode"] == mode)
            ]
            stop_frame = stops[
                (stops["dataset"] == dataset)
                & (stops["benchmark"] == f"{dataset}_all")
                & (stops["mode"] == mode)
            ]
            draw_panel(
                axes[row, col],
                frame,
                stop_frame,
                f"{dataset.upper()} — {'unit-cost' if mode == 'unit' else 'cost-aware'}",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="black", linestyle="--", lw=1.35)
    )
    labels.append("Stopping: vertical dashed mean, band ±1 SE")
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out_dir, "gsm8k_piqa_unit_cost_comparison")


def plot_mmlu(curves: pd.DataFrame, stops: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    for task_index, task in enumerate(MMLU_TASKS):
        row = task_index // 2
        base_col = (task_index % 2) * 2
        for mode_index, mode in enumerate(("unit", "cost")):
            frame = curves[
                (curves["dataset"] == "mmlu")
                & (curves["benchmark"] == task)
                & (curves["mode"] == mode)
            ]
            stop_frame = stops[
                (stops["dataset"] == "mmlu")
                & (stops["benchmark"] == task)
                & (stops["mode"] == mode)
            ]
            draw_panel(
                axes[row, base_col + mode_index],
                frame,
                stop_frame,
                f"{task.replace('_', ' ')} — {'unit' if mode == 'unit' else 'cost-aware'}",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="black", linestyle="--", lw=1.35)
    )
    labels.append("Stopping: vertical dashed mean, band ±1 SE")
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out_dir, "mmlu_8tasks_unit_cost_comparison")

    for task in MMLU_TASKS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for index, mode in enumerate(("unit", "cost")):
            frame = curves[
                (curves["dataset"] == "mmlu")
                & (curves["benchmark"] == task)
                & (curves["mode"] == mode)
            ]
            stop_frame = stops[
                (stops["dataset"] == "mmlu")
                & (stops["benchmark"] == task)
                & (stops["mode"] == mode)
            ]
            draw_panel(
                axes[index],
                frame,
                stop_frame,
                f"{task.replace('_', ' ')} — {'unit' if mode == 'unit' else 'cost-aware'}",
            )
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(
            Line2D([0], [0], color="black", linestyle="--", lw=1.35)
        )
        labels.append("Stopping: vertical dashed mean, band ±1 SE")
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(fig, out_dir, f"mmlu_{task}_unit_cost_comparison")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gittins_eb_pilot_baselines",
    )
    parser.add_argument(
        "--eb-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gittins_eb_pilot_runs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/figure/new_figure/final/gittins_eb_pilot_comparison",
    )
    parser.add_argument("--grid-size", type=int, default=201)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_curves = load_baseline_curves(args.baseline_root)
    eb_curves, prior_rows = load_eb_curves(args.eb_root)
    stop_frame = aggregate_stops(load_stop_records(args.baseline_root, args.eb_root))
    all_curves = baseline_curves + eb_curves
    curve_frame, summary_frame = aggregate(all_curves, args.grid_size)
    curve_frame.to_csv(args.out_dir / "regret_curves_mean_se.csv", index=False)
    summary_frame.to_csv(args.out_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(args.out_dir / "prior_comparison.csv", index=False)
    stop_frame.to_csv(args.out_dir / "stopping_summary.csv", index=False)
    plot_gsm_piqa(curve_frame, stop_frame, args.out_dir)
    plot_mmlu(curve_frame, stop_frame, args.out_dir)
    print(
        json.dumps(
            {
                "baseline_runs": len(baseline_curves),
                "eb_runs": len(eb_curves),
                "summary_rows": len(summary_frame),
                "prior_rows": len(prior_rows),
                "stopping_rows": len(stop_frame),
                "out_dir": str(args.out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
