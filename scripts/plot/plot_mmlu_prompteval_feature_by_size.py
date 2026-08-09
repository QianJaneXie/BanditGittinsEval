"""Plot MMLU PromptEval-BAI feature-block comparison by task size bucket.

This is a lightweight plotting helper for comparing the four PromptEval
feature blocks separately on Small/Medium/Large MMLU tasks. It reads the
existing PromptEval result npy files and does not modify any input data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_mmlu_easy_task_panels_bo5pct import TASK_SIZE


FEATURES = [
    ("One-hot", "tab:cyan"),
    ("Discrete", "tab:blue"),
    ("Sentence-T PCA", "tab:purple"),
    ("FT-BERT PCA", "tab:brown"),
]

SIZE_ORDER = [("S", "Small"), ("M", "Medium"), ("L", "Large")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            r"outputs\prompteval_downloads\wallclock_mmlu_gsm8k_piqa_20260807\results\formal\mmlu"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"outputs\figure\new_figure\final\prompteval_feature_by_size"),
    )
    p.add_argument("--stem", default="mmlu_prompteval_feature_block_by_size")
    p.add_argument("--x-right", type=float, default=10.0)
    p.add_argument(
        "--unit-only",
        action="store_true",
        help="Draw only the Unit-cost column, useful when raw cost-aware four-feature files are unavailable.",
    )
    return p.parse_args()


def raw_result_path(results_dir: Path, task: str, mode: str) -> Path:
    suffix = "combined" if mode == "unit" else "costaware_combined"
    return results_dir / f"bai_results_MMLU_{task}_{suffix}.npy"


def processed_result_path(results_dir: Path, task: str, mode: str) -> Path:
    suffix = "combined" if mode == "unit" else "costaware_combined"
    return results_dir / f"bai_processed_results_MMLU_{task}_{suffix}.npy"


def load_processed_feature_curves(path: Path) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    payload = np.load(path, allow_pickle=True).item()
    arr = np.asarray(payload["curves"], dtype=float)
    if arr.ndim != 5 or arr.shape[2] < len(FEATURES) or arr.shape[3] != 2:
        raise ValueError(f"Unexpected processed curve shape at {path}: {arr.shape}")
    curves_by_feature: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in FEATURES
    ]
    for feature_idx in range(len(FEATURES)):
        y = np.asarray(arr[0, 0, feature_idx, 0], dtype=float)
        x = np.asarray(arr[0, 0, feature_idx, 1], dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size == 0:
            continue
        x = x / float(x[-1]) * 10.0
        curves_by_feature[feature_idx].append((x, y))
    return curves_by_feature


def load_feature_curves(path: Path, mode: str) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    payload = np.load(path, allow_pickle=True).item()
    out = payload["out"]
    curves_by_feature: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in FEATURES
    ]
    for seed_out in out:
        for feature_idx, records in enumerate(seed_out[0]):
            xs: list[float] = []
            ys: list[float] = []
            for record in records:
                x_key = "budget_obs" if mode == "unit" else "budget_cost"
                x = float(record[x_key])
                y = float(record["simple_regret"])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
            if xs:
                x_arr = np.asarray(xs, dtype=float)
                y_arr = np.asarray(ys, dtype=float)
                # The x-axis in the original feature comparison is percentage
                # of that PromptEval run's own final 10% budget.
                x_arr = x_arr / float(x_arr[-1]) * 10.0
                curves_by_feature[feature_idx].append((x_arr, y_arr))
    return curves_by_feature


def aggregate_step(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sorted({float(x) for x_arr, _ in curves for x in x_arr})
    x_grid = np.asarray(xs, dtype=float)
    values: list[np.ndarray] = []
    for x_arr, y_arr in curves:
        mapped = []
        for x in x_grid:
            idx = np.searchsorted(x_arr, x, side="right") - 1
            idx = max(0, min(idx, len(y_arr) - 1))
            mapped.append(float(y_arr[idx]))
        values.append(np.asarray(mapped, dtype=float))
    arr = np.vstack(values)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    return x_grid, mean, se


def aggregate_linear(
    curves: list[tuple[np.ndarray, np.ndarray]], grid_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_x = max(float(np.nanmin(x_arr)) for x_arr, _ in curves)
    max_x = min(float(np.nanmax(x_arr)) for x_arr, _ in curves)
    x_grid = np.linspace(min_x, max_x, int(grid_size))
    values = []
    for x_arr, y_arr in curves:
        values.append(np.interp(x_grid, x_arr, y_arr))
    arr = np.vstack(values)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    return x_grid, mean, se


def collect_by_size(results_dir: Path, mode: str) -> dict[str, list[list[tuple[np.ndarray, np.ndarray]]]]:
    grouped: dict[str, list[list[tuple[np.ndarray, np.ndarray]]]] = {
        size: [[] for _ in FEATURES] for size, _ in SIZE_ORDER
    }
    for task, size in TASK_SIZE.items():
        processed_path = processed_result_path(results_dir, task, mode)
        raw_path = raw_result_path(results_dir, task, mode)
        if processed_path.is_file():
            task_curves = load_processed_feature_curves(processed_path)
        elif raw_path.is_file():
            task_curves = load_feature_curves(raw_path, mode)
        else:
            continue
        for feature_idx, curves in enumerate(task_curves):
            grouped[size][feature_idx].extend(curves)
    return grouped


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    modes = [("unit", "Unit-cost")] if args.unit_only else [("unit", "Unit-cost"), ("aware", "Cost-aware")]
    by_mode = {mode: collect_by_size(args.results_dir, mode) for mode, _ in modes}

    ncols = len(modes)
    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(9.6 if args.unit_only else 12.8, 11.4),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for row, (size, size_label) in enumerate(SIZE_ORDER):
        for col, (mode, title) in enumerate(modes):
            ax = axes[row, col]
            for feature_idx, (label, color) in enumerate(FEATURES):
                curves = by_mode[mode][size][feature_idx]
                if not curves:
                    continue
                x, mean, se = aggregate_step(curves)
                drawstyle = "steps-post"
                fill_kwargs = {"step": "post"}
                if len(x) > 0 and float(x[-1]) < float(args.x_right):
                    x = np.append(x, float(args.x_right))
                    mean = np.append(mean, mean[-1])
                    se = np.append(se, se[-1])
                ax.plot(
                    x,
                    mean,
                    color=color,
                    lw=2.8,
                    label=label,
                    drawstyle=drawstyle,
                    solid_capstyle="round",
                )
                ax.fill_between(
                    x,
                    mean - se,
                    mean + se,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                    **fill_kwargs,
                )
            ax.set_xlim(0, float(args.x_right))
            ax.grid(True, alpha=0.22, linewidth=0.7)
            ax.tick_params(axis="both", labelsize=15)
            if row == 0:
                ax.set_title(title, fontsize=27, pad=8)
            if col == 0:
                ax.set_ylabel(f"{size_label}\nSimple Regret", fontsize=21)
            if row == 2:
                ax.set_xlabel("Percentage of PromptEval Budget", fontsize=20)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=17 if args.unit_only else 20,
        bbox_to_anchor=(0.5, 0.018),
    )
    title = "MMLU PromptEval Feature Block Comparison by Task Size"
    if args.unit_only:
        title += " (Unit-cost)"
    fig.suptitle(title, fontsize=23 if args.unit_only else 31, y=0.982)
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.94))

    png = args.out_dir / f"{args.stem}.png"
    pdf = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    for mode, _ in modes:
        for size, label in SIZE_ORDER:
            counts = [len(by_mode[mode][size][i]) for i in range(len(FEATURES))]
            print(f"{mode:5s} {label:6s}: {counts}")


if __name__ == "__main__":
    main()
