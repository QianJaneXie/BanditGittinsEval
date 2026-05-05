from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize(task: str, mu: np.ndarray) -> dict:
    sorted_mu = np.sort(mu)
    best = float(sorted_mu[-1])
    second = float(sorted_mu[-2]) if len(sorted_mu) >= 2 else float("nan")
    return {
        "task": task,
        "n_arms": int(len(mu)),
        "mean": float(np.mean(mu)),
        "std": float(np.std(mu)),
        "min": float(np.min(mu)),
        "q05": float(np.quantile(mu, 0.05)),
        "q25": float(np.quantile(mu, 0.25)),
        "median": float(np.quantile(mu, 0.50)),
        "q75": float(np.quantile(mu, 0.75)),
        "q95": float(np.quantile(mu, 0.95)),
        "max_best_arm": best,
        "second_best_arm": second,
        "top_gap": best - second,
    }


def plot_one(task: str, mu: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(mu, bins=40, density=False, alpha=0.75)
    ax.axvline(0.5, linestyle="--", linewidth=1.6, label="default prior mean = 0.5")
    ax.axvline(0.75, linestyle="--", linewidth=1.6, label="high-bucket data prior mean = 0.75")
    ax.axvline(float(np.mean(mu)), linestyle="-", linewidth=1.8, label=f"empirical mean = {np.mean(mu):.3f}")
    ax.axvline(float(np.max(mu)), linestyle=":", linewidth=1.8, label=f"best arm = {np.max(mu):.3f}")

    ax.set_title(f"MMLU {task}: latent mean accuracy distribution")
    ax.set_xlabel("Latent mean accuracy per arm")
    ax.set_ylabel("Number of arms")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = out_dir / f"{task}_latent_mean_accuracy_distribution.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote", out_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", type=Path, default=Path("data/MMLU_matrices"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/latent_mean_accuracy/mmlu_high"))
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["us_foreign_policy", "jurisprudence", "management", "international_law"],
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for task in args.tasks:
        path = args.matrix_dir / f"{task}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}. Try checking your matrix directory.")

        mat = np.load(path)
        if mat.ndim != 2:
            raise ValueError(f"{path} should be 2D, got shape {mat.shape}")

        mu = np.nanmean(mat, axis=1)

        pd.DataFrame({
            "task": task,
            "arm_idx": np.arange(len(mu)),
            "latent_mean_accuracy": mu,
        }).to_csv(args.out_dir / f"{task}_latent_mean_accuracy.csv", index=False)

        summary_rows.append(summarize(task, mu))
        plot_one(task, mu, args.out_dir)

    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "mmlu_high_latent_mean_accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())