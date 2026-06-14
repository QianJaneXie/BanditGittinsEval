#!/usr/bin/env python3
"""
Generate synthetic 0/1 accuracy matrices.

Minimal version:
- Only saves final synthetic matrix .npy files.
- No latent means, no metadata, no manifest.

Settings:
1. GSM8K-like:
   shape copied from one real GSM8K matrix
   prior = default N(0.5, 0.04)

2. Optional GSM8K dataset-prior:
   shape copied from one real GSM8K matrix
   prior = dataset N(0.2, 0.01)

3. MMLU high/easy-like:
   one synthetic matrix per selected high/easy MMLU subject
   shape copied from the real MMLU subject matrix
   prior = high/easy N(0.75, 0.02)
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


DEFAULT_PRIOR = (0.5, 0.04)
GSM8K_DATASET_PRIOR = (0.2, 0.01)
MMLU_HIGH_PRIOR = (0.75, 0.01)

MMLU_HIGH_SUBJECTS = [
    "jurisprudence",
    "high_school_biology",
    "logical_fallacies",
    "human_sexuality",
    "computer_security",
    "management",
    "high_school_geography",
    "international_law",
    "world_religions",
    "high_school_us_history",
    "miscellaneous",
    "high_school_world_history",
    "high_school_psychology",
    "sociology",
    "high_school_government_and_politics",
    "us_foreign_policy",
    "marketing",
]


def stable_rng(base_seed: int, *tokens: object) -> np.random.Generator:
    text = "::".join([str(base_seed), *[str(t) for t in tokens]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed32 = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(seed32)


def load_shape(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference matrix not found: {path}")

    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}: {path}")

    return int(arr.shape[0]), int(arr.shape[1])


def sample_theta(
    *,
    rng: np.random.Generator,
    n_arms: int,
    prior_mean: float,
    prior_variance: float,
) -> np.ndarray:
    """
    Sample latent arm means:
        theta_k ~ Normal(prior_mean, prior_variance)

    Since theta is a Bernoulli probability, clip it into [0, 1].
    """
    std = float(np.sqrt(prior_variance))
    theta = rng.normal(loc=prior_mean, scale=std, size=n_arms)
    theta = np.clip(theta, 0.0, 1.0)
    return theta.astype(np.float64)


def generate_matrix(
    *,
    n_arms: int,
    n_examples: int,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate:
        X_kj ~ Bernoulli(theta_k)

    Output shape:
        (n_arms, n_examples)
    """
    probs = theta.reshape(n_arms, 1)
    matrix = rng.random((n_arms, n_examples)) < probs
    return matrix.astype(np.uint8)


def write_synthetic_matrix(
    *,
    reference_matrix: Path,
    output_matrix: Path,
    prior_mean: float,
    prior_variance: float,
    base_seed: int,
    matrix_seed: int,
    setting: str,
    subject: str | None,
    overwrite: bool,
) -> None:
    if output_matrix.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {output_matrix}")

    n_arms, n_examples = load_shape(reference_matrix)

    rng_theta = stable_rng(base_seed, matrix_seed, setting, subject or "none", "theta")
    rng_matrix = stable_rng(base_seed, matrix_seed, setting, subject or "none", "matrix")

    theta = sample_theta(
        rng=rng_theta,
        n_arms=n_arms,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )

    matrix = generate_matrix(
        n_arms=n_arms,
        n_examples=n_examples,
        theta=theta,
        rng=rng_matrix,
    )

    output_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_matrix, matrix)

    print(
        f"Wrote {output_matrix} | "
        f"shape={matrix.shape} | "
        f"prior=N({prior_mean}, {prior_variance}) | "
        f"theta_mean={theta.mean():.4f} | "
        f"matrix_mean={matrix.mean():.4f}"
    )


def parse_mmlu_subjects(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.lower() == "all":
        return list(MMLU_HIGH_SUBJECTS)

    subjects = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(subjects) - set(MMLU_HIGH_SUBJECTS))
    if unknown:
        raise ValueError(
            f"Unknown or non-high MMLU subjects: {unknown}\n"
            f"Allowed subjects: {MMLU_HIGH_SUBJECTS}"
        )

    return subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/synthetic_matrices"),
    )
    parser.add_argument(
        "--matrix-seeds",
        nargs="+",
        type=int,
        default=[1],
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260524,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--gsm8k-reference",
        type=Path,
        default=Path("data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy"),
    )
    parser.add_argument(
        "--mmlu-dir",
        type=Path,
        default=Path("data/MMLU_matrices"),
    )
    parser.add_argument(
        "--mmlu-high-subjects",
        default="all",
        help="Use 'all' or comma-separated subjects, e.g. jurisprudence,marketing",
    )

    parser.add_argument(
        "--skip-gsm8k",
        action="store_true",
    )
    parser.add_argument(
        "--skip-gsm8k-default",
        action="store_true",
    )
    parser.add_argument(
        "--include-gsm8k-dataset",
        action="store_true",
    )
    parser.add_argument(
        "--skip-mmlu",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = parse_mmlu_subjects(args.mmlu_high_subjects)

    total = 0

    for matrix_seed in args.matrix_seeds:
        seed_dir = f"seed{matrix_seed}"

        if not args.skip_gsm8k:
            if not args.skip_gsm8k_default:
                mu, var = DEFAULT_PRIOR
                out = (
                    args.out_dir
                    / "gsm8k_default"
                    / seed_dir
                    / f"gsm8k_default_synthetic_seed{matrix_seed}.npy"
                )

                write_synthetic_matrix(
                    reference_matrix=args.gsm8k_reference,
                    output_matrix=out,
                    prior_mean=mu,
                    prior_variance=var,
                    base_seed=args.base_seed,
                    matrix_seed=matrix_seed,
                    setting="gsm8k_default",
                    subject=None,
                    overwrite=args.overwrite,
                )
                total += 1

            if args.include_gsm8k_dataset:
                mu, var = GSM8K_DATASET_PRIOR
                out = (
                    args.out_dir
                    / "gsm8k_dataset"
                    / seed_dir
                    / f"gsm8k_dataset_synthetic_seed{matrix_seed}.npy"
                )

                write_synthetic_matrix(
                    reference_matrix=args.gsm8k_reference,
                    output_matrix=out,
                    prior_mean=mu,
                    prior_variance=var,
                    base_seed=args.base_seed,
                    matrix_seed=matrix_seed,
                    setting="gsm8k_dataset",
                    subject=None,
                    overwrite=args.overwrite,
                )
                total += 1

        if not args.skip_mmlu:
            mu, var = MMLU_HIGH_PRIOR

            for subject in subjects:
                reference = args.mmlu_dir / f"{subject}.npy"

                # Keep subject name in the synthetic filename.
                # The runner can infer MMLU task from the filename stem.
                out = (
                    args.out_dir
                    / "mmlu_high"
                    / seed_dir
                    / f"{subject}_synthetic_high_seed{matrix_seed}.npy"
                )

                write_synthetic_matrix(
                    reference_matrix=reference,
                    output_matrix=out,
                    prior_mean=mu,
                    prior_variance=var,
                    base_seed=args.base_seed,
                    matrix_seed=matrix_seed,
                    setting="mmlu_high",
                    subject=subject,
                    overwrite=args.overwrite,
                )
                total += 1

    print(f"Done. Wrote {total} synthetic matrix files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
