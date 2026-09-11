#!/usr/bin/env python3
"""Download only the existing Gittins-G/S runs needed by the EB pilot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER = REPO_ROOT / "scripts" / "download_wandb_simple_regret.py"


@dataclass(frozen=True)
class Spec:
    key: str
    dataset: str
    sweep_id: str
    batch_size: int
    matrices: tuple[str, ...]


SPECS = (
    Spec(
        "gsm8k",
        "gsm8k",
        "nerftdgl",
        8,
        tuple(
            f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{i}.npy"
            for i in range(1, 6)
        ),
    ),
    Spec(
        "piqa",
        "piqa",
        "ajgqpgqg",
        8,
        tuple(
            f"data/BanditEval_matrices/piqa_1_samples_various_models_seed{i}.npy"
            for i in range(1, 6)
        ),
    ),
    Spec(
        "mmlu_small",
        "mmlu",
        "wbvkm3vs",
        2,
        (
            "data/MMLU_matrices/abstract_algebra.npy",
            "data/MMLU_matrices/computer_security.npy",
            "data/MMLU_matrices/management.npy",
        ),
    ),
    Spec(
        "mmlu_medium",
        "mmlu",
        "wb4w1d88",
        4,
        (
            "data/MMLU_matrices/virology.npy",
            "data/MMLU_matrices/marketing.npy",
            "data/MMLU_matrices/security_studies.npy",
            "data/MMLU_matrices/high_school_government_and_politics.npy",
            "data/MMLU_matrices/prehistory.npy",
        ),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True, choices=range(len(SPECS)))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gittins_eb_pilot_baselines",
    )
    args = parser.parse_args()
    spec = SPECS[args.index]
    variants = tuple(
        f"gittins_{mode}_B{spec.batch_size}_scale1e-4_{prior}"
        for mode in ("unit", "cost")
        for prior in ("default", "dataset")
    )
    command = [
        sys.executable,
        str(DOWNLOADER),
        "--entity",
        "EfficientLLMEval",
        "--project",
        "GittinsBanditEval",
        "--sweep-id",
        spec.sweep_id,
        "--dataset",
        spec.dataset,
        "--raw-dir",
        str((args.output_root / spec.key).resolve()),
        "--states",
        "finished",
        "--page-size",
        "10000",
        "--print-every",
        "20",
    ]
    for variant in variants:
        command.extend(["--variant", variant])
    for matrix in spec.matrices:
        command.extend(["--matrix", matrix])
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
