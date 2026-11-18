#!/usr/bin/env python3
"""Run paired matrix/metadata BOHB jobs; dry run lists the exact experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=["gsm8k", "piqa", "mmlu", "alpaca"], default=["gsm8k", "piqa", "mmlu"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--matrix-seeds", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    p.add_argument("--cost-modes", nargs="+", choices=["unit", "cost"], default=["unit", "cost"])
    p.add_argument("--subjects", nargs="+", help="Optional MMLU subject stems; default all 57 subjects.")
    p.add_argument("--alpaca-matrix", type=Path, help="Explicit matrix version matching the comparison runs.")
    p.add_argument("--alpaca-bo-inputs", type=Path)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/bohb")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="Skip a job only when both its trace and JSON sidecar exist.")
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    args = p.parse_args()
    pairs = []
    for dataset in args.datasets:
        if dataset in ("gsm8k", "piqa"):
            for seed in args.matrix_seeds:
                stem = f"{dataset}_1_samples_various_models_seed{seed}"
                pairs.append((ROOT / f"data/BanditEval_matrices/{stem}.npy", ROOT / f"data/bo_inputs/{dataset}/{stem}_bo_inputs.npz"))
        elif dataset == "mmlu":
            subjects = args.subjects or [path.stem for path in sorted((ROOT / "data/MMLU_matrices").glob("*.npy"))]
            if not subjects or (args.subjects is None and len(subjects) != 57):
                p.error("Expected all 57 MMLU subjects; specify --subjects for an intentional subset")
            for subject in subjects:
                if Path(subject).name != subject:
                    p.error("--subjects must contain subject stems")
                pairs.append((ROOT / f"data/MMLU_matrices/{subject}.npy", ROOT / f"data/bo_inputs/mmlu/{subject}_bo_inputs.npz"))
        else:
            if args.alpaca_matrix is None or args.alpaca_bo_inputs is None:
                p.error("AlpacaEval requires explicit --alpaca-matrix and --alpaca-bo-inputs matching the comparison data version")
            pairs.append((args.alpaca_matrix, args.alpaca_bo_inputs))
    jobs = []
    for matrix, inputs in pairs:
        if not matrix.is_file() or not inputs.is_file():
            p.error(f"Missing matrix/metadata pair: {matrix}, {inputs}")
        for seed in args.seeds:
            for mode in args.cost_modes:
                out = args.out_dir / f"{matrix.stem}__bohb_{mode}__seed{seed}.npz"
                if args.skip_existing and out.is_file() and out.with_suffix(".json").is_file():
                    continue
                jobs.append([sys.executable, str(ROOT / "scripts/run_bohb_baseline.py"),
                             "--matrix", str(matrix), "--bo-inputs", str(inputs), "--seed", str(seed),
                             "--experiment-variant", f"bohb_{mode}", "--out", str(out),
                             "--wandb-mode", args.wandb_mode])
    if args.dry_run:
        print(json.dumps({"num_jobs": len(jobs), "commands": jobs}, indent=2))
        return 0
    for i, command in enumerate(jobs):
        print(f"BOHB job {i+1}/{len(jobs)}: {Path(command[3]).stem} {command[9]}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
