#!/usr/bin/env python3
"""Generate MMLU aggregate plots by directly averaging raw simple regret."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cost-mode", choices=["unit", "aware", "both"], default="unit")
    p.add_argument("--small-batch-size", type=int, default=2)
    p.add_argument("--large-batch-size", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path(r"outputs\wandb_plots_new\paper_figures"))
    p.add_argument("--include-large-lrf", action="store_true", default=True)
    p.add_argument("--shared-y-label-x", type=float, default=-0.05)
    p.add_argument("--shared-x-label-y", type=float, default=0.235)
    return p.parse_args()


def run_one(args: argparse.Namespace, cost_mode: str) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("plot_mmlu_aggregate_2x2_normalized_fast_shared_labels.py")),
        "--out-dir",
        str(args.out_dir),
        "--small-batch-size",
        str(args.small_batch_size),
        "--large-batch-size",
        str(args.large_batch_size),
        "--normalize-y",
        "none",
        "--preserve-lrf-bo-x-offset",
        "--shared-y-label-x",
        str(args.shared_y_label_x),
        "--shared-x-label-y",
        str(args.shared_x_label_y),
    ]
    if args.include_large_lrf:
        cmd.append("--include-large-lrf")
    if cost_mode == "aware":
        cmd.extend(["--cost-mode", "aware"])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    modes = ["unit", "aware"] if args.cost_mode == "both" else [args.cost_mode]
    for mode in modes:
        run_one(args, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
