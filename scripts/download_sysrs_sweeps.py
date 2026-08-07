#!/usr/bin/env python3
"""Download all completed SySRs W&B sweeps with resume and completeness checks.

By default, data is written below ``outputs/wandb_downloads_new/sysrs``.
Each sweep gets an independent raw directory so its manifest and checkpoint do
not conflict with earlier experiment downloads. MMLU histories are additionally
split into one directory per task.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOAD_ROOT = REPO_ROOT / "outputs" / "wandb_downloads_new" / "sysrs"
DOWNLOADER = REPO_ROOT / "scripts" / "download_wandb_simple_regret.py"


@dataclass(frozen=True)
class SweepSpec:
    key: str
    sweep_id: str
    dataset: str
    expected_runs: int
    config: Path
    mmlu_size_bucket: str = "all"


SWEEPS = (
    SweepSpec(
        key="gsm8k",
        sweep_id="guaigadq",
        dataset="gsm8k",
        expected_runs=200,
        config=REPO_ROOT / "scripts" / "config" / "GSM8KSimpleRegretSweep_sysrs.yml",
    ),
    SweepSpec(
        key="piqa",
        sweep_id="mxnhrzk2",
        dataset="piqa",
        expected_runs=200,
        config=REPO_ROOT / "scripts" / "config" / "PIQASimpleRegretSweep_sysrs.yml",
    ),
    SweepSpec(
        key="alpaca",
        sweep_id="id3prbke",
        dataset="alpaca",
        expected_runs=40,
        config=REPO_ROOT / "scripts" / "config" / "AlpacaSimpleRegretSweep_sysrs.yml",
    ),
    SweepSpec(
        key="mmlu_small",
        sweep_id="yejrrrk7",
        dataset="mmlu",
        expected_runs=880,
        config=REPO_ROOT
        / "scripts"
        / "config"
        / "MMLUSimpleRegretPilot_small_sysrs.yml",
        mmlu_size_bucket="small",
    ),
    SweepSpec(
        key="mmlu_medium",
        sweep_id="cj4vipp8",
        dataset="mmlu",
        expected_runs=1200,
        config=REPO_ROOT
        / "scripts"
        / "config"
        / "MMLUSimpleRegretPilot_medium_sysrs.yml",
        mmlu_size_bucket="medium",
    ),
    SweepSpec(
        key="mmlu_large",
        sweep_id="0w7o8byj",
        dataset="mmlu",
        expected_runs=200,
        config=REPO_ROOT
        / "scripts"
        / "config"
        / "MMLUSimpleRegretPilot_large_sysrs.yml",
        mmlu_size_bucket="large",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="EfficientLLMEval")
    parser.add_argument("--project", default="GittinsBanditEval")
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.key for spec in SWEEPS],
        default=None,
        help="Download only selected sweep keys; default downloads all six.",
    )
    parser.add_argument("--states", default="finished")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, spec: SweepSpec) -> list[str]:
    raw_dir = args.download_root / spec.key
    command = [
        sys.executable,
        str(DOWNLOADER),
        "--entity",
        args.entity,
        "--project",
        args.project,
        "--sweep-id",
        spec.sweep_id,
        "--dataset",
        spec.dataset,
        "--states",
        args.states,
        "--raw-dir",
        str(raw_dir),
        "--expected-grid-yaml",
        str(spec.config),
        "--recover-missing-targeted",
    ]
    if spec.dataset == "mmlu":
        command.extend(
            [
                "--mmlu-size-bucket",
                spec.mmlu_size_bucket,
                "--split-mmlu-by-task",
                "--out-root",
                str(args.download_root / f"{spec.key}_by_task"),
            ]
        )
    if args.no_history:
        command.append("--no-history")
    if args.refresh_manifest:
        command.append("--refresh-manifest")
    return command


def main() -> int:
    args = parse_args()
    selected = [
        spec for spec in SWEEPS if args.only is None or spec.key in set(args.only)
    ]
    for path in [DOWNLOADER, *(spec.config for spec in selected)]:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(
        f"Selected {len(selected)} sweeps; expected runs="
        f"{sum(spec.expected_runs for spec in selected)}"
    )
    for index, spec in enumerate(selected, start=1):
        command = build_command(args, spec)
        print(
            f"\n[{index}/{len(selected)}] {spec.key}: sweep={spec.sweep_id}, "
            f"expected_runs={spec.expected_runs}",
            flush=True,
        )
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if args.dry_run:
        print("\nDry run complete; no W&B requests or output writes were performed.")
    else:
        print(f"\nAll selected downloads completed under: {args.download_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
