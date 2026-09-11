#!/usr/bin/env python3
"""Run one deterministic grid item from a W&B-style YAML without using W&B."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_simple_regret_wandb.py"


def safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def parameter_values(spec: dict[str, Any]) -> list[Any]:
    if "values" in spec:
        return list(spec["values"])
    if "value" in spec:
        return [spec["value"]]
    raise ValueError(f"parameter needs value or values: {spec}")


def grid_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config["parameters"]
    keys = list(parameters)
    values = [parameter_values(parameters[key]) for key in keys]
    return [dict(zip(keys, row, strict=True)) for row in itertools.product(*values)]


def completed_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("summary", {}).get("duplicate_observation_count") == 0
        and payload.get("summary", {}).get("final_observed_cell_count")
        == payload.get("summary", {}).get("final_cum_eval")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument(
        "--estimate-prior-variance",
        action="store_true",
        help="Estimate prior variance from the warm-batch arm means.",
    )
    parser.add_argument(
        "--disable-eb-warm-start",
        action="store_true",
        help="Run the configured Gittins variant without the empirical-Bayes warm start.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "gittins_eb_pilot_runs",
    )
    args = parser.parse_args()
    if args.disable_eb_warm_start and args.estimate_prior_variance:
        parser.error("--estimate-prior-variance requires empirical-Bayes warm start")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    items = grid_items(config)
    if not 0 <= args.index < len(items):
        raise SystemExit(f"index {args.index} outside [0, {len(items)})")
    item = items[args.index]

    matrix = Path(str(item["matrix"]))
    matrix_label = matrix.stem
    variant = str(item["experiment_variant"])
    run_seed = int(item["run_seed"])
    output_dir = args.output_root.resolve() / safe_token(config_path.stem)
    if args.disable_eb_warm_start:
        output_name = f"{safe_token(matrix_label)}__{safe_token(variant)}__seed{run_seed:02d}.json"
    else:
        eb_suffix = "ebwarm_estvar" if args.estimate_prior_variance else "ebwarm"
        output_name = f"{safe_token(matrix_label)}__{safe_token(variant)}_{eb_suffix}__seed{run_seed:02d}.json"
    output_path = output_dir / output_name
    if completed_output(output_path):
        print(f"SKIP completed {output_path}", flush=True)
        return 0

    command = [sys.executable, str(RUNNER)]
    if not args.disable_eb_warm_start:
        command.append("--gittins-empirical-bayes-warm-start")
    if args.estimate_prior_variance:
        command.append("--gittins-empirical-bayes-estimate-prior-variance")
    command.extend(
        [
            "--wandb-mode",
            "disabled",
            "--output-json",
            str(output_path),
        ]
    )
    for key, value in item.items():
        if key in {"wandb_project", "wandb_group", "wandb_mode"}:
            continue
        command.extend([f"--{key.replace('_', '-')}", str(value)])

    task_token = safe_token(
        f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', args.index)}"
    )
    task_tmp = args.output_root.resolve() / "_tmp" / task_token
    task_tmp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(task_tmp)
    env["XDG_CACHE_HOME"] = str(task_tmp / "cache")
    env["WANDB_SILENT"] = "true"

    print(f"RUN index={args.index}/{len(items)} output={output_path}", flush=True)
    print("COMMAND " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        print(f"FAILED returncode={completed.returncode} output={output_path}", file=sys.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
