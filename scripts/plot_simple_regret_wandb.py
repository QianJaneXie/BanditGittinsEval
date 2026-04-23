#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot_simple_regret.py"

ALGORITHM_SETS: dict[str, list[str]] = {
    "full": ["rr", "ucb", "lrf", "gittins"],
    "no_lrf": ["rr", "ucb", "gittins"],
    "gittins_only": ["gittins"],
    "default": ["ucb", "lrf", "gittins"],
}


def _safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _build_paths(
    *,
    matrix: str | None,
    experiment: str,
    seed: int,
    algorithm_set: str,
    cost_mode: str,
    run_id: str,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if matrix:
        base = Path(matrix).stem
    else:
        base = experiment

    stem = _safe_token(f"{base}__{algorithm_set}__{cost_mode}__seed{seed}__{run_id}")
    fig_path = out_dir / f"{stem}.png"
    traces_path = out_dir / f"{stem}_traces.npz"
    meta_path = out_dir / f"{stem}_traces.meta.json"
    return fig_path, traces_path, meta_path


def _stream_subprocess(cmd: list[str], cwd: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    return process.wait()


def _log_curve(
    run: wandb.sdk.wandb_run.Run,
    z: Any,
    *,
    prefix: str,
    x_eval_key: str,
    y_key: str,
    x_cost_key: str | None = None,
) -> None:
    if x_eval_key not in z.files or y_key not in z.files:
        return

    xs_eval = np.asarray(z[x_eval_key]).reshape(-1)
    ys = np.asarray(z[y_key]).reshape(-1)

    if xs_eval.size == 0 or ys.size == 0:
        return
    if xs_eval.shape[0] != ys.shape[0]:
        return

    xs_cost: np.ndarray | None = None
    if x_cost_key and x_cost_key in z.files:
        tmp = np.asarray(z[x_cost_key]).reshape(-1)
        if tmp.shape[0] == ys.shape[0]:
            xs_cost = tmp

    for i in range(ys.shape[0]):
        payload: dict[str, float | int] = {
            f"{prefix}_cum_eval": int(xs_eval[i]),
            f"{prefix}_simple_regret": float(ys[i]),
        }
        if xs_cost is not None:
            payload[f"{prefix}_cum_original_cost"] = float(xs_cost[i])
        run.log(payload)

    run.summary[f"final_{prefix}_simple_regret"] = float(ys[-1])
    run.summary[f"final_{prefix}_cum_eval"] = int(xs_eval[-1])
    if xs_cost is not None:
        run.summary[f"final_{prefix}_cum_original_cost"] = float(xs_cost[-1])


def _log_outputs(
    run: wandb.sdk.wandb_run.Run,
    *,
    fig_path: Path,
    traces_path: Path,
    meta_path: Path,
) -> None:
    if not traces_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {traces_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    z = np.load(traces_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    run.summary["matrix"] = meta.get("matrix")
    run.summary["figure_path"] = str(fig_path.resolve())
    run.summary["traces_path"] = str(traces_path.resolve())
    run.summary["meta_path"] = str(meta_path.resolve())
    run.summary["budget_stops_by"] = meta.get("budget_stops_by")
    run.summary["gittins_cost_mode"] = meta.get("gittins_cost_mode")
    run.summary["eval_budget_fraction"] = meta.get("eval_budget_fraction")
    run.summary["seed"] = meta.get("seed")
    run.summary["algorithms"] = ",".join(meta.get("algorithms", []))

    _log_curve(
        run,
        z,
        prefix="rr",
        x_eval_key="rr_x",
        y_key="rr_regret",
        x_cost_key="rr_x_original_cost",
    )
    _log_curve(
        run,
        z,
        prefix="ucb",
        x_eval_key="ucb_x",
        y_key="ucb_regret",
        x_cost_key="ucb_x_original_cost",
    )
    _log_curve(
        run,
        z,
        prefix="lrf",
        x_eval_key="lrf_x_plot",
        y_key="lrf_regret_plot",
        x_cost_key="lrf_x_plot_original_cost",
    )
    _log_curve(
        run,
        z,
        prefix="gittins",
        x_eval_key="gittins_x",
        y_key="gittins_regret",
        x_cost_key="gittins_x_original_cost",
    )

    if fig_path.is_file():
        run.log({"simple_regret_figure": wandb.Image(str(fig_path))})

    artifact = wandb.Artifact(
        name=f"simple-regret-{run.id}",
        type="experiment_outputs",
        description="Figure + traces + meta from plot_simple_regret.py",
    )
    if fig_path.is_file():
        artifact.add_file(str(fig_path))
    artifact.add_file(str(traces_path))
    artifact.add_file(str(meta_path))
    run.log_artifact(artifact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run plot_simple_regret.py, then upload traces/figure to W&B."
    )

    parser.add_argument("--entity", type=str, default=None, help="W&B entity/team")
    parser.add_argument("--project", type=str, default=None, help="W&B project")
    parser.add_argument("--group", type=str, default=None, help="Optional W&B group name")

    parser.add_argument(
        "--experiment",
        type=str,
        default="gsm8k_various_model",
        help="Experiment preset name passed through to plot_simple_regret.py",
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        help="Matrix path; if omitted, plot_simple_regret.py uses --experiment default",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--algorithm-set",
        "--algorithm_set",
        dest="algorithm_set",
        choices=sorted(ALGORITHM_SETS.keys()),
        default="full",
        help="Named algorithm bundle",
    )
    parser.add_argument(
        "--eval-budget-fraction",
        "--eval_budget_fraction",
        dest="eval_budget_fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument(
        "--gittins-batch-size",
        "--gittins_batch_size",
        dest="gittins_batch_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--warmup-percentage",
        "--warmup_percentage",
        dest="warmup_percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument("--ucb-a", "--ucb_a", dest="ucb_a", type=int, default=1)
    parser.add_argument("--lrf-device", "--lrf_device", dest="lrf_device", type=str, default="cpu")
    parser.add_argument(
        "--gittins-grid-points",
        "--gittins_grid_points",
        dest="gittins_grid_points",
        type=int,
        default=1025,
    )
    parser.add_argument(
        "--cost-mode",
        "--cost_mode",
        dest="cost_mode",
        choices=["aware", "unaware"],
        default="aware",
    )
    parser.add_argument(
        "--gittins-cost-vector",
        "--gittins_cost_vector",
        dest="gittins_cost_vector",
        type=str,
        default=None,
        help="Required when --cost-mode aware",
    )
    parser.add_argument(
        "--gittins-per-cell-dp",
        "--gittins_per_cell_dp",
        dest="gittins_per_cell_dp",
        action="store_true",
    )
    parser.add_argument(
        "--gittins-prior-mean",
        "--gittins_prior_mean",
        dest="gittins_prior_mean",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--gittins-prior-variance",
        "--gittins_prior_variance",
        dest="gittins_prior_variance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "wandb_runs",
        help="Directory for generated figure/traces",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cost_mode == "aware" and not args.gittins_cost_vector:
        print("--cost-mode aware requires --gittins-cost-vector", file=sys.stderr)
        return 2

    algorithms = ALGORITHM_SETS[args.algorithm_set]

    config: dict[str, Any] = {
        "experiment": args.experiment,
        "matrix": args.matrix,
        "seed": args.seed,
        "algorithm_set": args.algorithm_set,
        "algorithms": algorithms,
        "eval_budget_fraction": args.eval_budget_fraction,
        "batch_size": args.batch_size,
        "gittins_batch_size": args.gittins_batch_size,
        "warmup_percentage": args.warmup_percentage,
        "ucb_a": args.ucb_a,
        "lrf_device": args.lrf_device,
        "gittins_grid_points": args.gittins_grid_points,
        "cost_mode": args.cost_mode,
        "gittins_cost_vector": args.gittins_cost_vector,
        "gittins_per_cell_dp": args.gittins_per_cell_dp,
        "gittins_prior_mean": args.gittins_prior_mean,
        "gittins_prior_variance": args.gittins_prior_variance,
    }

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        job_type="simple_regret",
        config=config,
    )

    assert run is not None

    fig_path, traces_path, meta_path = _build_paths(
        matrix=args.matrix,
        experiment=args.experiment,
        seed=args.seed,
        algorithm_set=args.algorithm_set,
        cost_mode=args.cost_mode,
        run_id=run.id,
        out_dir=args.out_dir,
    )

    run.config.update(
        {
            "resolved_out": str(fig_path),
            "resolved_traces_out": str(traces_path),
            "resolved_meta_out": str(meta_path),
        },
        allow_val_change=True,
    )

    cmd: list[str] = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--experiment",
        args.experiment,
        "--out",
        str(fig_path),
        "--traces-out",
        str(traces_path),
        "--seed",
        str(args.seed),
        "--algorithms",
        *algorithms,
        "--eval-budget-fraction",
        str(args.eval_budget_fraction),
        "--batch-size",
        str(args.batch_size),
        "--gittins-batch-size",
        str(args.gittins_batch_size),
        "--warmup-percentage",
        str(args.warmup_percentage),
        "--ucb-a",
        str(args.ucb_a),
        "--lrf-device",
        args.lrf_device,
        "--gittins-grid-points",
        str(args.gittins_grid_points),
        "--gittins-cost-mode",
        args.cost_mode,
    ]

    if args.matrix:
        cmd.extend(["--matrix", args.matrix])

    if args.cost_mode == "aware" and args.gittins_cost_vector:
        cmd.extend(["--gittins-cost-vector", args.gittins_cost_vector])

    if args.gittins_per_cell_dp:
        cmd.append("--gittins-per-cell-dp")

    if args.gittins_prior_mean is not None:
        cmd.extend(["--gittins-prior-mean", str(args.gittins_prior_mean)])

    if args.gittins_prior_variance is not None:
        cmd.extend(["--gittins-prior-variance", str(args.gittins_prior_variance)])

    print("\n=== Running underlying experiment ===")
    print(" ".join(cmd))
    print("====================================\n")

    t0 = time.time()
    return_code = _stream_subprocess(cmd, cwd=REPO_ROOT)
    t1 = time.time()

    run.summary["wallclock_total_s"] = float(t1 - t0)
    run.summary["subprocess_return_code"] = int(return_code)

    if return_code != 0:
        run.finish(exit_code=return_code)
        return return_code

    try:
        _log_outputs(
            run,
            fig_path=fig_path,
            traces_path=traces_path,
            meta_path=meta_path,
        )
    except Exception as e:
        run.summary["postprocess_error"] = repr(e)
        run.finish(exit_code=1)
        raise

    run.finish()
    print(f"\nDone. W&B run: {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())