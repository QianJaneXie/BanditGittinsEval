#!/usr/bin/env python3
"""Author-default BOHB baseline with categorical features and item subsampling."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bohb_policy import default_parameters, load_data, run_bohb


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bo-inputs", "--bo_inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", "--run-seed", "--run_seed", type=int, default=0)
    parser.add_argument("--experiment-variant", "--experiment_variant", choices=["bohb_unit", "bohb_cost"], default="bohb_unit")
    parser.add_argument("--eval-budget-fraction", "--eval_budget_fraction", type=float, default=0.1)
    parser.add_argument("--min-fidelity", "--min_fidelity", type=float, default=0.01,
                        help="Author default lower bound; actual lowest rung is 1/81 with eta=3.")
    parser.add_argument("--eta", type=float, default=3)
    parser.add_argument("--max-jobs", "--max_jobs", type=int, help="Optional job cap for implementation smoke tests.")
    parser.add_argument("--dataset-tag", "--dataset_tag")
    parser.add_argument("--wandb-mode", "--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", "--wandb_project", default="GittinsBanditEval")
    parser.add_argument("--wandb-entity", "--wandb_entity")
    parser.add_argument("--wandb-group", "--wandb_group")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.perf_counter()
    data = load_data(args.matrix, args.bo_inputs)
    cost_budgeted = args.experiment_variant == "bohb_cost"
    if args.out is None:
        args.out = ROOT / "outputs/bohb" / f"{args.matrix.stem}__{args.experiment_variant}__seed{args.seed}.npz"
    if args.out.suffix != ".npz":
        raise ValueError("--out must end in .npz")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(f"Result already exists: {args.out}")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(policy_family="bohb", dataset_tag=args.dataset_tag or data.dataset,
                  run_seed=args.seed, cost_mode="cost" if cost_budgeted else "unit",
                  allocation_cost_aware=False, native_defaults=default_parameters(),
                  feature_names=list(data.feature_names), n_arms=data.matrix.shape[0],
                  n_examples=data.matrix.shape[1], recommendation_rule="hpbandster_default_incumbent_trajectory")
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group, mode=args.wandb_mode, config=config,
                         name=f"{args.matrix.stem}_{args.experiment_variant}_seed{args.seed}")
        run.define_metric("cum_eval")
        run.define_metric("simple_regret", step_metric="cum_eval")

    def on_step(row):
        if run is not None:
            run.log({**row, "cum_eval": row["x"], "cum_original_cost": row["x_original_cost"],
                     "simple_regret": row["regret"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    native_logger = logging.getLogger("hpbandster")
    log_handler = logging.FileHandler(args.out.with_suffix(".log"))
    log_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    native_logger.addHandler(log_handler)
    old_propagate = native_logger.propagate
    native_logger.propagate = False
    try:
        result = run_bohb(data, seed=args.seed, eval_budget_fraction=args.eval_budget_fraction,
                          cost_budgeted=cost_budgeted, min_fidelity=args.min_fidelity,
                          eta=args.eta, max_jobs=args.max_jobs, on_step=on_step)
        n_jobs = len(result["x"])
        summary = {"num_jobs": n_jobs, "final_simple_regret": float(result["regret"][-1]) if n_jobs else None,
                   "final_cum_eval": int(result["x"][-1]) if n_jobs else 0,
                   "final_cum_original_cost": float(result["x_original_cost"][-1]) if n_jobs else 0,
                   "budget": result["budget"], "budget_spent": result["budget_spent_total"],
                   "unused_budget": result["unused_budget"], "stop_reason": result["stop_reason"],
                   "full_fidelity_recommended_arm": result["full_fidelity_recommended_arm"],
                   "num_model_based_jobs": int(result["model_based_pick"].sum()),
                   "total_wall_time_s": time.perf_counter() - start}
        aliases = {f"bohb_{key}": result[key] for key in
                   ("x", "x_original_cost", "regret", "recommended_arm", "recommended_mean")}
        np.savez_compressed(args.out, **result, **aliases, metadata_json=json.dumps(config),
                            cost_per_arm_original=data.per_cell_cost, features=data.features)
        args.out.with_suffix(".json").write_text(json.dumps({**config, **summary}, indent=2, allow_nan=False) + "\n")
        if run is not None:
            run.summary.update(summary)
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    finally:
        native_logger.removeHandler(log_handler)
        native_logger.propagate = old_propagate
        log_handler.close()
    if run is not None:
        run.finish()
    print(json.dumps({"out": str(args.out), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
