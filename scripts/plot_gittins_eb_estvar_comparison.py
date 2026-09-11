#!/usr/bin/env python3
"""Compare fixed-variance and warm-sample-variance Gittins-EB on MMLU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import plot_gittins_eb_pilot_comparison as base


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_EB = "Gittins-EB (var=0.04)"
ESTVAR_EB = "Gittins-EB (sample var)"


def relabel_curves(curves: list[base.RunCurve], label: str) -> list[base.RunCurve]:
    selected = [curve for curve in curves if curve.dataset == "mmlu"]
    for curve in selected:
        curve.method = label
    return selected


def relabel_stops(
    records: list[dict[str, object]], old: str, new: str
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for record in records:
        if record["dataset"] != "mmlu" or record["method"] != old:
            continue
        selected.append({**record, "method": new})
    return selected


def estimated_prior_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        summary = payload["summary"]
        task = str(config["mmlu_task"])
        seed = int(summary["run_seed"])
        key = (task, seed)
        if key in seen:
            continue
        seen.add(key)
        true_means = np.asarray(payload["true_arm_means"], dtype=float)
        true_variance = float(np.var(true_means, ddof=0))
        estimated_variance = float(summary["prior_variance_resolved"])
        estimated_std = float(
            summary.get("estimated_prior_std", math.sqrt(estimated_variance))
        )
        rows.append(
            {
                "task": task,
                "run_seed": seed,
                "estimated_prior_mean": float(summary["prior_mean_resolved"]),
                "estimated_prior_std": estimated_std,
                "estimated_prior_variance": estimated_variance,
                "raw_estimated_prior_variance": summary.get(
                    "raw_estimated_prior_variance", estimated_variance
                ),
                "prior_variance_was_floored": bool(
                    summary.get("prior_variance_was_floored", False)
                ),
                "true_arm_mean": float(np.mean(true_means)),
                "true_prior_std": math.sqrt(true_variance),
                "true_prior_variance": true_variance,
                "std_ratio_estimated_to_true": (
                    estimated_std / math.sqrt(true_variance)
                    if true_variance > 0.0
                    else float("nan")
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "outputs/gittins_eb_pilot_baselines",
    )
    parser.add_argument(
        "--fixed-eb-root",
        type=Path,
        default=REPO_ROOT / "outputs/gittins_eb_pilot_runs",
    )
    parser.add_argument(
        "--estimated-variance-eb-root",
        type=Path,
        default=REPO_ROOT / "outputs/gittins_eb_estvar_pilot_runs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/figure/new_figure/final/gittins_eb_estvar_comparison",
    )
    parser.add_argument("--grid-size", type=int, default=201)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base.METHODS = ("Gittins-G", "Gittins-S", FIXED_EB, ESTVAR_EB)
    base.STYLE = {
        "Gittins-G": {"color": "tab:green", "lw": 2.0},
        "Gittins-S": {"color": "tab:orange", "lw": 2.0},
        FIXED_EB: {"color": "tab:blue", "lw": 2.2},
        ESTVAR_EB: {"color": "tab:purple", "lw": 2.2},
    }

    baseline_curves = [
        curve
        for curve in base.load_baseline_curves(args.baseline_root)
        if curve.dataset == "mmlu"
    ]
    fixed_curves_raw, _ = base.load_eb_curves(args.fixed_eb_root)
    estvar_curves_raw, _ = base.load_eb_curves(args.estimated_variance_eb_root)
    fixed_curves = relabel_curves(fixed_curves_raw, FIXED_EB)
    estvar_curves = relabel_curves(estvar_curves_raw, ESTVAR_EB)

    baseline_and_fixed_stops = base.load_stop_records(
        args.baseline_root, args.fixed_eb_root
    )
    stop_records = [
        record
        for record in baseline_and_fixed_stops
        if record["dataset"] == "mmlu" and record["method"] in {"Gittins-G", "Gittins-S"}
    ]
    stop_records.extend(
        relabel_stops(baseline_and_fixed_stops, "Gittins-EB", FIXED_EB)
    )
    estvar_stop_records = base.load_stop_records(
        args.baseline_root, args.estimated_variance_eb_root
    )
    stop_records.extend(
        relabel_stops(estvar_stop_records, "Gittins-EB", ESTVAR_EB)
    )
    stop_frame = base.aggregate_stops(stop_records)

    curves, summary = base.aggregate(
        baseline_curves + fixed_curves + estvar_curves, args.grid_size
    )
    curves.to_csv(args.out_dir / "regret_curves_mean_se.csv", index=False)
    summary.to_csv(args.out_dir / "summary_metrics.csv", index=False)
    stop_frame.to_csv(args.out_dir / "stopping_summary.csv", index=False)
    prior_frame = pd.DataFrame(estimated_prior_rows(args.estimated_variance_eb_root))
    prior_frame.to_csv(args.out_dir / "estimated_variance_prior_summary.csv", index=False)
    base.plot_mmlu(curves, stop_frame, args.out_dir)
    print(
        json.dumps(
            {
                "baseline_runs": len(baseline_curves),
                "fixed_eb_runs": len(fixed_curves),
                "estimated_variance_eb_runs": len(estvar_curves),
                "prior_rows": len(prior_frame),
                "out_dir": str(args.out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
