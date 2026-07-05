#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_token(s: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "unknown"


def parse_arms(row: pd.Series) -> list[int]:
    if "pulled_arms_json" in row.index:
        v = row.get("pulled_arms_json")
        if pd.notna(v) and str(v).strip() and str(v).lower() not in {"nan", "none"}:
            try:
                arr = json.loads(str(v))
                return sorted({int(x) for x in arr})
            except Exception:
                pass

    v = row.get("pulled_arm")
    if pd.isna(v):
        return []
    return [int(float(v))]


def concentration_metrics(counts: np.ndarray) -> dict[str, float | int]:
    total = float(counts.sum())
    if total <= 0:
        return {
            "unique_arms_pulled": 0,
            "max_pulls_one_arm": 0,
            "top1_share": 0.0,
            "top5_share": 0.0,
            "hhi": 0.0,
        }

    sorted_counts = np.sort(counts)[::-1]
    shares = counts / total
    return {
        "unique_arms_pulled": int((counts > 0).sum()),
        "max_pulls_one_arm": int(sorted_counts[0]),
        "top1_share": float(sorted_counts[:1].sum() / total),
        "top5_share": float(sorted_counts[:5].sum() / total),
        "hhi": float(np.sum(shares ** 2)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history-csv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dataset", default=None)
    p.add_argument("--matrix-seed", default=None)
    p.add_argument("--run-seed", default=None)
    p.add_argument("--variant-regex", default="ucb|gittins")
    p.add_argument("--max-runs", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.history_csv)

    if args.dataset is not None:
        ds = args.dataset.lower()
        mask = pd.Series(False, index=df.index)
        for col in ["dataset_tag_resolved", "dataset_tag"]:
            if col in df.columns:
                mask = mask | (df[col].astype(str).str.lower() == ds)
        df = df[mask]

    if args.matrix_seed is not None and "matrix_seed" in df.columns:
        df = df[df["matrix_seed"].astype(str) == str(args.matrix_seed)]

    if args.run_seed is not None and "run_seed" in df.columns:
        df = df[df["run_seed"].astype(str) == str(args.run_seed)]

    if args.variant_regex is not None and "experiment_variant" in df.columns:
        rx = re.compile(args.variant_regex)
        df = df[df["experiment_variant"].astype(str).map(lambda x: bool(rx.search(x)))]

    if df.empty:
        raise SystemExit("No rows left after filtering.")

    sort_col = "step_idx" if "step_idx" in df.columns else "cum_eval"
    summary_rows = []
    made = 0

    for run_id, g in df.groupby("run_id"):
        if made >= args.max_runs:
            break

        g = g.sort_values(sort_col)
        if g.empty:
            continue

        n_arms = None
        if "n_arms" in g.columns and pd.notna(g["n_arms"].iloc[0]):
            try:
                n_arms = int(float(g["n_arms"].iloc[0]))
            except Exception:
                n_arms = None

        all_arms = []
        arms_per_step = []
        for _, row in g.iterrows():
            arms = parse_arms(row)
            arms_per_step.append(arms)
            all_arms.extend(arms)

        if not all_arms:
            continue

        if n_arms is None:
            n_arms = max(all_arms) + 1

        counts = np.zeros(n_arms, dtype=int)
        snapshots = []

        for arms in arms_per_step:
            for a in set(arms):
                if 0 <= a < n_arms:
                    counts[a] += 1
            snapshots.append(counts.copy())

        snap = np.vstack(snapshots)
        x = np.arange(1, snap.shape[0] + 1)

        variant = str(g["experiment_variant"].iloc[0]) if "experiment_variant" in g.columns else "unknown"
        run_seed = str(g["run_seed"].iloc[0]) if "run_seed" in g.columns else "unknown"
        matrix_seed = str(g["matrix_seed"].iloc[0]) if "matrix_seed" in g.columns else "unknown"
        matrix_task = str(g["matrix_task"].iloc[0]) if "matrix_task" in g.columns else ""

        plt.figure(figsize=(10, 4.5))
        for a in range(n_arms):
            if snap[-1, a] > 0:
                plt.plot(x, snap[:, a], linewidth=0.8, alpha=0.8)

        plt.xlabel("Iteration index / batch step")
        plt.ylabel("Cumulative batch pulls per arm")
        plt.title(f"{variant} | run_seed={run_seed} | matrix_seed={matrix_seed} {matrix_task}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        out = args.out_dir / f"{safe_token(variant)}_runseed{safe_token(run_seed)}_matrixseed{safe_token(matrix_seed)}_{safe_token(run_id)}.png"
        plt.savefig(out, dpi=160)
        plt.close()

        metrics = concentration_metrics(counts)
        summary_rows.append({
            "run_id": run_id,
            "experiment_variant": variant,
            "run_seed": run_seed,
            "matrix_seed": matrix_seed,
            "matrix_task": matrix_task,
            "n_steps": int(snap.shape[0]),
            "n_arms": int(n_arms),
            **metrics,
            "plot_path": str(out),
        })

        made += 1
        print(f"Wrote {out}")

    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "rollout_concentration_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
