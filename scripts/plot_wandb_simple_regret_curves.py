#!/usr/bin/env python3
"""Plot mean simple-regret curves from downloaded W&B history.

Aggregates repeated run_seeds by method (default: ``method_label``).
Supports merging bandit history from ``download_wandb_simple_regret.py`` and
BO history from ``download_wandb_bo_baseline.py``.

Optional stopping overlays:
- pass ``--summary-csv`` files together with ``--plot-stopping``;
- stopping x-positions are aggregated by the same ``--group-by`` field as curves;
- a vertical line shows the mean/median stop location, and the shaded vertical
  band shows std or stderr depending on ``--range``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _str_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a string Series even if ``col`` is missing."""
    if col in df.columns:
        return df[col].astype(str)
    return pd.Series("", index=df.index, dtype=str)


def infer_benchmark_key_row(row: pd.Series) -> str:
    if pd.notna(row.get("benchmark_key")) and str(row.get("benchmark_key")).strip():
        return str(row.get("benchmark_key")).strip()
    dataset = str(row.get("dataset_tag_resolved") or row.get("dataset_tag") or "").lower()
    matrix = str(row.get("matrix") or row.get("bo_inputs") or "")
    if "mmlu" in dataset or "/mmlu/" in matrix.lower():
        task = row.get("mmlu_task")
        if pd.notna(task) and str(task).strip():
            return f"mmlu_{safe_token(str(task))}"
        stem = Path(matrix).stem.removesuffix("_bo_inputs").removesuffix("_bo")
        if stem:
            return f"mmlu_{safe_token(stem)}"
    seed = row.get("matrix_seed")
    if pd.isna(seed) or not str(seed).strip():
        m = re.search(r"seed(\d+)", matrix)
        seed = m.group(1) if m else "NA"
    ds = (
        "gsm8k"
        if "gsm8k" in dataset or "gsm8k" in matrix.lower()
        else "piqa"
        if "piqa" in dataset or "piqa" in matrix.lower()
        else dataset or "unknown"
    )
    return f"{ds}_seed{seed}"


def derive_method_label_row(row: pd.Series) -> str:
    if pd.notna(row.get("method_label")) and str(row.get("method_label")).strip():
        return str(row.get("method_label")).strip()
    family = str(row.get("policy_family") or "").strip().lower()
    variant = str(row.get("experiment_variant") or "").strip()
    if family == "bo":
        acq = str(row.get("acquisition") or variant).lower()
        cost_mode = str(row.get("cost_mode") or "unit").lower()
        if acq == "pbgi":
            return "BO PBGI (cost)" if cost_mode == "cost" else "BO PBGI"
        if acq == "logei":
            return "BO LogEI"
        if acq == "logeipc":
            return "BO LogEIPC"
        return f"BO {variant or acq}"
    if family == "gittins":
        return "Bandit Gittins (cost)" if "cost" in variant or "aware" in variant else "Bandit Gittins"
    if family == "lrf":
        return "Bandit UCB-E-LRF (cost)" if "cost" in variant else "Bandit UCB-E-LRF"
    if family == "ucb" or variant.startswith("ucb"):
        return "Bandit UCB-E (cost)" if "cost" in variant else "Bandit UCB-E"
    return variant or family or "unknown"


def load_history_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_summary_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_bo_phase_filter(df: pd.DataFrame, bo_phase: str) -> pd.DataFrame:
    if bo_phase != "post_init":
        return df
    if "selection_phase" not in df.columns:
        return df
    phase = df["selection_phase"].astype(str)
    is_bo = _str_series(df, "policy_family").str.lower() == "bo"
    keep = (~is_bo) | (phase != "random_init")
    return df.loc[keep].copy()


def apply_lrf_phase_filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Optionally drop the LRF warmup stage from history rows.

    LRF warmup is measured in cumulative evaluated cells, even when plotting
    against cumulative original cost. This removes rows before
    ceil(n_cells * warmup_percentage) for LRF variants only.
    """
    if args.lrf_phase != "post_warmup":
        return df

    family = _str_series(df, "policy_family").str.lower()
    variant = _str_series(df, "experiment_variant").str.strip().str.lower()
    is_lrf = (family == "lrf") | variant.str.startswith("lrf")
    if not bool(is_lrf.any()):
        return df

    cum_eval = pd.to_numeric(df.get("cum_eval", pd.Series(np.nan, index=df.index)), errors="coerce")

    if args.lrf_warmup_evals is not None:
        threshold = pd.Series(float(args.lrf_warmup_evals), index=df.index)
    elif "n_cells" in df.columns:
        n_cells = pd.to_numeric(df["n_cells"], errors="coerce")
        if "warmup_percentage" in df.columns:
            warmup_fraction = pd.to_numeric(df["warmup_percentage"], errors="coerce").fillna(
                float(args.lrf_warmup_fraction)
            )
        else:
            warmup_fraction = pd.Series(float(args.lrf_warmup_fraction), index=df.index)
        threshold = np.ceil(n_cells * warmup_fraction)
    elif "n_arms" in df.columns and "n_examples" in df.columns:
        n_arms = pd.to_numeric(df["n_arms"], errors="coerce")
        n_examples = pd.to_numeric(df["n_examples"], errors="coerce")
        threshold = np.ceil(n_arms * n_examples * float(args.lrf_warmup_fraction))
    else:
        threshold = pd.Series(0.0, index=df.index)

    keep = (~is_lrf) | (cum_eval >= threshold)
    out = df.loc[keep].copy()
    removed = int((~keep & is_lrf).sum())
    kept = int((keep & is_lrf).sum())
    print(
        f"LRF post-warmup filter: removed {removed} warmup rows, "
        f"kept {kept} LRF rows.",
        flush=True,
    )
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["benchmark_key"] = df.apply(infer_benchmark_key_row, axis=1)
    df["method_label"] = df.apply(derive_method_label_row, axis=1)
    return df


def filter_common(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply dataset/benchmark/seed/variant/method filters shared by history and summary."""
    if args.dataset is not None:
        dataset = args.dataset.lower()
        mask = (_str_series(df, "dataset_tag_resolved").str.lower() == dataset) | (
            _str_series(df, "dataset_tag").str.lower() == dataset
        )
        df = df[mask]
    if args.benchmark_key is not None:
        df = df[df["benchmark_key"].astype(str) == str(args.benchmark_key)]
    elif args.matrix_seed is not None and "matrix_seed" in df.columns:
        df = df[df["matrix_seed"].astype(str) == str(args.matrix_seed)]
    if args.run_seed is not None and "run_seed" in df.columns:
        df = df[df["run_seed"].astype(str) == str(args.run_seed)]
    if args.variant_regex is not None and "experiment_variant" in df.columns:
        rx = re.compile(args.variant_regex)
        df = df[df["experiment_variant"].astype(str).map(lambda x: bool(rx.search(x)))]
    if args.method_regex is not None:
        rx = re.compile(args.method_regex)
        df = df[df["method_label"].astype(str).map(lambda x: bool(rx.search(x)))]
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--history-csv",
        action="append",
        required=True,
        type=Path,
        help="One or more history CSV / CSV.GZ files (bandit and/or BO). Repeat to merge.",
    )
    p.add_argument(
        "--summary-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional run summary CSV files. Repeat to merge. Required only when "
            "using --plot-stopping."
        ),
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. gsm8k or mmlu")
    p.add_argument(
        "--benchmark-key",
        default=None,
        help="Filter to one benchmark, e.g. gsm8k_seed1 or mmlu_abstract_algebra",
    )
    p.add_argument(
        "--matrix-seed",
        default=None,
        help="Optional legacy filter when benchmark-key is not set",
    )
    p.add_argument(
        "--run-seed",
        default=None,
        help="Optional algorithm run seed filter. Usually leave unset for mean/std aggregation.",
    )
    p.add_argument("--x-axis", choices=["cum_eval", "cum_original_cost", "step_idx"], default="cum_eval")
    p.add_argument("--y-axis", default="simple_regret")
    p.add_argument("--group-by", default="method_label", help="Column used for curve labels / aggregation groups")
    p.add_argument("--bo-phase", choices=["full", "post_init"], default="full", help="For BO rows, post_init drops random-init steps")
    p.add_argument(
        "--lrf-phase",
        choices=["full", "post_warmup"],
        default="full",
        help="For LRF rows, post_warmup drops the warmup stage before plotting.",
    )
    p.add_argument(
        "--lrf-warmup-fraction",
        type=float,
        default=0.05,
        help="Fallback LRF warmup fraction used when warmup_percentage is missing.",
    )
    p.add_argument(
        "--lrf-warmup-evals",
        type=float,
        default=None,
        help="Optional explicit warmup threshold in cumulative evaluations/cells.",
    )
    p.add_argument("--variant-regex", default=None, help="Optional regex filter on experiment_variant")
    p.add_argument("--method-regex", default=None, help="Optional regex filter on derived method_label")
    p.add_argument("--max-variants", type=int, default=12, help="Limit number of plotted groups after filtering")
    p.add_argument("--grid-size", type=int, default=300)
    p.add_argument("--range", choices=["std", "stderr", "none"], default="std")
    p.add_argument("--title", default=None)

    # Minimal stopping-line additions.
    p.add_argument(
        "--plot-stopping",
        action="store_true",
        help=(
            "Overlay aggregated stopping markers from --summary-csv. "
            "Uses the same filters and --group-by as the regret curves."
        ),
    )
    p.add_argument(
        "--stop-aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Center statistic for vertical stopping lines.",
    )
    p.add_argument(
        "--plot-recommendation-stop",
        action="store_true",
        help="Also plot Gittins recommendation-aware stopping lines.",
    )
    return p.parse_args()


def aggregate_variant(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runs = []
    max_x = 0.0
    for _, g in df.groupby("run_id"):
        gg = g[[x_col, y_col]].dropna().sort_values(x_col)
        if gg.empty:
            continue
        gg = gg.groupby(x_col, as_index=False)[y_col].last()
        x = gg[x_col].to_numpy(dtype=float)
        y = gg[y_col].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        runs.append((x, y))
        max_x = max(max_x, float(np.nanmax(x)))

    if not runs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_grid = np.linspace(0.0, max_x, grid_size)
    ys = []
    for x, y in runs:
        yi = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
        ys.append(yi)
    y_arr = np.vstack(ys)
    mean = np.nanmean(y_arr, axis=0)
    std = np.nanstd(y_arr, axis=0)
    n = np.sum(~np.isnan(y_arr), axis=0)
    stderr = std / np.sqrt(np.maximum(n, 1))
    return x_grid, mean, std, stderr


def _numeric_stop_values(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    return vals


def _stop_columns_for_xaxis(x_axis: str, plot_recommendation_stop: bool) -> list[tuple[str, str, str]]:
    if x_axis == "cum_original_cost":
        out = [
            ("gittins_stop_cum_original_cost", "gittins_index_stop", "--"),
            ("bo_stop_cum_original_cost", "bo_stop", "--"),
        ]
        if plot_recommendation_stop:
            out.append(
                (
                    "gittins_recommendation_aware_stop_cum_original_cost",
                    "gittins_recommendation_stop",
                    ":",
                )
            )
        return out
    if x_axis == "cum_eval":
        out = [
            ("gittins_stop_cum_eval", "gittins_index_stop", "--"),
            ("bo_stop_cum_eval", "bo_stop", "--"),
        ]
        if plot_recommendation_stop:
            out.append(
                (
                    "gittins_recommendation_aware_stop_cum_eval",
                    "gittins_recommendation_stop",
                    ":",
                )
            )
        return out
    return []


def plot_stopping_overlays(
    *,
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    variants: list[str],
    args: argparse.Namespace,
    color_by_variant: dict[str, str],
) -> list[dict[str, object]]:
    stop_specs = _stop_columns_for_xaxis(args.x_axis, args.plot_recommendation_stop)
    stop_rows: list[dict[str, object]] = []
    if not stop_specs:
        if args.plot_stopping:
            print(f"Stopping overlay skipped: x-axis {args.x_axis!r} has no stopping columns.")
        return stop_rows

    for variant in variants:
        if args.group_by not in summary_df.columns:
            continue
        vg = summary_df[summary_df[args.group_by].astype(str) == variant]
        if vg.empty:
            continue

        color = color_by_variant.get(variant, None)
        for col, stop_type, linestyle in stop_specs:
            vals = _numeric_stop_values(vg, col)
            if vals.size == 0:
                continue

            center = float(np.nanmean(vals)) if args.stop_aggregate == "mean" else float(np.nanmedian(vals))
            std = float(np.nanstd(vals))
            stderr = float(std / np.sqrt(max(int(vals.size), 1)))
            band = std if args.range == "std" else stderr if args.range == "stderr" else 0.0

            if args.range != "none" and band > 0:
                ax.axvspan(center - band, center + band, color=color, alpha=0.08, linewidth=0)

            suffix = {
                "gittins_index_stop": "Gittins stop",
                "gittins_recommendation_stop": "Gittins rec-stop",
                "bo_stop": "BO stop",
            }.get(stop_type, "stop")
            label = f"{variant} {suffix}"
            ax.axvline(
                center,
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                alpha=0.78,
                label=label,
            )
            stop_rows.append(
                {
                    "variant": variant,
                    "x_axis": args.x_axis,
                    "stop_type": stop_type,
                    "stop_column": col,
                    "stop_aggregate": args.stop_aggregate,
                    "x": center,
                    "std": std,
                    "stderr": stderr,
                    "n": int(vals.size),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                }
            )

    if args.plot_stopping:
        print(f"Plotted {len(stop_rows)} aggregated stopping markers.")
    return stop_rows


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_history_csv(path) for path in args.history_csv]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = add_derived_columns(df)
    df = apply_bo_phase_filter(df, args.bo_phase)
    df = apply_lrf_phase_filter(df, args)
    df = filter_common(df, args)

    df = df.dropna(subset=[args.x_axis, args.y_axis, args.group_by, "run_id"])
    variants = sorted(df[args.group_by].dropna().astype(str).unique().tolist())

    if len(variants) > args.max_variants:
        final_rows = []
        for variant, vg in df.groupby(args.group_by):
            vals = []
            for _, rg in vg.groupby("run_id"):
                rr = rg[[args.x_axis, args.y_axis]].dropna().sort_values(args.x_axis)
                if not rr.empty:
                    vals.append(float(rr[args.y_axis].iloc[-1]))
            if vals:
                final_rows.append((str(variant), float(np.mean(vals))))
        variants = [v for v, _ in sorted(final_rows, key=lambda t: t[1])[: args.max_variants]]
        print(f"Too many groups; plotting top {len(variants)} by mean final {args.y_axis}:")
        for v in variants:
            print("  ", v)

    if not variants:
        raise SystemExit("No groups left after filtering.")

    fig, ax = plt.subplots(figsize=(10, 6))
    aggregate_rows = []
    color_by_variant: dict[str, str] = {}

    for variant in variants:
        vg = df[df[args.group_by].astype(str) == variant]
        x_grid, mean, std, stderr = aggregate_variant(vg, args.x_axis, args.y_axis, args.grid_size)
        if x_grid.size == 0:
            continue
        (line,) = ax.plot(x_grid, mean, label=variant, linewidth=1.8)
        color_by_variant[variant] = line.get_color()

        if args.range != "none":
            band = std if args.range == "std" else stderr
            ax.fill_between(x_grid, mean - band, mean + band, alpha=0.15, color=line.get_color())

        for x, m, s, se in zip(x_grid, mean, std, stderr):
            aggregate_rows.append(
                {
                    "variant": variant,
                    "x_axis": args.x_axis,
                    "x": x,
                    "mean": m,
                    "std": s,
                    "stderr": se,
                }
            )

    stop_rows: list[dict[str, object]] = []
    if args.plot_stopping:
        if not args.summary_csv:
            raise SystemExit("--plot-stopping requires at least one --summary-csv.")
        summary_frames = [load_summary_csv(path) for path in args.summary_csv]
        summary_df = pd.concat(summary_frames, ignore_index=True, sort=False)
        summary_df = add_derived_columns(summary_df)
        summary_df = filter_common(summary_df, args)
        stop_rows = plot_stopping_overlays(
            ax=ax,
            summary_df=summary_df,
            variants=variants,
            args=args,
            color_by_variant=color_by_variant,
        )

    bench = args.benchmark_key or f"seed{args.matrix_seed or 'all'}"
    title = args.title or f"{args.dataset or 'all'} {bench}: {args.y_axis} vs {args.x_axis}"
    ax.set_title(title)
    ax.set_xlabel(args.x_axis)
    ax.set_ylabel(args.y_axis)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    name_parts = [args.dataset or "all", safe_token(str(bench)), args.y_axis, "vs", args.x_axis]
    if args.bo_phase == "post_init":
        name_parts.append("bo_post_init")
    if args.lrf_phase == "post_warmup":
        name_parts.append("lrf_post_warmup")
    if args.plot_stopping:
        name_parts.append("stopping")
    if args.method_regex:
        name_parts.append(safe_token(args.method_regex)[:80])
    out_png = args.out_dir / ("_".join(name_parts) + ".png")
    out_csv = args.out_dir / ("_".join(name_parts) + "_aggregated.csv")
    out_stop_csv = args.out_dir / ("_".join(name_parts) + "_stopping.csv")

    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    pd.DataFrame(aggregate_rows).to_csv(out_csv, index=False)
    if args.plot_stopping:
        pd.DataFrame(stop_rows).to_csv(out_stop_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    if args.plot_stopping:
        print(f"Wrote {out_stop_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
