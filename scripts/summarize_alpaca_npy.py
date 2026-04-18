#!/usr/bin/env python3
"""
Load Alpaca GPT-4-turbo weighted 2D comparison matrices (*.npy) and write
machine-readable stats plus a short text summary under data_analysis/.

Matrices are produced by extract_matrices.py from the Zhang et al. benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Presentation order (matches common discussion: continuous -> debias -> discretizations / options).
ALPACA_NPY_STEMS: tuple[str, ...] = (
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_no_rounding",
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_no_rounding_debias",
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_non_trivial_rounding",
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_rounding",
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_option_1",
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_option_2",
)


def _count_close(m: np.ndarray, target: float) -> int:
    return int(np.sum(np.isclose(m, target, rtol=0.0, atol=0.0)))


def analyze_matrix(path: Path, m: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "file": str(path.resolve()),
        "stem": path.stem,
        "shape": list(int(x) for x in m.shape),
        "ndim": int(m.ndim),
        "dtype": str(m.dtype),
        "nbytes": int(m.nbytes),
    }
    if m.ndim != 2:
        row["warning"] = f"expected 2D matrix, got ndim={m.ndim}"

    finite = np.isfinite(m)
    row["nan_count"] = int(np.isnan(m).sum())
    row["inf_count"] = int(np.isinf(m).sum())
    row["nonfinite_count"] = int(m.size - int(finite.sum()))

    mf = m[finite] if finite.any() else m.ravel()[:0]
    if mf.size == 0:
        row["min"] = None
        row["max"] = None
        row["mean"] = None
    else:
        row["min"] = float(np.min(mf))
        row["max"] = float(np.max(mf))
        row["mean"] = float(np.mean(m))

    u = np.unique(m)
    row["num_unique_values"] = int(u.size)
    if u.size <= 12:
        row["unique_values"] = [float(x) for x in u.tolist()]

    row["count_neg2"] = _count_close(m, -2.0)
    row["count_0"] = _count_close(m, 0.0)
    row["count_0_5"] = _count_close(m, 0.5)
    row["count_1"] = _count_close(m, 1.0)

    row["values_in_unit_interval"] = bool(float(m.min()) >= 0.0 and float(m.max()) <= 1.0)
    row["is_strict_0_1"] = bool(np.all((m == 0) | (m == 1)))

    if row["count_neg2"] > 0:
        idx = np.argwhere(np.isclose(m, -2.0))
        row["neg2_positions"] = [[int(r), int(c)] for r, c in idx]
    else:
        row["neg2_positions"] = []

    return row


def verify_non_trivial_vs_no_rounding(no: np.ndarray, nt: np.ndarray) -> dict[str, Any]:
    if no.shape != nt.shape:
        return {"match_shapes": False, "non_trivial_matches_threshold_rule": None}
    pred = np.where(no > 0.5, 1.0, np.where(no < 0.5, 0.0, 0.5))
    ok = bool(np.allclose(nt.astype(np.float64), pred.astype(np.float64)))
    return {
        "match_shapes": True,
        "non_trivial_matches_threshold_rule": ok,
        "counts": {
            "no_gt_0_5": int(np.sum(no > 0.5)),
            "no_lt_0_5": int(np.sum(no < 0.5)),
            "no_eq_0_5": int(np.sum(no == 0.5)),
            "nt_eq_1": int(_count_close(nt, 1.0)),
            "nt_eq_0": int(_count_close(nt, 0.0)),
            "nt_eq_0_5": int(_count_close(nt, 0.5)),
        },
    }


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    cols = [
        "stem",
        "rows",
        "cols",
        "dtype",
        "min",
        "max",
        "mean",
        "nan_count",
        "inf_count",
        "num_unique_values",
        "count_neg2",
        "count_0",
        "count_0_5",
        "count_1",
        "values_in_unit_interval",
        "is_strict_0_1",
    ]

    def cell(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bool):
            return "1" if x else "0"
        return str(x)

    lines = ["\t".join(cols) + "\n"]
    for r in rows:
        sh = r["shape"]
        line = [
            r["stem"],
            str(sh[0]),
            str(sh[1]),
            r["dtype"],
            cell(r["min"]),
            cell(r["max"]),
            cell(r["mean"]),
            str(r["nan_count"]),
            str(r["inf_count"]),
            str(r["num_unique_values"]),
            str(r["count_neg2"]),
            str(r["count_0"]),
            str(r["count_0_5"]),
            str(r["count_1"]),
            cell(r["values_in_unit_interval"]),
            cell(r["is_strict_0_1"]),
        ]
        lines.append("\t".join(line) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_summary_text(
    rows: list[dict[str, Any]],
    verification: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    parts.append(
        "Alpaca (GPT-4 Turbo weighted) — six 2D comparison matrices — .npy statistics summary\n"
    )
    parts.append(
        "Data source: matching .npy files under data/matrices/ exported from PKL.\n"
    )

    title_map = {
        "no_rounding": "Continuous probabilities (no rounding)",
        "no_rounding_debias": "Continuous probabilities + debias (row count may differ by 1)",
        "non_trivial_rounding": "Three-level discrete: {0, 0.5, 1}",
        "rounding": "Four-level discrete: {-2, 0, 0.5, 1} (aggressive rounding; few 0.5s)",
        "option_1": "Mostly like no rounding + small manual fixes (includes -2 sentinel)",
        "option_2": "Four-level discrete (similar to rounding; even fewer 0.5s)",
    }

    for r in rows:
        stem = r["stem"]
        short = stem.split("2d_comparisons_")[-1] if "2d_comparisons_" in stem else stem
        label = title_map.get(short, short)
        sh = r["shape"]
        parts.append(f"\n— {label} —\n")
        parts.append(f"File: {r['stem']}.npy\n")
        parts.append(f"shape: ({sh[0]}, {sh[1]})  dtype: {r['dtype']}\n")
        parts.append(
            f"min / max / mean: {r['min']} / {r['max']} / {r['mean']}\n"
            if r["mean"] is not None
            else ""
        )
        parts.append(f"Unique value count: {r['num_unique_values']}\n")
        if "unique_values" in r:
            parts.append(f"All distinct values: {r['unique_values']}\n")
        parts.append(
            f"Counts — -2: {r['count_neg2']} | 0: {r['count_0']} | 0.5: {r['count_0_5']} | 1: {r['count_1']}\n"
        )
        if r["count_neg2"] and r["neg2_positions"]:
            n = len(r["neg2_positions"])
            parts.append(f"Positions where -2 appears (row, col), {n} total: {r['neg2_positions']}\n")
        parts.append(
            f"Strictly binary {{0,1}}: {r['is_strict_0_1']}; all values in [0,1]: {r['values_in_unit_interval']}\n"
        )

    if verification:
        parts.append("\n— Consistency check (non_trivial_rounding) —\n")
        parts.append(f"Same shape as no_rounding: {verification.get('match_shapes')}\n")
        v = verification.get("non_trivial_matches_threshold_rule")
        if v is not None:
            parts.append(
                "Exactly matches non_trivial if no_rounding is thresholded with "
                ">0.5→1, <0.5→0, ==0.5→0.5: "
                f"{v}\n"
            )
        if "counts" in verification:
            parts.append(f"Auxiliary counts: {verification['counts']}\n")

    parts.append(
        "\nNote: in rounding / option_2, many original 0.5 values (ties) are collapsed to 0 or 1; "
        "see the paper or generation code for the exact tie-break rule. "
        "This summary only reports statistics of the stored matrix values.\n"
    )
    return "".join(parts)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Summarize Alpaca *.npy matrices to data_analysis/.")
    parser.add_argument(
        "--matrices-dir",
        type=Path,
        default=root / "data" / "matrices",
        help="Directory containing Alpaca .npy files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data_analysis",
        help="Where to write alpaca_npy_stats.* and summary txt",
    )
    args = parser.parse_args()

    matrices_dir: Path = args.matrices_dir
    out_dir: Path = args.out_dir
    if not matrices_dir.is_dir():
        print(f"Matrices directory not found: {matrices_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    loaded: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for stem in ALPACA_NPY_STEMS:
        path = matrices_dir / f"{stem}.npy"
        if not path.is_file():
            missing.append(stem)
            continue
        m = np.load(path)
        rows.append(analyze_matrix(path, m))
        loaded[stem] = m

    if missing:
        print("Missing .npy files:", file=sys.stderr)
        for s in missing:
            print(f"  {s}.npy", file=sys.stderr)
        return 1

    verification: dict[str, Any] | None = None
    no_k = "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_no_rounding"
    nt_k = "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_non_trivial_rounding"
    if no_k in loaded and nt_k in loaded:
        verification = verify_non_trivial_vs_no_rounding(loaded[no_k], loaded[nt_k])

    tsv_path = out_dir / "alpaca_npy_stats.tsv"
    json_path = out_dir / "alpaca_npy_stats.json"
    txt_path = out_dir / "alpaca_npy_summary.txt"

    write_tsv(rows, tsv_path)
    payload = {
        "matrices_dir": str(matrices_dir.resolve()),
        "rows": rows,
        "non_trivial_verification": verification,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(build_summary_text(rows, verification), encoding="utf-8")

    print(f"Wrote {tsv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
