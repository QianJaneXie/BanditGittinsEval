#!/usr/bin/env python3
"""
Inspect .npy matrix files: shape/dtype/stats, optional corner preview, full print
(for small arrays), or CSV export (for viewing every cell in Excel / editor).

Batch: use --export-csv-dir DIR to write every *.npy under the given path(s) as
DIR/<stem>.csv (no positional paths defaults to <repo>/data/BanditEval_matrices).
"""

from __future__ import annotations

import argparse
import csv
import glob as glob_mod
import sys
from pathlib import Path

import numpy as np


def summarize(m: np.ndarray) -> None:
    print(f"shape: {m.shape}  dtype: {m.dtype}  nbytes: {m.nbytes}")
    print(f"min: {float(np.min(m)):.6g}  max: {float(np.max(m)):.6g}  mean: {float(np.mean(m)):.6g}")
    finite = np.isfinite(m)
    if not np.all(finite):
        n_bad = int(np.size(m) - np.sum(finite))
        print(f"non-finite count: {n_bad}")


def preview_block(m: np.ndarray, max_rows: int, max_cols: int) -> None:
    r = min(max_rows, m.shape[0])
    c = min(max_cols, m.shape[1])
    block = m[:r, :c]
    with np.printoptions(precision=6, suppress=True, linewidth=200):
        print(f"\n--- top-left {r} x {c} ---")
        print(block)


def write_csv(m: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i in range(m.shape[0]):
            row = m[i, :]
            w.writerow([float(x) if np.isfinite(x) else "" for x in row.tolist()])


def view_one(
    npy_path: Path,
    *,
    preview_rows: int,
    preview_cols: int,
    full: bool,
    csv_out: Path | None,
    summary_only: bool,
) -> None:
    if not npy_path.is_file():
        print(f"Not found: {npy_path}", file=sys.stderr)
        return

    m = np.load(npy_path)
    print(f"\n== {npy_path} ==")
    if m.ndim != 2:
        print(f"Warning: expected 2D matrix, got ndim={m.ndim}", file=sys.stderr)
    summarize(m)

    if csv_out is not None:
        write_csv(m, csv_out)
        print(f"Wrote CSV: {csv_out} ({m.shape[0]} rows x {m.shape[1]} cols)")

    if summary_only:
        return

    n_el = int(m.size)
    if full:
        if n_el > 50_000:
            print(
                f"Refusing --full: {n_el} elements (cap 50000). Use --csv or increase preview.",
                file=sys.stderr,
            )
        else:
            with np.printoptions(precision=8, suppress=True, linewidth=200, threshold=sys.maxsize):
                print("\n--- full array ---")
                print(m)
        return

    preview_block(m, preview_rows, preview_cols)
    if m.shape[0] > preview_rows or m.shape[1] > preview_cols:
        print(
            f"\n(Array truncated in preview; use --csv FILE.csv for all values, "
            f"or --preview-rows/--preview-cols, or --full for <=50k elements.)"
        )


def expand_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        path = Path(p)
        if any(c in p for c in "*?[]"):
            out.extend(Path(x) for x in sorted(glob_mod.glob(p, recursive=False)))
            continue
        if path.is_dir():
            out.extend(sorted(path.glob("*.npy")))
        else:
            out.append(path)
    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for q in out:
        s = str(q.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(q)
    return uniq


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="View .npy evaluation matrices.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="One or more .npy files, a directory (all *.npy), or glob patterns. "
        "Omit when using --export-csv-dir to export from data/BanditEval_matrices by default.",
    )
    parser.add_argument("--preview-rows", type=int, default=8, help="Rows of top-left preview")
    parser.add_argument("--preview-cols", type=int, default=12, help="Cols of top-left preview")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print entire array (only if element count <= 50000)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        metavar="FILE_OR_DIR",
        help="Export full matrix: one .csv path if a single .npy is given; if multiple inputs, pass a directory and each {stem}.csv is written there",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print shape/stats (no numeric preview)",
    )
    parser.add_argument(
        "--export-csv-dir",
        type=Path,
        metavar="DIR",
        help="Write each matched .npy to DIR/<stem>.csv (one line of stats per file; no matrix preview). "
        "If no paths are given, exports all *.npy from <repo>/data/BanditEval_matrices.",
    )
    args = parser.parse_args()

    if args.csv is not None and args.export_csv_dir is not None:
        print("Use either --csv or --export-csv-dir, not both.", file=sys.stderr)
        return 2

    paths_in: list[str] = list(args.paths) if args.paths else []
    if not paths_in and args.export_csv_dir is not None:
        paths_in = [str(root / "data" / "BanditEval_matrices")]
    elif not paths_in:
        parser.error("need at least one PATH, or use --export-csv-dir alone to export default data/BanditEval_matrices")

    paths = expand_paths(paths_in)
    if not paths:
        print("No matching .npy files.", file=sys.stderr)
        return 1

    export_dir: Path | None = args.export_csv_dir.resolve() if args.export_csv_dir is not None else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    csv_base = args.csv
    if csv_base is not None and len(paths) > 1:
        if csv_base.suffix.lower() == ".csv":
            print("With multiple .npy inputs, --csv must be a directory, not a .csv file.", file=sys.stderr)
            return 2
        csv_base.mkdir(parents=True, exist_ok=True)

    for npy_path in paths:
        csv_out: Path | None = None
        if export_dir is not None:
            csv_out = export_dir / f"{npy_path.stem}.csv"
        elif csv_base is not None:
            if len(paths) == 1:
                if csv_base.is_dir():
                    csv_out = csv_base / f"{npy_path.stem}.csv"
                else:
                    csv_base.parent.mkdir(parents=True, exist_ok=True)
                    csv_out = csv_base
            else:
                csv_out = csv_base / f"{npy_path.stem}.csv"

        summary_only = args.summary_only or (export_dir is not None)
        view_one(
            npy_path,
            preview_rows=args.preview_rows,
            preview_cols=args.preview_cols,
            full=args.full and export_dir is None,
            csv_out=csv_out,
            summary_only=summary_only,
        )

    if export_dir is not None:
        print(f"\nExported {len(paths)} CSV file(s) to {export_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
