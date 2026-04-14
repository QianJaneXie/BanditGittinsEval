#!/usr/bin/env python3
"""
Load each *.pkl under performance_prediction_data/, take the ``matrix`` ndarray,
and write a matching *.npy plus a JSON manifest (shapes, dtypes, value stats).

See Zhang et al., "On Speeding Up Language Model Evaluation" (arXiv:2407.06172),
Table 2 for expected matrix sizes per benchmark setting.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


def matrix_stats(m: np.ndarray) -> dict[str, Any]:
    info: dict[str, Any] = {
        "shape": list(m.shape),
        "dtype": str(m.dtype),
        "size_bytes": int(m.nbytes),
        "min": float(np.min(m)),
        "max": float(np.max(m)),
        "mean": float(np.mean(m)),
    }
    # Heuristic: binary vs graded scores (paper uses S in [0,1] and O in {0,1}; PKLs store scores)
    u = np.unique(m)
    info["num_unique_values"] = int(u.size)
    if u.size <= 4:
        info["unique_values_sample"] = [float(x) for x in u.tolist()]
    is_01 = np.all((m == 0) | (m == 1))
    info["is_strict_0_1"] = bool(is_01)
    nearly_01 = float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    info["values_in_unit_interval"] = bool(nearly_01)
    return info


def extract_one(pkl_path: Path, out_dir: Path) -> dict[str, Any]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{pkl_path.name}: expected dict, got {type(data)}")
    if "matrix" not in data:
        raise KeyError(f"{pkl_path.name}: missing 'matrix' key")
    m = data["matrix"]
    if not isinstance(m, np.ndarray):
        m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError(f"{pkl_path.name}: matrix must be 2D, got shape {m.shape}")

    stem = pkl_path.stem
    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, m)

    row: dict[str, Any] = {
        "source_pkl": pkl_path.name,
        "npy_file": npy_path.name,
        **matrix_stats(m),
    }
    extra_keys = [k for k in data if k != "matrix"]
    if extra_keys:
        row["other_keys_in_pkl"] = sorted(extra_keys)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract matrix arrays from PKL files to NPY + manifest.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "performance_prediction_data",
        help="Directory containing *.pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "matrices",
        help="Where to write *.npy and manifest.json",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    pkls = sorted(input_dir.glob("*.pkl"))
    if not pkls:
        print(f"No .pkl files in {input_dir}", file=sys.stderr)
        return 1

    manifest: list[dict[str, Any]] = []
    errors: list[str] = []
    for pkl in pkls:
        try:
            manifest.append(extract_one(pkl, output_dir))
        except Exception as e:
            errors.append(f"{pkl.name}: {e}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.tsv"
    header = "source_pkl\trows\tcols\tdtype\tmin\tmax\tstrict_0_1\n"
    lines = [header]
    for r in manifest:
        sh = r["shape"]
        lines.append(
            f"{r['source_pkl']}\t{sh[0]}\t{sh[1]}\t{r['dtype']}\t{r['min']}\t{r['max']}\t{r['is_strict_0_1']}\n"
        )
    summary_path.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {len(manifest)} .npy files to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"TSV summary: {summary_path}")
    if errors:
        print("\nErrors:", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
