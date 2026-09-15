#!/usr/bin/env python3
"""
Build PromptEval-style Ys / Xs pickles for BanditEval GSM8K, PIQA, and AlpacaEval matrices.

BanditEval .npy files are (n_models, n_samples): rows = models (arms), columns = i.i.d.
samples. There are no prompt templates, so Xs covariate views are empty lists and BAI
runs the baseline (X=None) path only. GSM8K and PIQA are binary; AlpacaEval keeps its
original continuous preference scores in [0, 1]. The BAI runner applies the paper's
0.5 fit-time binarization while retaining these continuous values for evaluation.

Output (default ``prompteval/banditeval_pickle/``):

- ``Ys.pickle`` — ``{"GSM8K": {task: [Y], ...}, "PIQA": {task: [Y], ...}}``
- ``Xs.pickle`` — same keys; ``Xs[bench][task] == [[]]`` (one pseudo-LLM slot, no views)

Run BAI afterward, e.g.::

    python prompteval/bai_evaluation.py --bench GSM8K

or::

    from bai_evaluation import run_bai_evaluation
    run_bai_evaluation(
        data_path="prompteval/banditeval_pickle/",
        bench="GSM8K",
        tasks=["various_models_seed1"],
        combine_models=False,
        random_seeds=5,
    )
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

BANDITEVAL_STEM_RE = re.compile(
    r"^(?P<prefix>gsm8k|piqa|alpaca_eval)_(?P<body>.+)$",
    re.IGNORECASE,
)

DEFAULT_GSM8K_GLOB = "gsm8k_1_samples_various_models_seed*.npy"
DEFAULT_PIQA_GLOB = "piqa_1_samples_various_models_seed*.npy"
DEFAULT_ALPACA_GLOB = (
    "alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_no_rounding_debias.npy"
)

BENCH_NAME = {"gsm8k": "GSM8K", "piqa": "PIQA", "alpaca_eval": "ALPACA"}


def task_id_from_stem(stem: str) -> str:
    """``gsm8k_1_samples_various_models_seed1`` → ``various_models_seed1``."""
    m = BANDITEVAL_STEM_RE.match(stem.strip())
    if not m:
        return stem
    body = m.group("body")
    prefix = m.group("prefix").lower()
    if body.startswith("1_samples_"):
        return body[len("1_samples_") :]
    return body


def bench_from_stem(stem: str) -> str:
    m = BANDITEVAL_STEM_RE.match(stem.strip())
    if not m:
        raise ValueError(f"Cannot infer benchmark from stem {stem!r}")
    return BENCH_NAME[m.group("prefix").lower()]


def load_matrix(path: Path, *, allow_continuous: bool = False) -> np.ndarray:
    y = np.load(path)
    if y.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {path}, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{path}: matrix contains non-finite values")
    if allow_continuous:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if y_min < 0.0 or y_max > 1.0:
            raise ValueError(
                f"{path}: expected bounded values in [0, 1], got min={y_min}, max={y_max}"
            )
        return np.asarray(y, dtype=np.float32)
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(
            f"{path}: expected binary values in {{0, 1}}, got unique {uniq[:10]}..."
        )
    return np.asarray(y, dtype=np.int8)


def discover_matrices(
    root: Path,
    *,
    gsm8k_glob: str,
    piqa_glob: str,
    alpaca_glob: str,
    include_gsm8k: bool,
    include_piqa: bool,
    include_alpaca: bool,
) -> list[Path]:
    paths: list[Path] = []
    if include_gsm8k:
        paths.extend(sorted(root.glob(gsm8k_glob), key=lambda p: p.name.casefold()))
    if include_piqa:
        paths.extend(sorted(root.glob(piqa_glob), key=lambda p: p.name.casefold()))
    if include_alpaca:
        paths.extend(sorted(root.glob(alpaca_glob), key=lambda p: p.name.casefold()))
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p)
    return out


def build_pickles(paths: list[Path]) -> tuple[dict, dict]:
    ys: dict[str, dict[str, list[np.ndarray]]] = {}
    xs: dict[str, dict[str, list[list]]] = {}

    for path in paths:
        bench = bench_from_stem(path.stem)
        task = task_id_from_stem(path.stem)
        y = load_matrix(path, allow_continuous=bench == "ALPACA")

        ys.setdefault(bench, {})[task] = [y]
        xs.setdefault(bench, {})[task] = [[]]

    return ys, xs


def save_pickles(ys: dict, xs: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ys_path = out_dir / "Ys.pickle"
    xs_path = out_dir / "Xs.pickle"
    with ys_path.open("wb") as handle:
        pickle.dump(ys, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with xs_path.open("wb") as handle:
        pickle.dump(xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ys_path, xs_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Build PromptEval Ys/Xs pickles from BanditEval .npy matrices."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "BanditEval_matrices",
        help="Directory containing gsm8k_*.npy / piqa_*.npy files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "banditeval_pickle",
        help="Output directory for Ys.pickle and Xs.pickle (default: prompteval/banditeval_pickle/).",
    )
    p.add_argument(
        "--gsm8k-glob",
        default=DEFAULT_GSM8K_GLOB,
        help=f"Glob under --data-dir for GSM8K (default: {DEFAULT_GSM8K_GLOB!r}).",
    )
    p.add_argument(
        "--piqa-glob",
        default=DEFAULT_PIQA_GLOB,
        help=f"Glob under --data-dir for PIQA (default: {DEFAULT_PIQA_GLOB!r}).",
    )
    p.add_argument(
        "--alpaca-glob",
        default=DEFAULT_ALPACA_GLOB,
        help=f"Glob under --data-dir for AlpacaEval (default: {DEFAULT_ALPACA_GLOB!r}).",
    )
    p.add_argument("--no-gsm8k", action="store_true", help="Skip GSM8K matrices.")
    p.add_argument("--no-piqa", action="store_true", help="Skip PIQA matrices.")
    p.add_argument("--no-alpaca", action="store_true", help="Skip AlpacaEval matrices.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data_dir
    if not root.is_dir():
        print(f"ERROR: data directory not found: {root}", file=sys.stderr)
        return 1

    paths = discover_matrices(
        root,
        gsm8k_glob=args.gsm8k_glob,
        piqa_glob=args.piqa_glob,
        alpaca_glob=args.alpaca_glob,
        include_gsm8k=not args.no_gsm8k,
        include_piqa=not args.no_piqa,
        include_alpaca=not args.no_alpaca,
    )
    if not paths:
        print(
            f"ERROR: no matrices under {root} "
            f"(gsm8k={not args.no_gsm8k}, piqa={not args.no_piqa}, "
            f"alpaca={not args.no_alpaca})",
            file=sys.stderr,
        )
        return 1

    ys, xs = build_pickles(paths)
    ys_path, xs_path = save_pickles(ys, xs, args.out_dir)

    print(f"Wrote {ys_path}")
    print(f"Wrote {xs_path}")
    for bench in sorted(ys):
        tasks = sorted(ys[bench])
        y0 = ys[bench][tasks[0]][0]
        print(
            f"  {bench}: {len(tasks)} task(s), example shape {y0.shape} "
            f"(rows=models, cols=samples); tasks={tasks}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
