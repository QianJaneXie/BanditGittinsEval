"""
Compare MMLU correctness tensors in Ys.pickle (PromptEval layout) with
pre-flattened matrices under data/MMLU_matrix/ (see manifest.json).

Ys["MMLU"][task] is a list of length n_models; each element has shape
(n_prompts, n_questions). The matrix file stacks these in model order:

    np.load(task + ".npy") == np.vstack(Ys["MMLU"][task])

Row convention (from MMLU_matrix/manifest.json):
    row_index = model_idx * n_prompts + prompt_idx
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def _default_paths(repo_root: Path) -> tuple[Path, Path]:
    ys = repo_root / "prompteval" / "data" / "Ys.pickle"
    matrix_dir = repo_root / "data" / "MMLU_matrix"
    return ys, matrix_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (parent of prompteval/). Default: parent of this file's directory.",
    )
    parser.add_argument(
        "--ys-pickle",
        type=Path,
        default=None,
        help="Path to Ys.pickle. Default: <repo-root>/prompteval/data/Ys.pickle",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=None,
        help="Directory with manifest.json and *.npy. Default: <repo-root>/data/MMLU_matrix",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="If set, only compare the first N tasks (after sorting names).",
    )
    parser.add_argument(
        "--value-check",
        action="store_true",
        default=True,
        help="Load each .npy and check element-wise equality with vstack(Ys) (default: on).",
    )
    parser.add_argument(
        "--no-value-check",
        action="store_false",
        dest="value_check",
        help="Only compare shapes / manifest metadata, skip loading .npy arrays.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root or Path(__file__).resolve().parent.parent
    ys_path, matrix_dir = _default_paths(repo_root)
    if args.ys_pickle is not None:
        ys_path = args.ys_pickle
    if args.matrix_dir is not None:
        matrix_dir = args.matrix_dir

    manifest_path = matrix_dir / "manifest.json"
    if not ys_path.is_file():
        print(f"ERROR: Ys pickle not found: {ys_path}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(ys_path, "rb") as f:
        ys = pickle.load(f)

    if "MMLU" not in ys:
        print('ERROR: Ys has no "MMLU" key.', file=sys.stderr)
        return 1

    mmlu = ys["MMLU"]
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_tasks = {entry["task"] for entry in manifest["files"]}
    ys_tasks = set(mmlu.keys())

    only_in_ys = sorted(ys_tasks - manifest_tasks)
    only_in_manifest = sorted(manifest_tasks - ys_tasks)
    print(f"Ys.pickle: {ys_path}")
    print(f"MMLU_matrix: {matrix_dir}")
    print(f"Manifest dataset: {manifest.get('dataset', '?')}")
    print(f"Tasks in Ys['MMLU']: {len(ys_tasks)}")
    print(f"Tasks in manifest files: {len(manifest_tasks)}")
    if only_in_ys:
        print(f"Tasks only in Ys (no manifest entry): {only_in_ys}")
    if only_in_manifest:
        print(f"Tasks only in manifest (no Ys key): {only_in_manifest}")

    n_models_manifest = len(manifest.get("models", []))
    manifest_by_task = {e["task"]: e for e in manifest["files"]}

    tasks = sorted(ys_tasks & manifest_tasks)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    failures: list[str] = []
    for task in tasks:
        ys_models = mmlu[task]
        n_models = len(ys_models)
        if n_models_manifest and n_models != n_models_manifest:
            failures.append(
                f"{task}: model count {n_models} != manifest models {n_models_manifest}"
            )
            continue

        shapes = [np.asarray(y).shape for y in ys_models]
        if len(set(shapes)) != 1:
            failures.append(f"{task}: inconsistent Y shapes across models: {shapes}")
            continue

        n_prompts, n_questions = shapes[0]
        meta = manifest_by_task.get(task)
        if meta is not None:
            exp_shape = tuple(meta["shape"])
            stacked_shape = (n_models * n_prompts, n_questions)
            if exp_shape != stacked_shape:
                failures.append(
                    f"{task}: stacked shape {stacked_shape} != manifest {exp_shape}"
                )

        if not args.value_check:
            continue

        npy_path = matrix_dir / f"{task}.npy"
        if not npy_path.is_file():
            failures.append(f"{task}: missing {npy_path.name}")
            continue

        mat = np.load(npy_path)
        stacked = np.vstack([np.asarray(y) for y in ys_models])
        if mat.shape != stacked.shape:
            failures.append(f"{task}: npy shape {mat.shape} != vstack(Ys) {stacked.shape}")
            continue
        if not np.array_equal(stacked, mat):
            diff = np.sum(stacked != mat)
            failures.append(f"{task}: {int(diff)} / {stacked.size} entries differ")

    if failures:
        print("\nComparison failures:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print(f"\nOK: compared {len(tasks)} tasks" + (" (metadata only)" if not args.value_check else " (values match)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
