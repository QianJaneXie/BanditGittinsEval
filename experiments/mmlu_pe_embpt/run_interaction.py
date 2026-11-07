"""Run the model-specific prompt-effect experiment on the bundled MMLU inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import interaction as experiment


DEFAULT_OUT = experiment.REPO / "outputs" / "mmlu_rasch389" / "results"


def checkpoint_path(root: Path, seed: int, variant: str) -> Path:
    return root / f"seed_{seed:02d}_{variant}.json"


def read_checkpoint(path: Path, subject: str, seed: int, variant: str, fingerprint: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["subject"], record["seed"], record["variant"], record["fingerprint"]) != (
        subject, seed, variant, fingerprint
    ):
        raise ValueError(f"Checkpoint configuration differs from this run: {path}")
    updates = record["updates"]
    bai = experiment.STATE["bai"]
    if (
        len(updates) != bai.MAX_BAI_PHASES
        or any(update.keys() != set(bai.UPDATE_FIELDS) for update in updates)
        or record["updates_sha256"] != experiment.digest_json({"updates": updates})
    ):
        raise ValueError(f"Checkpoint records are incomplete or differ from their hash: {path}")
    return record


def export_subject(out_dir: Path, root: Path, provenance: dict, fingerprint: str) -> Path | None:
    subject = provenance["subject"]
    paths = {
        (seed, variant): checkpoint_path(root, seed, variant)
        for seed in experiment.SEEDS
        for variant in experiment.VARIANTS
    }
    if any(not path.is_file() for path in paths.values()):
        return None
    records = {
        (seed, variant): read_checkpoint(path, subject, seed, variant, fingerprint)
        for (seed, variant), path in paths.items()
    }
    out = [
        [[records[seed, variant]["updates"] for variant in experiment.VARIANTS]]
        for seed in experiment.SEEDS
    ]
    target = out_dir / f"bai_results_MMLU_{subject}_combined.npy"
    temporary = target.with_suffix(f".{os.getpid()}.tmp.npy")
    bai = experiment.STATE["bai"]
    bai.save_bai_raw(
        str(temporary),
        out=out,
        jobs=[(subject, seed) for seed in experiment.SEEDS],
        tasks=[subject],
        seeds=experiment.SEEDS,
        bench="MMLU",
        combine_models=True,
        costs_recorded=True,
        n_models_stacked=experiment.N_MODELS,
        out_by_task=[out],
        cost_file=experiment.STATE["prices_path"].relative_to(experiment.REPO).as_posix(),
    )
    payload = np.load(temporary, allow_pickle=True).item()
    payload["feature_names"] = experiment.VARIANTS
    payload["reproduction"] = provenance | {
        "fingerprint": fingerprint,
        "checkpoint_sha256": {f"{seed}:{variant}": experiment.sha256(path) for (seed, variant), path in paths.items()},
        "convergence_warning_counts": {
            f"{seed}:{variant}": len(record["convergence_warnings"])
            for (seed, variant), record in records.items()
        },
    }
    np.save(temporary, payload)
    temporary.replace(target)
    return target


def run_subject(subject: str, seeds: tuple[int, ...], variants: tuple[str, ...], out_dir: Path) -> None:
    matrix, features, provenance, fingerprint = experiment.prepare_subject(subject)
    root = out_dir / "checkpoints" / subject
    manifest = root / "manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError(f"Subject configuration differs from {manifest}; use a fresh output directory.")
    else:
        experiment.write_json(manifest, provenance | {"fingerprint": fingerprint})

    computed = reused = 0
    tick = time.perf_counter()
    for seed in seeds:
        for variant in variants:
            path = checkpoint_path(root, seed, variant)
            if path.is_file():
                read_checkpoint(path, subject, seed, variant, fingerprint)
                reused += 1
                continue
            begin = time.perf_counter()
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                updates = experiment.replay_case(matrix, features, provenance["observation_budget"], seed, variant)
            convergence = [str(item.message) for item in recorded if issubclass(item.category, ConvergenceWarning)]
            elapsed = time.perf_counter() - begin
            experiment.write_json(path, {
                "subject": subject, "seed": seed, "variant": variant, "fingerprint": fingerprint,
                "updates": updates, "updates_sha256": experiment.digest_json({"updates": updates}),
                "elapsed_seconds": elapsed, "convergence_warnings": convergence,
            })
            computed += 1
            print(json.dumps({
                "event": "trajectory_complete", "subject": subject, "seed": seed, "variant": variant,
                "seconds": round(elapsed, 3), "convergence_warnings": len(convergence),
            }), flush=True)
    archive = export_subject(out_dir, root, provenance, fingerprint)
    print(json.dumps({
        "event": "subject_complete", "subject": subject, "computed": computed, "reused": reused,
        "seconds": round(time.perf_counter() - tick, 2), "archive": archive.name if archive else None,
    }), flush=True)


def main() -> None:
    manifest = json.loads((experiment.MATRIX_DIR / "manifest.json").read_text(encoding="utf-8"))
    sizes = {row["task"]: row["shape"][1] for row in manifest["files"]}
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--subjects", nargs="+", choices=sorted(sizes))
    selection.add_argument("--all-subjects", action="store_true", help="Run all 57 MMLU subjects, the default.")
    parser.add_argument("--seeds", nargs="+", type=int, choices=experiment.SEEDS, default=experiment.SEEDS)
    parser.add_argument("--variants", nargs="+", choices=experiment.VARIANTS, default=experiment.VARIANTS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-chunk-size", type=int, default=2,
                        help="Seeds per independent subject job, to distribute large subjects across workers.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.seed_chunk_size < 1:
        parser.error("--seed-chunk-size must be at least one")
    subjects = sorted(set(args.subjects or sizes), key=lambda subject: (-sizes[subject], subject))
    seeds = tuple(sorted(set(args.seeds)))
    variants = tuple(variant for variant in experiment.VARIANTS if variant in args.variants)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_chunks = [seeds[start:start + args.seed_chunk_size] for start in range(0, len(seeds), args.seed_chunk_size)]
    with experiment.one_thread_workers():
        with ProcessPoolExecutor(
            max_workers=min(len(subjects) * len(seed_chunks), args.workers), initializer=experiment.initialize_worker,
        ) as pool:
            jobs = [pool.submit(run_subject, subject, chunk, variants, out_dir)
                    for subject in subjects for chunk in seed_chunks]
            for job in as_completed(jobs):
                job.result()


if __name__ == "__main__":
    main()
