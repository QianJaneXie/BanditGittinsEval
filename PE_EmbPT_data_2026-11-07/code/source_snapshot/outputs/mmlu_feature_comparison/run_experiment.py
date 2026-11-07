"""Replay the historical PromptEval BAI with its four MMLU feature blocks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np


HERE = Path(__file__).resolve().parent
HELPER_DIR = HERE.parent / "mmlu_onehot_reproduction"
sys.path.insert(0, str(HELPER_DIR))
import run_reproduction as replay


MATRIX_DIR = replay.REPO / "data" / "MMLU_matrices"
YS_PATH = replay.PROMPTEVAL / "data" / "Ys.pickle"
XS_PATH = replay.PROMPTEVAL / "data" / "Xs.pickle"
SEEDS = tuple(range(20))
FEATURE_NAMES = ("one_hot", "discrete", "pretrained_embedding", "fine_tuned_embedding")
BLOCKS = tuple(range(len(FEATURE_NAMES)))
STATE = {}


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{contiguous.dtype.str}:{contiguous.shape}".encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def initialize_worker() -> None:
    from threadpoolctl import threadpool_limits

    bai = replay.import_runner()
    # Use the already validated same-order sampler and sparse identity matrix.
    replay.install_optimized_onehot(bai)
    import methods

    sparse_identity = methods.GenXY

    def feature_design(seen_items, Y, X, Z):
        rows = np.nonzero(seen_items)[0]
        if X.shape == (Y.shape[0], Y.shape[0]):
            return sparse_identity(seen_items, Y, X, Z)
        # LogReg discards the item columns of the original GenXY output. It
        # still appends its own intercept and calls the same LIBLINEAR fit.
        return X[rows], Y[seen_items].astype(float)

    methods.GenXY = feature_design
    price_path = replay.REPO / bai.DEFAULT_COST_FILES["MMLU"]
    ys, _ = replay.load_trusted_pickle(lambda: replay.read_ys(YS_PATH))
    xs, _ = replay.load_trusted_pickle(lambda: replay.read_ys(XS_PATH))
    source_files = (
        Path(__file__).resolve(), HELPER_DIR / "run_reproduction.py",
        *(replay.PROMPTEVAL / name for name in ("bai_evaluation.py", "methods.py", "methods_gpu.py", "utils.py")),
    )
    STATE.update(
        bai=bai,
        ys=ys,
        xs=xs,
        prices=bai.load_arm_costs(str(price_path)),
        price_path=price_path,
        thread_limits=threadpool_limits(limits=1),
        provenance={
            "version": 1,
            "python_version": sys.version,
            "packages": replay.installed_packages(),
            "source_sha256": {path.relative_to(replay.REPO).as_posix(): replay.sha256(path) for path in source_files},
            "input_sha256": {path.relative_to(replay.REPO).as_posix(): replay.sha256(path) for path in (YS_PATH, XS_PATH, price_path)},
            "backend": "sklearn",
            "allocation": "historical successive halving with random unseen columns",
            "scoring": "historical arm-feature L2 logistic regression, C=100, LIBLINEAR, ranking by arm score",
            "fit_acceleration": "same-order NumPy sampler and equivalent sparse identity or dense arm-feature matrix",
            "cost_policy": "sampling and budget count observations; heterogeneous cost is recorded alongside",
            "feature_names": FEATURE_NAMES,
            "fine_tuned_embedding_note": "Historical stacking zero-pads different odd/even widths and shares feature indices across the two encoders.",
            "budget_fraction": bai.OBSERVATION_BUDGET_MAX_FRACTION,
            "phase_cap": bai.MAX_BAI_PHASES,
        },
    )


def prepare_subject(subject: str):
    bai = STATE["bai"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MMLU/.*: Xs view .* right-padded")
        matrix, views = bai.combine_models_y_xs(STATE["ys"], STATE["xs"], "MMLU", subject)
    matrix_path = MATRIX_DIR / f"{subject}.npy"
    if not np.array_equal(matrix, np.load(matrix_path)):
        raise ValueError(f"Stacked PromptEval data do not match {matrix_path}")
    if len(views) != len(BLOCKS) - 1 or STATE["prices"].shape != (matrix.shape[0],):
        raise ValueError(f"Unexpected feature count or price shape for {subject}")

    budget = bai.budgets_from_y(matrix)[0]
    provenance = STATE["provenance"] | {
        "subject": subject,
        "matrix_shape": list(matrix.shape),
        "matrix_sha256": replay.sha256(matrix_path),
        "observation_budget": budget,
        "feature_sha256": {
            FEATURE_NAMES[0]: digest_json({"feature": "identity", "dimension": matrix.shape[0]}),
            **{name: digest_array(view) for name, view in zip(FEATURE_NAMES[1:], views, strict=True)},
        },
        "feature_widths": [matrix.shape[0], *(view.shape[1] for view in views)],
        "stacked_models": len(STATE["ys"]["MMLU"][subject]),
    }
    return matrix, (None, *views), provenance, digest_json(provenance)


def checkpoint_path(root: Path, seed: int, block: int) -> Path:
    return root / f"seed_{seed:02d}_block_{block}.json"


def load_checkpoint(path: Path, subject: str, seed: int, block: int, fingerprint: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("subject"), result.get("seed"), result.get("block"), result.get("fingerprint")) != (
        subject, seed, block, fingerprint
    ):
        raise ValueError(f"Checkpoint configuration differs from this run: {path}")
    updates = result.get("updates", [])
    if len(updates) != STATE["bai"].MAX_BAI_PHASES or any(
        update.keys() != set(STATE["bai"].UPDATE_FIELDS) for update in updates
    ):
        raise ValueError(f"Checkpoint lacks the expected historical phase records: {path}")
    return result


def historical_updates(subject: str) -> tuple[dict, str | None]:
    path = replay.PROMPTEVAL / "results" / f"bai_results_MMLU_{subject}_combined.npy"
    if not path.is_file():
        return {}, None
    payload, _ = replay.load_trusted_pickle(lambda: np.load(path, allow_pickle=True).item())
    if payload["update_fields"] != STATE["bai"].UPDATE_FIELDS:
        raise ValueError(f"Unexpected historical schema: {path}")
    updates = {
        (seed, block): block_updates
        for (task, seed), run in zip(payload["jobs"], payload["out"], strict=True)
        if task == subject
        for block, block_updates in enumerate(run[0])
    }
    return updates, replay.sha256(path)


def export_subject(out_dir: Path, root: Path, subject: str, provenance: dict, fingerprint: str) -> Path | None:
    if any(not checkpoint_path(root, seed, block).is_file() for seed in SEEDS for block in BLOCKS):
        return None
    bai = STATE["bai"]
    out = [
        [[load_checkpoint(checkpoint_path(root, seed, block), subject, seed, block, fingerprint)["updates"] for block in BLOCKS]]
        for seed in SEEDS
    ]
    target = out_dir / f"bai_results_MMLU_{subject}_combined.npy"
    temporary = target.with_suffix(f".{os.getpid()}.tmp.npy")
    bai.save_bai_raw(
        str(temporary),
        out=out,
        jobs=[(subject, seed) for seed in SEEDS],
        tasks=[subject],
        seeds=SEEDS,
        bench="MMLU",
        combine_models=True,
        costs_recorded=True,
        n_models_stacked=provenance["stacked_models"],
        out_by_task=[out],
        cost_file=STATE["price_path"].relative_to(replay.REPO).as_posix(),
    )
    payload = np.load(temporary, allow_pickle=True).item()
    payload["reproduction"] = provenance | {"fingerprint": fingerprint}
    payload["feature_names"] = FEATURE_NAMES
    np.save(temporary, payload)
    temporary.replace(target)
    return target


def run_subject(subject: str, seeds: tuple[int, ...], blocks: tuple[int, ...], out_dir: Path) -> dict:
    bai = STATE["bai"]
    matrix, features, provenance, fingerprint = prepare_subject(subject)
    root = out_dir / "checkpoints" / subject
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(f"Subject configuration differs from {manifest_path}; use a fresh output directory")
    else:
        replay.write_json(manifest_path, provenance | {"fingerprint": fingerprint})

    historical, historical_hash = historical_updates(subject)
    counts = {"computed": 0, "reused": 0, "historical_cases": 0, "historical_exact": 0}
    comparisons = []
    tick = time.perf_counter()
    for seed in seeds:
        for block in blocks:
            path = checkpoint_path(root, seed, block)
            if path.is_file():
                result = load_checkpoint(path, subject, seed, block, fingerprint)
                counts["reused"] += 1
            else:
                start = time.perf_counter()
                updates = bai.compute_regrets(
                    provenance["observation_budget"], matrix, features[block],
                    random_seed=seed, backend="sklearn", costs=STATE["prices"],
                )
                result = {
                    "subject": subject, "seed": seed, "block": block, "feature_name": FEATURE_NAMES[block],
                    "fingerprint": fingerprint, "updates": updates, "elapsed_seconds": time.perf_counter() - start,
                }
                replay.write_json(path, result)
                counts["computed"] += 1
            if (seed, block) in historical:
                comparison = replay.compare_updates(historical[seed, block], result["updates"], bai.UPDATE_FIELDS)
                comparisons.append({"seed": seed, "block": block, **comparison})
                counts["historical_cases"] += 1
                counts["historical_exact"] += comparison["exact_match"]

    if comparisons:
        replay.write_json(
            root / "historical_validation.json",
            {"subject": subject, "saved_trace_sha256": historical_hash, "comparisons": comparisons},
        )
    archive = export_subject(out_dir, root, subject, provenance, fingerprint)
    return {
        "subject": subject, "seconds": round(time.perf_counter() - tick, 2), **counts,
        "archive": str(archive) if archive else None,
    }


@contextmanager
def one_thread_worker_environment():
    variables = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    original = {name: os.environ.get(name) for name in variables}
    try:
        os.environ.update({name: "1" for name in variables})
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    manifest = json.loads((MATRIX_DIR / "manifest.json").read_text(encoding="utf-8"))
    sizes = {row["task"]: row["shape"][1] for row in manifest["files"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", choices=sorted(sizes), default=sorted(sizes))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--blocks", nargs="+", type=int, choices=BLOCKS, default=BLOCKS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--require-historical-match", action="store_true", help="Fail unless all selected cases have an exact saved match.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    subjects = sorted(set(args.subjects), key=lambda subject: (-sizes[subject], subject))
    seeds, blocks = tuple(sorted(set(args.seeds))), tuple(sorted(set(args.blocks)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exact = 0
    with one_thread_worker_environment():
        with ProcessPoolExecutor(max_workers=min(len(subjects), args.workers), initializer=initialize_worker) as pool:
            jobs = [pool.submit(run_subject, subject, seeds, blocks, args.out_dir.resolve()) for subject in subjects]
            for job in as_completed(jobs):
                summary = job.result()
                exact += summary["historical_exact"]
                print(json.dumps(summary), flush=True)
    return int(args.require_historical_match and exact != len(subjects) * len(seeds) * len(blocks))


if __name__ == "__main__":
    raise SystemExit(main())
