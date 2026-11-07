"""Replay pooled MMLU BAI with X39 plus one question effect per item."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
LEGACY_DIR = HERE.parent / "mmlu_feature_comparison"
sys.path.insert(0, str(LEGACY_DIR))
import run_experiment as legacy


REPLAY = legacy.replay
SEEDS = legacy.SEEDS
VARIANTS = ("rasch39_random", "rasch39_balanced")
PRETRAINED_BLOCK = legacy.FEATURE_NAMES.index("pretrained_embedding")
FIGURE3_TASKS = REPLAY.REPO / "Figure3_data_2026-11-07" / "data" / "selected_tasks.csv"
STATE = {}


def sparse_rasch_design(seen_items, y, x, z):
    """Return native GenXY's row-major observations in CSR form."""
    n_questions = y.shape[1]
    if z.shape != (n_questions, n_questions) or np.count_nonzero(z) != n_questions or not np.all(z.diagonal() == 1):
        raise ValueError("The Rasch comparison requires the original question identity features.")
    arms, questions = np.nonzero(seen_items)
    item_features = sparse.csr_matrix(
        (np.ones(len(arms)), questions, np.arange(len(arms) + 1)),
        shape=(len(arms), n_questions),
    )
    return sparse.hstack((sparse.csr_matrix(x[arms]), item_features), format="csr"), y[arms, questions].astype(float)


def check_multicollinearity(x, tol=1e-6):
    # Same criterion as the upstream utils function, without its model-loading dependencies.
    pca = PCA().fit(x)
    if not np.all(pca.explained_variance_ratio_ > tol):
        raise AssertionError("The Rasch prompt/model covariates are not full rank.")


def sample_balanced(seen_examples, max_seen, random_seed, active_arms=None, random_column=False):
    """Replay native StratSample with the least-covered available question rule."""
    if not isinstance(active_arms, (list, np.ndarray)):
        raise ValueError("Expected the list of active BAI arms.")
    active = np.asarray(active_arms, dtype=np.intp)
    rng = np.random.RandomState(random_seed)
    row_counts = seen_examples.sum(axis=1)
    column_counts = seen_examples.sum(axis=0)
    n_seen = int(row_counts.sum())
    while n_seen < max_seen:
        active_counts = row_counts[active]
        row = rng.choice(active[active_counts == active_counts.min()])
        available = np.flatnonzero(~seen_examples[row])
        if not available.size:
            break
        available_counts = column_counts[available]
        column = rng.choice(available[available_counts == available_counts.min()])
        seen_examples[row, column] = True
        row_counts[row] += 1
        column_counts[column] += 1
        n_seen += 1
    return seen_examples


class RaschScore:
    """Adapt native PE row scores to the local phase-by-phase BAI driver."""

    def fit(self, seen, matrix, features):
        model = STATE["methods"].ExtendedRaschModel()
        model.fit(seen, matrix, features, Z=None)
        self.thetas = model.get_Y_hat().mean(axis=1)


def initialize_worker() -> None:
    legacy.initialize_worker()
    import methods

    methods.GenXY = sparse_rasch_design
    methods.check_multicolinearity = check_multicollinearity
    STATE.update(methods=methods, random_sampler=legacy.STATE["bai"].StratSample)


def replay_case(matrix, features, budget, seed, variant):
    bai = legacy.STATE["bai"]
    old_model, old_sampler = bai.LogReg, bai.StratSample
    bai.LogReg = RaschScore
    bai.StratSample = STATE["random_sampler"] if variant == VARIANTS[0] else sample_balanced
    try:
        return bai.compute_regrets(
            budget, matrix, features, random_seed=seed, backend="sklearn", costs=legacy.STATE["prices"],
        )
    finally:
        bai.LogReg, bai.StratSample = old_model, old_sampler


def archive_path(root: Path, subject: str) -> Path:
    return root / f"bai_results_MMLU_{subject}_combined.npy"


def checkpoint_path(root: Path, seed: int, variant: str) -> Path:
    return root / f"seed_{seed:02d}_{variant}.json"


def read_checkpoint(path: Path, subject: str, seed: int, variant: str, fingerprint: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["subject"], record["seed"], record["variant"], record["fingerprint"]) != (
        subject, seed, variant, fingerprint
    ):
        raise ValueError(f"Checkpoint configuration differs: {path}")
    updates = record["updates"]
    bai = legacy.STATE["bai"]
    if len(updates) != bai.MAX_BAI_PHASES or any(update.keys() != set(bai.UPDATE_FIELDS) for update in updates):
        raise ValueError(f"Checkpoint does not contain the complete historical phase schema: {path}")
    if record["updates_sha256"] != legacy.digest_json({"updates": updates}):
        raise ValueError(f"Checkpoint records differ from their hash: {path}")
    return record


def export_subject(out_dir: Path, root: Path, provenance: dict, fingerprint: str) -> Path | None:
    subject = provenance["subject"]
    paths = {(seed, variant): checkpoint_path(root, seed, variant) for seed in SEEDS for variant in VARIANTS}
    if any(not path.is_file() for path in paths.values()):
        return None
    records = {
        (seed, variant): read_checkpoint(path, subject, seed, variant, fingerprint)
        for (seed, variant), path in paths.items()
    }
    out = [[[records[seed, variant]["updates"] for variant in VARIANTS]] for seed in SEEDS]
    target = archive_path(out_dir, subject)
    temporary = target.with_suffix(f".{os.getpid()}.tmp.npy")
    bai = legacy.STATE["bai"]
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
        cost_file=legacy.STATE["price_path"].relative_to(REPLAY.REPO).as_posix(),
    )
    payload = np.load(temporary, allow_pickle=True).item()
    payload["feature_names"] = VARIANTS
    payload["reproduction"] = provenance | {
        "fingerprint": fingerprint,
        "checkpoint_sha256": {f"{seed}:{variant}": REPLAY.sha256(path) for (seed, variant), path in paths.items()},
    }
    np.save(temporary, payload)
    temporary.replace(target)
    return target


def prepare_subject(subject: str, reference_dir: Path):
    matrix, features, old, old_fingerprint = legacy.prepare_subject(subject)
    x39 = features[PRETRAINED_BLOCK]
    if x39.shape != (1500, 39) or not np.array_equal(x39[:, -14:], np.repeat(np.eye(15)[:, 1:], 100, axis=0)):
        raise ValueError(f"Expected the archived 25 prompt features and 14 model indicators for {subject}.")
    reference = archive_path(reference_dir, subject)
    archived, _ = REPLAY.load_trusted_pickle(lambda: np.load(reference, allow_pickle=True).item())
    if (
        archived["tasks"] != [subject]
        or archived["seeds"] != list(SEEDS)
        or tuple(archived["feature_names"]) != legacy.FEATURE_NAMES
        or archived["reproduction"]["fingerprint"] != old_fingerprint
    ):
        raise ValueError(f"The archived logistic comparison differs from the loaded pipeline: {reference}")

    source = Path(__file__).resolve()
    x39_hash = legacy.digest_array(x39)
    provenance = old | {
        "source_sha256": old["source_sha256"] | {source.relative_to(REPLAY.REPO).as_posix(): REPLAY.sha256(source)},
        "feature_names": VARIANTS,
        "feature_widths": [39 + matrix.shape[1]] * len(VARIANTS),
        "feature_sha256": {variant: x39_hash for variant in VARIANTS},
        "prompt_feature_sha256": x39_hash,
        "prompt_feature_width": 39,
        "question_feature_width": matrix.shape[1],
        "fitted_coefficients": 39 + matrix.shape[1],
        "reference_logistic_archive_sha256": REPLAY.sha256(reference),
        "reference_logistic_fingerprint": old_fingerprint,
        "allocation": "Pooled 1,500-arm successive halving with the historical five-phase 10% observation budget.",
        "samplers": {
            VARIANTS[0]: "Same random unseen questions and least-sampled active arm rule as the pooled logistic comparison.",
            VARIANTS[1]: "Upstream PE least-covered unseen question rule using column counts across all rows.",
        },
        "scoring": "ExtendedRaschModel: keep observed labels and use fitted probabilities for unseen entries, then average across questions.",
        "fit_acceleration": "Equivalent CSR design, native L2 LIBLINEAR logistic regression at C=100 and no separate intercept.",
        "recommendation": "Recommend highest PE row score among active arms at each of the five phases, using historical tie behavior.",
        "adaptation": "Pooled model-prompt candidates and phase-level recommendations; upstream per-model tasks and terminal empirical selection are separate.",
    }
    return matrix, x39, provenance, legacy.digest_json(provenance)


def run_subject(subject: str, seeds: tuple[int, ...], variants: tuple[str, ...], out_dir: Path, reference_dir: Path):
    matrix, x39, provenance, fingerprint = prepare_subject(subject, reference_dir)
    root = out_dir / "checkpoints" / subject
    manifest = root / "manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError(f"Subject configuration differs from {manifest}; use a fresh output directory.")
    else:
        REPLAY.write_json(manifest, provenance | {"fingerprint": fingerprint})

    timings = {}
    reused = 0
    tick = time.perf_counter()
    for seed in seeds:
        for variant in variants:
            path = checkpoint_path(root, seed, variant)
            if path.is_file():
                read_checkpoint(path, subject, seed, variant, fingerprint)
                reused += 1
                continue
            begin = time.perf_counter()
            updates = replay_case(matrix, x39, provenance["observation_budget"], seed, variant)
            elapsed = time.perf_counter() - begin
            REPLAY.write_json(path, {
                "subject": subject, "seed": seed, "variant": variant, "fingerprint": fingerprint,
                "updates": updates, "updates_sha256": legacy.digest_json({"updates": updates}), "elapsed_seconds": elapsed,
            })
            timings[f"{seed}:{variant}"] = round(elapsed, 3)
    archive = export_subject(out_dir, root, provenance, fingerprint)
    return {
        "subject": subject, "computed": len(timings), "reused": reused, "case_seconds": timings,
        "seconds": round(time.perf_counter() - tick, 2), "archive": str(archive) if archive else None,
    }


def main() -> None:
    manifest = json.loads((legacy.MATRIX_DIR / "manifest.json").read_text(encoding="utf-8"))
    sizes = {row["task"]: row["shape"][1] for row in manifest["files"]}
    with FIGURE3_TASKS.open(newline="", encoding="utf-8-sig") as handle:
        default_subjects = [row["task"] for row in csv.DictReader(handle)]
    if len(default_subjects) != 19 or len(set(default_subjects)) != 19 or not set(default_subjects) <= sizes.keys():
        raise ValueError("Expected the 19 verified Figure 3 subjects.")
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--subjects", nargs="+", choices=sorted(sizes), default=default_subjects)
    selection.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--reference-dir", type=Path, default=LEGACY_DIR / "results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    subjects = sorted(sizes if args.all_subjects else set(args.subjects), key=lambda subject: (-sizes[subject], subject))
    seeds = tuple(sorted(set(args.seeds)))
    variants = tuple(variant for variant in VARIANTS if variant in args.variants)
    out_dir, reference_dir = args.out_dir.resolve(), args.reference_dir.resolve()
    if out_dir == reference_dir:
        parser.error("Write Rasch results to a separate directory from the archived logistic results.")
    out_dir.mkdir(parents=True, exist_ok=True)
    with legacy.one_thread_worker_environment():
        with ProcessPoolExecutor(
            max_workers=min(len(subjects), args.workers), initializer=initialize_worker,
        ) as pool:
            jobs = [pool.submit(run_subject, subject, seeds, variants, out_dir, reference_dir) for subject in subjects]
            for job in as_completed(jobs):
                print(json.dumps(job.result()), flush=True)


if __name__ == "__main__":
    main()
