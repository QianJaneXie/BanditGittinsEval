"""Replay the submitted one-hot PromptEval BAI on its two saved MMLU subjects."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import types

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROMPTEVAL = REPO / "prompteval"
SUBJECTS = ("abstract_algebra", "professional_law")
FLOAT_FIELDS = {"budget_cost", "chosen_mean", "oracle_mean", "simple_regret"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def installed_packages() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "tqdm", "joblib"):
        distribution = importlib.metadata.distribution(name)
        record = distribution.read_text("RECORD")
        packages[name] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None,
        }
    return packages


def import_runner():
    # The historical sklearn one-hot path uses no utils functions. Importing the
    # full utils module unnecessarily requires Torch and Transformers.
    def unexpected_covariate_check(*args, **kwargs):
        raise RuntimeError("The historical one-hot path should never check covariates.")

    utils = types.ModuleType("utils")
    utils.check_multicolinearity = unexpected_covariate_check
    sys.modules["utils"] = utils
    sys.path.insert(0, str(PROMPTEVAL))
    import bai_evaluation

    return bai_evaluation


def install_optimized_onehot(bai) -> None:
    from scipy.sparse import csr_matrix
    import methods

    def sparse_gen_xy(seen_items, Y, X, Z):
        n_arms = Y.shape[0]
        if X.shape != (n_arms, n_arms) or np.count_nonzero(X) != n_arms or not np.all(X.diagonal() == 1):
            raise ValueError("The sparse replay only supports the original one-hot arm features.")
        rows = np.nonzero(seen_items)[0]
        features = csr_matrix(
            (np.ones(len(rows)), rows, np.arange(len(rows) + 1)),
            shape=(len(rows), n_arms),
        )
        return features, Y[seen_items].astype(float)

    def strat_sample(seen_examples, max_seen, random_seed, active_arms=None, random_column=False):
        if type(active_arms) not in (list, np.ndarray) or not random_column:
            raise ValueError("The optimized replay only supports the original BAI random-column path.")
        # Keep the supplied order of active arms and ascending order of unseen
        # columns. Repeat the original two RandomState.choice calls per sample.
        active = np.asarray(active_arms, dtype=np.intp)
        rng = np.random.RandomState(random_seed)
        counts = seen_examples.sum(axis=1)
        n_seen = counts.sum()
        while n_seen < max_seen:
            active_counts = counts[active]
            row = rng.choice(active[active_counts == active_counts.min()])
            columns = np.flatnonzero(~seen_examples[row])
            if not columns.size:
                break
            column = rng.choice(columns)
            seen_examples[row, column] = True
            counts[row] += 1
            n_seen += 1
        return seen_examples

    methods.GenXY = sparse_gen_xy
    bai.StratSample = strat_sample


def load_trusted_pickle(reader):
    try:
        return reader(), False
    except ModuleNotFoundError as error:
        if error.name not in {"numpy._core", "numpy._core.multiarray"}:
            raise
        # NumPy 2 moved the internal pickle import path used by the saved runs.
        # This alias resolves their arrays without changing their contents.
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        return reader(), True


def read_ys(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_historical(path: Path):
    payload, used_alias = load_trusted_pickle(lambda: np.load(path, allow_pickle=True).item())
    historical = {}
    for (task, seed), blocks in zip(payload["jobs"], payload["out"], strict=True):
        if (task, seed) in historical:
            raise ValueError(f"Duplicate historical job for {task}, seed {seed}")
        historical[task, seed] = blocks[0][0]
    return payload, historical, used_alias


def compare_updates(old, new, fields):
    if len(old) != len(new):
        return {
            "exact_match": False,
            "within_tolerance": False,
            "phase_counts": {"historical": len(old), "rerun": len(new)},
            "differences": [],
        }
    differences = []
    for index, (historical, rerun) in enumerate(zip(old, new, strict=True)):
        for field in fields:
            before, after = historical[field], rerun[field]
            if before == after:
                continue
            close = (
                field in FLOAT_FIELDS
                and before is not None
                and after is not None
                and math.isclose(before, after, rel_tol=1e-12, abs_tol=1e-12)
            )
            differences.append(
                {"phase": index, "field": field, "historical": before, "rerun": after, "within_tolerance": close}
            )
    return {
        "exact_match": not differences,
        "within_tolerance": all(item["within_tolerance"] for item in differences),
        "phase_counts": {"historical": len(old), "rerun": len(new)},
        "differences": differences,
    }


def output_path(out_dir: Path, subject: str, seed: int, first_phase: bool = False) -> Path:
    prefix = "first_phase_seed" if first_phase else "seed"
    return out_dir / subject / f"{prefix}_{seed:02d}.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def reproduce(args) -> bool:
    bai = import_runner()
    if args.optimized:
        install_optimized_onehot(bai)
    ys_path = PROMPTEVAL / "data" / "Ys.pickle"
    ys, ys_alias = load_trusted_pickle(lambda: read_ys(ys_path))
    price_path = REPO / bai.DEFAULT_COST_FILES["MMLU"]
    prices = bai.load_arm_costs(str(price_path))
    common_files = [
        Path(__file__).resolve(), ys_path, price_path,
        *(PROMPTEVAL / name for name in ("bai_evaluation.py", "methods.py", "methods_gpu.py", "utils.py")),
    ]
    common_hashes = {str(path.relative_to(REPO)): sha256(path) for path in common_files}
    runtime = {
        "executable": sys.executable,
        "python": sys.version,
        "packages": installed_packages(),
        "utils_import": "Guarded because the legacy sklearn one-hot path uses no utils functions.",
        "legacy_sampler_and_feature_builder": "vectorized same-order sampler, sparse identity design"
        if args.optimized else "unmodified",
    }

    matched = True
    for subject in args.subjects:
        saved_path = PROMPTEVAL / "results" / f"bai_results_MMLU_{subject}_combined.npy"
        old, old_by_job, old_alias = load_historical(saved_path)
        if old["bench"] != "MMLU" or not old["combine_models"] or not old["costs_recorded"]:
            raise ValueError(f"Unexpected historical configuration for {subject}")
        if list(old["update_fields"]) != bai.UPDATE_FIELDS or REPO / old["cost_file"] != price_path:
            raise ValueError(f"Historical output schema or cost file differs for {subject}")
        matrix_path = REPO / "data" / "MMLU_matrices" / f"{subject}.npy"
        subject_models = ys["MMLU"][subject]
        matrix = np.vstack(subject_models)
        if len(subject_models) != old["n_models_stacked"] or not np.array_equal(matrix, np.load(matrix_path)):
            raise ValueError(f"Stacked PromptEval data do not match the benchmark matrix for {subject}")
        hashes = common_hashes | {str(path.relative_to(REPO)): sha256(path) for path in (matrix_path, saved_path)}
        budget = bai.budgets_from_y(matrix)[0]
        first_phase_only = args.first_phase_only
        phase_cap = bai.MAX_BAI_PHASES
        if first_phase_only:
            n_phases = min(int(np.ceil(np.log2(matrix.shape[0]))), phase_cap)
            budget -= (n_phases - 1) * (budget // n_phases)
        for seed in args.seeds:
            historical = old_by_job[subject, seed]
            if first_phase_only:
                historical = historical[:1]
            print(f"Start {subject}, seed {seed}, {budget:,} observations", flush=True)
            tick = time.perf_counter()
            try:
                if first_phase_only:
                    bai.MAX_BAI_PHASES = 1
                rerun = bai.compute_regrets(budget, matrix, None, random_seed=seed, backend="sklearn", costs=prices)
            finally:
                bai.MAX_BAI_PHASES = phase_cap
            elapsed = time.perf_counter() - tick
            comparison = compare_updates(historical, rerun, bai.UPDATE_FIELDS)
            matched = matched and comparison["within_tolerance"]
            result = {
                "subject": subject,
                "seed": seed,
                "backend": "sklearn",
                "feature_block": 0,
                "optimized": args.optimized,
                "matrix_shape": list(matrix.shape),
                "observation_budget": budget,
                "first_phase_only": first_phase_only,
                "exhaustive_observations": int(matrix.size),
                "exhaustive_cost": float(matrix.shape[1] * prices.sum()),
                "elapsed_seconds": elapsed,
                "historical": historical,
                "rerun": rerun,
                "comparison": comparison,
                "source_and_input_sha256": hashes,
                "runtime": runtime,
                "numpy_legacy_pickle_alias": {
                    "installed": "numpy._core" in sys.modules and int(np.__version__.split(".")[0]) < 2,
                    "triggered_on_response_pickle": ys_alias,
                    "triggered_on_historical_trace": old_alias,
                },
            }
            write_json(output_path(args.out_dir, subject, seed, first_phase_only), result)
            status = "exact match" if comparison["exact_match"] else (
                "match within tolerance" if comparison["within_tolerance"] else "mismatch"
            )
            print(f"Finish {subject}, seed {seed}, {elapsed:.2f}s, {status}", flush=True)
    return matched


def plot_results(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bai = import_runner()
    for subject in args.subjects:
        results = [
            json.loads(output_path(args.out_dir, subject, seed).read_text(encoding="utf-8"))
            for seed in args.seeds
        ]
        cost_grid = np.unique([
            phase["budget_cost"]
            for result in results
            for key in ("historical", "rerun")
            for phase in result[key]
        ])
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharey=True, layout="constrained")
        for ax, cost_aware, mode in zip(axes, (False, True), ("Unit cost", "Heterogeneous cost"), strict=True):
            for key, label, color, marker, style in (
                ("historical", "Saved one-hot", "#1665a8", "o", "-"),
                ("rerun", "One-hot rerun", "#bd4c19", "x", "--"),
            ):
                if cost_aware:
                    # Match the historical processed curve. Use the actual spend
                    # at each completed phase, then hold the previous recommendation.
                    x = 100 * cost_grid / results[0]["exhaustive_cost"]
                    y = np.array([
                        bai._step_regret_on_grid(
                            np.array([p["budget_cost"] for p in result[key]]),
                            np.array([p["simple_regret"] for p in result[key]], dtype=float),
                            cost_grid,
                        )
                        for result in results
                    ])
                else:
                    x = np.array([
                        [100 * p["budget_obs"] / r["exhaustive_observations"] for p in r[key]]
                        for r in results
                    ]).mean(axis=0)
                    y = np.array([[p["simple_regret"] for p in r[key]] for r in results], dtype=float)
                counts = np.isfinite(y).sum(axis=0)
                mean = np.divide(np.nansum(y, axis=0), counts, out=np.full(x.shape, np.nan), where=counts > 0)
                if len(results) > 1:
                    squared_deviations = np.nansum((y - mean) ** 2, axis=0)
                    se = np.sqrt(np.divide(
                        squared_deviations,
                        counts * (counts - 1),
                        out=np.full(x.shape, np.nan),
                        where=counts > 1,
                    ))
                    if key == "historical":
                        ax.fill_between(x, mean - se, mean + se, step="post", color=color, alpha=0.16)
                ax.step(
                    x, mean, where="post", label=label, color=color,
                    marker=None if cost_aware else marker, linestyle=style,
                )
            ax.set(title=mode, xlabel="Exhaustive evaluation spend (%)")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Simple regret")
        axes[0].legend(frameon=False)
        fig.suptitle(f"MMLU {subject.replace('_', ' ')} · {len(results)} paired seeds")
        path = args.out_dir / subject / "onehot_reproduction.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(path, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", "--subject", nargs="+", choices=SUBJECTS, default=SUBJECTS)
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, choices=range(20), default=list(range(20)))
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--optimized", action="store_true", help="Use the equivalent vectorized sampler and sparse one-hot design.")
    parser.add_argument("--first-phase-only", action="store_true", help="Replay just the first original phase for validation.")
    parser.add_argument("--plot-only", action="store_true", help="Plot saved comparison records for the selected seeds.")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args)
        return 0
    return 0 if reproduce(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
