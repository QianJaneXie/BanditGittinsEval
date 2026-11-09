#!/usr/bin/env python3
"""Paired GSM8K independent and configuration-correlated Gaussian TS.

Both policies use fixed-benchmark empirical targets, B=8 and LCB coefficient
one by default. Configuration correlations come from the same categorical
kernel family used by the existing mixed-input BO surrogate. Prices affect
the stopping budget, never the sampled accuracy objective. Outputs stay local.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compare_gittins_ts import (  # noqa: E402
    PRIORS, _atomic_npz, exact_normalized_auc, load_paper_costs,
)
from run_mfkg_baseline import load_data  # noqa: E402

MODEL_VERSION = "categorical_empirical_correlated_ts_v1"
METHODS = ("ts", "correlated_ts")
DEFAULT_REFERENCE_DIR = ROOT / "outputs/gsm8k_gittins_ts_batch_lcb100"


def simulate_correlated_ts(
    matrix: np.ndarray, *, features: torch.Tensor, seed: int,
    prior_mean: float, prior_variance: float, batch_size: int,
    per_arm_cost: np.ndarray, cost_mode: str, budget_fraction: float = 0.1,
    tau_sq_cell: float = 0.25, std_penalty: float = 1.0,
    kernel_mode: str = "fit", fit_every: int = 32, fit_maxiter: int = 25,
    max_steps: int | None = None, record_cells: bool = True,
) -> dict[str, Any]:
    """Supply the policy with revealed batches only, and score after feedback."""
    from correlated_ts_policy import CorrelatedGaussianTS

    raw = np.asarray(matrix)
    if raw.ndim != 2 or min(raw.shape) < 1 or not np.isfinite(raw).all():
        raise ValueError("matrix must be finite and nonempty")
    if cost_mode not in {"unit", "cost"}:
        raise ValueError("cost_mode must be unit or cost")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(budget_fraction) or not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must lie in (0,1]")
    if not math.isfinite(std_penalty) or std_penalty < 0:
        raise ValueError("std_penalty must be finite and nonnegative")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    n_arms, n_examples = raw.shape
    costs = np.asarray(per_arm_cost, dtype=np.float64)
    if costs.shape != (n_arms,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("per_arm_cost must be positive, finite and row-aligned")
    if tuple(features.shape[:1]) != (n_arms,):
        raise ValueError("features must align with matrix rows")
    model = CorrelatedGaussianTS(
        features, n_examples=n_examples, prior_mean=prior_mean,
        prior_variance=prior_variance, tau_sq_cell=tau_sq_cell,
        kernel_mode=kernel_mode, fit_every=fit_every, fit_maxiter=fit_maxiter,
    )
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observations = np.asarray(raw, dtype=np.float32)
    true_means = np.asarray(raw, dtype=np.float64).mean(axis=1)
    best = float(true_means.max())
    initial_regret = best - float(true_means[0])
    seen = np.zeros((n_arms, n_examples), dtype=bool)
    if cost_mode == "unit":
        full_budget = float(n_arms * n_examples)
        budget = float(max(1, round(budget_fraction * full_budget)))
    else:
        full_budget = float(n_examples * costs.sum())
        budget = budget_fraction * full_budget
    history = {name: [] for name in (
        "x", "x_original_cost", "recommended_arm", "recommended_mean", "regret",
        "pulled_arm", "n_new_samples",
    )}
    columns_history, fit_history = [], []
    evaluated, spent, stop_reason = 0, 0.0, "budget"
    initial_kernel = json.loads(json.dumps(model.kernel_metadata))
    while (evaluated if cost_mode == "unit" else spent) < budget:
        if max_steps is not None and len(history["x"]) >= max_steps:
            stop_reason = "max_steps"
            break
        complete = model.counts == n_examples
        if bool(complete.all()):
            stop_reason = "all_arms_complete"
            break
        draws = model.sample_target(generator=generator).masked_fill(complete, -torch.inf)
        arm = int(draws.argmax())
        unseen = np.flatnonzero(~seen[arm])
        size = min(batch_size, len(unseen))
        if cost_mode == "unit":
            size = min(size, int(budget - evaluated))
        order = torch.randperm(len(unseen), generator=generator).numpy()
        columns = unseen[order[:size]]
        seen[arm, columns] = True
        old_fit_count = model.fit_count
        model.observe(arm, torch.as_tensor(observations[arm, columns], dtype=torch.float64))
        evaluated += size
        spent += float(torch.full((size,), costs[arm], dtype=torch.float64).sum())
        if model.fit_count != old_fit_count:
            kernel = model.kernel_metadata
            fit_history.append({"batch": len(history["x"]) + 1, "evaluations": evaluated,
                                "kernel": {k: v for k, v in kernel.items() if k != "fit_diagnostics"},
                                "diagnostic": kernel.get("fit_diagnostics", [None])[-1]})
        means, covariance = model.empirical_target_moments()
        recommendation = int((means - std_penalty * covariance.diagonal().clamp_min(0).sqrt()).argmax())
        values = dict(
            x=evaluated, x_original_cost=spent, recommended_arm=recommendation,
            recommended_mean=float(means[recommendation]),
            regret=best - float(true_means[recommendation]), pulled_arm=arm,
            n_new_samples=size,
        )
        for name, value in values.items():
            history[name].append(value)
        if record_cells:
            padded = np.full(batch_size, -1, dtype=np.int64)
            padded[:size] = columns
            columns_history.append(padded)
    integer = {"x", "recommended_arm", "pulled_arm", "n_new_samples"}
    out = {k: np.asarray(v, dtype=np.int64 if k in integer else np.float64)
           for k, v in history.items()}
    out.update(
        policy="correlated_ts", cost_mode=cost_mode, budget=budget,
        full_budget=full_budget, counts=model.counts.numpy().copy(),
        observed_sums=model.sums.numpy().copy(), elapsed_s=time.perf_counter() - started,
        stop_reason=stop_reason, initial_recommended_arm=0, initial_regret=initial_regret,
        fit_count=model.fit_count, kernel_initial_json=json.dumps(initial_kernel),
        kernel_final_json=json.dumps(model.kernel_metadata),
        kernel_fit_history_json=json.dumps(fit_history),
    )
    if record_cells:
        out["sampled_columns"] = np.asarray(columns_history, dtype=np.int64).reshape(-1, batch_size)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _simulator_sha256(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    node = next((n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == "simulate_correlated_ts"), None)
    if node is None:
        raise ValueError(f"Missing correlated TS simulator in source snapshot: {path}")
    return hashlib.sha256(ast.get_source_segment(source, node).strip().encode()).hexdigest()


def _case_names(settings: dict):
    for matrix in settings["matrix_seeds"]:
        for seed in settings["run_seeds"]:
            for prior in PRIORS:
                for mode in ("unit", "cost"):
                    yield f"matrix{matrix}_seed{seed}_{prior}_{mode}.npz", matrix, seed, prior, mode


def validate_reference_manifest(settings: dict, reference: dict) -> None:
    """Require compatible, previously completed empirical-target baselines."""
    for key in ("batch_size", "budget_fraction", "per_cell_noise_variance", "recommendation_std_penalty", "priors"):
        if settings[key] != reference.get(key):
            raise ValueError(f"Reference baseline has a different protocol: {key}")
    if (not set(settings["matrix_seeds"]) <= set(reference.get("matrix_seeds", [])) or
            not set(settings["run_seeds"]) <= set(reference.get("run_seeds", []))):
        raise ValueError("Reference baseline does not cover the requested matrix/trial seeds")
    if (settings["n_arms"] != reference.get("n_models") or
            settings["n_examples"] != reference.get("n_examples")):
        raise ValueError("Reference baseline has different matrix dimensions")
    for seed, digest in settings["matrix_sha256"].items():
        if reference.get("matrix_sha256", {}).get(seed) != digest:
            raise ValueError(f"Reference baseline has different response data: matrix seed {seed}")
    if settings["ordered_per_cell_cost_sha256"] != reference.get("ordered_cost_vector_sha256"):
        raise ValueError("Reference baseline has a different ordered price vector")
    for new_key, old_key in (("unit_exhaustive_cost", "full_evals"),
                              ("heterogeneous_exhaustive_cost", "full_original_cost")):
        if not np.isclose(settings[new_key], reference.get(old_key, np.nan), rtol=0, atol=1e-6):
            raise ValueError(f"Reference baseline has a different exhaustive budget: {new_key}")
    if (reference.get("gittins_target") != "finite_population_mean" or
            reference.get("ts_target") != "finite_population_mean" or
            reference.get("gittins_model_version") != "batch_level_paper_v1" or
            reference.get("gittins_dp_transitions") != "batch" or
            reference.get("gittins_dp_horizon") != math.ceil(settings["n_examples"] / settings["batch_size"])):
        raise ValueError("Reference must contain empirical-target TS and paper batch-level Gittins")


def _validate_case_identity(archive, source_settings: dict, identity: tuple, path: Path) -> None:
    if "settings_hash" not in archive or str(archive["settings_hash"].item()) != source_settings["settings_hash"]:
        raise ValueError(f"Source case settings do not match its source manifest: {path}")
    for key, expected in zip(("matrix_seed", "run_seed", "prior_name", "cost_mode"), identity):
        if key not in archive or np.asarray(archive[key]).item() != expected:
            raise ValueError(f"Source case identity differs: {path}: {key}")


def _validate_reference_case(
    archive, settings: dict, identity: tuple, matrix: np.ndarray, costs: np.ndarray, path: Path,
    *, methods: tuple[str, ...] = ("ts", "gittins"),
) -> None:
    """Validate stored trajectories without evaluating either old policy again."""
    _, _, _, mode = identity
    full = float(settings["unit_exhaustive_cost" if mode == "unit" else "heterogeneous_exhaustive_cost"])
    budget = max(1, round(settings["budget_fraction"] * full)) if mode == "unit" else settings["budget_fraction"] * full
    truth = np.asarray(matrix, dtype=np.float64).mean(1)
    for method in methods:
        required = ("x", "x_original_cost", "pulled_arm", "n_new_samples", "recommended_arm",
                    "recommended_mean", "regret", "budget", "full_budget", "elapsed_s", "auc_normalized")
        for field in required:
            if f"{method}_{field}" not in archive:
                raise ValueError(f"Reference baseline is missing {method}_{field}: {path}")
        x = np.asarray(archive[f"{method}_x"])
        prices = np.asarray(archive[f"{method}_x_original_cost"], dtype=np.float64)
        arms = np.asarray(archive[f"{method}_pulled_arm"], dtype=np.int64)
        sizes = np.asarray(archive[f"{method}_n_new_samples"], dtype=np.int64)
        rec = np.asarray(archive[f"{method}_recommended_arm"], dtype=np.int64)
        if (x.ndim != 1 or x.size == 0 or any(value.shape != x.shape for value in (prices, arms, sizes, rec)) or
                (arms < 0).any() or (arms >= len(costs)).any() or (rec < 0).any() or (rec >= len(costs)).any() or
                (sizes < 1).any() or (sizes > settings["batch_size"]).any()):
            raise ValueError(f"Reference baseline has invalid action arrays: {path}: {method}")
        np.testing.assert_array_equal(x, sizes.cumsum(), err_msg=f"Reference item accounting differs: {path}")
        np.testing.assert_allclose(prices, np.cumsum(sizes * costs[arms]), rtol=0, atol=1e-8,
                                   err_msg=f"Reference price accounting differs: {path}")
        np.testing.assert_allclose(truth.max() - truth[rec], archive[f"{method}_regret"], rtol=0, atol=1e-12)
        for key, expected in (("budget", budget), ("full_budget", full)):
            np.testing.assert_allclose(float(archive[f"{method}_{key}"].item()), expected, rtol=0, atol=1e-6)
        axis = x if mode == "unit" else prices
        require_budget = method in ("ts", "gittins") or settings.get("max_steps") is None
        if require_budget and (axis[-1] < budget - 1e-8 or (len(axis) > 1 and axis[-2] >= budget + 1e-8)):
            raise ValueError(f"Reference baseline did not finish at its specified budget: {path}: {method}")


def validate_reference_study(settings: dict, directory: Path, matrices: dict[int, np.ndarray], costs: np.ndarray) -> dict:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Existing baseline reference is required; no TS or Gittins rerun fallback: {path}")
    reference = json.loads(path.read_text())
    validate_reference_manifest(settings, reference)
    roots_path = directory / "roots.npz"
    with np.load(roots_path, allow_pickle=False) as roots:
        if str(roots["settings_hash"].item()) != reference["settings_hash"]:
            raise ValueError("Reference Gittins root cache does not match its manifest")
        horizon = reference["gittins_dp_horizon"]
        for prior in PRIORS:
            for mode in ("unit", "cost"):
                values = roots[f"{prior}_{mode}_roots"]
                if values.shape != (settings["n_arms"], horizon + 1) or not np.isfinite(values).all() or np.any(values[:, -1] != 0):
                    raise ValueError("Reference Gittins root cache is not a completed batch-level table")
    hashes = {}
    for name, *identity in _case_names(settings):
        case = directory / "runs" / name
        if not case.is_file():
            raise FileNotFoundError(f"Reference baseline case is missing; baseline policies will not be rerun: {case}")
        with np.load(case, allow_pickle=False) as archive:
            _validate_case_identity(archive, reference, tuple(identity), case)
            _validate_reference_case(archive, settings, tuple(identity), matrices[identity[0]], costs, case)
        hashes[name] = _sha256(case)
    return dict(directory=str(directory.resolve()), settings_hash=reference["settings_hash"],
                manifest_sha256=_sha256(path), root_cache_sha256=_sha256(roots_path),
                case_sha256=hashes, policies_reused=["ts", "gittins"],
                baseline_simulation_allowed=False)


def validate_correlated_reuse(settings: dict, directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing correlated TS source manifest: {path}")
    source = json.loads(path.read_text())
    for key in ("model_version", "batch_size", "budget_fraction", "priors", "per_cell_noise_variance",
                "recommendation_std_penalty", "target", "kernel_mode", "fit_every", "fit_maxiter", "kernel",
                "kernel_marginal_variance", "max_steps", "ordered_per_cell_cost_sha256", "botorch_version", "torch_version"):
        if settings[key] != source.get(key):
            raise ValueError(f"Existing correlated TS has a different algorithm/protocol: {key}")
    for field in ("matrix_sha256", "ordered_feature_sha256"):
        for seed, digest in settings[field].items():
            if source.get(field, {}).get(seed) != digest:
                raise ValueError(f"Existing correlated TS has different data: {field}, seed {seed}")
    if source.get("source_sha256", {}).get("src/correlated_ts_policy.py") != settings["source_sha256"]["src/correlated_ts_policy.py"]:
        raise ValueError("Existing correlated TS used a different policy core")
    simulator_hash = source.get("correlated_simulator_sha256")
    if simulator_hash is None:
        snapshot = directory / "source_snapshot/scripts/compare_correlated_ts.py"
        if not snapshot.is_file() or _sha256(snapshot) != source.get("source_sha256", {}).get("scripts/compare_correlated_ts.py"):
            raise ValueError("Cannot verify the existing correlated TS simulator; its matching source snapshot is required")
        simulator_hash = _simulator_sha256(snapshot)
    if simulator_hash != settings["correlated_simulator_sha256"]:
        raise ValueError("Existing correlated TS used a different simulator")
    return dict(directory=str(directory.resolve()), settings_hash=source["settings_hash"],
                manifest_sha256=_sha256(path), correlated_simulator_sha256=simulator_hash,
                policy_core_sha256=settings["source_sha256"]["src/correlated_ts_policy.py"])


def prepare(args: argparse.Namespace) -> dict:
    """Fingerprint protocol, alignment and numerical source before resuming."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_hashes, feature_hashes, shapes, matrices = {}, {}, set(), {}
    costs = load_paper_costs()
    for seed in args.matrix_seeds:
        matrix = ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{seed}.npy"
        bo = ROOT / f"data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed{seed}_bo_inputs.npz"
        data = load_data(matrix, bo)
        matrices[seed] = data.matrix
        shapes.add(data.matrix.shape)
        np.testing.assert_allclose(data.per_cell_cost, costs, rtol=0, atol=1e-12)
        matrix_hashes[str(seed)] = _sha256(matrix)
        feature_hashes[str(seed)] = hashlib.sha256(data.features.numpy().astype("<f8").tobytes()).hexdigest()
    if len(shapes) != 1:
        raise ValueError("All paired response matrices must have the same shape")
    n_arms, n_examples = next(iter(shapes))
    sources = ["src/correlated_ts_policy.py", "scripts/compare_correlated_ts.py", "scripts/run_mfkg_baseline.py"]
    import botorch
    settings = dict(
        model_version=MODEL_VERSION, methods=list(METHODS), matrix_seeds=args.matrix_seeds,
        run_seeds=args.run_seeds, n_arms=n_arms, n_examples=n_examples,
        batch_size=args.batch_size, budget_fraction=args.eval_budget_fraction,
        priors={k: list(v) for k, v in PRIORS.items()}, per_cell_noise_variance=0.25,
        recommendation_std_penalty=1.0, target="complete_fixed_benchmark_empirical_mean",
        kernel_mode=args.kernel_mode, fit_every=args.fit_every, fit_maxiter=args.fit_maxiter,
        kernel="MixedSingleTaskGP_all_categorical_configuration_covariance",
        kernel_marginal_variance="normalized_to_same_prior_variance_as_independent_TS",
        fit_data="observed_arm_sample_means_only_with_noise_0.25_divided_by_observed_count",
        prior_mean_and_variance="fixed_per_panel_not_fit", warmup_batches=0,
        cost_modes=["unit", "cost"], allocation_cost_usage="none_for_both_methods",
        budget_convention="TS_unit_final_batch_cap; cost_post_batch_stop; metrics_exclude_over_budget_feedback",
        unit_exhaustive_cost=n_arms * n_examples,
        heterogeneous_exhaustive_cost=float(n_examples * costs.sum()),
        max_steps=args.max_steps, matrix_sha256=matrix_hashes,
        ordered_feature_sha256=feature_hashes,
        ordered_per_cell_cost_sha256=hashlib.sha256(costs.astype("<f8").tobytes()).hexdigest(),
        source_sha256={name: _sha256(ROOT / name) for name in sources},
        correlated_simulator_sha256=_simulator_sha256(Path(__file__).resolve()),
        botorch_version=botorch.__version__, torch_version=torch.__version__,
    )
    reference_directory = Path(getattr(args, "reference_results_dir", DEFAULT_REFERENCE_DIR)).resolve()
    settings["baseline_reference"] = validate_reference_study(settings, reference_directory, matrices, costs)
    reuse_directory = getattr(args, "reuse_correlated_from", None)
    existing_path = args.out_dir / "manifest.json"
    existing_source = (json.loads(existing_path.read_text()).get("correlated_reuse_source")
                       if existing_path.is_file() else None)
    if reuse_directory is not None and (Path(reuse_directory) / "manifest.json").is_file():
        settings["correlated_reuse_source"] = validate_correlated_reuse(settings, Path(reuse_directory).resolve())
    elif existing_source is not None and (reuse_directory is None or
            str(Path(reuse_directory).resolve()) == existing_source["directory"]):
        # The import origin may have been removed after successful import.
        # Completed output cases and the persistent TS reference are sufficient
        # for resuming; absent correlated cases can still be computed alone.
        settings["correlated_reuse_source"] = existing_source
    elif reuse_directory is not None:
        raise FileNotFoundError(f"Missing correlated TS import source: {reuse_directory}")
    else:
        settings["correlated_reuse_source"] = None
    # Import origins are audit metadata, not part of the algorithm/cache
    # identity. They may be removed once their correlated arrays are imported.
    identity_settings = {key: value for key, value in settings.items() if key != "correlated_reuse_source"}
    settings["settings_hash"] = hashlib.sha256(json.dumps(identity_settings, sort_keys=True).encode()).hexdigest()
    path = args.out_dir / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()).get("settings_hash") != settings["settings_hash"]:
            raise ValueError("Output directory has different protocol/data/source; choose a new --out-dir")
    else:
        temp = path.with_name(f"manifest.{args.worker_index}.tmp")
        temp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    return settings


def run_worker(args: argparse.Namespace, settings: dict) -> None:
    costs = load_paper_costs()
    pairs = [(matrix, run) for matrix in args.matrix_seeds for run in args.run_seeds]
    cache, completed, computed_correlated, reused_correlated = {}, 0, 0, 0
    started = time.perf_counter()
    for index, (matrix_seed, seed) in enumerate(pairs):
        if index % args.workers != args.worker_index:
            continue
        if matrix_seed not in cache:
            cache[matrix_seed] = load_data(
                ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{matrix_seed}.npy",
                ROOT / f"data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed{matrix_seed}_bo_inputs.npz",
            )
        data = cache[matrix_seed]
        for prior, (mean, variance) in PRIORS.items():
            for mode in ("unit", "cost"):
                path = args.out_dir / "runs" / f"matrix{matrix_seed}_seed{seed}_{prior}_{mode}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as archive:
                        if str(archive["settings_hash"].item()) != settings["settings_hash"]:
                            raise ValueError(f"Incompatible existing case: {path}")
                else:
                    case = dict(matrix_seed=matrix_seed, run_seed=seed, prior_name=prior,
                                cost_mode=mode, settings_hash=settings["settings_hash"])
                    reference = settings["baseline_reference"]
                    reference_path = Path(reference["directory"]) / "runs" / path.name
                    if _sha256(reference_path) != reference["case_sha256"][path.name]:
                        raise ValueError(f"Reference baseline changed after validation: {reference_path}")
                    case.update(baseline_reference_case=str(reference_path),
                                baseline_reference_case_sha256=reference["case_sha256"][path.name],
                                baseline_reference_settings_hash=reference["settings_hash"],
                                baseline_policies_recomputed=False)
                    reuse = settings["correlated_reuse_source"]
                    reuse_path = Path(reuse["directory"]) / "runs" / path.name if reuse is not None else None
                    if reuse_path is not None and reuse_path.is_file():
                        source_settings = json.loads((Path(reuse["directory"]) / "manifest.json").read_text())
                        if _sha256(Path(reuse["directory"]) / "manifest.json") != reuse["manifest_sha256"]:
                            raise ValueError("Existing correlated TS source manifest changed after validation")
                        with np.load(reuse_path, allow_pickle=False) as archive:
                            _validate_case_identity(archive, source_settings, (matrix_seed, seed, prior, mode), reuse_path)
                            _validate_reference_case(archive, settings, (matrix_seed, seed, prior, mode),
                                                     data.matrix, costs, reuse_path, methods=("correlated_ts",))
                            copied = {key: archive[key].copy() for key in archive.files if key.startswith("correlated_ts_")}
                            if not copied or "correlated_ts_x" not in copied:
                                raise ValueError(f"Existing case lacks correlated TS history: {reuse_path}")
                            case.update(copied)
                        case.update(correlated_ts_reused=True, correlated_ts_source_case=str(reuse_path),
                                    correlated_ts_source_case_sha256=_sha256(reuse_path),
                                    correlated_ts_source_settings_hash=reuse["settings_hash"])
                        reused_correlated += 1
                    else:
                        common = dict(seed=seed, prior_mean=mean, prior_variance=variance,
                                  batch_size=args.batch_size, per_arm_cost=costs,
                                  cost_mode=mode, budget_fraction=args.eval_budget_fraction,
                                  max_steps=args.max_steps, record_cells=True)
                        trace = simulate_correlated_ts(
                            data.matrix, features=data.features, kernel_mode=args.kernel_mode,
                            fit_every=args.fit_every, fit_maxiter=args.fit_maxiter, **common)
                        case.update({f"correlated_ts_{key}": value for key, value in trace.items()})
                        case["correlated_ts_auc_normalized"] = exact_normalized_auc(trace, args.eval_budget_fraction)
                        case["correlated_ts_reused"] = False
                        computed_correlated += 1
                    _atomic_npz(path, **case)
                completed += 1
                progress = dict(worker_index=args.worker_index, cases_complete=completed,
                                matrix_seed=matrix_seed, run_seed=seed, prior_name=prior,
                                cost_mode=mode, computed_correlated_cases=computed_correlated,
                                reused_correlated_cases=reused_correlated, baseline_policies_recomputed=False,
                                elapsed_s=time.perf_counter() - started)
                (args.out_dir / f"worker_{args.worker_index}_progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    print(json.dumps({"worker_index": args.worker_index, "complete": completed,
                      "computed_correlated_cases": computed_correlated, "reused_correlated_cases": reused_correlated,
                      "baseline_policies_recomputed": False,
                      "elapsed_s": time.perf_counter() - started}), flush=True)
    existing, imported = 0, 0
    for name, *_ in _case_names(settings):
        path = args.out_dir / "runs" / name
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["settings_hash"].item()) != settings["settings_hash"]:
                raise ValueError(f"Output case settings differ: {path}")
            existing += 1
            imported += int(archive["correlated_ts_reused"].item())
    total = len(settings["matrix_seeds"]) * len(settings["run_seeds"]) * len(PRIORS) * 2
    summary = dict(expected_cases=total, completed_cases=existing, all_cases_complete=existing == total,
                   reused_TS_cases=existing, reused_Gittins_cases=existing,
                   imported_correlated_TS_cases=imported,
                   newly_computed_correlated_TS_cases=existing - imported,
                   TS_simulations=0, Gittins_simulations=0,
                   baseline_reference=settings["baseline_reference"]["directory"],
                   correlated_reuse_source=settings["correlated_reuse_source"],
                   settings_hash=settings["settings_hash"])
    target = args.out_dir / "reuse_summary.json"
    temporary = target.with_name(f"reuse_summary.{args.worker_index}.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/gsm8k_correlated_ts_reuse_lcb100")
    parser.add_argument("--reference-results-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
                        help="Required existing TS/Gittins trajectories; baseline policies are never rerun.")
    parser.add_argument("--reuse-correlated-from", type=Path,
                        help="Import matching completed correlated-TS cases; compute only missing correlated-TS cases.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix-seeds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-budget-fraction", type=float, default=0.1)
    parser.add_argument("--kernel-mode", choices=("fixed", "fit"), default="fit")
    parser.add_argument("--fit-every", type=int, default=32)
    parser.add_argument("--fit-maxiter", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=None, help="Explicit smoke cap; omitted for complete trials.")
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("worker-index must lie in [0,workers)")
    if args.batch_size < 1 or not 0 < args.eval_budget_fraction <= 1:
        parser.error("Invalid batch size or budget fraction")
    if args.fit_every < 1 or args.fit_maxiter < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Fit controls and any max-steps cap must be positive")
    if (not args.matrix_seeds or not args.run_seeds or
            len(set(args.matrix_seeds)) != len(args.matrix_seeds) or
            len(set(args.run_seeds)) != len(args.run_seeds) or
            not set(args.matrix_seeds) <= set(range(1, 6)) or min(args.run_seeds) < 0):
        parser.error("Use distinct matrix seeds1..5 and nonnegative run seeds")
    torch.set_num_threads(1)
    settings = prepare(args)
    if not args.prepare_only:
        run_worker(args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
