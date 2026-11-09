"""Existing TS/Gittins trajectories are references, never new simulations."""

import copy
import hashlib
import json
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_correlated_ts as comparison
from plot_correlated_ts_comparison import ts_trajectory_archive


@pytest.fixture
def studies(tmp_path):
    matrix = np.array([[0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]], dtype=float)
    costs = np.array([.25, .45, 3.2])
    price_hash = hashlib.sha256(costs.astype("<f8").tobytes()).hexdigest()
    reference_dir, correlated_dir = tmp_path / "reference", tmp_path / "correlated"
    for directory in (reference_dir, correlated_dir):
        (directory / "runs").mkdir(parents=True)
    settings = dict(
        model_version=comparison.MODEL_VERSION, matrix_seeds=[1], run_seeds=[0],
        n_arms=3, n_examples=4, batch_size=2, budget_fraction=1.,
        priors={k: list(v) for k, v in comparison.PRIORS.items()},
        per_cell_noise_variance=.25, recommendation_std_penalty=1.,
        unit_exhaustive_cost=12, heterogeneous_exhaustive_cost=float(4 * costs.sum()),
        matrix_sha256={"1": "matrix"}, ordered_feature_sha256={"1": "features"},
        ordered_per_cell_cost_sha256=price_hash, max_steps=None,
        target="complete_fixed_benchmark_empirical_mean", kernel_mode="fixed",
        fit_every=32, fit_maxiter=25,
        kernel="MixedSingleTaskGP_all_categorical_configuration_covariance",
        kernel_marginal_variance="normalized_to_same_prior_variance_as_independent_TS",
        source_sha256={"src/correlated_ts_policy.py": comparison._sha256(ROOT / "src/correlated_ts_policy.py")},
        correlated_simulator_sha256=comparison._simulator_sha256(ROOT / "scripts/compare_correlated_ts.py"),
        botorch_version="test", torch_version="test", settings_hash="new-study",
    )
    reference = dict(
        n_models=3, n_examples=4, batch_size=2, budget_fraction=1.,
        matrix_seeds=[1], run_seeds=[0], priors=settings["priors"],
        per_cell_noise_variance=.25, recommendation_std_penalty=1.,
        matrix_sha256=settings["matrix_sha256"], ordered_cost_vector_sha256=price_hash,
        full_evals=12, full_original_cost=settings["heterogeneous_exhaustive_cost"],
        gittins_target="finite_population_mean", ts_target="finite_population_mean",
        gittins_model_version="batch_level_paper_v1", gittins_dp_transitions="batch",
        gittins_dp_horizon=2, settings_hash="reference-study",
    )
    source = {**copy.deepcopy(settings), "settings_hash": "old-correlated"}
    (reference_dir / "manifest.json").write_text(json.dumps(reference))
    (correlated_dir / "manifest.json").write_text(json.dumps(source))
    np.savez(reference_dir / "roots.npz", settings_hash=reference["settings_hash"],
             **{f"{prior}_{mode}_roots": np.zeros((3, 3))
                for prior in comparison.PRIORS for mode in ("unit", "cost")})
    sizes, arms = np.full(6, 2), np.array([0, 1, 2, 0, 1, 2])
    for prior in comparison.PRIORS:
        for mode in ("unit", "cost"):
            identity = dict(matrix_seed=1, run_seed=0, prior_name=prior, cost_mode=mode)
            full = settings["unit_exhaustive_cost" if mode == "unit" else "heterogeneous_exhaustive_cost"]
            trace = dict(x=sizes.cumsum(), x_original_cost=np.cumsum(sizes * costs[arms]),
                         pulled_arm=arms, n_new_samples=sizes, recommended_arm=np.ones(6, dtype=int),
                         recommended_mean=np.full(6, .75), regret=np.zeros(6),
                         budget=full, full_budget=full, elapsed_s=.01, auc_normalized=0.)
            name = f"matrix1_seed0_{prior}_{mode}.npz"
            np.savez(reference_dir / "runs" / name, **identity, settings_hash=reference["settings_hash"],
                     **{f"{method}_{k}": v for method in ("ts", "gittins") for k, v in trace.items()})
            np.savez(correlated_dir / "runs" / name, **identity, settings_hash=source["settings_hash"],
                     **{f"correlated_ts_{k}": v for k, v in trace.items()})
    settings["baseline_reference"] = comparison.validate_reference_study(settings, reference_dir, {1: matrix}, costs)
    settings["correlated_reuse_source"] = comparison.validate_correlated_reuse(settings, correlated_dir)
    args = SimpleNamespace(out_dir=tmp_path / "result", matrix_seeds=[1], run_seeds=[0],
                           worker_index=0, workers=1, batch_size=2, eval_budget_fraction=1.,
                           kernel_mode="fixed", fit_every=32, fit_maxiter=25, max_steps=None)
    args.out_dir.mkdir()
    return args, settings, matrix, costs, reference_dir, correlated_dir


def test_importing_completed_cts_calls_no_simulator_and_persists_no_baseline_arrays(studies, monkeypatch):
    args, settings, matrix, costs, reference_dir, _ = studies
    import compare_gittins_ts

    def forbidden(*args, **kwargs):
        raise AssertionError("No policy may be simulated on a complete reuse path")

    monkeypatch.setattr(comparison, "simulate_correlated_ts", forbidden)
    monkeypatch.setattr(compare_gittins_ts, "simulate_policy", forbidden)
    monkeypatch.setattr(comparison, "load_paper_costs", lambda: costs)
    monkeypatch.setattr(comparison, "load_data", lambda *args, **kwargs:
                        SimpleNamespace(matrix=matrix, features=torch.arange(3, dtype=torch.float64)[:, None]))
    comparison.run_worker(args, settings)
    for path in (args.out_dir / "runs").glob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            assert not any(key.startswith(("ts_", "gittins_")) for key in archive.files)
            assert archive["correlated_ts_reused"].item()
            assert Path(archive["baseline_reference_case"].item()).parent == reference_dir / "runs"
            with ts_trajectory_archive(archive) as source:
                np.testing.assert_array_equal(source["ts_pulled_arm"], [0, 1, 2, 0, 1, 2])
    summary = json.loads((args.out_dir / "reuse_summary.json").read_text())
    assert summary["imported_correlated_TS_cases"] == 4
    assert summary["newly_computed_correlated_TS_cases"] == 0
    assert summary["reused_TS_cases"] == summary["reused_Gittins_cases"] == 4
    assert summary["TS_simulations"] == summary["Gittins_simulations"] == 0
    comparison.run_worker(args, settings)  # Resume also cannot invoke a model.


def test_missing_reference_fails_without_falling_back_to_policy_simulation(studies):
    _, settings, matrix, costs, reference_dir, _ = studies
    next((reference_dir / "runs").glob("*.npz")).unlink()
    with pytest.raises(FileNotFoundError, match="will not be rerun"):
        comparison.validate_reference_study(settings, reference_dir, {1: matrix}, costs)


@pytest.mark.parametrize("field,value", [("kernel_mode", "fit"), ("fit_every", 16),
                                         ("ordered_per_cell_cost_sha256", "new-prices")])
def test_mismatched_correlated_source_cannot_be_imported(studies, field, value):
    _, settings, _, _, _, correlated_dir = studies
    settings = copy.deepcopy(settings)
    settings[field] = value
    with pytest.raises(ValueError, match="different algorithm/protocol"):
        comparison.validate_correlated_reuse(settings, correlated_dir)


def test_original_ts_reference_is_readonly_and_detects_changed_source(studies, tmp_path):
    _, _, _, _, reference_dir, _ = studies
    path = next((reference_dir / "runs").glob("*.npz"))
    raw = dict(baseline_reference_case=np.array(str(path)),
               baseline_reference_case_sha256=np.array("wrong-digest"),
               baseline_reference_settings_hash=np.array("reference-study"))
    np.savez(tmp_path / "reference.npz", **raw)
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="missing or changed"):
            with ts_trajectory_archive(archive):
                pass


def test_completed_reuse_cache_resumes_after_correlated_import_origin_is_removed(studies, monkeypatch):
    args, _, matrix, costs, reference_dir, correlated_dir = studies
    args.reference_results_dir = reference_dir
    args.reuse_correlated_from = None
    monkeypatch.setattr(comparison, "load_paper_costs", lambda: costs)
    monkeypatch.setattr(comparison, "load_data", lambda *a, **k:
                        SimpleNamespace(matrix=matrix, per_cell_cost=costs,
                                        features=torch.arange(3, dtype=torch.float64)[:, None]))
    original_sha = comparison._sha256
    monkeypatch.setattr(comparison, "_sha256", lambda path:
                        "matrix" if Path(path).name.startswith("gsm8k_1_samples_various_models_seed")
                        and Path(path).suffix == ".npy" else original_sha(path))
    first = comparison.prepare(args)
    source = {**first, "settings_hash": "old-correlated"}
    (correlated_dir / "manifest.json").write_text(json.dumps(source))
    (args.out_dir / "manifest.json").unlink()
    args.reuse_correlated_from = correlated_dir
    imported = comparison.prepare(args)
    assert imported["settings_hash"] == first["settings_hash"]

    def forbidden(*a, **k):
        raise AssertionError("Completed cache must not invoke a simulator")

    monkeypatch.setattr(comparison, "simulate_correlated_ts", forbidden)
    comparison.run_worker(args, imported)
    shutil.rmtree(correlated_dir)
    args.reuse_correlated_from = None
    resumed = comparison.prepare(args)
    assert resumed["settings_hash"] == imported["settings_hash"]
    comparison.run_worker(args, resumed)
