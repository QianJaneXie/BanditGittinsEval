"""End-to-end item accounting and experiment identity for correlated TS."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_correlated_ts import prepare, simulate_correlated_ts


@pytest.fixture
def matrix():
    return np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)


def run(matrix, **overrides):
    kwargs = dict(
        features=torch.tensor([[0., 0.], [0., 1.], [1., 0.]], dtype=torch.float64),
        seed=7, prior_mean=.2, prior_variance=.01, batch_size=4,
        per_arm_cost=np.array([.25, .45, 3.2]), cost_mode="unit",
        budget_fraction=1., kernel_mode="fixed", record_cells=True,
    )
    kwargs.update(overrides)
    return simulate_correlated_ts(matrix, **kwargs)


@pytest.mark.parametrize("kernel_mode", ["fixed", "fit"])
def test_complete_matrix_reveals_each_item_once_and_handles_partial_last_batch(matrix, kernel_mode):
    trace = run(matrix, kernel_mode=kernel_mode, fit_every=2, fit_maxiter=4)
    assert trace["x"][-1] == matrix.size
    assert 3 in trace["n_new_samples"]  # N=11, B=4 gives a final batch of three.
    seen = np.zeros(matrix.shape, dtype=bool)
    expected_cost = 0.
    prices = np.array([.25, .45, 3.2])
    for step, (arm, size) in enumerate(zip(trace["pulled_arm"], trace["n_new_samples"])):
        padded = trace["sampled_columns"][step]
        columns = padded[:size]
        assert np.all(padded[size:] == -1)
        assert len(np.unique(columns)) == size
        assert not seen[arm, columns].any()
        seen[arm, columns] = True
        expected_cost += size * prices[arm]
        assert trace["x_original_cost"][step] == pytest.approx(expected_cost)
        true_regret = matrix.mean(axis=1).max() - matrix.mean(axis=1)[trace["recommended_arm"][step]]
        assert trace["regret"][step] == pytest.approx(float(true_regret), abs=1e-7)
    assert seen.all()
    np.testing.assert_array_equal(trace["counts"], [11, 11, 11])
    np.testing.assert_allclose(trace["observed_sums"], matrix.sum(axis=1))
    assert trace["recommended_arm"][-1] == 1
    assert trace["regret"][-1] == 0.


@pytest.mark.parametrize("kernel_mode", ["fixed", "fit"])
def test_prices_do_not_change_allocations_before_either_budget_ends(matrix, kernel_mode):
    common = dict(cost_mode="cost", max_steps=7, kernel_mode=kernel_mode,
                  fit_every=2, fit_maxiter=4)
    first = run(matrix, per_arm_cost=np.ones(3), **common)
    second = run(matrix, per_arm_cost=np.array([.25, .45, 3.2]), **common)
    for key in ("pulled_arm", "sampled_columns", "recommended_arm", "recommended_mean", "x"):
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["x_original_cost"], second["x_original_cost"])


def test_kernel_mode_is_part_of_the_resumable_experiment_identity(tmp_path):
    args = SimpleNamespace(
        out_dir=tmp_path, matrix_seeds=[1], run_seeds=[0], batch_size=8,
        eval_budget_fraction=.1, kernel_mode="fixed", fit_every=32,
        fit_maxiter=25, max_steps=2, worker_index=0,
    )
    first = prepare(args)
    assert prepare(args) == first
    args.kernel_mode = "fit"
    with pytest.raises(ValueError, match="different protocol"):
        prepare(args)


def test_generic_cli_does_not_train_on_archive_answers(tmp_path, matrix, monkeypatch):
    matrix_path, archive_path = tmp_path / "scores.npy", tmp_path / "inputs.npz"
    np.save(matrix_path, matrix)
    np.savez(archive_path, X=np.array([[0, 0], [0, 1], [1, 0]]),
             arm_ids=np.arange(3), cost=np.ones(3) * matrix.shape[1],
             Y=np.full((3, 1), np.nan), raw_scores_mean=np.full(3, np.nan))
    output = tmp_path / "trace.npz"
    spec = importlib.util.spec_from_file_location("cts_cli_under_test", ROOT / "scripts/run_correlated_ts_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["run_correlated_ts_baseline.py", "--matrix", str(matrix_path),
        "--bo-inputs", str(archive_path), "--out", str(output), "--batch-size", "4",
        "--eval-budget-fraction", "1", "--kernel-mode", "fit", "--fit-every", "2", "--fit-maxiter", "4"])
    assert module.main() == 0
    with np.load(output, allow_pickle=False) as trace:
        assert trace["x"][-1] == matrix.size
        assert trace["regret"][-1] == 0
        assert trace["fit_count"].item() > 0
    assert output.with_suffix(".json").is_file()
