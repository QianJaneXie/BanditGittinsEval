"""BO costs must be converted and aligned without opening score arrays."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation_costs import load_bo_per_cell_costs, validate_per_cell_costs  # noqa: E402


def make_archive(path, *, costs=None, ids=None, metadata=None):
    np.savez(
        path,
        cost=np.array([30.0, 10.0, 20.0]) if costs is None else np.asarray(costs),
        arm_ids=np.array([2, 0, 1]) if ids is None else np.asarray(ids),
        metadata_json=json.dumps({"n_examples": 10} if metadata is None else metadata),
        # Loading either score array would fail with allow_pickle=False.
        Y=np.array([object()], dtype=object),
        raw_scores_mean=np.array([object()], dtype=object),
    )


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"cost_test_{name}", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["simulate_simple_regret", "run_simple_regret_wandb"])
def bandit_runner(request):
    return load_script(request.param)


def test_archive_costs_are_divided_by_benchmark_size_and_reordered(tmp_path):
    path = tmp_path / "bo.npz"
    make_archive(path)
    np.testing.assert_array_equal(load_bo_per_cell_costs(path, 3), [1.0, 2.0, 3.0])


def test_both_bandit_entrypoints_accept_bo_cost_archives(bandit_runner, tmp_path):
    path = tmp_path / "bo.npz"
    make_archive(path)
    torch.testing.assert_close(
        bandit_runner.load_cost_vector(path, 3),
        torch.tensor([1, 2, 3], dtype=torch.float64),
    )


@pytest.mark.parametrize("n_examples", [None, 0, -1, float("nan"), float("inf"), True, 2.5, "10"])
def test_rejects_unknown_or_invalid_benchmark_size(tmp_path, n_examples):
    path = tmp_path / "bo.npz"
    metadata = {} if n_examples is None else {"n_examples": n_examples}
    make_archive(path, metadata=metadata)
    with pytest.raises(ValueError, match="n_examples.*finite positive integer"):
        load_bo_per_cell_costs(path, 3)


@pytest.mark.parametrize("ids", [[0, 0, 2], [0, 1, 3], [0.0, 1.0, 2.0], [[0], [1], [2]]])
def test_rejects_non_permutation_arm_ids(tmp_path, ids):
    path = tmp_path / "bo.npz"
    make_archive(path, ids=ids)
    with pytest.raises(ValueError, match="integer permutation"):
        load_bo_per_cell_costs(path, 3)


@pytest.mark.parametrize("costs", [[0, 1, 2], [-1, 1, 2], [float("nan"), 1, 2], [float("inf"), 1, 2]])
def test_rejects_nonpositive_or_nonfinite_archive_costs(tmp_path, costs):
    path = tmp_path / "bo.npz"
    make_archive(path, costs=costs)
    with pytest.raises(ValueError, match="finite and positive"):
        load_bo_per_cell_costs(path, 3)


def test_rejects_wrong_cost_dimensions_and_missing_metadata(tmp_path):
    path = tmp_path / "bo.npz"
    make_archive(path, costs=[[10], [20], [30]])
    with pytest.raises(ValueError, match="shape"):
        load_bo_per_cell_costs(path, 3)
    np.savez(path, cost=[10, 20, 30], arm_ids=[0, 1, 2])
    with pytest.raises(ValueError, match="missing required.*metadata_json"):
        load_bo_per_cell_costs(path, 3)


@pytest.mark.parametrize("kind", ["npy", "list", "cost_per_arm", "pricing"])
def test_existing_cost_formats_preserve_per_cell_units(bandit_runner, tmp_path, kind):
    prices = [0.25, 2.5, 10.0]
    if kind == "npy":
        path = tmp_path / "prices.npy"
        np.save(path, prices)
    else:
        path = tmp_path / "prices.json"
        value = prices
        if kind == "cost_per_arm":
            value = {"cost_per_arm": prices}
        elif kind == "pricing":
            value = {str(i): {"estimated_cost_per_1m_input_tokens": c} for i, c in enumerate(prices)}
        path.write_text(json.dumps(value), encoding="utf-8")
    np.testing.assert_array_equal(bandit_runner.load_cost_vector(path, 3).numpy(), prices)


def test_wandb_legacy_cost_dict_format_is_preserved(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"0": {"cost": 1.0}, "1": {"cost": 2.0}}))
    np.testing.assert_array_equal(load_script("run_simple_regret_wandb").load_cost_vector(path, 2), [1, 2])


def test_all_entrypoints_reject_invalid_legacy_costs(bandit_runner, tmp_path):
    path = tmp_path / "prices.npy"
    np.save(path, [1.0, 0.0, 3.0])
    with pytest.raises(ValueError, match="finite and positive"):
        bandit_runner.load_cost_vector(path, 3)


def test_validation_does_not_flatten_archive_costs():
    with pytest.raises(ValueError, match="shape"):
        validate_per_cell_costs(np.ones((3, 1)), 3)
