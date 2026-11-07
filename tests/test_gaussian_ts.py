"""Behavioral checks for Gaussian TS and its two experiment entry points."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gaussian_ts_policy import gaussian_thompson_sampling
from simple_regret_recommend import finite_population_posterior_moments, posterior_incumbent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_full_population_posterior_and_completed_recommendation():
    obs = torch.tensor([[1.0, float("nan")], [0.0, 0.0]])
    # Prior N(.5, .25), cell noise .25: after observing 1, latent
    # posterior is N(.75, .125). The full 2-cell mean is N(.875, .09375).
    means, variances = finite_population_posterior_moments(
        obs, prior_mean=0.5, prior_variance=0.25, tau_sq_cell=0.25,
    )
    torch.testing.assert_close(means, torch.tensor([0.875, 0.0]))
    torch.testing.assert_close(variances, torch.tensor([0.09375, 0.0], dtype=torch.float64))
    completed = torch.tensor([[1.0, 1.0], [0.0, float("nan")]])
    arm, _ = posterior_incumbent(completed, prior_mean=.5, prior_variance=.25, tau_sq_cell=.25)
    assert arm == 0
    batch = gaussian_thompson_sampling(completed, batch_size=8)
    assert batch.tolist() == [[1], [1]]


def test_unseen_cells_only_batch_cap_and_seed_reproducibility():
    obs = torch.tensor([[1., float("nan"), 0., float("nan")], [float("nan")] * 4])
    first = gaussian_thompson_sampling(obs, batch_size=3, max_batch_size=2,
                                      generator=torch.Generator().manual_seed(42))
    second = gaussian_thompson_sampling(obs, batch_size=3, max_batch_size=2,
                                       generator=torch.Generator().manual_seed(42))
    assert torch.equal(first, second)
    assert first.shape == (2, 2)
    assert len(set(first[1].tolist())) == 2
    assert bool(obs[first[0], first[1]].isnan().all())
    assert gaussian_thompson_sampling(torch.zeros((2, 3))) is None
    assert gaussian_thompson_sampling(obs, max_batch_size=0) is None


def test_posterior_sample_can_explore_lower_mean():
    obs = torch.tensor([[.9, float("nan")], [.1, float("nan")]])
    with patch("gaussian_ts_policy.torch.randn", return_value=torch.tensor([-10., 10.], dtype=torch.float64)):
        batch = gaussian_thompson_sampling(obs, batch_size=1)
    assert int(batch[0, 0]) == 1


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"prior_variance": 0}, {"obs_noise_variance": -1},
    {"prior_mean": float("nan")}, {"batch_size": 1.5},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        gaussian_thompson_sampling(torch.full((2, 3), float("nan")), **kwargs)


def test_simulation_cli_saves_reproducible_traces(tmp_path):
    matrix = tmp_path / "matrix.npy"
    np.save(matrix, np.array([[1., 1., 0., 1.], [0., 0., 1., 0.]], dtype=np.float32))
    paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
    for out in paths:
        subprocess.run([
            sys.executable, str(ROOT / "scripts/simulate_simple_regret.py"),
            "--matrix", str(matrix), "--out", str(out), "--algorithms", "ts",
            "--eval-budget-fraction", "0.625", "--ts-batch-size", "3", "--seed", "2",
            "--ts-prior-mean", ".2", "--ts-prior-variance", ".01",
        ], check=True, capture_output=True, text=True)
    with np.load(paths[0]) as first, np.load(paths[1]) as second:
        for key in ("ts_x", "ts_regret", "ts_recommended_arm", "ts_recommended_mean"):
            np.testing.assert_equal(first[key], second[key])
        assert first["ts_x"][-1] == 5
        np.testing.assert_equal(first["ts_x"], first["ts_x_original_cost"])
        assert str(first["ts_target"]) == "finite_population_mean"
        assert float(first["ts_obs_noise_variance"]) == 1 / 12


def test_wandb_variants_and_cost_budget_without_network():
    runner = load_script("run_simple_regret_wandb")
    for raw, cost, prior in [
        ("ts_B16", "unit", "default"), ("ts_cost_B16", "cost", "default"),
        ("ts_unit_B16_dataset", "unit", "dataset"),
        ("ts_aware_B16_dataset", "cost", "dataset"),
    ]:
        variant = runner.parse_experiment_variant(raw)
        assert (variant.policy_family, variant.cost_mode, variant.prior_type) == ("ts", cost, prior)
        assert variant.batch_size == 16
    args = SimpleNamespace(run_seed=7, gittins_obs_noise_variance=None, ts_obs_noise_variance=None)
    result = runner.run_simple_regret_experiment(
        ground_truth=torch.tensor([[1., 1., 0.], [0., 0., 1.]]),
        original_cost_per_arm=torch.tensor([2., 3.], dtype=torch.float64),
        variant=runner.parse_experiment_variant("ts_cost_B1"), args=args,
        prior_mean=.5, prior_variance=.04, max_evaluations=6, max_original_cost=5.,
        run=None, log_step_metrics=False,
    )
    increments = np.diff([0.] + result["x_original_cost"])
    np.testing.assert_equal(increments, [2. if arm == 0 else 3. for arm in result["pulled_arm"]])
    assert result["x_original_cost"][-1] >= 5.
    assert result["x_original_cost"][-2] < 5. if len(increments) > 1 else True
    assert result["lookup_table_s"] is None
