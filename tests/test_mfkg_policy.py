"""Checks for the finite-set MF-KG objective and its BoTorch integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from botorch.models import MixedSingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfkg_policy import (  # noqa: E402
    DiscretePosteriorMaximum,
    MarginalCostWeightedUtility,
    fit_mfkg_model,
    score_mfkg_candidates,
    score_mfkg_candidates_native,
    target_posterior_means,
)


@pytest.fixture
def setup():
    # Widely separated categorical codes ensure the acquisition only evaluates
    # the finite legal arm set, without interpolating numerical identifiers.
    train_X = torch.tensor([[0, 0.2], [100, 0.2], [0, 0.5]], dtype=torch.float64)
    train_Y = torch.tensor([[0.3], [0.4], [0.5]], dtype=torch.float64)
    model = MixedSingleTaskGP(
        train_X,
        train_Y,
        cat_dims=[0],
        train_Yvar=torch.full_like(train_Y, 1e-6),
        outcome_transform=Standardize(m=1),
    )
    model.eval()
    candidates = torch.tensor(
        [[0, 0.75], [100, 0.5], [500, 0.2], [500, 1.0]], dtype=torch.float64
    )
    targets = torch.tensor([[0, 1.0], [100, 1.0], [500, 1.0]], dtype=torch.float64)
    costs = torch.tensor([2.0, 3.0, 2.0, 10.0], dtype=torch.float64)
    return model, candidates, targets, costs


def _direct_fantasy_objective(model, candidates, targets, costs, *, seed=17):
    """Independent reference: enumerate the posterior means after each fantasy."""
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=seed)
    with torch.no_grad():
        current = model.posterior(targets).mean.max()
        fantasy = model.fantasize(candidates.unsqueeze(-2), sampler=sampler)
        posterior_means = fantasy.posterior(targets[None, None]).mean.squeeze(-1)
        return (posterior_means.max(dim=-1).values - current).mean(dim=0) / costs


def test_botorch_mfkg_matches_direct_discrete_fantasies(setup):
    model, candidates, targets, costs = setup
    scores = score_mfkg_candidates(
        model, candidates, targets, costs, num_fantasies=16, seed=17,
        candidate_batch_size=4, target_batch_size=2,
    )
    expected = _direct_fantasy_objective(model, candidates, targets, costs)
    torch.testing.assert_close(scores, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("standardize", [False, True])
def test_analytic_q1_matches_native_with_heterogeneous_observation_noise(setup, standardize):
    _, candidates, targets, costs = setup
    train_X = torch.tensor([[0, 0.2], [100, 0.2], [0, 0.5]], dtype=torch.float64)
    train_Y = torch.tensor([[20.0], [25.0], [19.0]], dtype=torch.float64)
    model = MixedSingleTaskGP(
        train_X, train_Y, cat_dims=[0],
        train_Yvar=torch.tensor([[1e-5], [2e-4], [3e-3]], dtype=torch.float64),
        outcome_transform=Standardize(m=1) if standardize else None,
    )
    model.eval()
    actual = score_mfkg_candidates(
        model, candidates, targets, costs, num_fantasies=16, seed=11,
        candidate_batch_size=2, target_batch_size=1,
    )
    native = score_mfkg_candidates_native(
        model, candidates, targets, costs, num_fantasies=16, seed=11,
        candidate_batch_size=4, target_batch_size=3,
    )
    torch.testing.assert_close(actual, native, rtol=1e-9, atol=1e-10)


def test_analytic_q1_bounds_joint_covariance_size(setup, monkeypatch):
    model, candidates, targets, costs = setup
    posterior = model.posterior
    observed_sizes = []

    def record_points(X, **kwargs):
        observed_sizes.append(X.shape[-2])
        return posterior(X, **kwargs)

    monkeypatch.setattr(model, "posterior", record_points)
    score_mfkg_candidates(
        model, candidates, targets, costs, num_fantasies=8,
        candidate_batch_size=2, target_batch_size=1,
    )
    assert max(observed_sizes) <= 3  # candidate chunk + terminal chunk


def test_chunking_preserves_common_fantasies_and_scores(setup):
    model, candidates, targets, costs = setup
    unchunked = score_mfkg_candidates(
        model, candidates, targets, costs, num_fantasies=16, seed=5,
        candidate_batch_size=4, target_batch_size=3,
    )
    for candidate_size, target_size in ((1, 1), (2, 2), (3, 1)):
        chunked = score_mfkg_candidates(
            model, candidates, targets, costs, num_fantasies=16, seed=5,
            candidate_batch_size=candidate_size, target_batch_size=target_size,
        )
        torch.testing.assert_close(unchunked, chunked, rtol=1e-9, atol=1e-11)


def test_cost_scaling_uses_physical_units_even_below_botorch_cost_floor(setup):
    model, candidates, targets, costs = setup
    scores = score_mfkg_candidates(model, candidates, targets, costs, num_fantasies=16)
    cheaper = score_mfkg_candidates(
        model, candidates, targets, costs * 1e-5, num_fantasies=16
    )
    torch.testing.assert_close(cheaper, scores / 1e-5)
    doubled = score_mfkg_candidates(
        model, candidates, targets, 2 * costs, num_fantasies=16
    )
    torch.testing.assert_close(doubled, scores / 2)


def test_negative_fantasy_deltas_are_inverse_weighted_with_same_units():
    X = torch.tensor([[[0, 0.5]], [[1, 0.5]]], dtype=torch.float64)
    deltas = torch.tensor([[-0.3, 0.7], [0.2, -0.1]], dtype=torch.float64)
    for scale in (1.0, 1e-5, 100.0):
        cost_model = GenericDeterministicModel(
            f=lambda points: scale * (points[..., :1] + 1.0)
        )
        utility = MarginalCostWeightedUtility(cost_model=cost_model)
        expected = deltas / (scale * torch.tensor([1.0, 2.0], dtype=torch.float64))
        torch.testing.assert_close(utility(X, deltas), expected)


def test_discrete_terminal_value_respects_target_set(setup):
    model, _, targets, _ = setup
    value_fn = DiscretePosteriorMaximum(model, target_X=targets, target_batch_size=1)
    with torch.no_grad():
        variance_before = model.posterior(targets).variance.clone()
        expected = model.posterior(targets).mean.max()
        actual = value_fn(torch.tensor([[12345, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        target_posterior_means(model, targets, target_batch_size=1),
        model.posterior(targets).mean.squeeze(-1),
    )
    # Skipping terminal variances must not leak into subsequent candidate
    # posterior evaluations, whose uncertainty generates the KG fantasies.
    assert bool((variance_before > 0).all())
    torch.testing.assert_close(model.posterior(targets).variance, variance_before)


def test_actual_model_fit_and_scoring():
    train_X = torch.tensor([[0, 0.2], [1, 0.2], [0, 0.5]], dtype=torch.float64)
    train_Y = torch.tensor([[0.3], [0.4], [0.5]], dtype=torch.float64)
    model = fit_mfkg_model(train_X, train_Y, cat_dims=[0], fit_maxiter=200)
    targets = torch.tensor([[0, 1.0], [1, 1.0]], dtype=torch.float64)
    candidates = torch.tensor([[0, 0.75], [1, 0.5]], dtype=torch.float64)
    scores = score_mfkg_candidates(
        model, candidates, targets, torch.ones(2, dtype=torch.float64), num_fantasies=8
    )
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_bounded_fit_reports_iteration_limit_without_discarding_parameters():
    train_X = torch.tensor([[0, 0.2], [1, 0.2], [0, 0.5]], dtype=torch.float64)
    train_Y = torch.tensor([[0.3], [0.4], [0.5]], dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="fit reached fit_maxiter"):
        model = fit_mfkg_model(train_X, train_Y, cat_dims=[0], fit_maxiter=1)
    assert not model.training
    assert torch.isfinite(model.posterior(train_X).mean).all()


@pytest.mark.parametrize("bad_costs", ([0, 1, 1, 1], [-1, 1, 1, 1], [float('nan'), 1, 1, 1]))
def test_rejects_invalid_costs(setup, bad_costs):
    model, candidates, targets, _ = setup
    with pytest.raises(ValueError, match="finite positive"):
        score_mfkg_candidates(
            model, candidates, targets, torch.tensor(bad_costs, dtype=torch.float64)
        )


def test_rejects_non_target_fidelity_and_invalid_configuration(setup):
    model, candidates, targets, costs = setup
    invalid_targets = targets.clone()
    invalid_targets[0, -1] = 0.5
    with pytest.raises(ValueError, match="full-fidelity"):
        score_mfkg_candidates(model, candidates, invalid_targets, costs)
    invalid_candidates = candidates.clone()
    invalid_candidates[0, 0] = 1234
    with pytest.raises(ValueError, match="finite target"):
        score_mfkg_candidates(model, invalid_candidates, targets, costs)


def test_empty_candidate_set(setup):
    model, candidates, targets, costs = setup
    result = score_mfkg_candidates(model, candidates[:0], targets, costs[:0])
    assert result.shape == (0,)


def test_rejects_categorical_fidelity():
    train_X = torch.tensor([[0, 0.2], [1, 0.2]], dtype=torch.float64)
    train_Y = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
    with pytest.raises(ValueError, match="exclude fidelity"):
        fit_mfkg_model(train_X, train_Y, cat_dims=[0, 1])
