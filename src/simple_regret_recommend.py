"""Simple-regret recommendations for the complete, fixed evaluation matrix."""

from __future__ import annotations

import math
from typing import Any

import torch


def recommend_from_means(mus: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Argmax over per-arm means, treating NaN as -inf."""
    mus = mus.detach()
    scores = torch.where(
        torch.isnan(mus), torch.full_like(mus, -float("inf")), mus.to(torch.float32)
    )
    if not torch.isfinite(scores).any():
        return 0, mus
    return int(torch.argmax(scores).item()), mus


def empirical_incumbent(obs: torch.Tensor, _aux: Any = None) -> tuple[int, torch.Tensor]:
    """Recommend by empirical row mean (UCB-E / LRF post-pull rule)."""
    mus = torch.nanmean(obs, dim=1)
    return recommend_from_means(mus)


def _latent_posterior_moments_float64(
    obs: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute latent moments without rounding the posterior mean to float32."""
    for name, value in (("prior_variance", prior_variance), ("tau_sq_cell", tau_sq_cell)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    counts = (~obs.isnan()).sum(dim=1).to(torch.float64)
    obs_sum = torch.nan_to_num(obs, nan=0.0).sum(dim=1).to(torch.float64)
    v0 = float(prior_variance)
    prec = 1.0 / v0 + counts / float(tau_sq_cell)
    v_t = 1.0 / prec
    mus = v_t * (float(prior_mean) / v0 + obs_sum / float(tau_sq_cell))
    mus[counts == 0] = float(prior_mean)
    return mus, v_t


def posterior_moments(
    obs: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent normal-normal posterior moments using per-cell noise variance."""
    means, variances = _latent_posterior_moments_float64(
        obs,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
    )
    return means.to(torch.float32), variances


def posterior_means(
    obs: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> torch.Tensor:
    return posterior_moments(
        obs,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
    )[0]


def finite_population_posterior_moments(
    obs: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior moments of each full, realized row mean, including unrevealed cells.

    For N cells, n observations with sum S, and latent posterior (mu, v), the
    full-row mean has expectation (S + (N-n)*mu) / N and variance
    ((N-n)**2*v + (N-n)*tau_sq_cell) / N**2. The second variance term accounts
    for unrevealed cell noise. Completed rows have their empirical mean and
    zero variance; wholly unobserved rows retain the prior mean.
    """
    if obs.ndim != 2 or obs.shape[1] == 0:
        raise ValueError("obs must be a matrix with at least one cell per arm")
    # Keep both observed sums and latent means in float64 until the final cast;
    # intermediate rounding could split equal posterior means across counts.
    observations = obs.to(torch.float64)
    latent_means, latent_variances = _latent_posterior_moments_float64(
        observations,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
    )
    n_cells = obs.shape[1]
    remaining = (n_cells - (~obs.isnan()).sum(dim=1)).to(torch.float64)
    observed_sum = torch.nan_to_num(observations, nan=0.0).sum(dim=1)
    means = (observed_sum + remaining * latent_means) / n_cells
    variances = (
        remaining.square() * latent_variances + remaining * float(tau_sq_cell)
    ) / n_cells**2
    return means.to(torch.float32), variances


def recommend_from_posterior(
    mus: torch.Tensor,
    variances: torch.Tensor,
    *,
    std_penalty: float = 0.0,
) -> tuple[int, torch.Tensor]:
    """Maximize μ - std_penalty * σ over all arms (posterior lower bound).

    A penalty of 1 gives μ - σ; 0 selects the largest supplied posterior mean.
    Return the unmodified posterior means, not the penalized scores. Exact
    ties use the lowest arm index, and NaN scores are treated as -inf.
    """
    if not math.isfinite(std_penalty) or std_penalty < 0:
        raise ValueError("std_penalty must be nonnegative and finite")
    if std_penalty == 0:
        return recommend_from_means(mus)
    scores = mus.detach().to(torch.float64) - float(std_penalty) * variances.detach().to(torch.float64).sqrt()
    scores = torch.where(torch.isnan(scores), torch.full_like(scores, -float("inf")), scores)
    if not bool(torch.isfinite(scores).any()):
        return 0, mus.detach()
    return int(torch.argmax(scores).item()), mus.detach()


def posterior_incumbent(
    obs: torch.Tensor,
    _aux: Any = None,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
    std_penalty: float = 0.0,
) -> tuple[int, torch.Tensor]:
    """Recommend the full fixed-row posterior mean, with an optional std penalty."""
    mus, variances = finite_population_posterior_moments(
        obs,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
    )
    return recommend_from_posterior(
        mus,
        variances,
        std_penalty=std_penalty,
    )
