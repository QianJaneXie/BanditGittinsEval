"""Shared simple-regret recommendation helpers (empirical / posterior incumbent)."""

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


def posterior_moments(
    obs: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal-normal posterior means and variances using per-cell noise variance."""
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
    return mus.to(torch.float32), v_t


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


def recommend_from_posterior(
    mus: torch.Tensor,
    variances: torch.Tensor,
    *,
    std_penalty: float = 0.0,
) -> tuple[int, torch.Tensor]:
    """Maximize μ - std_penalty * σ over all arms (posterior lower bound).

    A penalty of 1 gives μ - σ; 0 retains the original posterior-mean rule.
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
    """Recommend by posterior mean minus a configurable posterior-std penalty."""
    mus, variances = posterior_moments(
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
