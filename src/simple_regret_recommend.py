"""Shared simple-regret recommendation helpers (empirical / posterior incumbent)."""

from __future__ import annotations

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


def posterior_means(
    obs: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> torch.Tensor:
    counts = (~obs.isnan()).sum(dim=1).to(torch.float64)
    obs_sum = torch.nan_to_num(obs, nan=0.0).sum(dim=1).to(torch.float64)
    v0 = float(prior_variance)
    prec = 1.0 / v0 + counts / float(tau_sq_cell)
    v_t = 1.0 / prec
    mus = v_t * (float(prior_mean) / v0 + obs_sum / float(tau_sq_cell))
    mus[counts == 0] = float(prior_mean)
    return mus.to(torch.float32)


def posterior_incumbent(
    obs: torch.Tensor,
    _aux: Any = None,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> tuple[int, torch.Tensor]:
    """Recommend by posterior mean E[θ_k | D_t] under the normal-normal model."""
    mus = posterior_means(
        obs,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell,
    )
    return recommend_from_means(mus)
