"""Gaussian Thompson sampling for the full, fixed evaluation-set mean.

The normal-normal observation model and full-row posterior are the same as
Gittins'. Sampling ranks posterior draws of accuracy; it does not divide them
by arm costs, which would change the objective to accuracy per dollar. A
cost-budgeted TS run uses the same allocation rule with cost-based stopping.
Only observed cells are consumed by this policy.
"""

from __future__ import annotations

import math

import torch

from simple_regret_recommend import finite_population_posterior_moments


def gaussian_thompson_sampling(
    obs: torch.Tensor,
    *,
    prior_mean: float = 0.5,
    prior_variance: float = 0.04,
    obs_noise_variance: float | None = None,
    batch_size: int = 32,
    max_batch_size: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Draw full-row means and reveal unseen cells from the best unfinished arm.

    ``obs_noise_variance`` denotes variance of a nominal batch mean, matching
    Gittins. Its default is ``1/(4*batch_size)``; the equivalent cell variance
    is always ``obs_noise_variance * batch_size``, including partial batches.
    Completed arms remain eligible for recommendation but cannot be sampled.
    ``max_batch_size`` can cap a final batch to the remaining experiment budget.
    """
    if obs.ndim != 2 or min(obs.shape) <= 0:
        raise ValueError("obs must be a nonempty (n_arms, n_examples) matrix")
    if not obs.is_floating_point():
        raise ValueError("obs must be floating point, with NaN for unseen cells")
    if not math.isfinite(prior_mean):
        raise ValueError("prior_mean must be finite")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_batch_size is not None:
        if isinstance(max_batch_size, bool) or int(max_batch_size) != max_batch_size:
            raise ValueError("max_batch_size must be an integer")
        if max_batch_size <= 0:
            return None
    tau_sq_batch = 1.0 / (4.0 * batch_size) if obs_noise_variance is None else float(obs_noise_variance)
    if not math.isfinite(tau_sq_batch) or tau_sq_batch <= 0:
        raise ValueError("obs_noise_variance must be positive and finite")
    means, variances = finite_population_posterior_moments(
        obs,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        tau_sq_cell=tau_sq_batch * batch_size,
    )
    unfinished = obs.isnan().any(dim=1)
    if not bool(unfinished.any()):
        return None
    draws = means.to(torch.float64) + variances.sqrt() * torch.randn(
        means.shape, dtype=torch.float64, device=obs.device, generator=generator,
    )
    draws = draws.masked_fill(~unfinished, -float("inf"))
    arm = int(draws.argmax().item())
    unseen = torch.where(obs[arm].isnan())[0]
    n_batch = min(int(batch_size), int(unseen.numel()))
    if max_batch_size is not None:
        n_batch = min(n_batch, int(max_batch_size))
    columns = unseen[torch.randperm(unseen.numel(), device=obs.device, generator=generator)[:n_batch]]
    return torch.stack((torch.full_like(columns, arm), columns))
