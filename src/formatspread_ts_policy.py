"""Maximization-only FormatSpread-TS with fixed, observation-independent priors.

For binary scores the update is Beta--Bernoulli. For bounded fractional
scores it is a fractional-count (power-likelihood) extension, not an exact
Bernoulli posterior. No rounding or access to unobserved scores is used.
"""
from __future__ import annotations

import math
import torch


def beta_parameters(obs, *, prior_alpha=1.0, prior_beta=1.0):
    if obs.ndim != 2 or min(obs.shape) <= 0 or not obs.is_floating_point():
        raise ValueError("obs must be a nonempty floating matrix with NaN for unseen cells")
    for value in (prior_alpha, prior_beta):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Beta prior parameters must be positive and finite")
    seen = ~obs.isnan()
    values = obs[seen]
    if not bool((torch.isfinite(values) & (values >= 0) & (values <= 1)).all()):
        raise ValueError("FormatSpread-TS requires scores in [0, 1]; scores are never clipped")
    successes = torch.where(seen, obs, 0).to(torch.float64).sum(1)
    counts = seen.sum(1)
    return successes + prior_alpha, counts - successes + prior_beta


def formatspread_thompson_sampling(
    obs, *, batch_size=8, prior_alpha=1.0, prior_beta=1.0,
    max_batch_size=None, per_arm_cost=None, remaining_cost=None,
):
    """Draw Beta values and reveal an affordable batch from the maximizing arm.

    Costs constrain eligible actions; they do not divide sampled accuracy.
    A final partial batch may be used. Completed arms cannot be pulled but
    remain eligible for empirical-mean recommendation in the experiment runner.
    Uses the seeded global torch RNG, including the Beta draw.
    """
    alpha, beta = beta_parameters(obs, prior_alpha=prior_alpha, prior_beta=prior_beta)
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_batch_size is not None:
        if isinstance(max_batch_size, bool) or int(max_batch_size) != max_batch_size or max_batch_size < 0:
            raise ValueError("max_batch_size must be a nonnegative integer")
        batch_size = min(batch_size, max_batch_size)
    sizes = obs.isnan().sum(1).clamp(max=batch_size)
    if remaining_cost is not None:
        if not math.isfinite(remaining_cost) or remaining_cost < 0:
            raise ValueError("remaining_cost must be finite and nonnegative")
        costs = torch.as_tensor(per_arm_cost, dtype=torch.float64, device=obs.device)
        if costs.shape != sizes.shape or not bool((torch.isfinite(costs) & (costs > 0)).all()):
            raise ValueError("per_arm_cost must contain positive finite per-cell costs")
        affordable = torch.floor(remaining_cost / costs + 1e-10).clamp(min=0).long()
        sizes = torch.minimum(sizes, affordable)
    eligible = sizes > 0
    if not bool(eligible.any()):
        return None
    draws = torch.distributions.Beta(alpha, beta).sample()
    arm = int(draws.masked_fill(~eligible, -torch.inf).argmax())
    unseen = torch.where(obs[arm].isnan())[0]
    columns = unseen[torch.randperm(len(unseen), device=obs.device)[:int(sizes[arm])]]
    return torch.stack((torch.full_like(columns, arm), columns))


def formatspread_prior(mean, *, alpha=None, beta=None):
    """Original beta_0=5, alpha_0=max(5*x/(1-x),1.1), or explicit pair."""
    if (alpha is None) != (beta is None):
        raise ValueError("Specify both prior alpha and beta, or neither")
    if alpha is None:
        if not math.isfinite(mean) or not 0 < mean < 1:
            raise ValueError("FormatSpread prior mean x must be strictly between 0 and 1")
        alpha, beta = max(5.0 * mean / (1.0 - mean), 1.1), 5.0
    if any(not math.isfinite(v) or v <= 0 for v in (alpha, beta)):
        raise ValueError("Beta prior parameters must be positive and finite")
    return float(alpha), float(beta)


def dataset_prior_mean(matrix):
    """Reuse the project's coarse data-specific means, without reading scores."""
    from pathlib import Path
    import json
    path = Path(matrix)
    name = str(path).lower()
    for tag, mean in (("gsm8k", .2), ("piqa", .4), ("alpaca", .2)):
        if tag in name:
            return mean
    if 'mmlu' in name:
        metadata = Path(__file__).resolve().parents[1] / 'data/MMLU_matrices/task_metadata.json'
        if metadata.is_file():
            task = path.stem.split('_synthetic_', 1)[0]
            for row in json.loads(metadata.read_text()).get('tasks', []):
                if row.get('task') == task:
                    return {'low': .4, 'medium': .6, 'high': .75}.get(row.get('dataset_prior_bucket'), .5)
    return .5
