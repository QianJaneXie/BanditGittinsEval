# Copyright 2026 Qian Xie
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Matrix bandit exploration using Gittins indices (Gaussian conjugate / normal–normal)."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import torch

from gittins_shrinking_posterior import (
    compute_gittins_shrinking_posterior_walk_batch_mean,
    compute_gittins_shrinking_posterior_walk_per_observation,
)


def _cost_vector_per_arm(
    cost_per_transition: float | Sequence[float] | torch.Tensor,
    n_arms: int,
) -> torch.Tensor:
    """Return ``(n_arms,)`` float64 CPU tensor; broadcast scalars, validate vector length."""
    if isinstance(cost_per_transition, torch.Tensor):
        v = cost_per_transition.detach().reshape(-1).cpu().to(torch.float64)
    elif isinstance(cost_per_transition, numbers.Real):
        return torch.full((n_arms,), float(cost_per_transition), dtype=torch.float64)
    else:
        v = torch.tensor(list(cost_per_transition), dtype=torch.float64)
    if v.numel() == 1:
        return v.expand(n_arms).contiguous()
    if v.numel() != n_arms:
        raise ValueError(
            "cost_per_transition must be a scalar, a length-1 tensor, or length n_arms "
            f"(got {v.numel()} values, n_arms={n_arms})"
        )
    return v


def _normal_normal_posterior(
    mu0: float,
    v0: float,
    tau_sq: float,
    obs_sum: float,
    t: int,
) -> tuple[float, float]:
    """Conjugate N(μ0,v0) prior on θ and Y|θ ~ N(θ, τ²); return posterior mean and variance after t sums."""
    if t <= 0:
        return mu0, v0
    prec = 1.0 / v0 + t / tau_sq
    v_t = 1.0 / prec
    mu_t = v_t * (mu0 / v0 + obs_sum / tau_sq)
    return float(mu_t), float(v_t)


def gittins_index_exploration(
    observed_matrix: torch.Tensor,
    *,
    prior_mean: float = 0.7,
    prior_variance: float = 0.01,
    obs_noise_variance: float = 0.01,
    cost_per_transition: float | Sequence[float] | torch.Tensor = 1e-4,
    n_gittins_grid_points: int = 2**10 + 1,
    batch_size: int = 32,
    return_mus: bool = False,
    cached_scores: torch.Tensor | None = None,
    recompute_arms: Iterable[int] | None = None,
    use_batch_mean_gittins_dp: bool = False,
):
    """
    One step of Gittins-index exploration on a masked observation matrix.

    Rows are arms (methods); columns are i.i.d. example indices. Each revealed entry is modeled as
    Y | θ_k ~ N(θ_k, τ²) with τ² = ``obs_noise_variance``. A common worst-case bound for
    [0, 1]-valued outcomes uses τ² approximately 1/(4B), where **B is** ``batch_size`` (examples per
    arm per step). To
    match that approximation, set ``obs_noise_variance`` to ``1.0 / (4 * batch_size)`` (or your
    chosen B). Default prior on each θ_k is N(0.7, 0.01); override with ``prior_mean`` and
    ``prior_variance``.

    **Batch semantics (not a mixed pair minibatch):** compute the Gittins index for every arm,
    choose the single arm k* with the largest index, then evaluate **that method** on
    ``batch_size`` **distinct unevaluated examples** (columns), chosen uniformly at random. So
    ``batch[0]`` is constant (k* repeated); ``batch[1]`` lists example indices. This is
    ``batch_size`` new **example** evaluations for **one** method—not ``batch_size`` arbitrary
    (method, example) pairs that could split across methods.

    The tensor layout matches ``upper_confidence_bound_exploration`` (stacked row/col indices) so
    the same simulator can call either policy.

    Args:
        observed_matrix: (n_arms, n_examples) with NaN for unevaluated cells.
        prior_mean: μ_0 in the Gaussian prior on each θ_k (default 0.7).
        prior_variance: v_0 in the Gaussian prior on each θ_k (default 0.01).
        obs_noise_variance: τ² in Y | θ_k ~ N(θ_k, τ²). With the 1/(4B) bound above, τ² is
            approximately 1/(4B) when B = ``batch_size``.
        cost_per_transition: Cost per **transition** in the Gittins DP for each arm. A scalar
            applies to every arm; otherwise pass a sequence or ``(n_arms,)`` tensor with one value
            per row of ``observed_matrix``. The default is ``1e-4`` for light testing.
            If ``use_batch_mean_gittins_dp`` is False (default), each transition is one new per-cell
            observation and the horizon has ``n_examples`` stages. If True, each transition is one
            **batch mean** (mean of up to ``batch_size`` new cells), so the horizon has one stage per
            future batch (about ``ceil(remaining / batch_size)`` from the current posterior).
        n_gittins_grid_points: Grid size for tabular Q (see ``tabular_q_estimate``).
        batch_size: Number of **examples** to evaluate on the chosen arm this step (capped by how
            many columns are still NaN in that row).
        return_mus: If True, return ``(batch, mus)`` with ``mus`` the row-wise nanmean (same as UCB).
        cached_scores: Optional ``(n_arms,)`` float32 buffer updated **in place** when passed. After
            one step, only the pulled arm’s posterior changes, so pass the same buffer and set
            ``recompute_arms`` to that arm’s index to avoid recomputing every arm’s Gittins index.
            First call: pass ``None`` (allocates internally) or pass a buffer with
            ``recompute_arms=None`` to fill all arms.
        recompute_arms: Used only when ``cached_scores`` is not ``None``. If ``None``, recompute
            every arm that is not fully observed. Otherwise recompute only the listed arm indices
            (typically the arm evaluated on the previous step).
        use_batch_mean_gittins_dp: If True, each DP step observes the **mean** of the next batch of
            per-cell draws (likelihood variance ``τ² / b`` for batch size ``b``), matching the idea
            that learning advances once per simulator batch instead of once per matrix cell.

    Returns:
        ``batch`` with shape ``(2, b)``, ``b ≤ batch_size``, or ``None`` if every cell is observed.
        Indexing matches ``observed_matrix[row, col]`` (rows = arms/methods, columns = examples):

        * ``batch[0, i]`` — arm index (method); here the same ``k*`` for all ``i`` (only one arm
          chosen per step).
        * ``batch[1, i]`` — example index (column); ``b`` distinct previously unobserved columns for
          that arm.

        Reveal cells by assigning into ``observed_matrix[batch[0], batch[1]]``. Optionally
        ``(batch, mus)``.
    """
    observed_matrix = observed_matrix.detach()
    if observed_matrix.device.type != "cpu":
        observed_matrix = observed_matrix.cpu()

    m_methods, n_examples = observed_matrix.shape
    mus = observed_matrix.nanmean(dim=1)
    counts = (~observed_matrix.isnan()).sum(1)
    completely_sensed_mask = counts == n_examples

    if completely_sensed_mask.sum() == m_methods:
        return (None, mus) if return_mus else None

    arm_costs = _cost_vector_per_arm(cost_per_transition, m_methods)
    n_pts = jnp.uint32(int(n_gittins_grid_points))

    if cached_scores is None:
        scores = torch.empty((m_methods,), dtype=torch.float32)
        arm_indices = range(m_methods)
    else:
        if cached_scores.shape != (m_methods,) or cached_scores.dtype != torch.float32:
            raise ValueError(
                "cached_scores must have shape (n_arms,) and dtype float32; "
                f"got shape {tuple(cached_scores.shape)}, dtype {cached_scores.dtype}"
            )
        scores = cached_scores
        if recompute_arms is None:
            arm_indices = range(m_methods)
        else:
            arm_indices = sorted({int(k) for k in recompute_arms if 0 <= int(k) < m_methods})

    for k in arm_indices:
        if completely_sensed_mask[k]:
            scores[k] = float("-inf")
            continue
        t = int(counts[k].item())
        row = observed_matrix[k]
        valid = ~torch.isnan(row)
        obs_sum = float(row[valid].sum().item())
        mu_kt, v_kt = _normal_normal_posterior(
            prior_mean, prior_variance, obs_noise_variance, obs_sum, t
        )
        c_k = float(arm_costs[k].item())
        if use_batch_mean_gittins_dp:
            remaining = n_examples - t
            bsz_plan = int(batch_size)
            n_batch = (remaining + bsz_plan - 1) // bsz_plan
            transition_costs_bm = jnp.full((n_batch,), c_k, dtype=jnp.float32)
            g = compute_gittins_shrinking_posterior_walk_batch_mean(
                jnp.float32(mu_kt),
                jnp.float32(v_kt),
                jnp.float32(obs_noise_variance),
                int(batch_size),
                remaining,
                transition_costs_bm,
                n_pts,
            )
        else:
            transition_costs_per_cell = jnp.full((n_examples,), c_k, dtype=jnp.float32)
            g = compute_gittins_shrinking_posterior_walk_per_observation(
                jnp.uint32(t),
                jnp.float32(mu_kt),
                jnp.float32(prior_variance),
                jnp.float32(obs_noise_variance),
                transition_costs_per_cell,
                n_pts,
            )
        scores[k] = float(jax.device_get(g))

    best_method_index = int(torch.argmax(scores).item())
    unobserved_column_indices = (
        observed_matrix[best_method_index].isnan().nonzero().flatten()
    )
    n_unobserved = int(unobserved_column_indices.size(0))
    bsz = min(int(batch_size), n_unobserved)
    perm = torch.randperm(n_unobserved)[:bsz]
    batch = torch.stack(
        [
            torch.full((bsz,), best_method_index, dtype=torch.long),
            unobserved_column_indices[perm].long(),
        ]
    )

    if return_mus:
        return batch, mus
    return batch
