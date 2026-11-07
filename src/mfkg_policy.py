"""Discrete, cost-aware multi-fidelity knowledge gradient using BoTorch.

Inputs contain categorical configuration features followed by a continuous
fidelity (the fraction of a fixed, seeded example prefix evaluated).  The
surrogate is a joint GP over configuration and fidelity.  Prefix means are
deterministic lower-fidelity objectives, so repeated observations at different
fidelities are correlated by the GP rather than treated as independent noisy
estimates of the same objective.  ``observation_noise`` is a numerical floor.

The objective is BoTorch's ``qMultiFidelityKnowledgeGradient`` with q=1.
An exact Gaussian conditioning identity evaluates its fantasy terminal means
efficiently; the native BoTorch evaluation remains available as a reference.
Its terminal value is the largest posterior mean over the *actual finite arm
set* at full fidelity.  Enumerating this terminal set avoids optimizing numeric
category identifiers as though they were continuous configuration parameters.
"""

from __future__ import annotations

import math
import warnings

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.fit import DEFAULT_WARNING_HANDLER, fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import skip_posterior_variances


def _validate_points(X: torch.Tensor, *, name: str, allow_empty: bool = False) -> None:
    if X.ndim != 2 or X.shape[1] < 2 or (not allow_empty and X.shape[0] == 0):
        raise ValueError(f"{name} must have shape (n, d), with d >= 2 and n > 0")
    if X.dtype not in (torch.float32, torch.float64) or not torch.isfinite(X).all():
        raise ValueError(f"{name} must contain finite float32 or float64 values")
    if bool(((X[:, -1] <= 0) | (X[:, -1] > 1)).any()):
        raise ValueError(f"{name}'s last column must be a fidelity in (0, 1]")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _bounded_fit_warning_handler(warning: warnings.WarningMessage) -> bool:
    # Reaching an explicit iteration budget leaves usable fitted parameters.
    # Report that fact once, rather than asking BoTorch to restart five times
    # and roll those parameters back.  Other optimizer failures retain
    # BoTorch's normal retry/error policy.
    message = str(warning.message)
    if "TOTAL NO. OF ITERATIONS REACHED LIMIT" in message:
        warnings.warn(
            "MF-KG GP fit reached fit_maxiter; using the parameters at that "
            "iteration. Increase fit_maxiter to check convergence.",
            RuntimeWarning,
            stacklevel=3,
        )
        return True
    return DEFAULT_WARNING_HANDLER(warning)


def fit_mfkg_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    *,
    cat_dims: list[int],
    observation_noise: float = 1e-6,
    fit_maxiter: int = 100,
) -> MixedSingleTaskGP:
    """Fit a categorical/configuration + continuous/fidelity joint GP.

    All configuration columns must be categorical.  Use an arm-id column when
    configuration metadata is unavailable.  ``train_Y`` consists only of the
    prefix means actually revealed so far; full-row means are never required.
    """
    _validate_points(train_X, name="train_X")
    if train_Y.shape != (train_X.shape[0], 1):
        raise ValueError("train_Y must have shape (len(train_X), 1)")
    if train_Y.dtype != train_X.dtype or train_Y.device != train_X.device:
        raise ValueError("train_X and train_Y must have the same dtype and device")
    if not torch.isfinite(train_Y).all():
        raise ValueError("train_Y must contain finite values")
    if sorted(cat_dims) != list(range(train_X.shape[1] - 1)):
        raise ValueError("cat_dims must contain every configuration column and exclude fidelity")
    if not math.isfinite(observation_noise) or observation_noise <= 0:
        raise ValueError("observation_noise must be positive and finite")
    _positive_integer(fit_maxiter, "fit_maxiter")
    model = MixedSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=cat_dims,
        train_Yvar=torch.full_like(train_Y, float(observation_noise)),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": fit_maxiter}},
        warning_handler=_bounded_fit_warning_handler,
    )
    model.eval()
    model.likelihood.eval()
    return model


class DiscretePosteriorMaximum(AcquisitionFunction):
    """Exact finite-set terminal value for each fantasy model.

    ``X`` carries the fantasy/t-batch shape required by BoTorch's one-shot KG
    interface.  Its terminal point coordinates are unused because the inner
    maximization is performed exactly over ``target_X`` instead.
    """

    def __init__(
        self,
        model: Model,
        *,
        target_X: torch.Tensor,
        target_batch_size: int = 128,
        posterior_transform=None,
    ) -> None:
        super().__init__(model=model)
        self.register_buffer("target_X", target_X)
        self.target_batch_size = target_batch_size
        self.posterior_transform = posterior_transform

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        best = None
        for points in self.target_X.split(self.target_batch_size):
            # Singleton batch dimensions broadcast over fantasy and candidate
            # batches without materializing a covariance over all target arms.
            points = points.view(*([1] * len(self.model.batch_shape)), *points.shape)
            # Terminal KG values only use posterior means. Avoid constructing
            # predictive covariance terms across every fantasy/target batch;
            # candidate fantasization outside this scope still uses variance.
            with skip_posterior_variances(True):
                means = self.model.posterior(
                    points, posterior_transform=self.posterior_transform
                ).mean.squeeze(-1)
            chunk_best = means.max(dim=-1).values
            best = chunk_best if best is None else torch.maximum(best, chunk_best)
        return best


def target_posterior_means(
    model: Model,
    target_X: torch.Tensor,
    *,
    target_batch_size: int = 128,
) -> torch.Tensor:
    """Posterior means over the full-fidelity finite arm set, in input order."""
    _validate_points(target_X, name="target_X")
    _positive_integer(target_batch_size, "target_batch_size")
    if not torch.allclose(target_X[:, -1], torch.ones_like(target_X[:, -1])):
        raise ValueError("target_X must contain only full-fidelity points (last column = 1)")
    with torch.no_grad(), skip_posterior_variances(True):
        return torch.cat(
            [model.posterior(points).mean.squeeze(-1) for points in target_X.split(target_batch_size)],
            dim=-1,
        )


def _project_full_fidelity(X: torch.Tensor) -> torch.Tensor:
    projected = X.clone()
    projected[..., -1] = 1.0
    return projected


def _candidate_cost_model(
    candidate_X: torch.Tensor, normalized_costs: torch.Tensor
) -> GenericDeterministicModel:
    def cost(X: torch.Tensor) -> torch.Tensor:
        matches = (X.unsqueeze(-2) == candidate_X).all(dim=-1)
        if not bool(matches.any(dim=-1).all()):
            raise ValueError("Cost requested for a point outside the candidate set")
        indices = matches.to(torch.int64).argmax(dim=-1)
        return normalized_costs[indices].unsqueeze(-1)

    return GenericDeterministicModel(f=cost)


class MarginalCostWeightedUtility(InverseCostWeightedUtility):
    """Use E[terminal improvement] / known cost consistently across versions.

    BoTorch 0.9.5 inverse-weights every fantasy delta.  Some newer releases
    multiply negative deltas by cost instead, changing the estimate and even
    its units under a cost rescaling.  With deterministic costs the intended
    KG-per-cost objective is unambiguously the mean improvement divided by
    cost; this override preserves that objective on both versions.
    """

    def forward(self, X, deltas, sampler=None, **kwargs):
        costs = self.cost_objective(self.cost_model.posterior(X).mean).sum(dim=-1)
        if not bool((costs > 0).all()):
            raise ValueError("Candidate evaluation costs must be strictly positive")
        return deltas / costs


def score_mfkg_candidates(
    model: Model,
    candidate_X: torch.Tensor,
    target_X: torch.Tensor,
    candidate_costs: torch.Tensor,
    *,
    num_fantasies: int = 32,
    seed: int = 0,
    candidate_batch_size: int = 8,
    target_batch_size: int = 128,
    use_analytic_conditioning: bool = True,
) -> torch.Tensor:
    """Return Monte Carlo MF-KG / marginal cost for every legal candidate.

    ``candidate_costs`` must be the *additional* evaluation costs after prefix
    reuse, including arm-dependent prices when appropriate.  Unit-cost runs
    still charge the number of newly revealed examples, so querying a larger
    subset is never free.  Returned scores retain the original cost units.

    Shared, seeded Sobol fantasy samples make comparisons reproducible and
    independent of the candidate chunk size.  Estimates can be slightly
    negative at finite fantasy count; no clipping or heuristic fallback changes
    the acquisition objective.

    For the single-output, fixed-noise Gaussian surrogate used here, q=1
    fantasy posterior means are evaluated by the exact Gaussian conditioning
    identity. This shares BoTorch's noisy posterior and Sobol sampler, while
    avoiding repeated fantasy-model kernel evaluations. Set
    ``use_analytic_conditioning=False`` to use native qMFKG as a reference.
    Other surrogate types retain the native implementation.
    """
    _validate_points(candidate_X, name="candidate_X", allow_empty=True)
    _validate_points(target_X, name="target_X")
    for name, value in (("num_fantasies", num_fantasies),
                        ("candidate_batch_size", candidate_batch_size),
                        ("target_batch_size", target_batch_size)):
        _positive_integer(value, name)
    if candidate_X.shape[1] != target_X.shape[1]:
        raise ValueError("candidate_X and target_X must have the same feature dimension")
    if candidate_X.dtype != target_X.dtype or candidate_X.device != target_X.device:
        raise ValueError("candidate_X and target_X must have the same dtype and device")
    if candidate_costs.shape != (candidate_X.shape[0],):
        raise ValueError("candidate_costs must have shape (len(candidate_X),)")
    if candidate_costs.dtype != candidate_X.dtype or candidate_costs.device != candidate_X.device:
        raise ValueError("candidate_costs must have the same dtype and device as candidate_X")
    if not torch.isfinite(candidate_costs).all() or bool((candidate_costs <= 0).any()):
        raise ValueError("candidate_costs must contain finite positive values")
    if not torch.allclose(target_X[:, -1], torch.ones_like(target_X[:, -1])):
        raise ValueError("target_X must contain only full-fidelity points (last column = 1)")
    if candidate_X.shape[0] == 0:
        return candidate_costs.clone()
    if torch.unique(candidate_X, dim=0).shape[0] != candidate_X.shape[0]:
        raise ValueError("candidate_X must not contain duplicate points")
    _, configuration_ids = torch.unique(
        torch.cat([candidate_X[:, :-1], target_X[:, :-1]]), dim=0, return_inverse=True
    )
    if not bool(torch.isin(
        configuration_ids[:len(candidate_X)], configuration_ids[len(candidate_X):]
    ).all()):
        raise ValueError("Every candidate configuration must belong to the finite target arm set")

    # An unbatched surrogate is fitted for each experiment; fantasy/candidate
    # batches are created below.  This keeps the returned score vector 1D.
    if len(model.batch_shape) != 0:
        raise ValueError("score_mfkg_candidates requires an unbatched surrogate")
    current_value = target_posterior_means(
        model, target_X, target_batch_size=target_batch_size
    ).max()
    if (
        use_analytic_conditioning
        and isinstance(model, MixedSingleTaskGP)
        and isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
        and model.num_outputs == 1
        and (
            not hasattr(model, "outcome_transform")
            or isinstance(model.outcome_transform, Standardize)
        )
    ):
        return _score_q1_gaussian_conditioning(
            model, candidate_X, target_X, candidate_costs,
            current_value=current_value, num_fantasies=num_fantasies, seed=seed,
            candidate_batch_size=candidate_batch_size,
            target_batch_size=target_batch_size,
        )
    cost_scale = candidate_costs.min()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]), seed=int(seed))
    scores = []
    with torch.no_grad():
        for start in range(0, candidate_X.shape[0], candidate_batch_size):
            points = candidate_X[start:start + candidate_batch_size]
            costs = candidate_costs[start:start + candidate_batch_size]
            cost_model = _candidate_cost_model(points, costs / cost_scale)
            acquisition = qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=num_fantasies,
                sampler=sampler,
                current_value=current_value,
                cost_aware_utility=MarginalCostWeightedUtility(cost_model=cost_model),
                project=_project_full_fidelity,
                valfunc_cls=DiscretePosteriorMaximum,
                valfunc_argfac=lambda model: {
                    "target_X": target_X,
                    "target_batch_size": target_batch_size,
                },
            )
            # q=1 actual point + the interface's fantasy terminal coordinates.
            # The exact finite-set terminal value ignores these placeholders.
            dummy_terminals = target_X[0].expand(points.shape[0], num_fantasies, -1)
            augmented_X = torch.cat([points.unsqueeze(-2), dummy_terminals], dim=-2)
            scores.append(acquisition(augmented_X) / cost_scale)
    result = torch.cat(scores)
    if not torch.isfinite(result).all():
        raise RuntimeError("MF-KG returned non-finite candidate scores")
    return result


def _score_q1_gaussian_conditioning(
    model: MixedSingleTaskGP,
    candidate_X: torch.Tensor,
    target_X: torch.Tensor,
    candidate_costs: torch.Tensor,
    *,
    current_value: torch.Tensor,
    num_fantasies: int,
    seed: int,
    candidate_batch_size: int,
    target_batch_size: int,
) -> torch.Tensor:
    """Exact q=1 KG via conditional means, with bounded covariance blocks.

    For a noisy fantasy Y at x, the terminal mean is
    mu(T) + Cov(f(T), f(x)) / Var(Y) * (Y - mu(x)). Observation noise and
    Standardize transforms match ``FantasizeMixin.fantasize``. No terminal
    arms or candidate fidelities are pruned, and every fantasy is retained.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]), seed=int(seed))
    # BoTorch fantasize uses mean fixed noise in outcome-transformed units.
    observation_noise = model.likelihood.noise.unsqueeze(-1).mean(dim=-2, keepdim=True)
    scores = []
    with torch.no_grad():
        for start in range(0, candidate_X.shape[0], candidate_batch_size):
            points = candidate_X[start:start + candidate_batch_size]
            noisy_posterior = model.posterior(
                points.unsqueeze(-2), observation_noise=observation_noise
            )
            fantasy_residuals = (
                sampler(noisy_posterior) - noisy_posterior.mean
            ).squeeze(-1).squeeze(-1)
            noisy_variances = noisy_posterior.variance.squeeze(-1).squeeze(-1)
            best = None
            for targets in target_X.split(target_batch_size):
                n_targets = targets.shape[0]
                joint = model.posterior(torch.cat([targets, points], dim=-2))
                cross = joint.distribution.covariance_matrix[:n_targets, n_targets:].transpose(0, 1)
                terminal_means = (
                    joint.mean[:n_targets, 0][None, None, :]
                    + fantasy_residuals[:, :, None]
                    * (cross / noisy_variances[:, None])[None, :, :]
                )
                chunk_best = terminal_means.max(dim=-1).values
                best = chunk_best if best is None else torch.maximum(best, chunk_best)
            scores.append(
                (best - current_value).mean(dim=0)
                / candidate_costs[start:start + points.shape[0]]
            )
    result = torch.cat(scores)
    if not torch.isfinite(result).all():
        raise RuntimeError("MF-KG returned non-finite candidate scores")
    return result


def score_mfkg_candidates_native(*args, **kwargs) -> torch.Tensor:
    """Native BoTorch qMFKG reference for the same finite-set q=1 objective."""
    kwargs["use_analytic_conditioning"] = False
    return score_mfkg_candidates(*args, **kwargs)
