"""Correlated Gaussian Thompson sampling for full fixed-benchmark means.

The latent arm means have a configuration-kernel Gaussian prior.  The kernel
is BoTorch ``MixedSingleTaskGP``'s all-categorical covariance, as in the
ordinary BO baseline and the categorical components of the MF-KG surrogate.
Its diagonal is normalized before applying the supplied prior variance, so
adding correlation does not change the independent TS marginal prior.

Only revealed scores enter inference.  The empirical target additionally
contains the still-unseen cell noise; sampling a latent GP mean alone would
omit this uncertainty and would not match the independent Gaussian TS target.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.models import MixedSingleTaskGP


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def make_configuration_kernel(features: torch.Tensor):
    """Construct the existing all-categorical BO kernel without fitting data.

    The dummy targets are used only to ask BoTorch for its covariance module;
    they are discarded and never condition the correlated TS posterior.
    Configuration IDs are categorical even when represented by numbers.
    """
    if features.ndim != 2 or min(features.shape) < 1 or not torch.isfinite(features).all():
        raise ValueError("features must be a finite nonempty (n_arms, n_features) tensor")
    X = features.detach().to(device="cpu", dtype=torch.float64).clone()
    dummy = torch.zeros((len(X), 1), dtype=X.dtype)
    with warnings.catch_warnings():
        # Constant dummy values are intentional; no targets are being fitted.
        warnings.filterwarnings("ignore", category=InputDataWarning)
        model = MixedSingleTaskGP(
            train_X=X,
            train_Y=dummy,
            train_Yvar=torch.full_like(dummy, 1e-6),
            cat_dims=list(range(X.shape[1])),
            outcome_transform=None,
        )
    kernel = model.covar_module
    # Normalization fixes the latent marginal variance, making output scale
    # unidentified.  Fit the categorical lengthscales only.
    for name, parameter in kernel.named_parameters():
        parameter.requires_grad_("raw_lengthscale" in name)
    return kernel


def normalized_configuration_correlation(kernel, features: torch.Tensor) -> torch.Tensor:
    """Evaluate a kernel as a unit-diagonal correlation matrix."""
    covariance = kernel(features).to_dense()
    scale = covariance.diagonal().sqrt()
    correlation = covariance / (scale[:, None] * scale[None, :])
    return (correlation + correlation.T) * 0.5


class CorrelatedGaussianTS:
    """Sufficient-statistic correlated Gaussian posterior and joint TS draws.

    ``kernel_mode='fixed'`` keeps BoTorch's initial categorical lengthscales.
    ``'fit'`` maximizes the Gaussian marginal likelihood of the revealed arm
    means every ``fit_every`` observed batches.  Their likelihood variances
    are ``tau_sq_cell / counts``; nested cumulative means are never treated as
    independent observations.  Prior mean and marginal variance stay fixed.

    This object has no access to an accuracy matrix, costs, or complete-score
    BO archive targets.  Call ``observe`` only with newly revealed scores.
    """

    def __init__(
        self,
        features: torch.Tensor,
        *,
        n_examples: int,
        prior_mean: float = 0.5,
        prior_variance: float = 0.04,
        tau_sq_cell: float = 0.25,
        kernel_mode: str = "fixed",
        fit_every: int = 32,
        fit_maxiter: int = 25,
        correlation_matrix: torch.Tensor | None = None,
    ) -> None:
        _positive_integer(n_examples, "n_examples")
        _positive_integer(fit_every, "fit_every")
        _positive_integer(fit_maxiter, "fit_maxiter")
        if not math.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        for name, value in (("prior_variance", prior_variance), ("tau_sq_cell", tau_sq_cell)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if kernel_mode not in {"fixed", "fit"}:
            raise ValueError("kernel_mode must be 'fixed' or 'fit'")
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        if features.ndim != 2 or min(features.shape) < 1 or not torch.isfinite(features).all():
            raise ValueError("features must be a finite nonempty (n_arms, n_features) tensor")
        self.features = features.detach().to(device="cpu", dtype=torch.float64).clone()
        self.n_arms = len(self.features)
        self.n_examples = int(n_examples)
        self.prior_mean = float(prior_mean)
        self.prior_variance = float(prior_variance)
        self.tau_sq_cell = float(tau_sq_cell)
        self.kernel_mode = kernel_mode
        self.fit_every = int(fit_every)
        self.fit_maxiter = int(fit_maxiter)
        self.kernel = make_configuration_kernel(self.features)
        self._fixed_correlation = None
        if correlation_matrix is not None:
            if kernel_mode != "fixed":
                raise ValueError("an explicit correlation_matrix requires kernel_mode='fixed'")
            correlation = torch.as_tensor(correlation_matrix, dtype=torch.float64, device="cpu").clone()
            if correlation.shape != (self.n_arms, self.n_arms) or not torch.isfinite(correlation).all():
                raise ValueError("correlation_matrix must be finite with shape (n_arms, n_arms)")
            if not torch.allclose(correlation, correlation.T, atol=1e-12, rtol=1e-12):
                raise ValueError("correlation_matrix must be symmetric")
            if not torch.allclose(correlation.diagonal(), torch.ones(self.n_arms, dtype=torch.float64)):
                raise ValueError("correlation_matrix must have unit diagonal")
            if float(torch.linalg.eigvalsh(correlation).min()) < -1e-10:
                raise ValueError("correlation_matrix must be positive semidefinite")
            self._fixed_correlation = correlation
        with torch.no_grad():
            self.prior_covariance = self._kernel_covariance().detach()
        self.counts = torch.zeros(self.n_arms, dtype=torch.int64)
        self.sums = torch.zeros(self.n_arms, dtype=torch.float64)
        self.latent_mean = torch.full((self.n_arms,), self.prior_mean, dtype=torch.float64)
        self.latent_covariance = self.prior_covariance.clone()
        self.batch_count = 0
        self.fit_count = 0
        self.fit_diagnostics: list[dict[str, Any]] = []
        self._target_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._draw_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _kernel_covariance(self) -> torch.Tensor:
        correlation = (
            self._fixed_correlation if self._fixed_correlation is not None
            else normalized_configuration_correlation(self.kernel, self.features)
        )
        return self.prior_variance * correlation

    def _invalidate_cache(self) -> None:
        self._target_cache = None
        self._draw_cache = None

    @property
    def kernel_metadata(self) -> dict[str, Any]:
        lengthscales = {
            name: module.lengthscale.detach().reshape(-1).tolist()
            for name, module in self.kernel.named_modules()
            if "CategoricalKernel" in type(module).__name__
        }
        return {
            "kernel": "MixedSingleTaskGP_all_categorical_normalized",
            "kernel_mode": self.kernel_mode,
            "configuration_dimensions": self.features.shape[1],
            "categorical_lengthscales": lengthscales,
            "prior_mean": self.prior_mean,
            "prior_variance": self.prior_variance,
            "tau_sq_cell": self.tau_sq_cell,
            "fit_every": self.fit_every,
            "fit_maxiter": self.fit_maxiter,
            "fit_count": self.fit_count,
            "fit_diagnostics": list(self.fit_diagnostics),
            "explicit_correlation_matrix": self._fixed_correlation is not None,
            "target": "complete_fixed_benchmark_empirical_mean",
        }

    def observe(self, arm: int, values: torch.Tensor) -> None:
        """Condition on one batch of new scores; repeated cells are caller-invalid."""
        if isinstance(arm, bool) or int(arm) != arm or not 0 <= int(arm) < self.n_arms:
            raise ValueError("arm must be a valid integer arm index")
        arm = int(arm)
        values = torch.as_tensor(values, dtype=torch.float64, device="cpu").detach()
        if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
            raise ValueError("values must be a nonempty finite one-dimensional batch")
        size = values.numel()
        if int(self.counts[arm]) + size > self.n_examples:
            raise ValueError("observations would exceed n_examples for this arm")
        with torch.no_grad():
            column = self.latent_covariance[:, arm].clone()
            denominator = self.latent_covariance[arm, arm] + self.tau_sq_cell / size
            innovation = values.mean() - self.latent_mean[arm]
            self.latent_mean = self.latent_mean + column * (innovation / denominator)
            covariance = self.latent_covariance - torch.outer(column, column) / denominator
            self.latent_covariance = (covariance + covariance.T) * 0.5
            self.counts[arm] += size
            self.sums[arm] += values.sum()
        self.batch_count += 1
        self._invalidate_cache()
        if self.kernel_mode == "fit" and self.batch_count % self.fit_every == 0:
            self.fit_kernel()

    def empirical_target_moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and covariance of each full fixed-row mean."""
        if self._target_cache is None:
            remaining = (self.n_examples - self.counts).to(torch.float64)
            fraction = remaining / self.n_examples
            means = self.sums / self.n_examples + fraction * self.latent_mean
            covariance = fraction[:, None] * self.latent_covariance * fraction[None, :]
            covariance = covariance + torch.diag(remaining * self.tau_sq_cell / self.n_examples**2)
            self._target_cache = (means, (covariance + covariance.T) * 0.5)
        # Return copies so a caller cannot corrupt the inference cache.
        return tuple(value.clone() for value in self._target_cache)

    def sample_target(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw the joint empirical target; completed arms are deterministic."""
        means, covariance = self.empirical_target_moments()
        if self._draw_cache is None:
            unfinished = torch.where(self.counts < self.n_examples)[0]
            if unfinished.numel():
                submatrix = covariance[unfinished][:, unfinished]
                factor, info = torch.linalg.cholesky_ex(submatrix)
                if bool(info):
                    # Unseen-cell noise makes this submatrix strictly PD.
                    # An eigendecomposition handles only roundoff-scale PSD
                    # loss without adding a model-changing observation nugget.
                    eigenvalues, eigenvectors = torch.linalg.eigh(submatrix)
                    tolerance = 100 * torch.finfo(torch.float64).eps * max(1.0, float(submatrix.abs().max()))
                    if float(eigenvalues.min()) < -tolerance:
                        raise RuntimeError("empirical-target covariance is not positive semidefinite")
                    factor = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None, :]
            else:
                factor = torch.empty((0, 0), dtype=torch.float64)
            self._draw_cache = (unfinished, factor)
        unfinished, factor = self._draw_cache
        # One normal per arm mirrors independent TS's RNG consumption; the
        # full matrix factor makes these draws correlated rather than marginal.
        noise = torch.randn((self.n_arms,), dtype=torch.float64, generator=generator)
        draws = means.clone()
        if unfinished.numel():
            draws[unfinished] += factor @ noise[unfinished]
        return draws

    def _recompute_latent_posterior(self) -> None:
        """Recondition sufficient statistics after changing kernel parameters."""
        with torch.no_grad():
            prior = self._kernel_covariance().detach()
            observed = torch.where(self.counts > 0)[0]
            if not observed.numel():
                mean = torch.full_like(self.latent_mean, self.prior_mean)
                covariance = prior
            else:
                cross = prior[:, observed]
                observed_counts = self.counts[observed].to(torch.float64)
                system = prior[observed][:, observed] + torch.diag(self.tau_sq_cell / observed_counts)
                factor = torch.linalg.cholesky(system)
                residual = self.sums[observed] / observed_counts - self.prior_mean
                alpha = torch.cholesky_solve(residual[:, None], factor)
                mean = self.prior_mean + (cross @ alpha).squeeze(-1)
                covariance = prior - cross @ torch.cholesky_solve(cross.T, factor)
            self.prior_covariance = prior
            self.latent_mean = mean
            self.latent_covariance = (covariance + covariance.T) * 0.5
            self._invalidate_cache()

    def fit_kernel(self) -> bool:
        """Fit categorical lengthscales from current revealed-score statistics.

        The fixed mean and variance isolate cross-configuration borrowing from
        changes in the marginal prior.  An unsuccessful fit is rolled back.
        """
        if self._fixed_correlation is not None:
            raise ValueError("cannot fit an explicit fixed correlation_matrix")
        observed = torch.where(self.counts > 0)[0]
        if observed.numel() < 2:
            return False
        parameters = [parameter for parameter in self.kernel.parameters() if parameter.requires_grad]
        initial = [parameter.detach().clone() for parameter in parameters]
        counts = self.counts[observed].to(torch.float64)
        residual = self.sums[observed] / counts - self.prior_mean
        diagonal_noise = self.tau_sq_cell / counts

        def objective() -> torch.Tensor:
            covariance = self._kernel_covariance()[observed][:, observed] + torch.diag(diagonal_noise)
            factor = torch.linalg.cholesky(covariance)
            alpha = torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
            return 0.5 * residual.dot(alpha) + factor.diagonal().log().sum()

        before = float(objective().detach())
        optimizer = torch.optim.LBFGS(parameters, max_iter=self.fit_maxiter, line_search_fn="strong_wolfe")
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite categorical-kernel marginal likelihood")
            loss.backward()
            evaluations += 1
            return loss

        failure = None
        try:
            optimizer.step(closure)
            after = float(objective().detach())
            if not math.isfinite(after) or after > before + 1e-8:
                failure = "fit did not improve the marginal likelihood"
        except (RuntimeError, ValueError) as error:
            failure = str(error)
            after = before
        if failure is not None:
            with torch.no_grad():
                for parameter, value in zip(parameters, initial):
                    parameter.copy_(value)
            after = before
        self.fit_count += 1
        self.fit_diagnostics.append({
            "batch": self.batch_count,
            "observed_arms": int(observed.numel()),
            "negative_log_likelihood_before": before,
            "negative_log_likelihood_after": after,
            "objective_evaluations": evaluations,
            "accepted": failure is None,
            "failure": failure,
        })
        self._recompute_latent_posterior()
        return failure is None
