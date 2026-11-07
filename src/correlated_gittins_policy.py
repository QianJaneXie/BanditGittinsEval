"""Correlation-aware Gittins allocation using a marginal-index approximation.

Beliefs are updated jointly using exactly the correlated Gaussian TS model.
Each arm's index then solves its *own* stopping problem through an offline
variance-dependent root table. Future observations of other arms are ignored
inside that stopping problem. The resulting indices and stopping diagnostics
are heuristic for correlated arms; independent-arm optimality does not extend.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from correlated_ts_policy import CorrelatedGaussianTS
from gittins_policy import evaluate_gittins_stopping_rules
from simple_regret_recommend import recommend_from_posterior

if TYPE_CHECKING:
    from marginal_gittins_lookup import MarginalGittinsRootTable


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class MarginalIndexGittins:
    """Joint Gaussian beliefs with single-arm marginal stopping indices.

    ``per_arm_cost`` contains raw per-cell prices. The supplied table must
    already use the same ``cost_scaling_factor`` and batch schedule. No FFT,
    dynamic program, posterior sample, or extra acquisition bonus runs online.
    A kernel refit follows correlated TS's cadence and observed-data rules.
    """

    def __init__(
        self,
        features: torch.Tensor,
        *,
        root_table: MarginalGittinsRootTable,
        n_examples: int,
        batch_size: int,
        per_arm_cost: torch.Tensor | np.ndarray | float,
        prior_mean: float = 0.5,
        prior_variance: float = 0.04,
        tau_sq_cell: float = 0.25,
        cost_scaling_factor: float = 1e-4,
        kernel_mode: str = "fit",
        fit_every: int = 32,
        fit_maxiter: int = 25,
        correlation_matrix: torch.Tensor | None = None,
    ) -> None:
        self.n_examples = _positive_integer(n_examples, "n_examples")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        if not math.isfinite(cost_scaling_factor) or cost_scaling_factor <= 0:
            raise ValueError("cost_scaling_factor must be positive and finite")
        self.cost_scaling_factor = float(cost_scaling_factor)
        expected = {
            "n_examples": self.n_examples,
            "batch_size": self.batch_size,
            "prior_variance": float(prior_variance),
            "tau_sq_cell": float(tau_sq_cell),
            "cost_scaling_factor": self.cost_scaling_factor,
        }
        for name, value in expected.items():
            actual = getattr(root_table, name, None)
            if actual is None or not math.isclose(float(actual), float(value), rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"root_table has incompatible {name}: expected {value}, got {actual}")
        self.root_table = root_table
        self.belief = CorrelatedGaussianTS(
            features,
            n_examples=self.n_examples,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            tau_sq_cell=tau_sq_cell,
            kernel_mode=kernel_mode,
            fit_every=fit_every,
            fit_maxiter=fit_maxiter,
            correlation_matrix=correlation_matrix,
        )
        self.n_arms = self.belief.n_arms
        if self.n_arms > int(getattr(root_table, "n_arms", self.n_arms)):
            raise ValueError("root_table variance grid was built for fewer arms")
        if isinstance(per_arm_cost, torch.Tensor):
            prices = per_arm_cost.detach().cpu().to(torch.float64).numpy().reshape(-1)
        else:
            prices = np.asarray(per_arm_cost, dtype=np.float64).reshape(-1)
        if prices.size == 1:
            prices = np.full(self.n_arms, float(prices[0]), dtype=np.float64)
        if prices.shape != (self.n_arms,) or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("per_arm_cost must contain one positive finite raw price per arm")
        self.per_arm_cost = prices.copy()
        self._indices_cache: torch.Tensor | None = None
        # Validate coverage immediately rather than discovering a wrong price
        # cache only after spending an evaluation batch.
        self.indices()

    @property
    def counts(self) -> torch.Tensor:
        return self.belief.counts

    @property
    def sums(self) -> torch.Tensor:
        return self.belief.sums

    @property
    def kernel_metadata(self) -> dict[str, Any]:
        return self.belief.kernel_metadata

    def empirical_target_moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Full fixed-benchmark target moments, including unseen item noise."""
        return self.belief.empirical_target_moments()

    def indices(self) -> torch.Tensor:
        """Return all marginal indices, with exact empirical terminal values.

        Float32 subtraction matches the existing independent Gittins policy's
        numerical convention at the table's exact independent-prior anchor.
        """
        if self._indices_cache is None:
            means, _ = self.empirical_target_moments()
            scores = means.to(torch.float32)
            unfinished = self.counts < self.n_examples
            if bool(unfinished.any()):
                variances = self.belief.latent_covariance.diagonal().detach()
                tolerance = 1e-12 * self.belief.prior_variance
                if bool((variances < -tolerance).any()):
                    raise RuntimeError("joint posterior has a negative marginal variance")
                selected = unfinished.numpy()
                roots = self.root_table.lookup(
                    variances.clamp_min(0).numpy()[selected],
                    (self.n_examples - self.counts).numpy()[selected],
                    self.per_arm_cost[selected],
                )
                roots = np.asarray(roots, dtype=np.float32)
                if roots.shape != (int(unfinished.sum()),) or not np.isfinite(roots).all():
                    raise RuntimeError("root table returned invalid marginal stopping roots")
                scores[unfinished] -= torch.tensor(roots, dtype=torch.float32)
            self._indices_cache = scores
        return self._indices_cache.clone()

    def observe(self, arm: int, values: torch.Tensor) -> None:
        """Update the whole belief, then refresh every arm's marginal index."""
        self.belief.observe(arm, values)
        self._indices_cache = None
        self.indices()

    def recommend(self, std_penalty: float = 1.0) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Recommend by empirical-target LCB, retaining Gittins float32 ranking.

        Means are quantized only for ranking, as in independent GittinsEval;
        the returned moments retain the joint posterior's float64 precision.
        """
        means, covariance = self.empirical_target_moments()
        arm, _ = recommend_from_posterior(
            means.to(torch.float32), covariance.diagonal().clamp_min(0), std_penalty=std_penalty,
        )
        return arm, means, covariance

    def select_arm(self, allow_early_stop: bool = False) -> int | None:
        """Choose the best unfinished arm, or honor the inherited stop signal."""
        scores = self.indices()
        complete = self.counts == self.n_examples
        if bool(complete.all()):
            return None
        if allow_early_stop and bool(complete[int(scores.argmax())]):
            return None
        return int(scores.masked_fill(complete, -torch.inf).argmax())

    def stopping_diagnostics(
        self,
        sim_cum_eval: int,
        natural_stop_cum_eval_holder: list[int | None] | None = None,
        recommendation_aware_stop_cum_eval_holder: list[int | None] | None = None,
        std_penalty: float = 1.0,
    ) -> None:
        """Record the original post-batch stopping rules as heuristic signals.

        LCB selects the recommendation. Its *unpenalized* target mean remains
        the recommendation-aware stopping threshold, as in GittinsEval.
        """
        recommended, means, _ = self.recommend(std_penalty=std_penalty)
        evaluate_gittins_stopping_rules(
            self.indices(), means.to(torch.float32), self.counts == self.n_examples,
            sim_cum_eval=int(sim_cum_eval),
            natural_stop_cum_eval_holder=natural_stop_cum_eval_holder,
            recommendation_aware_stop_cum_eval_holder=recommendation_aware_stop_cum_eval_holder,
            recommended_arm=recommended,
        )
