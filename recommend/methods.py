"""Recommendation rules applied to a shared Gittins observation trajectory.

The Gaussian rules use the repository's normal--normal working model.  The
finite-population rule instead assumes bounded outcomes in [0, 1] revealed in a
uniform random order without replacement within each arm.  No rule sees hidden
matrix entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    family: str
    parameter: float | None


@dataclass(frozen=True)
class Recommendation:
    arm: int
    score: float
    eligible_count: int


def method_specs() -> list[MethodSpec]:
    """Return the prespecified comparison grid, with no outcome-based tuning."""
    methods = [
        MethodSpec("posterior_mean", "Current posterior mean", "baseline", None),
        MethodSpec("empirical_mean", "Observed sample mean", "baseline", None),
    ]
    methods.extend(
        MethodSpec(
            f"variance_gate_{rho:g}",
            f"Variance gate: v/v0 <= {rho:g}",
            "variance_gate",
            rho,
        )
        for rho in (0.5, 0.2, 0.1, 0.05)
    )
    methods.extend(
        MethodSpec(
            f"mean_variance_{alpha:g}",
            f"Mean - {alpha:g} x variance",
            "mean_variance",
            alpha,
        )
        for alpha in (1.0, 5.0, 10.0, 20.0)
    )
    methods.extend(
        MethodSpec(
            f"gaussian_lcb_{z:g}",
            f"Gaussian lower bound: mean - {z:g} x SD",
            "gaussian_lcb",
            z,
        )
        for z in (1.0, 1.645, 1.96, 2.576)
    )
    methods.extend(
        MethodSpec(
            f"finite_gaussian_lcb_{z:g}",
            f"Finite-row Gaussian bound: mean - {z:g} x SD",
            "finite_gaussian_lcb",
            z,
        )
        for z in (1.0, 1.645, 1.96)
    )
    methods.append(
        MethodSpec(
            "finite_population_lcb",
            "Anytime finite-population lower bound (delta=0.05)",
            "confidence",
            0.05,
        )
    )
    return methods


class RecommendationRules:
    """Evaluate competing terminal decisions on the same sufficient statistics.

    An abstaining rule returns ``Recommendation(-1, nan, 0)``.  Every maximizer
    uses the first row on exact ties.  Only the current posterior-mean baseline
    casts scores to float32 before argmax, matching the native implementation.
    """

    def __init__(
        self,
        n_arms: int,
        n_examples: int,
        prior_mean: float,
        prior_variance: float,
        tau_sq_cell: float = 0.25,
    ) -> None:
        if isinstance(n_arms, bool) or int(n_arms) != n_arms or n_arms < 1:
            raise ValueError("n_arms must be a positive integer")
        if isinstance(n_examples, bool) or int(n_examples) != n_examples or n_examples < 1:
            raise ValueError("n_examples must be a positive integer")
        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        if not np.isfinite(prior_variance) or prior_variance <= 0:
            raise ValueError("prior_variance must be finite and positive")
        if not np.isfinite(tau_sq_cell) or tau_sq_cell <= 0:
            raise ValueError("tau_sq_cell must be finite and positive")
        self.n_arms = int(n_arms)
        self.n_examples = int(n_examples)
        self.prior_mean = float(prior_mean)
        self.prior_variance = float(prior_variance)
        self.tau_sq_cell = float(tau_sq_cell)
        self.specs = method_specs()
        self._radii: dict[float, np.ndarray] = {}

    def _statistics(self, counts, sums) -> tuple[np.ndarray, np.ndarray]:
        counts = np.asarray(counts, dtype=np.float64)
        sums = np.asarray(sums, dtype=np.float64)
        if counts.shape != (self.n_arms,) or sums.shape != (self.n_arms,):
            raise ValueError("counts and sums must have one entry per arm")
        if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(sums)):
            raise ValueError("counts and sums must be finite")
        if np.any(counts < 0) or np.any(counts > self.n_examples):
            raise ValueError("counts must lie between zero and n_examples")
        if np.any(counts != np.floor(counts)):
            raise ValueError("counts must be integers")
        if np.any(sums < 0) or np.any(sums > counts):
            raise ValueError("bounded [0, 1] observations require 0 <= sums <= counts")
        return counts.astype(np.int64), sums

    def posterior(self, counts, sums) -> tuple[np.ndarray, np.ndarray]:
        """Return Gaussian posterior means and variances in cell units."""
        counts, sums = self._statistics(counts, sums)
        variance = 1.0 / (1.0 / self.prior_variance + counts / self.tau_sq_cell)
        mean = variance * (
            self.prior_mean / self.prior_variance + sums / self.tau_sq_cell
        )
        mean[counts == 0] = self.prior_mean
        return mean, variance

    def _confidence_radius(self, delta: float) -> np.ndarray:
        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if delta not in self._radii:
            radius = np.zeros(self.n_examples + 1, dtype=np.float64)
            radius[0] = np.inf
            n = np.arange(1, self.n_examples, dtype=np.float64)
            # Bardenet--Maillard (2015), Propositions 2.2 and 2.3.
            correction = np.minimum(
                1.0 - (n - 1.0) / self.n_examples,
                (1.0 - n / self.n_examples) * (1.0 + 1.0 / n),
            )
            # Spend delta / (K n(n+1)) on the two-sided event at count n.
            # Sum_{n>=1} 1/(n(n+1)) = 1, so all arms and times are covered.
            radius[1:-1] = np.sqrt(
                correction
                * np.log(2.0 * self.n_arms * n * (n + 1.0) / delta)
                / (2.0 * n)
            )
            self._radii[delta] = radius
        return self._radii[delta]

    def finite_posterior(self, counts, sums) -> tuple[np.ndarray, np.ndarray]:
        """Predictive Gaussian mean/variance of the actual finite row average.

        Observed cells are known exactly. Hidden cells share the uncertain arm
        mean and each adds independent observation noise under the working model.
        """
        counts, sums = self._statistics(counts, sums)
        mean, variance = self.posterior(counts, sums)
        remaining = self.n_examples - counts
        row_mean = (sums + remaining * mean) / self.n_examples
        row_variance = (
            remaining**2 * variance + remaining * self.tau_sq_cell
        ) / self.n_examples**2
        return row_mean, row_variance

    def confidence_bounds(
        self, counts, sums, delta: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simultaneous anytime bounds for each finite matrix-row average.

        Bounds intersect Hoeffding--Serfling intervals with the deterministic
        bounds from the known sum and the number of still-hidden [0,1] cells.
        At n=0 the interval is [0,1]; at n=N it is the exact row average.
        Validity requires uniform sampling without replacement within each arm;
        adaptive selection among arms is allowed.
        """
        counts, sums = self._statistics(counts, sums)
        radius = self._confidence_radius(float(delta))[counts]
        empirical = np.divide(
            sums, counts, out=np.zeros(self.n_arms), where=counts > 0
        )
        lower = np.maximum(sums / self.n_examples, empirical - radius)
        upper = np.minimum(
            (sums + self.n_examples - counts) / self.n_examples,
            empirical + radius,
        )
        return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)

    @staticmethod
    def _recommend(scores: np.ndarray, eligible: np.ndarray) -> Recommendation:
        eligible_count = int(np.count_nonzero(eligible))
        if eligible_count == 0:
            return Recommendation(-1, float("nan"), 0)
        arm = int(np.argmax(np.where(eligible, scores, -np.inf)))
        return Recommendation(arm, float(scores[arm]), eligible_count)

    def evaluate(self, counts, sums) -> dict[str, Recommendation]:
        counts, sums = self._statistics(counts, sums)
        mean, variance = self.posterior(counts, sums)
        all_arms = np.ones(self.n_arms, dtype=bool)
        empirical = np.divide(
            sums, counts, out=np.zeros(self.n_arms), where=counts > 0
        )
        recommendations = {}
        finite_mean, finite_variance = self.finite_posterior(counts, sums)
        for spec in self.specs:
            eligible = all_arms
            if spec.key == "posterior_mean":
                score = mean.astype(np.float32)
            elif spec.key == "empirical_mean":
                score = empirical
                eligible = counts > 0
            elif spec.family == "variance_gate":
                score = mean
                eligible = variance / self.prior_variance <= spec.parameter
            elif spec.family == "mean_variance":
                score = mean - spec.parameter * variance
            elif spec.family == "gaussian_lcb":
                score = mean - spec.parameter * np.sqrt(variance)
            elif spec.family == "finite_gaussian_lcb":
                score = finite_mean - spec.parameter * np.sqrt(finite_variance)
            elif spec.family == "confidence":
                score, _ = self.confidence_bounds(counts, sums, spec.parameter)
            else:
                raise ValueError(f"unknown recommendation family: {spec.family}")
            recommendations[spec.key] = self._recommend(score, eligible)
        return recommendations
