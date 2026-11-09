"""Offline single-arm roots for the correlated marginal-index approximation.

Only the *current* marginal variance is used. Future information sharing is
ignored by the single-arm stopping problem; this is not an optimal index for
correlated arms. All dynamic programs run while building the table, never in
``lookup``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gittins_lookup import _compute_roots_for_random_walk
from gittins_shrinking_posterior import compute_finite_population_batch_roots_lookup_table

ROOT_TABLE_VERSION = "marginal_index_effective_prior_variance_v1"


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if converted != value or converted < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return converted


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _batches(n_examples: int, batch_size: int, residue: int = 0) -> np.ndarray:
    sizes = [residue] if residue else []
    remaining = n_examples - residue
    while remaining:
        size = min(batch_size, remaining)
        sizes.append(size)
        remaining -= size
    return np.asarray(sizes, dtype=np.float64)


def marginal_transition_stds(
    current_variance: float,
    remaining_items: int,
    *,
    n_examples: int,
    batch_size: int,
    tau_sq_cell: float,
) -> np.ndarray:
    """Future empirical-mean SDs when only this arm is evaluated from now on.

    For local marginal ``theta ~ N(mu,v)`` and ``m`` unseen items, the scale
    multiplying future latent posterior increments is ``(m+tau/v)/N``.
    The algebra below avoids division by ``v`` and includes its zero limit:
    known latent means still leave ``sqrt(b*tau)/N`` empirical-score noise.
    """
    n_examples = _integer(n_examples, "n_examples")
    batch_size = _integer(batch_size, "batch_size")
    remaining_items = _integer(remaining_items, "remaining_items", 0)
    if remaining_items > n_examples:
        raise ValueError("remaining_items cannot exceed n_examples")
    tau = _number(tau_sq_cell, "tau_sq_cell", positive=True)
    variance = _number(current_variance, "current_variance")
    batches = _batches(remaining_items, batch_size)
    if not batches.size:
        return np.empty(0, dtype=np.float64)
    consumed = np.r_[0.0, np.cumsum(batches)[:-1]]
    denominator = 1.0 + consumed * variance / tau
    return (remaining_items * variance + tau) / (
        n_examples * np.sqrt(denominator * (variance + tau * denominator / batches))
    )


@partial(jax.jit, static_argnames=("n_examples", "n_points"))
def _phase_roots(
    effective_variances: jax.Array,
    prices: jax.Array,
    batches: jax.Array,
    *,
    n_examples: int,
    tau_sq_cell: jax.Array,
    cost_scaling_factor: jax.Array,
    n_points: int,
) -> jax.Array:
    consumed = jnp.concatenate([jnp.zeros((1,), dtype=batches.dtype), jnp.cumsum(batches)[:-1]])

    def for_variance(w):
        # w=0 is handled by its finite, nonzero cell-noise limit.
        safe = jnp.where(w > 0, w, jnp.float32(1.0))
        posterior = 1.0 / (1.0 / safe + consumed / tau_sq_cell)
        scale = 1.0 + tau_sq_cell / (n_examples * safe)
        sigma = scale * posterior / jnp.sqrt(posterior + tau_sq_cell / batches)
        sigma = jnp.where(w > 0, sigma, jnp.sqrt(tau_sq_cell * batches) / n_examples)

        def for_price(price):
            _, roots = _compute_roots_for_random_walk(
                transition_stds=sigma,
                transition_costs=cost_scaling_factor * price * batches,
                n_points=n_points,
            )
            return roots.at[-1].set(0.0)

        return jax.vmap(for_price)(prices)

    return jax.vmap(for_variance)(effective_variances)


def compute_marginal_stopping_root(
    current_variance: float,
    remaining_items: int,
    raw_cost: float,
    *,
    n_examples: int,
    batch_size: int,
    tau_sq_cell: float,
    cost_scaling_factor: float = 1e-4,
    n_points: int = 1025,
) -> float:
    """Reference direct DP for an arbitrary current marginal (offline use).

    Reachable finite-prior states use the full-N numerical-grid convention of
    GittinsEval, including a prefix residue for irregular observation counts.
    A variance too large to represent by a positive full-N synthetic prior
    uses a suffix-only grid instead. Both solve the same future single-arm
    stopping problem; only their finite numerical grids differ.
    """
    n_examples = _integer(n_examples, "n_examples")
    batch_size = _integer(batch_size, "batch_size")
    remaining_items = _integer(remaining_items, "remaining_items", 0)
    n_points = _integer(n_points, "n_points", 2)
    variance = _number(current_variance, "current_variance")
    tau = _number(tau_sq_cell, "tau_sq_cell", positive=True)
    price = _number(raw_cost, "raw_cost")
    scale = _number(cost_scaling_factor, "cost_scaling_factor")
    if remaining_items > n_examples:
        raise ValueError("remaining_items cannot exceed n_examples")
    if remaining_items == 0:
        return 0.0
    observed = n_examples - remaining_items
    denominator = 1.0 - observed * variance / tau
    if variance == 0 or denominator > 0:
        w = variance / denominator if variance else 0.0
        batches = _batches(n_examples, batch_size, observed % batch_size)
        counts = np.r_[0, np.cumsum(batches).astype(np.int64)]
        stage = int(np.flatnonzero(counts == observed)[0])
        values = _phase_roots(
            jnp.asarray([w], dtype=jnp.float32), jnp.asarray([price], dtype=jnp.float32),
            jnp.asarray(batches, dtype=jnp.float32), n_examples=n_examples,
            tau_sq_cell=jnp.float32(tau), cost_scaling_factor=jnp.float32(scale), n_points=n_points,
        )
        return float(np.asarray(values)[0, 0, stage])
    batches = _batches(remaining_items, batch_size)
    sigmas = marginal_transition_stds(variance, remaining_items, n_examples=n_examples,
                                    batch_size=batch_size, tau_sq_cell=tau)
    _, roots = _compute_roots_for_random_walk(
        transition_stds=jnp.asarray(sigmas, dtype=jnp.float32),
        transition_costs=jnp.asarray(scale * price * batches, dtype=jnp.float32),
        n_points=n_points,
    )
    return float(np.asarray(roots)[0])


@dataclass(frozen=True)
class MarginalGittinsRootTable:
    """Roots indexed ``[raw-price, effective-prior-variance, remaining-items]``.

    The effective prior variance is ``w = 1/(1/v - n/tau)`` with ``n=N-m``.
    Each w supplies every remaining horizon in one backward DP per batch
    residue. Online interpolation is linear in the *current* variance v, not
    in w. The original independent-prior row is retained exactly.
    """

    n_examples: int
    batch_size: int
    tau_sq_cell: float
    prior_variance: float
    cost_scaling_factor: float
    n_points: int
    n_arms: int
    effective_prior_variances: np.ndarray
    costs_per_arm: np.ndarray
    roots: np.ndarray

    def __post_init__(self):
        for name, minimum in (("n_examples", 1), ("batch_size", 1), ("n_points", 2), ("n_arms", 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        for name in ("tau_sq_cell", "prior_variance"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        object.__setattr__(self, "cost_scaling_factor", _number(self.cost_scaling_factor, "cost_scaling_factor"))
        grid = np.asarray(self.effective_prior_variances, dtype=np.float64).copy()
        costs = np.asarray(self.costs_per_arm, dtype=np.float64).copy()
        roots = np.asarray(self.roots, dtype=np.float32).copy()
        if (grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
                or grid[0] != 0 or grid[-1] != self.prior_variance or (np.diff(grid) <= 0).any()):
            raise ValueError("effective_prior_variances must increase strictly from zero to prior_variance")
        if costs.ndim != 1 or not costs.size or not np.isfinite(costs).all() or (costs < 0).any() or (np.diff(costs) <= 0).any():
            raise ValueError("costs_per_arm must be finite, nonnegative, sorted unique raw prices")
        if roots.shape != (costs.size, grid.size, self.n_examples + 1) or not np.isfinite(roots).all():
            raise ValueError("roots must have shape (n_costs, n_variances, N+1) and be finite")
        if not np.array_equal(roots[:, :, 0], np.zeros_like(roots[:, :, 0])):
            raise ValueError("completed-arm roots must be exactly zero")
        for name, array in (("effective_prior_variances", grid), ("costs_per_arm", costs), ("roots", roots)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "version": ROOT_TABLE_VERSION, "n_examples": self.n_examples,
            "batch_size": self.batch_size, "tau_sq_cell": self.tau_sq_cell,
            "prior_variance": self.prior_variance, "cost_scaling_factor": self.cost_scaling_factor,
            "n_points": self.n_points, "n_arms": self.n_arms,
            "effective_prior_variances": self.effective_prior_variances.tolist(),
            "costs_per_arm": self.costs_per_arm.tolist(),
            "root_axis": "raw_cost,effective_prior_variance,remaining_items",
            "interpolation": "linear_in_current_marginal_variance",
            "transition_cost": "cost_scaling_factor * actual_batch_size * raw_per_cell_price",
            "target": "complete_fixed_benchmark_empirical_mean",
            "policy_guarantee": "marginal_index_heuristic_no_correlated_optimality_guarantee",
        }

    @property
    def settings_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def lookup(self, variances, remaining_items, costs_per_arm) -> np.ndarray:
        """Vectorized interpolation without any DP or silent extrapolation."""
        incoming = np.asarray(variances)
        epsilon = np.finfo(incoming.dtype).eps if np.issubdtype(incoming.dtype, np.floating) else np.finfo(float).eps
        v, m_float, costs = np.broadcast_arrays(np.asarray(variances, dtype=np.float64),
                                              np.asarray(remaining_items, dtype=np.float64),
                                              np.asarray(costs_per_arm, dtype=np.float64))
        if not np.isfinite(v).all() or (v < 0).any():
            raise ValueError("variances must be finite and nonnegative")
        if (not np.isfinite(m_float).all() or (m_float != np.floor(m_float)).any()
                or (m_float < 0).any() or (m_float > self.n_examples).any()):
            raise ValueError("remaining_items must contain integers in [0,N]")
        if not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("costs_per_arm must contain finite nonnegative raw prices")
        shape = v.shape
        v, m, costs = v.ravel(), m_float.astype(np.int64).ravel(), costs.ravel()
        result = np.zeros(v.size, dtype=np.float32)
        active = m > 0
        if not active.any():
            return result.reshape(shape)
        vv, mm, cc = v[active], m[active], costs[active]
        observed = self.n_examples - mm
        maximum = self.prior_variance / (1.0 + observed * self.prior_variance / self.tau_sq_cell)
        tolerance = 16 * epsilon * maximum + 64 * np.finfo(float).eps * self.prior_variance
        if (vv > maximum + tolerance).any():
            raise ValueError("current variance exceeds the reachable independent-own-observation bound")
        anchored = np.abs(vv - maximum) <= tolerance
        vv = np.minimum(vv, maximum)
        effective = vv / (1.0 - observed * vv / self.tau_sq_cell)
        effective = np.clip(effective, 0.0, self.prior_variance)
        right = np.clip(np.searchsorted(self.effective_prior_variances, effective, side="right"),
                        1, len(self.effective_prior_variances) - 1)
        left = right - 1
        w_left, w_right = self.effective_prior_variances[left], self.effective_prior_variances[right]
        v_left = w_left / (1.0 + observed * w_left / self.tau_sq_cell)
        v_right = w_right / (1.0 + observed * w_right / self.tau_sq_cell)
        weight = np.clip((vv - v_left) / (v_right - v_left), 0.0, 1.0)
        position = np.clip(np.searchsorted(self.costs_per_arm, cc), 0, len(self.costs_per_arm) - 1)
        previous = np.maximum(position - 1, 0)
        position = np.where(np.abs(self.costs_per_arm[previous] - cc) < np.abs(self.costs_per_arm[position] - cc), previous, position)
        if not np.isclose(self.costs_per_arm[position], cc, rtol=2e-7, atol=1e-12).all():
            raise ValueError("raw price is not present in the offline root table")
        lower, upper = self.roots[position, left, mm], self.roots[position, right, mm]
        values = ((1.0 - weight) * lower + weight * upper).astype(np.float32)
        values[anchored] = self.roots[position[anchored], -1, mm[anchored]]
        result[active] = values
        return result.reshape(shape)

    def _payload_hash(self) -> str:
        digest = hashlib.sha256(self.settings_hash.encode())
        for values in (self.effective_prior_variances, self.costs_per_arm, self.roots):
            digest.update(str(values.dtype).encode())
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            # Each writer owns its temporary file. Keep it on the same filesystem
            # for atomic replacement, and close it first for Windows compatibility.
            with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                    suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, metadata_json=json.dumps(self.metadata, sort_keys=True),
                                    settings_hash=self.settings_hash, payload_sha256=self._payload_hash(),
                                    effective_prior_variances=self.effective_prior_variances,
                                    costs_per_arm=self.costs_per_arm, roots=self.roots)
            for attempt in range(5):
                try:
                    temporary.replace(path)
                    break
                except PermissionError as error:
                    # Windows can briefly deny a replace while another writer
                    # replaces the same destination. Do not retry other errors.
                    if os.name != "nt" or error.winerror not in (5, 32) or attempt == 4:
                        raise
                    time.sleep(.01 * 2**attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, **expected_settings) -> "MarginalGittinsRootTable":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("version") != ROOT_TABLE_VERSION:
                raise ValueError("Incompatible marginal Gittins root-table version")
            fields = ("n_examples", "batch_size", "tau_sq_cell", "prior_variance", "cost_scaling_factor", "n_points", "n_arms")
            table = cls(**{name: metadata[name] for name in fields},
                        effective_prior_variances=data["effective_prior_variances"],
                        costs_per_arm=data["costs_per_arm"], roots=data["roots"])
            if metadata != table.metadata or str(data["settings_hash"].item()) != table.settings_hash:
                raise ValueError("Marginal root-table metadata/hash mismatch")
            if str(data["payload_sha256"].item()) != table._payload_hash():
                raise ValueError("Marginal root-table payload hash mismatch")
        for name, expected in expected_settings.items():
            if name == "n_variance_points":
                actual = len(table.effective_prior_variances)
            elif name in table.metadata:
                actual = table.metadata[name]
            else:
                raise ValueError(f"Unknown expected root-table setting: {name}")
            if name == "costs_per_arm":
                expected = np.unique(np.asarray(expected, dtype=np.float64))
            if isinstance(actual, list) or isinstance(expected, (list, tuple, np.ndarray)):
                matches = np.array_equal(np.asarray(actual), np.asarray(expected))
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"Marginal root-table setting mismatch: {name}")
        return table


def load_marginal_gittins_root_table(path: str | Path, **expected_settings) -> MarginalGittinsRootTable:
    return MarginalGittinsRootTable.load(path, **expected_settings)


def build_marginal_gittins_root_table(
    *,
    prior_variance: float,
    tau_sq_cell: float,
    n_examples: int,
    batch_size: int,
    costs_per_arm,
    n_arms: int = 122,
    n_variance_points: int = 65,
    n_points: int = 1025,
    cost_scaling_factor: float = 1e-4,
    effective_prior_variances=None,
) -> MarginalGittinsRootTable:
    """Precompute all current variances/horizons using at most B phase scans.

    The grid has zero, a geometric positive part, and the exact original prior
    anchor. Its smallest positive node uses a conservative bound on all other
    arms' possible observation precision. The zero node also covers smaller
    positive variances, without online extrapolation or dynamic programming.
    """
    n_examples = _integer(n_examples, "n_examples")
    batch_size = _integer(batch_size, "batch_size")
    n_arms = _integer(n_arms, "n_arms")
    n_points = _integer(n_points, "n_points", 2)
    n_variance_points = _integer(n_variance_points, "n_variance_points", 2)
    prior = _number(prior_variance, "prior_variance", positive=True)
    tau = _number(tau_sq_cell, "tau_sq_cell", positive=True)
    scale = _number(cost_scaling_factor, "cost_scaling_factor")
    prices = np.unique(np.asarray(costs_per_arm, dtype=np.float64).reshape(-1))
    if not prices.size or not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("costs_per_arm must contain finite nonnegative raw prices")
    if effective_prior_variances is None:
        smallest = 1.0 / (1.0 / prior + (n_arms - 1) * n_examples / tau)
        if n_variance_points == 2 or smallest == prior:
            grid = np.asarray([0.0, prior], dtype=np.float64)
        else:
            grid = np.r_[0.0, np.geomspace(smallest, prior, n_variance_points - 1)]
            grid[-1] = prior
    else:
        grid = np.asarray(effective_prior_variances, dtype=np.float64)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or grid[0] != 0
            or grid[-1] != prior or (np.diff(grid) <= 0).any()):
        raise ValueError("effective_prior_variances must increase strictly from zero to prior_variance")
    roots = np.full((len(prices), len(grid), n_examples + 1), np.nan, dtype=np.float32)
    # A 16-by-8 chunk bounds scan intermediates independently of the number of
    # distinct configurations/prices in the caller's experiment.
    for residue in range(min(batch_size, n_examples)):
        batches = _batches(n_examples, batch_size, residue)
        observed = np.r_[0, np.cumsum(batches).astype(np.int64)]
        start = 1 if residue else 0
        remaining = n_examples - observed[start:]
        for w_start in range(0, len(grid), 16):
            w_stop = min(w_start + 16, len(grid))
            for c_start in range(0, len(prices), 8):
                c_stop = min(c_start + 8, len(prices))
                phase = np.asarray(_phase_roots(
                    jnp.asarray(grid[w_start:w_stop], dtype=jnp.float32),
                    jnp.asarray(prices[c_start:c_stop], dtype=jnp.float32),
                    jnp.asarray(batches, dtype=jnp.float32), n_examples=n_examples,
                    tau_sq_cell=jnp.float32(tau), cost_scaling_factor=jnp.float32(scale), n_points=n_points,
                ))
                # phase is [w,c,stage]; assign remaining-count columns directly.
                for local_cost, cost_index in enumerate(range(c_start, c_stop)):
                    roots[cost_index, w_start:w_stop, remaining] = phase[:, local_cost, start:].T
    # Use the established helper, rather than an algebraically equivalent
    # float32 expression, at all legal independent-arm batch boundaries.
    independent = np.asarray(compute_finite_population_batch_roots_lookup_table(
        prior_variance=prior, tau_sq_cell=tau, n_examples=n_examples,
        batch_size=batch_size, costs_per_arm=prices, cost_scaling_factor=scale, n_points=n_points,
    ), dtype=np.float32)
    natural_counts = np.r_[np.arange(0, n_examples, batch_size), n_examples]
    roots[:, -1, n_examples - natural_counts] = independent
    roots[:, :, 0] = 0.0
    if not np.isfinite(roots).all():
        raise RuntimeError("Offline root phases did not cover all remaining-item counts")
    return MarginalGittinsRootTable(
        n_examples=n_examples, batch_size=batch_size, tau_sq_cell=tau,
        prior_variance=prior, cost_scaling_factor=scale, n_points=n_points, n_arms=n_arms,
        effective_prior_variances=grid, costs_per_arm=prices, roots=roots,
    )
