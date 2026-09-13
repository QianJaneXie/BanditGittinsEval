"""NumPy/SciPy implementation of the existing m-diff Gaussian Gittins DP.

Equations and grid bounds match src/q_estimation.py and src/gittins_lookup.py.
The DP uses float64; indices/posterior means use float32 as in the saved runs.
This backend does not require Torch or JAX. Sampling uses NumPy PCG64.

Derived from q_estimation.py, Copyright 2025 Qian Xie, Theo Brown, Ziv Scully,
Alexander Terenin, under the repository's MIT license (see LICENSE).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def gaussian_convolve(y, sigma, dx):
    """Expectation of piecewise-linear y, with boundary slopes zero and one."""
    n = len(y)
    slope_changes = np.diff(np.r_[0.0, np.diff(y) / dx, 1.0])
    offsets = (np.arange(2 * n - 1) - (n - 1)) * dx
    z = offsets / sigma
    kernel = offsets * ndtr(z) + sigma * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    fft_size = 1 << (3 * n - 3).bit_length()
    convolution = np.fft.irfft(
        np.fft.rfft(slope_changes, fft_size) * np.fft.rfft(kernel, fft_size), fft_size)
    return y[0] + convolution[n - 1:2 * n - 1]


def compute_roots(prior_variance, n_examples, cost_scale=1e-4, tau_sq_cell=0.25, grid_points=1025):
    stages = np.arange(n_examples, dtype=np.float64)
    variance = 1.0 / (1.0 / prior_variance + stages / tau_sq_cell)
    stds = np.sqrt(variance**2 / (variance + tau_sq_cell))
    total_std = np.sqrt(np.sum(stds**2))
    grid = np.linspace(-5 * total_std, 1.01 * n_examples * cost_scale + 5 * total_std, grid_points)
    roots = np.empty(n_examples + 1, dtype=np.float32)
    q = grid.copy()
    roots[-1] = grid[np.clip(np.searchsorted(q, 0), 0, grid_points - 1)]
    for stage in range(n_examples - 1, -1, -1):
        q = gaussian_convolve(np.maximum(q, 0.0), stds[stage], grid[1] - grid[0]) - cost_scale
        roots[stage] = grid[np.clip(np.searchsorted(q, 0), 0, grid_points - 1)]
    return roots


def posterior(counts, sums, prior_mean, prior_variance, tau_sq_cell):
    variance = 1.0 / (1.0 / prior_variance + counts.astype(np.float64) / tau_sq_cell)
    mean = variance * (prior_mean / prior_variance + sums / tau_sq_cell)
    return mean.astype(np.float32), variance


def next_arm(counts, sums, roots, n_examples, prior_mean, prior_variance, tau_sq_cell):
    mean, _ = posterior(counts, sums, prior_mean, prior_variance, tau_sq_cell)
    scores = mean - roots[counts]
    scores[counts == n_examples] = -np.inf
    return int(np.argmax(scores))
