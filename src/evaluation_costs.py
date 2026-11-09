"""Read evaluation prices without reading any complete ground-truth scores."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


def validate_per_cell_costs(values: np.ndarray, n_arms: int) -> np.ndarray:
    """Return a finite, positive cost vector in matrix-row order."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n_arms,):
        raise ValueError(f"cost vector must have shape ({n_arms},), got {values.shape}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("per-cell costs must be finite and positive")
    return values


def load_bo_per_cell_costs(path: Path, n_arms: int) -> np.ndarray:
    """Convert a BO archive's full-evaluation costs into per-cell prices.

    ``arm_ids`` aligns archive rows with matrix rows; ``metadata_json`` must
    identify the positive benchmark size ``n_examples`` used to compute the
    archived full-evaluation costs. Score arrays such as ``Y`` are never read.
    """
    if isinstance(n_arms, bool) or not isinstance(n_arms, (int, np.integer)) or n_arms < 1:
        raise ValueError("n_arms must be a positive integer")
    with np.load(path, allow_pickle=False) as archive:
        required = {"cost", "arm_ids", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"BO cost archive is missing required arrays: {sorted(missing)}")
        arm_ids = np.asarray(archive["arm_ids"])
        if (
            arm_ids.shape != (n_arms,)
            or not np.issubdtype(arm_ids.dtype, np.integer)
            or not np.array_equal(np.sort(arm_ids), np.arange(n_arms))
        ):
            raise ValueError("BO arm_ids must be an integer permutation of matrix row IDs")
        full_costs = validate_per_cell_costs(archive["cost"], n_arms)
        try:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        except (ValueError, TypeError) as exc:
            raise ValueError("BO metadata_json must be a scalar JSON object") from exc
        n_examples = metadata.get("n_examples") if isinstance(metadata, dict) else None
        if (
            isinstance(n_examples, bool)
            or not isinstance(n_examples, Real)
            or not math.isfinite(n_examples)
            or n_examples <= 0
            or not float(n_examples).is_integer()
        ):
            raise ValueError("BO metadata n_examples must be a finite positive integer")
        return validate_per_cell_costs(full_costs[np.argsort(arm_ids)] / n_examples, n_arms)
