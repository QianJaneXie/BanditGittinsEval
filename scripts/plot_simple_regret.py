#!/usr/bin/env python3
"""
Simulate exploration on a fixed accuracy matrix (rows = models, columns = i.i.d. samples) by
revealing masked entries, and plot simple regret:

    μ* − μ_{âₜ}

where μ* is the best true row mean and âₜ is the recommended model after each batch. For UCB-E /
UCB-E-LRF / round-robin, âₜ is the arm with largest **row empirical mean** (among rows with data).
For **Gittins**, âₜ uses the **conjugate Gaussian posterior mean** per arm (same N(μ₀, v₀) prior and
τ² as in ``gittins_index_exploration``), not the sample mean.

Compared algorithms:
  - upper_confidence_bound_exploration (UCB-E)
  - upper_confidence_bound_exploration_low_rank_factorization (UCB-E-LRF)
  - gittins_index_exploration with τ² = 1/(4B), B = ``--gittins-batch-size`` (default 20; worst-case
    bound on [0, 1]). Gittins uses **batch-mean DP** (one DP stage per that batch) by default; pass
    ``--gittins-per-cell-dp`` for one DP stage per matrix cell instead. UCB-E / LRF use ``--batch-size``.

The horizontal axis is cumulative matrix entries evaluated (cost-unaware) or cumulative cost
(cost-aware). **Budget:** ``--eval-budget-fraction`` caps **evaluations** as a fraction of matrix
cells when ``--gittins-cost-mode unaware``; when **aware**, the same fraction applies to **total
monetary cost** to fill the whole matrix (each cell of arm ``k`` costs ``c_k``). UCB-E runs over the full budget from
the first query. UCB-E-LRF spends ``--warmup-percentage`` (e.g. 5%) on uniform random probing, then
runs the low-rank UCB rule for the rest; **only the post–warm-up segment** is drawn for UCB-E-LRF
so the curve begins where that policy starts (at ~5% cumulative evals when the defaults are 5%
warm-up and 10% total).

**Experiments:** ``--experiment NAME`` (default ``gsm8k_various_model``) picks a registered matrix and
figure path from ``experiment_specs()``; ``--matrix`` / ``--out`` override those defaults so the script
is benchmark-agnostic (any ``(n_models, n_examples)`` accuracy ``.npy`` works). Gittins prior defaults
to N(0.5, 0.04); the ``gsm8k_various_model`` preset uses N(0.2, 0.01). Override with
``--gittins-prior-mean`` / ``--gittins-prior-variance``.

By default all three algorithms are simulated and plotted. For a quick test (e.g. Gittins only with a
small budget), use ``--algorithms gittins`` and ``--eval-budget-fraction 0.02`` (or similar).

Use ``--verbose`` to print each step: distinct arms in the batch, incumbent arm, and simple regret
(tagged ``ucb`` / ``lrf`` / ``gittins`` when multiple algorithms run).

**Gittins cost** (same semantics as ``gittins_policy.cost_per_transition``): ``--gittins-cost-mode
unaware`` passes **1.0** per arm (uniform cost) and plots regret vs cumulative matrix evaluations.
``--gittins-cost-mode aware`` loads **per-arm monetary costs** (e.g. dollars per transition) from
``--gittins-cost-vector``; the DP scales them via ``cost_scaling_factor``. For the bundled GSM8K
configurations pricing JSON, values are **USD per 1M input tokens** and the cumulative-cost axis
uses that unit (other pricing files follow whatever unit the file declares). If UCB/LRF (or RR) run
alongside cost-aware Gittins, the figure plots **one** panel with
cumulative cost on the x-axis for every curve when traces include per-method cumulative cost (saved
automatically for new runs). Legacy trace bundles without ``ucb_x_original_cost`` /
``lrf_x_original_cost`` still use two panels (evals vs cost).

**Traces:** by default, cumulative-evaluation counts and simple regret series are written next to the
figure as ``<figure_stem>_traces.npz`` (see ``--traces-out`` / ``--no-save-traces``). Arrays:

- ``ucb_x``, ``ucb_regret`` — UCB-E (empty if not run)
- ``ucb_x_original_cost`` — cumulative monetary cost after each UCB batch (cost-aware only; same units
  as the cost vector)
- ``lrf_x_full``, ``lrf_regret_full`` — UCB-E-LRF full trace (empty if not run)
- ``lrf_x_plot``, ``lrf_regret_plot`` — LRF segment used in the figure (cum eval ≥ ``warmup_evals``)
- ``lrf_x_original_cost``, ``lrf_x_plot_original_cost`` — full and post–warm-up cumulative cost for LRF
  (cost-aware only)
- ``gittins_x``, ``gittins_regret`` — Gittins (empty if not run)
- ``gittins_x_original_cost`` — cumulative monetary cost after each step (same units as the cost
  vector; bundled GSM8K configurations pricing JSON: **USD per 1M input tokens**). Same length as
  ``gittins_regret`` when cost-aware Gittins runs.
- scalar ``gittins_stop_cum_eval`` — first cumulative eval count (before that policy step) where the
  top-scoring arm was already fully observed (nominal stopping time); ``-1`` if that never occurred.
  The Gittins regret curve still runs to the eval budget when using the simple-regret script.
- scalars ``warmup_evals``, ``budget_evals``, ``tau_sq_gittins``

**Gittins-only rerun with UCB/LRF unchanged:** pass ``--algorithms gittins`` and
``--merge-ucb-lrf-from PREVIOUS_traces.npz`` to copy UCB-E and UCB-E-LRF series from an earlier
full run and simulate only Gittins (same matrix / seed / budget recommended).

Replot without resimulating: ``python scripts/replot_simple_regret_from_traces.py --traces …`` (use
``--gittins-x-axis original_cost`` for cumulative monetary cost vs ``evals`` for cost-unaware-style
x-axis; see replot script for axis labels).

**Timing (``*_traces.meta.json``):** when traces are saved, ``timing`` records per-method wall-clock
stats from ``time.time()`` (see ``summary.iter_total`` / ``summary.iter_step``). Pass
``--timing-include-per-iter-series`` to also store every iteration's seconds in the meta file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch

from banditeval.bandits import (
    upper_confidence_bound_exploration,
    upper_confidence_bound_exploration_low_rank_factorization,
)

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from gittins_policy import _normal_normal_posterior, gittins_index_exploration

T = TypeVar("T", int, float)

# Cost-aware figures default x-axis label (matches the bundled GSM8K configurations pricing JSON,
# which stores ``estimated_cost_per_1m_input_tokens``). Other pricing files may use other units;
# adjust this label if you switch.
_GITTINS_COST_AWARE_CUMULATIVE_XLABEL = "Cumulative cost (USD per 1M input tokens)"


@dataclass(frozen=True)
class ExperimentSpec:
    """Named experiment: default matrix and figure output (paths relative to repo root)."""

    matrix: Path
    out: Path
    # Optional Gittins prior θ_k ~ N(μ_0, v_0); None → use global default N(0.5, 0.04) unless CLI overrides.
    gittins_prior_mean: float | None = None
    gittins_prior_variance: float | None = None


def experiment_specs(root: Path) -> dict[str, ExperimentSpec]:
    """Register named presets here. Each preset only fixes default ``--matrix`` / ``--out`` (and
    optionally a Gittins prior); CLI flags always override. Add new benchmarks by appending an entry."""
    return {
        "gsm8k_various_model": ExperimentSpec(
            matrix=root / "data" / "BanditEval_matrices" / "gsm8k_1_samples_various_models_seed1.npy",
            out=root / "outputs" / "figures" / "simple_regret_gsm8k_various_models_seed1.png",
            gittins_prior_mean=0.2,
            gittins_prior_variance=0.01,
        ),
        "piqa_various_models": ExperimentSpec(
            matrix=root / "data" / "BanditEval_matrices" / "piqa_1_samples_various_models_seed1.npy",
            out=root / "outputs" / "figures" / "simple_regret_piqa_various_models_seed1.png",
        ),
    }


def _resolve_gittins_prior(spec: ExperimentSpec, args: Namespace) -> tuple[float, float]:
    """Resolve Gittins prior (μ_0, v_0): CLI > experiment preset > global default N(0.5, 0.04).

    The ``gsm8k_various_model`` preset, for example, sets N(0.2, 0.01)."""
    default_mean, default_var = 0.5, 0.04
    if args.gittins_prior_mean is not None:
        mean = float(args.gittins_prior_mean)
    elif spec.gittins_prior_mean is not None:
        mean = float(spec.gittins_prior_mean)
    else:
        mean = default_mean
    if args.gittins_prior_variance is not None:
        var = float(args.gittins_prior_variance)
    elif spec.gittins_prior_variance is not None:
        var = float(spec.gittins_prior_variance)
    else:
        var = default_var
    return mean, var


def _is_configurations_pricing(data: object) -> bool:
    """Detect ``data_analysis/pricing/..._configurations_...json`` shape:
    ``{"0": {"estimated_cost_per_1m_input_tokens": ...}, "1": {...}, ...}``. The bundled GSM8K
    pricing JSON uses this layout; any benchmark that matches the same shape is also accepted."""
    if not isinstance(data, dict) or "0" not in data:
        return False
    z = data["0"]
    return isinstance(z, dict) and "estimated_cost_per_1m_input_tokens" in z


def _json_configurations_to_1d(data: dict, n_arms: int) -> np.ndarray:
    out: list[float] = []
    for i in range(n_arms):
        k = str(i)
        if k not in data:
            raise ValueError(f"missing configuration key {k!r} (expected 0..{n_arms - 1})")
        row = data[k]
        if not isinstance(row, dict) or "estimated_cost_per_1m_input_tokens" not in row:
            raise ValueError(f"entry {k!r} must include estimated_cost_per_1m_input_tokens")
        out.append(float(row["estimated_cost_per_1m_input_tokens"]))
    return np.asarray(out, dtype=np.float64)


def _json_to_cost_1d(data: object, n_arms: int) -> np.ndarray:
    """Interpret JSON root as a length-``n_arms`` vector (list or dict with an array field)."""
    if isinstance(data, list):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
    elif isinstance(data, dict):
        for key in ("cost_per_arm", "costs", "cost"):
            if key in data and isinstance(data[key], list):
                arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
                break
        else:
            raise ValueError(
                "JSON object must contain a list field: cost_per_arm, costs, or cost (or use a JSON array at the root)"
            )
    else:
        raise ValueError("JSON root must be an array or an object with a cost array")
    if arr.shape != (n_arms,):
        raise ValueError(f"cost vector length must be n_arms={n_arms}, got {arr.shape[0]}")
    return arr


def load_gittins_cost_vector_from_file(path: Path, n_arms: int) -> torch.Tensor:
    """
    Load per-arm per-cell costs from disk → ``(n_arms,)`` float64 tensor. **Units:** match your file
    (the bundled GSM8K ``configurations`` JSON uses **USD per 1M input tokens** per
    ``estimated_cost_per_1m_input_tokens``).

    - ``.npy``: ``numpy.load`` (shape ``(n_arms,)``).
    - ``.json``: flat array / ``cost_per_arm`` list, or a ``configurations`` export (keys ``\"0\"``…
      with ``estimated_cost_per_1m_input_tokens`` per arm; same shape as the bundled GSM8K file).

    Call this first, then pass the returned tensor into ``gittins_index_exploration`` (via
    ``make_gittins_step_with_score_cache``) and ``simulate(..., per_arm_original_cost=...)``.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.squeeze(np.load(path))
    elif suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if _is_configurations_pricing(data):
            arr = _json_configurations_to_1d(data, n_arms)
        else:
            arr = _json_to_cost_1d(data, n_arms)
    else:
        text = path.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(text)
            if _is_configurations_pricing(data):
                arr = _json_configurations_to_1d(data, n_arms)
            else:
                arr = _json_to_cost_1d(data, n_arms)
        except (json.JSONDecodeError, ValueError):
            arr = np.squeeze(np.load(path))
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.shape != (n_arms,):
        raise ValueError(f"cost vector must have shape (n_arms,) = ({n_arms},); got {arr.shape}")
    return torch.tensor(arr, dtype=torch.float64)


def parse_gittins_cost_string(s: str, n_arms: int) -> torch.Tensor:
    """Build ``(n_arms,)`` float64 cost tensor: float, comma-separated list, or JSON array."""
    s = s.strip()
    if not s:
        raise ValueError("empty gittins cost string")
    if s.startswith("["):
        data = json.loads(s)
        if not isinstance(data, list):
            raise ValueError("JSON cost must be a list of numbers")
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.shape != (n_arms,):
            raise ValueError(f"expected {n_arms} values in JSON list, got {arr.shape[0]}")
        return torch.tensor(arr, dtype=torch.float64)
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 1:
            return torch.full((n_arms,), float(parts[0]), dtype=torch.float64)
        if len(parts) != n_arms:
            raise ValueError(f"expected {n_arms} comma-separated values, got {len(parts)}")
        return torch.tensor([float(p) for p in parts], dtype=torch.float64)
    return torch.full((n_arms,), float(s), dtype=torch.float64)


def gittins_cost_to_meta_string(tensor: torch.Tensor) -> str:
    """Compact string for ``*_traces.meta.json``: scalar if all entries equal, else JSON array."""
    t = tensor.detach().cpu().reshape(-1).to(torch.float64)
    if t.numel() == 1 or bool(torch.all(t == t[0]).item()):
        return str(float(t[0].item()))
    return json.dumps([float(x) for x in t.tolist()])


def load_gittins_cost_from_meta(meta: dict, n_arms: int) -> torch.Tensor:
    """Rebuild ``(n_arms,)`` costs from trace meta (legacy paths and embedded JSON). Used by ``plot_arm_eval_rollouts``."""
    legacy = meta.get("gittins_cost_per_arm")
    if legacy:
        p = Path(legacy)
        if p.is_file():
            c_np = np.squeeze(np.load(p))
            if c_np.shape != (n_arms,):
                raise ValueError(
                    f"legacy gittins_cost_per_arm must have shape (n_arms,) = ({n_arms},); "
                    f"got {c_np.shape}"
                )
            return torch.tensor(c_np, dtype=torch.float64)

    raw = meta.get("gittins_cost", "1.0")
    if isinstance(raw, list):
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.shape != (n_arms,):
            raise ValueError(f"gittins_cost list must have length {n_arms}, got {arr.shape[0]}")
        return torch.tensor(arr, dtype=torch.float64)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return torch.full((n_arms,), float(raw), dtype=torch.float64)

    s = str(raw).strip()
    p = Path(s)
    if p.suffix == ".npy" and p.is_file():
        c_np = np.squeeze(np.load(p))
        if c_np.shape != (n_arms,):
            raise ValueError(
                f"gittins_cost .npy must have shape (n_arms,) = ({n_arms},); got {c_np.shape}"
            )
        return torch.tensor(c_np, dtype=torch.float64)

    return parse_gittins_cost_string(s, n_arms)


def make_gittins_step_with_score_cache(**gittins_kwargs):
    """Return ``step(obs, **_)`` that reuses per-arm Gittins scores and only recomputes the last pulled arm."""

    cache: dict[str, torch.Tensor | int | None] = {"scores": None, "prev_arm": None}

    def step(obs: torch.Tensor, **kwargs) -> torch.Tensor | None:
        sim_cum_eval = kwargs.pop("sim_cum_eval", None)
        m = int(obs.shape[0])
        scores = cache["scores"]
        if scores is None:
            scores = torch.full((m,), float("inf"), dtype=torch.float32)
            cache["scores"] = scores
            recompute_arms = None
        else:
            prev = cache["prev_arm"]
            recompute_arms = None if prev is None else [int(prev)]

        batch = gittins_index_exploration(
            obs,
            cached_scores=scores,
            recompute_arms=recompute_arms,
            sim_cum_eval=sim_cum_eval,
            **gittins_kwargs,
        )
        if batch is not None:
            cache["prev_arm"] = int(batch[0, 0].item())
        return batch

    return step


def make_round_robin_step(*, batch_size: int) -> Callable[[torch.Tensor], torch.Tensor | None]:
    """Round-robin baseline: cycle arms, evaluate ``batch_size`` new columns on that arm.

    Incumbent recommendation is still the sample-mean argmax (handled by ``simulate()``).
    """

    state: dict[str, int] = {"next_arm": 0}

    def step(obs: torch.Tensor, **_: object) -> torch.Tensor | None:
        m, n = obs.shape
        if m <= 0 or n <= 0:
            return None

        start = int(state["next_arm"]) % m
        chosen: int | None = None
        for i in range(m):
            k = (start + i) % m
            if torch.isnan(obs[k]).any():
                chosen = k
                break
        if chosen is None:
            return None

        unobs = torch.isnan(obs[chosen]).nonzero().flatten()
        if unobs.numel() == 0:
            return None
        bsz = min(int(batch_size), int(unobs.numel()))
        perm = torch.randperm(int(unobs.numel()))[:bsz]
        cols = unobs[perm].long()
        rows = torch.full((bsz,), int(chosen), dtype=torch.long)
        state["next_arm"] = (int(chosen) + 1) % m
        return torch.stack([rows, cols])

    return step


def _lists_from_npz_trace(z: Any, xkey: str, ykey: str) -> tuple[list[int], list[float]]:
    x = z[xkey]
    y = z[ykey]
    if x.size == 0:
        return [], []
    return x.astype(np.int64).tolist(), y.astype(np.float64).tolist()


def _trim_trace_from_cum_eval(xs: list[int], ys: list[T], min_x: int) -> tuple[list[int], list[T]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x >= min_x]
    if not pairs:
        return [], []
    ox, oy = zip(*pairs)
    return list(ox), list(oy)


def _cost_series_aligned(xs_eval: list[int], xs_cost: list[float], regrets: list[float]) -> bool:
    """True when cumulative-cost x-axis can pair with regret (same length, non-empty)."""
    return (
        len(xs_eval) == len(xs_cost) == len(regrets) and len(xs_cost) > 0
    )


def _cost_aligned_if_curve(
    *,
    want_curve: bool,
    xs_eval: list[int],
    xs_cost: list[float],
    regrets: list[float],
) -> bool:
    """If we draw this policy's curve (non-empty regret), require aligned cost series; else OK."""
    if not want_curve:
        return True
    if not regrets:
        return True
    return _cost_series_aligned(xs_eval, xs_cost, regrets)


def _cumulative_cost_at_eval_stop(
    xs_eval: list[int], xs_cost: list[float], stop_eval: int | None
) -> float | None:
    """First step where cumulative evals reach ``stop_eval``; return matching cumulative cost (cost-vector units)."""
    if stop_eval is None or stop_eval < 0 or not xs_eval or not xs_cost:
        return None
    for i, e in enumerate(xs_eval):
        if e >= stop_eval:
            return float(xs_cost[i]) if i < len(xs_cost) else float(xs_cost[-1])
    return float(xs_cost[-1])


def incumbent_from_empirical_means(observed: torch.Tensor) -> int:
    """Recommend arm with largest row nan-mean; rows with no data are excluded."""
    row_means = torch.nanmean(observed, dim=1)
    scores = torch.where(torch.isnan(row_means), torch.full_like(row_means, -float("inf")), row_means)
    if not torch.isfinite(scores).any():
        return 0
    return int(torch.argmax(scores).item())


def incumbent_from_gittins_posterior(
    observed: torch.Tensor,
    *,
    prior_mean: float,
    prior_variance: float,
    tau_sq: float,
) -> int:
    """Argmax arm by conjugate posterior mean E[θ_k | data], matching the Gittins Gaussian model."""
    observed = observed.detach()
    m = int(observed.shape[0])
    scores = torch.empty((m,), dtype=torch.float64)
    for k in range(m):
        row = observed[k]
        valid = ~torch.isnan(row)
        t = int(valid.sum().item())
        if t == 0:
            scores[k] = prior_mean
        else:
            obs_sum = float(row[valid].sum().item())
            mu_k, _ = _normal_normal_posterior(prior_mean, prior_variance, tau_sq, obs_sum, t)
            scores[k] = mu_k
    return int(torch.argmax(scores).item())


def simulate(
    ground_truth: torch.Tensor,
    step: Callable[..., torch.Tensor | None],
    *,
    step_kwargs: dict,
    seed: int,
    max_evaluations: int | None = None,
    max_cumulative_cost: float | None = None,
    verbose: bool = False,
    log_prefix: str = "",
    pass_sim_cum_eval: bool = False,
    per_arm_original_cost: torch.Tensor | None = None,
    incumbent_fn: Callable[[torch.Tensor], int] | None = None,
) -> tuple[list[float], list[int], dict[str, Any]]:
    """Run until a budget is reached, the matrix is exhausted, or ``batch is None``.

    Returns parallel lists: regret after each batch, and cumulative number of
    entries revealed (batch sizes summed), including warm-up queries for LRF.

    Pass at least one of ``max_evaluations`` (cap on revealed cells) or ``max_cumulative_cost``
    (cap on cumulative monetary cost, requires ``per_arm_original_cost``).

    If ``pass_sim_cum_eval`` is True, each ``step`` call also receives
    ``sim_cum_eval=<cumulative evals so far>`` (for Gittins nominal stopping-time bookkeeping).

    If ``per_arm_original_cost`` is a length-``(n_arms,)`` tensor, ``timing["cum_original_cost"]``
    records cumulative monetary cost after each batch in the same units as ``per_arm_original_cost``
    (e.g. dollars per transition when cost-aware; bundled GSM8K configurations JSON: **USD per 1M
    input tokens**), summed as
    ``c_k * n_cells_in_batch`` per step.

    If ``incumbent_fn`` is set, it maps ``observed_matrix`` → incumbent arm index for simple regret;
    otherwise ``incumbent_from_empirical_means`` is used.

    No new batch is started once cumulative evaluations reach ``max_evaluations`` (if set) or
    cumulative cost reaches ``max_cumulative_cost`` (if set, checked at the start of each iteration).

    If ``verbose``, prints after each batch: distinct row indices (arms) in the batch,
    incumbent arm (used for simple regret), and simple regret.
    """
    torch.manual_seed(seed)

    if max_evaluations is None and max_cumulative_cost is None:
        raise ValueError("simulate requires max_evaluations and/or max_cumulative_cost")
    if max_cumulative_cost is not None and per_arm_original_cost is None:
        raise ValueError("max_cumulative_cost requires per_arm_original_cost")

    if per_arm_original_cost is not None and tuple(per_arm_original_cost.shape) != (
        ground_truth.shape[0],
    ):
        raise ValueError(
            "per_arm_original_cost must have shape (n_arms,) = "
            f"({ground_truth.shape[0]},); got {tuple(per_arm_original_cost.shape)}"
        )

    obs = torch.full_like(ground_truth, float("nan"))
    true_means = ground_truth.mean(dim=1)
    mu_star = float(true_means.max().item())

    regrets: list[float] = []
    cum_evaluated: list[int] = []
    cum_original_cost: list[float] = []
    evaluated = 0
    total_original_cost = 0.0
    tag = log_prefix or "sim"
    iter_total_s: list[float] = []
    iter_step_s: list[float] = []

    while True:
        if max_evaluations is not None and evaluated >= max_evaluations:
            break
        if max_cumulative_cost is not None and total_original_cost >= max_cumulative_cost:
            break
        t_iter0 = time.time()
        call_kw = dict(step_kwargs)
        if pass_sim_cum_eval:
            call_kw["sim_cum_eval"] = evaluated
        t_step0 = time.time()
        batch = step(obs, **call_kw)
        t_step1 = time.time()
        if batch is None:
            break
        row_idx, col_idx = batch
        n_batch = int(row_idx.numel())
        obs[row_idx, col_idx] = ground_truth[row_idx, col_idx]
        evaluated += n_batch

        if per_arm_original_cost is not None:
            pulled_arm = int(row_idx[0].item())
            unit = float(per_arm_original_cost[pulled_arm].item())
            total_original_cost += unit * n_batch
            cum_original_cost.append(total_original_cost)

        if incumbent_fn is not None:
            arm = incumbent_fn(obs)
        else:
            arm = incumbent_from_empirical_means(obs)
        mu_sel = float(true_means[arm].item())
        simple_regret = mu_star - mu_sel
        regrets.append(simple_regret)
        cum_evaluated.append(evaluated)

        if verbose:
            distinct_arms = sorted({int(x) for x in row_idx.reshape(-1).tolist()})
            print(
                f"[{tag}] step {len(regrets)}: cum_eval={evaluated} "
                f"batch_arms(distinct)={distinct_arms} incumbent_arm={arm} "
                f"simple_regret={simple_regret:.6f}",
                flush=True,
            )

        t_iter1 = time.time()
        iter_total_s.append(float(t_iter1 - t_iter0))
        iter_step_s.append(float(t_step1 - t_step0))

    timing: dict[str, Any] = {
        "clock": "time.time",
        "unit": "seconds",
        "iter_total_s": iter_total_s,
        "iter_step_s": iter_step_s,
    }
    if per_arm_original_cost is not None:
        timing["cum_original_cost"] = cum_original_cost
    return regrets, cum_evaluated, timing


def _final_cum_cost(timing: dict[str, Any] | None) -> float | None:
    if timing is None:
        return None
    c = timing.get("cum_original_cost")
    if not c:
        return None
    return float(c[-1])


def _timing_summary(times_s: list[float]) -> dict[str, float | int]:
    if not times_s:
        return {"n": 0}
    arr = np.asarray(times_s, dtype=np.float64)
    return {
        "n": int(arr.size),
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "median_s": float(np.quantile(arr, 0.5)),
        "p90_s": float(np.quantile(arr, 0.9)),
        "p99_s": float(np.quantile(arr, 0.99)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def _timing_for_meta(
    timing: dict[str, Any] | None, *, include_per_iter_series: bool
) -> dict[str, Any] | None:
    """Serialize simulate() timing for JSON meta (summary always; full series optional)."""
    if timing is None:
        return None
    out: dict[str, Any] = {
        "clock": timing["clock"],
        "unit": timing["unit"],
        "summary": {
            "iter_total": _timing_summary(timing["iter_total_s"]),
            "iter_step": _timing_summary(timing["iter_step_s"]),
        },
    }
    if include_per_iter_series:
        out["iter_total_s"] = list(timing["iter_total_s"])
        out["iter_step_s"] = list(timing["iter_step_s"])
    return out


def save_trace_bundle(
    path: Path,
    *,
    xs_rr: list[int],
    regrets_rr: list[float],
    xs_ucbe: list[int],
    regrets_ucbe: list[float],
    xs_lrf: list[int],
    regrets_lrf: list[float],
    xs_lrf_plot: list[int],
    regrets_lrf_plot: list[float],
    xs_rr_original_cost: list[float] | None,
    xs_ucbe_original_cost: list[float] | None,
    xs_lrf_original_cost: list[float] | None,
    xs_lrf_plot_original_cost: list[float] | None,
    xs_gittins: list[int],
    regrets_gittins: list[float],
    xs_gittins_original_cost: list[float] | None,
    gittins_stop_cum_eval: int | None,
    warmup_evals: int,
    budget_evals: int,
    tau_sq_gittins: float,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_scalar = np.int64(-1 if gittins_stop_cum_eval is None else int(gittins_stop_cum_eval))
    np.savez_compressed(
        path,
        rr_x=np.asarray(xs_rr, dtype=np.int64),
        rr_regret=np.asarray(regrets_rr, dtype=np.float64),
        ucb_x=np.asarray(xs_ucbe, dtype=np.int64),
        ucb_regret=np.asarray(regrets_ucbe, dtype=np.float64),
        lrf_x_full=np.asarray(xs_lrf, dtype=np.int64),
        lrf_regret_full=np.asarray(regrets_lrf, dtype=np.float64),
        lrf_x_plot=np.asarray(xs_lrf_plot, dtype=np.int64),
        lrf_regret_plot=np.asarray(regrets_lrf_plot, dtype=np.float64),
        rr_x_original_cost=np.asarray(
            xs_rr_original_cost if xs_rr_original_cost is not None else [],
            dtype=np.float64,
        ),
        ucb_x_original_cost=np.asarray(
            xs_ucbe_original_cost if xs_ucbe_original_cost is not None else [],
            dtype=np.float64,
        ),
        lrf_x_original_cost=np.asarray(
            xs_lrf_original_cost if xs_lrf_original_cost is not None else [],
            dtype=np.float64,
        ),
        lrf_x_plot_original_cost=np.asarray(
            xs_lrf_plot_original_cost if xs_lrf_plot_original_cost is not None else [],
            dtype=np.float64,
        ),
        gittins_x=np.asarray(xs_gittins, dtype=np.int64),
        gittins_regret=np.asarray(regrets_gittins, dtype=np.float64),
        gittins_x_original_cost=np.asarray(
            xs_gittins_original_cost if xs_gittins_original_cost is not None else [],
            dtype=np.float64,
        ),
        gittins_stop_cum_eval=stop_scalar,
        warmup_evals=np.int64(warmup_evals),
        budget_evals=np.int64(budget_evals),
        tau_sq_gittins=np.float64(tau_sq_gittins),
    )
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


_GITTINS_COST_EPILOG = """
Gittins cost modes (--gittins-cost-mode), matching gittins_policy.cost_per_transition:
  unaware — per-arm cost 1.0 (uniform); plot Gittins vs cumulative matrix evaluations.
  aware — per-arm monetary costs from --gittins-cost-vector (bundled GSM8K configurations JSON: USD
    per 1M input tokens; other files follow whatever unit they declare); DP scales via
    cost_scaling_factor; plot vs cumulative cost (one panel for all policies when traces include
    UCB/LRF cost series; legacy traces may still use two panels).
  JSON shapes: flat array, cost_per_arm/costs/cost, or "configurations" export (keys "0".. with
    estimated_cost_per_1m_input_tokens per arm; same shape as the bundled GSM8K file).
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    specs = experiment_specs(root)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_GITTINS_COST_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=sorted(specs.keys()),
        default="gsm8k_various_model",
        metavar="NAME",
        help="Which preset to run (default matrix and --out); override with --matrix / --out.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Path to (n_models, n_examples) accuracy matrix (.npy); default from --experiment",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to save the figure; default from --experiment",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for exploration randomness")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="UCB-E and UCB-E-LRF: examples per step (default: 32)",
    )
    parser.add_argument(
        "--gittins-batch-size",
        type=int,
        default=20,
        help="Gittins: examples per step and B in τ² = 1/(4B) (default: 20)",
    )
    parser.add_argument(
        "--eval-budget-fraction",
        type=float,
        default=0.10,
        help="Budget fraction: cost-unaware = fraction of matrix cells (eval cap); cost-aware = fraction "
        "of total cost to evaluate the full matrix (sum over arms of c_k × n_examples) (default: 10%%)",
    )
    parser.add_argument("--ucb-a", type=int, default=1, dest="a")
    parser.add_argument(
        "--warmup-percentage",
        type=float,
        default=0.05,
        help="UCB-E-LRF: fraction of matrix that must be observed before low-rank UCB replaces random probing",
    )
    parser.add_argument("--lrf-device", type=str, default="cpu")
    parser.add_argument(
        "--gittins-grid-points",
        type=int,
        default=2**10 + 1,
        help="Tabular Q grid resolution for gittins_index_exploration",
    )
    parser.add_argument(
        "--gittins-cost-mode",
        type=str,
        choices=["unaware", "aware"],
        default="unaware",
        help="unaware: uniform cost 1.0 per arm (see gittins_policy); plot vs evals. aware: monetary "
        "per-arm costs from --gittins-cost-vector (bundled GSM8K configurations JSON: USD per 1M input "
        "tokens; other files follow their own unit); plot vs cumulative cost.",
    )
    parser.add_argument(
        "--gittins-cost",
        type=float,
        default=1.0,
        metavar="C",
        help="Unused for Gittins cost-unaware mode (DP uses 1.0). Reserved for uniform non-1 scenarios.",
    )
    parser.add_argument(
        "--gittins-cost-vector",
        type=Path,
        default=None,
        metavar="FILE",
        help="Required for --gittins-cost-mode aware: per-arm monetary costs (JSON or .npy); the "
        "bundled GSM8K configurations export uses USD per 1M input tokens. Ignored when unaware.",
    )
    parser.add_argument(
        "--gittins-per-cell-dp",
        action="store_true",
        help="Gittins: one DP stage per matrix cell (long horizon). Default is batch-mean DP aligned with --gittins-batch-size",
    )
    parser.add_argument(
        "--gittins-prior-mean",
        type=float,
        default=None,
        metavar="μ0",
        help="μ_0 for θ_k ~ N(μ_0, v_0). Default N(0.5, 0.04); experiment gsm8k_various_model uses (0.2, 0.01) unless set.",
    )
    parser.add_argument(
        "--gittins-prior-variance",
        type=float,
        default=None,
        metavar="V0",
        help="v_0 for θ_k ~ N(μ_0, v_0). Default N(0.5, 0.04); experiment gsm8k_various_model uses (0.2, 0.01) unless set.",
    )
    parser.add_argument(
        "--traces-out",
        type=Path,
        default=None,
        help="Save regret traces as compressed .npz (default: next to --out, stem + _traces.npz)",
    )
    parser.add_argument(
        "--no-save-traces",
        action="store_true",
        help="Do not write trace .npz or .meta.json",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["rr", "ucb", "lrf", "gittins"],
        default=["ucb", "lrf", "gittins"],
        metavar="NAME",
        help="Which algorithms to run and plot (default: all three). Example: --algorithms gittins",
    )
    parser.add_argument(
        "--merge-ucb-lrf-from",
        type=Path,
        default=None,
        help="Use with --algorithms gittins only: load UCB-E and UCB-E-LRF traces from this .npz "
        "and simulate only Gittins. LRF plot trim uses warmup_evals stored in that .npz when present.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each simulation step: arms in batch, incumbent arm, simple regret",
    )
    parser.add_argument(
        "--timing-include-per-iter-series",
        action="store_true",
        help="When saving traces meta, include per-iteration time arrays (iter_total_s / iter_step_s); "
        "default is summary statistics only to keep .meta.json small.",
    )
    args = parser.parse_args()

    spec = specs[args.experiment]
    if args.matrix is None:
        args.matrix = spec.matrix
    if args.out is None:
        args.out = spec.out

    gittins_prior_mean, gittins_prior_variance = _resolve_gittins_prior(spec, args)

    algorithms = list(dict.fromkeys(args.algorithms))
    merge_ucb_lrf = args.merge_ucb_lrf_from
    if merge_ucb_lrf is not None:
        if set(algorithms) != {"gittins"}:
            print(
                "With --merge-ucb-lrf-from, use exactly: --algorithms gittins",
                file=sys.stderr,
            )
            return 1
        if not merge_ucb_lrf.is_file():
            print(f"--merge-ucb-lrf-from not found: {merge_ucb_lrf}", file=sys.stderr)
            return 1

    if args.gittins_cost_mode == "aware" and "gittins" not in algorithms:
        print(
            "--gittins-cost-mode aware requires --algorithms to include gittins",
            file=sys.stderr,
        )
        return 1

    if args.batch_size <= 0:
        print("--batch-size must be positive", file=sys.stderr)
        return 1
    if args.gittins_batch_size <= 0:
        print("--gittins-batch-size must be positive", file=sys.stderr)
        return 1
    if not args.matrix.is_file():
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    gt_np = np.load(args.matrix)
    if gt_np.ndim != 2:
        print(f"Expected a 2D matrix, got shape {gt_np.shape}", file=sys.stderr)
        return 1

    ground_truth = torch.tensor(gt_np, dtype=torch.float32)
    n_arms = int(ground_truth.shape[0])

    cost_vector_path: Path | None = args.gittins_cost_vector
    if args.gittins_cost_mode == "aware":
        if cost_vector_path is None:
            print("--gittins-cost-mode aware requires --gittins-cost-vector", file=sys.stderr)
            return 1
        try:
            gittins_cost_tensor = load_gittins_cost_vector_from_file(cost_vector_path, n_arms)
        except FileNotFoundError:
            print(f"gittins cost file not found: {cost_vector_path}", file=sys.stderr)
            return 1
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"gittins cost vector: {e}", file=sys.stderr)
            return 1
    else:
        if cost_vector_path is not None:
            print(
                "Note: --gittins-cost-vector ignored in cost-unaware mode "
                "(Gittins DP uses 1.0 per transition per arm).",
                file=sys.stderr,
            )
        gittins_cost_tensor = torch.full((n_arms,), 1.0, dtype=torch.float64)

    n_cells = int(ground_truth.numel())
    n_examples = int(ground_truth.shape[1])
    total_full_matrix_cost: float | None = None
    budget_max_cumulative_cost: float | None = None
    if args.gittins_cost_mode == "aware":
        total_full_matrix_cost = float(n_examples * gittins_cost_tensor.sum().item())
        budget_max_cumulative_cost = float(args.eval_budget_fraction) * total_full_matrix_cost
        budget_max_evals = n_cells
    else:
        budget_max_evals = max(1, int(round(args.eval_budget_fraction * n_cells)))

    warmup_evals = int(np.ceil(args.warmup_percentage * n_cells))
    warmup_evals_lrf_trim = warmup_evals

    budget_evals = budget_max_evals

    if "lrf" in algorithms and warmup_evals >= budget_max_evals:
        print(
            "Warm-up threshold (ceil(warmup %% × n)) must be < eval budget cap; "
            "raise --eval-budget-fraction or lower --warmup-percentage.",
            file=sys.stderr,
        )
        return 1

    if args.gittins_cost_mode == "aware" and "lrf" in algorithms and total_full_matrix_cost is not None:
        max_c = float(gittins_cost_tensor.max().item())
        if warmup_evals * max_c > budget_max_cumulative_cost:
            print(
                "Warning: worst-case LRF warm-up cost (warmup_evals × max arm cost) exceeds "
                f"cost budget {budget_max_cumulative_cost:.6g}; consider lowering --warmup-percentage.",
                file=sys.stderr,
            )

    tau_sq_gittins = 1.0 / (4.0 * float(args.gittins_batch_size))

    sim_kwargs: dict[str, Any] = {
        "seed": args.seed,
        "max_evaluations": budget_max_evals,
        "verbose": args.verbose,
    }
    if budget_max_cumulative_cost is not None:
        sim_kwargs["max_cumulative_cost"] = budget_max_cumulative_cost

    per_arm_cost_for_simulate: torch.Tensor | None = (
        gittins_cost_tensor if args.gittins_cost_mode == "aware" else None
    )

    regrets_ucbe: list[float] = []
    xs_ucbe: list[int] = []
    xs_ucbe_original: list[float] = []
    regrets_lrf: list[float] = []
    xs_lrf: list[int] = []
    xs_lrf_original: list[float] = []
    regrets_rr: list[float] = []
    xs_rr: list[int] = []
    xs_rr_original: list[float] = []
    regrets_gittins: list[float] = []
    xs_gittins: list[int] = []
    xs_gittins_original: list[float] = []
    gittins_stop_cum_eval: int | None = None
    timing_ucb: dict[str, Any] | None = None
    timing_lrf: dict[str, Any] | None = None
    timing_rr: dict[str, Any] | None = None
    timing_gittins: dict[str, Any] | None = None

    if merge_ucb_lrf is not None:
        z_merge = np.load(merge_ucb_lrf)
        if "warmup_evals" in z_merge.files:
            warmup_evals_lrf_trim = int(z_merge["warmup_evals"].reshape(()))
        xs_ucbe, regrets_ucbe = _lists_from_npz_trace(z_merge, "ucb_x", "ucb_regret")
        xs_lrf, regrets_lrf = _lists_from_npz_trace(z_merge, "lrf_x_full", "lrf_regret_full")
        if not xs_ucbe and not xs_lrf:
            print(
                "Merge file has empty UCB and LRF traces; check --merge-ucb-lrf-from path.",
                file=sys.stderr,
            )
            return 1
        if "ucb_x_original_cost" in z_merge.files and z_merge["ucb_x_original_cost"].size:
            xs_ucbe_original = z_merge["ucb_x_original_cost"].astype(np.float64).tolist()
        if "lrf_x_original_cost" in z_merge.files and z_merge["lrf_x_original_cost"].size:
            xs_lrf_original = z_merge["lrf_x_original_cost"].astype(np.float64).tolist()
        if "rr_x_original_cost" in z_merge.files and z_merge["rr_x_original_cost"].size:
            xs_rr_original = z_merge["rr_x_original_cost"].astype(np.float64).tolist()

    if "ucb" in algorithms:
        regrets_ucbe, xs_ucbe, timing_ucb = simulate(
            ground_truth,
            upper_confidence_bound_exploration,
            step_kwargs={"a": args.a, "batch_size": args.batch_size, "return_mus": False},
            log_prefix="ucb",
            per_arm_original_cost=per_arm_cost_for_simulate,
            **sim_kwargs,
        )
        xs_ucbe_original = list(timing_ucb.get("cum_original_cost", []))
    if "lrf" in algorithms:
        regrets_lrf, xs_lrf, timing_lrf = simulate(
            ground_truth,
            upper_confidence_bound_exploration_low_rank_factorization,
            step_kwargs={
                "a": args.a,
                "batch_size": args.batch_size,
                "return_mus": False,
                "warmup_percentage": args.warmup_percentage,
                "device": args.lrf_device,
            },
            log_prefix="lrf",
            per_arm_original_cost=per_arm_cost_for_simulate,
            **sim_kwargs,
        )
        xs_lrf_original = list(timing_lrf.get("cum_original_cost", []))
    if "rr" in algorithms:
        regrets_rr, xs_rr, timing_rr = simulate(
            ground_truth,
            make_round_robin_step(batch_size=args.batch_size),
            step_kwargs={},
            log_prefix="rr",
            per_arm_original_cost=per_arm_cost_for_simulate,
            **sim_kwargs,
        )
        xs_rr_original = list(timing_rr.get("cum_original_cost", []))
    if "gittins" in algorithms:
        gittins_natural_stop_holder: list[int | None] = [None]
        regrets_gittins, xs_gittins, timing_gittins = simulate(
            ground_truth,
            make_gittins_step_with_score_cache(
                batch_size=args.gittins_batch_size,
                return_mus=False,
                obs_noise_variance=tau_sq_gittins,
                cost_per_transition=gittins_cost_tensor,
                n_gittins_grid_points=args.gittins_grid_points,
                prior_mean=gittins_prior_mean,
                prior_variance=gittins_prior_variance,
                use_batch_mean_gittins_dp=not args.gittins_per_cell_dp,
                allow_early_stop=False,
                natural_stop_cum_eval_holder=gittins_natural_stop_holder,
            ),
            step_kwargs={},
            log_prefix="gittins",
            pass_sim_cum_eval=True,
            per_arm_original_cost=per_arm_cost_for_simulate,
            incumbent_fn=lambda obs: incumbent_from_gittins_posterior(
                obs,
                prior_mean=gittins_prior_mean,
                prior_variance=gittins_prior_variance,
                tau_sq=tau_sq_gittins,
            ),
            **sim_kwargs,
        )
        gittins_stop_cum_eval = gittins_natural_stop_holder[0]
        xs_gittins_original = list(timing_gittins.get("cum_original_cost", []))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plot_algorithms = ["ucb", "lrf", "gittins"] if merge_ucb_lrf is not None else algorithms
    xs_lrf_plot, regrets_lrf_plot = _trim_trace_from_cum_eval(
        xs_lrf, regrets_lrf, warmup_evals_lrf_trim
    )
    # Returns (trimmed cum eval, trimmed ys); ys must be cumulative cost for cost-axis LRF plots.
    _, xs_lrf_plot_original = _trim_trace_from_cum_eval(
        xs_lrf, xs_lrf_original, warmup_evals_lrf_trim
    )

    batch_desc_parts: list[str] = []
    if "rr" in plot_algorithms or "ucb" in plot_algorithms or "lrf" in plot_algorithms:
        batch_desc_parts.append(f"UCB/LRF batch={args.batch_size}")
    if "gittins" in plot_algorithms:
        batch_desc_parts.append(f"Gittins batch={args.gittins_batch_size}")
    batch_desc = ", ".join(batch_desc_parts) if batch_desc_parts else f"batch={args.batch_size}"
    if (
        args.gittins_cost_mode == "aware"
        and total_full_matrix_cost is not None
        and budget_max_cumulative_cost is not None
    ):
        budget_str = (
            f"budget={args.eval_budget_fraction:.0%} of full-matrix cost "
            f"(cap {budget_max_cumulative_cost:.4g} / {total_full_matrix_cost:.4g} USD per 1M in-tokens)"
        )
    else:
        budget_str = f"budget={args.eval_budget_fraction:.0%} of {n_cells} cells (eval cap)"
    sub = f"seed={args.seed}, {batch_desc}, {budget_str}, gittins_cost_mode={args.gittins_cost_mode}"
    if "lrf" in plot_algorithms:
        sub += (
            f"\n(LRF: {args.warmup_percentage:.0%} random warm-up, then low-rank UCB; "
            f"curve starts at ~{warmup_evals_lrf_trim} evals)"
        )
    if merge_ucb_lrf is not None:
        sub += f"\n(UCB/LRF merged from {merge_ucb_lrf.name})"
    sub = f"algorithms={','.join(plot_algorithms)} | " + sub

    gittins_dp_part = "per-cell DP" if args.gittins_per_cell_dp else "batch-mean DP"
    gittins_cost_aware_plot = (
        args.gittins_cost_mode == "aware" and "gittins" in plot_algorithms
    )
    plot_eval_curves = any(a in plot_algorithms for a in ("rr", "ucb", "lrf"))
    single_panel_cost = (
        gittins_cost_aware_plot
        and plot_eval_curves
        and _cost_series_aligned(xs_gittins, xs_gittins_original, regrets_gittins)
        and _cost_aligned_if_curve(
            want_curve="rr" in plot_algorithms,
            xs_eval=xs_rr,
            xs_cost=xs_rr_original,
            regrets=regrets_rr,
        )
        and _cost_aligned_if_curve(
            want_curve="ucb" in plot_algorithms,
            xs_eval=xs_ucbe,
            xs_cost=xs_ucbe_original,
            regrets=regrets_ucbe,
        )
        and _cost_aligned_if_curve(
            want_curve="lrf" in plot_algorithms,
            xs_eval=xs_lrf_plot,
            xs_cost=xs_lrf_plot_original,
            regrets=regrets_lrf_plot,
        )
    )
    two_panel = gittins_cost_aware_plot and plot_eval_curves and not single_panel_cost
    if merge_ucb_lrf is not None and two_panel and gittins_cost_aware_plot:
        print(
            "Note: --merge-ucb-lrf-from bundle lacks aligned UCB/LRF cumulative cost series "
            "(ucb_x_original_cost / lrf_x_original_cost); using two panels. Regenerate the source "
            "traces with the current script for a single cost-axis figure.",
            file=sys.stderr,
        )
    if gittins_cost_aware_plot:
        if single_panel_cost:
            gittins_label = f"Gittins (τ² = 1/(4B), {gittins_dp_part}, cost-aware)"
        else:
            gittins_label = (
                f"Gittins (τ² = 1/(4B), {gittins_dp_part}, cost-aware, x = cum. cost)"
            )
    else:
        gittins_label = f"Gittins (τ² = 1/(4B), {gittins_dp_part})"

    if single_panel_cost:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if "rr" in plot_algorithms and regrets_rr:
            ax.plot(xs_rr_original, regrets_rr, label="Round-robin (sample mean)", linewidth=1.5)
        if "ucb" in plot_algorithms and regrets_ucbe:
            ax.plot(xs_ucbe_original, regrets_ucbe, label="UCB-E", linewidth=1.5)
        if "lrf" in plot_algorithms and regrets_lrf_plot:
            ax.plot(xs_lrf_plot_original, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
        if "gittins" in plot_algorithms:
            (line_g,) = ax.plot(
                xs_gittins_original, regrets_gittins, label=gittins_label, linewidth=1.5
            )
            stop_cost = _cumulative_cost_at_eval_stop(
                xs_gittins, xs_gittins_original, gittins_stop_cum_eval
            )
            if gittins_stop_cum_eval is not None and stop_cost is not None:
                ax.axvline(
                    stop_cost,
                    color=line_g.get_color(),
                    linestyle="--",
                    alpha=0.85,
                    linewidth=1.2,
                    label=f"Gittins nominal stop ({gittins_stop_cum_eval} evals)",
                )
        ax.set_xlabel(_GITTINS_COST_AWARE_CUMULATIVE_XLABEL)
        ax.set_ylabel("Simple regret")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle(f"Simple regret — {args.matrix.name}\n{sub}", fontsize=10, y=1.0)
        plt.savefig(args.out, dpi=150)
        plt.close()
    elif two_panel:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 9), sharey=True)
        if "rr" in plot_algorithms:
            ax0.plot(xs_rr, regrets_rr, label="Round-robin (sample mean)", linewidth=1.5)
        if "ucb" in plot_algorithms:
            ax0.plot(xs_ucbe, regrets_ucbe, label="UCB-E", linewidth=1.5)
        if "lrf" in plot_algorithms:
            ax0.plot(xs_lrf_plot, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
        ax0.set_xlabel("Cumulative examples evaluated (matrix entries revealed)")
        ax0.set_ylabel("Simple regret")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best")
        if "gittins" in plot_algorithms:
            (line_g,) = ax1.plot(
                xs_gittins_original, regrets_gittins, label=gittins_label, linewidth=1.5
            )
            stop_cost = _cumulative_cost_at_eval_stop(
                xs_gittins, xs_gittins_original, gittins_stop_cum_eval
            )
            if gittins_stop_cum_eval is not None and stop_cost is not None:
                ax1.axvline(
                    stop_cost,
                    color=line_g.get_color(),
                    linestyle="--",
                    alpha=0.85,
                    linewidth=1.2,
                    label=f"Gittins nominal stop ({gittins_stop_cum_eval} evals)",
                )
        ax1.set_xlabel(_GITTINS_COST_AWARE_CUMULATIVE_XLABEL)
        ax1.set_ylabel("Simple regret")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle(f"Simple regret — {args.matrix.name}\n{sub}", fontsize=10, y=1.0)
        plt.savefig(args.out, dpi=150)
        plt.close()
    elif gittins_cost_aware_plot:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        (line_g,) = ax.plot(
            xs_gittins_original, regrets_gittins, label=gittins_label, linewidth=1.5
        )
        stop_cost = _cumulative_cost_at_eval_stop(
            xs_gittins, xs_gittins_original, gittins_stop_cum_eval
        )
        if gittins_stop_cum_eval is not None and stop_cost is not None:
            ax.axvline(
                stop_cost,
                color=line_g.get_color(),
                linestyle="--",
                alpha=0.85,
                linewidth=1.2,
                label=f"Gittins nominal stop ({gittins_stop_cum_eval} evals)",
            )
        ax.set_xlabel(_GITTINS_COST_AWARE_CUMULATIVE_XLABEL)
        ax.set_ylabel("Simple regret")
        plt.title(f"Simple regret — {args.matrix.name}\n{sub}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        plt.close()
    else:
        plt.figure(figsize=(8, 5))
        if "rr" in plot_algorithms:
            plt.plot(xs_rr, regrets_rr, label="Round-robin (sample mean)", linewidth=1.5)
        if "ucb" in plot_algorithms:
            plt.plot(xs_ucbe, regrets_ucbe, label="UCB-E", linewidth=1.5)
        if "lrf" in plot_algorithms:
            plt.plot(xs_lrf_plot, regrets_lrf_plot, label="UCB-E-LRF", linewidth=1.5)
        if "gittins" in plot_algorithms:
            (line_gittins,) = plt.plot(
                xs_gittins, regrets_gittins, label=gittins_label, linewidth=1.5
            )
            if gittins_stop_cum_eval is not None:
                plt.axvline(
                    gittins_stop_cum_eval,
                    color=line_gittins.get_color(),
                    linestyle="--",
                    alpha=0.85,
                    linewidth=1.2,
                    label=f"Gittins nominal stop ({gittins_stop_cum_eval} evals)",
                )
        plt.xlabel("Cumulative examples evaluated (matrix entries revealed)")
        plt.ylabel("Simple regret")
        plt.title(f"Simple regret — {args.matrix.name}\n{sub}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        plt.close()

    if not args.no_save_traces:
        traces_path = args.traces_out
        if traces_path is None:
            traces_path = args.out.with_name(f"{args.out.stem}_traces.npz")
        timing_meta = {
            "rr": _timing_for_meta(
                timing_rr, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "ucb": _timing_for_meta(
                timing_ucb, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "lrf": _timing_for_meta(
                timing_lrf, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "gittins": _timing_for_meta(
                timing_gittins, include_per_iter_series=args.timing_include_per_iter_series
            ),
            "per_iter_series_included": bool(args.timing_include_per_iter_series),
        }
        save_trace_bundle(
            traces_path,
            xs_rr=xs_rr,
            regrets_rr=regrets_rr,
            xs_ucbe=xs_ucbe,
            regrets_ucbe=regrets_ucbe,
            xs_lrf=xs_lrf,
            regrets_lrf=regrets_lrf,
            xs_lrf_plot=xs_lrf_plot,
            regrets_lrf_plot=regrets_lrf_plot,
            xs_rr_original_cost=xs_rr_original if xs_rr_original else None,
            xs_ucbe_original_cost=xs_ucbe_original if xs_ucbe_original else None,
            xs_lrf_original_cost=xs_lrf_original if xs_lrf_original else None,
            xs_lrf_plot_original_cost=xs_lrf_plot_original if xs_lrf_plot_original else None,
            xs_gittins=xs_gittins,
            regrets_gittins=regrets_gittins,
            xs_gittins_original_cost=xs_gittins_original,
            gittins_stop_cum_eval=gittins_stop_cum_eval,
            warmup_evals=warmup_evals,
            budget_evals=budget_evals,
            tau_sq_gittins=tau_sq_gittins,
            meta={
                "matrix": str(args.matrix.resolve()),
                "figure": str(args.out.resolve()),
                "traces": str(traces_path.resolve()),
                "seed": args.seed,
                "batch_size": args.batch_size,
                "gittins_batch_size": args.gittins_batch_size,
                "eval_budget_fraction": args.eval_budget_fraction,
                "ucb_a": args.a,
                "warmup_percentage": args.warmup_percentage,
                "lrf_device": args.lrf_device,
                "gittins_grid_points": args.gittins_grid_points,
                "gittins_cost": gittins_cost_to_meta_string(gittins_cost_tensor),
                "gittins_cost_mode": args.gittins_cost_mode,
                "gittins_cost_scale": 1e-4,
                "gittins_cost_vector_file": str(cost_vector_path.resolve())
                if cost_vector_path and args.gittins_cost_mode == "aware"
                else None,
                "experiment": args.experiment,
                "gittins_prior_mean": gittins_prior_mean,
                "gittins_prior_variance": gittins_prior_variance,
                **(
                    {"gittins_simple_regret_incumbent": "posterior_mean"}
                    if "gittins" in algorithms
                    else {}
                ),
                "gittins_per_cell_dp": args.gittins_per_cell_dp,
                "n_cells": n_cells,
                "budget_stops_by": "cumulative_cost"
                if args.gittins_cost_mode == "aware"
                else "evaluations",
                "total_full_matrix_cost": total_full_matrix_cost,
                "budget_max_cumulative_cost": budget_max_cumulative_cost,
                "algorithms": plot_algorithms,
                "title": f"Simple regret — {args.matrix.name}\n{sub}",
                "gittins_stop_cum_eval": gittins_stop_cum_eval,
                "merge_ucb_lrf_from": str(merge_ucb_lrf.resolve()) if merge_ucb_lrf else None,
                "warmup_evals_lrf_plot_trim": warmup_evals_lrf_trim,
                "timing": timing_meta,
            },
        )
        print(f"Wrote traces {traces_path} and {traces_path.with_suffix('.meta.json')}")

    print(f"Wrote {args.out}")
    cost_aware_budget = (
        args.gittins_cost_mode == "aware" and budget_max_cumulative_cost is not None
    )
    if "rr" in plot_algorithms:
        if cost_aware_budget and "rr" in algorithms:
            print(
                f"Round-robin: {len(regrets_rr)} batches, {xs_rr[-1] if xs_rr else 0} evals, "
                f"cum cost {(_final_cum_cost(timing_rr) or 0.0):.6g} / {budget_max_cumulative_cost:.6g} "
                f"(B = {args.batch_size})"
            )
        else:
            print(
                f"Round-robin: {len(regrets_rr)} batches, {xs_rr[-1] if xs_rr else 0} / {budget_evals} budget evals "
                f"(B = {args.batch_size})"
            )
        if "rr" in algorithms:
            s_total = _timing_summary(timing_rr["iter_total_s"])
            s_step = _timing_summary(timing_rr["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    if "ucb" in plot_algorithms:
        src = " (merged)" if merge_ucb_lrf and "ucb" not in algorithms else ""
        if cost_aware_budget and "ucb" in algorithms:
            print(
                f"UCB-E{src}: {len(regrets_ucbe)} batches, {xs_ucbe[-1] if xs_ucbe else 0} evals, "
                f"cum cost {(_final_cum_cost(timing_ucb) or 0.0):.6g} / {budget_max_cumulative_cost:.6g}"
            )
        else:
            print(
                f"UCB-E{src}: {len(regrets_ucbe)} batches, {xs_ucbe[-1] if xs_ucbe else 0} / {budget_evals} budget evals"
            )
        if "ucb" in algorithms:
            s_total = _timing_summary(timing_ucb["iter_total_s"])
            s_step = _timing_summary(timing_ucb["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    if "lrf" in plot_algorithms:
        src = " (merged)" if merge_ucb_lrf and "lrf" not in algorithms else ""
        if cost_aware_budget and "lrf" in algorithms:
            print(
                f"UCB-E-LRF{src}: {len(regrets_lrf)} batches, {xs_lrf[-1] if xs_lrf else 0} evals, "
                f"cum cost {(_final_cum_cost(timing_lrf) or 0.0):.6g} / {budget_max_cumulative_cost:.6g} "
                f"({len(regrets_lrf_plot)} plotted points from cum_eval ≥ {warmup_evals_lrf_trim})"
            )
        else:
            print(
                f"UCB-E-LRF{src}: {len(regrets_lrf)} batches, {xs_lrf[-1] if xs_lrf else 0} / {budget_evals} budget evals "
                f"({len(regrets_lrf_plot)} plotted points from cum_eval ≥ {warmup_evals_lrf_trim})"
            )
        if "lrf" in algorithms:
            s_total = _timing_summary(timing_lrf["iter_total_s"])
            s_step = _timing_summary(timing_lrf["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    if "gittins" in plot_algorithms:
        stop_msg = (
            f", nominal stop marker at cum_eval={gittins_stop_cum_eval} (curve to budget)"
            if gittins_stop_cum_eval is not None
            else ""
        )
        if cost_aware_budget and "gittins" in algorithms:
            print(
                f"Gittins: {len(regrets_gittins)} batches, {xs_gittins[-1] if xs_gittins else 0} evals, "
                f"cum cost {(_final_cum_cost(timing_gittins) or 0.0):.6g} / {budget_max_cumulative_cost:.6g} "
                f"(B = {args.gittins_batch_size}, τ² = 1/(4B) = {tau_sq_gittins}){stop_msg}"
            )
        else:
            print(
                f"Gittins: {len(regrets_gittins)} batches, {xs_gittins[-1] if xs_gittins else 0} / {budget_evals} budget evals "
                f"(B = {args.gittins_batch_size}, τ² = 1/(4B) = {tau_sq_gittins}){stop_msg}"
            )
        if "gittins" in algorithms:
            s_total = _timing_summary(timing_gittins["iter_total_s"])
            s_step = _timing_summary(timing_gittins["iter_step_s"])
            print(
                "  timing (per-iteration): "
                f"total mean={s_total.get('mean_s', float('nan')):.6f}s median={s_total.get('median_s', float('nan')):.6f}s p90={s_total.get('p90_s', float('nan')):.6f}s "
                f"(n={s_total.get('n', 0)})"
            )
            print(
                "  timing (policy step only): "
                f"mean={s_step.get('mean_s', float('nan')):.6f}s median={s_step.get('median_s', float('nan')):.6f}s p90={s_step.get('p90_s', float('nan')):.6f}s "
                f"(n={s_step.get('n', 0)})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
