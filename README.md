# BanditGittinsEval

Exploration policies on a fixed accuracy matrix (rows = models, columns = i.i.d. samples), with simple-regret curves and optional per-arm rollout plots.

## Setup

From the repository root, install the project (editable install exposes modules under `src/`):

```bash
pip install -e .
# or: uv sync && uv run python ...
```

Dependencies include `banditeval`, `torch`, `matplotlib`, and `jax` (see `pyproject.toml`).

## Simple regret simulation and figure: `plot_simple_regret_gsm8k.py`

Simulates one or more algorithms on a masked matrix, reveals entries in batches, and plots **simple regret** \(\mu^* - \mu_{\hat{a}_t}\) versus **cumulative matrix entries evaluated**. Writes a PNG and, by default, a compressed trace bundle for later replotting.

**Basic run** (defaults assume a matrix under `data/matrices/`):

```bash
python scripts/plot_simple_regret_gsm8k.py
```

**Common overrides:**

| Argument | Role |
|----------|------|
| `--matrix PATH` | `(n_models, n_examples)` accuracy matrix `.npy` |
| `--out PATH` | Output figure (PNG) |
| `--seed N` | RNG seed for exploration |
| `--eval-budget-fraction F` | Stop after `F × (rows × cols)` evaluations (default `0.1`) |
| `--algorithms ucb lrf gittins` | Subset of algorithms to run (default: all three). Example: `--algorithms gittins` for a quick test |
| `--batch-size B` | UCB-E and UCB-E-LRF: cells per batch (default `32`) |
| `--gittins-batch-size B` | Gittins: batch size and \(B\) in \(\tau^2 = 1/(4B)\) (default `20`) |
| `--warmup-percentage W` | UCB-E-LRF: fraction of matrix observed with **uniform random** probing before low-rank UCB (default `0.05`). The plotted LRF curve uses only the post–warm-up segment so it aligns with where that policy actually runs |
| `--ucb-a A` | UCB exploration parameter (default `1`) |
| `--lrf-device DEVICE` | e.g. `cpu` or `cuda` for the low-rank factorization step |
| `--gittins-grid-points N` | Tabular DP grid size for Gittins (default `1025`) |
| `--gittins-cost C` | Scalar transition cost for Gittins DP (default `1e-4`) |
| `--gittins-cost-per-arm PATH` | Optional `(n_arms,)` vector `.npy`; overrides `--gittins-cost` |
| `--gittins-prior-mean`, `--gittins-prior-variance` | Prior \(\theta_k \sim \mathcal{N}(\mu_0, v_0)\) for Gittins |
| `--gittins-per-cell-dp` | Use one DP stage per matrix cell (slow); default is **batch-mean** DP aligned with `--gittins-batch-size` |
| `--traces-out PATH` | Where to save `*_traces.npz` (default: same directory as `--out`, stem + `_traces.npz`) |
| `--no-save-traces` | Skip writing trace `.npz` and `.meta.json` |
| `--verbose` | Print each batch: distinct arms, incumbent, simple regret |

**Example: small budget, Gittins only:**

```bash
python scripts/plot_simple_regret_gsm8k.py \
  --matrix data/matrices/gsm8k_1_samples_various_models_seed1.npy \
  --out outputs/figures/simple_regret_gittins_only.png \
  --algorithms gittins \
  --eval-budget-fraction 0.02 \
  --seed 0
```

### Trace files

By default, next to the figure you get:

- `<stem>_traces.npz` — arrays such as `ucb_x`, `ucb_regret`, `lrf_x_full`, `lrf_regret_full`, `lrf_x_plot`, `lrf_regret_plot`, `gittins_x`, `gittins_regret`, plus scalars `warmup_evals`, `budget_evals`, `tau_sq_gittins`.
- `<stem>_traces.meta.json` — paths, hyperparameters, and a title string for reproducibility.

**Replot the regret figure without resimulating:**

```bash
python scripts/replot_simple_regret_from_traces.py \
  --traces outputs/figures/simple_regret_gsm8k_various_models_seed1_traces.npz \
  --out outputs/figures/simple_regret_replot.png
```

## Per-arm rollout plot: `plot_arm_eval_rollouts.py`

The trace `.npz` does **not** store per-arm histories. This script reads the **same** `*_traces.meta.json` beside your traces, reloads the matrix, and **re-runs** the policies with the same settings and seed so the rollout matches the original run.

It produces a multi-panel figure: **x-axis** = batch iteration index \(1, \ldots, T\); **y-axis** = **cumulative batch pulls** per arm (each batch counts once per distinct arm that received at least one evaluated cell in that batch).

```bash
python scripts/plot_arm_eval_rollouts.py \
  --traces outputs/figures/simple_regret_gsm8k_various_models_seed1_traces.npz
```

Default output: `*_arm_rollouts_batch_pulls.png` next to the traces (e.g. `simple_regret_gsm8k_various_models_seed1_arm_rollouts_batch_pulls.png`).

| Argument | Role |
|----------|------|
| `--traces PATH` | Path to `*_traces.npz` (requires sibling `*_traces.meta.json`) |
| `--matrix PATH` | Optional: override matrix `.npy` if the path stored in meta is wrong or the file moved |
| `--out PATH` | Output PNG (default: derived from traces stem as above) |

Gittins re-simulation can be expensive for large matrices and budgets, similar to the main plotting script.
