# BanditGittinsEval

Exploration policies on a fixed accuracy matrix (rows = models, columns = i.i.d. samples), with simple-regret curves and optional per-arm rollout plots.

## Setup

From the repository root, install the project (editable install exposes modules under `src/`):

```bash
pip install -e .
# or: uv sync && uv run python ...
```

Dependencies include `banditeval`, `torch`, `matplotlib`, and `jax` (see `pyproject.toml`).

## Simple regret (recommended workflow): simulate + plot

For most experiments, it’s easier to **separate simulation from plotting**:

- `scripts/simulate_simple_regret.py`: runs exploration and writes a `.npz` bundle (no plotting).
- `scripts/plot_simple_regret_results.py`: loads that `.npz` and generates the figure.

Only **two algorithms** are supported here for now:

- **UCB-E**: arm **selection** uses the UCB bound; arm **recommendation** uses the row **empirical mean**.
- **Gittins**: arm **selection** uses the Gittins **index**; arm **recommendation** uses the **posterior mean** \(E[\theta_k \mid D_t]\) under the normal–normal model.

### Example: GSM8K matrix (`gsm8k_1_samples_various_models_seed1.npy`)

Simulate and save traces:

```bash
python scripts/simulate_simple_regret.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --out outputs/traces/gsm8k_various_models_seed1_ucb_gittins.npz \
  --seed 0 \
  --eval-budget-fraction 0.10 \
  --batch-size 32 \
  --gittins-batch-size 32
```

Notes:

- The UCB-E and Gittins batch sizes are separate flags; for a fair comparison, set them equal (as above).
- The simulation always tracks a cumulative-cost x-axis using a **per-arm cost vector**:
  - omit `--cost-vector` to use a homogeneous cost vector of all ones (so cumulative cost equals cumulative evaluations), or
  - pass `--cost-vector <.json/.npy>` for heterogeneous costs (e.g. pricing).
- By default, Gittins uses \(\tau^2 = 1/(4B)\) with \(B=\) `--gittins-batch-size`. Override with `--gittins-obs-noise-variance`.
- To run only one algorithm, use `--algorithms ucb` or `--algorithms gittins`.

Plot from the saved traces:

```bash
python scripts/plot_simple_regret_results.py \
  --traces outputs/traces/gsm8k_various_models_seed1_ucb_gittins.npz \
  --out outputs/figures/gsm8k_various_models_seed1_ucb_gittins.png \
  --x-axis evals
```

Plot versus cumulative monetary cost (when you used a pricing cost vector):

```bash
python scripts/plot_simple_regret_results.py \
  --traces outputs/traces/gsm8k_various_models_seed1_ucb_gittins.npz \
  --out outputs/figures/gsm8k_various_models_seed1_ucb_gittins_cost.png \
  --x-axis original_cost
```

## Simple regret simulation and figure: `plot_simple_regret.py`

Simulates one or more algorithms on a masked matrix, reveals entries in batches, and plots **simple regret** \(\mu^* - \mu_{\hat{a}_t}\) versus a cumulative **budget** on the x-axis: **matrix entries evaluated** when `--gittins-cost-mode unaware` (and for UCB/LRF in that mode), or **cumulative monetary cost** (same units as the pricing JSON—e.g. USD per 1M input tokens for the bundled GSM8K configurations file) when cost-aware. Writes a PNG and, by default, a compressed trace bundle for later replotting.

The script is **benchmark-agnostic** — it consumes any `(n_models, n_examples)` accuracy matrix `.npy`. Pre-registered presets (`--experiment NAME`) cover bundled benchmarks (`gsm8k_various_model`, `piqa_various_models`); pass `--matrix` / `--out` to point at any other matrix without touching `experiment_specs()`.

**Basic run** (default `--experiment gsm8k_various_model` sets matrix and figure paths; see `experiment_specs()` in the script):

```bash
python scripts/plot_simple_regret.py
```

**Common overrides:**

| Argument | Role |
|----------|------|
| `--experiment NAME` | Preset (`experiment_specs` in the script): default `--matrix` and `--out` (default `gsm8k_various_model`) |
| `--matrix PATH` | `(n_models, n_examples)` accuracy matrix `.npy` (overrides experiment default) |
| `--out PATH` | Output figure (PNG) (overrides experiment default) |
| `--seed N` | RNG seed for exploration |
| `--eval-budget-fraction F` | **Unaware:** stop after `F × (rows × cols)` evaluations. **Aware:** stop after cumulative spend reaches `F ×` (total cost to evaluate every cell: `n_examples × sum_k c_k` in pricing units) (default `0.1`) |
| `--algorithms ucb lrf gittins` | Subset of algorithms to run (default: all three). Example: `--algorithms gittins` for a quick test |
| `--batch-size B` | UCB-E and UCB-E-LRF: cells per batch (default `32`) |
| `--gittins-batch-size B` | Gittins: batch size and \(B\) in \(\tau^2 = 1/(4B)\) (default `20`) |
| `--warmup-percentage W` | UCB-E-LRF: fraction of matrix observed with **uniform random** probing before low-rank UCB (default `0.05`). The plotted LRF curve uses only the post–warm-up segment so it aligns with where that policy actually runs |
| `--ucb-a A` | UCB exploration parameter (default `1`) |
| `--lrf-device DEVICE` | e.g. `cpu` or `cuda` for the low-rank factorization step |
| `--gittins-grid-points N` | Tabular DP grid size for Gittins (default `1025`) |
| `--gittins-cost-mode unaware \| aware` | Same idea as `gittins_policy.cost_per_transition`: `unaware` uses **1.0** per arm (uniform cost); `aware` loads per-arm monetary costs. The bundled GSM8K configurations pricing JSON uses **USD per 1M input tokens**; other pricing files follow whatever unit they declare. When `aware`, the figure uses that unit on the x-axis. If UCB/LRF/RR run **alongside** cost-aware Gittins, the default is **one panel**: every curve is plotted vs cumulative cost (the trace bundle stores per-method cumulative cost; see **Trace files**). **Legacy** trace bundles without `ucb_x_original_cost` / `lrf_x_original_cost`, or `--merge-ucb-lrf-from` pointing at such a file, fall back to **two panels** (evals for UCB/LRF vs cost for Gittins). |
| `--gittins-cost C` | Ignored for cost-unaware Gittins (DP uses `1.0`); kept for compatibility |
| `--gittins-cost-vector FILE` | Required for `aware`: per-arm costs (JSON or `.npy`; bundled GSM8K configurations file: USD per 1M input tokens). Ignored when `unaware` |
| `--gittins-prior-mean`, `--gittins-prior-variance` | Prior \(\theta_k \sim \mathcal{N}(\mu_0, v_0)\) for Gittins (optional; see **Gittins prior** below) |
| `--gittins-per-cell-dp` | Use one DP stage per matrix cell (slow); default is **batch-mean** DP aligned with `--gittins-batch-size` |
| `--traces-out PATH` | Where to save `*_traces.npz` (default: same directory as `--out`, stem + `_traces.npz`) |
| `--no-save-traces` | Skip writing trace `.npz` and `.meta.json` |
| `--verbose` | Print each batch: distinct arms, incumbent, simple regret |
| `--merge-ucb-lrf-from PATH` | Reuse UCB-E and UCB-E-LRF traces from an earlier `*_traces.npz` and simulate only Gittins (`--algorithms gittins` required). For **cost-aware** runs, use a merge source produced by the current script so it includes `ucb_x_original_cost` and `lrf_x_original_cost`; otherwise the figure falls back to two panels. |

### Gittins prior (\(\theta_k \sim \mathcal{N}(\mu_0, v_0)\))

The script resolves \((\mu_0, v_0)\) in this order:

1. **CLI** — if you pass `--gittins-prior-mean` and/or `--gittins-prior-variance`, those values override everything else (omit a flag to leave that component to the next steps).
2. **Experiment preset** — `experiment_specs()` in `plot_simple_regret.py` can set optional `gittins_prior_mean` / `gittins_prior_variance` per `--experiment` name.
3. **Global default** — **\(\mu_0 = 0.5\)**, **\(v_0 = 0.04\)** (i.e. **N(0.5, 0.04)**) when the preset does not fix them.

The **`gsm8k_various_model`** preset uses **N(0.2, 0.01)** so GSM8K runs match the intended prior without extra flags. Add other experiments by extending `ExperimentSpec` the same way.

The resolved pair is stored in `*_traces.meta.json` as `gittins_prior_mean` and `gittins_prior_variance`.

**Example: small budget, Gittins only (cost-unaware, default):**

```bash
python scripts/plot_simple_regret.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --out outputs/figures/simple_regret_gittins_only.png \
  --algorithms gittins \
  --eval-budget-fraction 0.02 \
  --seed 0
```

**Example: cost-aware Gittins** (per-arm costs for the DP in USD per 1M input tokens; x-axis = cumulative cost in that unit). The pricing JSON below covers **all GSM8K matrices that share the same `various_models` arm configuration** — its name no longer mentions a specific samples count or seed because those don't change which arm is which:

```bash
python scripts/plot_simple_regret.py \
  --gittins-cost-mode aware \
  --gittins-cost-vector data_analysis/pricing/gsm8k_various_models_configurations_price_ratio_1to2_rounded.json \
  --algorithms gittins \
  --eval-budget-fraction 0.02
```

### Trace files

By default, next to the figure you get:

- `<stem>_traces.npz` — arrays such as `ucb_x`, `ucb_regret`, `lrf_x_full`, `lrf_regret_full`, `lrf_x_plot`, `lrf_regret_plot`, `gittins_x`, `gittins_regret`, plus scalars `warmup_evals`, `budget_evals`, `tau_sq_gittins`. **Cost-aware runs** also store cumulative monetary cost after each batch, aligned with the corresponding regret series: `gittins_x_original_cost`, `ucb_x_original_cost`, `lrf_x_original_cost`, `lrf_x_plot_original_cost` (post–warm-up LRF segment), and `rr_x_original_cost` if round-robin ran. These arrays are what allow a **single** cost-axis figure when multiple policies are plotted together.
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
