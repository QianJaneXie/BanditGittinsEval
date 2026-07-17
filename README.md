# BanditGittinsEval

Exploration policies on a fixed accuracy matrix (rows = models, columns = i.i.d. samples), with simple-regret curves and optional per-arm rollout plots.

Works with any `(n_models, n_examples)` accuracy matrix under `data/`. Full experiment sweeps (many GSM8K/PIQA/MMLU matrices, seeds, and hyperparameters) are launched via `run_simple_regret_wandb.py` and YAML configs in `scripts/config/`.

## Setup

```bash
pip install -e .
# or: uv sync && uv run python ...
```

See `pyproject.toml` for dependencies (`banditeval`, `torch`, `matplotlib`, `jax`, …).

## Workflow

Simulate bandit exploration, then plot — keep these steps separate:

```bash
python scripts/simulate_simple_regret.py --matrix <matrix.npy> --out <traces.npz> [options]
python scripts/plot_simple_regret_results.py --traces <traces.npz> --out <figure.png> [--x-axis evals|original_cost]
```

**Algorithms** compared in the W&B sweeps:

| Policy | Selection | Recommendation |
|--------|-----------|----------------|
| UCB-E | UCB bound | Empirical mean |
| SySRs | Synchronized successive rejects ([llm-bandits-sysrs](https://github.com/zifanlyu/llm-bandits-sysrs) Smart-SR) | Empirical mean among active arms |
| UCB-E-LRF | Low-rank UCB (after uniform warm-up) | Empirical mean |
| Gittins | Gittins index | Posterior mean \(E[\theta_k \mid D_t]\) |

`simulate_simple_regret.py` runs UCB-E, SySRs, and Gittins (`--algorithms ucb sysrs gittins`). UCB-E-LRF is run via `run_simple_regret_wandb.py` (`experiment_variant` such as `lrf_B32` / `lrf_cost_B32`; see `scripts/config/*_lrf.yml`). SySRs is hyperparameter-free (`sysrs` / `sysrs_cost`; schedule budget = `--eval-budget-fraction`).

**Common flags** — run any script with `--help` for the full list:

- `--eval-budget-fraction` — fraction of matrix cells (unit cost) or total full-evaluation cost (with `--cost-vector`).
- `--cost-vector <.json/.npy>` — heterogeneous per-arm costs; omit for unit cost (1 per evaluation).
- `--gittins-prior-mean` / `--gittins-prior-variance` — Gittins prior; MMLU bucket priors in [`docs/mmlu_prior_buckets.md`](docs/mmlu_prior_buckets.md).
- `--gittins-obs-noise-variance` — override default \(\tau^2 = 1/(4B)\) where \(B\) is `--gittins-batch-size`.

## Examples

Two representative benchmarks below. Swap paths for any other matrix in `data/`.

### GSM8K (various models)

Batch size 16; Gittins prior \(\mathcal{N}(0.2, 0.01)\).

Bandit (unit cost):

```bash
python scripts/simulate_simple_regret.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --out outputs/bandit_traces/gsm8k_seed1.npz \
  --seed 0 --eval-budget-fraction 0.10 \
  --batch-size 16 --gittins-batch-size 16 \
  --gittins-prior-mean 0.2 --gittins-prior-variance 0.01

python scripts/plot_simple_regret_results.py \
  --traces outputs/bandit_traces/gsm8k_seed1.npz \
  --out outputs/figures/gsm8k_seed1.png --x-axis evals
```

Cost-aware bandit: add `--cost-vector data_analysis/pricing/gsm8k_various_models_configurations_price_ratio_1to2_rounded.json` to simulate, then plot with `--x-axis original_cost`.

BayesOpt (unit cost):

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_bo_baseline.py \
  --bo-inputs data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz \
  --acquisition logei --seed 0 --out-dir outputs/bo_baselines
```

Cost-aware BayesOpt: use `--acquisition logeipc` (costs are in the `.npz`).

### MMLU (abstract algebra)

Batch size 4; Gittins prior \(\mathcal{N}(0.4, 0.02)\) (low bucket — see prior doc for other subjects).

Bandit (unit cost):

```bash
python scripts/simulate_simple_regret.py \
  --matrix data/MMLU_matrices/abstract_algebra.npy \
  --out outputs/bandit_traces/mmlu_abstract_algebra.npz \
  --seed 0 --eval-budget-fraction 0.10 \
  --batch-size 4 --gittins-batch-size 4 \
  --gittins-prior-mean 0.4 --gittins-prior-variance 0.02

python scripts/plot_simple_regret_results.py \
  --traces outputs/bandit_traces/mmlu_abstract_algebra.npz \
  --out outputs/figures/mmlu_abstract_algebra.png --x-axis evals
```

Cost-aware bandit: add `--cost-vector data_analysis/pricing/mmlu_prompt_eval_configurations_input_price.json` to simulate, then plot with `--x-axis original_cost`.

BayesOpt (unit cost):

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_bo_baseline.py \
  --bo-inputs data/bo_inputs/mmlu/abstract_algebra_bo_inputs.npz \
  --acquisition logei --seed 0 --out-dir outputs/bo_baselines
```

Cost-aware BayesOpt: use `--acquisition logeipc`.

Combine bandit and BayesOpt traces with `plot_bandit_bo_comparison.py` (pass `--bandit-trace`, `--pbgi-trace`, `--log-bo-trace`, and `--x-axis evals` or `original_cost`).

## Scripts

| Script | Purpose |
|--------|---------|
| `simulate_simple_regret.py` | Run UCB-E / Gittins; save trace `.npz` |
| `plot_simple_regret_results.py` | Plot regret from a trace bundle |
| `plot_arm_eval_rollouts.py` | Per-arm batch-pull rollouts for UCB-E / SySRs / Gittins (re-simulates from traces) |
| `run_bo_baseline.py` | BayesOpt baselines on `data/bo_inputs/*.npz` |
| `plot_bandit_bo_comparison.py` | Overlay bandit + BO curves |
| `convert_matrix_to_bo_inputs.py` | Build BO input files from a matrix |
| `run_simple_regret_wandb.py` | W&B sweep launcher (UCB-E, UCB-E-LRF, Gittins) |
| `run_bo_baseline_wandb.py` | W&B sweep launcher (BayesOpt baselines) |
| `download_wandb_simple_regret.py` | Pull bandit W&B sweep results |
| `download_wandb_bo_baseline.py` | Pull BayesOpt W&B sweep results |
| `plot_wandb_simple_regret_curves.py` | Plot downloaded W&B curves (merge bandit + BO histories) |

## BayesOpt baseline

BayesOpt runs on configuration-level inputs (`data/bo_inputs/*.npz`), not matrix cells. Each row is a full configuration (arm); evaluating one candidate reveals its aggregate score and consumes the stored cost.

**Budget.** Defaults to `eval_budget_fraction` (10% in the example sweeps) of configurations (unit cost) or total cost (cost-aware). Remaining budget after initialization goes to acquisition-driven BO steps.

**Random initialization.** Unless `--n-init` is set, the number of initial arms is

\[
n_{\mathrm{init}} = \min\bigl(d,\; \mathrm{round}(0.5 \times \texttt{eval\_budget\_fraction} \times n_{\mathrm{configs}})\bigr),
\]

where \(d\) is the **dominant dimension**: unique `model_id` levels for GSM8K/PIQA, unique `prompt_idx` levels for MMLU. With the default `eval_budget_fraction = 0.1`, the budget cap is **5%** of \(n_{\mathrm{configs}}\). Initial arms are sampled uniformly at random without replacement (not necessarily one per dominant level).

At the default 10% budget:

| Dataset | \(d\) | \(n_{\mathrm{configs}}\) | Default \(n_{\mathrm{init}}\) | BO steps after init |
|---------|------|---------------------------|-------------------------------|---------------------|
| GSM8K | 11 models | 122 | 6 | 6 |
| PIQA | 11 models | 103 | 5 | 5 |
| MMLU | 100 prompts | 1500 | 75 | 75 |

Override with `--n-init`. See **Examples** above for GSM8K and MMLU commands.

### W&B download and aggregate plots (bandit + BO)

Use separate download pipelines, then merge when plotting:

```bash
# Bandit sweep
python scripts/download_wandb_simple_regret.py \
  --entity <entity> --project GittinsBanditEval --sweep-id <bandit_sweep> \
  --dataset gsm8k --raw-dir outputs/wandb_downloads/gsm8k_bandit

# BO sweep
python scripts/download_wandb_bo_baseline.py \
  --entity <entity> --project GittinsBanditEval --sweep-id <bo_sweep> \
  --dataset gsm8k --raw-dir outputs/wandb_downloads/gsm8k_bo \
  --expected-grid-yaml scripts/config/GSM8KBOBaselineSweep.yml

# Combined aggregate plot (unit-cost example)
python scripts/plot_wandb_simple_regret_curves.py \
  --history-csv outputs/wandb_downloads/gsm8k_bandit/runs_history.csv.gz \
  --history-csv outputs/wandb_downloads/gsm8k_bo/runs_history.csv.gz \
  --dataset gsm8k --benchmark-key gsm8k_seed1 \
  --bo-phase post_init --x-axis cum_eval \
  --out-dir outputs/figures/wandb
```

`benchmark_key` links bandit and BO runs (`gsm8k_seed1`, `mmlu_abstract_algebra`, …). Use `--bo-phase post_init` to drop BO random-init steps (matching `plot_bandit_bo_comparison.py`).
