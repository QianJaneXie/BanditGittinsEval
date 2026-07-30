# W&B Logging Spec for New Baselines

This document specifies what a new algorithm must upload to **Weights & Biases** so it can be downloaded and plotted with the existing GittinsBanditEval pipeline.

Reference implementations:
- `scripts/run_simple_regret_wandb.py` (Gittins / UCB / LRF / SySRs-style)
- `scripts/run_bo_baseline_wandb.py` (BO baselines)
- Downloader schema: `scripts/download_wandb_simple_regret.py`

Default project: `GittinsBanditEval`

---

## 1. Run identity (required)

Each W&B run must be uniquely identifiable by config:

| Key | Type | Example | Notes |
|---|---|---|---|
| `dataset_tag` | str | `gsm8k`, `piqa`, `alpaca`, `mmlu` | Required |
| `matrix` | str/path | `data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy` | Required |
| `run_seed` | int | `0` … `19` | Required; independent trial seed |
| `experiment_variant` | str | `ucb_B8`, `sysrs`, `pbgi_unit`, `myalgo_unit` | Required; algorithm id used for filtering |

Recommended naming:
```text
wandb name:  {dataset_tag}_{matrix_or_task}_{experiment_variant}_runseed{run_seed}
wandb group: {dataset_tag}_simple_regret_sweep   # or a dedicated group for the new algo
```

---

## 2. Config fields

### 2.1 Required for all algorithms

| Key | Type | Meaning |
|---|---|---|
| `dataset_tag` | str | Dataset family |
| `matrix` | str | Path / id of reward matrix |
| `run_seed` | int | Trial seed |
| `experiment_variant` | str | Algorithm variant name |
| `eval_budget_fraction` | float | Usually `0.1` (10% of exhaustive evaluation) |
| `cost_mode` | str | `unit` / `cost` (or equivalent; must distinguish unit-cost vs cost-aware) |
| `n_arms` | int | Number of arms / configs |
| `n_examples` | int | Number of evaluation instances (questions) |
| `n_cells` | int | `n_arms * n_examples` |
| `budget_max_evals` | int | Max evaluations under unit-cost budget |
| `cost_aware_run` | bool | Whether this run is cost-aware |

### 2.2 Strongly recommended

| Key | Type | Meaning |
|---|---|---|
| `policy_variant` | str | Often same as / derived from `experiment_variant` |
| `policy_family` | str | e.g. `ucb`, `gittins`, `lrf`, `bo`, `sysrs`, `newalgo` |
| `batch_size` | int | Pull batch size if applicable |
| `cost_scaling_factor` | float | Cost scaling used in simulation |
| `cost_vector` | str/path | Path to per-arm cost file (needed for cost-aware) |
| `warmup_percentage` | float | e.g. LRF warmup `0.05` |
| `matrix_seed` | int/str | Matrix replicate id (GSM8K/PIQA seeds) |
| `total_brute_force_original_cost` | float | Full exhaustive original cost |
| `budget_original_cost` | float | Cost budget = `eval_budget_fraction * total_brute_force_original_cost` |

### 2.3 MMLU-only

| Key | Type | Meaning |
|---|---|---|
| `mmlu_task` | str | e.g. `computer_security` |
| `mmlu_size_bucket` | str | `small` / `medium` / `large` |

### 2.4 Algorithm-specific (only if applicable)

| Key | Used by | Meaning |
|---|---|---|
| `prior_type` / `prior_mean_resolved` / `prior_variance_resolved` | Gittins | Resolved prior |
| `prior_bucket` / `prior_source` | Gittins/MMLU | Prior metadata |
| `gittins_batch_size` | Gittins | Batch size |
| `ucb_a` | UCB-E | Exploration parameter |
| `acquisition` | BO | e.g. PBGI / LogEI |
| `n_init` / `n_steps` / `init_budget_fraction` | BO | Init / BO step schedule |

---

## 3. Per-step history (`wandb.log`, required)

Log **one row after each decision / batch**.

### 3.1 Required metrics for plotting

| Key | Type | Meaning |
|---|---|---|
| `cum_eval` | int | Cumulative number of cell evaluations so far |
| `cum_original_cost` | float | Cumulative original cost so far |
| `simple_regret` | float | `mu_star - true_mean[recommended_arm]` |
| `recommended_arm` | int | Current recommended best arm |
| `pulled_arm` | int | Arm pulled at this step (if single-arm pull) |

Metric definition in code:
```python
wandb.define_metric("cum_eval")
wandb.define_metric("simple_regret", step_metric="cum_eval")
wandb.define_metric("cum_original_cost")
```

### 3.2 Recommended extra history fields

| Key | Type | Meaning |
|---|---|---|
| `recommended_mean` | float | Estimated mean of recommended arm |
| `step_idx` | int | Step counter |
| `iter_step_s` / `iter_total_s` | float | Timing |
| `batch_cells` | int | Cells evaluated this step |

### 3.3 Optional / algorithm-specific history

| Key | Used by | Meaning |
|---|---|---|
| `gittins_index_pulled` | Gittins | Index of pulled arm |
| `posterior_mean_pulled` | Gittins | Posterior mean of pulled arm |
| `selection_phase` | BO | `random_init` or acquisition phase |
| `acquisition_value` | BO | Acquisition score |
| `pulled_arms_json` / `n_pulled_arms` | batched methods | Multi-arm pulls |
| `pulled_rows_json` / `pulled_cols_json` | cell-level logging | Optional diagnostics |

---

## 4. Run summary (`wandb.summary`, required)

### 4.1 Required for all algorithms

| Key | Type | Meaning |
|---|---|---|
| `final_simple_regret` | float | Simple regret at end of budget |
| `best_seen_regret` | float | Best (min) simple regret over the run |
| `final_cum_eval` | int | Final cumulative evaluations |
| `final_cum_original_cost` | float | Final cumulative original cost |
| `num_batches` | int | Number of logged steps |
| `experiment_variant` | str | Same as config |
| `run_seed` | int | Same as config |

### 4.2 Strongly recommended

| Key | Type | Meaning |
|---|---|---|
| `total_wall_time_s` | float | Wall-clock runtime |
| `matrix_seed` | int/str | Matrix replicate |
| `mmlu_task` / `mmlu_size_bucket` | str | MMLU metadata |

### 4.3 Early-stopping fields (if the method has a stop rule)

Use the naming convention of the method family:

**Gittins-style**
- `gittins_stop_cum_eval`
- `gittins_stop_cum_original_cost`
- `gittins_recommendation_aware_stop_cum_eval`
- `gittins_recommendation_aware_stop_cum_original_cost`

**BO-style**
- `bo_stop_cum_eval`
- `bo_stop_cum_original_cost`
- `bo_stop_index_value`
- (aliases also accepted) `pbgi_stop_cum_eval`, `pbgi_stop_cum_original_cost`

If the new algorithm has early stopping, please add:
- `{algo}_stop_cum_eval`
- `{algo}_stop_cum_original_cost`

and tell us the exact key names so plotting can pick them up.

---

## 5. Semantics that must match existing experiments

1. **Matrix layout**: rows = arms/configs, columns = evaluation instances.
2. **Unit-cost budget**: run until about `eval_budget_fraction * n_cells` evaluations (default 10%).
3. **Cost-aware budget**: run until about `eval_budget_fraction * total_brute_force_original_cost`.
4. **`simple_regret`**: always w.r.t. the true best arm mean on the matrix:
   ```text
   simple_regret = max_arm mean(matrix[arm, :]) - mean(matrix[recommended_arm, :])
   ```
5. **`cum_original_cost`**: sum of per-arm original costs of evaluated cells (same cost vector as existing runs).
6. Keep **unit-cost** and **cost-aware** as separate variants / runs (do not mix in one run).

---

## 6. Minimal working example (new algorithm)

```python
import wandb

run = wandb.init(
    project="GittinsBanditEval",
    group="gsm8k_simple_regret_sweep",
    name="gsm8k_seed1_myalgo_unit_runseed0",
    job_type="simple_regret",
    config={
        "dataset_tag": "gsm8k",
        "matrix": "data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy",
        "run_seed": 0,
        "experiment_variant": "myalgo_unit",
        "policy_family": "myalgo",
        "policy_variant": "myalgo_unit",
        "cost_mode": "unit",
        "cost_aware_run": False,
        "eval_budget_fraction": 0.1,
        "n_arms": 122,
        "n_examples": 1000,
        "n_cells": 122000,
        "budget_max_evals": 12200,
        "matrix_seed": 1,
    },
)
run.define_metric("cum_eval")
run.define_metric("simple_regret", step_metric="cum_eval")
run.define_metric("cum_original_cost")

for step in trajectory:
    run.log({
        "cum_eval": step.cum_eval,
        "cum_original_cost": step.cum_original_cost,
        "simple_regret": step.simple_regret,
        "recommended_arm": step.recommended_arm,
        "recommended_mean": step.recommended_mean,
        "pulled_arm": step.pulled_arm,
        "step_idx": step.idx,
    })

run.summary.update({
    "final_simple_regret": final_simple_regret,
    "best_seen_regret": best_seen_regret,
    "final_cum_eval": final_cum_eval,
    "final_cum_original_cost": final_cum_original_cost,
    "num_batches": num_steps,
    "experiment_variant": "myalgo_unit",
    "run_seed": 0,
    # if early stop exists:
    # "myalgo_stop_cum_eval": stop_eval,
    # "myalgo_stop_cum_original_cost": stop_cost,
})
run.finish()
```

---

## 7. What we currently log (complete checklist for alignment)

This section lists **exactly what our existing runners already upload**.
A new algorithm should match this set (at least the non-algorithm-specific parts).

Source of truth:
- main bandits: `scripts/run_simple_regret_wandb.py`
- BO baselines: `scripts/run_bo_baseline_wandb.py`

### 7.1 Experiment setup we already use

- [ ] Same matrices / cost vectors / seeds as existing baselines
- [ ] Separate unit-cost and cost-aware variants
- [ ] `eval_budget_fraction = 0.1` (10% of exhaustive evaluation)
- [ ] `simple_regret = true_best_mean - true_mean[recommended_arm]`
- [ ] Project default: `GittinsBanditEval`
- [ ] `wandb.define_metric("simple_regret", step_metric="cum_eval")`

### 7.2 Config we already write (`wandb.init(config=...)`)

From `run_simple_regret_wandb.py` (plus parsed `experiment_variant` fields):

- [ ] `dataset_tag`
- [ ] `dataset_tag_resolved`
- [ ] `matrix`
- [ ] `matrix_seed`
- [ ] `run_seed`
- [ ] `experiment_variant` (raw string, e.g. `ucb_B8`, `gittins_unit_B8_scale1e-4_dataset`)
- [ ] `policy_variant`
- [ ] `policy_family`
- [ ] `cost_mode` (`unit` / `cost`)
- [ ] `cost_aware_run`
- [ ] `batch_size`
- [ ] `gittins_batch_size`
- [ ] `cost_scaling_factor`
- [ ] `prior_type`
- [ ] `prior_mean_resolved`
- [ ] `prior_variance_resolved`
- [ ] `prior_bucket`
- [ ] `prior_source`
- [ ] `eval_budget_fraction`
- [ ] `n_arms`
- [ ] `n_examples`
- [ ] `n_cells`
- [ ] `budget_max_evals`
- [ ] `budget_original_cost`
- [ ] `total_brute_force_original_cost`
- [ ] `sim_max_evaluations`
- [ ] `cost_vector` (path; required for cost-aware)
- [ ] `ucb_a` (UCB runs)
- [ ] `warmup_percentage` (LRF runs)
- [ ] `mmlu_task` / `mmlu_size_bucket` (MMLU only)

Also present via `vars(args)` (CLI passthrough):  
`gittins_grid_points`, `gittins_obs_noise_variance`, `gittins_prior_mean`, `gittins_prior_variance`, `mmlu_task_metadata`, `log_step_metrics`, `wandb_*`, `lrf_device`.

### 7.3 History we already log every step (`wandb.log`)

Common fields (all current non-BO algorithms):

- [ ] `cum_eval`
- [ ] `cum_original_cost`
- [ ] `simple_regret`
- [ ] `recommended_arm`
- [ ] `recommended_mean`
- [ ] `pulled_arm`

Gittins-only extras (when applicable):

- [ ] `gittins_index_pulled`
- [ ] `posterior_mean_pulled`

BO extras (from `run_bo_baseline_wandb.py`, in addition to the common fields):

- [ ] `selected_config_arm_id`
- [ ] `observed_y`
- [ ] `acquisition_value`
- [ ] `selection_phase` (`random_init` / acquisition phase)
- [ ] `step_idx`
- [ ] `iter_fit_s`
- [ ] `iter_score_s`
- [ ] `iter_total_s`

### 7.4 Summary we already write (`wandb.summary.update`)

Common summary:

- [ ] `final_simple_regret`
- [ ] `best_seen_regret`
- [ ] `final_cum_eval`
- [ ] `final_cum_original_cost`
- [ ] `num_batches`
- [ ] `total_wall_time_s`
- [ ] `experiment_variant`
- [ ] `run_seed`
- [ ] `matrix_seed`
- [ ] `mmlu_task` / `mmlu_size_bucket` (MMLU)

Gittins / recommendation-stop fields we already store:

- [ ] `gittins_stop_cum_eval`
- [ ] `gittins_stop_cum_original_cost`
- [ ] `gittins_recommendation_aware_stop_cum_eval`
- [ ] `gittins_recommendation_aware_stop_cum_original_cost`
- [ ] `lookup_table_s`
- [ ] `prior_mean_resolved` / `prior_variance_resolved` / `prior_bucket` / `prior_source`

BO summary extras:

- [ ] `num_evaluated_configs`
- [ ] `total_fit_s` / `total_score_s` / `total_iter_logged_s`
- [ ] `policy_variant` / `policy_family` / `acquisition` / `cost_mode`
- [ ] `cost_aware_run` / `cost_scaling_factor` / `eval_budget_fraction`
- [ ] `n_init` / `n_steps` / `init_budget_fraction` / related init rules
- [ ] `bo_stop_cum_eval` / `bo_stop_cum_original_cost` / `bo_stop_index_value`
- [ ] aliases: `pbgi_stop_cum_eval` / `pbgi_stop_cum_original_cost` / `pbgi_stop_index_value`

### 7.5 Minimal “must match us” for a new algorithm

If the new algorithm is a normal bandit baseline (not BO), matching **7.1 + common fields in 7.2/7.3/7.4** is enough:

**config:**  
`dataset_tag`, `matrix`, `run_seed`, `experiment_variant`, `policy_variant`, `policy_family`, `cost_mode`, `cost_aware_run`, `eval_budget_fraction`, `n_arms`, `n_examples`, `n_cells`, `budget_max_evals`, `budget_original_cost`, `total_brute_force_original_cost`, `matrix_seed` (+ MMLU fields if needed)

**history:**  
`cum_eval`, `cum_original_cost`, `simple_regret`, `recommended_arm`, `recommended_mean`, `pulled_arm`

**summary:**  
`final_simple_regret`, `best_seen_regret`, `final_cum_eval`, `final_cum_original_cost`, `num_batches`, `total_wall_time_s`, `experiment_variant`, `run_seed`

If it has early stopping, also add clear stop keys analogous to `gittins_stop_*` / `bo_stop_*`.

---

## 8. What is NOT required for plotting

These appear in the downloader schema but are optional diagnostics:
- git commit / dirty flag
- slurm job ids / hostname / python path
- matrix/cost sha256
- memory peak stats
- detailed per-iteration timing percentiles

Nice to have for reproducibility, not required to draw the paper curves.
