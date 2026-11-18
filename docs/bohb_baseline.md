# BOHB baseline

BOHB uses the authors' HpBandSter 0.7.4 KDE configuration generator and
Hyperband/successive-halving scheduler. A synchronous driver replaces network
worker dispatch. Allocation and recommendation retain the author defaults;
the response-matrix oracle supplies nested item-subsampling fidelities.
MF-KG remains a separate multi-fidelity baseline.

## Defaults and fairness

The optimizer defaults are `eta=3`, `min_budget=0.01`, `max_budget=1`,
`top_n_percent=15`, `random_fraction=1/3`, `num_samples=64`,
`min_points_in_model=None` (active dimension plus one),
`bandwidth_factor=3`, and `min_bandwidth=0.001`.
We adopt the constructor's `0.01`--`1` resource bounds as fractions of the
benchmark examples. This mapping is our stated experimental choice; BOHB
accepts task-specific resource bounds and does not infer them from the number
of candidate configurations. The main protocol keeps this mapping across
benchmarks and retains BOHB's native geometric schedule. Evaluation counts
are rounded up to whole examples, without rounding to the bandit batch size
or forcing every BOHB job to reveal a fixed-size batch.
The realized fidelities are **1/81, 1/27, 1/9, 1/3, 1**. The lower-bound
argument `0.01` is not itself a rung. For 1,000 examples, rounding up gives
13, 38, 112, 334, and 1,000 examples. Small datasets can map two nominal
fidelities to the same integer count; the original nominal schedule is kept.

The comparison controls the matrix, seed structure, evaluation costs, and
nominal total resource budget, while retaining each policy's allocation and
recommendation mechanisms. Using library defaults makes the BOHB setup
reproducible; it does not by itself establish fairness. Both cost modes receive
10% of exhaustive evaluation. All initial observations
are charged. Increasing fidelity reveals additional examples from a fixed
seeded per-arm order, matching the MF-KG oracle. Previously observed cells
are cached across repeated jobs and brackets, and never charged twice.
Native duplicate proposals still reach the scheduler and KDE with their
cached objective values; the baseline does not change their selection rule.

Under heterogeneous costs, BOHB is **cost-unaware in allocation**. The objective
remains `1 - observed prefix mean`; it is not accuracy divided by price.
Actual incremental cost determines whether the next requested job fits.
The run stops before the first unaffordable job without resizing or skipping
that job. Report the recorded unused budget and hold the final recommendation
for the remainder of a budget-indexed curve. Do not classify this as a
cost-aware BOHB acquisition or a value-based stopping signal.

## Categorical configuration features

MMLU uses `(model_idx, prompt_idx)` (15 by 100). GSM8K and PIQA use
`(model_id, temp_id, max_len_id, prompt_id)`. All are unordered categorical
features. Combinations absent from the supplied matrix are forbidden in
ConfigSpace. Constant columns are dropped, retaining all configuration
identity information. AlpacaEval uses model identity and requires an explicit
matrix/metadata pair in the suite runner: the local 153-model archive must not
be silently substituted for paper figures reporting 152 models.

The loader reorders metadata using `arm_ids` and converts full-row costs to
per-example costs. It never reads the BO archive's `Y` or `raw_scores_mean`.
Full matrix means are used only to compute reporting regret.

## Default recommendations

Anytime recommendations reproduce HpBandSter's
`get_incumbent_trajectory(all_budgets=True, bigger_is_better=True,
non_decreasing_budget=True)`: prioritize the highest nominal fidelity
encountered so far, then the lowest observed loss at that fidelity.
No Gittins prior or LCB penalty is added.

HpBandSter's separate `get_incumbent_id()` final-result interface accepts only
maximum-fidelity evaluations. Accordingly, the trace also records
`full_fidelity_recommended_arm`, set to `-1` when no full-fidelity job finishes.
The lower-fidelity fallback is used for anytime reporting only.
Tests compare both conventions directly with the authors' result utilities.

## Run experiments

Install the optional baseline dependencies in the project environment:

```bash
python -m pip install -e '.[bohb]'
```

One matrix and one algorithm seed, with the full 10% budget:

```bash
python scripts/run_bohb_baseline.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --bo-inputs data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz \
  --seed 0 --experiment-variant bohb_unit
```

Use `bohb_cost` for heterogeneous-cost budgeting. The default output is
`outputs/bohb/{matrix}__{variant}__seed{seed}.npz`, with a JSON sidecar and
a `.log` file preserving native KDE/proposal warnings and random fallbacks.
Existing results are not overwritten. `--max-jobs` is only an implementation
smoke-test cap; omit it for benchmark results. W&B is disabled unless enabled
explicitly. A seed-1 W&B grid is in `scripts/config/GSM8KBOHBSweep.yml`.

The suite runner pairs each matrix with its metadata instead of crossing
unrelated matrix/input paths. Its default plan has 2,680 jobs: GSM8K and PIQA
each use five matrices and 20 seeds under two cost modes; MMLU uses all
57 subjects and 20 seeds under both modes.

```bash
python scripts/run_bohb_benchmarks.py --dry-run
python scripts/run_bohb_benchmarks.py --datasets mmlu --subjects abstract_algebra --seeds 0
python scripts/run_bohb_benchmarks.py --skip-existing
```

Raw traces include cumulative unique cells, cumulative original cost,
requested fidelity and prefix size, newly revealed cells, observed mean,
recommendation, bracket/rung, model-based proposal status, and library versions.
The `bohb_x`, `bohb_x_original_cost`, and `bohb_regret` aliases are accepted by
`plot_bandit_bo_comparison.py --bohb-trace PATH`. Use post-step curves and no
backfill before the first completed observation. For MMLU, average seeds within
each subject before equally weighting subjects, as for the other baselines.

## Validation

```bash
python -m pytest -q tests/test_bohb_baseline.py
```

Checks cover author-default fidelity levels, direct parity with native
recommendation utilities, real KDE proposals, sparse categorical grids,
cached-prefix accounting, strict budgets, cost-scale invariance,
unaffordable first jobs, and independence from unrevealed response scores.

Sources: [BOHB paper](https://proceedings.mlr.press/v80/falkner18a.html),
[optimizer source](https://github.com/automl/HpBandSter/blob/master/hpbandster/optimizers/bohb.py),
[result utilities](https://github.com/automl/HpBandSter/blob/master/hpbandster/core/result.py).
