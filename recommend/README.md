# Robust recommendation experiments

Completed comparisons: [GSM8K results](RESULTS.md) and
[PIQA + MMLU results](RESULTS_PIQA_MMLU.md).
Figures: [GSM8K](FIGURES.md) · [PIQA + MMLU, every method](FIGURES_PIQA_MMLU.md).

## AlpacaEval and all 57 MMLU subjects

The [full-suite report](RESULTS_FULL_SUITE.md) compares posterior-mean recommendation
with **LCB = mean − 1.645 × posterior SD** on the same Gittins observations.
The completed suite contains **2,320 trajectories**: 58 datasets × 20 sampling
seeds × two priors. These are offline evaluations against saved reward matrices.
MMLU selects an arm separately for each subject; the primary aggregate weights
the 57 subjects equally, with a question-weighted sensitivity analysis.

LCB reduces total downward movement, maximum drawdown, largest downward jump,
and recommendation switches on every MMLU subject under both priors, averaged
over seeds. Quality improvements are smaller and do not hold on every subject.
The report retains all regressions, paired uncertainty intervals, ten stability
and quality metrics, and links to all 58 individual figures.

- [Overview](figures/full_suite/overview.png)
- [Quality curves](figures/full_suite/quality_curves.png)
- [All MMLU subject effects](figures/full_suite/mmlu_subject_effects.png)
- [Aggregate statistics](figures/full_suite/aggregate_summary.csv)
- [Experiment manifest](results/full_suite/suite_manifest.json)
- [Independent audit summary](results/full_suite/audit_summary.json)
- [Independent statistical review](figures/full_suite/aggregate_validation.json)

Reproduce using NumPy, SciPy, and Matplotlib (see `requirements_followup.txt`):

```powershell
python recommend/test_full_suite_engine.py -v
python -m unittest recommend.test_full_suite_audit -v
python recommend/run_full_suite.py --workers 6
python recommend/validate_full_suite.py --root recommend/results/full_suite --workers 8
python recommend/summarize_full_suite.py --self-test
python recommend/summarize_full_suite.py
```

Use `--resume` for an existing unchanged experiment or `--out-root` for a fresh
directory. On this machine, run the experiment and analysis through the bundled
Python with `recommend/with_runtime.py .venv/Lib/site-packages`, as illustrated
below for the earlier follow-up. The compact engine saves every sampled cell
and every paired recommendation; its independent auditor reconstructs all
allocation decisions, scores, posterior statistics, and regrets. Each dataset's
`validation.json` records its audit and source hashes.

## Earlier temporal stability comparison

The [temporal stability comparison](ROBUSTNESS_RESULTS.md) analyzes recommendation
jumps across all 520 saved trajectories, covering cumulative deterioration,
drawdown, switches, tail risk, and budget-averaged quality. It includes figures,
per-run metrics, and paired uncertainty intervals for both priors.

```powershell
python -m unittest recommend.test_stability -v
python recommend/analyze_stability.py
```

This reanalysis uses the same NumPy/Matplotlib runtime as the follow-up below;
it does not perform new model evaluations or modify saved trajectories.

## PIQA and MMLU follow-up

The follow-up retains the 18 recommendation rules and fixes Gaussian LCB
`z=1.645` as the primary comparator. It contains 320 additional trajectories:
5 PIQA matrices × 20 sampling seeds × 2 priors, and one matrix × 20 seeds ×
2 priors for each of Computer Security, Anatomy, and Business Ethics.
PIQA uses batch size 16; MMLU uses 4. All runs stop at an exact 10% cell budget.
See the follow-up report for priors, numerical-backend changes, audits, and limits.

Run with NumPy/SciPy/Matplotlib installed (the follow-up does not require JAX or Torch):

```powershell
python recommend/run_benchmarks.py --benchmark piqa
python recommend/run_benchmarks.py --benchmark mmlu --tasks computer_security anatomy business_ethics
```

For each result directory (`piqa`, `mmlu_computer_security`, `mmlu_anatomy`,
`mmlu_business_ethics`), run the following with the corresponding name:

```powershell
python recommend/plot_results.py --results-dir recommend/results/piqa --out-dir recommend/figures/piqa
python recommend/validate_results.py --results-dir recommend/results/piqa
python recommend/audit_benchmarks.py recommend/results/piqa
```

After plotting all four datasets:

```powershell
python recommend/summarize_benchmarks.py
python recommend/validate_portable.py
```

Existing completed directories require `--resume` or a fresh `--out-root`.
The exact follow-up package versions are in `requirements_followup.txt` and
each manifest. On this machine, `with_runtime.py` appended the existing virtual
environment's extra packages after the bundled Python's packages:

```powershell
& 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' recommend/with_runtime.py .venv/Lib/site-packages recommend/run_benchmarks.py --benchmark piqa --resume
```

`results/<dataset>/decision_audit.json` records the independent audit of every
recommendation and allocation. `figures/piqa_mmlu/` contains cross-dataset
plots, all-method checkpoint CSVs, and coverage/unobserved-arm diagnostics.

## Original GSM8K comparison

This experiment compares recommendations on identical Gittins sampling histories.
The existing project policies and experiment scripts are unchanged. Each rule
uses only the revealed observations; full matrix row means are used for scoring.

## Experiment design

- All five `gsm8k_1_samples_various_models_seed*.npy` matrices: 122 configurations
  and 1,000 examples each.
- Twenty sampling seeds (0–19) for each matrix and each of two priors: the default
  `N(0.5, 0.04)` and the dataset prior `N(0.2, 0.01)`.
- 200 Gittins trajectories; 18 recommendations evaluated on every post-pull state.
- Unit evaluation cost, batch size 16, a 10% budget (12,200 cells), Gittins cost
  scale `1e-4`, per-cell observation variance `0.25`, and 1,025 DP grid points.
- Gittins uses the existing per-cell DP and continues beyond nominal stopping
  markers. Only the final batch is truncated to meet the exact budget; the
  original project runner can instead overshoot by part of a batch.
- Priors, parameter grids, and representative overview curves were fixed before
  inspecting results. This is an exploratory comparison, not held-out tuning.

## Recommendation families

| Family | Rule |
|---|---|
| Current baseline | Largest Gaussian posterior mean |
| Empirical baseline | Largest observed average among sampled arms |
| Variance gate | Largest posterior mean among arms with `v/v0 <= rho`; otherwise abstain |
| Mean–variance | Largest `mu - alpha*v` (Gaussian exponential-utility preference) |
| Gaussian lower quantile | Largest `mu - z*sqrt(v)` for the latent arm mean |
| Finite-row Gaussian lower quantile | Largest posterior lower quantile of the actual matrix-row average |
| Anytime finite-population confidence | Largest simultaneous Hoeffding–Serfling lower bound |

The gate grid is `rho = 0.5, 0.2, 0.1, 0.05`; variance penalties use
`alpha = 1, 5, 10, 20`; latent Gaussian quantiles use
`z = 1, 1.645, 1.96, 2.576`; finite-row quantiles use `z = 1, 1.645, 1.96`.
The confidence method uses a global error budget `delta = 0.05` across all arms
and observation counts within a run. All exact score ties choose the first row.

See [STATISTICAL_NOTES.md](STATISTICAL_NOTES.md) for formulas, targets, derivations,
and primary references. Posterior credibility and frequentist confidence are
different guarantees. A lower confidence score alone does not certify that the
chosen arm is best.

## Run and reproduce

From the project root, with Python 3.12:

```powershell
python -m pip install -r recommend/requirements.txt
python -m unittest recommend.test_methods -v
python recommend/run_gsm8k.py --out-dir recommend/results/gsm8k
python recommend/plot_results.py --results-dir recommend/results/gsm8k --out-dir recommend/figures/gsm8k
python recommend/validate_results.py --results-dir recommend/results/gsm8k
```

Use a new output directory for a changed experiment. `--resume` only accepts
unchanged configuration, source hashes, input hashes, and package versions.
The runner precomputes roots with the original JAX code, then maintains observed
sufficient statistics incrementally. It checks the first 128 pulls under each
prior against `src/gittins_policy.py`, including the sampled example indices.
The same Torch random generator is used for sampling; recommendations do not
consume randomness or affect allocation.

## Saved outputs

- `results/gsm8k/manifest.json`: parameters, method grid, input/source hashes,
  package versions, timestamps, and completion status.
- `results/gsm8k/roots/`: Gittins root tables used in these runs.
- `results/gsm8k/traces/`: per-run compressed arrays with every recommendation,
  its regret and selected arm's observation count/latent posterior variance, eligibility counts,
  selected cell indices, sufficient-statistic histories, and confidence bounds.
- `figures/gsm8k/`: comparison figures, a figure for each individual rule, and
  parameter-sensitivity figures. CSV summaries record endpoint and intermediate
  budget results.
- `RESULTS.md`: interpretation of the completed experiment.
- `FIGURES.md`: links to every individual method and parameter-family figure.

The saved `selected_variance` field always means latent posterior variance,
including for finite-row rules. Finite-row uncertainty is computed separately
inside those rules from the saved counts and sums.

An abstention is arm `-1` with undefined (`NaN`) regret, never zero regret.
Read gated regret together with recommendation coverage. Plot interpolation
holds the last recommendation only; it never uses future observations.
Uncertainty intervals describe Monte Carlo variation conditional on the five
fixed matrices, using sampling-seed blocks to retain pairing across matrices.
They do not establish generalization to unseen benchmarks or configurations.
