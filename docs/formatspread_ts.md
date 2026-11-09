# FormatSpread-TS

Main-text baseline: **FormatSpread-TS**. Gaussian TS remains a separate
appendix comparison (`ts`); historical TS results are not relabeled.

Based on Sclar et al., *Quantifying Language Models' Sensitivity to Spurious
Features in Prompt Design* (ICLR 2024), §3.2:
https://arxiv.org/html/2310.11324v2#S3.SS2
Official implementation:
https://github.com/msclar/formatspread/blob/main/format_evaluation.py

## Adaptation

- Treat each supplied model/prompt/decoding configuration as an arm. Only run
  maximization; all exploration budget searches for the best configuration.
- Independently initialize every arm using the existing data-specific prior mean x,
  with beta_0=5 and alpha_0=max(5*x/(1-x),1.1), following the original parameterization.
  Unlike the official implementation, do not fully evaluate a reference prompt
  to calibrate the prior. There is no free initialization or final evaluation.
- After observing scores y, use alpha = alpha_0 + sum(y), beta = beta_0 + n - sum(y).
  Draw one Beta value per unfinished, affordable arm and select the largest.
- Recommend the largest empirical mean among observed arms, including completed
  arms, following the original selection rule. This differs from Gittins' full
  test-set posterior recommendation; Gaussian TS isolates that modeling choice.
- Reveal a random batch of unseen items. Use the same nominal batch size and
  evaluation budget as the comparison methods. A final short batch can fit the
  remaining budget. Costs constrain affordability and stopping; sampled accuracy
  is never divided by price. This is a cost-budgeted baseline, not a cost-aware
  information-allocation algorithm. No finite-population posterior correction.

## AlpacaEval

The configured `no_rounding_debias.npy` matrix has 153 arms and 805 items, with
finite scores in [0,1]. Keep the raw fractional scores and original mean-regret
objective. The update above becomes a **fractional-count Beta extension**
(power likelihood y log(theta) + (1-y) log(1-theta)); it is not the exact
Beta--Bernoulli posterior for continuous scores. For binary matrices the same
implementation reduces to the usual conjugate update. This extension is our
adaptation, not a claim that the original method evaluated AlpacaEval.

Do not threshold or randomly binarize scores: either would alter the observed
feedback or add avoidable sampling noise. Reject scores outside [0,1] rather
than clipping them. In particular, some alternate Alpaca matrices contain -2
sentinels and must not be used directly.

## Running

```bash
python scripts/simulate_simple_regret.py \
  --matrix data/BanditEval_matrices/alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_no_rounding_debias.npy \
  --algorithms formatspread_ts --formatspread-batch-size 8 \
  --eval-budget-fraction 0.1 --seed 0 \
  --out outputs/formatspread_ts/alpaca_seed0.npz
python scripts/plot_simple_regret_results.py \
  --traces outputs/formatspread_ts/alpaca_seed0.npz \
  --out outputs/formatspread_ts/alpaca_seed0.png
```

For cost-budgeted simulation add `--cost-vector` with per-cell arm costs.
The W&B runner accepts `formatspread_ts_unit_B8` and
`formatspread_ts_cost_B8` (also B16/B32). Both entry points expose
`--formatspread-prior-mean` to override x, or an explicit pair of
`--formatspread-prior-alpha` and `--formatspread-prior-beta`.
Defaults: GSM8K/Alpaca x=0.2 (alpha_0=1.25, beta_0=5), PIQA x=0.4
(alpha_0=10/3, beta_0=5); MMLU reuses its task bucket mean. Unknown simulation
matrices fall back to x=0.5. The original alpha floor can shift the actual mean
when x < 1.1/6.1; both requested x and actual mean are recorded.
These priors reuse the existing experimental protocol; if its data-specific means
were calibrated on the evaluation data, that limitation also applies here.
Trace metadata records the update, prior, recommendation, and cost allocation.
GSM8K, PIQA and Alpaca sweep specifications are in
`scripts/config/*FormatSpreadTSSweep.yml`. Creating them does not launch or upload runs.

## Paired GSM8K comparison against GittinsEval-S (LCB)

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/compare_formatspread_gittins.py
```

This runs only FormatSpread-TS, using Beta(1.25,5), empirical recommendation,
B=8, a 10% budget, five matrix seeds and twenty trial seeds per matrix, under
unit and heterogeneous costs. The comparison reads GittinsEval-S's existing
LCB trajectories from `outputs/gsm8k_gittins_ts_batch_lcb100`, validates the
source protocol, matrix and ordered-price hashes, and verifies that source
archives remain unchanged. No Gittins policy or root computation is invoked.

Outputs: `outputs/gsm8k_formatspread_gittins_s/`, containing paired figures,
200 FormatSpread traces, per-run metrics, checkpoint/AUC summaries with paired
two-way matrix/seed bootstrap intervals, and source-file SHA256 provenance.
Plots use the previous completed observation, including the initial arm-0 tie;
no over-budget feedback enters fixed-budget metrics. Unlike the cached
Gittins batch-end runs, FormatSpread fits affordable final partial batches.
The Gittins LCB and FormatSpread empirical recommendation rules differ and
are retained as part of each method.
