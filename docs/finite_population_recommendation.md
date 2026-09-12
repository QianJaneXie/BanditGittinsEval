# Full fixed test-set posterior mean recommendation

Gittins recommendations now target the full realized row mean used to compute simple regret. By default, recommend the arm with the largest posterior expectation of that mean, without subtracting a standard deviation. This changes recommendation, while retaining the existing latent-mean Gittins sampling policy.

## Posterior moments

Let an arm have N fixed examples, n observed values with sum S, and m = N − n unrevealed examples. Under the existing model, the latent mean has posterior `theta | D ~ Normal(mu, v)`, and cells are conditionally independent with variance `tau_cell^2`. The full-test-set mean F has posterior moments

\[
\widetilde\mu = E[F\mid D] = \frac{S + m\mu}{N},
\qquad
\widetilde v = \operatorname{Var}(F\mid D)
= \frac{m^2 v + m\tau_{\mathrm{cell}}^2}{N^2}.
\]

The observed scores contribute their known values; the latent posterior predicts only the unrevealed scores. The variance includes both uncertainty in the latent mean and the unrevealed cells' conditional variation. It is not just the latent variance multiplied by the remaining fraction squared.

- Completed arm (`n = N`): exact full-row empirical mean, variance zero.
- Unobserved arm (`n = 0`): prior mean, variance `v0 + tau_cell^2 / N`. A high prior can therefore still cause an unobserved arm to be recommended.
- Partial arm: `n/N * observed_mean + (N-n)/N * latent_posterior_mean`. The implementation uses the observed sum to avoid `0 * NaN` for unobserved arms.

`posterior_moments` and `posterior_means` continue to represent the latent parameter. `finite_population_posterior_moments` represents F, and `posterior_incumbent` uses these finite-population moments. This keeps the two estimands explicit.

## Why the mean-only ranking is unchanged here

All arms in the current model share N, prior mean mu0, prior variance v0, and cell variance tau_cell^2. Writing c = tau_cell^2 / v0 gives `mu = (S + c * mu0) / (n + c)`. Substituting the observed sum into the finite-test-set mean yields

\[
\widetilde\mu
= \left(1+\frac{c}{N}\right)\mu - \frac{c}{N}\mu_0.
\]

This is the same strictly increasing affine transformation for every arm, even when their observed counts differ. Consequently, the mean-only recommendation ranking is identical in exact arithmetic. Changing the estimand corrects the reported means and the completed-arm uncertainty, but cannot itself remove mean-only recommendation oscillation in this shared-parameter model. Different arm-specific priors, noise variances, or test-set sizes would remove this particular invariance; those are not introduced here. The full-test-set std correction can change LCB rankings.

## Runners and optional LCB

Both simulation and W&B runners default to `--recommendation-std-penalty 0`. A positive coefficient lambda recommends the maximum of `finite_mean - lambda * sqrt(finite_variance)`. Returned and logged `recommended_mean` values are the unpenalized full-test-set means. `posterior_mean_pulled` remains the latent-mean sampling diagnostic.

The recommendation-aware stopping diagnostic compares the largest unfinished latent Gittins index to the selected arm's unpenalized full-test-set mean. It remains a diagnostic, not a newly derived optimal stopping rule for the finite-population objective. Fixed-budget runners do not stop at this crossing. The original index-induced stopping diagnostic retains its latent-mean semantics.

NPZ and W&B metadata use `finite_population_posterior_mean` or `finite_population_posterior_mean_minus_std`, distinguishing these runs from historical latent-mean runs.

Normal W&B downloads retain this metadata, and plotting separates the recommendation targets. Legacy expected-grid completeness and targeted-recovery keys cannot distinguish targets: those optional modes reject finite-population runs, including zero-penalty runs, rather than pooling them with latent-mean experiments.

## GSM8K paired experiment

The comparison uses the README's GSM8K matrix `gsm8k_1_samples_various_models_seed1.npy` (122 arms × 1,000 examples), sampling seed 0, batch size 16, unit costs, Gittins cost scale `1e-4`, and a 10% evaluation budget. It compares the general prior `Normal(0.5, 0.04)` and the README's data-specific prior `Normal(0.2, 0.01)`, with particular attention to the stronger oscillation under the general prior. Each prior has exactly one sampling trajectory on the fixed matrix; the old and new recommendation rules share that trajectory.

Two rules are evaluated after each identical batch: the old latent posterior mean and the new full-test-set posterior mean, both without a std penalty. Raw recommendation switches, upward regret jumps, total variation, and budget-weighted mean regret measure oscillation and performance separately. Plots show the single unsmoothed trajectory, with no averaging across seeds or standard-error bands.

```bash
.venv/bin/python scripts/compare_recommendation_rules.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --dataset-tag gsm8k --prior-types default dataset \
  --seeds 1 --seed-start 0 --batch-size 16 --budget-fraction 0.10 \
  --cost-scaling-factor 1e-4 --grid-points 1025 \
  --out-dir outputs/recommendation_finite_mean_gsm8k_seed1_run0
```

```bash
.venv/bin/python scripts/analyze_recommendation_oscillation.py \
  --out-dir outputs/recommendation_finite_mean_gsm8k_seed1_run0 \
  --prior-type default --expected-seeds 1

.venv/bin/python scripts/analyze_recommendation_oscillation.py \
  --out-dir outputs/recommendation_finite_mean_gsm8k_seed1_run0 \
  --prior-type dataset --expected-seeds 1
```

### Results: matrix seed1, sampling seed0

The mean-only recommendations match at every post-pull state: 766 of 766 batches under the general prior and 767 of 767 under the data-specific prior. Both trajectories end at exactly 12,200 evaluated cells. The old and new regret curves overlap exactly.

| Prior | Recommendation | Arm switches | Upward regret jumps | Regret total variation | Final regret |
|---|---|---:|---:|---:|---:|
| General `N(0.5, 0.04)` | Old latent mean | 141 | 65 | 10.632 | 0 |
| General `N(0.5, 0.04)` | New full-test-set mean | 141 | 65 | 10.632 | 0 |
| Data-specific `N(0.2, 0.01)` | Old latent mean | 41 | 18 | 1.222 | 0 |
| Data-specific `N(0.2, 0.01)` | New full-test-set mean | 41 | 18 | 1.222 | 0 |

The strong general-prior oscillation remains. The affine relation above explains this exactly: the general prior gives `finite_mean = 1.00625 * latent_mean - 0.003125`, while the data-specific prior gives `finite_mean = 1.025 * latent_mean - 0.005`. Both preserve rankings. The change makes recommendation means describe the intended full test set, but does not change which arm wins the mean-only comparison.

See [the unsmoothed regret comparison](../outputs/recommendation_finite_mean_gsm8k_seed1_run0/simple_regret.png), [recommendation switches](../outputs/recommendation_finite_mean_gsm8k_seed1_run0/recommendation_switches.png), and [saved configuration and results](../outputs/recommendation_finite_mean_gsm8k_seed1_run0/summary.json). General-prior and data-specific raw oscillation metrics are saved under `default/` and `dataset/` respectively. Only sampling seed 0 is included; these results do not average over seeds.

## Does LCB still help after changing the mean?

The follow-up comparison keeps exactly the same matrix, sampling seed 0, both priors, B16, unit costs, cost scale, and budget. The main comparison is full-test-set posterior mean versus full-test-set posterior mean minus one full-test-set posterior standard deviation (`lambda = 1`). The historical latent-mean rules are retained as controls. Sampling is shared across all four rules within each prior; changing the recommendation does not change the sampled data.

```bash
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/compare_recommendation_rules.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --dataset-tag gsm8k --prior-types default dataset \
  --seeds 1 --seed-start 0 --batch-size 16 --budget-fraction 0.10 \
  --include-lcb --std-penalty 1 --cost-scaling-factor 1e-4 --grid-points 1025 \
  --out-dir outputs/recommendation_finite_lcb_gsm8k_seed1_run0

MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/analyze_recommendation_oscillation.py \
  --out-dir outputs/recommendation_finite_lcb_gsm8k_seed1_run0 \
  --prior-type default --expected-seeds 1

MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/analyze_recommendation_oscillation.py \
  --out-dir outputs/recommendation_finite_lcb_gsm8k_seed1_run0 \
  --prior-type dataset --expected-seeds 1
```

The affine invariance of mean-only ranking does not extend to LCB. With `a = 1 + tau_cell^2/(N*v0)` and `r = (N-n)/N`, the full-test-set variance satisfies `finite_variance = a*r*latent_variance`. Consequently, ranking by finite LCB is equivalent to ranking by `latent_mean - lambda*sqrt(r/a)*latent_std`. The effective uncertainty penalty decreases as an arm approaches completion.

### Finite mean versus finite LCB: seed0 results

The mean-only control reproduces the previous run's observations, recommended arms, and regrets exactly for each prior. The following table compares the two **full-test-set** rules. Mean regret is weighted by evaluation budget over the observed interval, from 16 through 12,200 evaluated cells; it is not a smoothed curve or a mean over sampling seeds.

| Prior | Full-test-set recommendation | Switches | Upward regret jumps | Regret total variation | Budget-weighted mean regret | Final regret |
|---|---|---:|---:|---:|---:|---:|
| General | Mean | 141 | 65 | 10.632 | 0.04873014 | 0 |
| General | Mean − std | 48 | 22 | 2.210 | 0.03018385 | 0 |
| Data-specific | Mean | 41 | 18 | 1.222 | 0.03117663 | 0 |
| Data-specific | Mean − std | 41 | 18 | 1.058 | 0.03154563 | 0 |

Under the general prior, finite LCB reduces switches by 66.0%, regret total variation by 79.2%, and budget-weighted mean regret by 38.1%. It reduces but does not eliminate oscillation. The first zero-regret recommendation is later (1,696 → 2,032 evaluations), while the point after which regret stays zero through the budget is earlier (5,120 → 4,824). Final regret is unchanged at zero.

One direct effect is less frequent recommendation of unobserved arms: 118 → 34 of 766 post-pull states. An unobserved arm still has full-test-set posterior mean 0.5 under this prior, above this matrix's best realized row mean of 0.413. Subtracting the full-test-set std lowers its score to `0.5 - sqrt(0.04 + 0.25/1000) = 0.299376`. Correcting the mean target alone does not provide this reduction.

Under the data-specific prior, switching and upward-jump counts are unchanged. LCB reduces total variation by 13.4%, but raises budget-weighted mean regret by 1.18% and delays sustained zero regret (5,816 → 6,072 evaluations). The first zero-regret recommendation remains at 1,760 evaluations, and both rules recommend unobserved arms in 26 of 767 states. Thus smaller oscillation amplitude does not translate into better average regret in this case.

For this specified seed0 experiment, finite LCB has a clear practical benefit under the general prior, while mean-only remains preferable on average regret under the data-specific prior. The runner default remains mean-only; `--recommendation-std-penalty 1` explicitly enables finite LCB. This empirical benefit is not a Bayesian optimality requirement: under a trusted posterior and a posterior-expected-simple-regret objective, maximizing the full-test-set posterior mean remains the optimal terminal recommendation.

See [the focused finite mean versus LCB comparison](../outputs/recommendation_finite_lcb_gsm8k_seed1_run0/finite_mean_vs_lcb.png), [all four control curves](../outputs/recommendation_finite_lcb_gsm8k_seed1_run0/simple_regret.png), [general-prior metrics](../outputs/recommendation_finite_lcb_gsm8k_seed1_run0/default/oscillation_metrics.json), and [data-specific metrics](../outputs/recommendation_finite_lcb_gsm8k_seed1_run0/dataset/oscillation_metrics.json).
