# Historical latent posterior mean minus standard deviation recommendation

This document records the earlier **latent-mean** recommendation and its experiments. Current runners instead recommend using the **full fixed test-set mean**, with no standard-deviation penalty by default; see [the current rule and GSM8K comparison](finite_population_recommendation.md). The historical results below remain unchanged. The comparison script now evaluates both targets on the same trajectory, so its latent-mean rows still reproduce these baselines. The simulation command below now produces a full-test-set LCB trace, not the historical latent-mean trace.

The earlier Gittins recommendation maximized

\[
S_{k,t}=\mu_{k,t}-\lambda\sigma_{k,t}.
\]

For these historical results, `lambda = 1` implements latent posterior mean minus posterior standard deviation, and `lambda = 0` is the latent posterior-mean baseline. Every arm receives a score; there is no variance cutoff or candidate-set fallback. The earlier variance-filter implementation and its CLI parameters have been removed.

The minus sign gives a lower-confidence-bound (LCB) style recommendation: uncertainty reduces the score. Gittins still chooses the next arm to sample using its index. UCB-E, SySRs, and LRF retain their existing recommendation rules.

## Posterior and usage

The existing normal-normal model gives

\[
v_{k,t}=\left(\frac{1}{v_0}+\frac{n_k}{\tau_{\mathrm{cell}}^2}\right)^{-1},
\qquad
\mu_{k,t}=v_{k,t}\left(\frac{\mu_0}{v_0}+\frac{\sum_i y_{ki}}{\tau_{\mathrm{cell}}^2}\right),
\qquad
\sigma_{k,t}=\sqrt{v_{k,t}}.
\]

The penalty uses uncertainty about the latent arm mean, rather than the observation-noise standard deviation. With the default batch observation model, `tau_cell^2 = B * tau_batch^2 = 0.25`.

Both `simulate_simple_regret.py` and `run_simple_regret_wandb.py` accept `--recommendation-std-penalty 1`. The W&B runner also accepts `--recommendation_std_penalty 1`. The coefficient must be finite and nonnegative. Returned and logged recommendation means remain raw posterior means. The recommendation-aware stopping diagnostic compares the best unfinished Gittins index with the selected arm's raw posterior mean.

For example, the data-specific GSM8K prior is `N(0.2, 0.01)`:

```bash
.venv/bin/python scripts/simulate_simple_regret.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --out outputs/bandit_traces/gsm8k_seed1_lcb.npz \
  --algorithms gittins --seed 0 --eval-budget-fraction 0.10 \
  --gittins-batch-size 16 \
  --gittins-prior-mean 0.2 --gittins-prior-variance 0.01 \
  --recommendation-std-penalty 1
```

## GSM8K oscillation comparison

The README's reference configuration uses `gsm8k_1_samples_various_models_seed1.npy` (122 arms × 1,000 examples), sampling seed 0, batch size 16, unit evaluation costs, Gittins cost scaling `1e-4`, and a 10% evaluation budget. Commit `a9e501c` ("Match the batch size and prior used in the paper for illustration") explicitly sets GSM8K's paper illustration batch size to 16 for both UCB-E and Gittins. This B16 setting is used for the main LCB comparison; the B20 run below is only a historical diagnostic. Compare both priors:

| Prior | Mean | Variance |
|---|---:|---:|
| General (`default`) | 0.5 | 0.04 |
| Data-specific (`dataset`) | 0.2 | 0.01 |

Within each prior and seed, the two recommendation rules use the same observed data after every pull. Recommendation does not change sampling or stop the fixed-budget trajectory. Changing the prior changes the sampling trajectory, so the two priors are separate experiments.

The oscillation check uses unsmoothed seed-0 curves. Recommendation switches count changes of the recommended arm between adjacent batches. Upward regret jumps count positive changes in simple regret; total variation is the sum of absolute adjacent regret changes. These measure stability separately from final regret and regret averaged over the budget. A stable but consistently poor recommendation is not an improvement.

Reproduce the comparison:

```bash
.venv/bin/python scripts/compare_recommendation_rules.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --dataset-tag gsm8k --prior-types default dataset \
  --seeds 1 --seed-start 0 --batch-size 16 --budget-fraction 0.10 \
  --include-lcb --std-penalty 1 --cost-scaling-factor 1e-4 --grid-points 1025 \
  --out-dir outputs/recommendation_lcb_gsm8k_seed1_run0
```

### Results for matrix seed1 / run seed0

Both priors reached exactly 12,200 evaluations. Some arms exhaust their 1,000 examples in a partial final batch, giving 766 batches under the general prior and 767 under the data-specific prior. The curves are not averaged over seeds and have no standard-error bands.

| Prior | Recommendation | Arm switches | Upward regret jumps | Regret total variation | Mean regret per post-batch state | Final regret |
|---|---|---:|---:|---:|---:|---:|
| General | Posterior mean | 141 | 65 | 10.632 | 0.048450 | 0 |
| General | Posterior mean − std | 50 | 23 | 2.190 | 0.029948 | 0 |
| Data-specific | Posterior mean | 41 | 18 | 1.222 | 0.030983 | 0 |
| Data-specific | Posterior mean − std | 39 | 17 | 1.218 | 0.031140 | 0 |

For this seed, the standard-deviation penalty clearly reduces oscillation under the general prior: switches fall by 64.5%, total variation by 79.4%, and mean regret per batch by 38.2%. The benefit persists after 1,220 evaluations (1% of all matrix cells): switches fall from 65 to 17 and total variation from 4.573 to 0.418. Under the data-specific prior, oscillation changes little and average regret increases slightly.

A major source of the general-prior oscillation is recommending unobserved arms. Their posterior mean remains 0.5, above the best true row mean in this matrix (0.413). The penalty lowers their recommendation score to `0.5 - sqrt(0.04) = 0.3`. Post-batch states recommending unobserved arms fall from 118 to 34; those states account for about 96.9% of the total reduction in summed regret. This explains most of the improvement on this trajectory. Under the data-specific prior, both rules recommend an unobserved arm in 26 states.

The penalty has a tradeoff: under the general prior, the first zero-regret recommendation occurs later (1,696 → 2,032 evaluations), but the recommendation settles at zero regret through the rest of the run earlier (5,120 → 4,600). Under the data-specific prior, it settles at zero regret slightly later (5,816 → 5,960). These are observations from this particular seed and unit-cost setup.

Local outputs include [raw regret curves](../outputs/recommendation_lcb_gsm8k_seed1_run0/simple_regret.png), [cumulative recommendation switches](../outputs/recommendation_lcb_gsm8k_seed1_run0/recommendation_switches.png), [recommended arms](../outputs/recommendation_lcb_gsm8k_seed1_run0/recommended_arm.png), [oscillation metrics and definitions](../outputs/recommendation_lcb_gsm8k_seed1_run0/oscillation_metrics.json), and [settings and summaries](../outputs/recommendation_lcb_gsm8k_seed1_run0/summary.json). Full NPZ histories include the recommended arm's mean, standard deviation, and observed-cell count. `outputs/` is ignored by Git.

## Twenty-seed general-prior aggregate

This experiment fixes the classic GSM8K matrix seed1 and repeats sampling seeds
0–19 with general prior `N(0.5, 0.04)`, B16, unit costs, cost scale `1e-4`, and a
10% budget. It is a 20-run comparison on one matrix; the full sweep configuration
also contains four other matrix seeds, other batch sizes, and other cost scales.

```bash
.venv/bin/python scripts/compare_recommendation_rules.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --dataset-tag gsm8k --prior-types default \
  --seeds 20 --seed-start 0 --batch-size 16 --budget-fraction 0.10 \
  --include-lcb --std-penalty 1 --cost-scaling-factor 1e-4 --grid-points 1025 \
  --out-dir outputs/recommendation_lcb_gsm8k_seed1_20seeds

.venv/bin/python scripts/analyze_recommendation_oscillation.py \
  --out-dir outputs/recommendation_lcb_gsm8k_seed1_20seeds --expected-seeds 20
```

GSM8K has 1,000 examples per arm, so B16 produces partial eight-cell batches
when an arm is exhausted. Different seeds therefore have different observation
times. The comparison script saves each raw history under `raw/`, then takes
the union of event times and carries each seed's latest completed state forward.
The aggregate ends at 12,200 evaluations; raw final batches can end at 12,208.
No interpolation, smoothing, or future observations enter the aggregate.
`source_evaluations` records when each carried state was actually observed.

Plots show the unsmoothed mean ± one standard error across the 20 sampling seeds.
The existing W&B aggregate plotting script instead uses linear interpolation and
defaults to standard-deviation bands; its visual presentation is different.
Budget-weighted mean regret integrates the post-pull step curve from the first
observation at 16 evaluations through 12,200, divided by that interval's length.
Raw per-seed oscillation counts and aggregate-curve variation are reported
separately. Seed0 exactly reproduces the earlier B16 general-prior experiment.

| Metric, full observed budget | Posterior mean | Posterior mean − std |
|---|---:|---:|
| Aggregate curve total variation | 4.9168 | 1.7984 |
| Aggregate upward regret jumps | 149 | 97 |
| Mean switches per seed | 84.0 | 45.4 |
| Mean total variation per seed | 5.1563 | 2.0099 |
| Budget-weighted mean regret | 0.037959 | 0.032880 |
| Seeds with zero regret at the budget endpoint | 20/20 | 20/20 |

The posterior-mean aggregate still oscillates strongly early in the budget.
LCB reduces aggregate total variation by 63.4%, mean per-seed total variation by
61.0%, and budget-weighted regret by 13.4%. Eighteen of 20 seeds improve on
budget-weighted regret; the paired mean difference is `-0.005079 ± 0.001337 SE`.
Nineteen of 20 seeds have fewer recommendation switches. After the first 1% of
matrix evaluations (1,220 cells), aggregate total variation falls from 1.19965
to 0.24355, a 79.7% reduction.

The first zero-regret recommendation occurs later on average with LCB
(2,891.6 → 3,412.8 evaluations), while the mean time after which each seed
stays at zero through this budget is earlier (5,857.6 → 5,116.0).
The aggregate stays at zero from 8,856 evaluations for posterior mean and 8,872
for LCB. These are different summaries: the aggregate is zero only when all
20 seeds have zero regret. The general-prior result therefore supports reduced
oscillation and improved average regret, with a tradeoff in when the best arm
is first recommended. Sampling is identical between the paired rules.

See [the aggregate and early-budget plots](../outputs/recommendation_lcb_gsm8k_seed1_20seeds/aggregate_simple_regret.png),
[paired regret differences](../outputs/recommendation_lcb_gsm8k_seed1_20seeds/aggregate_paired_regret_difference.png),
[per-seed cumulative oscillation](../outputs/recommendation_lcb_gsm8k_seed1_20seeds/aggregate_oscillation.png),
[all metrics](../outputs/recommendation_lcb_gsm8k_seed1_20seeds/oscillation_metrics.json),
and [configuration](../outputs/recommendation_lcb_gsm8k_seed1_20seeds/summary.json).
The three additional figures are also saved as PDFs. The output directory
contains all 20 raw NPZ histories, the aligned NPZ, CSV metrics, and
`provenance.json` with source hashes and validation details.

## Why the recovered historical plot differs

The recovered `simple_regret_gsm8k_various_models_seed1.png` uses an older experiment. Its saved metadata and the code archived at `c96b65c` show the following differences:

| Component | Historical figure | Current paired LCB experiment |
|---|---|---|
| Matrix / sampling seed | GSM8K matrix seed1 / run seed0 | Same matrix values and same seed |
| Prior | Data-specific `N(0.2, 0.01)` only | Separate general and data-specific panels |
| Recommendation | Post-pull empirical mean; exclude unobserved arms | Post-pull posterior mean or posterior mean minus std over all arms |
| Gittins batch size | 20 | 16 in the initial LCB comparison |
| Noise used with per-cell counts | `1/(4B) = 0.0125` at B20 | `tau_cell^2 = 0.25` |
| DP transition | One batch mean | One cell observation |
| DP cost | `1e-4` per batch transition | `1e-4` per cell transition |

The old posterior update treated the batch-mean variance as the noise of each cell. Commit `45e0814` (April 28, 2026, "fix the batch observation bug") corrects the scaling. At B20, prior variance 0.01, and 20 observations, the old posterior variance is `1/(100 + 20/0.0125) = 0.000588`, whereas the corrected model gives `1/(100 + 20/0.25) = 0.005556`. This affects Gittins sampling, independently of the recommendation rule. The old plotting script's empirical recommendation is visible in [the archived helper and simulation](../outputs/gsm8k_historical_audit/historical_plot_script.py); the archived policy is [here](../outputs/gsm8k_historical_audit/historical_gittins_policy.py).

The matrix arrays were verified equal across versions. Keeping seed0 also does not fix the sampled observations when batch size, posterior updates, or acquisition scores change. On the plots, 10% of the current horizontal axis corresponds to 12,200 evaluations on the old axis.

As a further check, the current code was rerun with B20 and the old data-specific prior. All trajectories below use the same matrix, sampling seed, prior, nominal budget, and batch size; the archived and current algorithms still differ as listed above:

| Trace | Mean regret per batch | Upward jumps | First zero regret | Zero regret through the rest of the run |
|---|---:|---:|---:|---:|
| Archived old empirical recommendation | 0.027372 | 9 | 6,400 evals | 6,560 evals |
| Current posterior mean, B20 | 0.025736 | 19 | 3,360 evals | 5,100 evals |
| Current posterior mean minus std, B20 | 0.026164 | 15 | 3,520 evals | 4,080 evals |

See [the aligned-batch overlay](../outputs/gsm8k_historical_audit/historical_vs_current_b20.png), [archived metadata](../outputs/gsm8k_historical_audit/historical_metadata.json), and [matrix verification](../outputs/gsm8k_historical_audit/matrix_check.json). These checks show that the old-vs-new difference includes changes to sampling and recommendation. The earlier LCB improvement claim concerns the paired curves within each current prior setting. It does not isolate LCB's effect on the historical algorithm. The archived metadata does not record the executable Git revision, so the code-level audit uses the source stored alongside the historical artifact rather than claiming an exact rerun of that old executable.

## W&B metadata

NPZ and W&B metadata record the coefficient and recommendation rule. The downloader retains them, and default plotting labels distinguish posterior mean from each positive standard-deviation penalty. Use a fresh directory when downloading with the extended CSV schema. The legacy expected-grid and targeted-recovery keys do not include the coefficient; those optional operations reject positive penalties rather than merge distinct configurations. Normal downloads support the new rule.
