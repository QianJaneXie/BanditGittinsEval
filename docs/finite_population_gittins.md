# Gittins index for the full fixed test-set mean

Gittins acquisition, natural stopping, and recommendation now use the same target: the realized mean of all N examples in an arm's fixed test set. A completed arm's index is its exact empirical row mean. An unfinished arm's index includes the value of revealing its remaining examples under this target. The numerical `cost_scaling_factor` is unchanged.

This changes the acquisition DP as well as its current state. It can change sampled arms and stopping times even with `--recommendation-std-penalty 0`. The earlier recommendation-only change in commit `f4833e4` did not make this acquisition change.

## Posterior state and transition

For n observed examples with sum S, let the latent posterior have mean mu and variance v, with prior mean mu0, prior variance v0, and per-cell noise variance tau_cell^2. Write m = N - n. The finite test-set mean F has

\[
M = E[F\mid D] = \frac{S+m\mu}{N},\qquad
V_F = \frac{m^2v+m\tau_{\mathrm{cell}}^2}{N^2}.
\]

With shared N, prior, and noise, define

\[
a = 1+\frac{\tau_{\mathrm{cell}}^2}{Nv_0},\qquad
b=(1-a)\mu_0.
\]

Then `M = a*mu + b` at every observation count. For a batch of q fresh cells, the latent posterior mean's transition variance is `v^2 / (v + tau_cell^2/q)`. The finite mean's transition standard deviation is **a times** the latent transition standard deviation. The DP uses this transition uncertainty, not `sqrt(V_F)`: total remaining uncertainty and uncertainty resolved by one transition are different quantities.

At completion, `M = S/N` and `V_F = 0`; there is no future observation or exploration bonus. Partial final batches use their actual number of cells when computing transition noise.

## Cost and index

The existing random-walk DP retains its form:

\[
Q_t(s)=E[\max\{Q_{t+1}(s+\sigma^F_t Z),0\}]-c_t,
\qquad Q_N(s)=s,
\qquad \Gamma^F_t=M_t-r_t,
\]

where `Q_t(r_t)=0` and `sigma_F = a*sigma_latent`. The implementation recomputes roots with these scaled transitions and the original numerical costs, across lookup, per-observation DP, and batch-mean DP paths. A cost is still charged per transition of the selected DP mode; this change does not redefine the batching or cost model.

An equivalent identity for the exact DP is

\[
\Gamma^F(c)=a\,\Gamma^{\mathrm{latent}}(c/a)+b.
\]

Here `c/a` scales the entire remaining cost schedule. Simply transforming the old index as `a*Gamma_latent(c)+b` would also scale the finite-target cost to `a*c`. That is a different choice. We keep the configured cost fixed, so there is no claim of invariant acquisition rankings. Finite-grid numerical calculations approximate these identities.

For GSM8K with N=1,000 and tau_cell^2=0.25, a is 1.00625 under the general prior and 1.025 under the data-specific prior. These factors are close to one, but index comparisons can still change when candidates are close.

## Stopping, recommendation, and metadata

Natural stopping occurs when the largest index belongs to a completed arm. Both completed and unfinished indices now describe F. The LCB-aligned stopping rule instead compares the best unfinished finite-target index with the selected score `M - lambda*sqrt(V_F)`, so it can stop before any arm is complete. Fixed-budget runners record both first stopping times and continue to their evaluation budget. The older crossing against the selected arm's unpenalized M remains logged only for compatibility with existing results.

The default recommendation maximizes M. `--recommendation-std-penalty 1` instead maximizes `M - sqrt(V_F)`. The same coefficient is used by the LCB-aligned stopping threshold. LCB affects recommendation and the LCB-aligned stopping rule; it does not enter acquisition or natural stopping. Correcting the target does not eliminate optimistic recommendations for unobserved arms: their M remains the prior mean.

The low-level latent posterior and transition helpers retain their latent meaning. `gittins_index_exploration` and `gittins_post_pull_update` return finite-target means; `posterior_mean_pulled` consequently records M. Precomputed roots supplied to the policy must use the finite-target transition schedule. `compute_finite_population_roots_lookup_table` provides this construction and shares a single lookup table when all arm costs are identical.

New runs record `gittins_index_target = finite_population_mean` in addition to the existing recommendation-target and penalty fields. Historical runs without this field are treated as `latent_mean`. Plotting and download metadata distinguish these cases. The rollout script rejects historical latent-target traces when asked to re-simulate Gittins with the current policy; their recorded histories remain historical data.

## GSM8K verification

The paired comparison uses only sampling seed 0, both the general `Normal(0.5, 0.04)` and data-specific `Normal(0.2, 0.01)` priors, matrix `gsm8k_1_samples_various_models_seed1.npy`, batch size 16, unit costs, cost scale `1e-4`, grid size 1,025, and a 12,200-cell budget. The historical acquisition implementation is read from commit `f4833e4`; the new implementation uses the finite target at the same numerical cost. Each acquisition trajectory is scored with finite mean and finite LCB recommendations on identical observed data.

```bash
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/compare_gittins_targets.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --old-revision f4833e4 --seed 0 --prior-types default dataset \
  --batch-size 16 --budget 12200 --tau-sq-cell 0.25 \
  --cost-scaling-factor 1e-4 --grid-points 1025 \
  --out-dir outputs/unified_finite_gittins_gsm8k_seed1_run0
```

Use a fresh output directory for another invocation. The script archives both implementations, hashes the sources and matrix, and runs them in separate interpreters. The historical control exactly reproduces the previously saved evaluations, pulled arms, recommendations, regrets, and recommended-arm observation counts under both priors.

### Results at sampling seed 0

The sampling trajectories change: the first different batch ends at 2,304 evaluations under the general prior and at 5,352 under the data-specific prior. The following comparisons use finite-target recommendation on both old and new acquisition trajectories. Budget-weighted mean regret integrates the unsmoothed post-pull step curve from 16 through 12,200 evaluations.

| Prior | Acquisition | Recommendation | Switches | Upward regret jumps | Regret total variation | Budget-weighted mean regret |
|---|---|---|---:|---:|---:|---:|
| General | Old latent | Finite mean | 141 | 65 | 10.632 | 0.04873014 |
| General | Unified finite | Finite mean | 139 | 64 | 10.636 | 0.04824163 |
| General | Old latent | Finite LCB | 48 | 22 | 2.210 | 0.03018385 |
| General | Unified finite | Finite LCB | 38 | 17 | 2.102 | 0.02902692 |
| Data-specific | Old latent | Finite mean | 41 | 18 | 1.222 | 0.03117663 |
| Data-specific | Unified finite | Finite mean | 49 | 22 | 1.302 | 0.03128168 |
| Data-specific | Old latent | Finite LCB | 41 | 18 | 1.058 | 0.03154563 |
| Data-specific | Unified finite | Finite LCB | 37 | 16 | 1.018 | 0.03176888 |

All eight curves have zero regret at the budget endpoint. The new data-specific run completes a final batch at 12,208 cells; analysis excludes this overshooting observation and carries the state at 12,192 through the requested endpoint at 12,200. The other three acquisition trajectories finish exactly at 12,200. No smoothing, seed averaging, or future-state interpolation is used.

| Prior | Natural stop, old → unified | Recommendation-aware stop, old → unified |
|---|---:|---:|
| General | 4,360 → 3,528 | 2,096 → 2,096 |
| Data-specific | 2,984 → 2,984 | 576 → 576 |

These are first post-pull recorded crossings, not actual truncations of the fixed-budget experiments. For this seed, mean and LCB give the same first legacy recommendation-aware crossing within each acquisition/prior setting. That equality is an observed result, not a guarantee for other trajectories.

### LCB-aligned stopping pilot

On the current finite-target acquisition trajectories, replacing the selected raw mean in the stopping threshold with the selected finite-population LCB gives:

| Prior | Natural stop | LCB-aligned stop | True simple regret at LCB stop |
|---|---:|---:|---:|
| General `N(0.5, 0.04)` | 3,528 | 2,608 | 0 |
| Data-specific `N(0.2, 0.01)` | 2,984 | 2,064 | 0.024 |

Thus the LCB-aligned rule stops 920 evaluations before natural stopping in both seed-0 trajectories, without requiring a completed arm. These are single-matrix, single-sampling-seed results; the rule does not certify zero realized regret, as the data-specific row shows. The condition uses a strict inequality and records its first post-pull crossing even if continued fixed-budget sampling would later reverse it.

Simulation traces save this stopping time as `gittins_lcb_aligned_stop_cum_eval` and `gittins_lcb_aligned_stop_cum_original_cost`; W&B summaries and downloads use the same names. Existing natural-stop and legacy raw-mean fields retain their prior meanings.

The unified target does not remove general-prior recommendation oscillation. With the new index, LCB reduces switches from 139 to 38 and budget-weighted regret from 0.04824163 to 0.02902692. Under the data-specific prior it reduces switches from 49 to 37, while slightly increasing budget-weighted regret from 0.03128168 to 0.03176888. The default remains mean-only, with LCB available explicitly. Existing results are not interchangeable with the unified acquisition: even without LCB, both priors produce different sampling trajectories in this check.

See [the old/new regret curves](../outputs/unified_finite_gittins_gsm8k_seed1_run0/simple_regret.png), [recommendation switches](../outputs/unified_finite_gittins_gsm8k_seed1_run0/recommendation_switches.png), and [metrics, stopping times, source hashes, and reproduction audits](../outputs/unified_finite_gittins_gsm8k_seed1_run0/summary.json). Both figures are also saved as PDFs. These local artifacts are ignored by Git; the comparison script and this result record are committed.

## Validation

Eight finite-acquisition regression tests cover the remaining-variance identity, partial batches, the fixed-cost affine identity, nonuniform costs, lookup/direct-DP agreement, batch-noise conversion, exact completed indices, early stopping, and independence from the LCB coefficient. The 19 recommendation/stopping, two budget-analysis, and eight metadata tests also pass. Integration checks run the actual simulator on a small matrix through completion and the paired GSM8K comparison verifies that the production LCB-aligned holder reproduces the offline first-crossing calculation.
