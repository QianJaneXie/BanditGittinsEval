# GSM8K recommendation comparison — September 11, 2026

Uncertainty penalties improve recommendations early in some settings, but none
of these experiments supports a universally better replacement. The largest
early gain comes from avoiding recommendations of unobserved arms under the
optimistic default prior. A strict variance gate mainly delays making a choice.

## What was run

Five fixed GSM8K matrices × 20 sampling seeds × two Gaussian priors = **200
Gittins trajectories**. All **18 recommendation rules** see the same observations
within a trajectory. Each run reveals exactly **12,200 of 122,000 cells**,
with batches of up to 16, unit costs, and Gittins cost scale `1e-4`.

The priors are `N(0.5, 0.04)` (default) and `N(0.2, 0.01)` (GSM8K dataset prior).
Selection always uses the original Gittins index model. This experiment changes
recommendations only and does not tune the allocation in response to each rule.
See [README.md](README.md) and [manifest](results/gsm8k/manifest.json) for the full
prespecified parameter grid and provenance.

## At 1% of the full matrix

Here 1% means **1,220 evaluated cells**, one tenth of the maximum spending in this
experiment. Regret is measured in **accuracy percentage points**; lower is better.
Each prior column averages 100 runs. Coverage is the fraction of runs that make a
recommendation at that point.

| Rule | Default prior regret | Coverage | Dataset prior regret | Coverage |
|---|---:|---:|---:|---:|
| Current posterior mean | 16.877 | 100% | 6.819 | 100% |
| Empirical mean over observed arms | 11.260 | 100% | 7.738 | 100% |
| Variance gate `v/v0 <= 0.5` | 11.260 | 100% | 6.751 | 99% |
| Variance gate `v/v0 <= 0.1` | No recommendation | 0% | 1.200 | 32% |
| `mu - 5*v` | 11.260 | 100% | 6.779 | 100% |
| `mu - 1.645*sqrt(v)` | 11.260 | 100% | 6.750 | 100% |
| Finite-row Gaussian lower quantile, `z=1.645` | 11.260 | 100% | 6.602 | 100% |
| Anytime finite-population lower confidence bound | 11.260 | 100% | 6.815 | 100% |

The default-prior posterior-mean baseline recommends a completely **unobserved
arm in 33 of 100 runs** at this checkpoint. Those arms retain prior mean 0.5.
The uncertainty penalties avoid this particular failure and reduce mean regret
by **5.617 percentage points**. A sampling-seed-block bootstrap gives a paired
95% interval for the change of approximately **[-9.102, -2.379] pp**. This is an
exploratory, unadjusted interval conditional on the five fixed matrices.

The empirical baseline matches that early reduction. Thus this checkpoint does
not demonstrate that a sophisticated risk penalty is necessary: considering
only observed arms already solves the unobserved-arm problem here. Under the
dataset prior, the current baseline never recommends an unobserved arm at this
checkpoint, and empirical means perform worse than its shrinkage estimate.

The strict gate's dataset-prior regret of 1.200 pp is **conditional on its 32
covered runs**. On exactly those same runs, the current baseline has the same
regret: paired improvement is **zero**. Comparing 1.200 against the baseline's
all-run 6.819 would mistake selective abstention for improved decisions.

## Later budgets expose the conservatism tradeoff

At **2%** of the matrix, the default-prior baseline has mean regret **2.137 pp**.
The standard-deviation penalty (`z=1.645`) has **2.181 pp**, and the anytime
confidence method has **2.446 pp**. Under the dataset prior, the corresponding
values are **1.064, 1.072, and 1.072 pp**. Conservative rules can retain a well
measured incumbent when a less measured challenger would be a better choice.

The strict `v/v0 <= 0.1` gate covers 96% of default-prior runs and 94% of
dataset-prior runs at 2%. Its default-prior conditional regret is **1.965 pp**,
but its matched difference against the baseline is **+0.211 pp**, not an
improvement. On dataset-prior covered runs the paired difference remains zero.

By **10%**, all 18 rules recommend the same best configuration in every
default-prior run (**100/100**, zero regret), and in **99/100** dataset-prior
runs (mean regret **0.004 pp**). The remaining run misses the optimum by 0.4 pp.
These endpoint results leave almost no room for recommendation-only gains;
the useful differences are earlier in the learning curves.

## Which methods merit further evaluation?

- A **mild Gaussian lower quantile** is a useful candidate when a recommendation
  must always be supplied. `z=1.645` is an interpretable marginal posterior
  quantile. The finite-row version targets the benchmark's actual row average
  and has zero uncertainty after an entire row is observed. Its modest early
  dataset-prior gain does not persist uniformly over budgets.
- The **variance gate** is appropriate when abstention is acceptable. Under the
  current fixed-variance model it is simply a minimum-observation threshold;
  it is not a guarantee of correctness. Report coverage with every result.
- The **anytime finite-population method** has a confidence interpretation that
  tolerates adaptive repeated sampling and does not use the Gaussian prior.
  It is conservative and sometimes gives worse regret. A valid lower bound on
  an arm's quality does not establish that it beats every competitor.
- **Mean minus variance** also has a principled interpretation: under a
  Gaussian posterior it maximizes exponential expected utility with a chosen
  risk aversion. Its coefficient represents a preference, not a confidence level.

No production recommendation rule has been switched. Parameter choice still
needs validation on other matrices, priors, batch sizes, and evaluation budgets.
The alternative priors change Gittins allocation as well as posterior estimates,
so between-prior differences cannot be attributed to recommendation alone.

## Figures and result files

- [Overview](figures/gsm8k/overview.png) · [PDF](figures/gsm8k/overview.pdf)
- [Every individual method and family figure](FIGURES.md)
- [Endpoint summary CSV](figures/gsm8k/summary.csv)
- [1%, 2%, 5%, and 10% checkpoints CSV](figures/gsm8k/budget_checkpoints.csv)
- [Per-run endpoint metrics CSV](figures/gsm8k/per_run_metrics.csv)
- [Raw trajectories](results/gsm8k/traces/) · [Validation results](results/gsm8k/validation.json)

Curves use the last available observation state at each grid point, without
looking ahead or treating abstention as zero regret. Shaded 95% intervals use
1,000 bootstrap draws of 20 sampling-seed blocks, retaining all five matrices
inside each block. They quantify Monte Carlo variability on these fixed inputs,
not uncertainty across new benchmarks. Multiple methods and budgets have not
been adjusted for selection or multiplicity.

## Verification

- All 12 unit tests passed, including an exhaustive small binary-population
  confidence-coverage check and abstention/complete-row boundary cases.
- Independently reconstructed all **153,304 post-pull states** from selected
  cell indices across 200 runs; checked unique observations, exact budgets,
  counts/sums, selected-arm counts, and regret against the full-matrix oracle.
- The runner verified 128 native-policy pulls under each prior. An independent
  audit additionally replayed the **entire 766- and 767-pull first trajectories**
  under the default and dataset priors. Arm choices, sampled examples, and
  baseline recommendations matched the original implementation, with the
  deliberate final-budget truncation accounted for.
- Source and input hashes matched. No saved confidence interval missed its
  finite-row truth in these runs. This observation is not proof of coverage.
  No run produced a zero-regret certificate from the simultaneous intervals.

For the theory and source references, see [STATISTICAL_NOTES.md](STATISTICAL_NOTES.md).
