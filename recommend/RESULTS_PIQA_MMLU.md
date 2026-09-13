# PIQA and MMLU recommendation comparison — September 13, 2026

**LCB does not consistently beat posterior-mean recommendation on these datasets.**
The largest clear improvement at the recorded checkpoints is Computer Security
with the dataset prior at 5% of evaluations. PIQA is slightly worse early,
Anatomy's changes are small and uncertain, and Business Ethics mostly ties.
The large early default-prior improvement seen on GSM8K does not repeat here.

[All method figures](FIGURES_PIQA_MMLU.md) ·
[LCB comparison](figures/piqa_mmlu/cross_dataset_lcb.png) ·
[Paired differences](figures/piqa_mmlu/cross_dataset_lcb_effect.png) ·
[Every method at every checkpoint](figures/piqa_mmlu/all_methods_checkpoints.csv)

## Experiment

All 18 rules and parameter values from GSM8K were retained. The primary
comparison was fixed before running: **posterior mean versus
`mu - 1.645*sqrt(v)`**, the Gaussian lower posterior quantile for the latent arm
mean. This Gaussian LCB does not have the simultaneous frequentist guarantee
of the separately evaluated finite-population confidence method.

| Dataset | Matrix dimensions (arms × examples) | Matrices | Sampling seeds per prior | Dataset prior | Batch | Exact final cell budget |
|---|---:|---:|---:|---|---:|---:|
| PIQA | 103 × 1,000 | 5 | 20 per matrix | N(0.4, 0.02) | 16 | 10,300 |
| MMLU Computer Security | 1,500 × 100 | 1 | 20 | N(0.75, 0.01) | 4 | 15,000 |
| MMLU Anatomy | 1,500 × 135 | 1 | 20 | N(0.6, 0.02) | 4 | 20,250 |
| MMLU Business Ethics | 1,500 × 100 | 1 | 20 | N(0.6, 0.02) | 4 | 15,000 |

Every dataset also uses the default prior N(0.5, 0.04), giving **320 trajectories**:
200 PIQA and 40 for each MMLU subject. The second Normal parameter is variance.
MMLU priors come from the existing task metadata buckets; no prior or rule was
retuned after inspecting this comparison. MMLU's `matrix_seed=1` in filenames is
an identifier for its one fixed matrix, not an independent dataset replication.

All methods see identical observations within each trajectory and never affect
which arm is evaluated. Cells are sampled uniformly without replacement within
the chosen arm. The allocation uses the same Gaussian Gittins equations,
per-cell observation variance 0.25, cost scale 1e-4, and 1,025 DP grid points.
Runs continue to exactly 10% of the matrix, without diagnostic early stopping.
Each intermediate checkpoint uses the latest completed batch at or before its
budget; it never includes future observations. For example, PIQA's 1% checkpoint
is the latest completed batch at or before 1,030 cells.

These are offline replays of saved matrices, not new LLM evaluations. Full-row
accuracy is used only for oracle regret scoring. A 10% matrix budget averages
100 observations per PIQA arm but only 10 or 13.5 per MMLU arm; the comparisons
within each dataset are paired, but fractions do not imply equal precision
across datasets.

## Primary results

Each cell is **posterior-mean regret → LCB regret**, in accuracy percentage
points. Lower is better. Both rules have 100% recommendation coverage.

| Dataset | Prior | At 1% | At 5% | At 10% |
|---|---|---:|---:|---:|
| PIQA | Default | 3.422 → 3.651 | 0.443 → 0.392 | 0.024 → 0.024 |
| PIQA | Dataset | 3.098 → 3.211 | 0.701 → 0.670 | 0.085 → 0.043 |
| Computer Security | Default | 4.550 → 4.550 | 4.050 → 4.100 | 3.700 → 3.800 |
| Computer Security | Dataset | 4.700 → 4.700 | 4.050 → 3.300 | 2.500 → 2.350 |
| Anatomy | Default | 4.370 → 4.296 | 3.815 → 3.852 | 1.815 → 1.741 |
| Anatomy | Dataset | 3.778 → 3.444 | 3.185 → 3.111 | 1.926 → 1.889 |
| Business Ethics | Default | 1.450 → 1.450 | 1.500 → 1.500 | 0.550 → 0.550 |
| Business Ethics | Dataset | 0.700 → 0.550 | 0.600 → 0.600 | 0.400 → 0.400 |

The CSV also includes 2% budgets, 90th-percentile regret, selected-arm sample
counts, coverage, and paired confidence intervals for all 18 methods.

### PIQA: no broad LCB advantage

At 1%, LCB increases mean regret by 0.229 pp under the default prior and 0.113 pp
under the dataset prior. Their paired 95% bootstrap intervals are [0.000, 0.654]
and [-0.114, 0.424] pp, respectively. At 5%, the small improvements are 0.051 and
0.031 pp, with intervals including zero. By 10%, the default prior ties; the
dataset prior improves by 0.042 pp, with interval [-0.126, 0.000]. These estimates
do not establish a reliable, general advantage.

### Computer Security: useful with the dataset prior at intermediate budgets

At 5%, the dataset-prior mean regret drops **4.05 → 3.30 pp**, a paired change
of **-0.75 pp**, with 95% interval **[-1.55, -0.10] pp**. The 90th-percentile
regret also drops **7.10 → 5.10 pp**. This supports better average and upper-tail
performance in this particular setting.

The gain shrinks at 10%: **2.50 → 2.35 pp**, with paired interval [-0.45, 0.00].
Under the default prior, LCB offers no benefit at 1% or 2% and is slightly worse
at 5% and 10%. A claim about Computer Security must therefore specify the prior
and evaluation budget.

### Anatomy: small mean gains do not establish greater robustness

With the dataset prior at 1%, mean regret drops **3.778 → 3.444 pp** and the
90th percentile drops **7.704 → 5.037 pp**. The paired mean-change interval
is **[-0.889, 0.074] pp**, so the improvement is uncertain with these 20 seeds.

At 10%, the dataset-prior mean changes only **1.926 → 1.889 pp**, while its
90th-percentile regret actually rises **2.296 → 2.963 pp**. Lower average regret
alone would give an incomplete account of robustness.

### Business Ethics: mostly the same decisions' accuracy

The default-prior regret ties at all four recorded checkpoints. The dataset
prior has a small early gain at 1%, **0.70 → 0.55 pp**, with interval
[-0.45, 0.00] pp, and ties at 2%, 5%, and 10%. There is little support here for
replacing posterior mean solely to improve average recommendation accuracy.

## What happened to the other recommendation rules?

- **Finite-row lower bounds merit further evaluation on short rows.** On
  Computer Security with its dataset prior at 5%, finite-row Gaussian LCB
  (`z=1.645`) has 3.05 pp mean regret and the anytime finite-population LCB has
  3.00 pp, versus 4.05 pp for posterior mean and 3.30 pp for latent Gaussian
  LCB. On Business Ethics with its dataset prior at 10%, both finite-row rules
  achieve 0.15 pp versus 0.40 pp for posterior mean and latent LCB. That latter
  paired improvement has interval [-0.551, 0.050] pp, which includes zero.
  These are secondary exploratory comparisons, not a new universally best rule.
- **An observed sample mean can be much worse.** At 5% on Computer Security,
  dataset-prior sample-mean recommendation has 7.45 pp regret versus 4.05 pp
  for posterior mean; on Anatomy, 5.889 versus 3.185 pp. Retaining shrinkage is
  useful in these cases. Avoiding unobserved arms alone does not solve noisy
  ranking among sampled arms.
- **Variance gates often cannot make a recommendation.** For `v/v0 <= 0.1`,
  all MMLU settings have 0% coverage at 1%. At 10%, coverage is only 25% on
  default-prior Computer Security and still 0% on dataset-prior Computer
  Security and Business Ethics. Their apparently missing regret is abstention,
  not perfect performance. PIQA's dataset-prior gate covers only 48% at 1%.
- **Mean–variance penalties show a similar dependence on the setting.** For
  example, `mu - 5*v` matches latent LCB's 3.30 pp on dataset-prior Computer
  Security at 5%, but ties the posterior-mean baseline at several other
  checkpoints. All four penalty strengths are plotted rather than selecting
  the best-looking strength after the fact.

Under this fixed-noise Gaussian model, a variance gate is exactly a sample-count
threshold: `n >= ceil((0.25/v0)*(1/rho - 1))`. For rho=0.1 this requires
57 observations with the default prior, 225 for Computer Security's dataset
prior, and 113 for the Anatomy/Business Ethics dataset prior. A 100-question
row cannot satisfy the last two requirements, even if completely evaluated.
The thresholds were retained to expose this limitation.

## Interpretation and limits

At the 1% checkpoint the posterior-mean baseline recommends an unobserved arm
in **0% of runs in all eight new dataset/prior settings**. This differs from
33% on default-prior GSM8K and helps explain why its large gain did not repeat.
That is a diagnostic observation, not a causal decomposition of all differences.

The defensible conclusion is: **uncertainty penalties can help with noisy
recommendations, but improvement depends on the prior, budget, dataset, and
uncertainty target. The new results do not justify a blanket claim that latent
Gaussian LCB is more robust.** Finite-row methods have encouraging results on
some short MMLU rows, while hard variance gates can remain silent indefinitely.

Bootstrap intervals use 1,000 resamples of the 20 sampling-seed blocks, retaining
pairing across PIQA matrices. They quantify Monte Carlo variation conditional
on the fixed matrices. They are pointwise, unadjusted for the many methods and
checkpoints, and do not quantify performance on new questions, model calls,
subjects, or configurations. A 90th percentile from only 20 MMLU seeds is also
imprecise; reported changes should not be treated as precise tail guarantees.

## Numerical backend and verification

Windows Application Control blocked the previously installed Torch/JAX
binaries, including an approved attempt outside the sandbox. These runs used
the bundled Python's NumPy 2.3.5 with SciPy 1.18.1, a portable float64
implementation of the same m-diff Gittins DP, and **PCG64 sampling instead of
Torch sampling**. Original GSM8K sources and artifacts were preserved.

The portable DP passed an analytic Gaussian-convolution check. Compared with
the saved native GSM8K tables, 11/1,001 default-prior roots and 13/1,001
dataset-prior roots differed by more than 1e-6; all differences were within
one grid spacing plus numerical tolerance. The portable tables made the same
allocation choices on all **766 + 767 saved reference states**. This is useful
parity evidence, not a claim of bitwise equivalence or a new native execution.
See [portable validation](results/portable_validation.json).

All **320 runs, 632,112 post-pull states, and 11,378,016 recommendation decisions**
passed validation. Independent reconstruction checked source/input hashes,
complete seed grids, exact budgets, unique cells, counts/sums, full-matrix
regrets, all 18 rule choices/scores/eligibility counts, confidence intervals,
regret certificates, allocation decisions, and the exact sampled columns.
The 12 recommendation unit tests also passed.

There were no observed finite-population confidence-coverage violations, and
no final best-arm certificates. The absence of violations is a diagnostic;
the formal coverage claim depends on the assumptions in
[STATISTICAL_NOTES.md](STATISTICAL_NOTES.md).
