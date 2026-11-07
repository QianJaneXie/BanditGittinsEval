# Gaussian TS and item-subsampling MF-KG

These baselines address the human ICLR reviews: Mvv2 requests item-subsampling
multi-fidelity BO, and dNrr requests Thompson sampling. They operate on the
supplied accuracy matrices without LLM API calls. Full experimental sweeps have
not been run as part of implementation.

## Gaussian Thompson sampling

TS uses the same independent Gaussian observation model and finite-population
posterior as the current Gittins implementation. Each step samples a complete
test-set mean for each unfinished arm, chooses the largest draw, and reveals
unseen cells from that arm. Completed arms remain eligible for recommendation.
The recommendation maximizes the complete-test-set posterior mean. TS has no
separate warmup.

By default, `--ts-batch-size`, `--ts-prior-mean`, and `--ts-prior-variance`
inherit the corresponding Gittins settings. `--ts-obs-noise-variance` is the
nominal batch-mean variance; it inherits an explicit Gittins noise override,
otherwise defaults to `1/(4 * ts_batch_size)`. The equivalent per-cell variance
is always the batch variance times the nominal batch size, even on a partial
last batch.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/simulate_simple_regret.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --out outputs/ts/gsm8k_seed1.npz --algorithms ts \
  --seed 0 --eval-budget-fraction 0.1 \
  --gittins-batch-size 16 \
  --gittins-prior-mean 0.2 --gittins-prior-variance 0.01
```

To compare all matrix-bandit methods in one archive, use
`--algorithms ucb sysrs gittins ts`. Set `--batch-size 16` as well to match UCB's
batch size. Match priors, per-cell variance, and recommendation rules when
isolating allocation effects. The Gittins mean-minus-standard-deviation
recommendation is a separate ablation; TS recommends the posterior mean.

With a per-cell `--cost-vector`, TS uses the same sampling rule and a cost
stopping budget. It does not divide sampled accuracy by cost, which would
optimize a different quantity. Like the existing cost-budgeted bandit runners,
stopping occurs after a batch and may exceed the threshold by one batch; use
the recorded actual cumulative cost when comparing curves. The unit-cost last
TS batch is clipped to the remaining cell budget.

The bundled BO archives can also be passed as `--cost-vector`; their full costs
are divided by metadata `n_examples` and reordered by `arm_ids`, without reading
score arrays. For a runnable GSM8K cost experiment, add this to the TS command:

```bash
--cost-vector data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz
```

Use a separate output filename for the cost run.

W&B variants include `ts_unit_B16_dataset`, `ts_cost_B16_dataset`, `ts_B16`, and
`ts_cost_B16`. Dataset-prior variants use the existing prior-resolution logic;
explicit `--ts-prior-mean`, `--ts-prior-variance`, and
`--ts-obs-noise-variance` override it. Start from
`scripts/config/GSM8KThompsonSamplingSweep.yml`. It contains unit and cost runs
using prices from the supplied BO archive. Prices and arm ordering were checked
to be identical across its five GSM8K matrix seeds.

## MF-KG objective and observations

For each arm, draw a seeded permutation of its benchmark items before observing
any scores. Fidelity `s = n/N` reveals the prefix of `n` items. Define the
deterministic lower-fidelity objective as that prefix's mean; fidelity 1 is the
complete row mean. Promotion reuses the prefix and reveals only its new suffix.
Initialization, anchors, and all promotions appear in the cumulative cell and
cost traces. A promotion costs `(n_new - n_old) * per_cell_cost[arm]`.

The surrogate is BoTorch `MixedSingleTaskGP`, with categorical configuration
features and a continuous fidelity column. This preserves the shared
model/prompt/decoding structure in the existing BO inputs. Without
`--bo-inputs`, a categorical arm identity is used, which loses that structure.
The BO archive's `X` and `arm_ids` are used, and full-evaluation `cost` is divided
by `N`. `Y` and `raw_scores_mean` are never accessed. Use an archive aligned to
the supplied matrix; shape checks cannot detect a different matrix with the
same dimensions.

Overlapping prefix means are modeled as correlated deterministic function
values across fidelity. `--observation-noise` is a small numerical nugget, not
an independent observation-variance formula for overlapping averages. The
learned fidelity kernel approximates their relationship; it is not the exact
finite-population covariance model. Prefix curves need not be monotone.

Acquisition uses BoTorch `qMultiFidelityKnowledgeGradient` with `q=1`, seeded
Sobol fantasies, the current best posterior mean at full fidelity, and gain
divided by known marginal evaluation cost. Both the candidate and terminal
optimizations enumerate real configurations: a custom terminal value function
computes the exact maximum of fantasy posterior means over the finite arm set
at fidelity 1. No fractional categorical configurations are optimized. The
numerical GP nugget remains an approximation, including at observed full
fidelity. Recommendation uses posterior means at fidelity 1 and the exact
known mean of any completed arm.

For the fixed-noise Gaussian surrogate, the default score evaluation uses the
exact Gaussian conditional-mean formula for each fantasy. It retains every
candidate and fantasy but avoids constructing a separate fantasy GP to query
each terminal mean. `score_mfkg_candidates_native` (or
`use_analytic_conditioning=False`) retains the native BoTorch evaluation as a
reference. Tests compare the two evaluations, including outcome standardization
and heterogeneous observation noise; agreement is to numerical precision.

## Default: one query with adaptive fidelity

The default is **`q=1 + adaptive_fidelity`**. Each round enumerates all
configuration/fidelity pairs whose grid level is higher than that
configuration's observed count and whose newly revealed suffix fits the
remaining budget. MF-KG chooses one pair by expected full-fidelity gain per
incremental evaluation cost. It can jump across several fidelity levels, so
the number of new benchmark items varies by round. All new items and their
actual costs, including warmup, are included in the trace.

The default fidelity fractions are `0.016 0.032 0.064 0.128 0.25 0.5 1.0`.
Each is converted to `ceil(s * N)` items; duplicate counts are merged and full
fidelity is always included. For GSM8K with N=1,000, the resulting counts are
`16 32 64 128 250 500 1000`. Override the grid with `--fidelities ...` or
absolute counts, for example `--fidelity-samples 16 32 64 128 256 512 1000`.

`q=1` counts one `(configuration, fidelity)` query returning one cumulative
prefix mean. For example, advancing configuration A from 16 to 64 observed
items is one query revealing 48 new items. `--num-fantasies 64` controls
internal Monte Carlo precision; `--candidate-batch-size` and
`--target-batch-size` control computational chunking. These parameters do not
specify the number of benchmark items revealed per round.

The earlier fixed-step option remains available for compatibility through
`--allocation-mode fixed_batch --batch-size B`. It constrains each ordinary
query to B new items and rejects fidelity-grid options. The main examples and
MF-KG sweep use adaptive fidelity.

The deterministic-cost utility explicitly computes expected gain divided by
cost, including finite-Monte-Carlo negative fantasy deltas, to retain BoTorch
0.9.5 semantics across versions. Chunking bounds memory without pruning arms
or altering the terminal maximization.

This follows the [BoTorch discrete-fidelity KG setup](https://botorch.org/docs/v0.17.0/tutorials/discrete_multi_fidelity_bo),
with a mixed categorical surrogate and exact finite-configuration maximization
for this task. It does not use a single-fidelity LogEI score renamed as MF-KG.

## Warmup

The default warmup spends at most **10% of the experiment's budget**, including
anchors. With a 10% experiment budget, this is at most 1% of exhaustive cost.

1. Greedily seek coverage of category levels, with seeded random tie-breaking,
   at the lowest fidelity. The requested initial arm count defaults to the
   largest category-level count across feature dimensions, or 8 with arm
   identity features. The cost cap can truncate this coverage.
2. Randomly choose one of those arms, independently of its scores, and promote
   it to the highest affordable grid fidelity. Full fidelity is
   preferred when it fits the remaining warmup cap.
3. Let MF-KG select subsequent configurations and fidelity jumps. All candidate
   actions must fit the remaining experiment budget; there is no MF-KG budget
   overshoot. The fidelity grid can leave an unspendable remainder.

For GSM8K with 122 arms and 1,000 items, the default lowest fidelity is 16 items.
At a 12,200-cell experiment budget, 11 initial arms cost 176 cells. Promoting
one to 1,000 costs another 984, giving **1,160 cells**, within the 1,220-cell
warmup cap. Cost-aware runs use the same rules in actual cost units and can
choose a lower anchor fidelity.

This remains a cap on the existing coverage-and-anchor plan. Increasing
`--warmup-budget-fraction` to `0.5` allows at most 5% of exhaustive cost when
the experiment budget is 10%; it does not require the plan to spend that
amount. A matched 5%-of-exhaustive warmup target is a separate initialization
protocol.

Use `--n-init` to override the coverage count,
`--warmup-budget-fraction 0.05` for a smaller cap,
`--warmup-anchors 0` for the low-fidelity-only ablation, or
`--warmup-anchors 2` for two anchors. `--warmup-strategy random` replaces greedy
coverage with random initialization. If the cap cannot pay for even one
observation, the runner reports an error. The first grid count determines the
minimum observation size.

## Run and plot MF-KG

Install the project with the optional experiment and test dependencies:

```bash
python3 -m pip install -e '.[experiments,test]'
```

The environment was verified with BoTorch 0.9.5, GPyTorch 1.11, and PyTorch 2.9.1.
MF-KG additionally uses NumPy and SciPy; W&B is imported only when logging is
enabled. On this Mac, the existing duplicate OpenMP setup needs the same
`KMP_DUPLICATE_LIB_OK=TRUE` environment setting used by the other BO examples.

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/run_mfkg_baseline.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --bo-inputs data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz \
  --out outputs/mfkg/gsm8k_seed1_unit.npz \
  --seed 0 --eval-budget-fraction 0.1 \
  --allocation-mode adaptive_fidelity \
  --warmup-budget-fraction 0.1 --warmup-anchors 1 --num-fantasies 64
```

Add `--cost-aware` to use the BO archive's prices for both KG allocation and
the stopping budget, and give the run a separate output path. Alternatively,
`--cost-vector` supplies per-cell costs and enables cost-aware operation.
`--experiment-variant mfkg_unit` / `mfkg_cost` work in W&B sweeps.

For MMLU abstract algebra, substitute
`data/MMLU_matrices/abstract_algebra.npy`,
`data/bo_inputs/mmlu/abstract_algebra_bo_inputs.npz`, and
`--fidelity-samples 4 8 16 32 64 100`. MF-KG chooses its fidelity jump and does
not require the same per-round item count as TS/Gittins/UCB.

The trace has both BO-style `x`, `regret`, etc. and plot-compatible `mfkg_*`
arrays, plus fidelity, phase, new-sample counts, training data, prefix
permutations, allocation mode, and timing diagnostics. A companion JSON records settings and
summary metrics. `--out` defaults to a matrix/variant/seed-specific name under
`outputs/mfkg/`; choose separate paths when changing other settings.

```bash
python3 scripts/plot_simple_regret_results.py \
  --traces outputs/ts/gsm8k_seed1.npz \
  --mfkg-trace outputs/mfkg/gsm8k_seed1_unit.npz \
  --out outputs/figures/ts_mfkg_gsm8k.png --x-axis evals
```

Both plotting scripts support `--mfkg-trace` and retain its warmup points.
`plot_bandit_bo_comparison.py` also accepts comparisons without a PBGI trace.
Use `--x-axis original_cost` for cost comparisons.

MF-KG is substantially more expensive computationally than independent-arm
TS: it refits a GP and enumerates candidate queries and fantasy terminal arms.
Adaptive fidelity can have several candidate grid levels per unfinished
configuration. `--candidate-batch-size` and
`--target-batch-size` control memory;
`--num-fantasies` controls Monte Carlo precision. `--max-steps 2
--num-fantasies 4 --fit-maxiter 30` is a smoke-test setting, not the experimental
default. Iteration-limited GP fits emit a warning and retain their parameters;
increase `--fit-maxiter` to check convergence for reported experiments.

For W&B, use `--wandb-mode online` / `offline`. The runner defaults to disabled
logging. `scripts/config/GSM8KMFKGSweep.yml` contains a seed-1 pilot with both
cost modes, `q=1` adaptive-fidelity allocation, and 20 sampling seeds. It has no
smoke-test step cap. Run other
matrix seeds in paired matrix/archive sweeps to avoid a mismatched cross
product. Creating a sweep or uploading results is separate from implementing
and locally validating these baselines.

Use the bandit W&B downloader for TS and the BO W&B downloader for MF-KG. Both
retain the new methods' metadata and support the aggregate plotting script.
MF-KG warmup is retained even with `--bo-phase post_init`. Use a fresh download
directory when upgrading an existing CSV schema.

## GSM8K comparison with the paper's batch-level setup

The dedicated Gittins/TS comparison explicitly uses the full-benchmark empirical
target and the same `posterior mean - posterior std` recommendation for **both**
allocation policies. This differs from the standalone TS runner's default
posterior-mean recommendation. It uses B=8, cell-noise variance 0.25, Gittins
cost scale 1e-4, a 1025-point numerical grid, and a 10% exhaustive budget.
The general prior N(0.5, 0.04) and GSM8K prior N(0.2, 0.01) are each shared by
the two policies, under both unit and heterogeneous evaluation costs.

Each panel averages five matrix seeds and 20 sampling seeds per matrix,
giving 100 paired trials. TS allocation remains cost-unaware; actual prices
determine its stopping budget and the horizontal axis in the heterogeneous
setting. Both policies continue to the budget, with stopping signals retained
only as diagnostics. Gittins uses the manuscript's batch-level empirical-target
DP: each transition observes one batch with noise variance `0.25 / b` and costs
`1e-4 * b * raw_arm_price`, where `b` is the actual batch size. Roots are indexed
by completed local batches, including the smaller final batch when needed.
For GSM8K, the horizon is 125 batches and each root row has 126 entries,
including the terminal zero. The 1025-point grid approximates the exact roots.

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/compare_gittins_ts.py --prepare-only
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/compare_gittins_ts.py
python3 scripts/plot_gittins_ts_comparison.py
```

Outputs are saved under `outputs/gsm8k_gittins_ts_batch_lcb100/`. Root tables are cached
and completed cases resume automatically, with the batch model version, schedule,
and matrix/price hashes preventing reuse of mismatched results. The previous
`outputs/gsm8k_gittins_ts_lcb100/` experiment used per-cell roots and is superseded;
its figures and Gittins results do not follow the paper's batch-level setup.
The figure follows the paper's 350-point linear
interpolation and mean +/- one standard error. Numerical checkpoints and exact
budget-average regret use the last completed observation at or before each
budget, excluding over-budget feedback. Paired summaries include per-matrix
differences and a two-way matrix/trial-seed bootstrap; 100 trajectories do not
represent 100 independent datasets.

The corrected 100-pair experiment gives the following budget-average simple
regret over 0--10% (exact step-hold integral divided by 10%; lower is better):

| Costs | Prior | Batch-level Gittins | Gaussian TS |
|---|---|---:|---:|
| Unit | General | 0.02237811 | 0.02034700 |
| Unit | GSM8K | 0.01770500 | 0.01241134 |
| Heterogeneous | General | 0.00890544 | 0.01900626 |
| Heterogeneous | GSM8K | 0.00130103 | 0.01106157 |

Both methods have zero regret at the 10% checkpoint in all 400 cases. These
results do not establish uniform dominance: TS has lower average regret under
unit costs, while Gittins has lower average regret under heterogeneous costs.
The paired two-way bootstrap interval for the unit/general-prior difference
includes zero. All 400 TS histories were checked against the superseded
experiment and remain identical; Gittins histories were rerun with batch roots.

## Validation

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 -m pytest -q tests
```

Tests cover hand-calculated TS posterior moments and sampling behavior, actual
BoTorch KG scores against direct fantasy enumeration, cost-scale and chunk-size
invariance, prefix reuse, strict MF-KG budget and warmup accounting, score
leakage, row/cost alignment, fixed-batch item counts, and reproducibility. Real GSM8K local smoke traces
are under `outputs/baseline_smoke/`; these are implementation checks, not
evidence of relative benchmark performance.
