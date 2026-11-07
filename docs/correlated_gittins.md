# Correlation-aware GittinsEval: marginal-index approximation

This policy updates a joint Gaussian belief across configurations and assigns
each arm the stopping index of its current one-dimensional marginal. The
single-arm stopping problem ignores information that future evaluations of
other arms could provide. Allocation and index-based stopping are therefore
heuristic for correlated arms. The independent-arm optimality guarantee does
not extend to this policy.

## Shared belief and empirical target

`MarginalIndexGittins` composes the existing `CorrelatedGaussianTS` belief. It
uses the same normalized categorical configuration kernel, fixed marginal
prior mean and variance, Gaussian cell-noise model, and observed-data kernel
fitting. The default is a lengthscale refit every 32 batches, with 25 L-BFGS
iterations per fit. It does not sample a TS acquisition vector.

After any evaluation, the joint update changes beliefs of all correlated arms,
so all indices are refreshed. For arm k, let its latent marginal have mean mu
and variance v, with observed sum S, N total items, and m unobserved items:

```text
M = (S + m * mu) / N
V = (m^2 * v + m * tau_cell^2) / N^2
```

The current empirical mean M is the index state. The root table takes latent
variance v, rather than V. The default recommendation is the same empirical
LCB rule, `M - sqrt(V)`. This recommendation penalty is not an acquisition bonus.
Completed arms have their observed full-row scores as terminal indices.

## Single-arm stopping roots

Starting from the current marginal, hold the current model fixed inside its
stopping problem and suppose only this arm is evaluated in future. Any actual
later kernel refit supplies new marginals and new lookup indices. For u
additional observed items and next batch size
`b = min(B, m-u)`, the single-arm chain has

```text
v_u = 1 / (1/v + u/tau_cell^2)
a = (m + tau_cell^2/v) / N
sigma_u = a * v_u / sqrt(v_u + tau_cell^2/b)
transition_cost = lambda * b * raw_arm_price
```

This is the original Gaussian single-arm STOP/GO recursion, evaluated with the
existing FFT convolution. The final short batch uses its actual size for both
noise and cost. The implementation evaluates the algebra in a stable form
that retains `sigma_u = sqrt(b*tau_cell^2)/N` when v is zero: unobserved item
noise remains even if the latent mean is known.

The approximate unfinished-arm index is `M - root(v, m, raw_arm_price)`.
No global dynamic program, future cross-arm sharing term, or additional
acquisition bonus is introduced.

## Offline precomputation and online interpolation

Writing n = N-m, define an effective prior variance

```text
w = 1 / (1/v - n/tau_cell^2)
```

For the shared fixed marginal prior variance v0, conditioning on other arms
implies `0 <= w <= v0`. A full single-arm chain initialized at w gives the
required current variance at count n. One backward scan therefore supplies
roots for all stages along that chain, instead of solving a new DP for every
combination of current variance and remaining items.

The default table includes 65 effective-variance nodes, including the zero
limit and exact v0 anchor. Up to B residue-phase scans cover every integer
remaining-item count, including irregular partial histories. Online lookup
brackets these nodes and interpolates in the **actual current variance** v.
It never runs an FFT or DP online. Costs are tabulated at the exact supplied
raw prices. Unsupported prices and substantial variance-domain violations
raise an error rather than being silently extrapolated.

The v0 anchor copies the original offline roots with their existing float32
arithmetic. Maintaining zero cross-arm covariance therefore reproduces the
original Gittins indices, actions, LCB recommendations and stopping signals,
including partial terminal batches. The lookup cache records its numerical
settings and checks its payload on load.

## Running one matrix

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/run_correlated_gittins_baseline.py \
  --matrix data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy \
  --bo-inputs data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz \
  --batch-size 8 --prior-mean 0.2 --prior-variance 0.01 \
  --kernel-mode fit --fit-every 32 --fit-maxiter 25 \
  --out outputs/correlated_gittins/gsm8k_seed1_S_unit.npz
```

Add `--cost-aware` for price-dependent indices and a cost budget. Use
`--root-cache <path>` to share an existing compatible table; otherwise tables
are cached by numerical settings and prices under
`outputs/correlated_gittins_root_cache/`. `--prepare-only` builds or validates
the table without evaluating any items. Dataset priors and batch sizes must
be supplied consistently when running other benchmarks.

By default, the first natural and recommendation-aware index crossings are
recorded and sampling continues to the fixed budget, matching the paper's
evaluation protocol. `--allow-early-stop` instead ends at the inherited
heuristic index signal. Its choice is recorded in the standalone trace.

## Paired GSM8K comparison

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/compare_correlated_gittins.py --prepare-only
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/compare_correlated_gittins.py
python3 scripts/plot_correlated_gittins_comparison.py
```

Only the new marginal-index policy is evaluated. Independent Gittins and
correlated Gaussian TS must already exist under the two reference directories
(overridable with `--gittins-results-dir` and `--correlated-ts-results-dir`).
Preflight checks data, feature order, prices, prior, batch size, noise, fitting
schedule and recommendation settings. Missing or mismatched baseline data
fails explicitly; there is no baseline simulation fallback. New case files
contain only the new policy's arrays and verified baseline references.

The defaults use five response matrices and 20 sampling seeds per matrix,
both priors, both cost settings, B=8, cell noise 0.25, LCB coefficient one,
lambda=1e-4, a 1025-point stopping grid, and a 10% exhaustive budget. Root
tables are shared across matrices and seeds. Figures use 350-point linear
interpolation with mean +/- one standard error; numerical checkpoints and
budget-average regret use completed observations only. Paired uncertainty
resamples matrix and shared algorithm-seed IDs and remains descriptive with
only five matrices. All generated results and root caches remain local under
the ignored `outputs/` directory.
