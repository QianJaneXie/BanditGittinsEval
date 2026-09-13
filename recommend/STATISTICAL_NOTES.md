# Robust recommendation in the GSM8K replay experiment

The experiment changes the recommendation made from observed scores. Within each
matrix/prior/sampling-seed combination, every recommendation rule sees the same
Gittins sampling trajectory. A rule cannot improve its result by purchasing more
evaluations. Full matrix values are used only to measure performance, never to
form a recommendation.

## Quantities and objectives

There are `K` arms with `N` bounded scores in each row. Let `n_k` and `S_k` be the
number and sum of revealed cells. The measured target is the actual finite row
average `R_k`; simple regret is `max_j R_j - R_recommended`.

The existing normal-normal model instead introduces a latent arm mean `theta_k`:

```
lambda = tau_sq_cell / v0
v_k = 1 / (1/v0 + n_k/tau_sq_cell)
mu_k = (S_k + lambda*mu0) / (n_k + lambda)
```

`v_k` is posterior uncertainty about the latent mean, not the variance of an
individual answer or the empirical variance of observed scores. For the default
prior `mu0=0.5`, `v0=0.04`, and `tau_sq_cell=0.25`, `lambda=6.25`.

The posterior-mean baseline maximizes `mu_k`. Under its model, maximizing a
posterior expected reward minimizes posterior expected simple regret: the term
`E[max_j R_j | data]` does not depend on the recommended arm. Robust alternatives
introduce a different risk preference or a confidence guarantee. They are not
universally better estimators of the original Bayes objective.

## Variance eligibility gate

An arm is eligible when `v_k/v0 <= rho`. Among eligible arms, choose the largest
posterior mean. This directly implements the proposal to demand substantial
uncertainty reduction before considering an arm.

Because the assumed observation variance is fixed, eligibility depends only on
the observation count:

```
n_k >= ceil(lambda * (1/rho - 1))
```

With the default prior, `rho=0.5, 0.2, 0.1, 0.05` requires at least
`7, 25, 57, 119` cells respectively. With batches of 16, eligibility first occurs
at `16, 32, 64, 128` observations. Different prior variances change these counts.

An empty eligible set means abstention, represented by arm `-1`; it must not
silently fall back to an ineligible arm. Regret is undefined for that checkpoint.
Report recommendation coverage alongside regret conditional on recommending.
For pairwise regret differences, evaluate both methods on the same covered
replicates and report how many remain. A method that abstains often cannot be
declared better solely because its conditional regret is smaller.

The gate guarantees a specified contraction of the assumed posterior variance.
It does not guarantee that the selected arm is good, that the prior is correct,
or that the model's uncertainty is calibrated.

## Posterior mean minus variance

Choose the arm maximizing `mu_k - alpha*v_k`. This penalty has a precise
risk-sensitive interpretation under the Gaussian posterior. If utility is
`u(theta)=-exp(-eta*theta)`, then

```
E[u(theta_k) | data] = -exp(-eta*mu_k + eta^2*v_k/2).
```

Maximizing that expected utility is equivalent to maximizing
`mu_k - (eta/2)*v_k`; hence `eta=2*alpha`. Larger `alpha` means stronger
aversion to posterior uncertainty. The parameter is a utility preference, not
a confidence level. This derivation is conditional on the Gaussian model and
does not make its uncertainty empirically calibrated.

## Gaussian lower posterior quantile

Choose the arm maximizing `mu_k - z*sqrt(v_k)`. A penalty on standard deviation
has the same score units as the mean. Under the Gaussian posterior it is the
`Phi(-z)` quantile of the latent mean. For example, `z=1.644854` gives a one-sided
95% posterior lower bound and `z=1.959964` a one-sided 97.5% bound.

These are model-based marginal posterior quantiles. They are not frequentist
confidence sequences and must not be described as guaranteeing correct best-arm
selection with 95% probability. Even a valid lower bound on the chosen arm's
score does not establish that it beats competitors.

## Finite-row Gaussian lower posterior quantile

The finite-row version targets the quantity used to evaluate regret. Under the
same Gaussian generative model, observed cells are known and only the remaining
cells are uncertain. Its posterior predictive mean and variance are

```
q_k = (S_k + (N-n_k)*mu_k) / N
w_k = ((N-n_k)^2*v_k + (N-n_k)*tau_sq_cell) / N^2
```

Choose the largest `q_k - z*sqrt(w_k)`. When an entire row is observed, `q_k` is
its exact mean and `w_k=0`. With equal row lengths and a shared prior,

```
q_k = ((N+lambda)*mu_k - lambda*mu0) / N,
```

so the unpenalized posterior-mean rankings agree. Their uncertainty penalties
differ, making this a useful alternative to penalizing the latent mean.
This remains a Gaussian approximation to binary GSM8K outcomes, not a
distribution-free bound.

## Simultaneous Hoeffding-Serfling confidence bounds

This comparator directly bounds each fixed matrix-row mean using only that the
scores lie in `[0,1]` and that each pulled arm samples uniformly from its remaining
columns. The existing Gittins policy does exactly this in
`src/gittins_policy.py`. Different arms' scores need not be independent.

For `1 <= n < N`, define

```
f(n,N) = min(1 - (n-1)/N, (1 - n/N)*(1 + 1/n))
delta_kn = delta / (K*n*(n+1))
radius = sqrt(f(n,N) * log(2/delta_kn) / (2*n))
L_k = max(0, S_k/N, S_k/n - radius)
U_k = min(1, (S_k + N-n)/N, S_k/n + radius)
```

Use `[L_k,U_k]=[0,1]` at `n=0` and the singleton `[S_k/N,S_k/N]` at `n=N`.
The extra bounds `S_k/N` and `(S_k+N-n)/N` are deterministic: all unknown
values lie between zero and one. Rank arms by `L_k`; the posterior mean can
break exact ties without changing the confidence guarantee. The primary
setting is `delta=0.05`.

The fixed-sample concentration ingredients are Propositions 2.2 and 2.3 of
[Bardenet and Maillard (2015)](https://arxiv.org/pdf/1309.4029).
Each bounds the same moment-generating function. Taking the smaller deterministic
correction costs no extra failure probability. The factor `2` inside the
logarithm accounts for the two tails.

Our time-uniform construction allocates failure probability `delta_kn` and uses
the union bound. Since `sum_{n>=1} 1/(n*(n+1))=1`, all intervals cover all `K`
row means at all observation counts with probability at least `1-delta`.
Adaptive arm choices merely reveal prefixes of per-arm random permutations;
they do not invalidate this simultaneous event. This proof requires uniformly
random unseen columns and would not apply to selecting columns by their
unrevealed scores.

An optional certificate for a recommended arm `a` is

```
certified_regret_bound = max(0, max_{j != a} U_j - L_a).
```

On the simultaneous coverage event, actual simple regret is at most this
quantity, for any adaptively selected arm and checkpoint. Reporting an arm
with the greatest lower bound does not by itself certify it is best; exact
best-arm certification requires `L_a >= max_{j != a} U_j`. If there is only one
arm, the regret bound is zero.

The construction deliberately uses a simple conservative union bound. Tighter
without-replacement confidence sequences are possible; see
[Waudby-Smith and Ramdas (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html).
Their methods are a potential follow-up, not an implementation claim here.

## Reading this experiment

The formulas above are also used unchanged in the
[PIQA and MMLU follow-up](RESULTS_PIQA_MMLU.md). Its MMLU batch size is 4 and its
dataset-specific priors and matrix dimensions differ. The setup below describes
the original GSM8K experiment.

The study uses a 10% evaluation budget, batches of 16, unit evaluation costs,
and Gittins cost scale `1e-4`. Both the default prior and the existing GSM8K prior
are examined. Each has five matrix seeds and twenty sampling seeds per matrix;
recommendation comparisons are paired within each resulting trajectory.

Useful comparisons include mean regret across budget checkpoints, endpoint
regret, its upper tail, the selected arm's observation count, and coverage for
gated rules. Confidence bands describe replay variability, not uncertainty
about future datasets or future LLM calls. Matrix and sampling seeds have
different roles; uncertainty calculations should preserve this grouping when
claiming variation across matrices.

The finite matrix oracle is appropriate for scoring saved recommendations, but
not for choosing the rule or its parameter. Parameter sweeps on GSM8K are
exploratory: selecting their apparent winner and quoting the same runs as
independent evidence would overstate the conclusion. Further confirmation
should use new matrix draws or held-out benchmarks with a fixed rule.

These experiments leave the Gittins allocation policy unchanged. A conservative
recommendation can prevent promotion of a lightly sampled arm, but it cannot
create missing evidence. A later experiment could spend a reserved part of the
budget on final comparison of promising arms; that would be a different
sampling policy and needs its own budget-matched evaluation.
