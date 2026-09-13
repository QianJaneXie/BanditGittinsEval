### Gaussian Random Walk Evolution of the Posterior Mean

**Notation.** θ<sub>k</sub> is the unknown mean for arm *k* (latent). After data *D*<sub>t</sub>, the latent belief has μ<sub>k,t</sub> = **E**[θ<sub>k</sub> | *D*<sub>t</sub>] and variance v<sub>k,t</sub>. The Gittins state is now s<sub>k,t</sub> := **E**[F<sub>k</sub> | *D*<sub>t</sub>], where F<sub>k</sub> is the realized full fixed test-set mean. The latent posterior formulas below provide its predictive model; the finite-target transition is derived separately below. See [the unified index](finite_population_gittins.md).

We model each arm *k* with a Gaussian prior:

θ<sub>k</sub> ∼ N(μ<sub>0</sub>, v<sub>0</sub>), e.g. N(0.5, 0.04).

---

## Batch Observation Model

In practice, each evaluation consists of a batch of *B* independent observations.  
For example, in model evaluation, these correspond to correctness indicators:

X<sub>k,i</sub> ∼ Bernoulli(θ<sub>k</sub>), i = 1, ..., B

The observed quantity is the empirical mean:

Y<sub>t</sub> = (1 / B) ∑<sub>i=1</sub><sup>B</sup> X<sub>k,i</sub>

By the central limit theorem:

Y<sub>t</sub> | θ<sub>k</sub> ≈ N(θ<sub>k</sub>, θ<sub>k</sub>(1 − θ<sub>k</sub>) / B)

For tractability, we use the worst-case approximation:

τ² ≈ 1 / (4B)

Thus:

Y<sub>t</sub> | θ<sub>k</sub> ∼ N(θ<sub>k</sub>, τ²)

---

After *t* observations, the posterior remains Gaussian:

θ<sub>k</sub> | D<sub>t</sub> ∼ N(μ<sub>k,t</sub>, v<sub>k,t</sub>)

---

## Recursive Update as a Gaussian Random Walk

The posterior mean evolves as:

μ<sub>k,t+1</sub> = μ<sub>k,t</sub> + η<sub>k,t+1</sub>

where:

η<sub>k,t+1</sub> ∼ N(0, σ<sub>k,t</sub><sup>2</sup>)

and

σ<sub>k,t</sub><sup>2</sup> = v<sub>k,t</sub><sup>2</sup> / (v<sub>k,t</sub> + τ²)

---

## Variance Shrinkage

v<sub>k,t+1</sub> = (1 / v<sub>k,t</sub> + 1 / τ²)<sup>−1</sup>

As more data is collected:

- v<sub>k,t</sub> ↓ 0  
- σ<sub>k,t</sub><sup>2</sup> ↓ 0  

---

## Interpretation

- Posterior mean follows a **Gaussian random walk with shrinking variance**
- Early stage → large uncertainty
- Later stage → stable estimates
- Learning “freezes” over time

---

## Arm Selection via Gittins Index

For N fixed examples per arm, n observed scores with sum S, and per-cell noise variance τ²<sub>cell</sub>, the state is

s<sub>k,n</sub> = (S + (N − n) μ<sub>k,n</sub>) / N = a μ<sub>k,n</sub> + (1 − a) μ<sub>0</sub>,

where a = 1 + τ²<sub>cell</sub> / (N v<sub>0</sub>). Thus the state's transition standard deviation is a times the latent posterior-mean transition standard deviation. The DP uses that finite-target transition with the configured costs unchanged.

At each step:

A<sub>t</sub> = argmax<sub>k</sub> Γ<sub>k,n<sub>k</sub>(t)</sub>

where:

Γ<sub>k,n</sub> = s<sub>k,n</sub> − r<sub>n</sub>

- s<sub>k,n</sub> = full-test-set posterior mean
- r<sub>n</sub> = precomputed root satisfying Q<sub>n</sub>(r<sub>n</sub>) = 0  

---

## Global Stopping Rule (Index-Induced)

The stopping rule is **implicitly determined by the same index selection rule**.

At each step:

1. Select arm with largest Gittins index  
2. If the selected arm is already **completed (final stage)** → stop  

A completed arm's index equals its exact full-row empirical mean. Unfinished indices use the same finite target. Fixed-budget experiment runners record this first stopping time and continue collecting their budgeted observations.

Stopping time:

T = inf { t : A<sub>t</sub> is completed }

Key points:

- No global threshold on indices  
- Not based on posterior means  
- Same rule drives selection and stopping  

---

## Selection vs Recommendation

### Selection (evaluation)
A<sub>t</sub> = argmax<sub>k</sub> Γ<sub>k,n<sub>k</sub>(t)</sub>

### Recommendation (reporting)
k̂<sub>t</sub> = argmax<sub>k</sub> E[F<sub>k</sub> | D<sub>t</sub>], where F<sub>k</sub> is the full fixed test-set mean.

For N examples per arm and n<sub>k</sub> observations, this mean is
(sum of observed scores + (N − n<sub>k</sub>) μ<sub>k,t</sub>) / N.
Completed arms therefore use the exact empirical row mean. An optional standard-deviation penalty uses the posterior uncertainty of F<sub>k</sub>, including the unrevealed cells' variation. See [the finite-test-set recommendation](finite_population_recommendation.md).

- Index → exploration decision  
- Full-test-set posterior mean → final choice (default, no std penalty)

---

## Efficient Gittins Index Computation

### Offline Precomputation

Variance evolution:

v<sub>t+1</sub> = (1 / v<sub>t</sub> + 1 / τ²)<sup>−1</sup>

Precompute:

v<sub>0</sub>, v<sub>1</sub>, v<sub>2</sub>, ...

Latent posterior-mean transition variance:

σ<sub>t</sub><sup>2</sup> = v<sub>t</sub><sup>2</sup> / (v<sub>t</sub> + τ²)

For the finite-target DP, use (a σ<sub>t</sub>)². In per-cell mode τ² = τ²<sub>cell</sub>; for a batch of q cells use τ² = τ²<sub>cell</sub> / q. Costs retain their configured numerical values.

Compute roots:

Q<sub>t</sub>(r<sub>t</sub>) = 0

Store {r<sub>t</sub>}.

---

### Initialization

s<sub>k,0</sub> = μ<sub>0</sub>  
Γ<sub>k,0</sub> = s<sub>k,0</sub> − r<sub>0</sub>

---

### Online Updates

After observing batch mean Y<sub>t+1</sub>:

s<sub>t+1</sub> = s<sub>t</sub> + η<sub>t+1</sub>

η<sub>t+1</sub> = a (v<sub>t</sub> / (v<sub>t</sub> + τ²)) (Y<sub>t+1</sub> − μ<sub>t</sub>)

Update index:

Γ<sub>k,t</sub> = s<sub>k,t</sub> − r<sub>k,t</sub>

---

### Closed-form posterior (and an equivalent per-cell view)

Under the Gaussian conjugate model with prior θ ∼ **N**(μ<sub>0</sub>, v<sub>0</sub>) and observations
Y | θ ∼ **N**(θ, τ²) (with the same τ² for every observation), the posterior after *t* observations has
the closed form:

- v<sub>t</sub> = (1 / v<sub>0</sub> + t / τ²)<sup>−1</sup>
- μ<sub>t</sub> = v<sub>t</sub> (μ<sub>0</sub> / v<sub>0</sub> + (∑<sub>i=1</sub><sup>t</sup> Y<sub>i</sub>) / τ²)

This is equivalent to applying the one-step recursion above repeatedly: the Kalman gain
v<sub>t</sub>/(v<sub>t</sub> + τ²) changes with *t*, but the final posterior depends on the data only through
the sufficient statistics (*t*, ∑Y<sub>i</sub>) when τ² is constant.

**Batch-mean observation model vs per-cell observations.** If each step observes a batch mean
Ȳ of *B* fresh i.i.d. per-cell draws with variance τ²<sub>cell</sub>, then
Var(Ȳ | θ) = τ²<sub>cell</sub> / B. Thus a batch-mean noise level τ² can be viewed equivalently as a per-cell
noise level τ²<sub>cell</sub> = B τ² (so that τ²<sub>cell</sub> / B = τ²). In particular, the common bound
τ² = 1/(4B) corresponds to τ²<sub>cell</sub> = 1/4.

---

## Summary

- Batch observations → empirical mean → Gaussian approximation  
- Posterior updates → Gaussian random walk  
- Gittins index → optimal exploration decision  
- Stopping → induced by index rule  
- Efficient implementation:
  - offline DP (roots)
  - online update + lookup + subtraction  

---

## Overall Interpretation

This framework unifies:

- Bayesian learning (posterior updates)  
- Sequential decision making (Gittins index)  
- Efficient computation (precomputation + lookup)  

The algorithm adaptively balances:

- uncertainty reduction  
- evaluation cost  
- final decision quality
