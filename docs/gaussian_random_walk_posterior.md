### Gaussian Random Walk Evolution of the Posterior Mean

**Notation.** θ<sub>k</sub> is the unknown mean for arm *k* (latent). After data *D*<sub>t</sub>, the belief is summarized by μ<sub>k,t</sub> = **E**[θ<sub>k</sub> | *D*<sub>t</sub>] and *v*<sub>k,t</sub>. In a stopping or Gittins formulation, a one-dimensional **state** *s*<sub>k,t</sub> is often the sufficient statistic you act on; here it is natural to set *s*<sub>k,t</sub> := μ<sub>k,t</sub>. Do **not** identify *s*<sub>k,t</sub> with θ<sub>k</sub>: the former is an estimate, the latter is the parameter being learned.

We model each arm *k* with a Gaussian prior:

θ<sub>k</sub> ∼ **N**(μ<sub>0</sub>, *v*<sub>0</sub>), e.g. **N**(0.5, 0.04).

At each evaluation, we observe a noisy estimate:

*Y*<sub>t</sub> | θ<sub>k</sub> ∼ **N**(θ<sub>k</sub>, τ²),

where under the worst-case approximation:

τ² ≈ 1 / (4*B*).

Here ***B*** is the number of examples evaluated on the chosen arm in one step—the same quantity as **`batch_size`** in `gittins_index_exploration`. Larger batches imply a smaller effective observation variance τ².

After *t* observations, the posterior remains Gaussian:

θ<sub>k</sub> | *D*<sub>t</sub> ∼ **N**(μ<sub>k,t</sub>, *v*<sub>k,t</sub>).

---

### Recursive Update as a Gaussian Random Walk

The posterior mean μ<sub>k,t</sub> is the natural state. Each new observation induces the update:

μ<sub>k,t+1</sub> = μ<sub>k,t</sub> + η<sub>k,t+1</sub>,

where the increment is Gaussian:

η<sub>k,t+1</sub> ∼ **N**(0, σ<sub>k,t</sub><sup>2</sup>),

with variance:

σ<sub>k,t</sub><sup>2</sup> = *v*<sub>k,t</sub><sup>2</sup> / (*v*<sub>k,t</sub> + τ²).

---

### Variance Shrinkage

The posterior variance evolves deterministically:

*v*<sub>k,t+1</sub> = (1/*v*<sub>k,t</sub> + 1/τ²)<sup>−1</sup>.

As more observations are collected:

* *v*<sub>k,t</sub> ↓ 0  
* σ<sub>k,t</sub><sup>2</sup> ↓ 0  

---

### Interpretation

* The posterior mean follows a **Gaussian random walk with shrinking step size**.
* Early observations cause **large updates** (high uncertainty).
* Later observations produce **small refinements** (high confidence).
* The process transitions from **exploration (high variance)** to **stabilization (low variance)**.

Thus, learning corresponds to a random walk that gradually “freezes” as uncertainty vanishes.

---

## Arm Selection via Gittins Index

Given the posterior state, decisions are made using the **Gittins index**, which captures the value of further evaluation.

At each global step *t*, the algorithm selects the arm with the largest index:

A<sub>t</sub> ∈ argmax<sub>k</sub> Γ<sub>k,n<sub>k</sub>(t)</sub>

where:
- n<sub>k</sub>(t) is the number of times arm *k* has been evaluated,
- Γ<sub>k,n</sub> = s<sub>k,n</sub> − r<sub>n</sub> is the Gittins index,
- r<sub>n</sub> is the precomputed root satisfying Q<sub>n</sub>(r<sub>n</sub>) = 0.

---

## Global Stopping Rule (Index-Induced)

The stopping rule is **induced by the same Gittins index selection rule**.

At each step:
1. Select the arm with the largest Gittins index.
2. If the selected arm is already in its **completed (final) stage**, then the next evaluation transitions the system to the terminal state.

Thus, the stopping time is:

T = inf { t : A<sub>t</sub> is completed }

Key points:
- Stopping depends on the **argmax Gittins index**, not posterior means.
- There is **no global threshold across arms**.
- The same rule governs both **selection and stopping**.

---

## Selection vs Recommendation

It is important to distinguish two roles:

### Arm selection (for evaluation)
A<sub>t</sub> = argmax<sub>k</sub> Γ<sub>k,n<sub>k</sub>(t)</sub>

### Arm recommendation (for reporting)
k̂<sub>t</sub> = argmax<sub>k</sub> μ<sub>k,t</sub>

- The **Gittins index** determines which arm to evaluate next.
- The **posterior mean** determines which arm is believed to be best.

---

## Efficient Gittins Index Computation

The Gaussian structure enables an efficient implementation by separating offline precomputation from online updates.

### Offline Precomputation

Since the observation noise τ² is fixed, the posterior variance evolves deterministically:

\[
v_{k,t+1} = \left(\frac{1}{v_{k,t}} + \frac{1}{\tau^2}\right)^{-1}.
\]

Thus, we can precompute:

\[
v_0, v_1, v_2, \dots
\]

This induces transition variances:

\[
\sigma_t^2 = \frac{v_t^2}{v_t + \tau^2}.
\]

Using these, we solve the dynamic program once to compute roots:

\[
Q_t(r_t) = 0.
\]

The roots \(\{r_t\}\) are stored in a lookup table.

---

### Initialization

At the beginning:

\[
s_{k,0} = \mu_0, \quad \Gamma_{k,0} = s_{k,0} - r_0.
\]

Arms are selected via the Gittins index rule.

---

### Online Updates

After selecting an arm and observing \(Y_{t+1}\):

\[
s_{t+1} = s_t + \eta_{t+1},
\qquad
\eta_{t+1} := \frac{v_t}{v_t+\tau^2}(Y_{t+1}-\mu_t).
\]

Then update the index:

\[
\Gamma_{k,t} = s_{k,t} - r_{k,t}.
\]

---

### Summary

- Posterior variance updates are **deterministic and precomputed**.
- Gittins roots \(\{r_t\}\) are computed **offline once**.
- Online computation requires:
  - updating the posterior mean,
  - looking up \(r_t\),
  - computing one subtraction.

This yields an efficient implementation of Gittins index policies.

---

## Overall Interpretation

- Learning is a **shrinking Gaussian random walk**.
- The Gittins index measures **value of information**.
- Selection is driven by **index maximization**.
- Stopping occurs when the **highest-index arm is already completed**.

This unifies:
- Bayesian learning (posterior updates)
- optimal stopping (Gittins index)
- practical decision-making (selection vs recommendation)