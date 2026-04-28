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

v<sub>t+1</sub> = (1 / v<sub>t</sub> + 1 / τ²)<sup>−1</sup>

Thus, we can precompute:

v<sub>0</sub>, v<sub>1</sub>, v<sub>2</sub>, ...

This induces transition variances:

σ<sub>t</sub><sup>2</sup> = v<sub>t</sub><sup>2</sup> / (v<sub>t</sub> + τ²)

Using these, we solve the dynamic program once to compute roots:

Q<sub>t</sub>(r<sub>t</sub>) = 0

The roots {r<sub>t</sub>} are stored in a lookup table.

---

### Initialization

At the beginning:

s<sub>k,0</sub> = μ<sub>0</sub>  
Γ<sub>k,0</sub> = s<sub>k,0</sub> − r<sub>0</sub>

Arms are selected via:

A<sub>t</sub> = argmax<sub>k</sub> Γ<sub>k,n<sub>k</sub>(t)</sub>

---

### Online Updates

After selecting an arm and observing Y<sub>t+1</sub>:

s<sub>t+1</sub> = s<sub>t</sub> + η<sub>t+1</sub>

η<sub>t+1</sub> = (v<sub>t</sub> / (v<sub>t</sub> + τ²)) (Y<sub>t+1</sub> − μ<sub>t</sub>)

Then update the index:

Γ<sub>k,t</sub> = s<sub>k,t</sub> − r<sub>k,t</sub>

---

### Summary

- Posterior variance updates are deterministic and precomputed.
- Gittins roots {r<sub>t</sub>} are computed offline once.
- Online computation requires:
  - updating the posterior mean,
  - looking up r<sub>t</sub>,
  - one subtraction.

This yields an efficient implementation of Gittins index policies.