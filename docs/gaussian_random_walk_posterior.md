### Gaussian Random Walk Evolution of the Posterior Mean

**Notation.** θ<sub>k</sub> is the unknown mean for arm *k* (latent). After data *D*<sub>t</sub>, the belief is summarized by μ<sub>k,t</sub> = **E**[θ<sub>k</sub> | *D*<sub>t</sub>] and *v*<sub>k,t</sub>. In a stopping or Gittins formulation, a one-dimensional **state** *s*<sub>k,t</sub> is often the sufficient statistic you act on; here it is natural to set *s*<sub>k,t</sub> := μ<sub>k,t</sub>. Do **not** identify *s*<sub>k,t</sub> with θ<sub>k</sub>: the former is an estimate, the latter is the parameter being learned.

We model each arm *k* with a Gaussian prior:

θ<sub>k</sub> ∼ **N**(μ<sub>0</sub>, *v*<sub>0</sub>), e.g. **N**(0.7, 0.01).

At each evaluation, we observe a noisy estimate:

*Y*<sub>t</sub> | θ<sub>k</sub> ∼ **N**(θ<sub>k</sub>, τ²),

where under the worst-case approximation:

τ² ≈ 1 / (4*B*).

Here ***B*** is the number of examples evaluated on the chosen arm in one step—the same quantity as **`batch_size`** in ``gittins_index_exploration`` (and the per-step evaluation batch for that method). Larger batches imply a smaller effective observation variance τ² under this bound.

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
