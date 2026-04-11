### Gaussian Random Walk Evolution of the Posterior Mean

**Notation.** \(\theta_k\) is the unknown mean for arm \(k\) (latent). After data \(\mathcal{D}_t\), the belief is summarized by \(\mu_{k,t} = \mathbb{E}[\theta_k \mid \mathcal{D}_t]\) and \(v_{k,t}\). In a stopping or Gittins formulation, a one-dimensional **state** \(s_{k,t}\) is often the sufficient statistic you act on; here it is natural to set \(s_{k,t} := \mu_{k,t}\). Do **not** identify \(s_{k,t}\) with \(\theta_k\): the former is an estimate, the latter is the parameter being learned.

We model each arm \(k\) with a Gaussian prior:
\[
\theta_k \sim \mathcal{N}(\mu_0, v_0), \quad \text{e.g., } \mathcal{N}(0.7, 0.01).
\]

At each evaluation, we observe a noisy estimate:
\[
Y_t \mid \theta_k \sim \mathcal{N}(\theta_k, \tau^2),
\]
where under the worst-case approximation:
\[
\tau^2 \approx \frac{1}{4B}.
\]
Here **\(B\)** is the number of examples evaluated on the chosen arm in one step—the same quantity as **`batch_size`** in ``gittins_index_exploration`` (and the per-step evaluation batch for that method). Larger batches imply a smaller effective observation variance \(\tau^2\) under this bound.

After \(t\) observations, the posterior remains Gaussian:
\[
\theta_k \mid \mathcal{D}_t \sim \mathcal{N}(\mu_{k,t}, v_{k,t}).
\]

---

### Recursive Update as a Gaussian Random Walk

The posterior mean \(\mu_{k,t}\) is the natural state. Each new observation induces the update:
\[
\mu_{k,t+1} = \mu_{k,t} + \eta_{k,t+1},
\]
where the increment is Gaussian:
\[
\eta_{k,t+1} \sim \mathcal{N}(0, \sigma^2_{k,t}),
\]
with variance:
\[
\sigma^2_{k,t} = \frac{v_{k,t}^2}{v_{k,t} + \tau^2}.
\]

---

### Variance Shrinkage

The posterior variance evolves deterministically:
\[
v_{k,t+1} = \left( \frac{1}{v_{k,t}} + \frac{1}{\tau^2} \right)^{-1}.
\]

As more observations are collected:

* \(v_{k,t} \downarrow 0\)
* \(\sigma^2_{k,t} \downarrow 0\)

---

### Interpretation

* The posterior mean follows a **Gaussian random walk with shrinking step size**.
* Early observations cause **large updates** (high uncertainty).
* Later observations produce **small refinements** (high confidence).
* The process transitions from **exploration (high variance)** to **stabilization (low variance)**.

Thus, learning corresponds to a random walk that gradually “freezes” as uncertainty vanishes.