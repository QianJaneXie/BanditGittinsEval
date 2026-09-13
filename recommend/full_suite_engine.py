"""Compact paired posterior-mean/LCB replay with unchanged Gittins allocation.

Only the pulled arm's sufficient statistics and index change. Cached vectors
avoid evaluating unused recommendation families or storing dense state history.
All sampled cells are retained so every decision can be independently audited.
"""
from __future__ import annotations
import numpy as np

METHOD_KEYS = ["posterior_mean", "gaussian_lcb_1.645"]
Z = 1.645


def run_path(matrix, roots, *, prior_tag, prior_pair, run_seed, matrix_seed=1,
             batch_size=4, budget_fraction=.1, tau_sq_cell=.25):
    matrix = np.asarray(matrix, dtype=np.float64)
    k, n = matrix.shape
    mean0, variance0 = prior_pair
    budget = max(1, round(budget_fraction * matrix.size))
    counts = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    observed = np.zeros((k, n), dtype=bool)
    variance = 1 / (1 / variance0 + np.arange(n+1, dtype=np.float64) / tau_sq_cell)
    deviations = np.sqrt(variance)
    means32 = np.full(k, mean0, dtype=np.float32)
    lcbs = np.full(k, mean0-Z*deviations[0], dtype=np.float64)
    indices = means32-roots[0]
    generator = np.random.default_rng(run_seed)
    truth = matrix.mean(axis=1, dtype=np.float64)
    best = float(truth.max())
    max_steps = (budget+batch_size-1)//batch_size+k
    history = {
        "x": np.empty(max_steps, dtype=np.int64),
        "pulled_arm": np.empty(max_steps, dtype=np.int32),
        "pulled_columns": np.full((max_steps,batch_size), -1, dtype=np.int32),
        "recommended_arm": np.empty((max_steps,2), dtype=np.int32),
        "regret": np.empty((max_steps,2), dtype=np.float64),
        "scores": np.empty((max_steps,2), dtype=np.float64),
        "selected_n": np.empty((max_steps,2), dtype=np.int32),
        "selected_variance": np.empty((max_steps,2), dtype=np.float64),
    }
    evaluated=step=0
    while evaluated < budget:
        arm=int(np.argmax(indices))
        remaining=np.flatnonzero(~observed[arm])
        size=min(batch_size,len(remaining),budget-evaluated)
        assert size>0
        columns=remaining[generator.permutation(len(remaining))[:size]]
        observed[arm,columns]=True
        counts[arm]+=size
        sums[arm]+=matrix[arm,columns].sum(dtype=np.float64)
        count=counts[arm]
        mean=variance[count]*(mean0/variance0+sums[arm]/tau_sq_cell)
        means32[arm]=mean
        lcbs[arm]=mean-Z*deviations[count]
        indices[arm]=means32[arm]-roots[count] if count<n else -np.inf
        evaluated+=size
        chosen=np.array([np.argmax(means32),np.argmax(lcbs)],dtype=np.int32)
        history["x"][step]=evaluated
        history["pulled_arm"][step]=arm
        history["pulled_columns"][step,:size]=columns
        history["recommended_arm"][step]=chosen
        history["regret"][step]=best-truth[chosen]
        history["scores"][step]=[means32[chosen[0]],lcbs[chosen[1]]]
        history["selected_n"][step]=counts[chosen]
        history["selected_variance"][step]=variance[counts[chosen]]
        step+=1
    assert evaluated==budget==int(counts.sum())==int(observed.sum())
    assert np.all(counts<=n)
    result={key:value[:step].copy() for key,value in history.items()}
    result.update(prior_tag=np.asarray(prior_tag),prior_mean=np.asarray(mean0),
        prior_variance=np.asarray(variance0),run_seed=np.asarray(run_seed),
        matrix_seed=np.asarray(matrix_seed),method_keys=np.asarray(METHOD_KEYS),
        n_arms=np.asarray(k),n_examples=np.asarray(n),true_means=truth,
        final_counts=counts,final_sums=sums,sampling_generator=np.asarray("numpy.random.PCG64"))
    return result
