#!/usr/bin/env python3
"""Run only marginal-index correlation-aware Gittins; reuse existing baselines.

The stopping indices ignore future information sharing. They are a heuristic
for correlated arms, without the independent-arm optimality guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_correlated_ts import _validate_case_identity, _validate_reference_case, validate_reference_study
from compare_gittins_ts import PRIORS, _atomic_npz, exact_normalized_auc, load_paper_costs
from run_mfkg_baseline import load_data

MODEL_VERSION = "correlation_aware_gittins_marginal_index_v1"
POLICY = "correlated_gittins"
BASELINES = ("gittins", "correlated_ts")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate_correlated_gittins(
    matrix, *, features, root_table, seed, prior_mean, prior_variance,
    batch_size, per_arm_cost, cost_mode, budget_fraction=.1, tau_sq_cell=.25,
    std_penalty=1., cost_scaling_factor=1e-4, kernel_mode="fit", fit_every=32,
    fit_maxiter=25, max_steps=None, record_cells=True, allow_early_stop=False,
    correlation_matrix=None,
):
    """Evaluate only selected fresh cells; no baseline acquisition is invoked."""
    from correlated_gittins_policy import MarginalIndexGittins

    truth = np.asarray(matrix)
    if truth.ndim != 2 or min(truth.shape) < 1 or not np.isfinite(truth).all():
        raise ValueError("matrix must be a finite nonempty two-dimensional array")
    if cost_mode not in {"unit", "cost"} or not 0 < budget_fraction <= 1:
        raise ValueError("invalid cost mode or budget fraction")
    if not np.isfinite(std_penalty) or std_penalty < 0:
        raise ValueError("std_penalty must be finite and nonnegative")
    if max_steps is not None and (int(max_steps) != max_steps or max_steps < 1):
        raise ValueError("max_steps must be positive when supplied")
    costs = np.asarray(per_arm_cost, dtype=np.float64)
    n_arms, n_examples = truth.shape
    if costs.shape != (n_arms,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("per_arm_cost must contain positive row-aligned prices")
    decision_costs = np.ones_like(costs) if cost_mode == "unit" else costs
    policy = MarginalIndexGittins(
        features, root_table=root_table, n_examples=n_examples, batch_size=batch_size,
        per_arm_cost=decision_costs, prior_mean=prior_mean, prior_variance=prior_variance,
        tau_sq_cell=tau_sq_cell, cost_scaling_factor=cost_scaling_factor,
        kernel_mode=kernel_mode, fit_every=fit_every, fit_maxiter=fit_maxiter,
        correlation_matrix=correlation_matrix,
    )
    initial_kernel = json.loads(json.dumps(policy.kernel_metadata))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scores = np.asarray(truth, dtype=np.float32)
    means_true = np.asarray(truth, dtype=np.float64).mean(1)
    best = float(means_true.max())
    full_budget = float(n_arms * n_examples if cost_mode == "unit" else n_examples * costs.sum())
    budget = float(max(1, round(budget_fraction * full_budget))) if cost_mode == "unit" else budget_fraction * full_budget
    seen = np.zeros(scores.shape, dtype=bool)
    history = {k: [] for k in ("x", "x_original_cost", "pulled_arm", "n_new_samples",
                               "recommended_arm", "recommended_mean", "regret")}
    columns_history, fits = [], []
    evaluated, spent, reason = 0, 0., "budget"
    natural, recommendation_aware = [None], [None]
    started = time.perf_counter()
    while (evaluated if cost_mode == "unit" else spent) < budget:
        if max_steps is not None and len(history["x"]) >= max_steps:
            reason = "max_steps"
            break
        arm = policy.select_arm(allow_early_stop=allow_early_stop)
        if arm is None:
            reason = "all_arms_complete" if bool((policy.counts == n_examples).all()) else "index_stopping_signal"
            break
        unseen = np.flatnonzero(~seen[arm])
        size = min(batch_size, len(unseen))
        columns = unseen[torch.randperm(len(unseen), generator=generator).numpy()[:size]]
        seen[arm, columns] = True
        old_fits = policy.belief.fit_count
        policy.observe(arm, torch.as_tensor(scores[arm, columns], dtype=torch.float64))
        evaluated += size
        spent += float(torch.full((size,), costs[arm], dtype=torch.float64).sum())
        policy.stopping_diagnostics(evaluated, natural_stop_cum_eval_holder=natural,
                                   recommendation_aware_stop_cum_eval_holder=recommendation_aware,
                                   std_penalty=std_penalty)
        rec, means, _ = policy.recommend(std_penalty=std_penalty)
        for key, value in dict(x=evaluated, x_original_cost=spent, pulled_arm=arm,
                               n_new_samples=size, recommended_arm=rec,
                               recommended_mean=float(means[rec]),
                               regret=best - float(means_true[rec])).items():
            history[key].append(value)
        if policy.belief.fit_count != old_fits:
            k = policy.kernel_metadata
            fits.append({"batch": len(history["x"]), "evaluations": evaluated,
                         "categorical_lengthscales": k["categorical_lengthscales"],
                         "diagnostic": k.get("fit_diagnostics", [None])[-1]})
        if record_cells:
            padded = np.full(batch_size, -1, dtype=np.int64)
            padded[:size] = columns
            columns_history.append(padded)
    integer = {"x", "pulled_arm", "n_new_samples", "recommended_arm"}
    result = {key: np.asarray(value, dtype=np.int64 if key in integer else np.float64)
              for key, value in history.items()}
    result.update(policy=POLICY, cost_mode=cost_mode, budget=budget, full_budget=full_budget,
                  initial_recommended_arm=0, initial_regret=best - float(means_true[0]),
                  counts=policy.counts.numpy().copy(), observed_sums=policy.sums.numpy().copy(),
                  elapsed_s=time.perf_counter() - started, stop_reason=reason,
                  natural_stop_cum_eval=-1 if natural[0] is None else natural[0],
                  recommendation_aware_stop_cum_eval=-1 if recommendation_aware[0] is None else recommendation_aware[0],
                  fit_count=policy.belief.fit_count, kernel_initial_json=json.dumps(initial_kernel),
                  kernel_final_json=json.dumps(policy.kernel_metadata), kernel_fit_history_json=json.dumps(fits),
                  approximation="single_arm_stopping_ignores_future_cross_arm_information",
                  independent_arm_optimality_guarantee=False)
    if record_cells:
        result["sampled_columns"] = np.asarray(columns_history, dtype=np.int64).reshape(-1, batch_size)
    return result


def prepare(args):
    """Check reusable baselines before any offline computation or new rollout."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    costs, matrices, features, matrix_hashes, feature_hashes = load_paper_costs(), {}, {}, {}, {}
    for seed in args.matrix_seeds:
        path = ROOT / f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{seed}.npy"
        bo = ROOT / f"data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed{seed}_bo_inputs.npz"
        data = load_data(path, bo)
        np.testing.assert_allclose(costs, data.per_cell_cost, rtol=0, atol=1e-12)
        matrices[seed], features[seed] = data.matrix, data.features
        matrix_hashes[str(seed)], feature_hashes[str(seed)] = sha256(path), hashlib.sha256(data.features.numpy().astype("<f8").tobytes()).hexdigest()
    shapes = {m.shape for m in matrices.values()}
    if len(shapes) != 1:
        raise ValueError("Matrix shapes must match")
    n_arms, n_examples = next(iter(shapes))
    sources = ["src/correlated_gittins_policy.py", "src/marginal_gittins_lookup.py",
               "src/correlated_ts_policy.py", "src/gittins_lookup.py", "src/q_estimation.py",
               "src/gittins_shrinking_posterior.py", "scripts/compare_correlated_gittins.py"]
    settings = dict(model_version=MODEL_VERSION, methods=["gittins", "correlated_ts", POLICY],
                    matrix_seeds=args.matrix_seeds, run_seeds=args.run_seeds, n_arms=n_arms,
                    n_examples=n_examples, batch_size=args.batch_size, budget_fraction=args.eval_budget_fraction,
                    per_cell_noise_variance=.25, recommendation_std_penalty=1.,
                    priors={k: list(v) for k, v in PRIORS.items()}, target="complete_fixed_benchmark_empirical_mean",
                    cost_scaling_factor=args.cost_scaling_factor, kernel_mode=args.kernel_mode,
                    fit_every=args.fit_every, fit_maxiter=args.fit_maxiter,
                    variance_grid_size=args.variance_grid_size, gittins_grid_points=args.gittins_grid_points,
                    max_steps=args.max_steps, stopping="record_first_index_signals_then_continue_to_budget",
                    unit_exhaustive_cost=n_arms*n_examples,
                    heterogeneous_exhaustive_cost=float(n_examples*costs.sum()),
                    matrix_sha256=matrix_hashes, ordered_feature_sha256=feature_hashes,
                    ordered_per_cell_cost_sha256=hashlib.sha256(costs.astype("<f8").tobytes()).hexdigest(),
                    source_sha256={f: sha256(ROOT/f) for f in sources}, baseline_simulation_allowed=False,
                    approximation="marginal_single_arm_stopping_no_future_cross_arm_learning",
                    optimality_guarantee_for_correlated_arms=False)
    independent = validate_reference_study(settings, args.gittins_results_dir, matrices, costs)
    g_manifest = json.loads((args.gittins_results_dir/"manifest.json").read_text())
    if g_manifest.get("cost_scaling_factor") != args.cost_scaling_factor:
        raise ValueError("Independent Gittins cost scaling differs")
    cpath = args.correlated_ts_results_dir/"manifest.json"
    if not cpath.is_file():
        raise FileNotFoundError("Existing correlated TS results are required; they will not be rerun")
    c_manifest = json.loads(cpath.read_text())
    for key in ("batch_size", "budget_fraction", "per_cell_noise_variance", "recommendation_std_penalty",
                "priors", "target", "kernel_mode", "fit_every", "fit_maxiter", "ordered_per_cell_cost_sha256"):
        if c_manifest.get(key) != settings[key]:
            raise ValueError(f"Correlated TS reference differs: {key}")
    if c_manifest.get("max_steps") is not None:
        raise ValueError("Correlated TS reference must be a complete experiment")
    if c_manifest.get("source_sha256", {}).get("src/correlated_ts_policy.py") != settings["source_sha256"]["src/correlated_ts_policy.py"]:
        raise ValueError("Correlated TS belief implementation differs from the reference")
    hashes = {}
    for seed in args.matrix_seeds:
        if c_manifest.get("matrix_sha256", {}).get(str(seed)) != matrix_hashes[str(seed)] or c_manifest.get("ordered_feature_sha256", {}).get(str(seed)) != feature_hashes[str(seed)]:
            raise ValueError("Correlated TS reference data or features differ")
        for run in args.run_seeds:
            for prior in PRIORS:
                for mode in ("unit", "cost"):
                    name=f"matrix{seed}_seed{run}_{prior}_{mode}.npz"
                    path=args.correlated_ts_results_dir/"runs"/name
                    with np.load(path, allow_pickle=False) as archive:
                        identity=(seed,run,prior,mode)
                        _validate_case_identity(archive,c_manifest,identity,path)
                        _validate_reference_case(archive,settings,identity,matrices[seed],costs,path,methods=("correlated_ts",))
                    hashes[name]=sha256(path)
    settings["baseline_references"]={"gittins":independent,
        "correlated_ts":dict(directory=str(args.correlated_ts_results_dir.resolve()),
                              settings_hash=c_manifest["settings_hash"],manifest_sha256=sha256(cpath),case_sha256=hashes)}
    settings["settings_hash"]=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    path=args.out_dir/"manifest.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Different study/source settings; use a separate output directory")
    if not path.exists():
        temp=path.with_name(f"manifest.{args.worker_index}.tmp")
        temp.write_text(json.dumps(settings,indent=2)+"\n");temp.replace(path)
    return settings


def prepare_tables(args, settings):
    from marginal_gittins_lookup import MarginalGittinsRootTable, build_marginal_gittins_root_table
    prices=np.unique(np.r_[1.,load_paper_costs()])
    tables={}
    for name,(_,variance) in PRIORS.items():
        path=args.out_dir/f"marginal_roots_{name}.npz"
        expected=dict(n_examples=settings["n_examples"],batch_size=args.batch_size,
                      tau_sq_cell=.25,prior_variance=variance,cost_scaling_factor=args.cost_scaling_factor,
                      n_points=args.gittins_grid_points,n_arms=settings["n_arms"])
        if path.exists():
            table=MarginalGittinsRootTable.load(path)
            for key,value in expected.items():
                if getattr(table,key)!=value: raise ValueError(f"Wrong root cache: {key}")
            if len(table.effective_prior_variances)!=args.variance_grid_size or not np.array_equal(table.costs_per_arm,prices):
                raise ValueError("Wrong root cache variance or cost grid")
        else:
            start=time.perf_counter()
            table=build_marginal_gittins_root_table(**expected,costs_per_arm=prices,n_variance_points=args.variance_grid_size)
            table.save(path)
            print(json.dumps({"offline_prior":name,"elapsed_s":time.perf_counter()-start}),flush=True)
        tables[name]=table
    return tables


def run_worker(args,settings,tables):
    pairs=[(m,s) for m in args.matrix_seeds for s in args.run_seeds]
    costs,cache=load_paper_costs(),{}
    completed=computed=0
    for i,(matrix_seed,seed) in enumerate(pairs):
        if i%args.workers!=args.worker_index:continue
        if matrix_seed not in cache:
            cache[matrix_seed]=load_data(ROOT/f"data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{matrix_seed}.npy",ROOT/f"data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed{matrix_seed}_bo_inputs.npz")
        data=cache[matrix_seed]
        for prior,(mean,variance) in PRIORS.items():
            for mode in ("unit","cost"):
                name=f"matrix{matrix_seed}_seed{seed}_{prior}_{mode}.npz"; path=args.out_dir/"runs"/name
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        if z['settings_hash'].item()!=settings['settings_hash']:raise ValueError("Stale correlated Gittins case")
                else:
                    case=dict(matrix_seed=matrix_seed,run_seed=seed,prior_name=prior,cost_mode=mode,settings_hash=settings["settings_hash"])
                    for method in BASELINES:
                        ref=settings["baseline_references"][method]; source=Path(ref["directory"])/"runs"/name
                        if sha256(source)!=ref['case_sha256'][name]:raise ValueError("Baseline source changed")
                        case.update({f"{method}_reference_case":str(source),f"{method}_reference_sha256":ref['case_sha256'][name],f"{method}_reference_settings_hash":ref['settings_hash']})
                    trace=simulate_correlated_gittins(data.matrix,features=data.features,root_table=tables[prior],seed=seed,prior_mean=mean,prior_variance=variance,batch_size=args.batch_size,per_arm_cost=costs,cost_mode=mode,budget_fraction=args.eval_budget_fraction,cost_scaling_factor=args.cost_scaling_factor,kernel_mode=args.kernel_mode,fit_every=args.fit_every,fit_maxiter=args.fit_maxiter,max_steps=args.max_steps)
                    case.update({f"{POLICY}_{key}":value for key,value in trace.items()})
                    case[f"{POLICY}_auc_normalized"]=exact_normalized_auc(trace,args.eval_budget_fraction)
                    _atomic_npz(path,**case);computed+=1
                completed+=1
                progress=dict(worker_index=args.worker_index,cases_complete=completed,new_correlated_gittins_cases=computed,baseline_simulations=0,matrix_seed=matrix_seed,run_seed=seed,prior=prior,cost_mode=mode)
                (args.out_dir/f"worker_{args.worker_index}_progress.json").write_text(json.dumps(progress,indent=2))
                print(json.dumps(progress),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=ROOT/'outputs/gsm8k_correlated_gittins_lcb100')
    p.add_argument('--gittins-results-dir',type=Path,default=ROOT/'outputs/gsm8k_gittins_ts_batch_lcb100')
    p.add_argument('--correlated-ts-results-dir',type=Path,default=ROOT/'outputs/gsm8k_correlated_ts_reuse_lcb100')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--matrix-seeds',type=int,nargs='+',default=list(range(1,6)));p.add_argument('--run-seeds',type=int,nargs='+',default=list(range(20)))
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--eval-budget-fraction',type=float,default=.1)
    p.add_argument('--cost-scaling-factor',type=float,default=1e-4);p.add_argument('--gittins-grid-points',type=int,default=1025);p.add_argument('--variance-grid-size',type=int,default=65)
    p.add_argument('--kernel-mode',choices=['fixed','fit'],default='fit');p.add_argument('--fit-every',type=int,default=32);p.add_argument('--fit-maxiter',type=int,default=25);p.add_argument('--max-steps',type=int)
    a=p.parse_args()
    if not 0<=a.worker_index<a.workers or a.batch_size<1 or not 0<a.eval_budget_fraction<=1:p.error('Invalid worker, batch, or budget setting')
    if not a.matrix_seeds or not set(a.matrix_seeds)<=set(range(1,6)) or len(set(a.matrix_seeds))!=len(a.matrix_seeds) or not a.run_seeds or min(a.run_seeds)<0 or len(set(a.run_seeds))!=len(a.run_seeds):p.error('Use distinct matrix seeds1..5 and nonnegative run seeds')
    if a.fit_every<1 or a.fit_maxiter<1 or a.variance_grid_size<3 or a.gittins_grid_points<3 or a.cost_scaling_factor<=0 or (a.max_steps is not None and a.max_steps<1):p.error('Invalid fitting or numerical settings')
    torch.set_num_threads(1)
    settings=prepare(a);tables=prepare_tables(a,settings)
    if not a.prepare_only:run_worker(a,settings,tables)


if __name__=='__main__':main()
