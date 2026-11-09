#!/usr/bin/env python3
"""Run the correlation-aware marginal-index Gittins heuristic on one matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from compare_correlated_gittins import MODEL_VERSION, POLICY, simulate_correlated_gittins
from compare_gittins_ts import _atomic_npz, exact_normalized_auc
from run_mfkg_baseline import load_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--bo-inputs',type=Path)
    parser.add_argument('--cost-vector',type=Path)
    parser.add_argument('--cost-aware',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--root-cache',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--prior-mean',type=float,default=.5)
    parser.add_argument('--prior-variance',type=float,default=.04)
    parser.add_argument('--tau-sq-cell',type=float,default=.25)
    parser.add_argument('--cost-scaling-factor',type=float,default=1e-4)
    parser.add_argument('--eval-budget-fraction',type=float,default=.1)
    parser.add_argument('--recommendation-std-penalty',type=float,default=1.)
    parser.add_argument('--kernel-mode',choices=('fixed','fit'),default='fit')
    parser.add_argument('--fit-every',type=int,default=32)
    parser.add_argument('--fit-maxiter',type=int,default=25)
    parser.add_argument('--variance-grid-size',type=int,default=65)
    parser.add_argument('--gittins-grid-points',type=int,default=1025)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--allow-early-stop',action='store_true',help='End at the heuristic index signal; default records it and continues to budget.')
    args=parser.parse_args()
    if args.seed<0 or args.batch_size<1 or not 0<args.eval_budget_fraction<=1:
        parser.error('Use a nonnegative seed, positive batch size, and budget fraction in (0,1]')
    if args.fit_every<1 or args.fit_maxiter<1 or args.variance_grid_size<2 or args.gittins_grid_points<3:
        parser.error('Invalid fitting or numerical settings')
    data=load_data(args.matrix,args.bo_inputs,args.cost_vector)
    n_arms,n_examples=data.matrix.shape
    variance_points=2 if n_arms==1 else args.variance_grid_size
    mode='cost' if args.cost_aware or args.cost_vector is not None else 'unit'
    prices=np.unique(data.per_cell_cost if mode=='cost' else np.ones(n_arms))
    parameters=dict(n_examples=n_examples,n_arms=n_arms,batch_size=args.batch_size,
                    tau_sq_cell=args.tau_sq_cell,prior_variance=args.prior_variance,
                    cost_scaling_factor=args.cost_scaling_factor,n_points=args.gittins_grid_points)
    cache_identity=dict(**parameters,costs_per_arm=prices.tolist(),n_variance_points=variance_points)
    fingerprint=hashlib.sha256(json.dumps(cache_identity,sort_keys=True).encode()).hexdigest()[:16]
    root_cache=args.root_cache or Path('outputs/correlated_gittins_root_cache')/f'roots_{fingerprint}.npz'
    from marginal_gittins_lookup import MarginalGittinsRootTable, build_marginal_gittins_root_table
    start=time.perf_counter()
    if root_cache.exists():
        table=MarginalGittinsRootTable.load(root_cache,**parameters,n_variance_points=variance_points)
        if not np.isin(prices,table.costs_per_arm).all():raise ValueError('Root cache does not cover the requested raw prices')
    else:
        table=build_marginal_gittins_root_table(**parameters,costs_per_arm=prices,n_variance_points=variance_points)
        table.save(root_cache)
    offline_s=time.perf_counter()-start
    if args.prepare_only:
        print(f'Root cache ready: {root_cache}')
        return 0
    trace=simulate_correlated_gittins(
        data.matrix,features=data.features,root_table=table,seed=args.seed,
        prior_mean=args.prior_mean,prior_variance=args.prior_variance,batch_size=args.batch_size,
        per_arm_cost=data.per_cell_cost,cost_mode=mode,budget_fraction=args.eval_budget_fraction,
        tau_sq_cell=args.tau_sq_cell,std_penalty=args.recommendation_std_penalty,
        cost_scaling_factor=args.cost_scaling_factor,kernel_mode=args.kernel_mode,
        fit_every=args.fit_every,fit_maxiter=args.fit_maxiter,max_steps=args.max_steps,
        allow_early_stop=args.allow_early_stop)
    metadata=dict(model_version=MODEL_VERSION,policy_family=POLICY,matrix=str(args.matrix.resolve()),
                  matrix_sha256=hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
                  features_sha256=hashlib.sha256(data.features.numpy().astype('<f8').tobytes()).hexdigest(),
                  ordered_costs_sha256=hashlib.sha256(data.per_cell_cost.astype('<f8').tobytes()).hexdigest(),
                  feature_source=data.feature_source,cost_mode=mode,seed=args.seed,
                  prior_mean=args.prior_mean,prior_variance=args.prior_variance,
                  batch_size=args.batch_size,budget_fraction=args.eval_budget_fraction,
                  recommendation_std_penalty=args.recommendation_std_penalty,
                  kernel_mode=args.kernel_mode,fit_every=args.fit_every,fit_maxiter=args.fit_maxiter,
                  root_cache=str(root_cache.resolve()),root_cache_sha256=hashlib.sha256(root_cache.read_bytes()).hexdigest(),
                  root_parameters=cache_identity,offline_prepare_or_load_s=offline_s,
                  index_approximation='single_arm_stopping_ignores_future_cross_arm_information',
                  correlated_arm_optimality_guarantee=False,
                  budget_average_regret=exact_normalized_auc(trace,args.eval_budget_fraction),
                  allow_early_stop=args.allow_early_stop,max_steps=args.max_steps)
    identity={key:metadata[key] for key in (
        'model_version','matrix_sha256','features_sha256','ordered_costs_sha256','cost_mode',
        'seed','prior_mean','prior_variance','batch_size','budget_fraction',
        'recommendation_std_penalty','kernel_mode','fit_every','fit_maxiter',
        'root_parameters','allow_early_stop','max_steps')}
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    output=args.out or Path('outputs/correlated_gittins')/f'{args.matrix.stem}_{mode}_seed{args.seed}_{run_id}.npz'
    if output.suffix!='.npz':parser.error('out must end in .npz')
    _atomic_npz(output,**trace,metadata_json=json.dumps(metadata))
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved {len(trace["x"])} batches to {output}')
    return 0


if __name__=='__main__':main()
