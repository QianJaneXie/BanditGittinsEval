"""Runner accounting and read-only reuse of the two completed baselines."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import compare_correlated_gittins as runner
from marginal_gittins_lookup import build_marginal_gittins_root_table


@pytest.fixture(scope='module')
def table():
    return build_marginal_gittins_root_table(prior_variance=.04,tau_sq_cell=.25,
        n_examples=11,batch_size=4,costs_per_arm=np.array([.25,.45,1.,3.2]),
        n_arms=3,n_variance_points=5,n_points=257,cost_scaling_factor=1e-4)


@pytest.fixture
def data():
    matrix=np.array([[0,1,0,0,1,0,0,0,0,0,0],
                     [1,1,1,0,1,1,1,0,1,1,1],
                     [0,0,0,0,0,0,0,0,0,0,0]],dtype=np.float32)
    features=torch.tensor([[0.,0.],[0.,1.],[1.,0.]],dtype=torch.float64)
    return matrix,features,np.array([3.2,.25,.45])


@pytest.mark.parametrize('mode',['unit','cost'])
@pytest.mark.parametrize('kernel',['fixed','fit'])
def test_new_policy_only_evaluates_fresh_items_and_reaches_exact_terminal_scores(table,data,mode,kernel):
    matrix,features,costs=data
    trace=runner.simulate_correlated_gittins(matrix,features=features,root_table=table,seed=4,
        prior_mean=.5,prior_variance=.04,batch_size=4,per_arm_cost=costs,cost_mode=mode,
        budget_fraction=1.,kernel_mode=kernel,fit_every=2,fit_maxiter=3)
    seen=np.zeros_like(matrix,dtype=bool);spent=0.
    for i,(arm,size) in enumerate(zip(trace['pulled_arm'],trace['n_new_samples'])):
        columns=trace['sampled_columns'][i,:size]
        assert not seen[arm,columns].any()
        assert len(np.unique(columns))==size
        seen[arm,columns]=True;spent+=size*costs[arm]
        assert trace['x_original_cost'][i]==pytest.approx(spent)
    assert seen.all();assert trace['x'][-1]==matrix.size
    assert 3 in trace['n_new_samples']
    assert trace['recommended_arm'][-1]==1
    assert trace['regret'][-1]==0.
    assert trace['independent_arm_optimality_guarantee'] is False


def test_marginal_indices_use_actual_prices_in_the_cost_condition(table,data):
    matrix,features,costs=data
    common=dict(features=features,root_table=table,seed=4,prior_mean=.5,prior_variance=.04,
                batch_size=4,per_arm_cost=costs,budget_fraction=1.,kernel_mode='fixed',max_steps=1)
    unit=runner.simulate_correlated_gittins(matrix,cost_mode='unit',**common)
    cost=runner.simulate_correlated_gittins(matrix,cost_mode='cost',**common)
    assert unit['pulled_arm'][0]==0
    assert cost['pulled_arm'][0]==1


@pytest.fixture
def reference_study(tmp_path, monkeypatch, data):
    """Build tiny stored histories; no baseline policy or local result is used."""
    workspace = tmp_path/'workspace'
    matrix, features, prices = data
    n_arms, n_examples = matrix.shape
    matrix_path = workspace/'data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy'
    archive_path = workspace/'data/bo_inputs/gsm8k/gsm8k_1_samples_various_models_seed1_bo_inputs.npz'
    matrix_path.parent.mkdir(parents=True)
    archive_path.parent.mkdir(parents=True)
    np.save(matrix_path, matrix)
    np.savez_compressed(archive_path, X=features.numpy(), arm_ids=np.arange(n_arms),
        cost=prices*n_examples, metadata_json=json.dumps({'n_examples': n_examples}),
        Y=np.full((n_arms, 1), np.nan), raw_scores_mean=np.full(n_arms, np.nan))
    source_paths = ['src/correlated_gittins_policy.py', 'src/marginal_gittins_lookup.py',
        'src/correlated_ts_policy.py', 'src/gittins_lookup.py', 'src/q_estimation.py',
        'src/gittins_shrinking_posterior.py', 'scripts/compare_correlated_gittins.py']
    for relative in source_paths:
        target = workspace/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    monkeypatch.setattr(runner, 'ROOT', workspace)
    monkeypatch.setattr(runner, 'load_paper_costs', lambda: prices.copy())
    matrix_digest = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    price_digest = hashlib.sha256(prices.astype('<f8').tobytes()).hexdigest()
    feature_digest = hashlib.sha256(features.numpy().astype('<f8').tobytes()).hexdigest()
    common = dict(matrix_seeds=[1], run_seeds=[0], n_examples=n_examples,
        batch_size=8, budget_fraction=.1, per_cell_noise_variance=.25,
        recommendation_std_penalty=1., priors={k:list(v) for k,v in runner.PRIORS.items()},
        matrix_sha256={'1':matrix_digest})
    independent = dict(**common, n_models=n_arms, full_evals=matrix.size,
        full_original_cost=float(n_examples*prices.sum()),
        ordered_cost_vector_sha256=price_digest, gittins_target='finite_population_mean',
        ts_target='finite_population_mean', gittins_model_version='batch_level_paper_v1',
        gittins_dp_transitions='batch', gittins_dp_horizon=(n_examples+7)//8,
        cost_scaling_factor=1e-4, gittins_grid_points=1025)
    correlated = dict(**common, n_arms=n_arms,
        model_version='categorical_empirical_correlated_ts_v1',
        target='complete_fixed_benchmark_empirical_mean', kernel_mode='fit',
        fit_every=32, fit_maxiter=25, max_steps=None,
        ordered_per_cell_cost_sha256=price_digest,
        ordered_feature_sha256={'1':feature_digest},
        source_sha256={'src/correlated_ts_policy.py':
            hashlib.sha256((workspace/'src/correlated_ts_policy.py').read_bytes()).hexdigest()})
    directories = {'gittins':workspace/'references/gittins',
                   'correlated_ts':workspace/'references/correlated_ts'}
    true_means = matrix.astype(np.float64).mean(1)
    initial_regret = float(true_means.max()-true_means[0])
    for name, manifest in [('gittins', independent), ('correlated_ts', correlated)]:
        directory = directories[name]
        (directory/'runs').mkdir(parents=True)
        manifest['settings_hash'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        (directory/'manifest.json').write_text(json.dumps(manifest))
        for prior in runner.PRIORS:
            for mode in ['unit', 'cost']:
                case = dict(matrix_seed=1, run_seed=0, prior_name=prior, cost_mode=mode,
                            settings_hash=manifest['settings_hash'])
                methods = ['ts', 'gittins'] if name=='gittins' else ['correlated_ts']
                for method in methods:
                    full = float(matrix.size if mode=='unit' else n_examples*prices.sum())
                    budget = float(max(1,round(.1*full))) if mode=='unit' else .1*full
                    # One recorded batch crosses/reaches the tiny nominal
                    # budget. These arrays test stored accounting, not a policy.
                    size = int(budget) if mode=='unit' and method!='gittins' else 8
                    trace = dict(x=np.array([size]), x_original_cost=np.array([size*prices[0]]),
                        pulled_arm=np.array([0]), n_new_samples=np.array([size]),
                        recommended_arm=np.array([1]), recommended_mean=np.array([runner.PRIORS[prior][0]]),
                        regret=np.array([0.]), budget=budget, full_budget=full,
                        elapsed_s=.001, cost_mode=mode, initial_regret=initial_regret)
                    trace['auc_normalized'] = runner.exact_normalized_auc(trace, .1)
                    case.update({f'{method}_{key}':value for key,value in trace.items()})
                np.savez_compressed(directory/'runs'/f'matrix1_seed0_{prior}_{mode}.npz', **case)
    roots = {f'{prior}_{mode}_roots':np.zeros((n_arms, independent['gittins_dp_horizon']+1), dtype=np.float32)
             for prior in runner.PRIORS for mode in ['unit','cost']}
    np.savez_compressed(directories['gittins']/'roots.npz', settings_hash=independent['settings_hash'], **roots)
    args = SimpleNamespace(out_dir=workspace/'results', matrix_seeds=[1], run_seeds=[0],
        batch_size=8, eval_budget_fraction=.1, cost_scaling_factor=1e-4,
        kernel_mode='fit', fit_every=32, fit_maxiter=25, variance_grid_size=65,
        gittins_grid_points=1025, max_steps=None, worker_index=0, workers=1,
        gittins_results_dir=directories['gittins'], correlated_ts_results_dir=directories['correlated_ts'])
    return args, matrix, prices


def test_preflight_reuses_existing_baselines_without_any_baseline_simulation(reference_study,monkeypatch):
    import compare_gittins_ts
    import compare_correlated_ts
    def forbidden(*args,**kwargs):
        raise AssertionError('A previously completed baseline must not be run again')
    monkeypatch.setattr(compare_gittins_ts,'simulate_policy',forbidden)
    monkeypatch.setattr(compare_correlated_ts,'simulate_correlated_ts',forbidden)
    args, _, _ = reference_study
    settings=runner.prepare(args)
    assert settings['baseline_simulation_allowed'] is False
    assert set(settings['baseline_references'])=={'gittins','correlated_ts'}
    assert list(args.out_dir.iterdir())==[args.out_dir/'manifest.json']
    saved=(args.out_dir/'manifest.json').read_bytes()
    assert runner.prepare(args)==settings
    assert (args.out_dir/'manifest.json').read_bytes()==saved


def test_worker_stores_no_copies_of_baseline_trajectories(reference_study,monkeypatch):
    import compare_gittins_ts
    import compare_correlated_ts
    args,matrix,prices=reference_study
    def forbidden(*args,**kwargs):
        raise AssertionError('A previously completed baseline must not be run again')
    monkeypatch.setattr(compare_gittins_ts,'simulate_policy',forbidden)
    monkeypatch.setattr(compare_correlated_ts,'simulate_correlated_ts',forbidden)
    settings=runner.prepare(args)
    baseline_files=[path for directory in [args.gittins_results_dir,args.correlated_ts_results_dir]
                    for path in directory.rglob('*') if path.is_file()]
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in baseline_files}
    def new_policy_only(truth,**kwargs):
        mode=kwargs['cost_mode']
        full=float(truth.size if mode=='unit' else truth.shape[1]*prices.sum())
        return dict(x=np.array([8]),x_original_cost=np.array([8*prices[0]]),regret=np.array([0.]),
                    cost_mode=mode,full_budget=full,
                    initial_regret=float(matrix.mean(1).max()-matrix.mean(1)[0]))
    monkeypatch.setattr(runner,'simulate_correlated_gittins',new_policy_only)
    runner.run_worker(args,settings,{'G':object(),'S':object()})
    files=list((args.out_dir/'runs').glob('*.npz'));assert len(files)==4
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            assert 'correlated_gittins_x' in z
            assert 'gittins_reference_case' in z and 'correlated_ts_reference_case' in z
            assert not any(k in z for k in ['gittins_x','correlated_ts_x','ts_x'])
    assert before=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in baseline_files}
