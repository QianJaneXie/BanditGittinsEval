"""FormatSpread binary/fractional updates, budget feasibility and runner wiring."""
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from formatspread_ts_policy import beta_parameters, formatspread_thompson_sampling
from simple_regret_recommend import empirical_incumbent


def test_binary_and_fractional_parameters():
    obs = torch.tensor([[1., 0., float('nan')], [.2, .8, .5]], dtype=torch.float64)
    a, b = beta_parameters(obs)
    torch.testing.assert_close(a, torch.tensor([2., 2.5], dtype=torch.float64))
    torch.testing.assert_close(b, torch.tensor([2., 2.5], dtype=torch.float64))


@pytest.mark.parametrize('value', [-2., 1.01, float('inf')])
def test_reject_invalid_scores(value):
    with pytest.raises(ValueError, match='scores'):
        beta_parameters(torch.tensor([[value]]))


def test_affordable_batch_completed_arm_and_repeatability():
    obs = torch.tensor([[1., 1., 1.], [float('nan')] * 3, [float('nan')] * 3])
    kwargs = dict(batch_size=8, per_arm_cost=[1., 10., 2.], remaining_cost=3.)
    torch.manual_seed(9)
    a = formatspread_thompson_sampling(obs, **kwargs)
    torch.manual_seed(9)
    b = formatspread_thompson_sampling(obs, **kwargs)
    assert torch.equal(a, b)
    assert a.shape == (2, 1) and a[0, 0] == 2
    assert empirical_incumbent(obs)[0] == 0
    assert formatspread_thompson_sampling(obs, per_arm_cost=[1., 10., 2.], remaining_cost=1.) is None
    assert formatspread_thompson_sampling(obs, max_batch_size=0) is None


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_wandb_runner_fractional_cost_budget():
    runner = load_runner('run_simple_regret_wandb')
    variant = runner.parse_experiment_variant('formatspread_ts_cost_B8')
    args = SimpleNamespace(formatspread_prior_alpha=1., formatspread_prior_beta=1.,
                           eval_budget_fraction=.4, run_seed=7)
    result = runner.run_simple_regret_experiment(
        ground_truth=torch.tensor([[.2, .8, .6], [.9, .4, .1]]),
        original_cost_per_arm=torch.tensor([2., 5.], dtype=torch.float64),
        variant=variant, args=args, prior_mean=.5, prior_variance=.04,
        max_evaluations=6, max_original_cost=8.4, run=None, log_step_metrics=False,
    )
    assert result['x_original_cost'][-1] <= 8.4
    assert result['x'][-1] > 0


def test_simulation_cli_fractional_scores_and_reproducibility(tmp_path):
    import subprocess
    import numpy as np
    matrix = tmp_path / 'matrix.npy'
    costs = tmp_path / 'costs.npy'
    np.save(matrix, np.array([[.2, .7, .4], [.3, .8, .9]], dtype=np.float32))
    np.save(costs, np.array([2., 5.]))
    outputs = [tmp_path / 'a.npz', tmp_path / 'b.npz']
    for output in outputs:
        subprocess.run([
            sys.executable, str(ROOT / 'scripts/simulate_simple_regret.py'),
            '--matrix', str(matrix), '--cost-vector', str(costs),
            '--algorithms', 'formatspread_ts', '--formatspread-batch-size', '3',
            '--eval-budget-fraction', '.4', '--seed', '42', '--out', str(output),
        ], check=True, capture_output=True, text=True)
    with np.load(outputs[0]) as a, np.load(outputs[1]) as b:
        assert a['formatspread_ts_label'] == 'FormatSpread-TS'
        assert a['formatspread_ts_x_original_cost'][-1] <= 8.4
        for key in a.files:
            if key.startswith('formatspread_ts'):
                np.testing.assert_equal(a[key], b[key])


def test_original_prior_parameterization_and_dataset_means():
    from formatspread_ts_policy import formatspread_prior, dataset_prior_mean
    for dataset, x in [('gsm8k', .2), ('piqa', .4), ('alpaca', .2)]:
        assert dataset_prior_mean(dataset + '.npy') == x
        a, b = formatspread_prior(x)
        assert b == 5
        assert a / (a+b) == pytest.approx(x)
    assert formatspread_prior(.01) == (1.1, 5.)
    with pytest.raises(ValueError):
        formatspread_prior(0.)
    with pytest.raises(ValueError):
        formatspread_prior(.2, alpha=1.)
