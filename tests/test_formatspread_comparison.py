"""Incremental comparison must preserve native TS choices and recommendations."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
from compare_formatspread_gittins import simulate
from formatspread_ts_policy import formatspread_thompson_sampling
from simple_regret_recommend import empirical_incumbent


def test_incremental_trace_equals_native():
    matrix = np.array([[1.,0.,1.,1.], [0.,1.,0.,0.], [1.,1.,0.,0.]], dtype=np.float32)
    matrix = np.tile(matrix, (1, 6))
    full_cost = matrix.shape[1] * 6
    costs = np.array([2.,3.,1.])
    trace = simulate(matrix, costs, 'cost', 4, fraction=.8)
    torch.manual_seed(4)
    obs = torch.full(matrix.shape, float('nan'))
    spent = 0.
    arms, recs = [], []
    while True:
        batch = formatspread_thompson_sampling(obs, batch_size=8, prior_alpha=1.25,
                    prior_beta=5., per_arm_cost=costs, remaining_cost=max(0.,.8*full_cost-spent))
        if batch is None:
            break
        obs[batch[0],batch[1]] = torch.from_numpy(matrix)[batch[0],batch[1]]
        arm = int(batch[0,0]); arms.append(arm)
        spent += costs[arm]*batch.shape[1]
        recs.append(empirical_incumbent(obs)[0])
    np.testing.assert_array_equal(trace['pulled_arm'], arms)
    np.testing.assert_array_equal(trace['recommended_arm'], recs)
    assert trace['x_original_cost'][-1] <= .8*full_cost
