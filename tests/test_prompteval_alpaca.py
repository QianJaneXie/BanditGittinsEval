from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from prompteval import bai_evaluation as bai
from prompteval.build_banditeval_pickle import load_matrix


def test_alpaca_loader_preserves_bounded_continuous_scores(tmp_path):
    path = tmp_path / "alpaca_eval_test.npy"
    expected = np.array([[0.0, 0.25, 0.75, 1.0]], dtype=np.float32)
    np.save(path, expected)

    actual = load_matrix(path, allow_continuous=True)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    with pytest.raises(ValueError, match="expected binary values"):
        load_matrix(path)


def test_fit_is_binarized_but_regret_uses_raw_continuous_scores():
    raw = np.array(
        [
            [0.49, 0.49, 0.49, 0.49],
            [0.90, 0.80, 0.70, 0.60],
            [0.10, 0.20, 0.30, 0.40],
            [0.51, 0.51, 0.51, 0.51],
        ],
        dtype=np.float64,
    )
    fitted_targets = []

    class RecordingLogReg:
        def fit(self, seen_examples, targets, features):
            fitted_targets.append(np.asarray(targets).copy())
            # Deterministically select arm 1 in the first phase.
            self.thetas = np.array([0.0, 4.0, 1.0, 2.0])

    with patch.object(bai, "LogReg", RecordingLogReg):
        updates = bai.compute_regrets(
            budget=4,
            Y=raw,
            X=None,
            random_seed=0,
            fit_binarize_threshold=0.5,
        )

    assert fitted_targets
    assert set(np.unique(fitted_targets[0])).issubset({0, 1})
    np.testing.assert_array_equal(fitted_targets[0], raw >= 0.5)

    first = updates[0]
    assert first["chosen_arm"] == 1
    assert first["chosen_mean"] == pytest.approx(raw[1].mean())
    assert first["oracle_mean"] == pytest.approx(raw.mean(axis=1).max())
    assert first["simple_regret"] == pytest.approx(0.0)


def test_alpaca_default_threshold_matches_paper():
    assert bai.ALPACA_FIT_BINARIZE_THRESHOLD == pytest.approx(0.5)


def test_cost_loader_ignores_metadata_key(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(
        json.dumps(
            {
                "_metadata": {"dataset": "alpaca_eval"},
                "0": {"estimated_cost_per_1m_input_tokens": 1.5},
                "1": {"estimated_cost_per_1m_input_tokens": 2.5},
            }
        ),
        encoding="utf-8",
    )

    np.testing.assert_array_equal(bai.load_arm_costs(str(path)), [1.5, 2.5])
