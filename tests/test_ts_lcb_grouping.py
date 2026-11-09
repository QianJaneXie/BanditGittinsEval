"""Separate TS recommendation rules without relabelling historical traces."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_bandit_bo_comparison import ts_recommendation_label as bo_ts_label
from plot_simple_regret_results import ts_recommendation_label
from plot_wandb_simple_regret_curves import derive_method_label_row, validate_recommendation_groups


def ts_row(penalty, **overrides):
    row = dict(
        policy_family="ts", experiment_variant="ts_unit_B8_dataset",
        cost_mode="unit", batch_size=8, prior_type="dataset",
        recommendation_std_penalty=penalty,
        recommendation_rule=("finite_population_posterior_mean_minus_std" if penalty > 0
                             else "finite_population_posterior_mean"),
    )
    row.update(overrides)
    return row


@pytest.mark.parametrize("family,variant", [
    ("ts", "ts_unit_B8_dataset"), (None, "ts_cost_B16_default"),
    ("correlated_ts", "correlated_ts"), (None, "correlated_ts_unit_B8"),
])
def test_wandb_labels_and_default_groups_separate_mean_and_lcb(family, variant):
    frame = pd.DataFrame([ts_row(p, policy_family=family, experiment_variant=variant) for p in (0., 1.)])
    frame["method_label"] = frame.apply(derive_method_label_row, axis=1)
    assert frame["method_label"].nunique() == 2
    assert "1 std" in frame["method_label"].iloc[1]
    validate_recommendation_groups(frame, "method_label")
    frame["same_custom_group"] = "TS"
    with pytest.raises(ValueError, match="recommendation rules, or std penalties"):
        validate_recommendation_groups(frame, "same_custom_group")


def test_custom_label_derivation_is_idempotent_and_can_change_rule():
    row = pd.Series(ts_row(1., method_label="Student TS"))
    label = derive_method_label_row(row)
    assert label.startswith("Student TS [recommend:")
    row["method_label"] = label
    assert derive_method_label_row(row) == label
    row["recommendation_std_penalty"] = 0.
    row["recommendation_rule"] = "finite_population_posterior_mean"
    assert derive_method_label_row(row) == "Student TS"


def test_same_rule_with_different_coefficients_cannot_be_pooled():
    frame = pd.DataFrame([ts_row(p) for p in (.5, 1.)])
    frame["method_label"] = frame.apply(derive_method_label_row, axis=1)
    assert frame["method_label"].nunique() == 2
    validate_recommendation_groups(frame, "method_label")
    with pytest.raises(ValueError, match="std penalties"):
        validate_recommendation_groups(frame, "experiment_variant")


def test_different_mean_rules_do_not_share_a_label_or_aggregation_key():
    frame = pd.DataFrame([ts_row(0.), ts_row(0., recommendation_rule="empirical_mean")])
    frame["method_label"] = frame.apply(derive_method_label_row, axis=1)
    assert frame["method_label"].nunique() == 2
    validate_recommendation_groups(frame, "method_label")
    with pytest.raises(ValueError, match="recommendation rules"):
        validate_recommendation_groups(frame, "experiment_variant")


def test_legacy_csv_and_explicit_mean_rule_remain_one_group():
    legacy = ts_row(0.)
    del legacy["recommendation_rule"]
    del legacy["recommendation_std_penalty"]
    frame = pd.DataFrame([legacy, ts_row(0.)])
    frame["method_label"] = frame.apply(derive_method_label_row, axis=1)
    assert frame["method_label"].nunique() == 1
    validate_recommendation_groups(frame, "experiment_variant")


@pytest.mark.parametrize("labeler", [ts_recommendation_label, bo_ts_label])
def test_legacy_local_ts_ignores_the_shared_gittins_lcb_metadata(labeler):
    traces = {"recommendation_std_penalty": np.array(1.),
              "recommendation_rule": np.array("finite_population_posterior_mean_minus_std")}
    label = labeler(traces)
    assert label == "Gaussian TS (recommend: posterior full test-set mean)"
    assert "- 1 std" not in label


@pytest.mark.parametrize("labeler", [ts_recommendation_label, bo_ts_label])
def test_local_ts_uses_its_own_explicit_lcb_setting(labeler):
    traces = {"recommendation_std_penalty": np.array(0.),
              "ts_recommendation_std_penalty": np.array(1.),
              "ts_recommendation_rule": np.array("finite_population_posterior_mean_minus_std")}
    assert "full test-set mean - 1 std" in labeler(traces)
    traces["ts_recommendation_std_penalty"] = np.array(0.)
    traces["ts_recommendation_rule"] = np.array("finite_population_posterior_mean")
    assert "- 1 std" not in labeler(traces)


@pytest.mark.parametrize("bad", [-1., np.inf, "invalid"])
def test_invalid_csv_penalty_is_not_silently_grouped_with_mean_only(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        derive_method_label_row(pd.Series(ts_row(0., recommendation_std_penalty=bad)))


def test_custom_groups_cannot_combine_independent_and_correlated_ts():
    frame = pd.DataFrame([ts_row(1.), ts_row(1., policy_family="correlated_ts", experiment_variant="correlated_ts")])
    frame["same_custom_group"] = "TS"
    with pytest.raises(ValueError, match="TS policies"):
        validate_recommendation_groups(frame, "same_custom_group")
