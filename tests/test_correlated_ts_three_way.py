"""Prevent incompatible histories from entering the three-policy comparison."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_correlated_ts_comparison import (
    PRIORS, join_three_way_case, summarize_three_way, validate_comparison_manifests,
    validate_shared_ts_trajectory,
)


@pytest.fixture
def manifests():
    common = dict(
        matrix_seeds=list(range(1, 6)), run_seeds=list(range(20)), batch_size=8,
        budget_fraction=.1, per_cell_noise_variance=.25, recommendation_std_penalty=1.,
        priors={k: list(v) for k, v in PRIORS.items()}, n_examples=1000,
        matrix_sha256={str(i): f"matrix-{i}" for i in range(1, 6)},
    )
    current = dict(
        **common, n_arms=122, max_steps=None, ordered_per_cell_cost_sha256="costs",
        unit_exhaustive_cost=122000., heterogeneous_exhaustive_cost=83900.,
        target="complete_fixed_benchmark_empirical_mean",
    )
    previous = dict(
        **common, n_models=122, full_evals=122000, full_original_cost=83900.,
        ordered_cost_vector_sha256="costs", gittins_target="finite_population_mean",
        ts_target="finite_population_mean", gittins_model_version="batch_level_paper_v1",
        gittins_dp_transitions="batch", gittins_dp_horizon=125,
        cost_scaling_factor=1e-4, gittins_grid_points=1025, settings_hash="old-settings",
    )
    return current, previous


def test_same_complete_paper_protocol_allows_reuse(manifests):
    validate_comparison_manifests(*manifests)


@pytest.mark.parametrize("field,value", [
    ("batch_size", 16), ("recommendation_std_penalty", 0.),
    ("budget_fraction", .05), ("per_cell_noise_variance", .03125),
    ("priors", {"G": [.5, .01], "S": [.2, .01]}),
    ("run_seeds", list(range(19))), ("matrix_sha256", {"1": "different"}),
    ("ordered_per_cell_cost_sha256", "different-order"),
    ("heterogeneous_exhaustive_cost", 122000.),
    ("target", "latent_mean"), ("max_steps", 5),
])
def test_protocol_or_data_mismatches_are_rejected(manifests, field, value):
    current, previous = copy.deepcopy(manifests)
    current[field] = value
    with pytest.raises(ValueError):
        validate_comparison_manifests(current, previous)


def test_per_cell_gittins_cannot_enter_three_way_figure(manifests):
    current, previous = copy.deepcopy(manifests)
    previous.update(gittins_dp_transitions="per_cell", gittins_dp_horizon=1000)
    with pytest.raises(ValueError, match="batch-level"):
        validate_comparison_manifests(current, previous)


@pytest.fixture
def ts_trace():
    return {
        "ts_x": np.array([8, 16]), "ts_x_original_cost": np.array([2., 4.]),
        "ts_pulled_arm": np.array([0, 1]), "ts_n_new_samples": np.array([8, 8]),
        "ts_recommended_arm": np.array([0, 1]), "ts_recommended_mean": np.array([.2, .3]),
        "ts_regret": np.array([.1, 0.]), "ts_budget": np.array(16.),
        "ts_full_budget": np.array(160.),
        "ts_sampled_columns": np.array([[0, 1], [2, 3]]),
    }


def test_exact_shared_ts_histories_pass_without_requiring_unrecorded_old_item_ids(ts_trace):
    previous = copy.deepcopy(ts_trace)
    previous.pop("ts_sampled_columns")
    fields = validate_shared_ts_trajectory(ts_trace, previous, case_name="paired")
    assert "ts_pulled_arm" in fields
    assert "ts_recommended_arm" in fields
    assert "ts_sampled_columns" not in fields


@pytest.mark.parametrize("field", ["ts_pulled_arm", "ts_recommended_arm", "ts_x_original_cost", "ts_sampled_columns"])
def test_equal_final_regret_does_not_hide_a_different_ts_trajectory(ts_trace, field):
    previous = copy.deepcopy(ts_trace)
    previous[field].flat[0] += 1
    with pytest.raises(ValueError, match=field):
        validate_shared_ts_trajectory(ts_trace, previous, case_name="paired")


def test_missing_required_shared_ts_action_field_is_rejected(ts_trace):
    previous = copy.deepcopy(ts_trace)
    previous.pop("ts_pulled_arm")
    with pytest.raises(ValueError, match="Missing.*ts_pulled_arm"):
        validate_shared_ts_trajectory(ts_trace, previous)


def test_valid_join_reuses_gittins_without_changing_either_new_ts_trace(ts_trace, tmp_path):
    current_path, old_path = tmp_path / "current.npz", tmp_path / "gittins.npz"
    np.savez(current_path, **ts_trace)
    old_raw = {k: v for k, v in ts_trace.items() if k != "ts_sampled_columns"}
    np.savez(old_path, **old_raw, gittins_elapsed_s=.25)
    identity = dict(matrix_seed=1, run_seed=0, prior="G", mode="unit")
    ts, correlated = {"metrics": {"auc_0_10%": .03}}, {"metrics": {"auc_0_10%": .01}}
    current = dict(**identity, methods={"ts": ts, "correlated_ts": correlated})
    old_gittins = dict(actual_final_budget_fraction=.1, metrics={"auc_0_10%": .02})
    old = dict(**identity, methods={"gittins": old_gittins})
    merged, checked = join_three_way_case(current, old, current_path, old_path)
    assert merged["methods"]["ts"] is ts
    assert merged["methods"]["correlated_ts"] is correlated
    assert merged["methods"]["gittins"]["metrics"] == old_gittins["metrics"]
    assert merged["methods"]["gittins"]["final_budget_fraction"] == .1
    assert "ts_pulled_arm" in checked
    assert "gittins" not in current["methods"]


def test_three_way_summary_includes_all_methods_and_both_paired_comparators(manifests, tmp_path):
    current, previous = manifests
    cases = []
    # Five crossed matrix IDs and twenty shared trial seeds exercise the same
    # grouping and two-way bootstrap used for the paper comparison.
    for matrix in range(1, 6):
        for seed in range(20):
            for prior in PRIORS:
                for mode in ("unit", "cost"):
                    methods = {}
                    for name, mean in (("gittins", .02), ("ts", .03), ("correlated_ts", .01)):
                        value = mean + matrix * .0001 + seed * .00001
                        methods[name] = dict(metrics={"auc_0_10%": value, "regret_at_10%": 0.},
                                             final_budget_fraction=.1, observation_rounds=1525)
                    cases.append(dict(matrix_seed=matrix, run_seed=seed, prior=prior, mode=mode, methods=methods))
    summary = summarize_three_way(current, previous, cases, {"gittins_recomputed": False}, tmp_path)
    panel = summary["panels"]["unit_G"]
    assert set(panel["methods"]) == {"gittins", "ts", "correlated_ts"}
    assert panel["n_paired_trials"] == 100
    for comparator, expected in (("gittins", -.01), ("ts", -.02)):
        stats = panel["paired_differences"][f"correlated_ts_minus_{comparator}"]["auc_0_10%"]
        assert stats["mean_difference"] == pytest.approx(expected)
        assert stats["ci95_two_way_bootstrap_lower"] == pytest.approx(expected)
        assert stats["ci95_two_way_bootstrap_upper"] == pytest.approx(expected)
    saved = json.loads((tmp_path / "three_way_summary.json").read_text())
    assert saved["gittins_protocol"]["cost_scaling_factor"] == 1e-4
    assert saved["display_names"]["ts"] == "Correlation-unaware TS"
    assert (tmp_path / "three_way_pairwise_differences.csv").is_file()
