"""Accounting and information-boundary checks for nested-prefix MF-KG runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mfkg_runner_under_test", ROOT / "scripts" / "run_mfkg_baseline.py"
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class ObservedOnlyModel:
    """Cheap stand-in which cannot access the original evaluation matrix."""

    def __init__(self, X, Y):
        self.X, self.Y = X.clone(), Y.clone()

    def posterior(self, X):
        means = []
        for point in X:
            same_arm = (self.X[:, :-1] == point[:-1]).all(dim=-1)
            means.append(self.Y[same_arm][-1, 0] if same_arm.any() else self.Y.mean())
        return SimpleNamespace(mean=torch.stack(means).unsqueeze(-1))


@pytest.fixture
def cheap_core(monkeypatch):
    fitted = []

    def fit(X, Y, **kwargs):
        fitted.append((X.clone(), Y.clone()))
        return ObservedOnlyModel(X, Y)

    def scores(model, candidates, targets, costs, **kwargs):
        # Information-free tie breaking favors cheap promotions, while also
        # depending on the fitted model's actually revealed observations.
        return -costs + 1e-3 * model.posterior(candidates).mean.squeeze(-1)

    monkeypatch.setattr(runner, "fit_mfkg_model", fit)
    monkeypatch.setattr(runner, "score_mfkg_candidates", scores)
    return fitted


def make_data(n_arms=4, n_examples=100, costs=None):
    values = np.random.default_rng(37).uniform(size=(n_arms, n_examples))
    return runner.MFData(
        matrix=values,
        features=torch.arange(n_arms, dtype=torch.float64).unsqueeze(-1),
        per_cell_cost=np.ones(n_arms) if costs is None else np.asarray(costs, dtype=float),
        feature_source="bo_categorical_features",
    )


@pytest.mark.parametrize("cost_aware", [False, True])
def test_every_new_cell_is_charged_once_and_hard_budget_respected(cheap_core, cost_aware):
    data = make_data(costs=[0.5, 1.5, 2.5, 5.5])
    result = runner.run_mfkg(
        data, allocation_mode="adaptive_fidelity", sample_counts=np.array([2, 5, 10, 25, 100]), seed=8,
        eval_budget_fraction=0.137, cost_aware=cost_aware,
    )
    counts = np.zeros(4, dtype=np.int64)
    cumulative_cost = 0.0
    for i, (arm, count, added) in enumerate(zip(
        result["pulled_arm"], result["n_observed_arm"], result["n_new_samples"]
    )):
        assert added == count - counts[arm] > 0
        counts[arm] = count
        cumulative_cost += added * data.per_cell_cost[arm]
        assert result["x"][i] == counts.sum()
        assert result["x_original_cost"][i] == pytest.approx(cumulative_cost)
        prefix = result["permutations"][arm, :count]
        assert result["observed_y"][i] == pytest.approx(data.matrix[arm, prefix].mean())
        np.testing.assert_allclose(result["train_X"][i], [arm, count / 100])
        assert result["train_Y"][i, 0] == pytest.approx(result["observed_y"][i])
    np.testing.assert_array_equal(result["final_counts"], counts)
    spent = np.dot(counts, data.per_cell_cost) if cost_aware else counts.sum()
    assert result["budget_spent"] == pytest.approx(spent)
    assert spent <= result["budget"] + 1e-9
    assert result["warmup_spent"] <= result["warmup_budget"] + 1e-9
    assert len(cheap_core) == len(result["x"])
    # A terminated run cannot afford any remaining legal promotion.
    for arm in range(4):
        remaining = result["sample_counts"][result["sample_counts"] > counts[arm]]
        if len(remaining):
            unit_price = data.per_cell_cost[arm] if cost_aware else 1.0
            assert (remaining[0] - counts[arm]) * unit_price > result["budget"] - spent - 1e-9


def test_warmup_has_low_fidelity_coverage_and_seeded_affordable_anchor(cheap_core):
    data = make_data()
    result = runner.run_mfkg(
        data, allocation_mode="adaptive_fidelity", sample_counts=np.array([2, 10, 20, 100]), seed=4,
        eval_budget_fraction=0.5, warmup_budget_fraction=0.1, max_steps=0,
    )
    assert result["n_init_actual"] == 4
    assert len(set(result["pulled_arm"][:4])) == 4
    assert result["n_observed_arm"][:4] == [2] * 4
    assert result["selection_phase"] == ["random_init"] * 4 + ["random_init_anchor"]
    assert result["n_observed_arm"][-1] == 10  # full fidelity cannot fit the cap
    assert result["warmup_spent"] == 16
    assert result["warmup_budget"] == 20
    assert result["stop_reason"] == "max_steps"


def test_hidden_scores_do_not_change_actions_or_gp_training(cheap_core):
    data = make_data()
    kwargs = dict(
        allocation_mode="adaptive_fidelity", sample_counts=np.array([2, 5, 10, 25, 100]), seed=11,
        eval_budget_fraction=0.15, max_steps=5,
    )
    original = runner.run_mfkg(data, **kwargs)
    changed_matrix = data.matrix.copy()
    for arm, count in enumerate(original["final_counts"]):
        hidden = original["permutations"][arm, count:]
        changed_matrix[arm, hidden] = 1000 + arm
    changed_data = runner.MFData(
        changed_matrix, data.features, data.per_cell_cost, data.feature_source
    )
    changed = runner.run_mfkg(changed_data, **kwargs)
    for key in ["pulled_arm", "n_observed_arm", "recommended_arm", "train_X", "train_Y", "permutations"]:
        np.testing.assert_array_equal(original[key], changed[key])
    # The complete matrix may change evaluation metrics, but not decisions.
    assert original["regret"] != changed["regret"]


def test_completed_arms_use_exact_means_for_recommendation(cheap_core):
    data = runner.MFData(
        np.array([[1, 1, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0]], dtype=float),
        torch.tensor([[0.0], [1.0]], dtype=torch.float64), np.ones(2), "arm_identity",
    )
    result = runner.run_mfkg(
        data, allocation_mode="adaptive_fidelity", sample_counts=np.array([1, 3, 6]), eval_budget_fraction=1,
        warmup_budget_fraction=0.25, n_init=2,
    )
    np.testing.assert_array_equal(result["final_counts"], [6, 6])
    assert result["recommended_arm"][-1] == 0
    assert result["recommended_mean"][-1] == pytest.approx(5 / 6)
    assert result["regret"][-1] == 0


def test_loader_aligns_row_ids_and_full_costs_without_reading_complete_scores(tmp_path):
    matrix = tmp_path / "matrix.npy"
    np.save(matrix, np.ones((3, 10)))
    archive = tmp_path / "bo.npz"
    np.savez(
        archive, X=np.array([[20], [0], [10]]), arm_ids=np.array([2, 0, 1]),
        cost=np.array([30.0, 10.0, 20.0]),
        metadata_json=json.dumps({"n_examples": 10}), dataset="toy",
        # Access would raise because load_data opens the archive with pickle disabled.
        Y=np.array([object()], dtype=object), raw_scores_mean=np.array([object()], dtype=object),
    )
    data = runner.load_data(matrix, archive)
    np.testing.assert_array_equal(data.features.numpy().ravel(), [0, 10, 20])
    np.testing.assert_array_equal(data.per_cell_cost, [1, 2, 3])
    assert data.dataset == "toy"


def test_tiny_warmup_budget_fails_explicitly(cheap_core):
    with pytest.raises(ValueError, match="cannot afford one"):
        runner.run_mfkg(
            make_data(), allocation_mode="adaptive_fidelity", sample_counts=np.array([16, 100]),
            eval_budget_fraction=0.1, warmup_budget_fraction=0.1,
        )


def test_coverage_choice_uses_categories_and_seeded_ties():
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # With (0,0) observed, (1,1) adds two levels and must be chosen.
    assert runner.select_coverage_arm(features, np.array([1, 2, 3]), [0], np.random.default_rng(9)) == 3


def test_fidelity_counts_always_include_exact_completion():
    np.testing.assert_array_equal(runner.fidelity_counts(10, [0.01, 0.11, 0.11]), [1, 2, 10])
    np.testing.assert_array_equal(runner.fidelity_counts(10, [], [2, 5, 5]), [2, 5, 10])


@pytest.mark.parametrize("cost_aware", [False, True])
def test_fixed_batch_charges_exactly_new_cells_and_clips_budget_tail(cheap_core, cost_aware):
    data = make_data(costs=[0.5, 1.5, 2.5, 5.5])
    batch_size = 7
    result = runner.run_mfkg(
        data, allocation_mode="fixed_batch", batch_size=batch_size, seed=8, eval_budget_fraction=0.137,
        cost_aware=cost_aware, n_init=1, warmup_anchors=0,
        warmup_budget_fraction=0.25,
    )
    counts = np.zeros(4, dtype=np.int64)
    spent = 0.0
    original_cost = 0.0
    prices = data.per_cell_cost if cost_aware else np.ones(4)
    for i, (arm, count, added, phase) in enumerate(zip(
        result["pulled_arm"], result["n_observed_arm"], result["n_new_samples"],
        result["selection_phase"],
    )):
        assert 1 <= added <= batch_size
        assert added == count - counts[arm]
        if phase == "mfkg":
            affordable = int(np.floor((result["budget"] - spent + 1e-9) / prices[arm]))
            assert added == min(batch_size, 100 - counts[arm], affordable)
        counts[arm] = count
        spent += added * prices[arm]
        original_cost += added * data.per_cell_cost[arm]
        assert result["x"][i] == counts.sum()
        assert result["x_original_cost"][i] == pytest.approx(original_cost)
        prefix = result["permutations"][arm, :count]
        assert len(np.unique(prefix)) == count
        assert result["observed_y"][i] == pytest.approx(data.matrix[arm, prefix].mean())
        np.testing.assert_allclose(result["train_X"][i], [arm, count / 100])
        assert result["train_Y"][i, 0] == pytest.approx(result["observed_y"][i])
    assert spent == pytest.approx(result["budget_spent"])
    assert spent <= result["budget"] + 1e-9
    np.testing.assert_array_equal(result["final_counts"], counts)
    # Every remaining cell is too expensive after the hard-budget tail.
    assert np.all(prices[counts < 100] > result["budget"] - spent - 1e-9)
    if not cost_aware:
        assert result["budget_spent"] == result["budget"] == 54
        assert result["n_new_samples"][-1] == 5


def test_fixed_batch_uses_arm_remainder_and_completes_every_arm(cheap_core):
    data = make_data(n_arms=3, n_examples=11)
    result = runner.run_mfkg(
        data, allocation_mode="fixed_batch", batch_size=4, seed=3, eval_budget_fraction=1,
        n_init=1, warmup_anchors=0, warmup_budget_fraction=0.25,
    )
    counts = np.zeros(3, dtype=np.int64)
    for arm, count, added in zip(result["pulled_arm"], result["n_observed_arm"], result["n_new_samples"]):
        assert added == min(4, 11 - counts[arm])
        counts[arm] = count
    np.testing.assert_array_equal(counts, [11, 11, 11])
    assert result["x"][-1] == 33
    assert result["n_new_samples"].count(3) == 3
    assert result["regret"][-1] == 0


def test_fixed_batch_anchor_promotions_are_split_into_feedback_rounds(cheap_core):
    data = make_data()
    result = runner.run_mfkg(
        data, allocation_mode="fixed_batch", batch_size=7, seed=4, eval_budget_fraction=0.5,
        n_init=4, warmup_anchors=1, warmup_budget_fraction=0.6, max_steps=1,
    )
    assert result["n_new_samples"][:4] == [7] * 4
    anchor = [i for i, phase in enumerate(result["selection_phase"]) if phase == "random_init_anchor"]
    assert len(anchor) > 1
    assert all(1 <= result["n_new_samples"][i] <= 7 for i in anchor)
    assert len({result["pulled_arm"][i] for i in anchor}) == 1
    assert result["warmup_spent"] == result["warmup_budget"] == 120
    assert result["n_new_samples"][anchor[-1]] == 1  # only the warmup cap ends this round
    assert result["selection_phase"][-1] == "mfkg"
    assert result["pulled_arm"][-1] == result["pulled_arm"][anchor[-1]]
    assert result["n_new_samples"][-1] == 1  # the next query completes that 99-item arm


@pytest.mark.parametrize("cost_aware", [False, True])
def test_fixed_batch_scores_one_legal_next_fidelity_per_arm(cheap_core, monkeypatch, cost_aware):
    data = make_data(n_arms=3, n_examples=23, costs=[0.7, 1.1, 2.3])
    batch_size = 5
    prices = data.per_cell_cost if cost_aware else np.ones(3)
    budget = 0.61 * 23 * prices.sum() if cost_aware else np.floor(0.61 * 23 * prices.sum())
    calls = []

    def score(model, candidates, targets, costs, **kwargs):
        counts = np.zeros(3, dtype=np.int64)
        for point in model.X:
            counts[int(point[0])] = int(round(float(point[-1]) * 23))
        spent = np.dot(counts, prices)
        arms = candidates[:, 0].to(torch.int64).numpy()
        assert len(np.unique(arms)) == len(arms)
        for i, (arm, point) in enumerate(zip(arms, candidates)):
            target_count = int(round(float(point[-1]) * 23))
            affordable = int(np.floor((budget - spent + 1e-9) / prices[arm]))
            added = min(batch_size, 23 - counts[arm], affordable)
            assert added > 0
            assert target_count == counts[arm] + added
            assert float(costs[i]) == pytest.approx(added * prices[arm])
        calls.append(candidates.clone())
        return -costs

    monkeypatch.setattr(runner, "score_mfkg_candidates", score)
    result = runner.run_mfkg(
        data, allocation_mode="fixed_batch", batch_size=batch_size, seed=12, eval_budget_fraction=0.61,
        cost_aware=cost_aware, n_init=1, warmup_anchors=0,
        warmup_budget_fraction=0.25,
    )
    assert calls
    assert len(calls) == result["selection_phase"].count("mfkg")


def test_fixed_batch_hidden_scores_cannot_change_decisions(cheap_core):
    data = make_data()
    kwargs = dict(
        allocation_mode="fixed_batch", batch_size=7, seed=11, eval_budget_fraction=0.15, max_steps=5,
        n_init=1, warmup_anchors=0, warmup_budget_fraction=0.25,
    )
    first = runner.run_mfkg(data, **kwargs)
    changed_matrix = data.matrix.copy()
    for arm, count in enumerate(first["final_counts"]):
        changed_matrix[arm, first["permutations"][arm, count:]] = 1000 + arm
    second = runner.run_mfkg(
        runner.MFData(changed_matrix, data.features, data.per_cell_cost, data.feature_source),
        **kwargs,
    )
    for key in ("pulled_arm", "n_observed_arm", "n_new_samples", "recommended_arm", "train_X", "train_Y"):
        np.testing.assert_array_equal(first[key], second[key])


def test_fixed_batch_rejects_a_fidelity_jump_grid(cheap_core):
    with pytest.raises(ValueError, match="(?i)(fixed|sample_counts|fidelity)"):
        runner.run_mfkg(make_data(), allocation_mode="fixed_batch", sample_counts=np.array([1, 4, 100]))


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_fixed_batch_rejects_invalid_batch_sizes(cheap_core, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        runner.run_mfkg(make_data(), allocation_mode="fixed_batch", batch_size=batch_size)


def test_fixed_batch_cli_saves_round_size_and_allocation_metadata(cheap_core, monkeypatch, tmp_path):
    matrix = tmp_path / "matrix.npy"
    output = tmp_path / "fixed.npz"
    np.save(matrix, make_data().matrix)
    monkeypatch.setattr(sys, "argv", [
        "run_mfkg_baseline.py", "--matrix", str(matrix), "--out", str(output),
        "--allocation-mode", "fixed_batch",
        "--batch-size", "7", "--n-init", "1", "--warmup-anchors", "0",
        "--warmup-budget-fraction", ".25", "--eval-budget-fraction", ".137",
        "--max-steps", "2",
    ])
    assert runner.main() == 0
    with np.load(output, allow_pickle=False) as archive:
        assert archive["allocation_mode"].item() == "fixed_batch"
        assert archive["batch_size"].item() == 7
        assert archive["q"].item() == 1
        np.testing.assert_array_equal(archive["n_new_samples"], [7, 7, 7])
        np.testing.assert_array_equal(archive["mfkg_x"], [7, 14, 21])
        assert archive["selection_phase"].tolist() == ["random_init", "mfkg", "mfkg"]
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert (metadata["allocation_mode"], metadata["batch_size"], metadata["q"]) == ("fixed_batch", 7, 1)


@pytest.mark.parametrize("grid_flag, values", [
    ("--fidelity-samples", ["1", "4"]), ("--fidelities", [".1", ".5"]),
])
def test_explicit_fixed_cli_rejects_fidelity_jump_flags(monkeypatch, capsys, grid_flag, values):
    monkeypatch.setattr(sys, "argv", [
        "run_mfkg_baseline.py", "--matrix", "unused.npy",
        "--allocation-mode", "fixed_batch", grid_flag, *values,
    ])
    with pytest.raises(SystemExit) as error:
        runner.parse_args()
    assert error.value.code == 2
    assert "--allocation-mode adaptive_fidelity" in capsys.readouterr().err


def test_default_api_offers_fidelity_jumps_and_charges_the_selected_increment(cheap_core, monkeypatch):
    data = make_data()
    score_calls = []

    def prefer_largest_fidelity(model, candidates, targets, costs, **kwargs):
        arms = candidates[:, 0].to(torch.int64).numpy()
        # The default action space must offer multiple fidelity choices for
        # the same arm, rather than only its next fixed-size batch.
        assert any(np.sum(arms == arm) > 1 for arm in np.unique(arms))
        observed = np.zeros(4, dtype=np.int64)
        for point in model.X:
            observed[int(point[0])] = int(round(float(point[-1]) * 100))
        for point, price in zip(candidates, costs):
            arm = int(point[0])
            target_count = int(round(float(point[-1]) * 100))
            assert target_count > observed[arm]
            assert float(price) == target_count - observed[arm]
        score_calls.append(candidates.clone())
        # Break full-fidelity ties in favor of the already observed arm, so
        # the assertion below also checks reuse of its two warmup items.
        return candidates[:, -1] + 0.001 * (candidates[:, 0] == model.X[0, 0])

    monkeypatch.setattr(runner, "score_mfkg_candidates", prefer_largest_fidelity)
    result = runner.run_mfkg(
        data, n_init=1, warmup_anchors=0, eval_budget_fraction=0.5, max_steps=1,
    )
    assert result["allocation_mode"] == "adaptive_fidelity"
    assert result["q"] == 1
    np.testing.assert_array_equal(result["sample_counts"], [2, 4, 7, 13, 25, 50, 100])
    assert len(score_calls) == 1
    assert result["selection_phase"] == ["random_init", "mfkg"]
    assert result["n_new_samples"] == [2, 98]
    assert result["fidelity"][-1] == 1.0
    assert result["budget_spent"] == 100 <= result["budget"]


def test_default_cli_saves_q1_adaptive_fidelity_without_fixed_batch_metadata(cheap_core, monkeypatch, tmp_path):
    matrix = tmp_path / "matrix.npy"
    output = tmp_path / "adaptive.npz"
    np.save(matrix, make_data().matrix)
    monkeypatch.setattr(
        runner, "score_mfkg_candidates",
        lambda model, candidates, targets, costs, **kwargs:
            candidates[:, -1] + 0.001 * (candidates[:, 0] == model.X[0, 0]),
    )
    monkeypatch.setattr(sys, "argv", [
        "run_mfkg_baseline.py", "--matrix", str(matrix), "--out", str(output),
        "--n-init", "1", "--warmup-anchors", "0", "--eval-budget-fraction", ".5",
        "--max-steps", "1",
    ])
    assert runner.main() == 0
    with np.load(output, allow_pickle=False) as archive:
        assert archive["allocation_mode"].item() == "adaptive_fidelity"
        assert archive["batch_size"].item() == -1  # no fixed item batch applies
        assert archive["q"].item() == 1
        np.testing.assert_array_equal(archive["n_new_samples"], [2, 98])
        np.testing.assert_array_equal(archive["mfkg_x"], [2, 100])
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["allocation_mode"] == "adaptive_fidelity"
    assert metadata["batch_size"] is None
    assert metadata["q"] == 1


@pytest.mark.parametrize("grid_flag, values, field, expected", [
    ("--fidelity-samples", ["1", "4"], "fidelity_samples", [1, 4]),
    ("--fidelities", [".1", ".5"], "fidelities", [.1, .5]),
])
def test_default_cli_accepts_custom_fidelity_grid(monkeypatch, grid_flag, values, field, expected):
    monkeypatch.setattr(sys, "argv", [
        "run_mfkg_baseline.py", "--matrix", "unused.npy", grid_flag, *values,
    ])
    args = runner.parse_args()
    assert args.allocation_mode == "adaptive_fidelity"
    assert getattr(args, field) == expected
