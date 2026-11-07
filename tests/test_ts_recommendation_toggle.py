"""TS recommendation toggles must not change its posterior-sampling policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_lcb_toggle", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toy(tmp_path):
    matrix = tmp_path / "toy.npy"
    # Seed 4 first selects row 1. Its completed mean equals the unobserved
    # prior mean: mean-only ties choose row 0, while LCB chooses known row 1.
    np.save(matrix, np.array([[1., 1., 1., 1.], [0., 1., 0., 1.]], dtype=np.float32))
    costs = tmp_path / "prices.npy"
    np.save(costs, np.array([2., 3.]))
    return matrix, costs


def recorder(module, monkeypatch):
    original = module.gaussian_thompson_sampling
    batches = []

    def sample(*args, **kwargs):
        batch = original(*args, **kwargs)
        if batch is not None:
            batches.append(batch.clone())
        return batch

    monkeypatch.setattr(module, "gaussian_thompson_sampling", sample)
    return batches


@pytest.mark.parametrize("cost_aware", [False, True])
def test_standalone_toggle_changes_recommendation_and_metadata_only(tmp_path, toy, monkeypatch, cost_aware):
    module = load_script("simulate_simple_regret")
    matrix, costs = toy
    batches = recorder(module, monkeypatch)
    archives, actions = [], []
    for penalty in (0., 1.):
        output = tmp_path / f"trace_{penalty}.npz"
        args = ["runner", "--matrix", str(matrix), "--out", str(output),
                "--algorithms", "ts", "--ts-batch-size", "4", "--seed", "4",
                "--eval-budget-fraction", ".5",
                "--recommendation_std_penalty" if penalty else "--recommendation-std-penalty", str(penalty)]
        if cost_aware:
            args += ["--cost-vector", str(costs)]
        monkeypatch.setattr(sys, "argv", args)
        batches.clear()
        assert module.main() == 0
        actions.append([x.clone() for x in batches])
        with np.load(output, allow_pickle=False) as archive:
            archives.append({k: archive[k].copy() for k in archive.files})
    assert len(actions[0]) == len(actions[1]) == 1
    torch.testing.assert_close(actions[0][0], actions[1][0], rtol=0, atol=0)
    assert actions[0][0][0].tolist() == [1] * 4
    assert sorted(actions[0][0][1].tolist()) == list(range(4))
    for key in ("ts_x", "ts_x_original_cost", "ts_recommended_mean"):
        np.testing.assert_array_equal(archives[0][key], archives[1][key])
    assert archives[0]["ts_recommended_arm"].tolist() == [0]
    assert archives[1]["ts_recommended_arm"].tolist() == [1]
    for penalty, archive in zip((0., 1.), archives):
        expected = "finite_population_posterior_mean_minus_std" if penalty else "finite_population_posterior_mean"
        assert archive["ts_recommendation_std_penalty"].item() == penalty
        assert archive["recommendation_std_penalty"].item() == penalty
        assert archive["ts_recommendation_rule"].item() == expected
        assert archive["recommendation_rule"].item() == expected


@pytest.mark.parametrize("cost_aware", [False, True])
def test_standalone_default_applies_the_paper_lcb_rule(tmp_path, toy, monkeypatch, cost_aware):
    module = load_script("simulate_simple_regret")
    matrix, costs = toy
    output = tmp_path / "paper_default.npz"
    args = ["runner", "--matrix", str(matrix), "--out", str(output),
            "--algorithms", "ts", "--ts-batch-size", "4", "--seed", "4",
            "--eval-budget-fraction", ".5"]
    if cost_aware:
        args += ["--cost-vector", str(costs)]
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0
    with np.load(output, allow_pickle=False) as saved:
        assert saved["ts_recommendation_std_penalty"].item() == 1.
        assert saved["ts_recommendation_rule"].item() == "finite_population_posterior_mean_minus_std"
        assert saved["ts_recommended_arm"].tolist() == [1]


@pytest.mark.parametrize("mode", ["unit", "cost"])
def test_wandb_build_policy_uses_penalty_on_same_current_posterior(mode):
    module = load_script("run_simple_regret_wandb")
    observed = torch.tensor([[1.88, float("nan"), float("nan"), float("nan")],
                             [1., 1., 1., float("nan")]])
    selected = []
    for penalty in (0., 1.):
        _, recommend, _, _ = module.build_policy(
            variant=module.parse_experiment_variant(f"ts_{mode}_B4"),
            args=SimpleNamespace(ts_obs_noise_variance=.25, gittins_obs_noise_variance=None,
                                 recommendation_std_penalty=penalty),
            ground_truth=torch.zeros((2, 4)), actual_cost_per_arm=torch.tensor([2., 3.]),
            prior_mean=0., prior_variance=1., max_evaluations=8, n_cells=8,
            natural_stop_holder=[None], recommendation_aware_stop_holder=[None],
        )
        selected.append(recommend(observed, None)[0])
    assert selected == [0, 1]


@pytest.mark.parametrize("mode", ["unit", "cost"])
def test_wandb_entrypoint_records_rule_and_distinct_run_names_without_network(toy, monkeypatch, mode):
    module = load_script("run_simple_regret_wandb")
    matrix, costs = toy
    batches = recorder(module, monkeypatch)
    runs = []

    class LocalRun:
        def __init__(self, **kwargs):
            self.init = kwargs
            self.summary = {}
            self.history = []

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values):
            self.history.append(values)

        def finish(self, **kwargs):
            pass

    def local_init(**kwargs):
        run = LocalRun(**kwargs)
        runs.append(run)
        return run

    monkeypatch.setattr(module.wandb, "init", local_init)
    actions = []
    for penalty in (0., 1., None):
        args = ["runner", "--dataset-tag", "toy", "--matrix", str(matrix),
                "--experiment-variant", f"ts_{mode}_B4", "--run-seed", "4",
                "--eval-budget-fraction", ".5", "--wandb-mode", "offline"]
        if penalty is not None:
            args += ["--recommendation_std_penalty", str(penalty)]
        if mode == "cost":
            args += ["--cost-vector", str(costs)]
        monkeypatch.setattr(sys, "argv", args)
        batches.clear()
        assert module.main() == 0
        actions.append([x.clone() for x in batches])
    assert len(actions[0]) == len(actions[1]) == len(actions[2]) == 1
    torch.testing.assert_close(actions[0][0], actions[1][0], rtol=0, atol=0)
    torch.testing.assert_close(actions[1][0], actions[2][0], rtol=0, atol=0)
    assert [run.history[0]["recommended_arm"] for run in runs] == [0, 1, 1]
    assert runs[0].init["name"] != runs[1].init["name"]
    assert "_recstd" not in runs[0].init["name"]
    assert runs[1].init["name"].endswith("_recstd1")
    assert runs[2].init["name"] == runs[1].init["name"]
    for penalty, run in zip((0., 1., 1.), runs):
        expected = "finite_population_posterior_mean_minus_std" if penalty else "finite_population_posterior_mean"
        for saved in (run.init["config"], run.summary):
            assert saved["recommendation_std_penalty"] == penalty
            assert saved["ts_recommendation_std_penalty"] == penalty
            assert saved["recommendation_rule"] == expected
            assert saved["ts_recommendation_rule"] == expected


def test_wandb_missing_internal_penalty_defaults_to_paper_lcb():
    module = load_script("run_simple_regret_wandb")
    observed = torch.tensor([[1.88, float("nan"), float("nan"), float("nan")],
                             [1., 1., 1., float("nan")]])
    _, recommend, _, _ = module.build_policy(
        variant=module.parse_experiment_variant("ts_unit_B4"),
        args=SimpleNamespace(ts_obs_noise_variance=.25, gittins_obs_noise_variance=None),
        ground_truth=torch.zeros((2, 4)), actual_cost_per_arm=torch.ones(2),
        prior_mean=0., prior_variance=1., max_evaluations=8, n_cells=8,
        natural_stop_holder=[None], recommendation_aware_stop_holder=[None],
    )
    assert recommend(observed, None)[0] == 1


def test_formal_sweep_matches_the_paper_without_expanding_batch_sizes():
    import yaml
    setup = yaml.safe_load((ROOT / "scripts/config/GSM8KThompsonSamplingSweep.yml").read_text())
    params = setup["parameters"]
    assert params["experiment_variant"]["values"] == [
        "ts_unit_B8_default", "ts_cost_B8_default", "ts_unit_B8_dataset", "ts_cost_B8_dataset"]
    assert params["recommendation_std_penalty"] == {"value": 1.}
    assert params["eval_budget_fraction"] == {"value": .1}
    assert params["run_seed"]["values"] == list(range(20))
    assert len(params["matrix"]["values"]) == 5


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
@pytest.mark.parametrize("entrypoint", ["simulate_simple_regret", "run_simple_regret_wandb"])
def test_invalid_penalty_rejected_before_evaluation(toy, tmp_path, monkeypatch, value, entrypoint):
    module = load_script(entrypoint)
    matrix, _ = toy
    if entrypoint == "simulate_simple_regret":
        args = ["runner", "--matrix", str(matrix), "--out", str(tmp_path / "unused.npz"), "--algorithms", "ts"]
        function = module.main
    else:
        args = ["runner", "--dataset-tag", "toy", "--matrix", str(matrix),
                "--experiment-variant", "ts_unit_B4", "--wandb-mode", "disabled"]
        function = module.parse_args
    monkeypatch.setattr(sys, "argv", args + ["--recommendation-std-penalty", value])
    with pytest.raises(SystemExit) as error:
        function()
    assert error.value.code == 2
