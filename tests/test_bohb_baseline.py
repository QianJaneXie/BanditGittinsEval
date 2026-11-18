"""Native BOHB recommendation, accounting and information-boundary checks."""
import json
import importlib.util
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

pytest.importorskip("hpbandster")
pytest.importorskip("ConfigSpace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bohb_policy import BOHBData, default_parameters, fidelity_schedule, load_data, make_configspace, run_bohb

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def data(K=30, N=100, costs=None):
    return BOHBData(np.random.default_rng(12).integers(0, 2, (K, N)).astype(float),
                    np.arange(K)[:, None], np.ones(K) if costs is None else np.asarray(costs), ("arm",))


def test_defaults_match_author_and_realized_rungs():
    defaults = default_parameters()
    assert defaults["min_budget"] == 0.01
    assert defaults["max_budget"] == 1
    assert defaults["eta"] == 3
    assert defaults["min_points_in_model"] is None
    assert defaults["top_n_percent"] == 15
    assert defaults["random_fraction"] == 1 / 3
    assert defaults["num_samples"] == 64
    np.testing.assert_allclose(fidelity_schedule(), [1/81, 1/27, 1/9, 1/3, 1])
    np.testing.assert_array_equal(np.ceil(fidelity_schedule() * 1000), [13, 38, 112, 334, 1000])


def test_loader_aligns_rows_and_never_reads_answer_arrays(tmp_path):
    matrix = tmp_path / "matrix.npy"
    np.save(matrix, np.ones((3, 10)))
    archive = tmp_path / "inputs.npz"
    np.savez(archive, X=np.array([[20], [0], [10]]), arm_ids=np.array([2, 0, 1]),
             cost=np.array([30, 10, 20]), metadata_json=json.dumps({"n_examples": 10, "dataset": "toy"}),
             Y=np.array([object()], dtype=object), raw_scores_mean=np.array([object()], dtype=object))
    loaded = load_data(matrix, archive)
    np.testing.assert_array_equal(loaded.features.ravel(), [0, 10, 20])
    np.testing.assert_array_equal(loaded.per_cell_cost, [1, 2, 3])
    assert loaded.dataset == "toy"


@pytest.mark.parametrize("cost_budgeted", [False, True])
def test_prefix_reuse_strict_budget_and_observed_losses(cost_budgeted):
    D = data(costs=np.linspace(0.2, 3, 30))
    R = run_bohb(D, seed=7, cost_budgeted=cost_budgeted, eval_budget_fraction=0.15)
    counts = np.zeros(30, dtype=int)
    original = spent = 0.0
    for i, (arm, target, added) in enumerate(zip(R["pulled_arm"], R["n_requested_samples"], R["n_new_samples"])):
        assert added == max(0, target - counts[arm])
        counts[arm] = max(counts[arm], target)
        original += added * D.per_cell_cost[arm]
        spent += added * (D.per_cell_cost[arm] if cost_budgeted else 1)
        assert R["x"][i] == counts.sum()
        assert R["x_original_cost"][i] == pytest.approx(original)
        assert R["budget_spent"][i] == pytest.approx(spent)
        assert spent <= R["budget"] + 1e-8
        assert R["observed_y"][i] == pytest.approx(D.matrix[arm, R["permutations"][arm, :target]].mean())
    np.testing.assert_array_equal(counts, R["final_counts"])
    assert (R["n_new_samples"] == 0).any()  # Native duplicate jobs use the cache.
    assert R["stop_reason"] == "next_job_unaffordable"
    assert R["next_job_cost"] > R["unused_budget"]
    assert R["model_based_pick"].any()  # Exercises real KDE proposals, not a mock.


def test_unseen_scores_never_affect_native_allocations_or_recommendations():
    D = data()
    R = run_bohb(D, seed=3, max_jobs=40)
    changed = D.matrix.copy()
    for arm, n in enumerate(R["final_counts"]):
        hidden = R["permutations"][arm, n:]
        changed[arm, hidden] = 1 - changed[arm, hidden]
    D2 = BOHBData(changed, D.features, D.per_cell_cost, D.feature_names)
    R2 = run_bohb(D2, seed=3, max_jobs=40)
    for key in ["pulled_arm", "fidelity", "observed_y", "recommended_arm", "recommended_mean", "model_based_pick"]:
        np.testing.assert_array_equal(R[key], R2[key])


def test_anytime_and_final_recommendations_match_author_result_utilities():
    from hpbandster.core.base_iteration import Datum
    from hpbandster.core.result import Result
    R = run_bohb(data(K=50), seed=4, eval_budget_fraction=0.4)
    jobs = {}
    for i, (fidelity, mean) in enumerate(zip(R["fidelity"], R["observed_y"])):
        jobs[(i, 0, 0)] = Datum({}, {}, results={fidelity: {"loss": 1-mean, "info": {}}},
                                time_stamps={fidelity: {"submitted": i, "started": i, "finished": i+1}},
                                budget=fidelity)
    native = Result([jobs], {"max_budget": 1., "min_budget": 0.01, "time_ref": 0})
    trajectory = native.get_incumbent_trajectory()
    for i in range(len(R["x"])):
        event = np.searchsorted(trajectory["times_finished"], i+1, side="right") - 1
        selected_job = trajectory["config_ids"][event][0]
        assert R["recommended_arm"][i] == R["pulled_arm"][selected_job]
    final_id = native.get_incumbent_id()
    expected = -1 if final_id is None else R["pulled_arm"][final_id[0]]
    assert R["full_fidelity_recommended_arm"] == expected


def test_no_full_fidelity_job_has_no_final_incumbent():
    R = run_bohb(data(), max_jobs=10)
    assert R["full_fidelity_recommended_arm"] == -1
    assert (R["recommended_arm"] >= 0).all()
    assert R["incumbent_fidelity"].max() < 1


def test_sparse_grid_restricts_proposals_to_real_arms_and_drops_constant_columns():
    X = np.array([[0, 1, 7], [1, 0, 7], [1, 1, 7]])  # (0,0) is missing, including the default tuple.
    cs, hps, lookup, active = make_configspace(X, 9)
    assert active == [0, 1]
    for config in cs.sample_configuration(100):
        assert tuple(config[hp.name] for hp in hps) in lookup
    D = BOHBData(data(K=3).matrix, X, np.ones(3), ("model", "prompt", "constant"))
    R = run_bohb(D, max_jobs=20)
    assert set(R["pulled_arm"]).issubset(range(3))


def test_cost_scale_does_not_change_allocations_and_global_rng_is_restored():
    D = data(costs=np.linspace(0.5, 2, 30))
    np.random.seed(871)
    before = np.random.get_state()
    R = run_bohb(D, seed=5, cost_budgeted=True)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    D2 = BOHBData(D.matrix, D.features, D.per_cell_cost * 1e3, D.feature_names)
    R2 = run_bohb(D2, seed=5, cost_budgeted=True)
    np.testing.assert_array_equal(R["pulled_arm"], R2["pulled_arm"])
    np.testing.assert_array_equal(R["fidelity"], R2["fidelity"])
    np.testing.assert_allclose(R2["x_original_cost"], R["x_original_cost"] * 1e3)


def test_unaffordable_first_job_does_not_observe_anything():
    R = run_bohb(data(K=3), eval_budget_fraction=0.001)
    assert len(R["x"]) == 0
    assert R["budget_spent_total"] == 0
    assert R["full_fidelity_recommended_arm"] == -1
    assert not R["final_counts"].any()


def test_cli_writes_nonpickle_trace_and_refuses_to_overwrite(tmp_path):
    matrix, inputs, out = (tmp_path / name for name in ("matrix.npy", "inputs.npz", "out.npz"))
    np.save(matrix, data(K=6).matrix)
    np.savez(inputs, X=np.arange(6)[:, None], arm_ids=np.arange(6), cost=np.ones(6)*100,
             metadata_json=json.dumps({"dataset": "toy", "n_examples": 100}))
    script = Path(__file__).resolve().parents[1] / "scripts/run_bohb_baseline.py"
    command = [sys.executable, str(script), "--matrix", str(matrix), "--bo-inputs", str(inputs),
               "--out", str(out), "--max-jobs", "10"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with np.load(out, allow_pickle=False) as archive:
        for key in archive.files:
            archive[key]  # Every saved field must be readable without pickle.
        assert len(archive["bohb_regret"]) == 10
    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["num_jobs"] == 10
    assert summary["allocation_cost_aware"] is False
    assert out.with_suffix(".log").is_file()
    repeat = subprocess.run(command, capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "Result already exists" in repeat.stderr


def test_suite_pairs_matrix_versions_and_covers_all_repetitions(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/run_bohb_benchmarks.py"
    spec = importlib.util.spec_from_file_location("bohb_suite", script)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    for dataset in ["gsm8k", "piqa"]:
        for seed in range(1, 6):
            stem = f"{dataset}_1_samples_various_models_seed{seed}"
            for rel in [f"data/BanditEval_matrices/{stem}.npy", f"data/bo_inputs/{dataset}/{stem}_bo_inputs.npz"]:
                path = tmp_path / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
    for subject in range(57):
        for rel in [f"data/MMLU_matrices/subject_{subject}.npy", f"data/bo_inputs/mmlu/subject_{subject}_bo_inputs.npz"]:
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "--dry-run"])
    assert suite.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["num_jobs"] == 2680
    for command in plan["commands"]:
        matrix = Path(command[command.index("--matrix")+1])
        inputs = Path(command[command.index("--bo-inputs")+1])
        assert inputs.stem == matrix.stem + "_bo_inputs"
