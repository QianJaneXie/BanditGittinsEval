"""Offline checks for baseline metadata downloads and aggregate plot labels."""

import csv
import gzip
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    module_name = f"baseline_metadata_{name}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def downloader():
    return load_script("download_wandb_simple_regret")


@pytest.fixture(scope="module")
def plotter():
    return load_script("plot_wandb_simple_regret_curves")


@pytest.mark.parametrize("variant, mode, batch, prior", [
    ("ts_B16", "unit", 16, "default"),
    ("ts_cost_B8", "cost", 8, "default"),
    ("ts_unit_B32_dataset", "unit", 32, "dataset"),
    ("ts_aware_B16_dataset", "cost", 16, "dataset"),
    ("ts", "unit", 20, "default"),
    ("ts_cost", "cost", 20, "default"),
])
def test_ts_variant_fallback(downloader, variant, mode, batch, prior):
    parsed = downloader.parse_variant_fallback(variant)
    assert parsed["policy_family"] == "ts"
    assert parsed["policy_variant"] == f"ts_{mode}"
    assert parsed["cost_mode"] == mode
    assert parsed["batch_size"] == batch
    assert parsed["prior_type"] == prior
    assert parsed["ts_target"] == "finite_population_mean"


def test_ts_metadata_survives_csv_writers_without_wandb(downloader, tmp_path):
    run = SimpleNamespace(
        id="ts-offline", name="offline fake run", state="finished", url="",
        config={
            "experiment_variant": "ts_unit_B16_dataset", "dataset_tag": "gsm8k",
            "matrix": "data/gsm8k_seed1.npy", "run_seed": 0,
            "prior_mean_resolved": .2, "prior_variance_resolved": .01,
            "ts_prior_mean": .2, "ts_prior_variance": .01,
            "ts_obs_noise_variance": .03, "ts_obs_noise_variance_resolved": .03,
            "ts_tau_sq_cell": .48, "ts_cost_allocation": "standard_ts_cost_budget_stopping",
            "recommendation_rule": "finite_population_posterior_mean",
        },
        summary={"final_simple_regret": .05, "ts_obs_noise_variance_resolved": .04},
    )
    row = downloader.base_row(run)
    assert row["ts_target"] == "finite_population_mean"
    assert row["ts_obs_noise_variance_resolved"] == .04
    assert row["prior_mean_resolved"] == .2
    writers = downloader.DownloadWriters(tmp_path, None, no_history=False, split_mmlu_by_task=False)
    try:
        writers.write_run(task=None,
                          summary_row={**row, "final_simple_regret": .05},
                          history_rows=[{**row, "cum_eval": 16, "simple_regret": .05}])
    finally:
        writers.close()
    with gzip.open(tmp_path / "runs_history.csv.gz", "rt") as stream:
        history = list(csv.DictReader(stream))[0]
    summary = downloader.read_csv_rows(tmp_path / "runs_summary.csv")[0]
    for saved in (history, summary):
        assert saved["policy_family"] == "ts"
        assert saved["ts_target"] == "finite_population_mean"
        assert saved["ts_obs_noise_variance_resolved"] == "0.04"
        assert saved["ts_tau_sq_cell"] == "0.48"
        assert saved["prior_variance_resolved"] == "0.01"


def test_baseline_labels_distinguish_cost_batch_and_prior(plotter):
    label = plotter.derive_method_label_row(pd.Series({
        "policy_family": "ts", "experiment_variant": "ts_unit_B16_dataset",
        "cost_mode": "unit", "batch_size": 16, "prior_type": "dataset",
    }))
    assert label == "Gaussian TS (unit; B=16; prior=dataset)"
    assert plotter.derive_method_label_row(pd.Series({
        "experiment_variant": "ts_cost_B8_default",
    })) == "Gaussian TS (cost; B=8; prior=default)"
    assert plotter.derive_method_label_row(pd.Series({
        "policy_family": "mfkg", "cost_mode": "cost",
    })) == "BO MF-KG (cost)"
    assert plotter.derive_method_label_row(pd.Series({
        "policy_family": "ucb", "experiment_variant": "ucb_B16",
    })) == "Bandit UCB-E"


def test_post_init_keeps_all_mfkg_warmup_phases(plotter):
    rows = pd.DataFrame([
        {"run_id": "mf1", "policy_family": "mfkg", "selection_phase": "random_init"},
        {"run_id": "mf1", "policy_family": "mfkg", "selection_phase": "random_init_anchor"},
        {"run_id": "mf1", "policy_family": "mfkg", "selection_phase": "mfkg"},
        {"run_id": "old-mf", "policy_family": "bo", "acquisition": "mfkg", "selection_phase": "random_init"},
        {"run_id": "single", "policy_family": "bo", "acquisition": "logei", "selection_phase": "random_init"},
        {"run_id": "single", "policy_family": "bo", "acquisition": "logei", "selection_phase": "bo"},
        {"run_id": "ts", "policy_family": "ts", "selection_phase": "random_init"},
    ])
    filtered = plotter.apply_bo_phase_filter(rows, "post_init")
    assert filtered.index.tolist() == [0, 1, 2, 3, 5, 6]
    pd.testing.assert_frame_equal(plotter.apply_bo_phase_filter(rows, "full"), rows)


def test_mfkg_labels_distinguish_fixed_batches_and_adaptive_fidelity(plotter):
    base = {
        "policy_family": "mfkg", "experiment_variant": "mfkg_unit",
        "method_label": "BO MF-KG", "batch_size": 16,
    }
    assert plotter.derive_method_label_row(pd.Series({
        **base, "allocation_mode": "fixed_batch",
    })) == "BO MF-KG [fixed batch B=16]"
    assert plotter.derive_method_label_row(pd.Series({
        **base, "allocation_mode": "fixed_batch", "batch_size": 4.0,
    })) == "BO MF-KG [fixed batch B=4]"
    assert plotter.derive_method_label_row(pd.Series({
        **base, "allocation_mode": "adaptive_fidelity",
    })) == "BO MF-KG [adaptive fidelity]"
    # A previously derived label is refreshed from the actual saved protocol.
    assert plotter.derive_method_label_row(pd.Series({
        **base, "allocation_mode": "fixed_batch", "batch_size": 4,
        "method_label": "BO MF-KG [fixed batch B=16]",
    })) == "BO MF-KG [fixed batch B=4]"
    assert plotter.derive_method_label_row(pd.Series(base)) == "BO MF-KG"


def test_mfkg_custom_groups_cannot_pool_different_feedback_protocols(plotter):
    rows = pd.DataFrame([
        {"policy_family": "mfkg", "group": "pooled", "allocation_mode": "fixed_batch", "batch_size": 4},
        {"policy_family": "mfkg", "group": "pooled", "allocation_mode": "fixed_batch", "batch_size": 16},
    ])
    with pytest.raises(ValueError, match="MF-KG allocation modes or item batch sizes"):
        plotter.validate_recommendation_groups(rows, "group")
    # Adaptive fidelity does not use the nominal fixed batch_size setting.
    rows["allocation_mode"] = "adaptive_fidelity"
    plotter.validate_recommendation_groups(rows, "group")
