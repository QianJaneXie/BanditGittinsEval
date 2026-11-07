"""Offline checks for downloading both legacy BO and MF-KG run formats."""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_wandb_bo_baseline as downloader  # noqa: E402


def fake_run(config, summary=None):
    return SimpleNamespace(
        id="test_run", name="Test run", state="finished", url="https://wandb.invalid/run",
        config=config, summary=summary or {},
    )


@pytest.mark.parametrize("mode", ("unit", "cost"))
def test_mfkg_config_metadata_survives_download(mode):
    row = downloader.base_row(fake_run({
        "experiment_variant": f"mfkg_{mode}", "acquisition": "mfkg",
        "policy_family": "mfkg", "cost_mode": mode, "run_seed": 0,
        "dataset_tag": "gsm8k", "matrix": "data/gsm8k_seed1.npy",
        "bo_inputs": "data/bo_inputs/gsm8k_seed1_bo_inputs.npz",
        "n_arms": 122, "n_examples": 1319, "target_fidelity": 1.0,
        "fidelity_samples": [16, 32, 1319], "num_fantasies": 64,
        "feature_source": "bo_categorical_features", "warmup_budget_fraction": 0.1,
        "observation_model": "deterministic_nested_prefix_gp_with_numerical_nugget",
    }))
    assert row["policy_family"] == "mfkg"
    assert row["method_label"] == "BO MF-KG" + (" (cost)" if mode == "cost" else "")
    assert row["acquisition"] == "mfkg"
    assert row["run_seed"] == 0
    assert row["benchmark_key"] == "gsm8k_seed1"
    assert row["fidelity_samples"] == [16, 32, 1319]
    assert row["n_configs"] == 122
    assert row["warmup_budget_fraction"] == 0.1
    assert row["matrix"] == "data/gsm8k_seed1.npy"


def test_mfkg_variant_and_seed_fallback_with_matrix_only():
    row = downloader.base_row(fake_run({
        "experiment_variant": "mfkg_cost", "seed": 3,
        "matrix": "data/MMLU_matrices/abstract_algebra.npy",
        "cost_vector": "cost.npy", "cost_aware": True,
    }))
    assert row["policy_family"] == "mfkg"
    assert row["run_seed"] == 3
    assert row["benchmark_key"] == "mmlu_abstract_algebra"
    assert row["cost_mode"] == "cost"
    assert row["cost_aware"] is True
    assert downloader.config_key(row) == (
        "data/MMLU_matrices/abstract_algebra.npy", "3", "mfkg_cost"
    )


@pytest.mark.parametrize("mode,batch,suffix", [
    ("fixed_batch", 16, " [fixed batch B=16]"),
    ("fixed_batch", 4, " [fixed batch B=4]"),
    ("adaptive_fidelity", 16, " [adaptive fidelity]"),
])
def test_mfkg_allocation_protocol_survives_download_and_keys(mode, batch, suffix):
    row = downloader.base_row(fake_run({
        "experiment_variant": "mfkg_cost", "policy_family": "mfkg",
        "run_seed": 1, "matrix": "data/gsm8k_seed1.npy",
        "allocation_mode": mode, "batch_size": batch,
        "method_label": "BO MF-KG (cost)",
    }))
    assert row["allocation_mode"] == mode
    assert row["batch_size"] == batch
    assert row["method_label"] == "BO MF-KG (cost)" + suffix
    assert {"allocation_mode", "batch_size"}.issubset(downloader.HISTORY_COLUMNS)
    assert {"allocation_mode", "batch_size"}.issubset(downloader.SUMMARY_COLUMNS)
    assert downloader.config_key(row) == (
        "data/gsm8k_seed1.npy", "1", "mfkg_cost", mode,
        str(batch) if mode == "fixed_batch" else "",
    )


@pytest.mark.parametrize("variant,label,acquisition,cost_mode", [
    ("pbgi_unit", "BO PBGI", "pbgi", "unit"),
    ("pbgi_cost", "BO PBGI (cost)", "pbgi", "cost"),
    ("logei_unit", "BO LogEI", "logei", "unit"),
    ("logeipc_cost", "BO LogEIPC", "logeipc", "cost"),
])
def test_legacy_bo_labels_and_config_keys_unchanged(variant, label, acquisition, cost_mode):
    cfg = {"experiment_variant": variant, "run_seed": 4, "bo_inputs": "gsm8k_seed2_bo_inputs.npz"}
    row = downloader.base_row(fake_run(cfg))
    assert row["policy_family"] == "bo"
    assert row["method_label"] == label
    assert row["acquisition"] == acquisition
    assert row["cost_mode"] == cost_mode
    assert downloader.config_key(row) == ("gsm8k_seed2_bo_inputs.npz", "4", variant)


def test_history_scan_keeps_sparse_old_and_new_metric_rows_without_key_filter():
    def scan_history(**kwargs):
        assert kwargs == {"page_size": 20}
        yield {"_step": 0, "wall_clock": 1.2}
        yield {"cum_eval": 16, "simple_regret": 0.3, "selection_phase": "random_init"}
        yield {"cum_eval": 32, "simple_regret": 0.2, "selection_phase": "mfkg",
               "fidelity": 0.2, "n_new_samples": 16}
        yield {"x": 48, "regret": 0.1, "x_original_cost": 72, "bo_phase": "mfkg"}

    run = SimpleNamespace(id="test", scan_history=scan_history)
    rows = downloader.scan_history_with_retry(run, page_size=20, retries=1, sleep_s=0)
    assert len(rows) == 3
    assert rows[0]["selection_phase"] == "random_init"
    assert rows[1]["fidelity"] == 0.2
    assert rows[1]["n_new_samples"] == 16
    assert rows[2]["cum_eval"] == 48
    assert rows[2]["simple_regret"] == 0.1
    assert rows[2]["cum_original_cost"] == 72


@pytest.mark.parametrize("compressed", (False, True))
def test_existing_download_csv_schema_widens_without_changing_old_values(tmp_path, compressed):
    path = tmp_path / ("history.csv.gz" if compressed else "summary.csv")
    opener = gzip.open if compressed else Path.open
    with opener(path, "wt", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=["run_id", "simple_regret"])
        writer.writeheader()
        writer.writerow({"run_id": "legacy", "simple_regret": "0.123"})
    downloader.ensure_csv_header(
        path, ["run_id", "simple_regret", "fidelity"], gzip_file=compressed
    )
    with opener(path, "rt", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows == [{"run_id": "legacy", "simple_regret": "0.123", "fidelity": ""}]


def test_existing_unknown_schema_is_rejected_without_overwriting(tmp_path):
    path = tmp_path / "unknown.csv"
    original = "run_id,unexpected\nlegacy,keep_me\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="unknown columns"):
        downloader.ensure_csv_header(path, ["run_id", "fidelity"])
    assert path.read_text() == original


def test_expected_grid_accepts_single_value_bo_inputs_and_matrix_only(tmp_path):
    path = tmp_path / "sweep.yml"
    path.write_text("""parameters:
  matrix:
    value: data/gsm8k_seed1.npy
  bo_inputs:
    value: data/gsm8k_seed1_bo_inputs.npz
  run_seed:
    values: [0, 1]
  experiment_variant:
    values: [mfkg_unit, mfkg_cost]
""")
    rows = downloader.load_expected_rows_from_sweep_yaml(path, "auto")
    assert len(rows) == 4
    assert rows[0]["benchmark_key"] == "gsm8k_seed1"
    assert rows[0]["matrix"] == "data/gsm8k_seed1.npy"
    path.write_text("""parameters:
  matrix:
    value: data/MMLU_matrices/abstract_algebra.npy
  seed:
    value: 0
  experiment_variant:
    value: mfkg_unit
""")
    rows = downloader.load_expected_rows_from_sweep_yaml(path, "auto")
    assert len(rows) == 1
    assert rows[0]["benchmark_key"] == "mmlu_abstract_algebra"
    assert downloader.config_key(rows[0]) == (
        "data/MMLU_matrices/abstract_algebra.npy", "0", "mfkg_unit"
    )


def test_expected_grid_distinguishes_fixed_batches(tmp_path):
    path = tmp_path / "sweep.yml"
    path.write_text("""parameters:
  matrix:
    value: data/gsm8k_seed1.npy
  run_seed:
    value: 0
  experiment_variant:
    value: mfkg_unit
  allocation_mode:
    value: fixed_batch
  batch_size:
    values: [4, 16]
""")
    rows = downloader.load_expected_rows_from_sweep_yaml(path, "auto")
    assert len(rows) == 2
    assert len({downloader.config_key(row) for row in rows}) == 2
    assert {row["batch_size"] for row in rows} == {4, 16}
