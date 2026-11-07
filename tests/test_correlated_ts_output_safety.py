"""Regression tests for standalone result identity and overwrite protection."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_correlated_ts_baseline as cli


@pytest.fixture
def invoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    matrix = tmp_path / "matrix.npy"
    archive = tmp_path / "inputs.npz"
    np.save(matrix, np.array([[0., 1.], [1., 1.]], dtype=np.float32))
    np.savez(archive, X=np.array([[0], [1]]), arm_ids=np.arange(2), cost=np.ones(2)*2)
    calls = []

    def simulate(*args, **kwargs):
        calls.append(kwargs)
        return dict(x=np.array([1]), kernel_final_json="{}", elapsed_s=0.)

    monkeypatch.setattr(cli, "simulate_correlated_ts", simulate)
    monkeypatch.setattr(cli, "exact_normalized_auc", lambda *args: 0.)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["runner", "--matrix", str(matrix),
                                        "--bo-inputs", str(archive), *args])
        return cli.main()

    return run, calls


@pytest.mark.parametrize("option,value", [
    ("--prior-mean", ".2"), ("--prior-variance", ".01"),
    ("--batch-size", "2"), ("--tau-sq-cell", ".1"),
    ("--recommendation-std-penalty", "0"), ("--eval-budget-fraction", ".2"),
    ("--kernel-mode", "fixed"), ("--fit-every", "4"),
    ("--fit-maxiter", "3"), ("--max-steps", "1"), ("--seed", "1"),
])
def test_different_settings_preserve_both_default_outputs(invoke, option, value):
    run, calls = invoke
    assert run() == 0
    assert run(option, value) == 0
    outputs = list(Path("outputs/correlated_ts").glob("*.npz"))
    assert len(outputs) == len(calls) == 2
    fingerprints = []
    for output in outputs:
        with np.load(output, allow_pickle=False) as saved:
            fingerprints.append(json.loads(str(saved["metadata_json"].item()))["experiment_sha256"])
    assert len(set(fingerprints)) == 2


@pytest.mark.parametrize("changed_input", ["matrix", "features", "costs"])
def test_changed_input_contents_preserve_outputs(invoke, changed_input):
    run, calls = invoke
    run()
    if changed_input == "matrix":
        np.save("matrix.npy", np.ones((2, 2), dtype=np.float32))
    else:
        np.savez("inputs.npz", X=np.array([[0], [2 if changed_input == "features" else 1]]),
                 arm_ids=np.arange(2), cost=np.array([2., 4. if changed_input == "costs" else 2.]))
    run()
    assert len(list(Path("outputs/correlated_ts").glob("*.npz"))) == len(calls) == 2


def test_explicit_path_reuses_matching_result_and_rejects_conflict(invoke):
    run, calls = invoke
    run("--out", "trace.npz")
    original = Path("trace.npz").read_bytes()
    Path("trace.json").unlink()
    run("--out", "trace.npz")
    assert Path("trace.json").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run("--out", "trace.npz", "--prior-mean", ".2")
    assert Path("trace.npz").read_bytes() == original
    assert len(calls) == 1
    assert not Path("trace.npz.lock").exists()


def test_legacy_result_is_not_overwritten(invoke):
    run, calls = invoke
    np.savez("trace.npz", metadata_json="{}")
    original = Path("trace.npz").read_bytes()
    with pytest.raises(FileExistsError, match="legacy"):
        run("--out", "trace.npz")
    assert not calls
    assert Path("trace.npz").read_bytes() == original


def test_other_writer_lock_is_preserved(invoke):
    run, calls = invoke
    with cli._output_lock(Path("trace.npz")):
        with pytest.raises(FileExistsError, match="locked"):
            run("--out", "trace.npz")
        assert Path("trace.npz.lock").exists()
    assert not Path("trace.npz.lock").exists()
    assert not calls


def test_failed_run_releases_lock(invoke, monkeypatch):
    run, _ = invoke

    def fail(*args, **kwargs):
        raise RuntimeError("simulation failed")

    monkeypatch.setattr(cli, "simulate_correlated_ts", fail)
    with pytest.raises(RuntimeError, match="simulation failed"):
        run("--out", "trace.npz")
    assert not Path("trace.npz.lock").exists()
    assert not Path("trace.npz").exists()
