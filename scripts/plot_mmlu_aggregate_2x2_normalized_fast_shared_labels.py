"""Backward-compatible shim for scripts/plot/plot_mmlu_aggregate_2x2_normalized_fast_shared_labels.py."""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "plot" / "plot_mmlu_aggregate_2x2_normalized_fast_shared_labels.py"
_spec = importlib.util.spec_from_file_location(
    "_mmlu_aggregate_impl",
    _IMPL,
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules.setdefault("_mmlu_aggregate_impl", _mod)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})

if __name__ == "__main__":
    runpy.run_path(str(_IMPL), run_name="__main__")
