#!/usr/bin/env python3
"""Convert evaluation matrices to BO baseline inputs.

Core output:
  - data/bo_inputs/{dataset}/{matrix_stem}_bo_inputs.npz

Optional debug sidecar files with --write-sidecars:
  - data/bo_inputs/{dataset}/{matrix_stem}_bo_configs.csv
  - data/bo_inputs/{dataset}/{matrix_stem}_bo_metadata.json

The BO input .npz contains:
  X: shape (n_configs, n_features), integer configuration features
  Y: shape (n_configs, 1), average score over all examples
  arm_ids: shape (n_configs,)
  cost: shape (n_configs,), full-evaluation cost for the configuration
  metadata_json: JSON string containing feature maps and provenance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TEMP_ID_MAP = {0.0: 0, 0.5: 1, 1.0: 2}
MAX_LEN_ID_MAP = {128: 0, 512: 1}
PROMPT_ID_MAP = {"empty": 0, "step_by_step": 1}
PROMPT_NAME_TO_TYPE = {"empty": "direct", "step_by_step": "cot"}
BANDITEVAL_FEATURE_NAMES = ["model_id", "temp_id", "max_len_id", "prompt_id"]
MMLU_FEATURE_NAMES = ["model_idx", "prompt_idx"]
ALPACA_FEATURE_NAMES = ["model_idx"]
MODEL_ID_MAP_FIXED = {
    "gpt2": 0,
    "gpt2_large": 1,
    "codellama": 2,
    "tulu": 3,
    "tulu2": 4,
    "gemma_7b": 5,
    "phi2": 6,
    "llemma_7b": 7,
    "llama2_7b": 8,
    "mistral_7b": 9,
    "starcoder2_7b": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help=(
            "Input BanditEval matrix .npy path, e.g. "
            "data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy"
        ),
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help=(
            "Optional configuration/pricing JSON path. If omitted, infer from dataset "
            "using the matrix filename prefix."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/bo_inputs"),
        help="Output root directory. Final files are written under {out_dir}/{dataset}/.",
    )
    parser.add_argument(
        "--write-sidecars",
        action="store_true",
        help="Also write human-readable *_bo_configs.csv and *_bo_metadata.json files.",
    )
    return parser.parse_args()


def infer_dataset_and_seed(matrix_path: Path) -> tuple[str, int | None]:
    stem = matrix_path.stem.lower()

    if matrix_path.parent.name == "MMLU_matrices":
        return "mmlu", None

    if stem.startswith("gsm8k_"):
        dataset = "gsm8k"
    elif stem.startswith("piqa_"):
        dataset = "piqa"
    elif stem.startswith("alpaca_eval_"):
        return "alpaca", None
    else:
        raise ValueError(
            "Unable to infer dataset from matrix filename. "
            "Expected prefix 'gsm8k_', 'piqa_', or 'alpaca_eval_'. "
            f"Got: {matrix_path.name}"
        )

    seed_marker = "seed"
    idx = stem.rfind(seed_marker)
    if idx < 0:
        raise ValueError(
            f"Unable to infer seed from matrix filename because it does not contain 'seedN': "
            f"{matrix_path.name}"
        )

    suffix = stem[idx + len(seed_marker):]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break

    if not digits:
        raise ValueError(f"Unable to infer seed digits from matrix filename: {matrix_path.name}")

    return dataset, int("".join(digits))


def default_config_json_path(dataset: str) -> Path:
    if dataset == "gsm8k":
        return Path(
            "data_analysis/pricing/"
            "gsm8k_various_models_configurations_price_ratio_1to2_rounded.json"
        )
    if dataset == "piqa":
        return Path(
            "data_analysis/pricing/"
            "piqa_various_models_configurations_input_price.json"
        )
    if dataset == "mmlu":
        return Path("data_analysis/pricing/mmlu_prompt_eval_configurations_input_price.json")
    if dataset == "alpaca":
        return Path(
            "data_analysis/pricing/"
            "alpaca_153_models_no_rounding_debias_price_1to8.json"
        )
    raise ValueError(f"Unsupported dataset for default config JSON: {dataset}")


def load_matrix(matrix_path: Path) -> np.ndarray:
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    matrix = np.asarray(np.load(matrix_path), dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Matrix must be 2D with shape (n_configs, n_examples), got {matrix.shape}"
        )
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"Matrix must have positive dimensions, got {matrix.shape}")

    return matrix


def load_and_validate_configs(config_json_path: Path, n_configs: int, dataset: str) -> list[dict]:
    if not config_json_path.is_file():
        raise FileNotFoundError(f"Config JSON file not found: {config_json_path}")

    with config_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Config JSON must be a dict keyed by string indices: '0', '1', ...")

    numeric_items: dict[str, dict] = {}
    keys_int: list[int] = []
    for key, value in data.items():
        if str(key).startswith("_"):
            continue
        try:
            keys_int.append(int(key))
            numeric_items[str(int(key))] = value
        except ValueError as exc:
            raise ValueError(f"Config JSON key is not an integer string: {key!r}") from exc

    keys_sorted = sorted(keys_int)
    expected_keys = list(range(len(keys_sorted)))
    if keys_sorted != expected_keys:
        raise ValueError(
            "Config JSON keys must be continuous from 0 to N-1. "
            f"Got first keys {keys_sorted[:10]}, expected first keys {expected_keys[:10]}"
        )

    if len(numeric_items) != n_configs:
        raise ValueError(
            f"Config row count mismatch: len(configs)={len(numeric_items)}, "
            f"but matrix has n_configs={n_configs}"
        )

    if dataset == "mmlu":
        required_fields = [
            "model_name",
            "model_idx",
            "prompt_idx",
            "estimated_cost_per_1m_input_tokens",
        ]
    elif dataset == "alpaca":
        required_fields = [
            "model_name",
            "estimated_cost_per_1m_input_tokens",
        ]
    else:
        required_fields = [
            "model_name",
            "temperature",
            "max_tokens",
            "prompt_name",
            "estimated_cost_per_1m_input_tokens",
        ]

    rows: list[dict] = []
    for i in range(n_configs):
        row = numeric_items.get(str(i))
        if not isinstance(row, dict):
            raise ValueError(f"Config row {i} is missing or not a dict.")

        for field in required_fields:
            if field not in row:
                raise ValueError(f"Config row {i} missing required field: {field}")

        rows.append(row)

    return rows


def build_model_id_map(config_rows: list[dict]) -> dict[str, int]:
    """Return fixed model-id map and validate config model names.

    We intentionally keep a fixed mapping shared by GSM8K/PIQA:
      0 -> gpt2
      1 -> gpt2_large
      2 -> codellama
      3 -> tulu
      4 -> tulu2
      5 -> gemma_7b
      6 -> phi2
      7 -> llemma_7b
      8 -> llama2_7b
      9 -> mistral_7b
      10 -> starcoder2_7b
    """
    observed = {str(row["model_name"]) for row in config_rows}
    unknown = sorted(observed - set(MODEL_ID_MAP_FIXED.keys()))
    if unknown:
        raise ValueError(
            "Found model_name values not covered by fixed MODEL_ID_MAP_FIXED: "
            f"{unknown}"
        )
    return dict(MODEL_ID_MAP_FIXED)


def to_float_temperature(value: object, row_idx: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid temperature at row {row_idx}: {value!r}") from exc


def to_int_max_tokens(value: object, row_idx: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid max_tokens at row {row_idx}: {value!r}") from exc


def construct_arrays(
    config_rows: list[dict],
    model_id_map: dict[str, int],
    matrix: np.ndarray,
    dataset: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    list[str],
]:
    n_configs, n_examples = matrix.shape
    if dataset == "mmlu":
        feature_names = MMLU_FEATURE_NAMES
    elif dataset == "alpaca":
        feature_names = ALPACA_FEATURE_NAMES
    else:
        feature_names = BANDITEVAL_FEATURE_NAMES

    X = np.empty((n_configs, len(feature_names)), dtype=np.int64)
    Y = matrix.mean(axis=1, keepdims=True).astype(np.float64)
    arm_ids = np.arange(n_configs, dtype=np.int64)
    cost = np.empty((n_configs,), dtype=np.float64)

    model_names: list[str] = []
    prompt_names: list[str] = []
    prompt_types: list[str] = []
    temperatures = np.empty((n_configs,), dtype=np.float64)
    max_tokens_values = np.empty((n_configs,), dtype=np.int64)

    csv_rows: list[dict] = []

    for i, row in enumerate(config_rows):
        model_name = str(row["model_name"])
        cost[i] = float(row["estimated_cost_per_1m_input_tokens"]) * float(n_examples)
        model_names.append(model_name)

        if dataset == "mmlu":
            model_idx = int(row["model_idx"])
            prompt_idx = int(row["prompt_idx"])
            X[i, 0] = model_idx
            X[i, 1] = prompt_idx
            temperatures[i] = np.nan
            max_tokens_values[i] = -1
            prompt_names.append(f"prompt_{prompt_idx}")
            prompt_types.append("")
            csv_row = {
                "arm_id": int(arm_ids[i]),
                "model_idx": model_idx,
                "prompt_idx": prompt_idx,
                "model_name": model_name,
                "cost": float(cost[i]),
                "raw_estimated_cost_per_1m_input_tokens": float(
                    row["estimated_cost_per_1m_input_tokens"]
                ),
                "average_score": float(Y[i, 0]),
            }
        elif dataset == "alpaca":
            model_idx = int(row.get("model_idx", i))
            X[i, 0] = model_idx
            temperatures[i] = np.nan
            max_tokens_values[i] = -1
            prompt_names.append("")
            prompt_types.append("")
            csv_row = {
                "arm_id": int(arm_ids[i]),
                "model_idx": model_idx,
                "model_name": model_name,
                "cost": float(cost[i]),
                "raw_estimated_cost_per_1m_input_tokens": float(
                    row["estimated_cost_per_1m_input_tokens"]
                ),
                "average_score": float(Y[i, 0]),
            }
            if "original_index" in row:
                csv_row["original_index"] = int(row["original_index"])
        else:
            temperature = to_float_temperature(row["temperature"], i)
            max_tokens = to_int_max_tokens(row["max_tokens"], i)
            prompt_name = str(row["prompt_name"])

            if temperature not in TEMP_ID_MAP:
                raise ValueError(
                    f"Unsupported temperature at row {i}: {temperature}. "
                    f"Supported values: {list(TEMP_ID_MAP.keys())}"
                )

            if max_tokens not in MAX_LEN_ID_MAP:
                raise ValueError(
                    f"Unsupported max_tokens at row {i}: {max_tokens}. "
                    f"Supported values: {list(MAX_LEN_ID_MAP.keys())}"
                )

            if prompt_name not in PROMPT_ID_MAP:
                raise ValueError(
                    f"Unsupported prompt_name at row {i}: {prompt_name}. "
                    f"Supported values: {list(PROMPT_ID_MAP.keys())}"
                )

            prompt_type = PROMPT_NAME_TO_TYPE[prompt_name]

            X[i, 0] = model_id_map[model_name]
            X[i, 1] = TEMP_ID_MAP[temperature]
            X[i, 2] = MAX_LEN_ID_MAP[max_tokens]
            X[i, 3] = PROMPT_ID_MAP[prompt_name]
            temperatures[i] = temperature
            max_tokens_values[i] = max_tokens
            prompt_names.append(prompt_name)
            prompt_types.append(prompt_type)
            csv_row = {
                "arm_id": int(arm_ids[i]),
                "model_id": int(X[i, 0]),
                "temp_id": int(X[i, 1]),
                "max_len_id": int(X[i, 2]),
                "prompt_id": int(X[i, 3]),
                "model_name": model_name,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "prompt_name": prompt_name,
                "prompt_type": prompt_type,
                "cost": float(cost[i]),
                "raw_estimated_cost_per_1m_input_tokens": float(
                    row["estimated_cost_per_1m_input_tokens"]
                ),
                "average_score": float(Y[i, 0]),
            }
        csv_rows.append(csv_row)

    configs_df = pd.DataFrame(csv_rows)

    return (
        X,
        Y,
        arm_ids,
        cost,
        np.asarray(model_names, dtype=str),
        temperatures,
        max_tokens_values,
        np.asarray(prompt_names, dtype=str),
        np.asarray(prompt_types, dtype=str),
        configs_df,
        feature_names,
    )


def build_metadata(
    *,
    dataset: str,
    seed: int | None,
    matrix_path: Path,
    config_json_path: Path,
    matrix: np.ndarray,
    model_id_map: dict[str, int],
    feature_names: list[str],
) -> dict:
    feature_definition = {
        f"X[:, {idx}]": name for idx, name in enumerate(feature_names)
    }
    feature_definition["Y[:, 0]"] = (
        "average score over all examples for the same row/configuration"
    )

    metadata = {
        "dataset": dataset,
        "matrix_seed": None if seed is None else int(seed),
        "n_configs": int(matrix.shape[0]),
        "n_examples": int(matrix.shape[1]),
        "feature_names": feature_names,
        "feature_definition": feature_definition,
        "cost_multiplier_applied_to_bo_inputs": int(matrix.shape[1]),
        "cost_field_note": (
            "cost is the estimated full-evaluation cost for this configuration, not a "
            "per-example evaluation cost. It is computed from estimated_cost_per_1m_input_tokens "
            "by multiplying by the number of examples in the source matrix."
        ),
        "matrix_path": str(matrix_path),
        "config_json_path": str(config_json_path),
        "source_description": (
            "Converted from a BanditEval correctness matrix and its aligned "
            "configuration/pricing JSON. Rows are aligned by arm_id/index order. "
            "The per-example 0/1 matrix is compressed into Y by averaging across examples."
        ),
    }
    if dataset == "mmlu":
        metadata["feature_note"] = "MMLU rows are aligned by model_idx and prompt_idx."
    elif dataset == "alpaca":
        metadata["feature_note"] = (
            "Alpaca rows are aligned by the filtered no_rounding_debias model order. "
            "X[:, 0] stores the contiguous filtered row/model index; original_index is "
            "preserved in the sidecar configs when available."
        )
    else:
        model_id_to_name = [None] * len(model_id_map)
        for name, idx in model_id_map.items():
            model_id_to_name[idx] = name
        metadata.update(
            {
                "model_id_map": model_id_map,
                "model_id_to_name": model_id_to_name,
                "temp_id_map": {str(k): int(v) for k, v in TEMP_ID_MAP.items()},
                "max_len_id_map": {str(k): int(v) for k, v in MAX_LEN_ID_MAP.items()},
                "prompt_id_map": PROMPT_ID_MAP,
                "prompt_name_to_type": PROMPT_NAME_TO_TYPE,
            }
        )
    return metadata


def write_outputs(
    *,
    dataset: str,
    seed: int | None,
    out_dir: Path,
    matrix_path: Path,
    config_json_path: Path,
    matrix: np.ndarray,
    model_id_map: dict[str, int],
    X: np.ndarray,
    Y: np.ndarray,
    arm_ids: np.ndarray,
    cost: np.ndarray,
    model_names: np.ndarray,
    temperatures: np.ndarray,
    max_tokens_values: np.ndarray,
    prompt_names: np.ndarray,
    prompt_types: np.ndarray,
    configs_df: pd.DataFrame,
    feature_names: list[str],
    write_sidecars: bool,
) -> tuple[Path, Path | None, Path | None]:
    dataset_dir = out_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Full, unambiguous output names.
    output_stem = f"{matrix_path.stem}_bo"

    npz_path = dataset_dir / f"{output_stem}_inputs.npz"
    csv_path = dataset_dir / f"{output_stem}_configs.csv"
    metadata_path = dataset_dir / f"{output_stem}_metadata.json"

    metadata = build_metadata(
        dataset=dataset,
        seed=seed,
        matrix_path=matrix_path,
        config_json_path=config_json_path,
        matrix=matrix,
        model_id_map=model_id_map,
        feature_names=feature_names,
    )
    metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
    matrix_seed_array = (
        np.asarray(seed, dtype=np.int64)
        if seed is not None
        else np.asarray("", dtype="<U32")
    )

    np.savez(
        npz_path,
        X=X,
        Y=Y,
        arm_ids=arm_ids,
        cost=cost,
        raw_scores_mean=Y[:, 0].copy(),
        model_names=model_names,
        temperatures=temperatures,
        max_tokens=max_tokens_values,
        prompt_names=prompt_names,
        prompt_types=prompt_types,
        feature_names=np.asarray(feature_names, dtype="<U32"),
        matrix_path=np.asarray(str(matrix_path), dtype="<U512"),
        config_json_path=np.asarray(str(config_json_path), dtype="<U512"),
        dataset=np.asarray(dataset, dtype="<U32"),
        matrix_seed=matrix_seed_array,
        metadata_json=np.asarray(metadata_json),
    )

    written_csv_path: Path | None = None
    written_metadata_path: Path | None = None

    if write_sidecars:
        configs_df.to_csv(csv_path, index=False)
        metadata_path.write_text(metadata_json, encoding="utf-8")
        written_csv_path = csv_path
        written_metadata_path = metadata_path

    return npz_path, written_csv_path, written_metadata_path


def main() -> int:
    args = parse_args()

    matrix_path = args.matrix
    dataset, seed = infer_dataset_and_seed(matrix_path)

    config_json_path = (
        args.config_json
        if args.config_json is not None
        else default_config_json_path(dataset)
    )

    matrix = load_matrix(matrix_path)
    config_rows = load_and_validate_configs(
        config_json_path,
        n_configs=int(matrix.shape[0]),
        dataset=dataset,
    )
    model_id_map = {} if dataset in {"mmlu", "alpaca"} else build_model_id_map(config_rows)

    (
        X,
        Y,
        arm_ids,
        cost,
        model_names,
        temperatures,
        max_tokens_values,
        prompt_names,
        prompt_types,
        configs_df,
        feature_names,
    ) = construct_arrays(
        config_rows=config_rows,
        model_id_map=model_id_map,
        matrix=matrix,
        dataset=dataset,
    )

    npz_path, csv_path, metadata_path = write_outputs(
        dataset=dataset,
        seed=seed,
        out_dir=args.out_dir,
        matrix_path=matrix_path,
        config_json_path=config_json_path,
        matrix=matrix,
        model_id_map=model_id_map,
        X=X,
        Y=Y,
        arm_ids=arm_ids,
        cost=cost,
        model_names=model_names,
        temperatures=temperatures,
        max_tokens_values=max_tokens_values,
        prompt_names=prompt_names,
        prompt_types=prompt_types,
        configs_df=configs_df,
        feature_names=feature_names,
        write_sidecars=bool(args.write_sidecars),
    )

    print(f"dataset: {dataset}")
    print(f"seed: {seed}")
    print(f"matrix shape: {matrix.shape}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"cost shape: {cost.shape}")
    print("outputs:")
    print(f"  npz: {npz_path}")
    if csv_path is not None:
        print(f"  configs csv: {csv_path}")
    if metadata_path is not None:
        print(f"  metadata json: {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
