from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

MATS = [
    "abstract_algebra",
    "business_ethics",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "computer_security",
    "global_facts",
    "high_school_computer_science",
    "formal_logic",
    "human_sexuality",
    "anatomy",
    "college_biology",
    "electrical_engineering",
    "medical_genetics",
    "us_foreign_policy",
    "college_physics",
    "management",
    "jurisprudence",
    "public_relations",
    "machine_learning",
    "econometrics",
    "international_law",
]

ROOTS = [
    ("missing7", Path(r"outputs\wandb_downloads\mmlu_small_missing7_finished_with_stopping"), 3),
    ("selected_original", Path(r"outputs\wandb_downloads\mmlu_small_selected_matrices_finished"), 0),
    ("remaining_original", Path(r"outputs\wandb_downloads\mmlu_small_remaining_original_finished_with_stopping"), 0),
    ("rerun", Path(r"outputs\wandb_downloads\mmlu_small_rerun_finished_with_stopping"), 1),
    ("cleanup", Path(r"outputs\wandb_downloads\mmlu_small_cleanup_finished_with_stopping"), 2),
]

OUT_ROOT = Path(r"outputs\wandb_downloads\mmlu_small_merged_finished_with_stopping")
EXPECTED_PER_MATRIX = 840


def read_csv_maybe(path: Path):
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def get_run_id_col(df: pd.DataFrame) -> str:
    for c in ["run_id", "id"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find run id column in columns: {list(df.columns)}")


def merge_one_matrix(mat: str) -> dict:
    out_dir = OUT_ROOT / mat
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_parts = []

    for source_name, root, priority in ROOTS:
        p = root / mat / "runs_summary.csv"
        df = read_csv_maybe(p)
        if df is None:
            continue
        df["_source_name"] = source_name
        df["_source_priority"] = priority
        summary_parts.append(df)

    if not summary_parts:
        return {"matrix": mat, "n_summary": 0, "n_history_rows": 0, "n_stopping": 0, "missing": EXPECTED_PER_MATRIX}

    summary = pd.concat(summary_parts, ignore_index=True)

    if "experiment_variant" not in summary.columns or "run_seed" not in summary.columns:
        raise ValueError(f"{mat}: missing experiment_variant or run_seed in summary")

    run_id_col = get_run_id_col(summary)

    summary["_run_seed_str"] = summary["run_seed"].astype(str)
    summary["_dedupe_key"] = summary["experiment_variant"].astype(str) + "||" + summary["_run_seed_str"]

    summary = summary.sort_values(["_dedupe_key", "_source_priority"], ascending=[True, True])
    summary = summary.drop_duplicates("_dedupe_key", keep="last").copy()

    kept_run_ids = set(summary[run_id_col].astype(str))

    helper_cols = ["_source_priority", "_run_seed_str", "_dedupe_key"]
    summary_out = summary.drop(columns=[c for c in helper_cols if c in summary.columns])
    summary_out.to_csv(out_dir / "runs_summary.csv", index=False)

    # Merge history
    hist_parts = []
    for source_name, root, priority in ROOTS:
        p = root / mat / "runs_history.csv.gz"
        df = read_csv_maybe(p)
        if df is None:
            continue
        if run_id_col in df.columns:
            df = df[df[run_id_col].astype(str).isin(kept_run_ids)]
        else:
            h_run_id_col = get_run_id_col(df)
            df = df[df[h_run_id_col].astype(str).isin(kept_run_ids)]
        if len(df):
            hist_parts.append(df)

    n_history_rows = 0
    if hist_parts:
        hist = pd.concat(hist_parts, ignore_index=True)
        n_history_rows = len(hist)
        hist.to_csv(out_dir / "runs_history.csv.gz", index=False, compression="gzip")

    # Merge stopping
    stop_parts = []
    for source_name, root, priority in ROOTS:
        p = root / mat / "runs_stopping.csv"
        df = read_csv_maybe(p)
        if df is None:
            continue
        s_run_id_col = get_run_id_col(df)
        df = df[df[s_run_id_col].astype(str).isin(kept_run_ids)]
        if len(df):
            df["_source_name"] = source_name
            df["_source_priority"] = priority
            stop_parts.append(df)

    n_stopping = 0
    if stop_parts:
        stop = pd.concat(stop_parts, ignore_index=True)
        if "experiment_variant" in stop.columns and "run_seed" in stop.columns:
            stop["_run_seed_str"] = stop["run_seed"].astype(str)
            stop["_dedupe_key"] = stop["experiment_variant"].astype(str) + "||" + stop["_run_seed_str"]
            stop = stop.sort_values(["_dedupe_key", "_source_priority"], ascending=[True, True])
            stop = stop.drop_duplicates("_dedupe_key", keep="last")
        stop = stop.drop(columns=[c for c in ["_source_priority", "_run_seed_str", "_dedupe_key"] if c in stop.columns])
        n_stopping = len(stop)
        stop.to_csv(out_dir / "runs_stopping.csv", index=False)

    missing = EXPECTED_PER_MATRIX - len(summary_out)

    meta = {
        "matrix": mat,
        "n_summary": int(len(summary_out)),
        "n_history_rows": int(n_history_rows),
        "n_stopping": int(n_stopping),
        "expected": EXPECTED_PER_MATRIX,
        "missing": int(missing),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for mat in MATS:
        print(f"merging {mat} ...")
        meta = merge_one_matrix(mat)
        all_meta.append(meta)
        print(meta)

    out = {
        "out_root": str(OUT_ROOT),
        "expected_per_matrix": EXPECTED_PER_MATRIX,
        "total_expected": EXPECTED_PER_MATRIX * len(MATS),
        "total_summary": int(sum(x["n_summary"] for x in all_meta)),
        "total_missing": int(sum(max(0, x["missing"]) for x in all_meta)),
        "matrices": all_meta,
    }

    (OUT_ROOT / "metadata_all_matrices.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nDONE")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()