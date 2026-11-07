"""Plot completed 57-subject PromptEval feature comparisons and export budget checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MATRIX_DIR = REPO / "data" / "MMLU_matrices"
COST_FILE = REPO / "data_analysis" / "pricing" / "mmlu_prompt_eval_configurations_input_price.json"
FEATURES = ("One-hot", "Discrete", "Pretrained", "Finetuned (diagnostic)")
FEATURE_KEYS = ("one_hot", "discrete", "pretrained_embedding", "fine_tuned_embedding")
COST_SETTINGS = ("Unit cost", "Price-weighted spend")
SPEND_FIELDS = ("budget_obs", "budget_cost")
COLORS = ("#454c55", "#0072b2", "#d55e00", "#8754ad")
CHECKPOINTS = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
PLOT_GRID = np.linspace(0.0, 10.0, 401)
SEEDS = tuple(range(20))
BUDGET_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Trace:
    spend_percent: np.ndarray
    regret: np.ndarray


@dataclass(frozen=True)
class Subject:
    name: str
    size: str
    example_count: int
    exhaustive_observations: int
    exhaustive_price: float
    archive_sha256: str
    traces: tuple[tuple[Trace, ...], ...]


def size_group(example_count: int) -> str:
    if example_count <= 150:
        return "Small"
    if example_count <= 400:
        return "Medium"
    return "Large"


def load_subjects(results_dir: Path) -> list[Subject]:
    metadata = json.loads((MATRIX_DIR / "task_metadata.json").read_text(encoding="utf-8"))
    subject_names = sorted(task["task"] for task in metadata["tasks"])
    if len(set(subject_names)) != 57 or len(subject_names) != 57:
        raise ValueError("Expected 57 distinct subjects in the MMLU metadata.")

    price_bytes = COST_FILE.read_bytes()
    pricing = json.loads(price_bytes)
    price_hash = hashlib.sha256(price_bytes).hexdigest()
    price_relpath = COST_FILE.relative_to(REPO).as_posix()
    costs = np.array([pricing[str(arm)]["estimated_cost_per_1m_input_tokens"] for arm in range(1500)])
    if len(pricing) != 1500 or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("Expected one positive finite price for each of the 1,500 arms.")

    paths = {name: results_dir / f"bai_results_MMLU_{name}_combined.npy" for name in subject_names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed archives for {len(missing)} subjects: {', '.join(missing)}")

    subjects = []
    for name, path in paths.items():
        matrix_path = MATRIX_DIR / f"{name}.npy"
        arm_count, example_count = np.load(matrix_path, mmap_mode="r").shape
        if arm_count != 1500:
            raise ValueError(f"{name}: expected 1,500 arms, got {arm_count}.")
        exhaustive_observations = arm_count * example_count
        exhaustive_price = float(costs.sum() * example_count)
        denominators = np.array([exhaustive_observations, exhaustive_price])[:, None]
        payload = np.load(path, allow_pickle=True).item()
        expected_jobs = [(name, seed) for seed in SEEDS]
        if (
            [tuple(job) for job in payload["jobs"]] != expected_jobs
            or list(payload["seeds"]) != list(SEEDS)
            or len(payload["out"]) != len(SEEDS)
            or not payload["combine_models"]
            or payload["n_models_stacked"] != 15
            or not payload["costs_recorded"]
            or tuple(payload["feature_names"]) != FEATURE_KEYS
        ):
            raise ValueError(f"{path}: expected complete joint-arm results for seeds 0–19 with both spend channels.")
        provenance = payload["reproduction"]
        if provenance["input_sha256"].get(price_relpath) != price_hash:
            raise ValueError(f"{name}: the pricing file differs from the archived experiment input.")
        if provenance["matrix_sha256"] != hashlib.sha256(matrix_path.read_bytes()).hexdigest():
            raise ValueError(f"{name}: the matrix differs from the archived experiment input.")

        seed_traces = []
        for seed, run in zip(SEEDS, payload["out"]):
            if len(run) != 1 or len(run[0]) != len(FEATURES):
                raise ValueError(f"{name}, seed {seed}: expected all four feature variants in one budget run.")
            traces = []
            for feature, phases in zip(FEATURES, run[0]):
                if not 1 <= len(phases) <= 5:
                    raise ValueError(f"{name}, seed {seed}, {feature}: expected one to five phases.")
                if [phase["phase"] for phase in phases] != list(range(len(phases))):
                    raise ValueError(f"{name}, seed {seed}, {feature}: phases are out of order.")
                spend = np.array([[phase[field] for phase in phases] for field in SPEND_FIELDS], dtype=float)
                regret = np.array([phase["simple_regret"] for phase in phases], dtype=float)
                if (
                    not np.isfinite(spend).all()
                    or not np.isfinite(regret).all()
                    or np.any(spend <= 0)
                    or np.any(np.diff(spend, axis=1) < 0)
                    or np.any(regret < -BUDGET_TOLERANCE)
                    or np.any(regret > 1 + BUDGET_TOLERANCE)
                ):
                    raise ValueError(f"{name}, seed {seed}, {feature}: invalid spend or regret data.")
                traces.append(Trace(100.0 * spend / denominators, regret))
            seed_traces.append(tuple(traces))

        subjects.append(Subject(
            name, size_group(example_count), example_count, exhaustive_observations, exhaustive_price,
            hashlib.sha256(path.read_bytes()).hexdigest(), tuple(seed_traces),
        ))
    return subjects


def observations_at(subject: Subject, cost_index: int, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.full((len(FEATURES), len(SEEDS), len(grid)), np.nan)
    held_past_last = np.zeros(observed.shape, dtype=bool)
    for seed_index, traces in enumerate(subject.traces):
        for feature_index, trace in enumerate(traces):
            spend = trace.spend_percent[cost_index]
            indices = np.searchsorted(spend, grid + BUDGET_TOLERANCE, side="right") - 1
            available = indices >= 0
            observed[feature_index, seed_index, available] = trace.regret[indices[available]]
            held_past_last[feature_index, seed_index] = grid > spend[-1] + BUDGET_TOLERANCE
    return observed, held_past_last


def mean_and_se(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    mean = np.full(count.shape, np.nan)
    np.divide(np.nansum(values, axis=axis), count, where=count > 0, out=mean)
    residual = np.where(valid, values - np.expand_dims(mean, axis), 0.0)
    se = np.full(count.shape, np.nan)
    np.divide((residual * residual).sum(axis=axis), count * (count - 1), where=count > 1, out=se)
    return mean, np.sqrt(se), count


def measurements(subjects: list[Subject], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays indexed by cost setting, subject, feature, seed and budget."""
    both_settings = [
        [observations_at(subject, cost_index, grid) for subject in subjects]
        for cost_index, _ in enumerate(COST_SETTINGS)
    ]
    values = np.array([[observed for observed, _ in setting] for setting in both_settings])
    held = np.array([[past_last for _, past_last in setting] for setting in both_settings])
    return values, held


def cohorts(subjects: list[Subject]) -> dict[str, list[int]]:
    return {
        "All subjects": list(range(len(subjects))),
        **{size: [i for i, subject in enumerate(subjects) if subject.size == size] for size in ("Small", "Medium", "Large")},
    }


def draw_curves(
    ax: plt.Axes,
    means: np.ndarray,
    errors: np.ndarray,
    complete: np.ndarray,
    feature_indices: tuple[int, ...] = tuple(range(len(FEATURES))),
) -> None:
    x = PLOT_GRID[complete]
    for feature_index in feature_indices:
        label, color = FEATURES[feature_index], COLORS[feature_index]
        mean, error = means[feature_index, complete], errors[feature_index, complete]
        emphasis = feature_index in (0, 2)
        ax.step(
            x, mean, where="post", label=label, color=color,
            lw=2.3 if emphasis else 1.5, linestyle="--" if feature_index == 3 else "-",
            zorder=3 if emphasis else 2,
        )
        ax.fill_between(x, mean - error, mean + error, color=color, step="post", alpha=0.12, lw=0)
        ax.plot(x[-1:], mean[-1:], marker="o", markersize=3, color=color, zorder=4)
    ax.set(xlim=(0, 10.12), xticks=np.arange(0, 11, 2))
    ax.grid(axis="y", alpha=0.24, linewidth=0.6)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def render_onehot_vs_pretrained(out_dir: Path, values: np.ndarray, subject_means: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.95), sharex=True, sharey=True)
    focus = (0, 2)
    for cost_index, cost_setting in enumerate(COST_SETTINGS):
        ax = axes[cost_index]
        means, errors, _ = mean_and_se(subject_means[cost_index], axis=0)
        complete = np.isfinite(values[cost_index][:, focus]).all(axis=(0, 1, 2))
        draw_curves(ax, means, errors, complete, feature_indices=focus)
        ax.set_title(cost_setting, fontweight="semibold", pad=9)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Simple regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.054), ncol=2, frameon=False)
    fig.suptitle("MMLU: one-hot vs pretrained prompt embeddings", fontsize=16, fontweight="semibold", y=0.97)
    fig.text(
        0.5, 0.892,
        f"Existing logistic BAI  ·  {subject_means.shape[1]} subjects  ·  20 seeds per subject  ·  1,500 joint arms",
        ha="center",
    )
    fig.supxlabel("Percent of exhaustive evaluation", y=0.136)
    fig.text(
        0.5, 0.047,
        "Equal subject weights. Shading shows ±1 subject SE. Curves start once every run has a recommendation and carry the last one to 10%.",
        ha="center", fontsize=8.5, color="#454c55",
    )
    fig.subplots_adjust(left=0.09, right=0.978, bottom=0.23, top=0.795, wspace=0.13)
    save_figure(fig, out_dir / "mmlu_onehot_vs_pretrained")


def render_figures(out_dir: Path, subjects: list[Subject], values: np.ndarray) -> None:
    subject_means, subject_errors, _ = mean_and_se(values, axis=3)
    groups = cohorts(subjects)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "pdf.fonttype": 42})
    render_onehot_vs_pretrained(out_dir, values, subject_means)
    fig, axes = plt.subplots(2, 4, figsize=(13.3, 6.8), sharex=True, sharey="col")
    for cost_index, cost_setting in enumerate(COST_SETTINGS):
        for column, (group, members) in enumerate(groups.items()):
            ax = axes[cost_index, column]
            means, errors, _ = mean_and_se(subject_means[cost_index, members], axis=0)
            complete = np.isfinite(values[cost_index, members]).all(axis=(0, 1, 2))
            draw_curves(ax, means, errors, complete)
            if cost_index == 0:
                ax.set_title(f"{group} ({len(members)})", fontweight="semibold", pad=8)
            if column == 0:
                ax.set_ylabel(f"{cost_setting}\nSimple regret")
    for ax in axes[0]:
        ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.042), ncol=4, frameon=False)
    fig.suptitle("MMLU BAI feature comparison", fontsize=16, fontweight="semibold", y=0.975)
    fig.text(
        0.5, 0.91,
        "Existing logistic BAI  ·  20 seeds per subject  ·  Equal subject weights  ·  Shading shows ±1 subject SE",
        ha="center",
    )
    fig.supxlabel("Percent of exhaustive evaluation", y=0.109)
    fig.text(
        0.5, 0.024,
        "Each panel starts once every run in that cohort has a recommendation. Final recommendations carry forward to 10%.",
        ha="center", fontsize=8.5, color="#454c55",
    )
    fig.subplots_adjust(left=0.071, right=0.986, bottom=0.171, top=0.848, wspace=0.31, hspace=0.25)
    save_figure(fig, out_dir / "mmlu_feature_overview")

    subject_dir = out_dir / "subjects"
    subject_dir.mkdir(exist_ok=True)
    for subject_index, subject in enumerate(subjects):
        fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.05), sharex=True, sharey=True)
        for cost_index, cost_setting in enumerate(COST_SETTINGS):
            ax = axes[cost_index]
            complete = np.isfinite(values[cost_index, subject_index]).all(axis=(0, 1))
            draw_curves(ax, subject_means[cost_index, subject_index], subject_errors[cost_index, subject_index], complete)
            ax.set_title(cost_setting, fontweight="semibold")
        axes[0].set_ylim(bottom=0)
        axes[0].set_ylabel("Simple regret")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.042), ncol=4, frameon=False)
        fig.suptitle(subject.name.replace("_", " ").title(), fontsize=15, fontweight="semibold", y=0.974)
        fig.text(0.5, 0.884, f"{subject.size}  ·  {subject.example_count:,} examples  ·  ±1 standard error across seeds", ha="center")
        fig.supxlabel("Percent of exhaustive evaluation", y=0.119)
        fig.text(
            0.5, 0.023,
            "Each panel starts once every run has a recommendation. Final recommendations carry forward to 10%.",
            ha="center", fontsize=8, color="#454c55",
        )
        fig.subplots_adjust(left=0.093, right=0.974, bottom=0.225, top=0.795, wspace=0.14)
        save_figure(fig, subject_dir / subject.name)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_number(number: float) -> float | None:
    return float(number) if np.isfinite(number) else None


def export_checkpoints(out_dir: Path, subjects: list[Subject], values: np.ndarray, held: np.ndarray) -> list[dict]:
    mean, se, observed_count = mean_and_se(values, axis=3)
    difference, paired_se, paired_seed_count = mean_and_se(values - values[:, :, :1], axis=3)
    held_count = held.sum(axis=3)
    groups = cohorts(subjects)
    subject_rows = []
    aggregate_rows = []
    for cost_index, cost_setting in enumerate(COST_SETTINGS):
        for feature_index, feature in enumerate(FEATURES):
            for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                common = {"cost_setting": cost_setting, "budget_percent": checkpoint, "feature": feature}
                for subject_index, subject in enumerate(subjects):
                    at = cost_index, subject_index, feature_index, checkpoint_index
                    subject_rows.append({
                        **common, "subject": subject.name, "size": subject.size,
                        "mean_simple_regret": finite_number(mean[at]), "seed_se": finite_number(se[at]),
                        "available_seeds": observed_count[at], "total_seeds": len(SEEDS),
                        "seeds_held_past_last_phase": held_count[at],
                        "paired_delta_vs_one_hot": finite_number(difference[at]),
                        "paired_seed_se": finite_number(paired_se[at]), "paired_seeds": paired_seed_count[at],
                    })

                for group, members in groups.items():
                    at = cost_index, members, feature_index, checkpoint_index
                    group_mean, group_se, n_subjects = mean_and_se(mean[at], axis=0)
                    delta, delta_se, paired_subjects = mean_and_se(difference[at], axis=0)
                    aggregate_rows.append({
                        **common, "group": group,
                        "mean_simple_regret": finite_number(group_mean), "subject_se": finite_number(group_se),
                        "available_subjects": int(n_subjects), "total_subjects": len(members),
                        "available_seed_traces": int(observed_count[at].sum()),
                        "total_seed_traces": len(SEEDS) * len(members),
                        "traces_held_past_last_phase": int(held_count[at].sum()),
                        "paired_delta_vs_one_hot": finite_number(delta),
                        "paired_subject_se": finite_number(delta_se), "paired_subjects": int(paired_subjects),
                        "paired_seed_traces": int(paired_seed_count[at].sum()),
                    })

    write_csv(out_dir / "subject_checkpoints.csv", subject_rows)
    write_csv(out_dir / "checkpoint_summary.csv", aggregate_rows)
    return aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--out-dir", type=Path, default=HERE / "report")
    args = parser.parse_args()
    subjects = load_subjects(args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_values, held = measurements(subjects, CHECKPOINTS)
    checkpoints = export_checkpoints(args.out_dir, subjects, checkpoint_values, held)
    plot_values, _ = measurements(subjects, PLOT_GRID)
    render_figures(args.out_dir, subjects, plot_values)
    summary = {
        "method": "The repository's historical logistic-regression BAI with successive halving.",
        "variant_scope": "All four feature blocks use the same logistic BAI. This is a feature comparison within that implementation. The original PromptEval extended Rasch BAI is a separate method.",
        "logistic_scoring": {
            "solver": "LIBLINEAR",
            "penalty": "L2",
            "C": 100,
            "regularization_note": "The common C leaves differences in effective regularization from feature scale and dimension.",
        },
        "feature_source": {
            "path": "prompteval/data/Xs.pickle",
            "sha256": "2bd38d57a8c9a5de0e78ee803da795e9d7b7ac12657890fb398e798e954f6c2f",
            "provenance": "The bundled feature file matches the cached upstream PromptEval file byte for byte.",
        },
        "feature_descriptions": {
            "one_hot": {
                "joint_arm_identity_dimensions": 1500,
                "description": "One identity feature for each model and prompt pair.",
            },
            "discrete": {
                "template_metadata_fields": ["sep", "space", "op"],
                "description": "One-hot template metadata filtered by PCA.",
                "prompt_dimensions": 21,
                "model_dummy_dimensions": 14,
                "stacked_dimensions_before_logistic_intercept": 35,
            },
            "pretrained_embedding": {
                "encoder": "sentence-transformers/facebook-dpr-question_encoder-multiset-base",
                "description": "Encode the 100 template strings for each subject, then fit PCA on those embeddings. A second PCA filters components with explained-variance ratio at or below 1e-5. All 25 components survive in every subject. Before model dummies are appended, each subject's saved features are identical across its 15 models.",
                "uses_response_labels_for_features": False,
                "templates_per_subject": 100,
                "prompt_dimensions": 25,
                "model_dummy_dimensions": 14,
                "stacked_dimensions_before_logistic_intercept": 39,
            },
        },
        "finetuned_caveat": "Fine-tuned features were trained on responses from other candidate models. The original combined-model stack pads and concatenates folded PCA representations with incompatible coordinate systems. This block is shown as a diagnostic.",
        "cost_protocol": "The same evaluation trajectories are measured in observation counts or a spend proxy. The proxy uses each arm's quoted input-token price per million tokens as a fixed weight per observation. Token counts and realized dollar spend are unmeasured. The run budget is 10% of entries.",
        "budget_normalization": "Unit cost divides by all matrix entries. Price-weighted spend divides by the sum of price weights across every arm-example pair. Individual traces retain their actual endpoints.",
        "checkpoint_protocol": "A checkpoint uses the last completed phase at or before that spend. Before the first phase it is missing. Beyond the last phase the recommendation is held and counted separately. CSV blanks mean the estimate or its standard error is unavailable.",
        "plot_coverage": "Within each panel, plotted curves begin at the first grid point where every displayed feature has a recommendation from every seed of every included subject. Checkpoint tables retain available-case estimates and report their coverage.",
        "averaging": "Average available seeds within a subject, then weight subjects equally. Aggregate standard errors are computed across subject means. A paired delta is feature regret minus one-hot regret on the seeds available for both, then averaged equally across subjects.",
        "size_thresholds": {"small_max_examples": 150, "medium_max_examples": 400},
        "features_in_archive_order": FEATURES,
        "seeds": SEEDS,
        "checkpoint_percentages": CHECKPOINTS.tolist(),
        "plot_grid_spacing_percent": 0.025,
        "subject_count": len(subjects),
        "trace_count": len(subjects) * len(SEEDS) * len(FEATURES),
        "all_subjects_at_10_percent": [
            row for row in checkpoints if row["group"] == "All subjects" and row["budget_percent"] == 10
        ],
        "subjects": {
            subject.name: {
                "size": subject.size, "examples": subject.example_count,
                "exhaustive_observations": subject.exhaustive_observations,
                "exhaustive_price_weight": subject.exhaustive_price,
                "raw_archive_sha256": subject.archive_sha256,
            }
            for subject in subjects
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison, overview, {len(subjects)} subject figures and checkpoint tables to {args.out_dir}")


if __name__ == "__main__":
    main()
