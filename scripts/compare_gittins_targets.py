#!/usr/bin/env python3
"""Compare archived latent and current full-test-set Gittins at one sampling seed.

Each trajectory replays finite mean and finite mean-minus-std recommendations.
The numerical evaluation cost is held fixed across acquisition targets. Native
post-pull stop times are recorded without truncating the fixed-budget runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("gittins_policy.py", "gittins_shrinking_posterior.py", "gittins_lookup.py",
                "gittins_index_computation.py", "q_estimation.py", "simple_regret_recommend.py")
PRIORS = {"default": (.5, .04), "dataset": (.2, .01)}
RULES = ("Full-test mean", "Full-test mean − std")


def worker(args):
    # A separate interpreter and source directory isolate the archived imports.
    sys.path.insert(0, str(args.worker_source.resolve()))
    import torch
    import gittins_policy as policy
    import gittins_shrinking_posterior as posterior
    from gittins_lookup import compute_roots_lookup_table
    from simple_regret_recommend import finite_population_posterior_moments, recommend_from_posterior

    torch.set_num_threads(1)
    truth = torch.tensor(np.load(args.matrix), dtype=torch.float32)
    true_means = truth.to(torch.float64).mean(dim=1)
    best_mean = float(true_means.max())
    n_arms, n_examples = truth.shape
    summaries = []
    for prior_type in args.prior_types:
        prior = PRIORS[prior_type]
        if args.worker_target == "latent":
            transitions = posterior.transition_stds_shrinking_gaussian_posterior(
                np.float32(prior[1]), np.float32(args.tau_sq_cell), n_examples)
            roots_array = compute_roots_lookup_table(
                transition_stds=transitions, costs_per_arm=np.float32(args.cost_scaling_factor),
                n_points=args.grid_points)
        else:
            roots_array = posterior.compute_finite_population_roots_lookup_table(
                prior_variance=prior[1], tau_sq_cell=args.tau_sq_cell, n_examples=n_examples,
                costs_per_arm=np.float32(args.cost_scaling_factor), n_points=args.grid_points)
        roots = torch.tensor(np.asarray(roots_array), dtype=torch.float32)
        if roots.shape != (1, n_examples + 1):
            raise ValueError(f"Expected one shared unit-cost lookup row, found {tuple(roots.shape)}")
        torch.manual_seed(args.seed)
        obs = torch.full_like(truth, float("nan"))
        cache = torch.full((n_arms,), float("inf"), dtype=torch.float32)
        counts = torch.zeros(n_arms, dtype=torch.int64)
        natural, aware = [None], [[None], [None]]
        previous_arm, evaluated = None, 0
        history = {key: [] for key in (
            "evaluations", "pulled_arm", "pulled_columns", "regret", "recommendation",
            "recommended_mean", "recommended_std", "recommended_count", "acquisition_means",
            "acquisition_scores", "natural_stop_condition", "recommendation_stop_margin")}
        shared = dict(prior_mean=prior[0], prior_variance=prior[1],
                      obs_noise_variance=args.tau_sq_cell / args.batch_size,
                      batch_size=args.batch_size, batch_observation_model=True,
                      cost_per_transition=1., cost_scaling_factor=args.cost_scaling_factor,
                      n_gittins_grid_points=args.grid_points, roots_lookup_table=roots,
                      cached_scores=cache)
        while evaluated < args.budget:
            batch = policy.gittins_index_exploration(
                obs, **shared, recompute_arms=None if previous_arm is None else [previous_arm],
                allow_early_stop=False)
            if batch is None:
                break
            row, col = batch
            obs[row, col] = truth[row, col]
            previous_arm = int(row[0])
            evaluated += len(row)
            counts[previous_arm] += len(row)
            for rule, penalty in enumerate((0., 1.)):
                acquisition_means, _ = policy.gittins_post_pull_update(
                    obs, **shared, recompute_arms=[previous_arm], sim_cum_eval=evaluated,
                    natural_stop_cum_eval_holder=natural,
                    recommendation_aware_stop_cum_eval_holder=aware[rule],
                    recommendation_std_penalty=penalty)
                if rule == 0:
                    mean_rule_scores = cache.clone()
                elif not torch.equal(cache, mean_rule_scores):
                    raise AssertionError("Changing the recommendation penalty changed acquisition scores")
            means, variances = finite_population_posterior_moments(
                obs, prior_mean=prior[0], prior_variance=prior[1], tau_sq_cell=args.tau_sq_cell)
            recommendations = [recommend_from_posterior(means, variances, std_penalty=penalty)[0]
                               for penalty in (0., 1.)]
            incomplete = counts < n_examples
            max_unfinished = float(cache[incomplete].max()) if bool(incomplete.any()) else float("-inf")
            history["evaluations"].append(evaluated)
            history["pulled_arm"].append(previous_arm)
            history["pulled_columns"].append(np.pad(col.numpy(), (0, args.batch_size - len(col)), constant_values=-1))
            history["regret"].append([best_mean - float(true_means[arm]) for arm in recommendations])
            history["recommendation"].append(recommendations)
            history["recommended_mean"].append([float(means[arm]) for arm in recommendations])
            history["recommended_std"].append([float(variances[arm].sqrt()) for arm in recommendations])
            history["recommended_count"].append([int(counts[arm]) for arm in recommendations])
            history["acquisition_means"].append(acquisition_means.numpy().copy())
            history["acquisition_scores"].append(cache.numpy().copy())
            history["natural_stop_condition"].append(bool(counts[int(torch.argmax(cache))] == n_examples))
            history["recommendation_stop_margin"].append([max_unfinished - float(means[arm]) for arm in recommendations])
        history = {key: np.asarray(value) for key, value in history.items()}
        history_path = args.out_dir / "raw" / f"{args.worker_target}_{prior_type}_seed{args.seed}.npz"
        index_target = "latent_mean" if args.worker_target == "latent" else "finite_population_mean"
        np.savez_compressed(history_path, **history, rules=np.asarray(RULES),
                            gittins_index_target=np.asarray(index_target))
        # Cross-check native holders against the saved post-pull diagnostic conditions.
        within = history["evaluations"] <= args.budget
        def first(condition):
            indices = np.flatnonzero(within & condition)
            return int(history["evaluations"][indices[0]]) if indices.size else None
        natural_first = first(history["natural_stop_condition"])
        aware_first = [first(history["recommendation_stop_margin"][:, i] < 0) for i in (0, 1)]
        clip = lambda value: value if value is not None and value <= args.budget else None
        assert natural_first == clip(natural[0])
        assert aware_first == [clip(holder[0]) for holder in aware]
        entry = dict(acquisition_target=args.worker_target, gittins_index_target=index_target, prior_type=prior_type,
                     prior_mean=prior[0], prior_variance=prior[1], seed=args.seed,
                     raw_batches=len(history["evaluations"]), raw_final_evaluations=evaluated,
                     natural_stop_evaluations=natural_first,
                     recommendation_aware_stop_evaluations=aware_first,
                     finite_scale=1 + args.tau_sq_cell / (n_examples * prior[1]),
                     roots_shape=list(roots.shape), history=str(history_path.relative_to(args.out_dir)))
        summaries.append(entry)
        print(json.dumps(entry), flush=True)
    (args.out_dir / f"{args.worker_target}_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


def summarize_and_plot(args, revision, source_hashes):
    sys.path.insert(0, str(ROOT))
    from scripts.analyze_recommendation_oscillation import per_seed_metrics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories, metrics, trajectory_audits = {}, [], []
    setup_rows = []
    for target in ("latent", "finite"):
        setups = json.loads((args.out_dir / f"{target}_summary.json").read_text())
        setup_rows.extend(setups)
        for setup in setups:
            prior = setup["prior_type"]
            with np.load(args.out_dir / setup["history"]) as data:
                raw = {key: data[key] for key in data.files}
            histories[target, prior] = raw
            for rule, label in enumerate(RULES):
                entry = dict(acquisition_target=target, prior_type=prior, rule=label,
                             **per_seed_metrics(raw, rule, int(raw["evaluations"][0]), args.budget),
                             natural_stop_evaluations=setup["natural_stop_evaluations"],
                             recommendation_aware_stop_evaluations=setup["recommendation_aware_stop_evaluations"][rule],
                             unobserved_recommendation_count=int(np.sum((raw["recommended_count"][:, rule] == 0)
                                                                        & (raw["evaluations"] <= args.budget))))
                metrics.append(entry)
    old_reference = ROOT / "outputs/recommendation_finite_lcb_gsm8k_seed1_run0/raw"
    reference_summary_path = old_reference.parent / "summary.json"
    reference_summary = json.loads(reference_summary_path.read_text()) if reference_summary_path.exists() else {}
    reference_config = reference_summary.get("config", {})
    matrix_sha256 = hashlib.sha256(args.matrix.read_bytes()).hexdigest()
    for prior in args.prior_types:
        old, new = histories["latent", prior], histories["finite", prior]
        checks = {}
        path = old_reference / f"{args.matrix.stem}_{prior}_seed{args.seed}.npz"
        reference_setup = next((setup for setup in reference_summary.get("setups", [])
                                if setup["prior_type"] == prior and setup["dataset"] == args.matrix.stem), {})
        matching_config = all((
            reference_config.get("batch_size") == args.batch_size,
            reference_config.get("tau_sq_cell") == args.tau_sq_cell,
            reference_config.get("cost_scaling_factor") == args.cost_scaling_factor,
            reference_config.get("grid_points") == args.grid_points,
            reference_config.get("seed_start") == args.seed,
            reference_config.get("seeds") == 1,
            reference_setup.get("evaluation_budget") == args.budget,
            reference_setup.get("sha256") == matrix_sha256,
            reference_setup.get("prior_mean") == PRIORS[prior][0],
            reference_setup.get("prior_variance") == PRIORS[prior][1],
        ))
        audit_skip_reason = None if matching_config and path.exists() else (
            "Saved reference configuration or matrix hash does not match this comparison."
            if not matching_config else "Matching reference raw history is unavailable.")
        if audit_skip_reason is None:
            with np.load(path) as reference:
                for key in ("evaluations", "pulled_arm", "recommendation", "regret", "recommended_count"):
                    expected = reference[key][:, [1, 3]] if reference[key].ndim == 2 else reference[key]
                    checks[key] = bool(np.array_equal(old[key], expected))
            if not all(checks.values()):
                raise AssertionError(f"Archived latent control fails prior-artifact reproduction: {checks}")
        common_batches = min(len(old["evaluations"]), len(new["evaluations"]))
        same = ((old["pulled_arm"][:common_batches] == new["pulled_arm"][:common_batches])
                & np.all(old["pulled_columns"][:common_batches] == new["pulled_columns"][:common_batches], axis=1))
        different = np.flatnonzero(~same)
        first = int(different[0]) if different.size else None
        trajectory_audits.append(dict(prior_type=prior, old_reference=str(path.relative_to(ROOT)),
                                       archived_control_reproduces_saved_arrays=checks,
                                       archived_control_audit_skipped_reason=audit_skip_reason,
                                       sampling_identical=bool(same.all() and len(old["evaluations"]) == len(new["evaluations"])),
                                       first_differing_batch_index=first,
                                       first_differing_old_evaluations=int(old["evaluations"][first]) if first is not None else None,
                                       first_differing_new_evaluations=int(new["evaluations"][first]) if first is not None else None))
    styles = [("latent", 0, "Old acquisition · finite mean", "#34445d", "--"),
              ("latent", 1, "Old acquisition · finite LCB", "#c06a35", "--"),
              ("finite", 0, "Unified finite acquisition · mean", "#008579", "-"),
              ("finite", 1, "Unified finite acquisition · LCB", "#8d62a8", "-")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    n_cells = int(np.load(args.matrix, mmap_mode="r").size)
    full_percent = args.budget / n_cells * 100
    for cumulative in (False, True):
        fig, axes = plt.subplots(len(args.prior_types), 2, figsize=(12.4, 3.4 * len(args.prior_types) + .6), squeeze=False)
        for row, prior in enumerate(args.prior_types):
            for column, limit in enumerate((full_percent, min(3., full_percent))):
                ax = axes[row, column]
                for target, rule, label, color, linestyle in styles:
                    raw = histories[target, prior]
                    keep = raw["evaluations"] <= args.budget
                    x = raw["evaluations"][keep] / n_cells * 100
                    arms = raw["recommendation"][keep, rule]
                    values = np.r_[0, np.cumsum(arms[1:] != arms[:-1])] if cumulative else raw["regret"][keep, rule]
                    if x[-1] < full_percent:
                        x, values = np.r_[x, full_percent], np.r_[values, values[-1]]
                    ax.step(x, values, where="post", label=label, color=color, linestyle=linestyle, linewidth=1.15)
                mu, var = PRIORS[prior]
                ax.set_title(f"{'General' if prior == 'default' else 'Data-specific'} prior N({mu:g}, {var:g}) · 0–{limit:g}%")
                ax.set_xlim(0, limit)
                ax.set_ylim(bottom=0)
                ax.set_xlabel("Evaluated matrix cells (%)")
                ax.set_ylabel("Cumulative recommendation switches" if cumulative else "Simple regret (accuracy gap)")
                ax.grid(alpha=.18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=2, frameon=False)
        fig.suptitle(f"GSM8K · matrix seed1 · sampling seed{args.seed} · B{args.batch_size} · same numerical cost {args.cost_scaling_factor:g}", fontsize=12)
        fig.text(.5, .012, "Separate old/new acquisition trajectories; paired recommendation rules within each trajectory; no smoothing or seed averaging.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .14, 1, .95))
        name = "recommendation_switches" if cumulative else "simple_regret"
        for extension in ("png", "pdf"):
            fig.savefig(args.out_dir / f"{name}.{extension}", dpi=190)
        plt.close(fig)
    payload = dict(config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   old_revision=revision, source_sha256=source_hashes,
                   comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   matrix_sha256=matrix_sha256,
                   cost_semantics="Both targets use the same numeric per-cell cost. Finite acquisition uses full-test-mean reward units; this is equivalent to latent acquisition at c/a followed by the common positive affine reward transform, with a=1+tau_cell^2/(N*v0). Costs are not multiplied by a.",
                   stopping="First native post-pull natural/rec-aware condition within the budget; diagnostic recording does not truncate sampling. LCB changes recommendation and rec-aware stopping only; its raw mean is used for the stop comparison.",
                   trajectory_audits=trajectory_audits, setups=setup_rows, metrics=metrics)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(trajectory_audits=trajectory_audits, metrics=metrics), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/unified_finite_gittins_gsm8k_seed1_run0")
    parser.add_argument("--old-revision", default="f4833e4")
    parser.add_argument("--prior-types", nargs="+", choices=tuple(PRIORS), default=list(PRIORS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=12200)
    parser.add_argument("--tau-sq-cell", type=float, default=.25)
    parser.add_argument("--cost-scaling-factor", type=float, default=1e-4)
    parser.add_argument("--grid-points", type=int, default=1025)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-target", choices=("latent", "finite"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.matrix, args.out_dir = args.matrix.resolve(), args.out_dir.resolve()
    if args.seed < 0 or args.batch_size < 1 or args.budget < 1 or args.grid_points < 2:
        parser.error("Require seed >= 0, batch-size and budget >= 1, and grid-points >= 2.")
    if any(not np.isfinite(value) or value <= 0 for value in (args.tau_sq_cell, args.cost_scaling_factor)):
        parser.error("Noise variance and cost scaling must be finite and positive.")
    matrix = np.load(args.matrix, mmap_mode="r")
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        parser.error("Matrix must be nonempty, finite, and two-dimensional.")
    if args.budget > matrix.size:
        parser.error("Budget cannot exceed the number of matrix cells.")
    if args.worker_source is not None:
        worker(args)
        return
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new out-dir to preserve existing artifacts.")
    (args.out_dir / "raw").mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "--verify", args.old_revision], cwd=ROOT, text=True).strip()
    hashes = {}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    for target in ("latent", "finite"):
        source = args.out_dir / "source_snapshot" / target
        source.mkdir(parents=True, exist_ok=True)
        hashes[target] = {}
        for name in SOURCE_FILES:
            destination = source / name
            if target == "latent":
                destination.write_bytes(subprocess.check_output(["git", "show", f"{revision}:src/{name}"], cwd=ROOT))
            else:
                shutil.copyfile(ROOT / "src" / name, destination)
            hashes[target][name] = hashlib.sha256(destination.read_bytes()).hexdigest()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-source", str(source),
                   "--worker-target", target, "--matrix", str(args.matrix), "--out-dir", str(args.out_dir),
                   "--prior-types", *args.prior_types, "--seed", str(args.seed), "--batch-size", str(args.batch_size),
                   "--budget", str(args.budget), "--tau-sq-cell", str(args.tau_sq_cell),
                   "--cost-scaling-factor", str(args.cost_scaling_factor), "--grid-points", str(args.grid_points)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    summarize_and_plot(args, revision, hashes)


if __name__ == "__main__":
    main()
