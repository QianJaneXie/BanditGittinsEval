#!/usr/bin/env python3
"""Run only FormatSpread-TS; reuse verified GSM8K GittinsEval-S LCB traces."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
import matplotlib.pyplot as plt
from formatspread_ts_policy import formatspread_prior
from compare_gittins_ts import load_paper_costs
from plot_gittins_ts_comparison import load_cases, previous_regret, budget_average_regret, mean_and_se, paired_stats


def simulate(matrix, costs, mode, seed, fraction=.1):
    """Incremental equivalent of native policy: same Beta draws and unseen-cell RNG."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if not ((matrix == 0) | (matrix == 1)).all():
        raise ValueError('FormatSpread-TS requires binary scores (0 or 1)')
    torch.manual_seed(seed)
    k, n = matrix.shape
    counts = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    seen = np.zeros((k, n), dtype=bool)
    means = matrix.astype(np.float64).mean(1)
    best = means.max()
    full = float(k*n if mode == 'unit' else n*costs.sum())
    budget = round(fraction*full) if mode == 'unit' else fraction*full
    alpha, beta = formatspread_prior(.2)
    x, regret, arms, rec_means, pulls, sizes = [], [], [], [], [], []
    spent, evaluated = 0., 0
    original_cost = []
    while True:
        capacity = np.minimum(8, n-counts)
        if mode == 'unit':
            capacity = np.minimum(capacity, int(budget-evaluated))
        else:
            capacity = np.minimum(capacity, np.floor(max(0., budget-spent)/costs+1e-10).astype(int))
        eligible = capacity > 0
        if not eligible.any():
            break
        draws = torch.distributions.Beta(torch.from_numpy(sums+alpha),
                                        torch.from_numpy(counts-sums+beta)).sample().numpy()
        arm = int(np.where(eligible, draws, -np.inf).argmax())
        unseen = np.flatnonzero(~seen[arm])
        size = int(capacity[arm])
        columns = unseen[torch.randperm(len(unseen)).numpy()[:size]]
        seen[arm, columns] = True
        counts[arm] += size
        sums[arm] += matrix[arm, columns].astype(np.float64).sum()
        evaluated += size
        spent += size*costs[arm]
        # Native empirical_incumbent ranks float32 means.
        empirical = np.divide(sums, counts, out=np.full(k, -np.inf), where=counts > 0).astype(np.float32)
        rec = int(empirical.argmax())
        x.append(evaluated if mode == 'unit' else spent)
        original_cost.append(spent)
        regret.append(best-means[rec])
        arms.append(rec)
        rec_means.append(empirical[rec])
        pulls.append(arm)
        sizes.append(size)
    return dict(x=np.asarray(x)/full, regret=np.asarray(regret),
                recommended_arm=np.asarray(arms), recommended_mean=np.asarray(rec_means),
                pulled_arm=np.asarray(pulls), n_new_samples=np.asarray(sizes),
                x_original_cost=np.asarray(original_cost), initial_regret=best-means[0],
                full_budget=full, budget=budget, prior_alpha=alpha, prior_beta=beta,
                recommendation='empirical_mean', rng='torch_global_seeded')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/gsm8k_gittins_ts_batch_lcb100')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/gsm8k_formatspread_gittins_s')
    args = parser.parse_args()
    cases = [c for c in load_cases(args.source) if c['prior'] == 'S']
    manifest = json.loads((args.source/'manifest.json').read_text())
    matrices, matrix_hashes = {}, {}
    for m in range(1, 6):
        path = ROOT/f'data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{m}.npy'
        matrix_hashes[str(m)] = digest(path)
        if matrix_hashes[str(m)] != manifest['matrix_sha256'][str(m)]:
            raise ValueError('Source matrix hash mismatch')
        matrices[m] = np.load(path)
    costs = load_paper_costs()
    if hashlib.sha256(costs.astype("<f8").tobytes()).hexdigest() != manifest["ordered_cost_vector_sha256"]:
        raise ValueError("Source cost vector hash mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.out/'runs'
    runs.mkdir(exist_ok=True)
    source_hashes = {c['path']: digest(Path(c['path'])) for c in cases}
    rows, summary = [], {}
    curves = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, mode in zip(axes, ('unit', 'cost')):
        selected = sorted([c for c in cases if c['mode'] == mode], key=lambda c:(c['matrix_seed'],c['run_seed']))
        all_curves = {'GittinsEval-S (LCB)': [], 'FormatSpread-TS': []}
        metrics = {name: [] for name in all_curves}
        grid = np.linspace(0, .1, 351)
        for i, case in enumerate(selected):
            m, seed = case['matrix_seed'], case['run_seed']
            trace = simulate(matrices[m], costs, mode, seed)
            np.savez_compressed(runs/f'matrix{m}_seed{seed}_{mode}.npz', **trace)
            g = case['methods']['gittins']
            for name, x, y, initial in (
                ('GittinsEval-S (LCB)', g['x'], g['y'], g['initial_regret']),
                ('FormatSpread-TS', trace['x'], trace['regret'], trace['initial_regret'])):
                vals = [previous_regret(x, y, t, initial) for t in grid]
                all_curves[name].append(vals)
                record = dict(method=name, mode=mode, matrix_seed=m, run_seed=seed,
                              auc_0_10=budget_average_regret(x,y,.1,initial))
                for checkpoint in (.01,.02,.05,.1):
                    record[f'regret_at_{checkpoint:g}'] = previous_regret(x,y,checkpoint,initial)
                rows.append(record)
                metrics[name].append(record)
            if (i+1) % 20 == 0:
                print(f'{mode}: {i+1}/100 paired trials', flush=True)
        for name, color in zip(all_curves, ('C1','C4')):
            values = np.asarray(all_curves[name])
            mean, se = mean_and_se(values)
            ax.plot(grid*100, mean, label=name, color=color)
            ax.fill_between(grid*100, mean-se, mean+se, color=color, alpha=.18)
            curves[mode+'_'+('gittins' if name.startswith('Gittins') else 'formatspread')] = values
        panel = {}
        for field in ('regret_at_0.01','regret_at_0.02','regret_at_0.05','regret_at_0.1','auc_0_10'):
            g = np.array([r[field] for r in metrics['GittinsEval-S (LCB)']])
            ts = np.array([r[field] for r in metrics['FormatSpread-TS']])
            panel[field] = dict(gittins_mean=float(g.mean()), formatspread_mean=float(ts.mean()),
                               difference=paired_stats(ts-g, np.repeat(np.arange(1,6),20)))
        summary[mode] = panel
        ax.set(title='Unit cost' if mode=='unit' else 'Heterogeneous cost',
               xlabel='Evaluation budget (% of exhaustive cost)', ylabel='Simple regret', xlim=(0,10))
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(args.out/'comparison.png', dpi=180)
    fig.savefig(args.out/'comparison.pdf')
    np.savez_compressed(args.out/'comparison_curves.npz', grid_fraction=grid, **curves)
    with (args.out/'run_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    # Prove that source archives were not mutated; never call Gittins simulation.
    assert all(digest(Path(p)) == h for p,h in source_hashes.items())
    provenance = dict(source_directory=str(args.source.resolve()), source_sha256=source_hashes,
                      matrix_sha256=matrix_hashes, gittins_rerun=False,
                      n_trials_per_mode=100, batch_size=8, fraction=.1,
                      gittins='GittinsEval-S, finite-population LCB, prior N(.2,.01)',
                      formatspread='Beta(1.25,5), empirical-mean recommendation, fixed data-specific x=.2',
                      budget='FormatSpread uses affordable partial final batches; Gittins checkpoints exclude over-budget feedback',
                      figure='previous completed recommendation on 351-point grid, mean +/- one SE',
                      uncertainty='Five matrices x 20 seeds; paired two-way bootstrap is descriptive',
                      script_sha256=digest(Path(__file__)))
    (args.out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Wrote {args.out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
