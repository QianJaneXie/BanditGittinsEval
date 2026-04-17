import json, itertools
import numpy as np
from pathlib import Path

DATASET = "piqa"
d = Path(f"outputs/figures/{DATASET}")
configs = [
    ("pm0.2_pv0.01", 0.2, 0.01),
    ("pm0.2_pv0.04", 0.2, 0.04),
    ("pm0.3_pv0.04", 0.3, 0.04),
    ("pm0.5_pv0.01", 0.5, 0.01),
    ("pm0.5_pv0.02", 0.5, 0.02),
    ("pm0.5_pv0.04", 0.5, 0.04),
]

meta0 = json.loads((d / f"simple_regret_{DATASET}_{configs[0][0]}_traces.meta.json").read_text())
n_cells = int(meta0["n_cells"])
batch = int(meta0["gittins_batch_size"])
budget = int(round(0.1 * n_cells))
print(f"n_cells={n_cells}  budget(10%)={budget}  batch={batch}  max_batches={budget // batch}")
print()

hdr = (
    f"{'tag':>14} {'pm':>4} {'pv':>5} {'stop':>6} {'n_iter':>7} {'step_s':>8} "
    f"{'finalR':>8} {'R@10%':>8} {'R@25%':>8} {'R@50%':>8} {'first0':>8} {'last>0':>8} "
    f"{'meanR':>8} {'AUC':>10}"
)
print(hdr)
data = {}
for tag, pm, pv in configs:
    z = np.load(d / f"simple_regret_{DATASET}_{tag}_traces.npz")
    meta = json.loads((d / f"simple_regret_{DATASET}_{tag}_traces.meta.json").read_text())
    x = z["gittins_x"].astype(np.int64)
    r = z["gittins_regret"].astype(np.float64)
    data[tag] = (x, r)
    stop = int(z["gittins_stop_cum_eval"])
    step_s = meta["timing"]["gittins"]["summary"]["iter_step"]["total_s"]
    final_r = float(r[-1]) if r.size else float("nan")

    def R_at_frac(frac):
        cum = int(frac * budget)
        idx = np.searchsorted(x, cum, side="right") - 1
        return float(r[idx]) if idx >= 0 else float("nan")

    nz = np.where(r > 1e-9)[0]
    last_nonzero = int(x[nz[-1]]) if nz.size else 0
    zero_idx = np.where(r <= 1e-9)[0]
    first_zero = int(x[zero_idx[0]]) if zero_idx.size else -1
    mean_r = float(r.mean())
    auc = float(np.trapezoid(r, x))
    print(
        f"{tag:>14} {pm:>4} {pv:>5} {stop:>6} {len(x):>7} {step_s:>8.2f} "
        f"{final_r:>8.4f} {R_at_frac(0.1):>8.4f} {R_at_frac(0.25):>8.4f} {R_at_frac(0.5):>8.4f} "
        f"{first_zero:>8} {last_nonzero:>8} {mean_r:>8.4f} {auc:>10.2f}"
    )

print()
print("Pairwise bit-equal pairs (x and regret arrays):")
for a, b in itertools.combinations([c[0] for c in configs], 2):
    xa, ra = data[a]
    xb, rb = data[b]
    if np.array_equal(xa, xb) and np.array_equal(ra, rb):
        print(f"  {a}  ==  {b}")

# Also show true row means (top few) of the piqa matrix for context
import os
mat_path = Path("data/matrices/piqa_1_samples_various_models_seed1.npy")
if mat_path.exists():
    M = np.load(mat_path)
    mu = M.mean(axis=1)
    order = np.argsort(-mu)
    print()
    print(f"piqa matrix shape={M.shape}, mean(all)={mu.mean():.4f}, best arm idx={order[0]} mean={mu[order[0]]:.4f}")
    print("Top-5 arm means: ", [f"{mu[i]:.4f}" for i in order[:5]])
    print("Bottom-5 arm means: ", [f"{mu[i]:.4f}" for i in order[-5:]])
    print(f"arm means min={mu.min():.4f} median={np.median(mu):.4f} max={mu.max():.4f}")
