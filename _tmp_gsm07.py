import json, itertools
import numpy as np
from pathlib import Path

d = Path("outputs/figures/gsm8k")
configs = [
    ("pm0.2_pv0.01", 0.2, 0.01),
    ("pm0.2_pv0.04", 0.2, 0.04),
    ("pm0.3_pv0.04", 0.3, 0.04),
    ("pm0.5_pv0.01", 0.5, 0.01),
    ("pm0.5_pv0.02", 0.5, 0.02),
    ("pm0.5_pv0.04", 0.5, 0.04),
    ("pm0.7_pv0.04", 0.7, 0.04),
]
data = {}
print(f"{'tag':>14} {'pm':>4} {'pv':>5} {'stop':>6} {'n_iter':>7} {'step_s':>8} "
      f"{'finalR':>8} {'R@10%':>8} {'R@25%':>8} {'R@50%':>8} {'first0':>8} {'last>0':>8} {'meanR':>8} {'AUC':>9}")
for tag, pm, pv in configs:
    p = d / f"simple_regret_gsm8k_{tag}_traces.npz"
    if not p.exists():
        print(f"  (missing) {tag}")
        continue
    z = np.load(p)
    meta = json.loads((d / f"simple_regret_gsm8k_{tag}_traces.meta.json").read_text())
    x = z["gittins_x"].astype(np.int64); r = z["gittins_regret"].astype(np.float64)
    data[tag] = (x, r)
    budget = int(0.1*meta["n_cells"])
    step_s = meta["timing"]["gittins"]["summary"]["iter_step"]["total_s"]
    def R_at_frac(frac):
        cum = int(frac*budget)
        idx = np.searchsorted(x, cum, side="right") - 1
        return float(r[idx]) if idx >= 0 else float("nan")
    nz = np.where(r > 1e-9)[0]
    last_nz = int(x[nz[-1]]) if nz.size else 0
    zs = np.where(r <= 1e-9)[0]
    first0 = int(x[zs[0]]) if zs.size else -1
    print(f"{tag:>14} {pm:>4} {pv:>5} {int(z['gittins_stop_cum_eval']):>6} {len(x):>7} {step_s:>8.2f} "
          f"{float(r[-1]):>8.4f} {R_at_frac(0.1):>8.4f} {R_at_frac(0.25):>8.4f} {R_at_frac(0.5):>8.4f} "
          f"{first0:>8} {last_nz:>8} {float(r.mean()):>8.4f} {float(np.trapezoid(r,x)):>9.2f}")

print()
print("Pairwise bit-equal pairs:")
for a, b in itertools.combinations(list(data.keys()), 2):
    xa, ra = data[a]; xb, rb = data[b]
    if np.array_equal(xa, xb) and np.array_equal(ra, rb):
        print(f"  {a}  ==  {b}")
