import numpy as np
from pathlib import Path
d = Path("outputs/figures/gsm8k")
z5 = np.load(d / "simple_regret_gsm8k_pm0.5_pv0.04_traces.npz")
z7 = np.load(d / "simple_regret_gsm8k_pm0.7_pv0.04_traces.npz")
x5, r5 = z5["gittins_x"], z5["gittins_regret"]
x7, r7 = z7["gittins_x"], z7["gittins_regret"]
print(f"len x5={len(x5)} len x7={len(x7)}  x-equal={np.array_equal(x5,x7)}")
print(f"regret-equal={np.array_equal(r5,r7)}  max|diff|={np.max(np.abs(r5.astype(float)-r7.astype(float))):.5f}")
diff_idx = np.where(r5 != r7)[0]
print(f"differing steps (batch index): {len(diff_idx)}")
# print first 30 differing
print("step, cum_eval, r5, r7, diff")
for i in diff_idx[:60]:
    print(f"  step={int(i):>4}  cum={int(x5[i]):>6}  r5={float(r5[i]):.4f}  r7={float(r7[i]):.4f}  diff={float(r7[i]-r5[i]):+.4f}")
print("...")
print(f"last differing step idx={int(diff_idx[-1])} cum={int(x5[diff_idx[-1]])}")
