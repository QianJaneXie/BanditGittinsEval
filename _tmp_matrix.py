import numpy as np
for name in ["gsm8k", "piqa"]:
    M = np.load(f"data/matrices/{name}_1_samples_various_models_seed1.npy")
    mu = M.mean(axis=1)
    print(f"{name}: shape={M.shape}, overall_mean={mu.mean():.3f}, best={mu.max():.3f} (arm {int(np.argmax(mu))}), worst={mu.min():.3f}, median={float(np.median(mu)):.3f}")
    order = np.argsort(-mu)
    print(f"  top-5 arm indices: {order[:5].tolist()}")
    print(f"  top-5 arm means:   {[round(float(mu[i]),3) for i in order[:5]]}")
    # rank-position of best arm (0-indexed) = np.argmax(mu)
    print(f"  best-arm position in scan order: {int(np.argmax(mu))} (out of {M.shape[0]})")
