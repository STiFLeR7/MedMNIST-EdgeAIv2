# external_src/eval/bootstrap_ci.py
from __future__ import annotations
import numpy as np

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Basic bootstrap CI for mean of a vector."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    idx = np.arange(n)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        boots[b] = values[samp].mean()
    mean = values.mean()
    lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
    return float(mean), float(lo), float(hi)
