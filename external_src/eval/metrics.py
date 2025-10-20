# external_src/eval/metrics.py
from __future__ import annotations
import numpy as np

def _onehot(y: np.ndarray, C: int) -> np.ndarray:
    z = np.zeros((y.shape[0], C), dtype=np.float32)
    z[np.arange(y.shape[0]), y] = 1.0
    return z

def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits, dtype=np.float64)
    p = (exps / exps.sum(axis=1, keepdims=True)).astype(np.float32)
    return p

def accuracy(y_true: np.ndarray, logits: np.ndarray) -> float:
    pred = logits.argmax(1)
    return float((pred == y_true).mean())

def macro_f1(y_true: np.ndarray, logits: np.ndarray) -> float:
    pred = logits.argmax(1)
    C = logits.shape[1]
    f1s = []
    for c in range(C):
        tp = ((pred == c) & (y_true == c)).sum()
        fp = ((pred == c) & (y_true != c)).sum()
        fn = ((pred != c) & (y_true == c)).sum()
        if tp == 0 and (fp + fn) == 0:
            f1s.append(1.0)
        else:
            prec = tp / (tp + fp + 1e-12)
            rec  = tp / (tp + fn + 1e-12)
            if prec + rec == 0:
                f1s.append(0.0)
            else:
                f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1s))

def nll(logits: np.ndarray, y_true: np.ndarray) -> float:
    # negative log likelihood (cross-entropy)
    logits = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(logits).sum(axis=1))
    return float((logsumexp - logits[np.arange(logits.shape[0]), y_true]).mean())

def brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    C = probs.shape[1]
    target = _onehot(y_true, C)
    return float(((probs - target) ** 2).sum(axis=1).mean())

def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15, adaptive: bool = False):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc  = (pred == y_true).astype(np.float32)
    if adaptive:
        # equal-mass binning
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1] + (1e-12 if i + 1 == n_bins else 0)
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            bin_stats.append((float((lo+hi)/2), 0.0, 0.0, 0))
            continue
        conf_mean = float(conf[mask].mean())
        acc_mean  = float(acc[mask].mean())
        gap = abs(conf_mean - acc_mean)
        w = mask.mean()
        ece_val += w * gap
        bin_stats.append((conf_mean, acc_mean, gap, int(mask.sum())))
    return float(ece_val), bin_stats
