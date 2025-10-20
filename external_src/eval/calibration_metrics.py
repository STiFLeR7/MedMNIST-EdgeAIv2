# external_src/eval/calibration_metrics.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .io_utils import load_preds, scan_pred_files
from .metrics import probs_from_logits, ece, nll, brier

def fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 1000, lr: float = 0.01):
    T = 1.0
    for _ in range(max_iter):
        # gradient of NLL w.r.t T for temperature scaling
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        # d/dT NLL = (sum_i p_i z_i - z_y) / T
        zy = z[np.arange(z.shape[0]), y_true]
        grad = ((p * z).sum(axis=1) - zy).mean() / T
        T_new = T - lr * grad
        if T_new <= 1e-3: T_new = 1e-3
        if abs(T_new - T) < 1e-6:
            break
        T = float(T_new)
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-glob", type=str, required=True, help="glob of parquet/npz/dir with preds")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--adaptive", action="store_true", default=False)
    ap.add_argument("--fit-temp", action="store_true", default=False)
    args = ap.parse_args()

    files = scan_pred_files([args.preds_glob])
    rows = []
    temps = []
    for f in files:
        y, logits = load_preds(f)
        model = Path(f).parents[1].name if "seed_" in str(f) else Path(f).parent.name
        seed  = Path(f).parent.name if "seed_" in str(f) else "seed_-1"

        if args.fit_temp:
            T = fit_temperature(logits, y)
            z = logits / T
            probs = probs_from_logits(z)
            temps.append({"model": model, "seed": seed, "T": T})
        else:
            probs = probs_from_logits(logits)

        acc = float((probs.argmax(1) == y).mean())
        nll_val = nll(logits if not args.fit_temp else logits / T, y)
        brier_val = brier(probs, y)
        ece_u, _ = ece(probs, y, n_bins=args.bins, adaptive=False)
        ece_a, _ = ece(probs, y, n_bins=args.bins, adaptive=True)

        rows.append({
            "model": model, "seed": seed,
            "accuracy": acc, "nll": nll_val, "brier": brier_val,
            "ece_uniform": ece_u, "ece_adaptive": ece_a
        })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if args.fit_temp:
        tdf = pd.DataFrame(temps)
        tpath = str(Path(args.out).with_name(Path(args.out).stem + "_tempscale_params.csv"))
        tdf.to_csv(tpath, index=False)
        print(f"Wrote {tpath}")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
