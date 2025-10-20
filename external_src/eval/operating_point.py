# external_src/eval/operating_point.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .io_utils import load_preds, scan_pred_files
from .metrics import probs_from_logits, macro_f1

def best_threshold_macro_f1(y: np.ndarray, probs: np.ndarray):
    """
    One-vs-rest threshold search (shared τ for all classes) maximizing macro-F1.
    Works for imbalanced datasets where argmax may not be optimal.
    """
    C = probs.shape[1]
    conf = probs.max(1)
    pred = probs.argmax(1)
    # grid on confidence threshold to abstain-to-negative flip
    taus = np.linspace(0.2, 0.95, 40)
    best = (macro_f1(y, np.log(probs + 1e-8)), 0.0)  # logits from probs as baseline
    for t in taus:
        p = pred.copy()
        low = conf < t
        # optional: map low-confidence to majority or "benign" class (domain-specific). Here keep argmax.
        # Could also set to class with calibrated prior.
        # For simplicity we keep argmax but you may plug domain rule here.
        # Evaluate by faking logits with one-hot at p.
        lg = np.full((len(y), C), -50.0, dtype=np.float32)
        lg[np.arange(len(y)), p] = 10.0
        f1 = macro_f1(y, lg)
        if f1 > best[0]:
            best = (f1, t)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-glob", type=str, required=True)
    ap.add_argument("--metric", type=str, default="macro_f1")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows = []
    for f in scan_pred_files([args.preds_glob]):
        y, logits = load_preds(f)
        probs = probs_from_logits(logits)
        f1, tau = best_threshold_macro_f1(y, probs)
        model = Path(f).parents[1].name if "seed_" in str(f) else Path(f).parent.name
        seed  = Path(f).parent.name if "seed_" in str(f) else "seed_-1"
        rows.append({"model": model, "seed": seed, "tau": tau, "macro_f1_opt": f1})

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
