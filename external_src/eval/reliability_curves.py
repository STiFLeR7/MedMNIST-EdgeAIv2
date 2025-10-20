# external_src/eval/reliability_curves.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io_utils import load_preds, scan_pred_files
from .metrics import probs_from_logits, ece

def plot_reliability(probs: np.ndarray, y: np.ndarray, bins: int, adaptive: bool, out_pdf: Path):
    ece_val, bin_stats = ece(probs, y, n_bins=bins, adaptive=adaptive)
    conf = [b[0] for b in bin_stats]
    acc  = [b[1] for b in bin_stats]

    plt.figure()
    plt.plot([0,1], [0,1], linestyle="--", linewidth=1)
    plt.scatter(conf, acc, s=24)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability (ECE={ece_val:.3f})")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-glob", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="directory to place PDFs")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--adaptive", action="store_true", default=False)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for f in scan_pred_files([args.preds_glob]):
        y, logits = load_preds(f)
        probs = probs_from_logits(logits)
        model = Path(f).parents[1].name if "seed_" in str(f) else Path(f).parent.name
        seed  = Path(f).parent.name if "seed_" in str(f) else "seed_-1"
        out_pdf = outdir / f"{model}_{seed}_reliability.pdf"
        plot_reliability(probs, y, bins=args.bins, adaptive=args.adaptive, out_pdf=out_pdf)
        print("Wrote", out_pdf)

if __name__ == "__main__":
    main()
