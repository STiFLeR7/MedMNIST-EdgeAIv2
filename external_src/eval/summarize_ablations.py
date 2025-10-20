# external_src/eval/summarize_ablations.py
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

from .io_utils import load_preds, scan_pred_files
from .metrics import macro_f1

def parse_cfg_from_path(p: Path):
    # Expect path like .../arch=xxx_alpha=0.5_T=4_lambda=0.0005/seed_0/preds.parquet
    s = str(p)
    def find(k, default=None, pat=None):
        if pat is None:
            pat = rf"{k}=([0-9\.e-]+)"
        m = re.search(pat, s)
        return float(m.group(1)) if m else default
    return {
        "alpha": find("alpha", None),
        "T":     find("T", None, pat=r"T=([0-9]+)"),
        "lambda":find("lambda", None),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="models/students")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--summary", type=str, required=True)
    args = ap.parse_args()

    files = list(Path(args.root).glob("**/seed_*/test_preds.parquet"))
    rows = []
    for f in files:
        y, logits = load_preds(f)
        mf1 = macro_f1(y, logits)
        cfg = parse_cfg_from_path(f)
        rows.append({
            "model": f.parents[2].name,
            "seed": f.parent.name,
            "alpha": cfg["alpha"], "T": cfg["T"], "at_lambda": cfg["lambda"],
            "macro_f1": mf1
        })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # summarize best per model
    best = df.groupby(["model","alpha","T","at_lambda"], as_index=False)["macro_f1"].mean()
    idx = best.groupby("model")["macro_f1"].idxmax()
    top = best.loc[idx].reset_index(drop=True)
    top.to_csv(args.summary, index=False)
    print(f"Wrote {args.out} and {args.summary}")

if __name__ == "__main__":
    main()
