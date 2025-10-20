# external_src/eval/aggregate_runs.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .io_utils import load_preds
from .metrics import accuracy, macro_f1

TASK_TO_METRICS = {
    "derma": ["accuracy", "macro_f1"],
}

CAND_FILENAMES = (
    "test_preds.parquet",
    "preds.parquet",
    "preds.npz",
)

def _find_pred_files(root: Path) -> list[Path]:
    """Return a list of prediction files under root.

    Accept any of:
      - seed_*/{test_preds.parquet|preds.parquet|preds.npz}
      - eval/{...} (teacher often dumps here)
      - any recursive match to CAND_FILENAMES
    """
    files: list[Path] = []
    # seed_* convention
    for sd in sorted(root.glob("seed_*")):
        if sd.is_dir():
            for name in CAND_FILENAMES:
                p = sd / name
                if p.exists():
                    files.append(p)
    # eval/ fallback
    eval_dir = root / "eval"
    if eval_dir.is_dir():
        for name in CAND_FILENAMES:
            p = eval_dir / name
            if p.exists():
                files.append(p)
        # also tolerate any parquet here
        files += list(eval_dir.glob("*.parquet"))

    # recursive fallback (robust)
    if not files:
        for name in CAND_FILENAMES:
            files += list(root.rglob(name))
        # last resort: any parquet directly under root
        if not files:
            files += list(root.glob("*.parquet"))

    # unique, stable order
    uniq = []
    seen = set()
    for f in sorted(files):
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def _bootstrap_metric(y: np.ndarray, logits: np.ndarray, name: str, n_boot: int, alpha: float, seed: int = 0):
    """Bootstrap over example indices (pooled across seeds/files)."""
    def evaluate(y_, lg_):
        if name == "accuracy": return accuracy(y_, lg_)
        if name == "macro_f1": return macro_f1(y_, lg_)
        raise KeyError(name)

    n = y.shape[0]
    base = evaluate(y, logits)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(evaluate(y[idx], logits[idx]))
    lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
    return float(base), float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=str, required=True,
                    help="Comma-separated roots containing predictions (seed_*/ or eval/).")
    ap.add_argument("--task", type=str, default="derma", choices=list(TASK_TO_METRICS.keys()))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    roots = [Path(r.strip()) for r in args.roots.split(",") if r.strip()]
    rows = []

    for root in roots:
        pred_files = _find_pred_files(root)
        if not pred_files:
            print(f"[WARN] No prediction files under {root}. Expected seed_*/ or eval/ with {CAND_FILENAMES}.")
            continue

        Ys, LGs = [], []
        for pf in pred_files:
            y, lg = load_preds(pf)
            Ys.append(y); LGs.append(lg)
        Y = np.concatenate(Ys, axis=0)
        LG = np.concatenate(LGs, axis=0)

        row = {"model": root.name}
        for m in TASK_TO_METRICS[args.task]:
            mean, lo, hi = _bootstrap_metric(Y, LG, m, n_boot=args.n_boot, alpha=args.alpha)
            row[f"{m}_mean"] = mean
            row[f"{m}_lo"]   = lo
            row[f"{m}_hi"]   = hi
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("model")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
