#!/usr/bin/env python3
"""
Bootstrap CI utility for a single metric vector.
Usage: python scripts/bootstrap_ci.py --input metrics.csv --col accuracy --n 1000 --seed 42
Outputs median and 95% CI.
"""
import argparse
import numpy as np
import pandas as pd

def bootstrap_ci(arr, n=1000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    boots = []
    n_obs = len(arr)
    for _ in range(n):
        samp = rng.choice(arr, size=n_obs, replace=True)
        boots.append(np.mean(samp))
    lo = np.percentile(boots, 100 * (alpha/2))
    hi = np.percentile(boots, 100 * (1-alpha/2))
    median = np.median(boots)
    return median, lo, hi

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--col", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if args.col not in df.columns:
        raise SystemExit(f"Column {args.col} not found in {args.input}")
    arr = df[args.col].to_numpy()
    median, lo, hi = bootstrap_ci(arr, n=args.n, seed=args.seed)
    print(f"{args.col} median={median:.6f} 95%CI=[{lo:.6f}, {hi:.6f}]")
