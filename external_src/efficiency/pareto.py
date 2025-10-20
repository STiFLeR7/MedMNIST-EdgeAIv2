# external_src/efficiency/pareto.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=str, required=True, help="tables/core/ham10000_students_ci.csv")
    ap.add_argument("--lat", type=str, required=True, help="tables/efficiency/ham10000_latency_gpu.csv")
    ap.add_argument("--mem", type=str, required=True, help="tables/efficiency/ham10000_memory.csv")
    ap.add_argument("--out-table", type=str, required=True)
    ap.add_argument("--out-fig", type=str, required=True)
    args = ap.parse_args()

    core = pd.read_csv(args.core)   # columns: model, accuracy_mean, macro_f1_mean, ...
    lat  = pd.read_csv(args.lat)    # columns: model, seed, latency_median_ms, device=cuda
    mem  = pd.read_csv(args.mem)    # columns: model, artifact_size_bytes, peak_gpu_mem_bytes

    # aggregate latency across seeds
    latg = lat.groupby("model", as_index=False)["latency_median_ms"].median()
    memg = mem.groupby("model", as_index=False)[["artifact_size_bytes","peak_gpu_mem_bytes"]].median()

    df = core.merge(latg, on="model", how="left").merge(memg, on="model", how="left")
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_table, index=False)

    plt.figure()
    x = df["latency_median_ms"]
    y = df["macro_f1_mean"] if "macro_f1_mean" in df.columns else df["accuracy_mean"]
    for _, r in df.iterrows():
        plt.scatter(r["latency_median_ms"], r.get("macro_f1_mean", r["accuracy_mean"]))
        plt.text(r["latency_median_ms"], r.get("macro_f1_mean", r["accuracy_mean"]), r["model"], fontsize=8)
    plt.xlabel("Latency median (ms, batch=1, RTX 3050)")
    plt.ylabel("Macro-F1 (mean)")
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out_table} and {args.out_fig}")

if __name__ == "__main__":
    main()
