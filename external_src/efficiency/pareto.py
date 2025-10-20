# external_src/efficiency/pareto.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pick_first(df: pd.DataFrame, cols: list[str]) -> str:
    for c in cols:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns found: {cols}")


def load_core(core_csv: Path) -> pd.DataFrame:
    """
    core CSV from aggregate_runs.py (students):
      columns like: model, accuracy_mean/_lo/_hi, macro_f1_mean/_lo/_hi
    We choose metric column in this priority:
      macro_f1_mean > accuracy_mean > acc_mean
    """
    df = pd.read_csv(core_csv)
    # normalize model col if needed
    if "model" not in df.columns:
        raise KeyError("core table must contain a 'model' column")
    metric_col = None
    for cand in ["macro_f1_mean", "accuracy_mean", "acc_mean"]:
        if cand in df.columns:
            metric_col = cand; break
    if metric_col is None:
        # as a last resort, try any *_mean column that isn't CI bounds
        mean_cols = [c for c in df.columns if c.endswith("_mean")]
        if not mean_cols:
            raise KeyError("No usable *_mean metric column found in core table.")
        metric_col = mean_cols[0]
    out = df[["model", metric_col]].rename(columns={metric_col: "score"})
    return out


def load_latency(lat_csv: Path) -> pd.DataFrame:
    """
    latency CSV from measure_latency.py (updated):
      lat_ms_mean, lat_ms_std, lat_ms_p50, lat_ms_p90, lat_ms_p99
    Older scripts may have 'latency_median_ms' or 'median_ms'.
    We aggregate to per-model median latency across seeds/ckpts.
    """
    df = pd.read_csv(lat_csv)
    if "model" not in df.columns:
        # try to infer model from ckpt path if present
        if "ckpt" in df.columns:
            def infer_model(p):
                try:
                    return Path(p).parents[1].name
                except Exception:
                    return Path(p).stem
            df["model"] = df["ckpt"].map(infer_model)
        else:
            raise KeyError("latency table must contain 'model' or 'ckpt' to infer model")
    lat_col = None
    for cand in ["lat_ms_p50", "latency_median_ms", "median_ms", "lat_ms_mean", "latency_ms"]:
        if cand in df.columns:
            lat_col = cand; break
    if lat_col is None:
        raise KeyError("No usable latency column found. Expected one of lat_ms_p50/latency_median_ms/lat_ms_mean.")
    # per-model median of the chosen latency metric
    agg = df.groupby("model", as_index=False)[lat_col].median()
    agg = agg.rename(columns={lat_col: "latency_ms"})
    return agg


def load_memory(mem_csv: Path) -> pd.DataFrame:
    """
    memory CSV from measure_memory.py (updated):
      params_bytes, params_mib, peak_cuda_bytes, peak_cuda_mib, model, seed
    We aggregate per-model:
      params_mib -> median (identical across seeds), peak_cuda_mib -> median
    """
    df = pd.read_csv(mem_csv)
    if "model" not in df.columns:
        if "ckpt" in df.columns:
            def infer_model(p):
                try:
                    return Path(p).parents[1].name
                except Exception:
                    return Path(p).stem
            df["model"] = df["ckpt"].map(infer_model)
        else:
            raise KeyError("memory table must contain 'model' or 'ckpt' to infer model")

    # derive MiB columns if missing
    if "params_mib" not in df.columns and "params_bytes" in df.columns:
        df["params_mib"] = df["params_bytes"].astype(float) / (1024.0**2)
    if "peak_cuda_mib" not in df.columns and "peak_cuda_bytes" in df.columns:
        df["peak_cuda_mib"] = df["peak_cuda_bytes"].astype(float) / (1024.0**2)

    for req in ["params_mib", "peak_cuda_mib"]:
        if req not in df.columns:
            df[req] = np.nan

    agg = df.groupby("model", as_index=False)[["params_mib", "peak_cuda_mib"]].median()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=str, required=True, help="tables/core/ham10000_students_ci.csv")
    ap.add_argument("--lat", type=str, required=True, help="tables/efficiency/ham10000_latency_gpu.csv")
    ap.add_argument("--mem", type=str, required=True, help="tables/efficiency/ham10000_memory.csv")
    ap.add_argument("--out-table", type=str, required=True)
    ap.add_argument("--out-fig", type=str, required=True)
    ap.add_argument("--label", type=str, default="macro-F1", help="y-axis label")
    args = ap.parse_args()

    core = load_core(Path(args.core))
    lat  = load_latency(Path(args.lat))
    mem  = load_memory(Path(args.mem))

    # merge
    df = core.merge(lat, on="model", how="inner").merge(mem, on="model", how="left")

    # Pareto table
    out_table = Path(args.out_table); out_table.parent.mkdir(parents=True, exist_ok=True)
    df_sorted = df.sort_values(["latency_ms", "score"], ascending=[True, False])
    df_sorted.to_csv(out_table, index=False)

    # Figure: Acc (score) vs Latency, bubble size = params_mib, color by model (matplotlib default)
    out_fig = Path(args.out_fig); out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sizes = (df["params_mib"].fillna(df["params_mib"].median() if not df["params_mib"].isna().all() else 10.0))
    # scale bubble sizes to something visible
    s_scaled = 20.0 + 3.0 * sizes.values
    plt.scatter(df["latency_ms"], df["score"], s=s_scaled)
    # light annotations
    for _, r in df.iterrows():
        plt.annotate(r["model"], (r["latency_ms"], r["score"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    plt.xlabel("Latency (ms, median across seeds)")
    plt.ylabel(args.label)
    plt.title("HAM10000 — Accuracy vs Latency (bubble = params MiB)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()

    print(f"Wrote {out_table} and {out_fig}")


if __name__ == "__main__":
    main()
