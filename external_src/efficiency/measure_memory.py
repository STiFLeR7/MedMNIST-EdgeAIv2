# external_src/efficiency/measure_memory.py
from __future__ import annotations
import argparse
from pathlib import Path
import os, psutil
import numpy as np
import pandas as pd
import torch

def peak_gpu_bytes(fn):
    if not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats()
    ret = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return int(peak)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows = []
    for ck in sorted(Path().glob(args.ckpts)):
        model = torch.jit.load(str(ck)) if ck.suffix in (".ts", ".pt") else torch.load(str(ck), map_location="cpu")
        model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=("cuda" if torch.cuda.is_available() else "cpu"))

        def run():
            with torch.no_grad():
                _ = model(x); _ = model(x)
        peak = peak_gpu_bytes(run) if torch.cuda.is_available() else None
        size_bytes = os.path.getsize(ck)
        rows.append({
            "model": ck.parents[1].name,
            "seed": ck.parent.name,
            "artifact_size_bytes": int(size_bytes),
            "peak_gpu_mem_bytes": (int(peak) if peak is not None else -1),
            "ckpt_path": str(ck),
        })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
