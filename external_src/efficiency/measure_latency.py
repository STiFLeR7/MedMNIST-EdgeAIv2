# external_src/efficiency/measure_latency.py
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def median_p50_p90(model, dummy, iters=100, warmup=20, device="cuda"):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    arr = np.asarray(ts) * 1000.0
    return float(np.median(arr)), float(np.percentile(arr, 50)), float(np.percentile(arr, 90))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True, help="glob to best.pt/ts under seed_*")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows = []
    for ck in sorted(Path().glob(args.ckpts)):
        model = torch.jit.load(str(ck)) if ck.suffix in (".ts", ".pt") else torch.load(str(ck), map_location="cpu")
        model.eval()
        device = args.device
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA not available"
            model.to("cuda")
        dummy = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=("cuda" if device=="cuda" else "cpu"))
        p50, p50_same, p90 = median_p50_p90(model, dummy, iters=args.repeats, warmup=args.warmup, device=device)
        rows.append({
            "model": ck.parents[1].name,
            "seed": ck.parent.name,
            "device": device,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "latency_median_ms": p50,
            "latency_p50_ms": p50_same,
            "latency_p90_ms": p90,
            "ckpt_path": str(ck),
        })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
