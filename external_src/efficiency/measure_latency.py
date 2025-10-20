# external_src/efficiency/measure_latency.py
from __future__ import annotations
import argparse, glob, re, time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: F401


def _jit_first_load(ckpt: Path):
    # TorchScript-first; fallback to torch.load(weights_only=False) if it’s a full module
    try:
        m = torch.jit.load(str(ckpt), map_location="cpu")
        return m
    except Exception:
        obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if hasattr(obj, "eval") and callable(obj.eval):
            return obj
        raise RuntimeError(
            f"Unsupported checkpoint format for {ckpt}. "
            f"Export as TorchScript or save a full nn.Module."
        )


def _discover_ckpts(patterns: str) -> List[Path]:
    toks = [t.strip() for t in re.split(r"[;,]+", patterns) if t.strip()]
    hits: List[Path] = []
    for t in toks:
        hits += [Path(h) for h in glob.glob(t)]
    # stable order, de-dup
    uniq, seen = [], set()
    for p in sorted(hits):
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _warmup_and_benchmark(model: torch.nn.Module, device: str, batch: int, imgsz: int, warmup: int, repeats: int):
    dev = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model.eval().to(dev)

    x = torch.randn(batch, 3, imgsz, imgsz, device=dev)

    # some scripted nets need a first call to compile
    def _call():
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y

    if dev == "cuda":
        torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        _ = _call()
    if dev == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = _call()
        if dev == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    arr = np.array(times, dtype=np.float64)
    stats = {
        "lat_ms_mean": float(arr.mean()),
        "lat_ms_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "lat_ms_p50": float(np.percentile(arr, 50)),
        "lat_ms_p90": float(np.percentile(arr, 90)),
        "lat_ms_p99": float(np.percentile(arr, 99)),
        "throughput_fps": float((batch * 1000.0) / arr.mean()) if arr.mean() > 0 else float("nan"),
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True,
                    help="Checkpoint glob(s). Accepts absolute paths; separate multiple with ';' or ','")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    ckpts = _discover_ckpts(args.ckpts)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpts}")

    rows = []
    for ck in ckpts:
        model = _jit_first_load(ck)
        stats = _warmup_and_benchmark(model, args.device, args.batch, args.imgsz, args.warmup, args.repeats)

        # infer tags from path .../<model>/seed_*/best.pt
        try:
            model_name = ck.parents[1].name
            seed = ck.parent.name
        except Exception:
            model_name = ck.stem
            seed = "unknown"

        row = {
            "ckpt": str(ck),
            "model": model_name,
            "seed": seed,
            "device": args.device,
            "batch": args.batch,
            "imgsz": args.imgsz,
            **stats,
        }
        rows.append(row)

    df_new = pd.DataFrame(rows)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # append idempotently by (ckpt, device, batch, imgsz)
    if outp.exists():
        df_old = pd.read_csv(outp)
        key_cols = ["ckpt", "device", "batch", "imgsz"]
        merged = pd.concat([df_old, df_new], ignore_index=True)
        merged = merged.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")
        merged.to_csv(outp, index=False)
    else:
        df_new.to_csv(outp, index=False)

    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
