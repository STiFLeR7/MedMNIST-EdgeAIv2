# external_src/efficiency/measure_memory.py
from __future__ import annotations
import argparse, glob, re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch


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
    uniq, seen = [], set()
    for p in sorted(hits):
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq


def _num_params_bytes(model: torch.nn.Module) -> int:
    # If scripted, parameters() still works; otherwise best effort
    nbytes = 0
    try:
        for p in model.parameters():
            if p is None:
                continue
            nbytes += p.numel() * p.element_size()
        for b in getattr(model, "buffers", lambda: [])():
            if b is None:
                continue
            nbytes += b.numel() * b.element_size()
    except Exception:
        nbytes = 0
    return int(nbytes)


def _peak_cuda_bytes(model: torch.nn.Module, imgsz: int = 224, batch: int = 1) -> int:
    if not torch.cuda.is_available():
        return 0
    dev = "cuda"
    model = model.eval().to(dev)
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, imgsz, imgsz, device=dev)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    # move back to CPU to free GPU memory for safety
    model.to("cpu"); torch.cuda.empty_cache()
    return int(peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True,
                    help="Checkpoint glob(s). Accepts absolute paths; separate multiple with ';' or ','")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    ckpts = _discover_ckpts(args.ckpts)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpts}")

    rows = []
    for ck in ckpts:
        model = _jit_first_load(ck)

        # tags
        try:
            model_name = ck.parents[1].name
            seed = ck.parent.name
        except Exception:
            model_name = ck.stem
            seed = "unknown"

        nparam_bytes = _num_params_bytes(model)
        peak_cuda = _peak_cuda_bytes(model, imgsz=args.imgsz, batch=args.batch)

        row = {
            "ckpt": str(ck),
            "model": model_name,
            "seed": seed,
            "params_bytes": nparam_bytes,
            "params_mib": float(nparam_bytes) / (1024**2) if nparam_bytes else 0.0,
            "peak_cuda_bytes": peak_cuda,
            "peak_cuda_mib": float(peak_cuda) / (1024**2) if peak_cuda else 0.0,
            "imgsz": args.imgsz,
            "batch": args.batch,
        }
        rows.append(row)

    df_new = pd.DataFrame(rows)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # append idempotently by ckpt
    if outp.exists():
        df_old = pd.read_csv(outp)
        merged = pd.concat([df_old, df_new], ignore_index=True)
        merged = merged.sort_values(["ckpt"]).drop_duplicates(subset=["ckpt"], keep="last")
        merged.to_csv(outp, index=False)
    else:
        df_new.to_csv(outp, index=False)

    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
