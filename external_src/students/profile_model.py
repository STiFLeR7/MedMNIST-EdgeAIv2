#!/usr/bin/env python3
"""
Profile a model checkpoint:
  - Params
  - MACs and FLOPs (if ptflops installed)
  - TorchScript size
  - Latency (avg, p50, p90)
  - Peak GPU memory
  - GPU utilization trace (via pynvml or nvidia-smi fallback)

Usage (PowerShell):
python -m external_src.students.profile_model `
  --arch resnet18 `
  --checkpoint .\models\students\distilled_resnet18_ham10000\ckpt-best.pth `
  --device cuda:0 `
  --runs 200 `
  --trace-out .\models\students\distilled_resnet18_ham10000\export\traced.pt
"""

import argparse, json, time, subprocess
from pathlib import Path
import torch, torch.nn.functional as F
from torchvision import models
import numpy as np

# ------------------------- Optional deps -------------------------
try:
    from ptflops import get_model_complexity_info
    HAVE_PTFLOPS = True
except Exception:
    HAVE_PTFLOPS = False

try:
    import pynvml
    HAVE_PYNVML = True
except Exception:
    HAVE_PYNVML = False


# ------------------------- Model builders -------------------------
def build_model(arch, num_classes):
    a = arch.lower()
    if a == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    if a in ("mobilenet_v2", "mobilenetv2"):
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = torch.nn.Linear(m.last_channel, num_classes)
        return m
    if a in ("efficientnet_b0", "efficientnetb0"):
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = torch.nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unknown architecture: {arch}")


def load_checkpoint(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model_state", ck)
    new_sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    return model


# ------------------------- Profiling helpers -------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_flops_and_macs(model, input_res=(3, 224, 224)):
    if not HAVE_PTFLOPS:
        return None, None
    macs, params = get_model_complexity_info(model.cpu(), input_res, as_strings=False, print_per_layer_stat=False)
    flops = float(macs) * 2.0
    return float(macs), flops


def export_traced_size(model, out_path):
    model_cpu = model.cpu().eval()
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model_cpu, example)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    return out_path.stat().st_size


def measure_latency_and_peak_mem(model, device, runs=200, batch_size=1, warmup=20):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    inp = torch.randn((batch_size, 3, 224, 224)).to(dev)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inp)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = model(inp)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            if elapsed_ms > 0:
                times.append(elapsed_ms)

    if not times:
        return {"avg_ms": -1, "p50_ms": -1, "p90_ms": -1, "times_ms": [], "peak_gpu_mem_bytes": -1}

    avg = float(np.mean(times))
    p50 = float(np.percentile(times, 50))
    p90 = float(np.percentile(times, 90))
    peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else -1
    return {
        "avg_ms": avg,
        "p50_ms": p50,
        "p90_ms": p90,
        "times_ms": times,
        "peak_gpu_mem_bytes": int(peak)
    }


def sample_gpu_util_fallback(duration_s=8, sample_interval=0.25):
    """Try pynvml first, fallback to nvidia-smi."""
    samples = []

    # Attempt NVML
    if HAVE_PYNVML:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            end = time.time() + duration_s
            while time.time() < end:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                samples.append({
                    "t": time.time(),
                    "gpu_util": int(util.gpu),
                    "mem_used": int(mem.used)
                })
                time.sleep(sample_interval)
            pynvml.nvmlShutdown()
            return samples
        except Exception:
            pass

    # Fallback to nvidia-smi subprocess
    try:
        end = time.time() + duration_s
        while time.time() < end:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"
            ], stderr=subprocess.DEVNULL)
            line = out.decode("utf8").strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            util = int(parts[1])
            mem = int(parts[2]) * 1024 * 1024  # MiB → bytes
            samples.append({"t": time.time(), "gpu_util": util, "mem_used": mem})
            time.sleep(sample_interval)
    except Exception:
        samples = []

    return samples


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-classes", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--trace-out", default=None)
    args = ap.parse_args()

    model = build_model(args.arch, args.num_classes)
    model = load_checkpoint(model, args.checkpoint)

    print(f"Profiling {args.arch} on {args.device} ...")

    info = {"arch": args.arch, "checkpoint": str(args.checkpoint)}

    # Params + FLOPs
    info["params"] = count_params(model)
    macs, flops = compute_flops_and_macs(model)
    info["macs"] = macs
    info["flops"] = flops

    # TorchScript export size
    if args.trace_out:
        info["traced_size_bytes"] = export_traced_size(model, Path(args.trace_out))
    else:
        info["traced_size_bytes"] = None

    # Latency + memory
    perf = measure_latency_and_peak_mem(model, args.device, runs=args.runs)
    info.update(perf)

    # GPU utilization trace (8s)
    samples = sample_gpu_util_fallback(duration_s=8, sample_interval=0.2)
    info["gpu_util_trace"] = samples
    info["gpu_util_trace_len"] = len(samples)

    # Summary print
    print(json.dumps({
        k: (round(v, 3) if isinstance(v, (float, int)) and "times" not in k else v)
        for k, v in info.items() if k not in ["times_ms", "gpu_util_trace"]
    }, indent=2))

    # Save JSON
    out = Path(args.checkpoint).parent / "profile_summary.json"
    with open(out, "w", encoding="utf8") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
