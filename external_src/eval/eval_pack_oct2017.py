#!/usr/bin/env python3
"""
OCT2017 Eval Pack (Phase-2)
- Val/Test metrics
- Temperature scaling (val-fitted), ECE/NLL/Brier (val/test, raw+cal)
- Reliability curves (PNG)
- Optional corruptions robustness (Gaussian/JPEG/Contrast)
- Efficiency: GPU/CPU latency (median + p05/p95) and CUDA peak memory

Outputs (under --outdir):
  metrics.json
  calibration_temperature.json
  reliability_val_raw.png / reliability_val_cal.png
  reliability_test_raw.png / reliability_test_cal.png
  robustness_table.csv  (if --do-robust)
  efficiency_latency_gpu.csv  (if CUDA)
  efficiency_latency_cpu.csv
  efficiency_memory.csv
"""
import argparse, json, csv, io, os, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ----------------------- io helpers -----------------------
def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def dump_csv(rows, header, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)


# ----------------------- loaders/models -----------------------
def make_loader(root: Path, split: str, img_size=224, bs=32, nw=4):
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(str(root / split), transform=tf)
    dl = DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0)
    )
    return ds, dl

def make_model(arch: str, num_classes: int) -> nn.Module:
    a = arch.lower()
    if a == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif a == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif a == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif a == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m


# ----------------------- metrics -----------------------
@torch.no_grad()
def eval_pass(model, loader, device):
    model.eval()
    ce_sum, correct, seen = 0.0, 0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    logits_all, labels_all = [], []
    for x, y in tqdm(loader, ncols=120, desc="eval"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        ce_sum += float(ce(logits, y).item())
        correct += int((logits.argmax(1) == y).sum().item())
        seen += y.size(0)
        logits_all.append(logits.detach().float().cpu())
        labels_all.append(y.detach().cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    return {"loss": ce_sum / max(1, seen), "acc": correct / max(1, seen)}, logits, labels

def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, preds = probs.max(1)
    bins = torch.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m, n = bins[i], bins[i + 1]
        mask = (conf >= m) & (conf < n)
        if mask.sum() == 0: 
            continue
        acc = (preds[mask] == labels[mask]).float().mean()
        conf_avg = conf[mask].mean()
        out += float(mask.float().mean() * torch.abs(acc - conf_avg))
    return float(out)

def nll_loss(probs: torch.Tensor, labels: torch.Tensor) -> float:
    eps = 1e-12
    picked = probs[torch.arange(labels.size(0)), labels]
    return float((-torch.log(picked.clamp_min(eps))).mean().item())

def brier(probs: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    oh = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
    return float(((probs - oh) ** 2).sum(1).mean().item())

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.5, max_iter=50)
    def closure():
        opt.zero_grad()
        p = F.softmax(logits / T.clamp_min(1e-2), 1)
        eps = 1e-12
        picked = p[torch.arange(labels.size(0)), labels].clamp_min(eps)
        loss = -torch.log(picked).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().item())

def save_reliability(probs: torch.Tensor, labels: torch.Tensor, out_png: Path, n_bins=15, title="Reliability"):
    conf, preds = probs.max(1)
    bins = torch.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m, n = bins[i], bins[i + 1]
        mask = (conf >= m) & (conf < n)
        if mask.sum() == 0:
            continue
        xs.append(float(conf[mask].mean().item()))
        ys.append(float((preds[mask] == labels[mask]).float().mean().item()))
    plt.figure()
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(xs, ys, "o-")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight"); plt.close()


# ----------------------- corruptions -----------------------
def parse_levels(spec: str):
    """
    "gauss:0.1,0.2,0.3;jpeg:90,70,50;contrast:0.8,0.6"
    """
    out = {"gauss": [], "jpeg": [], "contrast": []}
    if not spec: return out
    for block in spec.split(";"):
        if not block.strip(): continue
        k, vs = block.split(":")
        vs = [v.strip() for v in vs.split(",") if v.strip()]
        if k == "gauss": out["gauss"] = [float(v) for v in vs]
        if k == "jpeg": out["jpeg"] = [int(v) for v in vs]
        if k == "contrast": out["contrast"] = [float(v) for v in vs]
    return out

def _apply_gauss(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0.0, sigma * 255.0, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def _apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(float(factor))

@torch.no_grad()
def eval_corruptions(model, root: Path, img_size: int, bs: int, device, spec: str, n_bins=15):
    base = datasets.ImageFolder(str(root / "test"))
    resize_crop = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
    ])
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    levels = parse_levels(spec)
    rows = []

    for cname, cvals in [("gauss", levels["gauss"]), ("jpeg", levels["jpeg"]), ("contrast", levels["contrast"])]:
        for cval in cvals:
            ce = nn.CrossEntropyLoss(reduction="sum")
            seen, correct, ce_sum = 0, 0, 0.0
            probs_all, labels_all = [], []

            Xb, yb = [], []
            def flush():
                nonlocal seen, correct, ce_sum, probs_all, labels_all, Xb, yb
                if not Xb: return
                x = torch.stack(Xb, 0).to(device, non_blocking=True)
                y = torch.tensor(yb, dtype=torch.long, device=device)
                logits = model(x)
                ce_sum += float(ce(logits, y).item())
                correct += int((logits.argmax(1) == y).sum().item())
                seen += y.size(0)
                probs_all.append(F.softmax(logits.detach().float().cpu(), 1))
                labels_all.append(y.detach().cpu())
                Xb.clear(); yb.clear()

            for img_path, y in tqdm(base.samples, desc=f"robust-{cname}-{cval}", ncols=120):
                img = Image.open(img_path).convert("RGB")
                img = resize_crop(img)
                if cname == "gauss":    img = _apply_gauss(img, cval)
                elif cname == "jpeg":   img = _apply_jpeg(img, cval)
                elif cname == "contrast": img = _apply_contrast(img, cval)
                x = to_tensor_norm(img)
                Xb.append(x); yb.append(y)
                if len(Xb) == bs: flush()
            flush()

            if probs_all:
                probs = torch.cat(probs_all, 0)
                labels = torch.cat(labels_all, 0)
                acc = correct / max(1, seen)
                rows.append([cname, cval, acc,
                             nll_loss(probs, labels),
                             ece(probs, labels, n_bins),
                             brier(probs, labels, probs.size(1)),
                             seen])
    return rows


# ----------------------- device-safe efficiency -----------------------
def _first_param_device(model: nn.Module):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

@torch.no_grad()
def latency_sweep(model, device, img_size, repeats, warmup, batch_list, use_amp=True):
    """
    Moves model to `device` for the sweep, restores to original device afterwards.
    """
    model.eval()
    orig_device = _first_param_device(model)
    if orig_device != device:
        model.to(device)

    results = []
    try:
        for bs in batch_list:
            x = torch.randn(bs, 3, img_size, img_size, device=device)
            # warmup
            for _ in range(warmup):
                with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(repeats):
                if device.type == "cuda":
                    t0 = torch.cuda.Event(enable_timing=True)
                    t1 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        _ = model(x)
                    t1.record()
                    torch.cuda.synchronize()
                    times.append(t0.elapsed_time(t1))  # ms
                else:
                    t_s = time.perf_counter()
                    _ = model(x)
                    t_e = time.perf_counter()
                    times.append((t_e - t_s) * 1000.0)
            p50 = float(np.median(times))
            p05 = float(np.percentile(times, 5))
            p95 = float(np.percentile(times, 95))
            vram = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
            results.append([bs, device.type, p50, p05, p95, vram])
    finally:
        if _first_param_device(model) != orig_device:
            model.to(orig_device)

    return results

@torch.no_grad()
def measure_memory(model, device, img_size):
    """
    Reports parameter size and peak CUDA memory for a bs=1 forward.
    Moves model to `device` for the measurement and restores it.
    """
    params_bytes = sum(p.numel() for p in model.parameters()) * 4  # fp32
    orig_device = _first_param_device(model)
    vram_peak = 0.0
    try:
        if orig_device != device:
            model.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(1, 3, img_size, img_size, device=device)
            _ = model(x)
            torch.cuda.synchronize()
            vram_peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    finally:
        if _first_param_device(model) != orig_device:
            model.to(orig_device)
    return params_bytes, params_bytes / (1024 * 1024), vram_peak


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bins", type=int, default=15)

    # Robustness & efficiency
    ap.add_argument("--do-robust", action="store_true")
    ap.add_argument("--levels", type=str, default="gauss:0.1,0.2,0.3;jpeg:90,70,50;contrast:0.8,0.6")

    ap.add_argument("--lat-gpu-repeats", type=int, default=100)
    ap.add_argument("--lat-gpu-warmup", type=int, default=20)
    ap.add_argument("--lat-gpu-batches", type=str, default="1,2,4,8")

    ap.add_argument("--lat-cpu-repeats", type=int, default=200)
    ap.add_argument("--lat-cpu-warmup", type=int, default=30)
    ap.add_argument("--lat-cpu-batches", type=str, default="1,2,4")

    ap.add_argument("--num-workers", type=int, default=4)

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    root = Path(args.dataset_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Discover classes from train/
    n_classes = len(datasets.ImageFolder(str(root / "train")).classes)

    # Loaders
    _, val_loader  = make_loader(root, "val",  args.img_size, args.eval_batch_size, args.num_workers)
    _, test_loader = make_loader(root, "test", args.img_size, args.eval_batch_size, args.num_workers)

    # Model + checkpoint
    model = make_model(args.arch, n_classes)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state", ck)
    model.load_state_dict(sd, strict=True)
    model = model.to(device)

    # --------- Val/Test core ---------
    val_metrics, val_logits, val_labels = eval_pass(model, val_loader, device)
    test_metrics, test_logits, test_labels = eval_pass(model, test_loader, device)

    # Temperature from val
    T = fit_temperature(val_logits.clone(), val_labels.clone())
    val_probs_raw  = F.softmax(val_logits, 1)
    test_probs_raw = F.softmax(test_logits, 1)
    val_probs_cal  = F.softmax(val_logits / T, 1)
    test_probs_cal = F.softmax(test_logits / T, 1)

    cal = {
        "temperature": T,
        "val": {
            "ece_raw":   ece(val_probs_raw,  val_labels, args.bins),
            "nll_raw":   nll_loss(val_probs_raw,  val_labels),
            "brier_raw": brier(val_probs_raw, val_labels, n_classes),
            "ece_cal":   ece(val_probs_cal,  val_labels, args.bins),
            "nll_cal":   nll_loss(val_probs_cal,  val_labels),
            "brier_cal": brier(val_probs_cal, val_labels, n_classes),
        },
        "test": {
            "ece_raw":   ece(test_probs_raw, test_labels, args.bins),
            "nll_raw":   nll_loss(test_probs_raw, test_labels),
            "brier_raw": brier(test_probs_raw, test_labels, n_classes),
            "ece_cal":   ece(test_probs_cal, test_labels, args.bins),
            "nll_cal":   nll_loss(test_probs_cal, test_labels),
            "brier_cal": brier(test_probs_cal, test_labels, n_classes),
        }
    }

    dump_json({"val": val_metrics, "test": test_metrics}, outdir / "metrics.json")
    dump_json(cal, outdir / "calibration_temperature.json")
    save_reliability(val_probs_raw,  val_labels,  outdir / "reliability_val_raw.png",  args.bins, "Reliability (Val, Raw)")
    save_reliability(val_probs_cal,  val_labels,  outdir / "reliability_val_cal.png",  args.bins, "Reliability (Val, Cal)")
    save_reliability(test_probs_raw, test_labels, outdir / "reliability_test_raw.png", args.bins, "Reliability (Test, Raw)")
    save_reliability(test_probs_cal, test_labels, outdir / "reliability_test_cal.png", args.bins, "Reliability (Test, Cal)")

    # --------- Robustness (optional) ---------
    if args.do_robust:
        robust_rows = eval_corruptions(model, root, args.img_size, args.eval_batch_size, device, args.levels, args.bins)
        dump_csv(robust_rows,
                 header=["corruption", "level", "acc", "nll", "ece", "brier", "n"],
                 path=outdir / "robustness_table.csv")

    # --------- Efficiency ---------
    eff_gpu_csv = outdir / "efficiency_latency_gpu.csv"
    eff_cpu_csv = outdir / "efficiency_latency_cpu.csv"
    mem_csv     = outdir / "efficiency_memory.csv"

    mem_params_b, mem_params_mib, mem_peak_cuda_mib = measure_memory(model, device, args.img_size)
    dump_csv(
        [[args.arch, mem_params_b, mem_params_mib, args.img_size, 1, mem_peak_cuda_mib]],
        header=["arch", "param_bytes", "param_mib", "img_size", "batch", "peak_cuda_mib"],
        path=mem_csv
    )

    # CPU sweep (moves model to CPU safely)
    bs_cpu = [int(x) for x in args.lat_cpu_batches.split(",") if x.strip()]
    res_cpu = latency_sweep(model, torch.device("cpu"), args.img_size, args.lat_cpu_repeats, args.lat_cpu_warmup, bs_cpu, use_amp=False)
    dump_csv(
        [[args.arch, bs, dev, p50, p05, p95, vram] for (bs, dev, p50, p05, p95, vram) in res_cpu],
        header=["arch", "batch", "device", "lat_ms_p50", "lat_ms_p05", "lat_ms_p95", "peak_cuda_mib"],
        path=eff_cpu_csv
    )

    # GPU sweep (moves back to CUDA safely if available)
    if torch.cuda.is_available():
        bs_gpu = [int(x) for x in args.lat_gpu_batches.split(",") if x.strip()]
        res_gpu = latency_sweep(model, torch.device("cuda"), args.img_size, args.lat_gpu_repeats, args.lat_gpu_warmup, bs_gpu, use_amp=True)
        dump_csv(
            [[args.arch, bs, dev, p50, p05, p95, vram] for (bs, dev, p50, p05, p95, vram) in res_gpu],
            header=["arch", "batch", "device", "lat_ms_p50", "lat_ms_p05", "lat_ms_p95", "peak_cuda_mib"],
            path=eff_gpu_csv
        )

    print("Done. Wrote:", outdir)

if __name__ == "__main__":
    main()
