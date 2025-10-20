#!/usr/bin/env python3
r"""
Generic evaluator + profiler for student checkpoints.

Usage examples (PowerShell):
# ResNet18
python .\external_src\students\evaluate_and_profile_generic.py `
  --arch resnet18 `
  --checkpoint .\models\students\distilled_resnet18_ham10000\ckpt-best.pth `
  --dataset HAM10000 `
  --data-root .\data `
  --save-dir .\models\students\distilled_resnet18_ham10000\eval `
  --device cuda:0

# MobileNetV2
python .\external_src\students\evaluate_and_profile_generic.py `
  --arch mobilenet_v2 `
  --checkpoint .\models\students\distilled_mobilenetv2_ham10000\ckpt-best.pth `
  --dataset HAM10000 `
  --data-root .\data `
  --save-dir .\models\students\distilled_mobilenetv2_ham10000\eval `
  --device cuda:0

# EfficientNet-B0
python .\external_src\students\evaluate_and_profile_generic.py `
  --arch efficientnet_b0 `
  --checkpoint .\models\students\distilled_efficientnetb0_ham10000\ckpt-best.pth `
  --dataset HAM10000 `
  --data-root .\data `
  --save-dir .\models\students\distilled_efficientnetb0_ham10000\eval `
  --device cuda:0
"""
import argparse, json, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchvision import models

from external_src.teachers.train_teacher import get_data_loaders

# optional FLOPs
try:
    from ptflops import get_model_complexity_info
    HAVE_PTFLOPS = True
except Exception:
    HAVE_PTFLOPS = False

# -------------------------
# Model factory
# -------------------------
def build_model(arch: str, num_classes: int):
    a = arch.lower()
    if a == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    if a in ("mobilenet_v2","mobilenetv2"):
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = torch.nn.Linear(m.last_channel, num_classes)
        return m
    if a in ("efficientnet_b0","efficientnetb0"):
        m = models.efficientnet_b0(weights=None)
        in_feats = m.classifier[1].in_features
        m.classifier[1] = torch.nn.Linear(in_feats, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")

# -------------------------
# Eval loop
# -------------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_trues, all_preds, all_probs = [], [], []
    rows = []
    idx_base = 0
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            bs = x.size(0)
            all_trues.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            for i in range(bs):
                rows.append({
                    "index": idx_base + i,
                    "true": int(y.cpu()[i].item()),
                    "pred": int(preds.cpu()[i].item()),
                    "probs": [float(p) for p in probs.cpu()[i].numpy().tolist()]
                })
            idx_base += bs
    report = classification_report(all_trues, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_trues, all_preds)
    return report, cm, rows

# -------------------------
# Profiling helpers
# -------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def compute_flops(model, input_res=(3,224,224)):
    if not HAVE_PTFLOPS:
        return None
    # ptflops wants CPU model
    with torch.no_grad():
        macs, params = get_model_complexity_info(model.cpu(), input_res, as_strings=False, print_per_layer_stat=False)
    # approximate flops = 2*MACs
    return float(macs) * 2.0

def export_traced_and_size(model, traced_path: Path):
    model_cpu = model.cpu().eval()
    example = torch.randn(1,3,224,224)
    traced = torch.jit.trace(model_cpu, example)
    traced_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(traced_path))
    return traced_path.stat().st_size

def measure_latency_gpu(model, device="cuda:0", runs=200, batch_size=1):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    inp = torch.randn((batch_size,3,224,224)).to(dev)
    # warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(inp)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.time()
            _ = model(inp)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1-t0)*1000.0)
    peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else -1
    return {"avg_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times,50)), "p90_ms": float(np.percentile(times,90)), "peak_gpu_mem_bytes": int(peak)}

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, type=str, choices=["resnet18","mobilenet_v2","efficientnet_b0"])
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--save-dir", required=True, type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--runs", type=int, default=150)
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(save_dir) / "eval"; eval_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloader (reuses get_data_loaders)
    train_loader, val_loader, num_classes, class_weights, label_map = get_data_loaders(
        args.dataset, Path(args.data_root), batch_size=args.batch_size, num_workers=2, input_size=224
    )

    # Build model and load ckpt
    model = build_model(args.arch, num_classes)
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck.get("model_state", ck)
    # strip module prefix
    new_sd = { (k[len("module."):] if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(new_sd, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate
    report, cm, rows = evaluate(model, val_loader, device)
    # Save metrics.json
    with open(eval_dir / "metrics.json", "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    np.savetxt(eval_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # Save preds.csv
    with open(eval_dir / "preds.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["index","true_label","pred_label","probs_json"])
        for r in rows:
            w.writerow([r["index"], r["true"], r["pred"], json.dumps(r["probs"])])

    # Plot confusion matrix
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(cm))
    if label_map:
        inv = {v:k for k,v in label_map.items()}
        labels = [inv.get(i,str(i)) for i in tick_marks]
    else:
        labels = [str(i) for i in tick_marks]
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    plt.colorbar()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Profile (params, flops, traced size)
    params = count_params(model)
    flops = compute_flops(model)  # may be None
    traced_path = Path(save_dir) / "export" / "traced.pt"
    try:
        traced_size = export_traced_and_size(model, traced_path)
    except Exception as e:
        print("Trace export failed:", e)
        traced_size = -1

    # Latency and peak GPU mem (on device)
    prof = measure_latency_gpu(model, device=args.device, runs=args.runs, batch_size=1)

    # Save profiling summary
    summary = {
        "arch": args.arch,
        "checkpoint": args.checkpoint,
        "params": int(params),
        "flops": float(flops) if flops is not None else None,
        "traced_size_bytes": int(traced_size),
        "latency_avg_ms": prof["avg_ms"],
        "latency_p50_ms": prof["p50_ms"],
        "latency_p90_ms": prof["p90_ms"],
        "peak_gpu_mem_bytes": prof["peak_gpu_mem_bytes"],
        "metrics": report
    }
    with open(Path(save_dir) / "profile_summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print("Saved eval+profile to:", eval_dir)
    print("Summary saved to:", Path(save_dir) / "profile_summary.json")

if __name__ == "__main__":
    main()
