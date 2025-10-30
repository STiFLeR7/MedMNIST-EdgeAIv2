#!/usr/bin/env python3
"""
ISIC Eval Pack (Phase-3)

Modes
-----
--mode ham : HAM→ISIC (label-compatible)
    - Requires --label-map (HAM label_map.json with {"class_name": int, ...} or {"int_str": int})
    - Optional --isic-to-ham-map mapping from ISIC class names to HAM class names or HAM indices
    - Outputs: metrics.json, calibration_temperature.json, reliability_* PNGs
    - If mapping cannot be resolved for some classes, they are skipped (logged).
    - If only melanoma is resolvable, falls back to melanoma-vs-rest aggregation (binary metrics).

--mode ood : OCT→ISIC (OOD)
    - Computes MSP/Entropy/MaxLogit AUROC/AUPR/FPR95 if --id-root is given (ID set).
      ID examples: data/OCT2017/test (must be a folder with class subdirs for ImageFolder).
    - Dumps histograms and summary stats. If --id-root is missing, AUROC/AUPR/FPR95 are omitted.
    - Output: ood_metrics.json, confidence_hist_*.png

Common
------
- Temperature scaling on validation (if available) then report raw+cal on val/test.
- Optional robustness corruptions over ISIC Test.
- Efficiency: CPU/GPU latency sweeps + peak CUDA memory and parameter size.

Usage (PowerShell examples)
---------------------------
# HAM→ISIC (with explicit name mapping)
python external_src/eval/eval_pack_isic.py `
  --mode ham `
  --dataset-root ./data/ISIC `
  --arch resnet50 `
  --ckpt ./models/teachers/isic_resnet50_v2/ckpt-best.pth `
  --outdir ./reports/phase3_isic/teacher_ham50 `
  --label-map ./models/teachers/runs_ham10000_resnet50/label_map.json `
  --isic-to-ham-map ./external_data/mappings/isic_to_ham.json `
  --bins 15 `
  --device cuda `
  --do-robust `
  --levels "gauss:0.1,0.2;jpeg:90,70;contrast:0.8"

# OCT→ISIC OOD (ID = OCT2017/test)
python external_src/eval/eval_pack_isic.py `
  --mode ood `
  --dataset-root ./data/ISIC `
  --id-root ./data/OCT2017/test `
  --arch resnet50 `
  --ckpt ./models/teachers/oct2017_resnet50/ckpt-best.pth `
  --outdir ./reports/phase3_isic/teacher_oct50_ood `
  --bins 30 `
  --device cuda
"""
import argparse, json, csv, io, os, time, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------- dataset utils -----------------------
def _disc(root: Path, name: str) -> Optional[Path]:
    # Accept lower, TitleCase (Train/Val/Test), and common synonyms
    name_l = name.lower()
    cand_names = {
        "train": {"train", "training"},
        "val": {"val", "valid", "validation"},
        "test": {"test", "testing"},
    }[name_l]
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in cand_names:
            return child
    tit = root / name.capitalize()
    return tit if tit.exists() else None

def make_loader_dir(dir_path: Path, img_size=224, bs=32, nw=4):
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(str(dir_path), transform=tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
                    persistent_workers=(nw > 0))
    return ds, dl

def discover_classes_from_dir(dir_path: Path) -> List[str]:
    ds = datasets.ImageFolder(str(dir_path))
    return list(ds.classes)


# ----------------------- model utils -----------------------
def make_model(arch: str, num_classes: int) -> nn.Module:
    a = arch.lower()
    if a == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif a == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif a == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif a in ("efficientnet_b0","efficientnet"):
        m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m

def extract_model_state(ckpt_obj) -> Dict[str, torch.Tensor]:
    sd = ckpt_obj
    if isinstance(sd, dict):
        for k in ["model_state", "state_dict", "model"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict.")
    out = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out

def infer_num_classes_from_sd(arch: str, sd: Dict[str, torch.Tensor]) -> Optional[int]:
    a = arch.lower()
    head_keys = []
    if a.startswith("resnet"):
        head_keys = ["fc.weight", "fc.bias"]
    elif a == "mobilenet_v2":
        head_keys = ["classifier.1.weight", "classifier.1.bias", "classifier.2.weight", "classifier.2.bias"]
    elif a in ("efficientnet_b0","efficientnet"):
        head_keys = ["classifier.1.weight", "classifier.1.bias"]
    for k in head_keys:
        if k in sd and sd[k] is not None and hasattr(sd[k], "shape"):
            return int(sd[k].shape[0])
    return None


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
    logits = torch.cat(logits_all, 0) if logits_all else torch.empty(0, device="cpu")
    labels = torch.cat(labels_all, 0) if labels_all else torch.empty(0, dtype=torch.long, device="cpu")
    return {"loss": ce_sum / max(1, seen), "acc": correct / max(1, seen)}, logits, labels

def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    if probs.numel() == 0: return float("nan")
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
    if probs.numel() == 0: return float("nan")
    eps = 1e-12
    picked = probs[torch.arange(labels.size(0)), labels]
    return float((-torch.log(picked.clamp_min(eps))).mean().item())

def brier(probs: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    if probs.numel() == 0: return float("nan")
    oh = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
    return float(((probs - oh) ** 2).sum(1).mean().item())

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 1.0
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
    if probs.numel() == 0: return
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
    if xs:
        plt.plot(xs, ys, "o-")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight"); plt.close()


# ----------------------- OOD metrics -----------------------
def msp_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, 1).max(1).values

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, 1)
    eps = 1e-12
    return (- (p * (p.clamp_min(eps).log())).sum(1))

def maxlogit_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.max(1).values

def _roc_pr_fpr95(scores_id: np.ndarray, scores_ood: np.ndarray, larger_is_id: bool = True):
    """
    Returns AUROC, AUPR (OOD as positive), and FPR@95%TPR on distinguishing ID vs OOD.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    # Convert to OOD scores: larger values mean OOD for PR consistency
    if larger_is_id:
        s_id  = -scores_id
        s_ood = -scores_ood
    else:
        s_id  = scores_id
        s_ood = scores_ood
    y = np.concatenate([np.zeros_like(s_id), np.ones_like(s_ood)])  # 1 = OOD
    s = np.concatenate([s_id, s_ood])

    auroc = float(roc_auc_score(y, s))
    aupr  = float(average_precision_score(y, s))

    # FPR@95%TPR with ID=0 class considered negative; TPR is for OOD-positive
    fpr, tpr, th = roc_curve(y, s)
    # We want TPR close to 0.95
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = float(fpr[idx])
    return auroc, aupr, fpr95


# ----------------------- corruptions -----------------------
def parse_levels(spec: str):
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
def eval_corruptions(model, test_dir: Path, img_size: int, bs: int, device, spec: str, n_bins=15):
    base = datasets.ImageFolder(str(test_dir))
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


# ----------------------- mapping (HAM→ISIC) -----------------------
def normalize_name(s: str) -> str:
    return s.strip().lower().replace("_"," ").replace("-"," ")

def build_isic_to_ham_map(isic_classes: List[str], ham_label_map: Dict, map_path: Optional[Path]):
    """
    Returns:
      - isic2ham_idx: dict {isic_idx -> ham_idx}
      - used, skipped: lists for logging
      - special_binary: if only melanoma was mappable -> ("melanoma_vs_rest", ham_mel_idx)
    ham_label_map can be {"mel":0,...} or {"0":"mel",...} — we'll invert if necessary.
    """
    # normalize HAM map to name->idx
    if all(isinstance(k, str) and isinstance(v, int) for k,v in ham_label_map.items()):
        ham_name2idx = {normalize_name(k): int(v) for k,v in ham_label_map.items()}
    elif all(isinstance(k, str) and k.isdigit() for k in ham_label_map.keys()):
        # {"0":"mel",...} unlikely, but guard
        inv = {normalize_name(v): int(k) for k,v in ham_label_map.items()}
        ham_name2idx = inv
    else:
        # or {"mel":0,"nv":1...} typical
        try:
            ham_name2idx = {normalize_name(k): int(v) for k,v in ham_label_map.items()}
        except Exception:
            raise RuntimeError("Unrecognized HAM label_map format.")

    explicit_map = {}
    if map_path and map_path.exists():
        raw = load_json(map_path)
        # allow {"isic_name":"ham_name_or_idx", ...}
        for k,v in raw.items():
            kn = normalize_name(str(k))
            if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
                explicit_map[kn] = int(v)
            else:
                vn = normalize_name(str(v))
                if vn in ham_name2idx:
                    explicit_map[kn] = int(ham_name2idx[vn])

    isic2ham_idx = {}
    used, skipped = [], []
    mel_ham_idx = None
    for i, ic in enumerate(isic_classes):
        nic = normalize_name(ic)
        if nic in explicit_map:
            isic2ham_idx[i] = explicit_map[nic]; used.append((ic, explicit_map[nic])); continue
        # heuristic: direct name match (e.g., "melanoma" -> "mel")
        if nic in ham_name2idx:
            isic2ham_idx[i] = int(ham_name2idx[nic]); used.append((ic, ham_name2idx[nic])); continue
        # melanoma special case
        if "melanoma" in nic:
            # find ham key that contains "mel"
            for hn, hid in ham_name2idx.items():
                if hn.startswith("mel"):
                    isic2ham_idx[i] = int(hid); mel_ham_idx = int(hid); used.append((ic, hid)); break
            if i in isic2ham_idx: continue
        skipped.append(ic)

    special_binary = None
    if len(isic2ham_idx) == 1 and mel_ham_idx is not None:
        special_binary = ("melanoma_vs_rest", mel_ham_idx)

    return isic2ham_idx, used, skipped, special_binary


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["ham","ood"])

    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=4)

    # HAM mapping inputs
    ap.add_argument("--label-map", type=str, default=None, help="HAM label_map.json (name->idx)")
    ap.add_argument("--isic-to-ham-map", type=str, default=None, help="Optional ISIC->HAM JSON map")

    # OOD ID root (required for AUROC/AUPR/FPR95)
    ap.add_argument("--id-root", type=str, default=None, help="In-distribution root (e.g., OCT2017/test) for OOD curves")

    # Robustness & efficiency
    ap.add_argument("--do-robust", action="store_true")
    ap.add_argument("--levels", type=str, default="gauss:0.1,0.2,0.3;jpeg:90,70,50;contrast:0.8,0.6")

    ap.add_argument("--lat-gpu-repeats", type=int, default=100)
    ap.add_argument("--lat-gpu-warmup", type=int, default=20)
    ap.add_argument("--lat-gpu-batches", type=str, default="1,2,4,8")
    ap.add_argument("--lat-cpu-repeats", type=int, default=200)
    ap.add_argument("--lat-cpu-warmup", type=int, default=30)
    ap.add_argument("--lat-cpu-batches", type=str, default="1,2,4")

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    root = Path(args.dataset_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve ISIC dirs
    train_dir = _disc(root, "train") or (root/"Train")
    val_dir   = _disc(root, "val")
    test_dir  = _disc(root, "test") or (root/"Test")
    if args.mode == "ham":
        if not train_dir or not train_dir.exists():
            raise RuntimeError(f"[ham] expected Train under {root}")
        # ISIC class discovery from Train (canonical)
        isic_classes = discover_classes_from_dir(train_dir)
    else:
        # OOD mode needs ISIC Test for OOD samples
        if not test_dir or not test_dir.exists():
            raise RuntimeError(f"[ood] expected Test under {root}")

    # ---- Load checkpoint & decide num classes
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = extract_model_state(ck)
    n_classes = infer_num_classes_from_sd(args.arch, sd)
    if n_classes is None:
        # fallback: discover from train (ham) or assume from ID-root if given (ood)
        if args.mode == "ham":
            if args.label_map:
                ham_lm = load_json(Path(args.label_map))
                # normalize to indices length
                if isinstance(next(iter(ham_lm.values())), int):
                    n_classes = int(max(ham_lm.values())) + 1
                else:
                    raise RuntimeError("Cannot infer n_classes from label_map.")
            else:
                raise RuntimeError("Provide --label-map for ham mode when ckpt head size is ambiguous.")
        else:
            if args.id_root:
                id_classes = discover_classes_from_dir(Path(args.id_root))
                n_classes = len(id_classes)
            else:
                raise RuntimeError("Provide --id-root or a ckpt with a resolvable head to infer n_classes in ood mode.")

    model = make_model(args.arch, n_classes)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        head_only = all(k.startswith("fc.") or k.startswith("classifier.") for k in missing)
        if not head_only:
            print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    model = model.to(device)

    # ============ MODE: HAM (HAM→ISIC) ============
    if args.mode == "ham":
        if not args.label_map:
            raise RuntimeError("--label-map is required for ham mode")
        ham_label_map = load_json(Path(args.label_map))
        isic2ham, used, skipped, special_binary = build_isic_to_ham_map(
            isic_classes, ham_label_map, (Path(args.isic_to_ham_map) if args.isic_to_ham_map else None)
        )
        mapping_report = {
            "isic_classes": isic_classes,
            "used": [(c, int(idx)) for c, idx in used],
            "skipped": skipped,
            "special_binary": special_binary
        }
        dump_json(mapping_report, outdir / "mapping_report.json")

        # Build mapped datasets (ImageFolder gives integer y as isic_idx)
        tf = transforms.Compose([
            transforms.Resize(int(args.img_size * 1.14)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        def remap_filter(ds: datasets.ImageFolder):
            # Keep only samples whose ISIC class is mapped
            keep = []
            for p, y_isic in ds.samples:
                if y_isic in isic2ham:
                    keep.append((p, isic2ham[y_isic]))
            class RemapDS(Dataset):
                def __init__(self, items, transform):
                    self.items = items; self.transform = transform
                def __len__(self): return len(self.items)
                def __getitem__(self, i):
                    p, y = self.items[i]
                    img = Image.open(p).convert("RGB")
                    x = self.transform(img) if self.transform else img
                    return x, y
            return RemapDS(keep, tf)

        # Val loader (if available) and Test loader
        val_loader = None
        if val_dir and val_dir.exists():
            ds_val = datasets.ImageFolder(str(val_dir))
            ds_val = remap_filter(ds_val)
            val_loader = DataLoader(ds_val, batch_size=args.eval_batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
        ds_test = datasets.ImageFolder(str(test_dir)) if test_dir and test_dir.exists() else None
        if ds_test is None:
            raise RuntimeError("[ham] ISIC Test not found.")
        ds_test = remap_filter(ds_test)
        test_loader = DataLoader(ds_test, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

        # If special_binary: aggregate to melanoma-vs-rest from HAM head
        def aggregate_binary_logits(logits: torch.Tensor, mel_idx: int) -> torch.Tensor:
            # logits: [B, C_ham] -> [B, 2] with [non-mel, mel]
            non_mel = torch.logsumexp(torch.cat([logits[:, :mel_idx], logits[:, mel_idx+1:]], dim=1), dim=1, keepdim=True)
            mel     = logits[:, mel_idx:mel_idx+1]
            return torch.cat([non_mel, mel], dim=1)

        # ---- Val/Test core ----
        metrics = {}
        if val_loader is not None:
            vm, v_logits, v_labels = eval_pass(model, val_loader, device)
            if special_binary:
                _, mel_idx = special_binary
                v_logits = aggregate_binary_logits(v_logits, mel_idx)
                v_labels = (v_labels == mel_idx).long()
            metrics["val"] = vm
        else:
            v_logits = torch.empty(0); v_labels = torch.empty(0, dtype=torch.long)

        tm, t_logits, t_labels = eval_pass(model, test_loader, device)
        if special_binary:
            _, mel_idx = special_binary
            t_logits = aggregate_binary_logits(t_logits, mel_idx)
            t_labels = (t_labels == mel_idx).long()
        metrics["test"] = tm

        # Temperature scaling from val if available, else skip calibration
        T = fit_temperature(v_logits.clone(), v_labels.clone()) if v_logits.numel() > 0 else 1.0
        val_probs_raw  = F.softmax(v_logits, 1) if v_logits.numel() > 0 else torch.empty(0)
        test_probs_raw = F.softmax(t_logits, 1) if t_logits.numel() > 0 else torch.empty(0)
        val_probs_cal  = F.softmax(v_logits / T, 1) if v_logits.numel() > 0 else torch.empty(0)
        test_probs_cal = F.softmax(t_logits / T, 1) if t_logits.numel() > 0 else torch.empty(0)

        cal = {
            "temperature": T,
            "val": {} if v_logits.numel()==0 else {
                "ece_raw":   ece(val_probs_raw,  v_labels, args.bins),
                "nll_raw":   nll_loss(val_probs_raw,  v_labels),
                "brier_raw": brier(val_probs_raw, v_labels, val_probs_raw.size(1)),
                "ece_cal":   ece(val_probs_cal,  v_labels, args.bins),
                "nll_cal":   nll_loss(val_probs_cal,  v_labels),
                "brier_cal": brier(val_probs_cal, v_labels, val_probs_cal.size(1)),
            },
            "test": {} if t_logits.numel()==0 else {
                "ece_raw":   ece(test_probs_raw, t_labels, args.bins),
                "nll_raw":   nll_loss(test_probs_raw, t_labels),
                "brier_raw": brier(test_probs_raw, t_labels, test_probs_raw.size(1)),
                "ece_cal":   ece(test_probs_cal, t_labels, args.bins),
                "nll_cal":   nll_loss(test_probs_cal, t_labels),
                "brier_cal": brier(test_probs_cal, t_labels, test_probs_cal.size(1)),
            }
        }

        dump_json(metrics, outdir / "metrics.json")
        dump_json(cal, outdir / "calibration_temperature.json")

        if v_logits.numel() > 0:
            save_reliability(val_probs_raw,  v_labels,  outdir / "reliability_val_raw.png",  args.bins, "Reliability (Val, Raw)")
            save_reliability(val_probs_cal,  v_labels,  outdir / "reliability_val_cal.png",  args.bins, "Reliability (Val, Cal)")
        if t_logits.numel() > 0:
            save_reliability(test_probs_raw, t_labels, outdir / "reliability_test_raw.png", args.bins, "Reliability (Test, Raw)")
            save_reliability(test_probs_cal, t_labels, outdir / "reliability_test_cal.png", args.bins, "Reliability (Test, Cal)")

        # Robustness (optional, over ISIC Test)
        if args.do_robust and test_dir and test_dir.exists():
            robust_rows = eval_corruptions(model, test_dir, args.img_size, args.eval_batch_size, device, args.levels, args.bins)
            dump_csv(robust_rows,
                     header=["corruption", "level", "acc", "nll", "ece", "brier", "n"],
                     path=outdir / "robustness_table.csv")

    # ============ MODE: OOD (OCT→ISIC) ============
    else:
        # Load OOD (ISIC Test) and optional ID loader
        _, ood_loader = make_loader_dir(test_dir, args.img_size, args.eval_batch_size, args.num_workers)
        id_loader = None
        if args.id_root:
            id_root = Path(args.id_root)
            if id_root.exists():
                # if the path points to .../test directory, load directly; else assume it has "test"
                id_dir = id_root if id_root.is_dir() and (id_root / "..").exists() else id_root
                if (id_dir / "train").exists() or (id_dir / "val").exists():
                    # prefer 'test' subset for ID
                    cand = id_dir / "test"
                    if not cand.exists():
                        raise RuntimeError(f"[ood] --id-root provided but no 'test' dir found under {id_dir}")
                    id_dir = cand
                _, id_loader = make_loader_dir(id_dir, args.img_size, args.eval_batch_size, args.num_workers)

        # Collect logits
        @torch.no_grad()
        def collect_logits(dl):
            model.eval()
            L = []
            for x, _ in tqdm(dl, ncols=120, desc="logits"):
                x = x.to(device, non_blocking=True)
                L.append(model(x).detach().float().cpu())
            return torch.cat(L, 0) if L else torch.empty(0)

        logits_ood = collect_logits(ood_loader)
        logits_id  = collect_logits(id_loader) if id_loader is not None else torch.empty(0)

        # Scores
        scores = {
            "msp": (msp_from_logits(logits_id).numpy() if logits_id.numel()>0 else None,
                    msp_from_logits(logits_ood).numpy()),
            "entropy": (entropy_from_logits(logits_id).numpy() if logits_id.numel()>0 else None,
                        entropy_from_logits(logits_ood).numpy()),
            "maxlogit": (maxlogit_from_logits(logits_id).numpy() if logits_id.numel()>0 else None,
                         maxlogit_from_logits(logits_ood).numpy()),
        }

        # Summaries and curves
        ood_metrics = {}
        for name, (sid, sood) in scores.items():
            entry = {}
            entry["ood_mean"] = float(np.mean(sood)) if sood.size > 0 else float("nan")
            entry["ood_std"]  = float(np.std(sood)) if sood.size > 0 else float("nan")
            if sid is not None and sid.size > 0:
                # AUROC/AUPR/FPR95
                larger_is_id = (name in ["msp", "maxlogit"])  # larger values indicate ID for these; entropy: larger=OOD
                auroc, aupr, fpr95 = _roc_pr_fpr95(sid, sood, larger_is_id=larger_is_id)
                entry.update({"auroc": auroc, "aupr": aupr, "fpr95": fpr95})
            else:
                entry.update({"auroc": None, "aupr": None, "fpr95": None})
            ood_metrics[name] = entry

            # hist plot
            plt.figure()
            if sid is not None and sid.size > 0:
                plt.hist(sid, bins=50, alpha=0.6, label="ID", density=True)
            plt.hist(sood, bins=50, alpha=0.6, label="ISIC (OOD)", density=True)
            plt.title(f"{name.upper()} distribution")
            plt.legend(); plt.xlabel(name); plt.ylabel("density")
            plt.savefig(outdir / f"conf_hist_{name}.png", bbox_inches="tight"); plt.close()

        dump_json(ood_metrics, outdir / "ood_metrics.json")

    # --------- Efficiency (both modes) ---------
    eff_gpu_csv = outdir / "efficiency_latency_gpu.csv"
    eff_cpu_csv = outdir / "efficiency_latency_cpu.csv"
    mem_csv     = outdir / "efficiency_memory.csv"

    mem_params_b, mem_params_mib, mem_peak_cuda_mib = measure_memory(model, device, args.img_size)
    dump_csv(
        [[args.arch, mem_params_b, mem_params_mib, args.img_size, 1, mem_peak_cuda_mib]],
        header=["arch", "param_bytes", "param_mib", "img_size", "batch", "peak_cuda_mib"],
        path=mem_csv
    )

    # CPU sweep
    bs_cpu = [int(x) for x in args.lat_cpu_batches.split(",") if x.strip()]
    res_cpu = latency_sweep(model, torch.device("cpu"), args.img_size, args.lat_cpu_repeats, args.lat_cpu_warmup, bs_cpu, use_amp=False)
    dump_csv(
        [[args.arch, bs, dev, p50, p05, p95, vram] for (bs, dev, p50, p05, p95, vram) in res_cpu],
        header=["arch", "batch", "device", "lat_ms_p50", "lat_ms_p05", "lat_ms_p95", "peak_cuda_mib"],
        path=eff_cpu_csv
    )

    # GPU sweep
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
