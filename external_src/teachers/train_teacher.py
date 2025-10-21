#!/usr/bin/env python3
"""
Train a teacher model (ResNet50) on HAM10000 or OCT2017.

Examples (PowerShell, repo root D:/MedMNIST-EdgeAIv2):

# HAM10000 (same as before)
python external_src/teachers/train_teacher.py ^
  --dataset HAM10000 ^
  --data-root ./data ^
  --epochs 30 ^
  --batch-size 16 ^
  --accum-steps 2 ^
  --lr 1e-4 ^
  --num-workers 4 ^
  --pretrained-backbone ./models/teachers/resnet50-0676ba61.pth ^
  --save-dir ./models/teachers/runs_ham10000_resnet50 ^
  --amp

# OCT2017 (Phase-2) — seeded manifests, test eval + metric dumps
python external_src/teachers/train_teacher.py ^
  --dataset OCT2017 ^
  --data-root ./data ^
  --epochs 60 ^
  --batch-size 16 ^
  --accum-steps 2 ^
  --lr 3e-4 ^
  --num-workers 4 ^
  --pretrained-backbone ./models/teachers/resnet50-0676ba61.pth ^
  --save-dir ./models/teachers/oct2017_resnet50 ^
  --split-manifest-dir ./external_data/splits/OCT2017 ^
  --index-parquet ./v2-rebuild/phase2_oct2017/data/processed/oct2017_index.parquet ^
  --seed 0 ^
  --eval-test ^
  --amp
"""
import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from PIL import Image
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# -------------------------
# Utilities
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_image_file(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"])

def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2)

# -------------------------
# Top-level RemapDataset (picklable for Windows workers)
# -------------------------
class RemapDataset(Dataset):
    def __init__(self, samples, transform, label_map):
        self.samples = samples
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        img = Image.open(p).convert("RGB")  # OCT may be grayscale; replicate to RGB
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[lbl]

# -------------------------
# HAM10000-specific metadata loader
# -------------------------
class HAM10000Dataset(Dataset):
    """
    Builds samples from HAM10000 metadata CSV and image-part folders.
    """
    def __init__(self, ds_root: Path, csv_path: Path, transform=None, img_col: str = None, label_col: str = None):
        import csv
        self.root = Path(ds_root)
        self.transform = transform
        self.samples = []
        part_dirs = [p for p in [self.root / "HAM10000_images_part_1", self.root / "HAM10000_images_part_2"] if p.exists()]
        if not part_dirs:
            candidates = [p for p in self.root.iterdir() if p.is_dir()]
            for c in candidates:
                if any(is_image_file(f.name) for f in c.iterdir()):
                    part_dirs.append(c)
        if not part_dirs:
            raise RuntimeError(f"No image part folders found under {self.root}.")
        images_map = {}
        for d in part_dirs:
            for f in d.iterdir():
                if f.is_file() and is_image_file(f.name):
                    images_map[f.name] = str(f.resolve())
                    images_map[f.stem] = str(f.resolve())
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not img_col:
                if 'image_id' in headers: img_col = 'image_id'
                elif 'image' in headers: img_col = 'image'
                else: img_col = headers[0]
            if not label_col:
                if 'dx' in headers: label_col = 'dx'
                elif 'label' in headers: label_col = 'label'
                elif 'diagnosis' in headers: label_col = 'diagnosis'
                else: label_col = headers[-1]
            for r in reader:
                raw_img = r.get(img_col, "").strip()
                raw_label = r.get(label_col, "").strip()
                if not raw_img:
                    continue
                candidates = []
                if is_image_file(raw_img):
                    candidates.append(raw_img)
                else:
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidates.append(raw_img + ext)
                chosen = None
                for c in candidates:
                    if c in images_map:
                        chosen = images_map[c]; break
                if not chosen and raw_img in images_map:
                    chosen = images_map[raw_img]
                if not chosen:
                    for k,v in images_map.items():
                        if raw_img in k:
                            chosen = v; break
                if not chosen:
                    continue
                self.samples.append((chosen, raw_label))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found from CSV {csv_path}; check paths and image parts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl

# -------------------------
# OCT2017 helpers (seeded manifests)
# -------------------------
def load_seed_manifest(manifest_dir: Path, seed: int) -> Optional[Dict[str, List[str]]]:
    """
    Expects files like seed_0.json with:
      { "seed": 0, "splits": { "train": [relpath,...], "val": [...], "test": [...] } }
    relpaths are relative to data/OCT2017 root (e.g., "train/CNV/xxx.jpeg").
    """
    if manifest_dir is None:
        return None
    manp = manifest_dir / f"seed_{seed}.json"
    if not manp.exists():
        return None
    data = json.loads(manp.read_text(encoding="utf8"))
    return data.get("splits", None)

def make_oct_dataset_from_manifest(root: Path, relpaths: List[str], transform) -> Dataset:
    """
    Build a dataset using relpaths' order. Label is inferred from folder name in relpath.
    Uses class_to_idx discovered from the train folder to ensure consistent mapping.
    """
    # Discover classes from train/ (canonical)
    if not (root/"train").exists():
        raise RuntimeError(f"OCT2017 expected {root}/train to exist.")
    base = datasets.ImageFolder(str(root/"train"))
    class_to_idx = base.class_to_idx  # e.g., {'CNV':0,'DME':1,'DRUSEN':2,'NORMAL':3}

    samples = []
    for rel in relpaths:
        p = (root / rel)
        if not p.exists() or not p.is_file() or not is_image_file(p.name):
            continue
        # rel like "train/CNV/img.jpg" -> label_name="CNV"
        parts = Path(rel).parts
        if len(parts) < 2:
            continue
        label_name = parts[1]
        if label_name not in class_to_idx:
            # Fallback: derive from actual parent folder
            label_name = p.parent.name
        if label_name not in class_to_idx:
            continue
        samples.append((str(p), class_to_idx[label_name]))

    class SimpleDS(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            p, y = self.samples[i]
            img = Image.open(p).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, y

    return SimpleDS(samples, transform)

def derive_class_weights_from_counts(counts: Dict[int,int]) -> torch.Tensor:
    # inverse frequency normalized to num_classes
    num_classes = max(counts.keys())+1 if counts else 1
    freq = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    freq = torch.clamp(freq, min=1.0)
    w = 1.0 / freq
    w = (w / w.sum()) * num_classes
    return w

def class_weights_from_index_parquet(index_parquet: Path, split="train", num_classes: int = 4) -> torch.Tensor:
    import pandas as pd
    df = pd.read_parquet(index_parquet)
    sub = df[df["split"] == split]
    counts = sub["label"].value_counts().to_dict()
    freq = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    freq = torch.clamp(freq, min=1.0)
    w = 1.0 / freq
    w = (w / w.sum()) * num_classes
    return w

# -------------------------
# Data loader builder
# -------------------------
def get_data_loaders(dataset_name: str,
                     data_root: Path,
                     batch_size: int,
                     num_workers: int,
                     input_size: int = 224,
                     seed: int = 42,
                     split_manifest_dir: Optional[Path] = None,
                     index_parquet: Optional[Path] = None,
                     eval_test: bool = False) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int, Optional[torch.Tensor], Dict]:
    """Returns train_loader, val_loader, test_loader(optional), num_classes, class_weight_tensor(optional), label_map (for HAM)"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    ds_root = Path(data_root) / dataset_name
    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {ds_root}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.05, 0.05, 0.0, 0.0),  # mild for OCT; texture-sensitive
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize(int(input_size*1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # HAM10000 path (unchanged)
    if dataset_name.upper().startswith("HAM10000"):
        csvp = ds_root / "HAM10000_metadata.csv"
        if not csvp.exists():
            csvp = next(ds_root.glob("*metadata*.csv"), None)
            if csvp is None:
                raise RuntimeError(f"HAM10000 metadata CSV not found under {ds_root}")
        full_ds = HAM10000Dataset(ds_root, csvp, transform=train_tf)
        labels = [lbl for _, lbl in full_ds.samples]
        unique_labels = sorted(list(set(labels)))
        label_map = {lab: idx for idx, lab in enumerate(unique_labels)}

        n = len(full_ds)
        idxs = list(range(n)); random.shuffle(idxs)
        split = int(0.8 * n)
        train_idx, val_idx = idxs[:split], idxs[split:]
        train_samples = [full_ds.samples[i] for i in train_idx]
        val_samples = [full_ds.samples[i] for i in val_idx]
        train_ds = RemapDataset(train_samples, train_tf, label_map)
        val_ds = RemapDataset(val_samples, test_tf, label_map)

        num_classes = len(unique_labels)
        lbls_int = [label_map[x] for _, x in full_ds.samples]
        cnt = Counter(lbls_int)
        cw = derive_class_weights_from_counts(cnt)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, None, num_classes, cw, label_map

    # OCT2017 path (Phase-2)
    if dataset_name.upper() == "OCT2017":
        # Try seed manifests
        man = load_seed_manifest(split_manifest_dir, seed) if split_manifest_dir else None

        # Discover classes for mapping
        base_train = datasets.ImageFolder(str(ds_root/"train"))
        num_classes = len(base_train.classes)

        # Build datasets (manifest-driven if available)
        if man:
            train_rel = [r for r in man.get("train", []) if r.startswith("train/")]
            val_rel   = [r for r in man.get("val",   []) if r.startswith("val/")]
            test_rel  = [r for r in man.get("test",  []) if r.startswith("test/")]
            train_ds = make_oct_dataset_from_manifest(ds_root, train_rel, transform=train_tf)
            val_ds   = make_oct_dataset_from_manifest(ds_root, val_rel,   transform=test_tf)
            test_ds  = make_oct_dataset_from_manifest(ds_root, test_rel,  transform=test_tf) if eval_test else None
        else:
            # Fallback to folder scanning in natural order
            train_ds = datasets.ImageFolder(str(ds_root/"train"), transform=train_tf)
            val_ds   = datasets.ImageFolder(str(ds_root/"val"),   transform=test_tf)
            test_ds  = datasets.ImageFolder(str(ds_root/"test"),  transform=test_tf) if eval_test else None

        # Class weights (prefer parquet if provided)
        if index_parquet and Path(index_parquet).exists():
            cw = class_weights_from_index_parquet(Path(index_parquet), split="train", num_classes=num_classes)
        else:
            # derive from train_ds samples
            counts = defaultdict(int)
            if hasattr(train_ds, "samples"):
                for _, y in train_ds.samples:
                    counts[int(y)] += 1
            else:
                # our SimpleDS uses .samples too; guard anyway
                for i in range(len(train_ds)):
                    _, y = train_ds[i]
                    counts[int(y)] += 1
            cw = derive_class_weights_from_counts(counts)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
        val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
        test_loader  = DataLoader(test_ds,  batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)) if eval_test and test_ds is not None else None
        return train_loader, val_loader, test_loader, num_classes, cw, None

    # Fallback generic (not used in our project)
    train_folder = ds_root / "train"
    val_folder = ds_root / "val"
    if not train_folder.exists() or not val_folder.exists():
        raise RuntimeError(f"Expected train/val under {ds_root}")
    train_ds = datasets.ImageFolder(str(train_folder), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_folder), transform=test_tf)
    num_classes = len(train_ds.classes)
    cw = None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, None, num_classes, cw, None

# -------------------------
# Model creation
# -------------------------
def make_model(num_classes: int, pretrained_backbone: Optional[str] = None) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if pretrained_backbone:
        sd = torch.load(pretrained_backbone, map_location="cpu")
        if isinstance(sd, dict) and 'state_dict' in sd and len(sd) > 1:
            sd = sd['state_dict']
        new_sd = {}
        for k, v in sd.items():
            nk = k[len('module.'):] if k.startswith('module.') else k
            if nk.startswith('fc') or nk.startswith('classifier'):
                continue
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    return model

# -------------------------
# Train / Validate / Test loops (with tqdm)
# -------------------------
def train_epoch(model, loader, optimizer, criterion, device, scaler, accum_steps=1, amp_enabled: bool = False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(loader), total=len(loader), desc="train", ncols=120)
    for i, (x,y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y) / max(1,accum_steps)
        if amp_enabled:
            scaler.scale(loss).backward()
            if (i+1) % accum_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i+1) % accum_steps == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.item() * max(1,accum_steps))
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.size(0)
        pbar.set_postfix({"loss": f"{float(loss.item()*max(1,accum_steps)):.4f}", "acc": f"{correct/total:.4f}"})
    avg_loss = running_loss / max(1, len(loader))
    acc = correct / total if total>0 else 0.0
    return avg_loss, acc

@torch.no_grad()
def eval_loop(model, loader, criterion, device, amp_enabled: bool = False, desc: str = "eval"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, total=len(loader), desc=desc, ncols=120)
    for x,y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        running_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.size(0)
        pbar.set_postfix({"loss": f"{float(loss.item()):.4f}", "acc": f"{correct/total:.4f}"})
    avg_loss = running_loss / max(1, len(loader))
    acc = correct / total if total>0 else 0.0
    return avg_loss, acc

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["HAM10000", "OCT2017"], help="Dataset name under data/")
    parser.add_argument("--data-root", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2, help="Keep modest on Windows + RTX3050")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/teachers/trained")
    parser.add_argument("--pretrained-backbone", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--save-every", type=int, default=1)

    # Phase-2 deltas
    parser.add_argument("--split-manifest-dir", type=str, default=None, help="Dir with seed_{k}.json manifests (OCT2017)")
    parser.add_argument("--index-parquet", type=str, default=None, help="index parquet for class weight derivation (optional)")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate on test split (OCT2017) and dump metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Quick run for debugging")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            print("Using CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            print("Using CUDA device")
    else:
        print("Using device:", device)

    amp_enabled = bool(args.amp)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if SummaryWriter else None

    # Data
    train_loader, val_loader, test_loader, num_classes, class_weight_tensor, label_map = get_data_loaders(
        args.dataset, Path(args.data_root), args.batch_size, args.num_workers,
        input_size=args.input_size, seed=args.seed,
        split_manifest_dir=(Path(args.split_manifest_dir) if args.split_manifest_dir else None),
        index_parquet=(Path(args.index_parquet) if args.index_parquet else None),
        eval_test=args.eval_test
    )
    print(f"Dataset: {args.dataset} | num_classes: {num_classes} | train batches: {len(train_loader)} | val batches: {len(val_loader)} | test: {bool(test_loader)}")

    if label_map:
        dump_json(label_map, out_dir / "label_map.json")
        print(f"Saved label_map.json with {len(label_map)} classes to {out_dir}")

    # Model
    model = make_model(num_classes, pretrained_backbone=args.pretrained_backbone).to(device)

    # Loss (class-weighted if available)
    weight = class_weight_tensor.to(device) if class_weight_tensor is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Optim + sched
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_acc = 0.0

    # Resume
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        scheduler.load_state_dict(ck.get('sched_state', scheduler.state_dict()))
        start_epoch = ck.get('epoch', 0) + 1
        best_val_acc = ck.get('best_val_acc', 0.0)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch} best_val_acc={best_val_acc}")

    metrics_path = out_dir / "metrics.json"
    hist = {"epochs": []}

    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler, accum_steps=args.accum_steps, amp_enabled=amp_enabled)
        val_loss, val_acc = eval_loop(model, val_loader, criterion, device, amp_enabled=amp_enabled, desc="val")
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # Save last + best + per-epoch
        ck = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }
        torch.save(ck, out_dir / "ckpt-last.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ck, out_dir / "ckpt-best.pth")
            print(f"Saved new best (val_acc={best_val_acc:.4f}) -> {out_dir/'ckpt-best.pth'}")
        if args.save_every > 0 and ((epoch % args.save_every) == 0):
            torch.save(ck, out_dir / f"ckpt-epoch{epoch:03d}.pth")

        # record epoch metrics
        hist["epochs"].append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(optimizer.param_groups[0]['lr'])
        })
        dump_json(hist, metrics_path)

        if args.dry_run:
            print("Dry run: stopping after 1 epoch")
            break

    # Optional test evaluation at the end
    test_summary = None
    if args.eval_test and test_loader is not None:
        t_loss, t_acc = eval_loop(model, test_loader, criterion, device, amp_enabled=amp_enabled, desc="test")
        test_summary = {"test_loss": float(t_loss), "test_acc": float(t_acc)}
        print(f"[TEST] loss={t_loss:.4f} acc={t_acc:.4f}")

    if writer:
        writer.close()

    final = {
        "best_val_acc": float(best_val_acc),
        "epochs": len(hist["epochs"]),
        "seed": int(args.seed),
        "dataset": args.dataset,
    }
    if test_summary:
        final.update(test_summary)
    dump_json(final, out_dir / "final_summary.json")
    print("Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
