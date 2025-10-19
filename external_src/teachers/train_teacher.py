#!/usr/bin/env python3
"""
Train a teacher model (ResNet50) on HAM10000 or OCT2017.

Example PowerShell run (from repo root D:/MedMNIST-EdgeAIv2):
python ./external_src/teachers/train_teacher.py ^
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
"""
import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Dict

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
    return any(name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])

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
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[lbl]

# -------------------------
# HAM10000-specific metadata loader
# -------------------------
class HAM10000Dataset(Dataset):
    """
    Builds samples from HAM10000 metadata CSV and the two image-part folders you have:
      - HAM10000_images_part_1
      - HAM10000_images_part_2
    Expects metadata CSV to have either:
      - 'image_id' and 'dx' columns (common HAM10000), or
      - 'image' and 'label' columns (more generic)
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
            raise RuntimeError(f"No image part folders found under {self.root}. Expected HAM10000_images_part_1/_part_2.")
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
# Generic CSV dataset (kept for other datasets)
# -------------------------
class CSVDataset(Dataset):
    def __init__(self, root: Path, csv_path: Path, transform=None, img_col: str = "image", label_col: str = "label"):
        import csv
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                img = r[img_col]
                lbl = r[label_col]
                p = (self.root / img)
                if p.exists():
                    self.samples.append((str(p), int(lbl)))
                else:
                    p2 = (self.root / Path(img).name)
                    if p2.exists():
                        self.samples.append((str(p2), int(lbl)))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, l

# -------------------------
# Data loader builder
# -------------------------
def get_data_loaders(dataset_name: str, data_root: Path, batch_size: int, num_workers: int,
                     input_size: int = 224, seed: int = 42) -> Tuple[DataLoader, DataLoader, int, Dict[int, float], Dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds_root = Path(data_root) / dataset_name
    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {ds_root}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(input_size*1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # HAM10000 special handling
    if dataset_name.upper().startswith("HAM10000".upper()):
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
        val_ds = RemapDataset(val_samples, val_tf, label_map)

        num_classes = len(unique_labels)
        lbls_int = [label_map[x] for _, x in full_ds.samples]
        cnt = Counter(lbls_int)
        tot = sum(cnt.values())
        class_weights = {i: (tot / (cnt.get(i,1))) for i in range(num_classes)}
        s = sum(class_weights.values())
        for k in list(class_weights.keys()):
            class_weights[k] = float(class_weights[k]) / s
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, num_classes, class_weights, label_map

    # Generic path: ImageFolder or metadata.csv fallback
    train_folder = ds_root / "train"
    val_folder = ds_root / "val"
    if train_folder.exists() and val_folder.exists():
        train_ds = datasets.ImageFolder(str(train_folder), transform=train_tf)
        val_ds = datasets.ImageFolder(str(val_folder), transform=val_tf)
        num_classes = len(train_ds.classes)
        class_weights = None
    else:
        csvp = ds_root / "metadata.csv"
        if csvp.exists():
            all_ds = CSVDataset(ds_root, csvp, transform=train_tf)
            n = len(all_ds); idxs = list(range(n)); random.shuffle(idxs)
            split = int(0.8 * n)
            from torch.utils.data import Subset
            train_ds = Subset(all_ds, idxs[:split])
            val_ds = Subset(all_ds, idxs[split:])
            labels = []
            for _, l in all_ds:
                labels.append(int(l))
            num_classes = max(labels)+1 if labels else 2
            class_weights = None
        else:
            ds = datasets.ImageFolder(str(ds_root), transform=train_tf)
            n = len(ds); idxs = list(range(n)); random.shuffle(idxs)
            split = int(0.8*n)
            from torch.utils.data import Subset
            train_ds = Subset(ds, idxs[:split])
            val_ds = Subset(ds, idxs[split:])
            num_classes = len(ds.classes)
            class_weights = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes, class_weights, None

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
# Train / Validate loops (with tqdm)
# -------------------------
def train_epoch(model, loader, optimizer, criterion, device, scaler, accum_steps=1, amp_enabled: bool = False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="train batches", ncols=120)
    for i, (x,y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y) / accum_steps
        if amp_enabled:
            scaler.scale(loss).backward()
            if (i+1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i+1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        running_loss += float(loss.item() * accum_steps)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.size(0)
        pbar.set_postfix({"batch_loss": f"{float(loss.item()*accum_steps):.4f}", "acc_sofar": f"{correct/total:.4f}"})
    avg_loss = running_loss / max(1, len(loader))
    acc = correct / total if total>0 else 0.0
    return avg_loss, acc

def validate(model, loader, criterion, device, amp_enabled: bool = False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, total=len(loader), desc="val batches", ncols=120)
    with torch.no_grad():
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
            pbar.set_postfix({"batch_loss": f"{float(loss.item()):.4f}", "acc_sofar": f"{correct/total:.4f}"})
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
    parser.add_argument("--num-workers", type=int, default=2, help="Workers: keep small on Windows + RTX3050")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/teachers/trained")
    parser.add_argument("--pretrained-backbone", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Quick run for debugging")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("Using device:", device)

    amp_enabled = bool(args.amp)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if SummaryWriter else None

    train_loader, val_loader, num_classes, class_weights, label_map = get_data_loaders(
        args.dataset, Path(args.data_root), args.batch_size, args.num_workers, input_size=args.input_size, seed=args.seed
    )
    print(f"Dataset: {args.dataset} | num_classes: {num_classes} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")
    if label_map:
        with open(out_dir / "label_map.json", "w", encoding="utf8") as f:
            json.dump(label_map, f, indent=2)
        print(f"Saved label_map.json with {len(label_map)} classes to {out_dir}")

    model = make_model(num_classes, pretrained_backbone=args.pretrained_backbone)
    model = model.to(device)

    if class_weights:
        weight = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32).to(device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        scheduler.load_state_dict(ck.get('sched_state', scheduler.state_dict()))
        start_epoch = ck.get('epoch', 0) + 1
        best_val_acc = ck.get('best_val_acc', 0.0)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch} best_val_acc={best_val_acc}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler, accum_steps=args.accum_steps, amp_enabled=amp_enabled)
        val_loss, val_acc = validate(model, val_loader, criterion, device, amp_enabled=amp_enabled)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        ck = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }
        last_path = out_dir / "ckpt-last.pth"
        torch.save(ck, last_path)
        if (val_acc > best_val_acc) or (epoch % args.save_every == 0 and args.save_every>0):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = out_dir / "ckpt-best.pth"
                torch.save(ck, best_path)
                print(f"Saved new best (acc={best_val_acc:.4f}) -> {best_path}")
            epoch_path = out_dir / f"ckpt-epoch{epoch:03d}.pth"
            torch.save(ck, epoch_path)

        if args.dry_run:
            print("Dry run: stopping after 1 epoch")
            break

    if writer:
        writer.close()
    print("Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
