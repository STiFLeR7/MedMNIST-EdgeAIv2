#!/usr/bin/env python3
"""
Train a teacher model (ResNet50 by default) on HAM10000, OCT2017, or ISIC
with stronger regularization and better optimization for ISIC stability/accuracy.

Key upgrades
- ImageNet weights by default (override with --pretrained-backbone to custom .pth)
- Stronger dermoscopy-friendly augs: RandAugment + RandomErasing + modest ColorJitter
- Larger default resolution (--input-size 256), center-crop eval
- Label smoothing CE
- Optional CutMix (--cutmix-p 0.5)
- Class-balanced sampler
- Cosine LR with warmup (--warmup-epochs)
- Optional SWA averaging (--swa-start epoch)
- Deterministic splits (seed manifests or stratified carve)

Example (ISIC, RTX 3050 friendly):
  python external_src/teachers/train_teacher.py ^
    --dataset ISIC ^
    --data-root ./data ^
    --arch resnet50 ^
    --epochs 50 ^
    --batch-size 32 ^
    --lr 3e-4 ^
    --weight-decay 1e-4 ^
    --num-workers 4 ^
    --save-dir ./models/teachers/isic_resnet50_v2 ^
    --split-manifest-dir ./external_data/splits/ISIC ^
    --index-parquet ./data/processed/isic_index.parquet ^
    --warmup-epochs 3 ^
    --cutmix-p 0.5 ^
    --swa-start 40 ^
    --seed 0 ^
    --eval-test ^
    --amp
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.transforms import RandAugment
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
    return any(
        name.lower().endswith(ext)
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]
    )


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2)


# -------------------------
# Global, picklable dataset helpers (Windows-safe)
# -------------------------


class RemapDataset(Dataset):
    """For HAM CSV -> label remap (string -> int)."""

    def __init__(self, samples: List[Tuple[str, str]], transform, label_map: Dict[str, int]):
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


class SampleListDataset(Dataset):
    """Simple (path, int_label) list dataset. Picklable because it's top-level."""

    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform
        self.classes = None  # optional, for parity

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


# -------------------------
# HAM10000 metadata loader (kept for completeness)
# -------------------------


class HAM10000Dataset(Dataset):
    def __init__(
        self, ds_root: Path, csv_path: Path, transform=None, img_col: str = None, label_col: str = None
    ):
        import csv

        self.root = Path(ds_root)
        self.transform = transform
        self.samples = []
        part_dirs = [
            p
            for p in [self.root / "HAM10000_images_part_1", self.root / "HAM10000_images_part_2"]
            if p.exists()
        ]
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

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            if not img_col:
                if "image_id" in headers:
                    img_col = "image_id"
                elif "image" in headers:
                    img_col = "image"
                else:
                    img_col = headers[0]
            if not label_col:
                if "dx" in headers:
                    label_col = "dx"
                elif "label" in headers:
                    label_col = "label"
                elif "diagnosis" in headers:
                    label_col = "diagnosis"
                else:
                    label_col = headers[-1]
            for r in reader:
                raw_img = (r.get(img_col, "") or "").strip()
                raw_label = (r.get(label_col, "") or "").strip()
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
                        chosen = images_map[c]
                        break
                if not chosen and raw_img in images_map:
                    chosen = images_map[raw_img]
                if not chosen:
                    for k, v in images_map.items():
                        if raw_img in k:
                            chosen = v
                            break
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
# Seeded manifest helpers (OCT/ISIC)
# -------------------------


def load_seed_manifest(manifest_dir: Optional[Path], seed: int) -> Optional[Dict[str, List[str]]]:
    if manifest_dir is None:
        return None
    manp = manifest_dir / f"seed_{seed}.json"
    if not manp.exists():
        return None
    data = json.loads(manp.read_text(encoding="utf8"))
    return data.get("splits", None)


def make_imagefolder_from_manifest(
    root: Path, relpaths: List[str], transform, class_to_idx: Dict[str, int]
) -> Dataset:
    samples = []
    for rel in relpaths:
        p = root / rel
        if not p.exists() or not p.is_file() or not is_image_file(p.name):
            continue
        parts = Path(rel).parts
        label_name = None
        if len(parts) >= 2:
            label_name = parts[1]
        if label_name is None:
            label_name = p.parent.name
        if label_name not in class_to_idx:
            lc = {k.lower(): k for k in class_to_idx.keys()}
            if label_name.lower() in lc:
                label_name = lc[label_name.lower()]
            else:
                continue
        samples.append((str(p), class_to_idx[label_name]))
    return SampleListDataset(samples, transform)


def derive_class_weights_from_counts(counts: Dict[int, int]) -> torch.Tensor:
    num_classes = max(counts.keys()) + 1 if counts else 1
    freq = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    freq = torch.clamp(freq, min=1.0)
    w = 1.0 / freq
    w = (w / w.sum()) * num_classes
    return w


def class_weights_from_index_parquet(
    index_parquet: Path, split: str = "train", num_classes: Optional[int] = None
) -> torch.Tensor:
    import pandas as pd

    df = pd.read_parquet(index_parquet)
    sub = df[df["split"] == split]
    if num_classes is None:
        num_classes = int(sub["label"].max()) + 1 if not sub.empty else 1
    counts = sub["label"].value_counts().to_dict()
    freq = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    freq = torch.clamp(freq, min=1.0)
    w = 1.0 / freq
    w = (w / w.sum()) * num_classes
    return w


# -------------------------
# Data loader builder + sampler
# -------------------------


def stratified_split_samples(
    samples: List[Tuple[str, int]], val_frac: float, seed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    by_cls = defaultdict(list)
    for p, y in samples:
        by_cls[int(y)].append((p, y))
    rng = random.Random(1000 + seed)
    train, val = [], []
    for _, lst in by_cls.items():
        rng.shuffle(lst)
        k = max(1, int(round(len(lst) * val_frac)))
        val.extend(lst[:k])
        train.extend(lst[k:])
    return train, val


def build_transforms(input_size: int, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
                RandAugment(num_ops=2, magnitude=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(int(input_size * 1.14)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def class_balanced_sampler(train_samples: List[Tuple[str, int]], num_classes: int) -> WeightedRandomSampler:
    counts = defaultdict(int)
    for _, y in train_samples:
        counts[int(y)] += 1
    weights = [1.0 / max(1, counts[int(y)]) for _, y in train_samples]
    return WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)


# -------------------------
# Model creation
# -------------------------


def make_model(arch: str, num_classes: int, pretrained_backbone: Optional[str] = None) -> nn.Module:
    arch = arch.lower()
    if arch in ["resnet50", "res50", "ham50", "oct50", "teacher50"]:
        if pretrained_backbone:
            m = models.resnet50(weights=None)
            sd = torch.load(pretrained_backbone, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and len(sd) > 1:
                sd = sd["state_dict"]
            new_sd = {}
            for k, v in sd.items():
                nk = k[len("module.") :] if k.startswith("module.") else k
                if nk.startswith("fc") or nk.startswith("classifier"):
                    continue
                new_sd[nk] = v
            m.load_state_dict(new_sd, strict=False)
        else:
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch in ["resnet18", "res18"]:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch in ["mobilenet_v2", "mbv2"]:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch in ["efficientnet_b0", "effb0", "efficientnet"]:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"Unsupported arch: {arch}")


# -------------------------
# Mix/CutMix helpers (CutMix only to stay lean)
# -------------------------


def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return int(bbx1), int(bby1), int(bbx2), int(bby2)


def apply_cutmix(x, y, beta=1.0):
    lam = np.random.beta(beta, beta)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


# -------------------------
# Train / Validate / Test loops
# -------------------------


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    accum_steps=1,
    amp_enabled: bool = False,
    cutmix_p: float = 0.0,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(loader), total=len(loader), desc="train", ncols=120)
    for i, (x, y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        use_cutmix = (cutmix_p > 0.0) and (random.random() < cutmix_p)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            if use_cutmix:
                x, y_a, y_b, lam = apply_cutmix(x, y)
                logits = model(x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss = loss / max(1, accum_steps)
        if amp_enabled:
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.item() * max(1, accum_steps))
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.size(0)
        pbar.set_postfix(
            {"loss": f"{float(loss.item() * max(1, accum_steps)):.4f}", "acc": f"{correct/total:.4f}"}
        )
    avg_loss = running_loss / max(1, len(loader))
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def eval_loop(model, loader, criterion, device, amp_enabled: bool = False, desc: str = "eval"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, total=len(loader), desc=desc, ncols=120)
    for x, y in pbar:
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
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["HAM10000", "OCT2017", "ISIC"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/teachers/trained")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet50", "resnet18", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument(
        "--pretrained-backbone",
        type=str,
        default=None,
        help="If provided, loads conv weights from this .pth instead of TorchVision weights",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    # Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutmix-p", type=float, default=0.0)
    # Warmup + scheduler
    parser.add_argument("--warmup-epochs", type=int, default=3)
    # SWA
    parser.add_argument("--swa-start", type=int, default=-1, help="Epoch to start SWA (-1 to disable)")
    # Manifests/index
    parser.add_argument("--split-manifest-dir", type=str, default=None)
    parser.add_argument("--index-parquet", type=str, default=None)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.1)

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

    # -------- Data
    def build_loaders():
        train_tf = build_transforms(args.input_size, train=True)
        test_tf = build_transforms(args.input_size, train=False)

        ds_root = Path(args.data_root) / args.dataset

        if args.dataset.upper() == "ISIC":
            # Flexible discovery: Train/Test titlecase is fine
            def _disc(root: Path, name: str) -> Optional[Path]:
                # Accept "train", "Train", etc.
                for child in root.iterdir():
                    if child.is_dir() and child.name.lower() in {
                        name,
                        {"train": "training", "val": "validation"}.get(name, name),
                    }:
                        return child
                cand = root / name.capitalize()
                return cand if cand.exists() else None

            train_dir = _disc(ds_root, "train") or (ds_root / "Train")
            val_dir = _disc(ds_root, "val")
            test_dir = _disc(ds_root, "test") or (ds_root / "Test")
            if not train_dir or not train_dir.exists():
                raise RuntimeError(f"ISIC expected Train under {ds_root}")

            base_train = datasets.ImageFolder(str(train_dir))
            class_to_idx = base_train.class_to_idx
            class_names = base_train.classes
            num_classes = len(class_names)

            man = load_seed_manifest(Path(args.split_manifest_dir) if args.split_manifest_dir else None, args.seed)
            if man:
                def _pick(split):
                    return [
                        r for r in man.get(split, []) if Path(r).parts and Path(r).parts[0].lower() == split
                    ]

                train_rel = _pick("train")
                val_rel = _pick("val")
                test_rel = _pick("test")

                train_ds = make_imagefolder_from_manifest(ds_root, train_rel, transform=train_tf, class_to_idx=class_to_idx)

                if val_rel:
                    val_ds = make_imagefolder_from_manifest(ds_root, val_rel, transform=test_tf, class_to_idx=class_to_idx)
                else:
                    tr, va = stratified_split_samples(getattr(train_ds, "samples", []), val_frac=args.val_frac, seed=args.seed)
                    train_ds = SampleListDataset(tr, transform=train_tf)
                    val_ds = SampleListDataset(va, transform=test_tf)

                if args.eval_test and test_rel:
                    test_ds = make_imagefolder_from_manifest(ds_root, test_rel, transform=test_tf, class_to_idx=class_to_idx)
                else:
                    test_ds = (
                        datasets.ImageFolder(str(test_dir), transform=test_tf)
                        if (args.eval_test and test_dir and test_dir.exists())
                        else None
                    )
            else:
                train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
                if val_dir and val_dir.exists():
                    val_ds = datasets.ImageFolder(str(val_dir), transform=test_tf)
                else:
                    tr, va = stratified_split_samples(train_ds.samples, val_frac=args.val_frac, seed=args.seed)
                    train_ds = SampleListDataset(tr, transform=train_tf)
                    val_ds = SampleListDataset(va, transform=test_tf)
                test_ds = (
                    datasets.ImageFolder(str(test_dir), transform=test_tf)
                    if (args.eval_test and test_dir and test_dir.exists())
                    else None
                )

            # Balanced sampler + weights
            train_samples = getattr(train_ds, "samples", [])
            sampler = class_balanced_sampler(train_samples, num_classes)

            if args.index_parquet and Path(args.index_parquet).exists():
                cw = class_weights_from_index_parquet(Path(args.index_parquet), split="train", num_classes=num_classes)
            else:
                counts = defaultdict(int)
                for _, y in train_samples:
                    counts[int(y)] += 1
                cw = derive_class_weights_from_counts(counts)

            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            test_loader = (
                DataLoader(
                    test_ds,
                    batch_size=args.batch_size * 2,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                )
                if (args.eval_test and test_ds is not None)
                else None
            )
            return train_loader, val_loader, test_loader, num_classes, cw, class_names

        if args.dataset.upper() == "OCT2017":
            base = datasets.ImageFolder(str((Path(args.data_root) / "OCT2017") / "train"))
            class_names = base.classes
            num_classes = len(class_names)
            train_ds = datasets.ImageFolder(str((Path(args.data_root) / "OCT2017") / "train"), transform=train_tf)
            val_ds = datasets.ImageFolder(str((Path(args.data_root) / "OCT2017") / "val"), transform=test_tf)
            test_ds = (
                datasets.ImageFolder(str((Path(args.data_root) / "OCT2017") / "test"), transform=test_tf)
                if args.eval_test
                else None
            )
            sampler = class_balanced_sampler(train_ds.samples, num_classes)
            counts = defaultdict(int)
            for _, y in train_ds.samples:
                counts[int(y)] += 1
            cw = derive_class_weights_from_counts(counts)
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            test_loader = (
                DataLoader(
                    test_ds,
                    batch_size=args.batch_size * 2,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                )
                if (args.eval_test and test_ds is not None)
                else None
            )
            return train_loader, val_loader, test_loader, num_classes, cw, class_names

        # HAM path: use your earlier trainer (kept in repo). This compact file focuses on ISIC/OCT.
        raise NotImplementedError("HAM path omitted in this compact rewrite; use previous trainer for HAM.")

    (
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        class_weight_tensor,
        class_names,
    ) = build_loaders()

    print(
        f"Dataset: {args.dataset} | num_classes: {num_classes} | "
        f"train batches: {len(train_loader)} | val batches: {len(val_loader)} | test: {bool(test_loader)}"
    )
    # Persist class map for downstream evals
    class_map = {"label_to_idx": {name: i for i, name in enumerate(class_names)}, "idx_to_label": list(class_names)}
    dump_json(class_map, Path(args.save_dir) / "class_map.json")

    # -------- Model
    model = make_model(args.arch, num_classes, pretrained_backbone=args.pretrained_backbone).to(device)

    # Loss (label smoothing + optional class weights)
    weight = class_weight_tensor.to(device) if class_weight_tensor is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(args.label_smoothing))

    # Optim + cosine with warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    def lr_lambda(current_epoch):
        # linear warmup then cosine
        if current_epoch < args.warmup_epochs:
            return max(1e-3, (current_epoch + 1) / max(1, args.warmup_epochs))
        # cosine over remaining epochs
        t = (current_epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Optional SWA
    swa_model = None
    if args.swa_start >= 0:
        from torch.optim.swa_utils import AveragedModel

        swa_model = AveragedModel(model)

    start_epoch = 0
    best_val_acc = 0.0

    # Resume
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val_acc = ck.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch} best_val_acc={best_val_acc}")

    metrics_path = Path(args.save_dir) / "metrics.json"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    hist = {"epochs": []}

    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            accum_steps=args.accum_steps,
            amp_enabled=amp_enabled,
            cutmix_p=args.cutmix_p,
        )
        val_loss, val_acc = eval_loop(model, val_loader, criterion, device, amp_enabled=amp_enabled, desc="val")
        scheduler.step()

        if swa_model is not None and (args.swa_start >= 0) and (epoch + 1) >= args.swa_start:
            swa_model.update_parameters(model)

        print(
            f"[Epoch {epoch + 1:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if SummaryWriter and writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save last + best + per-epoch
        ck = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }
        torch.save(ck, Path(args.save_dir) / "ckpt-last.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ck, Path(args.save_dir) / "ckpt-best.pth")
            print(f"Saved new best (val_acc={best_val_acc:.4f}) -> {Path(args.save_dir) / 'ckpt-best.pth'}")
        if args.save_every > 0 and ((epoch % args.save_every) == 0):
            torch.save(ck, Path(args.save_dir) / f"ckpt-epoch{epoch:03d}.pth")

        # record epoch metrics
        hist["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
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
        "arch": args.arch,
        "input_size": int(args.input_size),
        "label_smoothing": float(args.label_smoothing),
        "cutmix_p": float(args.cutmix_p),
    }
    if test_summary:
        final.update(test_summary)
    dump_json(final, Path(args.save_dir) / "final_summary.json")
    print("Training finished. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
