#!/usr/bin/env python3
"""
KD + Attention Transfer training script (ResNet50 teacher -> ResNet18 student).

Usage (PowerShell):
python .\external_src\students\train_student_kd.py `
  --dataset HAM10000 `
  --data-root .\data `
  --teacher-ckpt .\models\teachers\runs_ham10000_resnet50\ckpt-best.pth `
  --save-dir .\models\students\distilled_resnet18_ham10000 `
  --epochs 30 `
  --batch-size 32 `
  --accum-steps 2 `
  --lr 2e-4 `
  --alpha 0.5 --temp 4.0 --beta 1000.0 `
  --amp
"""
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models

# Reuse data loader & model builder from teacher script
from external_src.teachers.train_teacher import get_data_loaders

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

def make_student_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_teacher_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------------
# Feature hook helpers for attention transfer
# -------------------------
def register_feature_hooks(model: nn.Module, layer_names: List[str], features: Dict[str, torch.Tensor]):
    """
    Register forward hooks on modules named in layer_names (attributes on model).
    Expects layer_names like ['layer1','layer2','layer3','layer4'] (for ResNet).
    """
    handles = []
    for name in layer_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        def _hook(mod, inp, out, key=name):
            # store output tensor (detach later as needed)
            features[key] = out
        handles.append(module.register_forward_hook(_hook))
    return handles

def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

def attention_map_from_feature(f: torch.Tensor) -> torch.Tensor:
    """
    f: (B, C, H, W) -> att: (B, H*W) normalized per-sample
    Standard AT uses squared sum across channels, then L2 normalize spatial map.
    """
    # squared sum across channels -> (B, H, W)
    att = (f ** 2).sum(dim=1)  # (B, H, W)
    B = att.shape[0]
    att_flat = att.view(B, -1)  # (B, H*W)
    # L2 normalize each sample
    denom = torch.norm(att_flat, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    att_norm = att_flat / denom
    return att_norm  # (B, H*W)

# -------------------------
# Loss functions: KD + AT
# -------------------------
def kd_loss_fn(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    KLDiv between teacher soft targets and student log-softmax (with temperature).
    Multiply by T^2 as per Hinton et al.
    """
    # student: log_softmax, teacher: softmax
    s_log = F.log_softmax(student_logits / T, dim=1)
    t_soft = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(s_log, t_soft, reduction='batchmean') * (T * T)
    return kd

def attention_transfer_loss(student_feats: Dict[str, torch.Tensor], teacher_feats: Dict[str, torch.Tensor],
                            selected_layers: List[str]) -> torch.Tensor:
    """
    Compute AT loss between matching layers in selected_layers.
    Each layer contributes mean squared error between normalized attention maps.
    """
    losses = []
    for lname in selected_layers:
        if lname not in student_feats or lname not in teacher_feats:
            continue
        sf = student_feats[lname]
        tf = teacher_feats[lname]
        # ensure compatible spatial size: if different, we can interpolate student -> teacher
        if sf.shape[-2:] != tf.shape[-2:]:
            sf = F.interpolate(sf, size=tf.shape[-2:], mode='bilinear', align_corners=False)
        a_s = attention_map_from_feature(sf)  # (B, H*W) normalized
        a_t = attention_map_from_feature(tf)
        # MSE between attention maps
        losses.append(F.mse_loss(a_s, a_t, reduction='mean'))
    if not losses:
        return torch.tensor(0.0, device=next(iter(student_feats.values())).device)
    return sum(losses) / len(losses)

# -------------------------
# Training / Validation loops
# -------------------------
def train_one_epoch(student, teacher, train_loader, optimizer, criterion_ce, device, scaler,
                    accum_steps: int, alpha: float, T: float, beta: float, selected_layers: list, amp_enabled: bool):
    student.train()
    teacher.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    # feature holders and hooks
    s_feats = {}
    t_feats = {}
    s_handles = register_feature_hooks(student, selected_layers, s_feats)
    t_handles = register_feature_hooks(teacher, selected_layers, t_feats)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="train batches", ncols=120)
    for i, (x, y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits_s = student(x)
            with torch.no_grad():
                logits_t = teacher(x)
            loss_ce = criterion_ce(logits_s, y)
            loss_kd = kd_loss_fn(logits_s, logits_t, T)
            loss_at = attention_transfer_loss(s_feats, t_feats, selected_layers)
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd + beta * loss_at
            loss = loss / accum_steps

        if amp_enabled:
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        running_loss += float(loss.item() * accum_steps)
        preds = logits_s.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.size(0)

        pbar.set_postfix({"batch_loss": f"{float(loss.item()*accum_steps):.4f}", "acc_sofar": f"{correct/total:.4f}"})

        # clear feature buffers for next batch
        s_feats.clear()
        t_feats.clear()

    remove_hooks(s_handles)
    remove_hooks(t_handles)
    avg_loss = running_loss / max(1, len(train_loader))
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def validate(student, val_loader, device, criterion_ce, selected_layers: list, amp_enabled: bool):
    student.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # no teacher needed for pure validation accuracy/loss
    pbar = tqdm(val_loader, total=len(val_loader), desc="val batches", ncols=120)
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = student(x)
                loss = criterion_ce(logits, y)
            running_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.size(0)
            pbar.set_postfix({"batch_loss": f"{float(loss.item()):.4f}", "acc_sofar": f"{correct/total:.4f}"})
    avg_loss = running_loss / max(1, len(val_loader))
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["HAM10000", "OCT2017"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="models/students/distilled_resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for CE vs KD (alpha*CE + (1-alpha)*KD)")
    parser.add_argument("--temp", type=float, default=4.0, help="KD temperature")
    parser.add_argument("--beta", type=float, default=1000.0, help="AT loss weight")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = bool(args.amp)
    scaler = torch.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    # data loaders - reuse get_data_loaders; returns train,val,num_classes,class_weights,label_map
    train_loader, val_loader, num_classes, class_weights, label_map = get_data_loaders(
        args.dataset, Path(args.data_root), args.batch_size, args.num_workers, input_size=args.input_size, seed=args.seed
    )
    print(f"Dataset: {args.dataset} | num_classes: {num_classes} | train_batches: {len(train_loader)} val_batches: {len(val_loader)}")
    if label_map:
        with open(out_dir / "label_map.json", "w", encoding="utf8") as f:
            json.dump(label_map, f, indent=2)

    # build models
    teacher = make_teacher_resnet50(num_classes=num_classes)
    student = make_student_resnet18(num_classes=num_classes)

    # load teacher checkpoint (trained ckpt must include model_state)
    ck = torch.load(args.teacher_ckpt, map_location="cpu")
    sd = ck.get("model_state", ck)
    # load into teacher; ignore final fc mismatch if any
    new_sd = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_sd[nk] = v
    teacher.load_state_dict(new_sd, strict=False)
    teacher = teacher.to(device).eval()
    student = student.to(device)

    # losses & optimizer
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # selected layers for attention transfer - common ResNet block names
    selected_layers = ["layer1", "layer2", "layer3", "layer4"]

    start_epoch = 0
    best_val_acc = 0.0

    # resume logic (look for student ckpt)
    last_ckpt = out_dir / "ckpt-last.pth"
    if last_ckpt.exists():
        ck2 = torch.load(last_ckpt, map_location="cpu")
        student.load_state_dict(ck2['model_state'])
        optimizer.load_state_dict(ck2['optim_state'])
        scheduler.load_state_dict(ck2.get('sched_state', scheduler.state_dict()))
        start_epoch = ck2.get('epoch', 0) + 1
        best_val_acc = ck2.get('best_val_acc', 0.0)
        print(f"Resumed student from {last_ckpt}: start_epoch={start_epoch} best_val_acc={best_val_acc}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        train_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, criterion_ce, device, scaler,
            accum_steps=args.accum_steps, alpha=args.alpha, T=args.temp, beta=args.beta,
            selected_layers=selected_layers, amp_enabled=amp_enabled
        )
        val_loss, val_acc = validate(student, val_loader, device, criterion_ce, selected_layers, amp_enabled=amp_enabled)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # checkpoint
        ckst = {
            'epoch': epoch,
            'model_state': student.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }
        torch.save(ckst, out_dir / "ckpt-last.pth")
        if (val_acc > best_val_acc) or (epoch % args.save_every == 0 and args.save_every > 0):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(ckst, out_dir / "ckpt-best.pth")
                print(f"Saved new best student (acc={best_val_acc:.4f}) -> {out_dir / 'ckpt-best.pth'}")
            torch.save(ckst, out_dir / f"ckpt-epoch{epoch:03d}.pth")

        if args.dry_run:
            print("Dry run: stopping after 1 epoch")
            break

    print("Student training finished. Best val acc:", best_val_acc)
    # export best model path reminder
    print("Best student checkpoint:", out_dir / "ckpt-best.pth")

if __name__ == "__main__":
    main()
