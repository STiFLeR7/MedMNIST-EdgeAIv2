#!/usr/bin/env python3
r"""
KD + Attention Transfer training script
Teacher: ResNet50 (from ckpt)
Student: EfficientNet-B0

Usage (PowerShell):
python -m external_src.students.train_student_kd `
  --dataset HAM10000 `
  --data-root .\data `
  --teacher-ckpt .\models\teachers\runs_ham10000_resnet50\ckpt-best.pth `
  --save-dir .\models\students\distilled_efficientnetb0_ham10000 `
  --epochs 30 `
  --batch-size 32 `
  --accum-steps 2 `
  --lr 2e-4 `
  --alpha 0.5 --temp 4.0 --beta 900.0 `
  --amp
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

from external_src.teachers.train_teacher import get_data_loaders

# -------------------------
# Utilities
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_teacher_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_student_efficientnetb0(num_classes):
    """
    Build EfficientNet-B0 from torchvision and replace final classifier.
    """
    model = models.efficientnet_b0(weights=None)
    # model.classifier is typically Sequential(Dropout, Linear)
    in_features = model.classifier[1].in_features if hasattr(model.classifier[1], "in_features") else model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# -------------------------
# Attention hooks
# -------------------------
def register_feature_hooks(model, layers, feats):
    handles = []
    named_mods = dict(model.named_modules())
    for lname in layers:
        submodule = named_mods.get(lname)
        if submodule is None:
            # try fuzzy match by prefix
            for k in named_mods:
                if k.endswith(lname) or k.startswith(lname):
                    submodule = named_mods[k]
                    break
        if submodule is None:
            continue
        def _hook(mod, inp, out, key=lname): feats[key] = out
        handles.append(submodule.register_forward_hook(_hook))
    return handles

def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

def attention_map_from_feature(f):
    # f: (B, C, H, W) -> flattened normalized attention (B, H*W)
    att = (f ** 2).sum(dim=1)  # (B, H, W)
    att_flat = att.view(att.size(0), -1)
    denom = torch.norm(att_flat, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    att_norm = att_flat / denom
    return att_norm

def kd_loss(student_logits, teacher_logits, T):
    s_log = F.log_softmax(student_logits / T, dim=1)
    t_soft = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_log, t_soft, reduction="batchmean") * (T * T)

def at_loss(s_feats, t_feats, layers):
    losses = []
    for lname in layers:
        if lname not in s_feats or lname not in t_feats:
            continue
        sf, tf = s_feats[lname], t_feats[lname]
        # make spatial sizes compatible
        if sf.shape[-2:] != tf.shape[-2:]:
            sf = F.interpolate(sf, size=tf.shape[-2:], mode='bilinear', align_corners=False)
        a_s = attention_map_from_feature(sf)
        a_t = attention_map_from_feature(tf)
        losses.append(F.mse_loss(a_s, a_t))
    if not losses:
        # No matched layers; return zero tensor on correct device
        dev = next(iter(s_feats.values())).device if s_feats else torch.device("cpu")
        return torch.tensor(0.0, device=dev)
    return sum(losses) / len(losses)

# -------------------------
# Train / Validate loops
# -------------------------
def train_one_epoch(student, teacher, train_loader, optimizer, ce_loss, device, scaler,
                    accum_steps, alpha, T, beta, at_layers, amp_enabled):
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    s_feats = {}
    t_feats = {}
    s_hooks = register_feature_hooks(student, at_layers, s_feats)
    t_hooks = register_feature_hooks(teacher, at_layers, t_feats)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="train", ncols=120)
    for i, (x, y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            s_logits = student(x)
            with torch.no_grad():
                t_logits = teacher(x)
            loss_ce = ce_loss(s_logits, y)
            loss_kd = kd_loss(s_logits, t_logits, T)
            loss_at = at_loss(s_feats, t_feats, at_layers)
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

        total_loss += float(loss.item() * accum_steps)
        preds = s_logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.size(0)

        pbar.set_postfix({"loss": f"{(loss.item() * accum_steps):.4f}", "acc": f"{correct/total:.4f}"})

        # clear feature dicts for next batch
        s_feats.clear(); t_feats.clear()

    remove_hooks(s_hooks); remove_hooks(t_hooks)
    avg_loss = total_loss / max(1, len(train_loader))
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def validate(student, val_loader, device, ce_loss, amp_enabled):
    student.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(val_loader, total=len(val_loader), desc="val", ncols=120)
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = student(x)
                loss = ce_loss(logits, y)
            total_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
    avg_loss = total_loss / max(1, len(val_loader))
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temp", type=float, default=4.0)
    p.add_argument("--beta", type=float, default=900.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    seed_everything(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = bool(args.amp)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    train_loader, val_loader, num_classes, class_weights, label_map = get_data_loaders(
        args.dataset, Path(args.data_root), args.batch_size, args.num_workers, args.input_size
    )
    print(f"Dataset {args.dataset} | classes {num_classes}")

    if label_map:
        with open(out_dir / "label_map.json", "w", encoding="utf8") as f:
            json.dump(label_map, f, indent=2)

    teacher = make_teacher_resnet50(num_classes)
    student = make_student_efficientnetb0(num_classes)

    ck = torch.load(args.teacher_ckpt, map_location="cpu")
    sd = ck.get("model_state", ck)
    teacher.load_state_dict(sd, strict=False)
    teacher = teacher.to(device).eval()
    student = student.to(device)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # attention layers chosen across EfficientNet feature pyramid (approximate names)
    # verify with: print(list(student.named_modules())) if you want exact module ids
    at_layers = ["features.2", "features.4", "features.6", "features.8"]
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        train_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, ce_loss, device, scaler,
            args.accum_steps, args.alpha, args.temp, args.beta, at_layers, amp_enabled
        )
        val_loss, val_acc = validate(student, val_loader, device, ce_loss, amp_enabled)
        scheduler.step()

        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc
        }
        torch.save(ckpt, out_dir / "ckpt-last.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, out_dir / "ckpt-best.pth")
            print(f"Saved new best acc={best_val_acc:.4f} -> {out_dir / 'ckpt-best.pth'}")

        if args.dry_run:
            print("Dry run stop.")
            break

    print("Training done. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
