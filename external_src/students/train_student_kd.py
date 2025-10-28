#!/usr/bin/env python3
r"""
KD + Attention Transfer training script (Phase-2: OCT2017)
Teacher: ResNet50 (ckpt from your teacher run)
Student: EfficientNet-B0 (initialized from local ImageNet weights; no downloads)

Usage (PowerShell):
python -m external_src.students.train_student_kd `
  --dataset OCT2017 `
  --data-root .\data `
  --teacher-ckpt .\models\teachers\oct2017_resnet50_seed0\ckpt-best.pth `
  --save-dir .\models\students\oct2017_effb0_kdat_seed0 `
  --student-init .\models\students\efficientnet_b0_rwightman-3dd342df.pth `
  --epochs 40 `
  --batch-size 16 `
  --accum-steps 2 `
  --lr 2e-4 `
  --weight-decay 5e-2 `
  --seed 0 `
  --eval-test `
  --amp
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models

# Reuse teacher loaders to keep data contract identical
from external_src.teachers.train_teacher import get_data_loaders

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# -------------------------
# Utilities
# -------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2)


# -------------------------
# Teacher / Student builders
# -------------------------
def make_teacher_resnet50(num_classes: int) -> nn.Module:
    t = models.resnet50(weights=None)
    t.fc = nn.Linear(t.fc.in_features, num_classes)
    return t

def _load_local_state_dict(p: Path) -> dict:
    sd = torch.load(p, map_location="cpu")
    # Accept a few common wrappers
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    # Remove DistributedDataParallel prefix if present
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def make_student_efficientnetb0(num_classes: int, init_path: Path) -> nn.Module:
    """
    Build torchvision EfficientNet-B0 and initialize from a **local** ImageNet-1K checkpoint.
    The file `efficientnet_b0_rwightman-3dd342df.pth` matches torchvision's keying;
    we drop classifier weights and load the rest with strict=False.
    """
    m = models.efficientnet_b0(weights=None)
    # Replace classifier head *before* loading to avoid size mismatch
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)

    if init_path is None:
        raise FileNotFoundError("Missing --student-init path for EfficientNet-B0 weights.")
    init_path = Path(init_path)
    if not init_path.exists():
        raise FileNotFoundError(f"Student init weights not found: {init_path}")

    sd = _load_local_state_dict(init_path)

    # Drop classifier weights from the ImageNet checkpoint (1000-class)
    for k in list(sd.keys()):
        if k.startswith("classifier.1.") or k.startswith("classifier.fc.") or k in {"classifier.weight","classifier.bias"}:
            sd.pop(k, None)

    # Load backbone weights only
    m.load_state_dict(sd, strict=False)
    return m


# -------------------------
# Attention hooks (AT)
# -------------------------
class _FeatHook:
    def __init__(self, key): self.key = key; self.out = None
    def __call__(self, module, inp, out): self.out = out

def register_feature_hooks(model: nn.Module, layers: List[str]) -> Dict[str, _FeatHook]:
    hooks = {}
    named = dict(model.named_modules())
    for lname in layers:
        target = None
        if lname in named:
            target = named[lname]
        else:
            for k in named:
                if k.endswith(lname) or lname.endswith(k):
                    target = named[k]; break
        if target is None:  # skip silently
            continue
        hk = _FeatHook(lname)
        target.register_forward_hook(hk)
        hooks[lname] = hk
    return hooks

def attention_map(feat: torch.Tensor) -> torch.Tensor:
    # (B,C,H,W) -> L2-normalized spatial energy (B,1,H,W)
    att = (feat ** 2).sum(dim=1, keepdim=True)
    B, _, H, W = att.shape
    flat = att.view(B, -1)
    denom = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    flat = flat / denom
    return flat.view(B, 1, H, W)

def at_loss(student_feats, teacher_feats) -> torch.Tensor:
    loss = 0.0; n = 0
    for sf, tf in zip(student_feats, teacher_feats):
        if sf is None or tf is None:
            continue
        if sf.shape[-2:] != tf.shape[-2:]:
            sf = torch.nn.functional.interpolate(sf, size=tf.shape[-2:], mode="bilinear", align_corners=False)
        sA = attention_map(sf)
        tA = attention_map(tf)
        loss = loss + torch.nn.functional.mse_loss(sA, tA)
        n += 1
    if n == 0:
        dev = student_feats[0].device if student_feats and isinstance(student_feats[0], torch.Tensor) else "cpu"
        return torch.tensor(0.0, device=dev)
    return loss / n

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)


# -------------------------
# Train / Eval loops
# -------------------------
def train_epoch(student, teacher, loader, optimizer, ce, device, scaler,
                accum_steps, alpha, temperature, beta, s_hooks, t_hooks, amp):
    student.train(); teacher.eval()
    optimizer.zero_grad(set_to_none=True)

    run_loss = 0.0; run_acc = 0; seen = 0
    pbar = tqdm(enumerate(loader), total=len(loader), ncols=120, desc="train")
    for i, (x, y) in pbar:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=amp):
            t_logits = teacher(x)
            t_feats = [hk.out for hk in t_hooks.values()]

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            s_logits = student(x)
            s_feats = [hk.out for hk in s_hooks.values()]
            loss_ce = ce(s_logits, y)
            loss_kd = kd_loss(s_logits, t_logits, temperature)
            loss_at = at_loss(s_feats, t_feats)
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd + beta * loss_at
            loss = loss / max(1,accum_steps)

        if amp:
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)

        run_loss += float(loss.item() * max(1,accum_steps))
        run_acc  += int((s_logits.argmax(1) == y).sum().item())
        seen     += y.size(0)
        pbar.set_postfix({"loss": f"{run_loss/max(1,i+1):.4f}", "acc": f"{run_acc/max(1,seen):.4f}"})

    return run_loss / max(1,len(loader)), (run_acc / max(1,seen))


@torch.no_grad()
def eval_epoch(model, loader, ce, device, amp, desc="val"):
    model.eval()
    run_loss = 0.0; run_acc = 0; seen = 0
    for x, y in tqdm(loader, total=len(loader), ncols=120, desc=desc):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = ce(logits, y)
        run_loss += float(loss.item())
        run_acc  += int((logits.argmax(1) == y).sum().item())
        seen     += y.size(0)
    return run_loss / max(1,len(loader)), (run_acc / max(1,seen))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["OCT2017", "HAM10000"], help="OCT2017 for Phase-2; HAM10000 kept for parity.")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--student-init", type=str, default="models/students/efficientnet_b0_rwightman-3dd342df.pth",
                    help="Local EfficientNet-B0 ImageNet weights .pth (no download).")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)

    # KD/AT
    ap.add_argument("--alpha", type=float, default=0.2, help="Weight for CE(y); rest goes to KD.")
    ap.add_argument("--temp", type=float, default=4.0, help="KD temperature")
    ap.add_argument("--beta", type=float, default=0.2, help="AT weight (scaled to MSE of attention maps)")

    # runtime
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval-test", action="store_true", help="Run test set at the end if available")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    seed_everything(args.seed)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # mute TF log spam if present

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device if device.type == "cpu" else torch.cuda.get_device_name(0))

    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb")) if SummaryWriter else None
    scaler = torch.amp.GradScaler(enabled=args.amp)

    # ------------ Data (shared contract with teacher) ------------
    loaders = get_data_loaders(
        args.dataset,
        Path(args.data_root),
        args.batch_size,
        args.num_workers,
        input_size=args.input_size,
        seed=args.seed,
        eval_test=args.eval_test
    )
    if args.dataset.upper() == "OCT2017":
        train_loader, val_loader, test_loader, num_classes, class_w, _ = loaders
    else:
        train_loader, val_loader, num_classes, class_w, label_map = loaders
        test_loader = None
        dump_json(label_map, out_dir / "label_map.json")

    print(f"Dataset={args.dataset} | num_classes={num_classes} | train_batches={len(train_loader)} | val_batches={len(val_loader)} | test={bool(test_loader)}")

    # ------------ Models ------------
    teacher = make_teacher_resnet50(num_classes)
    ck = torch.load(args.teacher_ckpt, map_location="cpu")
    sd = ck.get("model_state", ck)
    teacher.load_state_dict(sd, strict=True)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False

    student = make_student_efficientnetb0(num_classes, init_path=Path(args.student_init)).to(device)

    # ------------ Loss / Optim ------------
    ce_weight = class_w.to(device) if class_w is not None else None
    ce = nn.CrossEntropyLoss(weight=ce_weight)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # EfficientNet-B0 taps (coarse pyramid). Teacher taps are resnet stages on OCT; on HAM we mirror student taps.
    at_layers_student = ["features.2", "features.4", "features.6", "features.8"]
    s_hooks = register_feature_hooks(student, at_layers_student)
    t_hooks = register_feature_hooks(teacher, ["layer1", "layer2", "layer3", "layer4"]) if args.dataset.upper() == "OCT2017" else register_feature_hooks(teacher, at_layers_student)

    best_val_acc = 0.0
    traj = {"epochs": []}

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        tr_loss, tr_acc = train_epoch(
            student, teacher, train_loader, optimizer, ce, device, scaler,
            args.accum_steps, args.alpha, args.temp, args.beta, s_hooks, t_hooks, args.amp
        )
        va_loss, va_acc = eval_epoch(student, val_loader, ce, device, args.amp, desc="val")
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if writer:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", va_loss, epoch)
            writer.add_scalar("acc/train", tr_acc, epoch)
            writer.add_scalar("acc/val", va_acc, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # save last/best and per-epoch
        ckpt = {
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "hparams": {
                "alpha": args.alpha, "temp": args.temp, "beta": args.beta,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "batch_size": args.batch_size, "accum_steps": args.accum_steps
            }
        }
        torch.save(ckpt, out_dir / "ckpt-last.pth")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(ckpt, out_dir / "ckpt-best.pth")
            print(f"Saved new best (val_acc={best_val_acc:.4f}) -> {out_dir/'ckpt-best.pth'}")

        traj["epochs"].append({
            "epoch": epoch, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
            "val_loss": float(va_loss), "val_acc": float(va_acc),
            "lr": float(optimizer.param_groups[0]["lr"])
        })
        dump_json(traj, out_dir / "metrics.json")

        if args.dry_run:
            print("Dry run: stopping after 1 epoch"); break

    # Optional test pass
    final = {"best_val_acc": float(best_val_acc), "epochs": len(traj["epochs"])}
    if args.eval_test and test_loader is not None:
        te_loss, te_acc = eval_epoch(student, test_loader, ce, device, args.amp, desc="test")
        print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")
        final.update({"test_loss": float(te_loss), "test_acc": float(te_acc)})

    if writer: writer.close()
    dump_json(final, out_dir / "final_summary.json")
    print("Student training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
