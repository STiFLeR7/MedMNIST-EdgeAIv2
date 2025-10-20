#!/usr/bin/env python3
"""
KD + Attention Transfer training script
Teacher: ResNet50 (from ckpt)
Student: MobileNetV2

Usage (PowerShell):
python .\external_src\students\train_student_kd.py `
  --dataset HAM10000 `
  --data-root .\data `
  --teacher-ckpt .\models\teachers\runs_ham10000_resnet50\ckpt-best.pth `
  --save-dir .\models\students\distilled_mobilenetv2_ham10000 `
  --epochs 30 `
  --batch-size 32 `
  --accum-steps 2 `
  --lr 2e-4 `
  --alpha 0.5 --temp 4.0 --beta 750.0 `
  --amp
"""
import argparse, json, random
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

def make_student_mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# -------------------------
# Attention hooks
# -------------------------
def register_feature_hooks(model, layers, feats):
    handles = []
    for lname in layers:
        submodule = dict(model.named_modules()).get(lname)
        if submodule is None: continue
        def _hook(mod, inp, out, key=lname): feats[key] = out
        handles.append(submodule.register_forward_hook(_hook))
    return handles

def remove_hooks(handles):
    for h in handles:
        try: h.remove()
        except Exception: pass

def attention_map_from_feature(f):
    att = (f ** 2).sum(dim=1)
    att_flat = att.view(att.size(0), -1)
    att_norm = att_flat / (torch.norm(att_flat, p=2, dim=1, keepdim=True) + 1e-6)
    return att_norm

def kd_loss(student_logits, teacher_logits, T):
    s_log = F.log_softmax(student_logits / T, dim=1)
    t_soft = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_log, t_soft, reduction="batchmean") * (T*T)

def at_loss(s_feats, t_feats, layers):
    losses = []
    for lname in layers:
        if lname not in s_feats or lname not in t_feats: continue
        sf, tf = s_feats[lname], t_feats[lname]
        if sf.shape[-2:] != tf.shape[-2:]:
            sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear", align_corners=False)
        a_s, a_t = attention_map_from_feature(sf), attention_map_from_feature(tf)
        losses.append(F.mse_loss(a_s, a_t))
    return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=next(iter(s_feats.values())).device)

# -------------------------
# Epoch loops
# -------------------------
def train_one_epoch(student, teacher, train_loader, optimizer, ce_loss, device, scaler,
                    accum_steps, alpha, T, beta, at_layers, amp_enabled):
    student.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    s_feats, t_feats = {}, {}
    s_hooks = register_feature_hooks(student, at_layers, s_feats)
    t_hooks = register_feature_hooks(teacher, at_layers, t_feats)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="train", ncols=120)
    for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            s_logits = student(x)
            with torch.no_grad():
                t_logits = teacher(x)
            loss_ce = ce_loss(s_logits, y)
            loss_kd = kd_loss(s_logits, t_logits, T)
            loss_at = at_loss(s_feats, t_feats, at_layers)
            loss = alpha*loss_ce + (1-alpha)*loss_kd + beta*loss_at
            loss /= accum_steps

        if amp_enabled:
            scaler.scale(loss).backward()
            if (i+1) % accum_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        else:
            loss.backward()
            if (i+1) % accum_steps == 0:
                optimizer.step(); optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds = s_logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}", "acc": f"{correct/total:.4f}"})
        s_feats.clear(); t_feats.clear()

    remove_hooks(s_hooks); remove_hooks(t_hooks)
    return total_loss/len(train_loader), correct/total

def validate(student, val_loader, device, ce_loss, amp_enabled):
    student.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(val_loader, total=len(val_loader), desc="val", ncols=120)
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = student(x)
                loss = ce_loss(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
    return total_loss/len(val_loader), correct/total

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--beta", type=float, default=750.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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
        with open(out_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

    teacher = make_teacher_resnet50(num_classes)
    student = make_student_mobilenetv2(num_classes)
    ck = torch.load(args.teacher_ckpt, map_location="cpu")
    sd = ck.get("model_state", ck)
    teacher.load_state_dict(sd, strict=False)
    teacher.to(device).eval()
    student.to(device)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    at_layers = ["features.2", "features.4", "features.7", "features.14"]
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        train_loss, train_acc = train_one_epoch(student, teacher, train_loader, optimizer, ce_loss, device,
                                                scaler, args.accum_steps, args.alpha, args.temp, args.beta, at_layers, amp_enabled)
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
