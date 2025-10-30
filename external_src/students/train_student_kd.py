#!/usr/bin/env python3
r"""
KD + Attention Transfer training (Phase-3: ISIC)
Teacher: ResNet50 (ckpt from your teacher run)
Students (trained sequentially in one run): ResNet18, MobileNetV2, EfficientNet-B0

Usage (PowerShell):
python -m external_src.students.train_student_kd `
  --dataset ISIC `
  --data-root .\data `
  --teacher-ckpt .\models\teachers\isic_resnet50_v2\ckpt-best.pth `
  --save-dir .\models\students\isic_kdat_all `
  --student-init-resnet18 .\models\students\resnet18-f37072fd.pth `
  --epochs 40 `
  --batch-size 32 `
  --lr 3e-4 `
  --weight-decay 5e-2 `
  --seed 0 `
  --eval-test `
  --amp
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, models, transforms

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# -------------------------
# Utilities / IO
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

def is_image_file(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"])


# -------------------------
# ISIC loaders (Train[/Val]/Test)
# -------------------------
def _disc(root: Path, name: str) -> Optional[Path]:
    # Accept lower or TitleCase (Train/Test)
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in {name, {"train":"training","val":"validation"}.get(name, name)}:
            return child
    cand = root / name.capitalize()
    return cand if cand.exists() else None

def _class_balanced_sampler(train_samples: List[Tuple[str,int]]):
    from torch.utils.data import WeightedRandomSampler
    from collections import defaultdict
    cnt = defaultdict(int)
    for _, y in train_samples: cnt[int(y)] += 1
    weights = [1.0 / max(1, cnt[int(y)]) for _, y in train_samples]
    return WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)

def _stratified_split(samples: List[Tuple[str,int]], val_frac: float, seed: int):
    from collections import defaultdict
    by = defaultdict(list)
    for p,y in samples: by[int(y)].append((p,y))
    rng = random.Random(1000+seed)
    tr, va = [], []
    for _, lst in by.items():
        rng.shuffle(lst)
        k = max(1, int(round(len(lst)*val_frac)))
        va.extend(lst[:k]); tr.extend(lst[k:])
    return tr, va

class SampleListDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,y = self.samples[i]
        from PIL import Image
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

def build_isic_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    input_size: int,
    seed: int,
    val_frac: float,
    eval_test: bool,
):
    ds_root = data_root / "ISIC"
    train_dir = _disc(ds_root, "train") or (ds_root/"Train")
    val_dir   = _disc(ds_root, "val")
    test_dir  = _disc(ds_root, "test") or (ds_root/"Test")
    if not train_dir or not train_dir.exists():
        raise RuntimeError(f"ISIC expected Train under {ds_root}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7,1.0), ratio=(0.9,1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.05,0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.12), ratio=(0.3,3.3), value="random"),
    ])
    test_tf  = transforms.Compose([
        transforms.Resize(int(input_size*1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    base = datasets.ImageFolder(str(train_dir))
    class_to_idx = base.class_to_idx
    class_names = base.classes
    num_classes = len(class_names)

    # Build datasets (optionally carve Val)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    if val_dir and val_dir.exists():
        val_ds = datasets.ImageFolder(str(val_dir), transform=test_tf)
    else:
        tr, va = _stratified_split(train_ds.samples, val_frac=val_frac, seed=seed)
        train_ds = SampleListDataset(tr, transform=train_tf)
        val_ds   = SampleListDataset(va, transform=test_tf)

    test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf) if (eval_test and test_dir and test_dir.exists()) else None

    # Balanced sampler for Train
    sampler = _class_balanced_sampler(getattr(train_ds, "samples", []))

    from collections import defaultdict
    counts = defaultdict(int)
    for _,y in getattr(train_ds, "samples", []):
        counts[int(y)] += 1
    # inverse-freq weights for CE
    max_cls = max(counts.keys()) if counts else -1
    freq = torch.tensor([counts.get(i,0) for i in range(max_cls+1)], dtype=torch.float32)
    freq = torch.clamp(freq, min=1.0)
    class_weights = (1.0 / freq)
    class_weights = (class_weights / class_weights.sum()) * (max_cls+1 if max_cls>=0 else 1)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size*2, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)
    )
    test_loader = (
        torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size*2, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)
        ) if (eval_test and test_ds is not None) else None
    )
    return train_loader, val_loader, test_loader, num_classes, class_weights, class_names


# -------------------------
# Models
# -------------------------
def extract_model_state(ckpt_obj) -> Dict[str, torch.Tensor]:
    """
    Robustly extract a state_dict from varied ckpt formats:
      - {'model_state': {...}}
      - {'state_dict': {...}}
      - {'model': {...}}
      - or a raw state_dict
    Strips optional 'module.' prefix from DDP.
    """
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

def make_teacher_resnet50(num_classes: int) -> nn.Module:
    t = models.resnet50(weights=None)
    t.fc = nn.Linear(t.fc.in_features, num_classes)
    return t

def init_from_local_or_tv(model: nn.Module, init_path: Optional[Path], drop_classifier_prefix: str):
    if init_path is not None:
        raw = torch.load(str(init_path), map_location="cpu")
        sd = raw
        if isinstance(sd, dict):
            # allow full checkpoint blobs too
            for k in ["state_dict", "model_state", "model"]:
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]; break
        clean = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k
            clean[nk] = v
        # drop classifier head keys
        for k in list(clean.keys()):
            if k.startswith(drop_classifier_prefix) or k in {f"{drop_classifier_prefix}weight", f"{drop_classifier_prefix}bias"}:
                clean.pop(k, None)
        model.load_state_dict(clean, strict=False)
        return "local"
    # caller should have built with torchvision weights if allowed; otherwise random
    return "tv/random"


def make_student(arch: str, num_classes: int,
                 init_resnet18: Optional[Path],
                 init_mbv2: Optional[Path],
                 init_effb0: Optional[Path],
                 allow_tv_weights: bool = True) -> Tuple[nn.Module, str]:
    arch_l = arch.lower()
    if arch_l == "resnet18":
        m = (models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
             if allow_tv_weights and init_resnet18 is None else models.resnet18(weights=None))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        src = init_from_local_or_tv(m, init_resnet18, "fc.")
        return m, src
    if arch_l == "mobilenet_v2":
        m = (models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
             if allow_tv_weights and init_mbv2 is None else models.mobilenet_v2(weights=None))
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        src = init_from_local_or_tv(m, init_mbv2, "classifier.")
        return m, src
    if arch_l in ["efficientnet_b0","effb0","efficientnet"]:
        m = (models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
             if allow_tv_weights and init_effb0 is None else models.efficientnet_b0(weights=None))
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        src = init_from_local_or_tv(m, init_effb0, "classifier.")
        return m, src
    raise ValueError(f"Unsupported student arch: {arch}")


# -------------------------
# Attention hooks (AT)
# -------------------------
class _FeatHook:
    def __init__(self): self.out = None
    def __call__(self, module, inp, out): self.out = out

def register_at_hooks(model: nn.Module, arch: str) -> List[_FeatHook]:
    named = dict(model.named_modules())
    hooks: List[_FeatHook] = []

    def add(name):
        if name in named:
            hk = _FeatHook(); named[name].register_forward_hook(hk); hooks.append(hk)

    arch = arch.lower()
    if arch.startswith("resnet"):
        for lname in ["layer1","layer2","layer3","layer4"]:
            add(lname)
    elif arch == "mobilenet_v2":
        for idx in [2, 4, 7, 14]:
            key = f"features.{idx}"
            add(key)
    elif arch in ["efficientnet_b0","effb0","efficientnet"]:
        for idx in [2, 3, 4, 6]:
            key = f"features.{idx}"
            add(key)
    else:
        # fallback: last few convs
        for k, m in list(named.items())[-4:]:
            if isinstance(m, nn.Conv2d):
                hk = _FeatHook(); m.register_forward_hook(hk); hooks.append(hk)
    return hooks

def attention_map(feat: torch.Tensor) -> torch.Tensor:
    att = (feat ** 2).sum(dim=1, keepdim=True)
    B, _, H, W = att.shape
    flat = att.view(B, -1)
    denom = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    flat = flat / denom
    return flat.view(B, 1, H, W)

def at_loss(student_feats: List[torch.Tensor], teacher_feats: List[torch.Tensor]) -> torch.Tensor:
    device = student_feats[0].device if student_feats and isinstance(student_feats[0], torch.Tensor) else "cpu"
    loss = torch.tensor(0.0, device=device)
    n = 0
    for sf, tf in zip(student_feats, teacher_feats):
        if sf is None or tf is None: continue
        if sf.shape[-2:] != tf.shape[-2:]:
            sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear", align_corners=False)
        sA = attention_map(sf); tA = attention_map(tf)
        loss = loss + F.mse_loss(sA, tA)
        n += 1
    return loss / max(1, n)

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
            t_feats = [hk.out for hk in t_hooks]

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            s_logits = student(x)
            s_feats = [hk.out for hk in s_hooks]
            loss_ce = ce(s_logits, y)
            loss_kd = kd_loss(s_logits, t_logits, temperature)
            loss_at = at_loss(s_feats, t_feats)
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd + beta * loss_at
            loss = loss / max(1,accum_steps)

        if amp:
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
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
# Orchestrator (three students)
# -------------------------
def train_one_student(
    arch: str,
    teacher: nn.Module,
    loaders,
    num_classes: int,
    out_dir: Path,
    args: argparse.Namespace,
    init_paths: Dict[str, Optional[Path]],
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_w = loaders

    # Build student
    student, init_src = make_student(
        arch,
        num_classes,
        init_resnet18=init_paths.get("resnet18"),
        init_mbv2=init_paths.get("mobilenet_v2"),
        init_effb0=init_paths.get("efficientnet_b0"),
        allow_tv_weights=not args.no_torchvision_weights
    )
    student = student.to(device)
    # Hooks
    s_hooks = register_at_hooks(student, arch)
    t_hooks = register_at_hooks(teacher, "resnet50")  # teacher is resnet50

    # Loss / Optim
    ce = nn.CrossEntropyLoss(weight=(class_w.to(device) if class_w is not None else None))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    # Writers
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb")) if SummaryWriter else None

    best_val = 0.0
    traj = {"epochs": [], "arch": arch, "init": init_src}

    for epoch in range(args.epochs):
        print(f"\n===== [{arch}] Epoch {epoch+1}/{args.epochs} =====")
        tr_loss, tr_acc = train_epoch(
            student, teacher, train_loader, optimizer, ce, device, scaler,
            args.accum_steps, args.alpha, args.temp, args.beta, s_hooks, t_hooks, args.amp
        )
        va_loss, va_acc = eval_epoch(student, val_loader, ce, device, args.amp, desc=f"val-{arch}")
        scheduler.step()

        print(f"[{arch}][Epoch {epoch+1:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if writer:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", va_loss, epoch)
            writer.add_scalar("acc/train", tr_acc, epoch)
            writer.add_scalar("acc/val", va_acc, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # save
        ckpt = {
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val_acc": best_val,
            "hparams": {
                "alpha": args.alpha, "temp": args.temp, "beta": args.beta,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "batch_size": args.batch_size, "accum_steps": args.accum_steps
            }
        }
        torch.save(ckpt, out_dir / "ckpt-last.pth")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(ckpt, out_dir / "ckpt-best.pth")
            print(f"[{arch}] Saved new best (val_acc={best_val:.4f}) -> {out_dir/'ckpt-best.pth'}")

        traj["epochs"].append({
            "epoch": epoch, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
            "val_loss": float(va_loss), "val_acc": float(va_acc),
            "lr": float(optimizer.param_groups[0]["lr"])
        })
        dump_json(traj, out_dir / "metrics.json")

        if args.dry_run: break

    final = {"best_val_acc": float(best_val), "epochs": len(traj["epochs"]), "arch": arch}
    if args.eval_test and (loaders[2] is not None):
        te_loss, te_acc = eval_epoch(student, loaders[2], ce, device, args.amp, desc=f"test-{arch}")
        print(f"[{arch}][TEST] loss={te_loss:.4f} acc={te_acc:.4f}")
        final.update({"test_loss": float(te_loss), "test_acc": float(te_acc)})
    dump_json(final, out_dir / "final_summary.json")
    if writer: writer.close()
    return final


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ISIC"], help="ISIC only in this trainer.")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--save-dir", required=True)

    # Optional local init weights
    ap.add_argument("--student-init-resnet18", type=str, default=None)
    ap.add_argument("--student-init-mobilenet_v2", type=str, default=None)
    ap.add_argument("--student-init-efficientnet_b0", type=str, default=None)
    ap.add_argument("--no-torchvision-weights", action="store_true", help="Disallow TV pretrained; use local or random.")

    ap.add_argument("--students", type=str, default="resnet18,mobilenet_v2,efficientnet_b0",
                    help="Comma-separated list; default trains all three.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.1, help="If Val missing, carve this fraction from Train per-class.")
    ap.add_argument("--seed", type=int, default=0)

    # KD/AT
    ap.add_argument("--alpha", type=float, default=0.2, help="Weight for CE(y); rest goes to KD.")
    ap.add_argument("--temp", type=float, default=4.0, help="KD temperature")
    ap.add_argument("--beta", type=float, default=0.2, help="AT weight (MSE of attention maps)")

    # runtime
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    seed_everything(args.seed)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device if device.type == "cpu" else torch.cuda.get_device_name(0))

    out_root = Path(args.save_dir); out_root.mkdir(parents=True, exist_ok=True)
    writer_root = SummaryWriter(str(out_root / "tb_global")) if SummaryWriter else None

    # ---------- Data
    train_loader, val_loader, test_loader, num_classes, class_w, class_names = build_isic_loaders(
        Path(args.data_root), args.batch_size, args.num_workers, args.input_size,
        seed=args.seed, val_frac=args.val_frac, eval_test=args.eval_test
    )
    print(f"Dataset=ISIC | num_classes={num_classes} | train_batches={len(train_loader)} | "
          f"val_batches={len(val_loader)} | test={bool(test_loader)}")

    # ---------- Teacher
    teacher = make_teacher_resnet50(num_classes)
    ck = torch.load(args.teacher_ckpt, map_location="cpu")
    sd = extract_model_state(ck)
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    # Allow head mismatch; warn if core layers missing
    if missing:
        head_only = all(k.startswith("fc.") for k in missing)
        if not head_only:
            print("[WARN] Missing keys in teacher load:", missing)
    if unexpected:
        print("[WARN] Unexpected keys in teacher load:", unexpected)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False

    loaders = (train_loader, val_loader, test_loader, class_w)

    # ---------- Students list
    students = [s.strip().lower() for s in args.students.split(",") if s.strip()]
    init_paths = {
        "resnet18": Path(args.student_init_resnet18) if args.student_init_resnet18 else None,
        "mobilenet_v2": Path(args.student_init_mobilenet_v2) if args.student_init_mobilenet_v2 else None,
        "efficientnet_b0": Path(args.student_init_efficientnet_b0) if args.student_init_efficientnet_b0 else None,
    }

    results = {}
    for arch in students:
        subdir_map = {
            "resnet18": "student_res18_isic_kdat",
            "mobilenet_v2": "student_mbv2_isic_kdat",
            "efficientnet_b0": "student_effb0_isic_kdat",
        }
        out_dir = out_root / subdir_map.get(arch, f"student_{arch}_isic_kdat")
        res = train_one_student(arch, teacher, loaders, num_classes, out_dir, args, init_paths)
        results[arch] = res

    dump_json(results, out_root / "summary_all_students.json")
    if writer_root: writer_root.close()
    print("KD+AT training complete for:", ", ".join(students))


if __name__ == "__main__":
    main()
