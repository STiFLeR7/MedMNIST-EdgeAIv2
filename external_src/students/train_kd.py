#!/usr/bin/env python3
"""
KD training template: student learns from teacher logits.
Implements a standard KD loss: alpha * CE(student, target) + (1-alpha) * KL(soft_student, soft_teacher)
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import torch.nn.functional as F
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def kd_loss_fn(student_logits, teacher_logits, targets, T, alpha):
    ce = F.cross_entropy(student_logits, targets)
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
    return alpha * ce + (1.0 - alpha) * kl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--out-dir", default="external_data/checkpoints/students")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Placeholder dataloaders (replace with real)
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train = datasets.FakeData(transform=transform, size=200)
    val = datasets.FakeData(transform=transform, size=50)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size)

    # models
    teacher = models.resnet50(pretrained=False, num_classes=10).to(device)
    student = models.resnet18(pretrained=False, num_classes=10).to(device)
    # load teacher ckpt
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(teacher_ckpt.get("model_state", teacher_ckpt))
    teacher.eval()

    opt = optim.Adam(student.parameters(), lr=1e-3)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            s_logits = student(x)
            with torch.no_grad():
                t_logits = teacher(x)
            loss = kd_loss_fn(s_logits, t_logits, y, args.T, args.alpha)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} train_loss {total_loss/len(train_loader):.4f}")
        torch.save({"student_state": student.state_dict(), "epoch": epoch, "seed": args.seed}, out_dir / f"kd_student_epoch{epoch}_T{args.T}_a{args.alpha}_s{args.seed}.pth")

    meta = {"teacher_ckpt": args.teacher_ckpt, "T": args.T, "alpha": args.alpha, "seed": args.seed}
    with open(out_dir / f"kd_meta_T{args.T}_a{args.alpha}_s{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
