#!/usr/bin/env python3
"""
Attention Transfer (AT) training scaffold.
Implements: L = CE(student, target) + lambda * attention_loss(student_feats, teacher_feats)
This is a minimal hook: you must define feature-extractors per architecture.
"""
import argparse
import json
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

def attention_map(x):
    # x: [B,C,H,W] -> attention: [B,1,H,W] normalized
    a = x.abs().sum(dim=1, keepdim=True)
    return a / (a.sum(dim=(2,3), keepdim=True) + 1e-8)

def attention_loss(s_feats, t_feats):
    s_a = attention_map(s_feats)
    t_a = attention_map(t_feats)
    return F.mse_loss(s_a, t_a)

# Minimal feature-hook extraction: for real use, adapt to particular modules
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        # return (logits, feature_map) as simple example
        features = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(self.backbone.conv1(x)))))
        logits = self.backbone.fc(torch.flatten(torch.adaptive_avg_pool2d(features, (1,1)), 1))
        return logits, features

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--lambda_at", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="external_data/checkpoints/students")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train = datasets.FakeData(transform=transform, size=200)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    teacher_backbone = models.resnet50(pretrained=False, num_classes=10)
    student_backbone = models.resnet18(pretrained=False, num_classes=10)
    teacher = FeatureExtractor(teacher_backbone).to(device)
    student = FeatureExtractor(student_backbone).to(device)

    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.backbone.load_state_dict(teacher_ckpt.get("model_state", teacher_ckpt))
    teacher.eval()

    opt = optim.Adam(student.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        student.train()
        tot_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            s_logits, s_feats = student(x)
            with torch.no_grad():
                t_logits, t_feats = teacher(x)
            loss = ce(s_logits, y) + args.lambda_at * attention_loss(s_feats, t_feats)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        print(f"Epoch {epoch} loss {tot_loss/len(train_loader):.4f}")
        torch.save({"student_state": student.state_dict(), "epoch": epoch, "seed": args.seed}, out_dir / f"at_student_epoch{epoch}_l{args.lambda_at}_s{args.seed}.pth")

    meta = {"teacher_ckpt": args.teacher_ckpt, "lambda_at": args.lambda_at, "seed": args.seed}
    with open(out_dir / f"at_meta_l{args.lambda_at}_s{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
