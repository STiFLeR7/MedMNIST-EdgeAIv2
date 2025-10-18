#!/usr/bin/env python3
"""
Teacher training template (ResNet50). Minimal training loop + checkpoint + metadata.
Fill dataset and training details as you iterate.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_dataloaders(data_dir, batch_size):
    # Replace with real dataset class (native-resolution)
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train = datasets.FakeData(transform=transform, size=200)  # placeholder
    val = datasets.FakeData(transform=transform, size=50)
    return torch.utils.data.DataLoader(train, batch_size=batch_size), torch.utils.data.DataLoader(val, batch_size=batch_size)

def train_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total_loss = 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return total_loss / len(loader), correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="external_data/HAM10000")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="external_data/checkpoints/teachers")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_dataloaders(args.data_dir, args.batch_size)

    model = models.resnet50(pretrained=False, num_classes=10)  # set num_classes appropriately
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, loss_fn, opt, device)
        val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={(time.time()-t0):.1f}s")
        ckpt = out_dir / f"resnet50_epoch{epoch}_seed{args.seed}.pth"
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "seed": args.seed}, ckpt)
    # metadata
    meta = {"dataset": args.data_dir, "epochs": args.epochs, "seed": args.seed}
    with open(out_dir / f"resnet50_meta_seed{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
