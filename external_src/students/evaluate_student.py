#!/usr/bin/env python3
"""
Evaluate student checkpoint (ResNet18) on validation set.

Usage (PowerShell, run from repo root):
python .\external_src\students\evaluate_student.py `
  --dataset HAM10000 `
  --data-root .\data `
  --checkpoint .\models\students\distilled_resnet18_ham10000\ckpt-best.pth `
  --save-dir .\models\students\distilled_resnet18_ham10000\eval
"""
import argparse, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from external_src.teachers.train_teacher import get_data_loaders
from torchvision import models

def build_student(num_classes):
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def run_eval(ckpt, dataset, data_root, save_dir, device):
    save_dir.mkdir(parents=True, exist_ok=True)
    _, val_loader, num_classes, _, label_map = get_data_loaders(dataset, data_root, batch_size=32, num_workers=2, input_size=224)

    model = build_student(num_classes)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model_state", ck)
    model.load_state_dict(sd)
    model.to(device).eval()

    all_trues, all_preds = [], []
    rows = []
    idx_base = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            bs = imgs.size(0)
            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            for i in range(bs):
                rows.append({
                    "index": idx_base + i,
                    "true": int(labels.cpu()[i].item()),
                    "pred": int(preds.cpu()[i].item()),
                    "probs": probs.cpu()[i].numpy().astype(float).tolist()
                })
            idx_base += bs

    report = classification_report(all_trues, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_trues, all_preds)

    with open(save_dir / "metrics.json", "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    np.savetxt(save_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    with open(save_dir / "preds.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["index","true","pred","probs_json"])
        for r in rows:
            w.writerow([r["index"], r["true"], r["pred"], json.dumps(r["probs"])])

    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(cm))
    if label_map:
        inv = {v:k for k,v in label_map.items()}
        labels = [inv.get(i,str(i)) for i in tick_marks]
    else:
        labels = [str(i) for i in tick_marks]
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved eval to", save_dir)
    print("Macro F1:", report.get("macro avg", {}).get("f1-score"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_eval(Path(args.checkpoint), args.dataset, Path(args.data_root), Path(args.save_dir), device)

if __name__ == "__main__":
    main()
