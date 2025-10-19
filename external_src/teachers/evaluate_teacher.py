#!/usr/bin/env python3
"""
Evaluate a trained teacher checkpoint on the validation split, saving:
 - metrics JSON
 - confusion matrix CSV + PNG
 - per-image predictions CSV

Run (PowerShell):
python -m external_src.teachers.evaluate_teacher `
  --dataset HAM10000 `
  --data-root .\data `
  --checkpoint .\models\teachers\runs_ham10000_resnet50\ckpt-best.pth `
  --save-dir .\models\teachers\runs_ham10000_resnet50\eval
"""
import argparse
import json
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# reuse data loader + model builder from train script
from external_src.teachers.train_teacher import get_data_loaders, make_model

def run_eval(checkpoint_path: Path, dataset: str, data_root: Path, save_dir: Path, device):
    save_dir.mkdir(parents=True, exist_ok=True)

    # get_data_loaders returns: train_loader, val_loader, num_classes, class_weights, label_map
    _, val_loader, num_classes, _, label_map = get_data_loaders(dataset, data_root, batch_size=32, num_workers=2, input_size=224)

    # build model and load checkpoint
    model = make_model(num_classes, pretrained_backbone=None)
    ck = torch.load(checkpoint_path, map_location="cpu")
    sd = ck.get("model_state", ck)
    model.load_state_dict(sd)
    model.to(device).eval()

    all_trues = []
    all_preds = []
    all_probs = []
    rows = []

    idx_base = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            batch_size = imgs.size(0)
            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            # store per-sample row; convert numpy float32 -> native floats for JSON
            for i in range(batch_size):
                probs_i = probs.cpu()[i].numpy().astype(float).tolist()
                rows.append({
                    "index": idx_base + i,
                    "true": int(labels.cpu()[i].item()),
                    "pred": int(preds.cpu()[i].item()),
                    "probs": probs_i
                })
            idx_base += batch_size

    # metrics
    report = classification_report(all_trues, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_trues, all_preds)

    # save metrics
    with open(save_dir / "metrics.json", "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)

    # save confusion matrix CSV
    np.savetxt(save_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # save per-image preds CSV (probs as JSON string), ensuring native floats
    with open(save_dir / "preds.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label", "probabilities_json"])
        for r in rows:
            writer.writerow([r["index"], r["true"], r["pred"], json.dumps(r["probs"])])

    # plot confusion matrix
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    if label_map:
        inv_label_map = {v:k for k,v in label_map.items()}
        labels = [inv_label_map.get(i, str(i)) for i in tick_marks]
    else:
        labels = [str(i) for i in tick_marks]
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print("Evaluation saved to:", save_dir)
    print("Overall macro avg F1:", report.get("macro avg", {}).get("f1-score"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="HAM10000 or OCT2017")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_eval(Path(args.checkpoint), args.dataset, Path(args.data_root), Path(args.save_dir), device)

if __name__ == "__main__":
    main()
