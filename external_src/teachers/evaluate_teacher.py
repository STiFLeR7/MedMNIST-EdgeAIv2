#!/usr/bin/env python3
"""
Evaluate a trained teacher checkpoint on the validation split, saving:
 - metrics JSON
 - confusion matrix CSV + PNG
 - per-image predictions CSV (JSON-encoded probs)
 - per-image predictions Parquet (labels, probs, logits)  <-- added for aggregators

Run (PowerShell):
python -m external_src.teachers.evaluate_teacher `
  --dataset HAM10000 `
  --data-root .\data `
  --checkpoint .\models\teachers\runs_ham10000_resnet50\ckpt-best.pth `
  --save-dir .\models\teachers\runs_ham10000_resnet50\eval
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# reuse data loader + model builder from train script
from external_src.teachers.train_teacher import get_data_loaders, make_model


def run_eval(checkpoint_path: Path, dataset: str, data_root: Path, save_dir: Path, device: torch.device):
    save_dir.mkdir(parents=True, exist_ok=True)

    # get_data_loaders returns: train_loader, val_loader, num_classes, class_weights, label_map
    _, val_loader, num_classes, _, label_map = get_data_loaders(
        dataset,
        data_root,
        batch_size=32,
        num_workers=2,
        input_size=224,
    )

    # build model and load checkpoint
    model = make_model(num_classes, pretrained_backbone=None)
    ck = torch.load(checkpoint_path, map_location="cpu")
    sd = ck.get("model_state", ck)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    all_trues: list[int] = []
    all_preds: list[int] = []
    rows_csv: list[dict] = []

    # For Parquet
    labels_list: list[int] = []
    probs_list: list[list[float]] = []
    logits_list: list[list[float]] = []

    idx_base = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            # ensure tensor shape [B, C]
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # accumulate for metrics
            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            # accumulate per-sample for CSV + Parquet
            logits_cpu = logits.detach().cpu().float().numpy()
            probs_cpu = probs.detach().cpu().float().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            for i in range(imgs.size(0)):
                # CSV row (JSON probs string; native Python floats)
                rows_csv.append({
                    "index": idx_base + i,
                    "true_label": int(labels_cpu[i].item() if hasattr(labels_cpu[i], "item") else labels_cpu[i]),
                    "pred_label": int(all_preds[-imgs.size(0) + i]),
                    "probabilities_json": json.dumps(probs_cpu[i].astype(float).tolist()),
                })
                # Parquet columns
                labels_list.append(int(labels_cpu[i].item() if hasattr(labels_cpu[i], "item") else labels_cpu[i]))
                probs_list.append(probs_cpu[i].astype(float).tolist())
                logits_list.append(logits_cpu[i].astype(float).tolist())

            idx_base += imgs.size(0)

    # metrics
    report = classification_report(all_trues, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_trues, all_preds)

    # save metrics JSON
    with open(save_dir / "metrics.json", "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)

    # save confusion matrix CSV
    np.savetxt(save_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # save per-image preds CSV (JSON probs)
    with open(save_dir / "preds.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label", "probabilities_json"])
        for r in rows_csv:
            writer.writerow([r["index"], r["true_label"], r["pred_label"], r["probabilities_json"]])

    # save Parquet for aggregators (labels, probs, logits)
    # io_utils.load_preds can read either 'logits' or 'probs'; we provide both.
    df = pd.DataFrame({
        "labels": labels_list,
        "probs": probs_list,
        "logits": logits_list,
    })
    df.to_parquet(save_dir / "test_preds.parquet", index=False)

    # plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])

    if label_map:
        inv_label_map = {v: k for k, v in label_map.items()}
        axis_labels = [inv_label_map.get(i, str(i)) for i in tick_marks]
    else:
        axis_labels = [str(i) for i in tick_marks]

    plt.xticks(tick_marks, axis_labels, rotation=45, ha="right")
    plt.yticks(tick_marks, axis_labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print("Evaluation saved to:", save_dir)
    print("Overall macro avg F1:", report.get("macro avg", {}).get("f1-score"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="HAM10000 or OCT2017")
    parser.add_argument("--data-root", required=True, type=str, help="Root containing dataset folders (e.g., .\\data)")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to teacher checkpoint (.pth)")
    parser.add_argument("--save-dir", required=True, type=str, help="Output directory for eval artifacts")
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_eval(Path(args.checkpoint), args.dataset, Path(args.data_root), Path(args.save_dir), device)


if __name__ == "__main__":
    main()
