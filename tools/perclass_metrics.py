import argparse, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

def find_pred_file(reports_root, dataset, model_tag):
    """
    Search typical artifacts:
      - reports/phase*/<dataset>/*<model_tag>*/inference_cache/test_predictions.npz
      - reports/<dataset>/*<model_tag>*/inference_cache/test_predictions.npz
    """
    patt = [
        os.path.join(reports_root, f"phase*_{dataset}", f"*{model_tag}*", "inference_cache", "test_predictions.npz"),
        os.path.join(reports_root, f"{dataset}",        f"*{model_tag}*", "inference_cache", "test_predictions.npz"),
    ]
    hits = []
    for p in patt:
        hits.extend(sorted(glob.glob(p)))
    return hits[-1] if hits else None

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def plot_confmat(cm, class_names, out_path):
    fig = plt.figure(figsize=(8,7))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.tight_layout()
    plt.ylabel('True'); plt.xlabel('Pred')
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def plot_pr_curves(y_true_oh, y_prob, class_names, out_path):
    fig = plt.figure(figsize=(8,7))
    for c in range(y_true_oh.shape[1]):
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_prob[:, c])
        ap = average_precision_score(y_true_oh[:, c], y_prob[:, c])
        plt.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.title("One-vs-Rest Precision–Recall")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def class_names_for(dataset):
    if dataset == "ham10000":
        return ["akiec","bcc","bkl","df","mel","nv","vasc"]
    if dataset == "isic":
        return ["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
    return ["akiec","bcc","bkl","df","mel","nv","vasc"]

def sanitize_labels(y_true, num_classes):
    """Make sure labels are 0..C-1. Handle 1-based or arbitrarily encoded labels."""
    y = np.array(y_true).reshape(-1)
    # if labels are 1..C, shift
    if y.max() == num_classes and y.min() in (1, 0):  # e.g., 1..8 for C=8
        y = y - 1
    # if any label >= C, remap by order of occurrence
    if y.max() >= num_classes or y.min() < 0:
        uniq = np.unique(y)
        mapping = {lbl: i for i, lbl in enumerate(uniq[:num_classes])}
        y = np.array([mapping.get(v, 0) for v in y], dtype=np.int64)
    return y.astype(np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ham10000","medmnist","isic"])
    ap.add_argument("--reports-root", default="./reports")
    ap.add_argument("--models", required=True, help="Comma list: resnet50,resnet18,mbv2,effb0")
    ap.add_argument("--out-figs", default="./figs")
    ap.add_argument("--out-tables", default="./tables")
    ap.add_argument("--save-latex", action="store_true")
    args = ap.parse_args()

    classes = class_names_for(args.dataset)
    C = len(classes)

    for model_tag in args.models.split(","):
        pred_file = find_pred_file(args.reports_root, args.dataset, model_tag)
        if pred_file is None:
            print(f"[WARN] No predictions found for {args.dataset}/{model_tag}. Skipping.")
            continue

        data = np.load(pred_file)
        y_true = data["y_true"].reshape(-1)
        y_prob = data["y_prob"]
        # guard for length mismatch
        N = min(len(y_true), y_prob.shape[0])
        y_true = y_true[:N]; y_prob = y_prob[:N, :C]

        y_true = sanitize_labels(y_true, C)
        y_pred = y_prob.argmax(1)
        y_true_oh = np.eye(C, dtype=np.float32)[y_true]

        # per-class metrics
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        df = pd.DataFrame(report).transpose()
        tab_dir = os.path.join(args.out_tables, args.dataset)
        fig_dir = os.path.join(args.out_figs, args.dataset)
        safe_mkdir(tab_dir); safe_mkdir(fig_dir)

        base = os.path.join(tab_dir, f"{model_tag}_perclass_metrics")
        df.to_csv(base + ".csv", index=True)
        if args.save_latex:
            keep = classes + ["macro avg","weighted avg","accuracy"]
            df.loc[keep].to_latex(base + ".tex", float_format="%.3f")

        cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
        plot_confmat(cm, classes, os.path.join(fig_dir, f"{model_tag}_confmat.png"))
        plot_pr_curves(y_true_oh, y_prob[:, :C], classes, os.path.join(fig_dir, f"{model_tag}_pr_curves.png"))
        print(f"[OK] {args.dataset}/{model_tag}: per-class tables + confusion + PR saved. ({pred_file})")

if __name__ == "__main__":
    main()
