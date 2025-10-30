import os, json, glob, re
from pathlib import Path
import numpy as np

# ----- YOU CAN TUNE THESE DEFAULTS -----
DEFAULTS = {
    "distilled": {"alpha":0.5, "tau":4.0, "beta":0.0},
    "kdat":      {"alpha":0.5, "tau":4.0, "beta":0.5},
}

def read_json_safe(p):
    try:
        with open(p,"r") as f: return json.load(f)
    except Exception:
        return None

def infer_ds_from_path(p):
    p = p.lower()
    for ds in ["ham10000","medmnist","isic","oct2017"]:
        if ds in p: return ds
    return None

def infer_student_from_path(p):
    p = p.lower()
    if "res18" in p or "resnet18" in p: return "resnet18"
    if "mbv2" in p or "mobilenetv2" in p: return "mbv2"
    if "effb0" in p or "efficientnetb0" in p: return "effb0"
    return None

def infer_hparams_from_path(p):
    p = p.lower()
    if "kdat" in p:
        d = DEFAULTS["kdat"].copy()
    elif "distilled" in p:
        d = DEFAULTS["distilled"].copy()
    else:
        d = DEFAULTS["distilled"].copy()

    # try to parse explicit a,t,b in names like a0.5_t4_b0.5
    m = re.search(r"a([0-9.]+)", p);  d["alpha"] = float(m.group(1)) if m else d["alpha"]
    m = re.search(r"t([0-9.]+)", p);  d["tau"]   = float(m.group(1)) if m else d["tau"]
    m = re.search(r"b([0-9.]+)", p);  d["beta"]  = float(m.group(1)) if m else d["beta"]
    return d

def macro_f1_from_metrics(metrics):
    # accept either scalar mac f1, or per-class report
    if "macro_f1" in metrics:
        return float(metrics["macro_f1"])
    # sometimes a per-class dict exists -> compute macro
    keys = [k for k in metrics.keys() if isinstance(metrics[k], dict) and "f1-score" in metrics[k]]
    if keys:
        vals = [metrics[k]["f1-score"] for k in keys if isinstance(metrics[k], dict) and "f1-score" in metrics[k]]
        if vals:
            return float(np.mean(vals))
    # last resort
    return float(metrics.get("f1", 0.0))

def ece_from_metrics(metrics):
    for k in ["ece","ECE","calibration_ece"]:
        if k in metrics: return float(metrics[k])
    return float("nan")

def main():
    out_root = Path("./reports")
    out_root.mkdir(exist_ok=True)
    # harvest all student metrics.json under models/students/**/eval/metrics.json OR final_summary.json
    files = []
    files += glob.glob("models/students/**/eval/metrics.json", recursive=True)
    files += glob.glob("models/students/**/metrics.json", recursive=True)
    files += glob.glob("models/students/**/final_summary.json", recursive=True)

    if not files:
        print("[INFO] No student metrics found.")
        return

    count = 0
    for fp in files:
        metrics = read_json_safe(fp)
        if metrics is None: 
            continue
        ds = infer_ds_from_path(fp)
        stu = infer_student_from_path(fp)
        if ds is None or stu is None:
            continue
        hp = infer_hparams_from_path(fp)
        macf1 = macro_f1_from_metrics(metrics)
        ece = ece_from_metrics(metrics)
        # write an ablation record
        out_dir = out_root/f"ablation_{ds}"/Path(fp).parent.name  # each run gets its own folder
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "epoch": metrics.get("epoch", metrics.get("best_epoch", 0)),
            "macro_f1": macf1,
            "ece": ece,
            "alpha": hp["alpha"], "tau": hp["tau"], "beta": hp["beta"],
            "student": stu, "dataset": ds,
            "source": fp
        }
        with open(out_dir/"metrics.jsonl", "a") as f:
            f.write(json.dumps(rec)+"\n")
        count += 1

    print(f"[OK] Backfilled {count} ablation records into ./reports/ablation_<ds>/**/metrics.jsonl")

if __name__ == "__main__":
    main()
