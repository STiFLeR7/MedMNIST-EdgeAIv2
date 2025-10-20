# external_src/robustness/eval_corruptions.py
from __future__ import annotations
import argparse, json, re, glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .corruptions import jpeg, gaussian, contrast, identity
from ..eval.metrics import macro_f1, accuracy


# --------------------------- Dataset ---------------------------

class HAMDataset(Dataset):
    """
    Minimal test dataset driven by a split JSON:
      {
        "test": [{"path": "<abs or relative to data root>", "label": int}, ...],
        "class_order": [...]
      }
    """
    def __init__(self, root: str, split_json: str, transform=None, corruption=None):
        self.root = Path(root)
        self.transform = transform
        self.corruption = corruption or (lambda im: im)
        data = json.loads(Path(split_json).read_text())
        self.samples = data["test"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]
        p = Path(rec["path"])
        if not p.is_file():
            p = self.root / rec["path"]
        img = Image.open(p).convert("RGB")
        img = self.corruption(img)
        if self.transform:
            img = self.transform(img)
        y = int(rec["label"])
        return img, y


# ---------------------- Corruption helpers ---------------------

def parse_levels(spec: str):
    """
    Parse level spec like:
      "gauss:0.1,0.2,0.3;jpeg:90,70,50;contrast:0.8,0.6"
    """
    out = []
    for block in spec.split(";"):
        block = block.strip()
        if not block:
            continue
        name, vals = block.split(":")
        vals = [v.strip() for v in vals.split(",") if v.strip()]
        out.append((name.strip(), vals))
    return out

def get_corruption(name: str, val: str):
    if name in ("gauss", "gaussian"):
        v = float(val)
        return (lambda im: gaussian(im, v)), f"gaussian_{v}"
    if name == "jpeg":
        v = int(val)
        return (lambda im: jpeg(im, v)), f"jpeg_{v}"
    if name == "contrast":
        v = float(val)
        return (lambda im: contrast(im, v)), f"contrast_{v}"
    if name in ("none", "identity"):
        return identity, "clean"
    raise KeyError(name)


# ------------------------ Model loading ------------------------

def _try_load_torchscript(ckpt: Path):
    try:
        m = torch.jit.load(str(ckpt), map_location="cpu")
        return m
    except Exception:
        return None

def load_model(ckpt: Path, device: str) -> torch.nn.Module:
    """
    Robust loader:
      1) Try TorchScript first (covers .pt/.ts that are scripted)
      2) Fallback to torch.load(weights_only=False) — trusted local checkpoints
         - If an nn.Module: use directly
         - Else raise with helpful message
    """
    m = _try_load_torchscript(ckpt)
    if m is None:
        obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if hasattr(obj, "eval") and callable(obj.eval):
            m = obj
        else:
            raise RuntimeError(
                f"Unsupported checkpoint format for {ckpt}. "
                f"Export as TorchScript or save full nn.Module."
            )
    m.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        m.to("cuda")
    else:
        m.to("cpu")
    return m


# ------------------------ Eval routine -------------------------

def run_eval(ckpt: Path, data_root: Path, split: Path, device: str, batch_size: int, corruption_fn, tag: str):
    model = load_model(ckpt, device=device)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = HAMDataset(str(data_root), str(split), transform=tfm, corruption=corruption_fn)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.startswith("cuda") and torch.cuda.is_available())
    )

    logits_all, labels_all = [], []
    dev = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    with torch.no_grad():
        for x, y in dl:
            x = x.to(dev, non_blocking=True)
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            logits_all.append(out.detach().cpu().float().numpy())
            labels_all.append(y.numpy())

    L = np.concatenate(logits_all, axis=0)
    Y = np.concatenate(labels_all, axis=0)
    return {
        "acc": float(accuracy(Y, L)),
        "macro_f1": float(macro_f1(Y, L)),
        "tag": tag,
    }


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-ckpt-glob", type=str, required=True,
                    help="glob for student checkpoints (accepts absolute paths; separate multiple with ';' or ',')")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--splits", type=str, required=True, help="split json for a specific seed")
    ap.add_argument("--levels", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fig", type=str, required=True)
    args = ap.parse_args()

    # Resolve absolute and multi-pattern globs
    patterns = [t.strip() for t in re.split(r"[;,]+", args.pred_ckpt_glob) if t.strip()]
    hits = []
    for pat in patterns:
        hits += glob.glob(pat)
    ckpts = sorted(map(Path, hits))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.pred_ckpt_glob}")

    levels = parse_levels(args.levels)

    rows = []
    for ck in ckpts:
        for name, vals in levels:
            for v in vals:
                corr, tag = get_corruption(name, v)
                r = run_eval(ck, Path(args.data), Path(args.splits), args.device, args.batch_size, corr, tag)
                r["model"] = ck.parents[1].name  # .../<model>/seed_*/best.pt
                r["seed"] = ck.parent.name
                rows.append(r)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Degradation plot (macro-F1 vs corruption level, averaged across models)
    import matplotlib.pyplot as plt
    plt.figure()

    def parse_level(t):
        try:
            return float(t.split("_")[-1])
        except Exception:
            return np.nan

    for cname in sorted(set(t.split("_")[0] for t in df["tag"].unique())):
        sub = df[df["tag"].str.startswith(cname)]
        lvls = sorted(sub["tag"].unique(), key=parse_level)
        xs = [parse_level(t) for t in lvls]
        ys = [sub[sub["tag"] == t]["macro_f1"].mean() for t in lvls]
        plt.plot(xs, ys, marker="o", label=cname)

    plt.xlabel("Corruption level")
    plt.ylabel("Macro-F1 (avg across models)")
    plt.legend()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.fig, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out} and {args.fig}")


if __name__ == "__main__":
    main()
