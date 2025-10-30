import os, csv, random
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image, ImageFile
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as F

# Make PIL robust to partial/truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Class-name registry
# ----------------------------
def class_names_for(dataset: str):
    dataset = dataset.lower()
    if dataset == "ham10000":
        return ["akiec","bcc","bkl","df","mel","nv","vasc"]  # 7
    if dataset == "isic":  # ISIC 2019 (8)
        return ["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
    if dataset == "oct2017":  # Retinal OCT 2017 (4)
        return ["CNV","DME","DRUSEN","NORMAL"]
    # MedMNIST dermaMNIST (7)
    return ["akiec","bcc","bkl","df","mel","nv","vasc"]

# ----------------------------
# Transforms
# ----------------------------
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def _default_transform_pil():
    # For PIL images
    return transforms.Compose([
        transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

def _tensor_from_path(path: str):
    """
    Fast path using torchvision.io.read_image (uint8 CxHxW) + normalization.
    Falls back to PIL if needed.
    """
    try:
        # read_image returns uint8 tensor [C,H,W]
        x = read_image(path, mode=ImageReadMode.RGB)  # RGB guaranteed
        # Resize (antialias) then convert to float [0,1] and normalize
        x = F.resize(x, [224,224], antialias=True).float().div(255.0)
        x = F.normalize(x, mean=_MEAN, std=_STD)
        return x
    except Exception:
        # Fallback to PIL route
        img = Image.open(path).convert("RGB")
        return _default_transform_pil()(img)

# ----------------------------
# CSV-backed dataset
# ----------------------------
class CSVImageDataset(Dataset):
    """
    CSV with columns: path,label  (label can be int or string).
    'path' may be absolute, repo-relative, or relative to dataset root.
    No double-prefixing: we detect if the row path already contains the root.
    """
    def __init__(self, root: str, csv_file: str):
        self.root = Path(root).as_posix()
        self.samples: List[Tuple[str,int]] = []

        with open(csv_file, "r", newline="") as f:
            rows = list(csv.DictReader(f))

        tmp: List[Tuple[str, object]] = []
        for row in rows:
            p = row.get("path") or row.get("image") or row.get("filepath")
            y = row.get("label") or row.get("target") or row.get("class")
            if p is None or y is None:
                continue
            tmp.append((self._join_under_root(p), y))

        # Map string labels deterministically
        if tmp and not isinstance(tmp[0][1], int):
            uniq = sorted({y for _, y in tmp})
            labmap = {s: i for i, s in enumerate(uniq)}
            self.samples = [(p, labmap[y]) for p, y in tmp]
        else:
            self.samples = [(p, int(y)) for p, y in tmp]

    def _join_under_root(self, p_in: str) -> str:
        p_in = p_in.replace("\\", "/")
        root_norm = self.root.replace("\\", "/")
        # Absolute path? keep
        if os.path.isabs(p_in): return p_in
        # Already under root? keep
        if p_in.startswith(root_norm + "/") or p_in == root_norm: return p_in
        # Repo-relative 'data/...' already including dataset folder? keep
        root_tail = Path(self.root).name
        if p_in.startswith("data/") and f"/{root_tail}/" in p_in: return p_in
        # Otherwise treat as relative to dataset root
        return (Path(self.root) / p_in).as_posix()

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        # Final guard against duplicated root once:
        dup = f"{self.root}/"
        if p.replace("\\","/").count(dup) > 1:
            p = p.replace(dup, "", 1)
            p = (Path(self.root) / p).as_posix()
        x = _tensor_from_path(p)
        return x, y

# ----------------------------
# ImageFolder fallback (case-insensitive)
# ----------------------------
def _imagefolder_split(rootdir: str, split: str):
    root = Path(rootdir)

    # exact match
    splitdir = root / split
    if splitdir.exists():
        return ImageFolder(splitdir.as_posix(), transform=_default_transform_pil())

    # common alternatives (case-insensitive)
    candidates = ["val","valid","validation","test","train"]
    for alt in candidates + [a.capitalize() for a in candidates] + [a.upper() for a in candidates]:
        d = root / alt
        if d.exists():
            return ImageFolder(d.as_posix(), transform=_default_transform_pil())

    raise FileNotFoundError(f"No split dir found under {rootdir} (expected {rootdir}/{split}/class/...)")

# ----------------------------
# MedMNIST: DermaMNIST
# ----------------------------
def _get_medmnist(split: str):
    try:
        from medmnist import DermaMNIST
        return DermaMNIST(split=split, transform=_default_transform_pil(), download=True)
    except Exception:
        return _imagefolder_split("./data/medmnist_derma", split)

# ----------------------------
# OCT2017 (Retinal OCT)
# ----------------------------
def _get_oct2017(split: str):
    root = "./data/OCT2017"
    csv_path = Path(root) / f"{split}.csv"
    if csv_path.exists():
        return CSVImageDataset(root, csv_path.as_posix())
    return _imagefolder_split(root, split)

# ----------------------------
# HAM10000 helpers (auto-split from metadata if needed)
# ----------------------------
def _resolve_ham_paths(root: Path) -> dict:
    paths = {}
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        d = root / part
        if not d.exists(): continue
        for p in d.glob("*.jpg"):
            img_id = p.stem
            rel = p.relative_to(root).as_posix()
            paths[img_id] = rel
    return paths

def _ensure_ham_csv_splits(root: Path, seed: int = 42, force: bool = True) -> None:
    train_csv = root / "train.csv"
    val_csv   = root / "val.csv"
    test_csv  = root / "test.csv"
    if (train_csv.exists() and val_csv.exists() and test_csv.exists()) and not force:
        return

    meta = root / "HAM10000_metadata.csv"
    if not meta.exists(): return

    df = pd.read_csv(meta)
    if "image_id" not in df.columns or "dx" not in df.columns:
        raise RuntimeError("HAM10000_metadata.csv must contain columns: image_id, dx")

    id2path = _resolve_ham_paths(root)
    df["path"] = df["image_id"].map(lambda s: id2path.get(str(s)))
    df = df[~df["path"].isna()].copy()

    valid = set(["akiec","bcc","bkl","df","mel","nv","vasc"])
    df = df[df["dx"].isin(valid)].copy()

    # Stratified 70/10/20 split
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1.0, random_state=rng)  # shuffle

    splits = {"train": [], "val": [], "test": []}
    for k, g in df.groupby("dx"):
        n = len(g)
        n_train = int(round(0.70 * n))
        n_val   = int(round(0.10 * n))
        n_test  = n - n_train - n_val
        splits["train"].append(g.iloc[:n_train])
        splits["val"].append(g.iloc[n_train:n_train+n_val])
        splits["test"].append(g.iloc[n_train+n_val:])

    for split in ["train","val","test"]:
        gg = pd.concat(splits[split], axis=0)
        rows = [{"path": r["path"], "label": r["dx"]} for _, r in gg.iterrows()]
        fp = root / f"{split}.csv"
        with open(fp.as_posix(), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path","label"])
            w.writeheader(); w.writerows(rows)
    print(f"[ham10000] wrote CSV splits under {root.as_posix()}")

def _get_ham10000(split: str):
    root = Path("./data/HAM10000")
    _ensure_ham_csv_splits(root, force=True)
    csv_path = root / f"{split}.csv"
    if csv_path.exists():
        return CSVImageDataset(root.as_posix(), csv_path.as_posix())
    return _imagefolder_split(root.as_posix(), split)

# ----------------------------
# Public API
# ----------------------------
def get_dataset_split(dataset: str, split: str = "test"):
    dataset = dataset.lower()
    if dataset == "medmnist":
        return _get_medmnist(split)
    if dataset == "ham10000":
        return _get_ham10000(split)
    if dataset == "isic":
        root = "./data/ISIC"
        csv_path = Path(root) / f"{split}.csv"
        if csv_path.exists():
            return CSVImageDataset(root, csv_path.as_posix())
        return _imagefolder_split(root, split)
    if dataset == "oct2017":
        return _get_oct2017(split)
    raise ValueError(f"Unsupported dataset: {dataset}")
