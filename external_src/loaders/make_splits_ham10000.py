# external_src/loaders/make_splits_ham10000.py
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import pandas as pd
from collections import defaultdict

# HAM10000 canonical class order used by DermaMNIST
CLASS_ORDER = ["akiec","bcc","bkl","df","mel","nv","vasc"]
CLASS2ID = {c:i for i,c in enumerate(CLASS_ORDER)}

def infer_metadata_csv(root: Path) -> Path:
    for name in ["HAM10000_metadata.csv","metadata.csv","ham10000_metadata.csv"]:
        p = root / name
        if p.exists(): return p
    # fallback: any csv in root
    cands = list(root.glob("*.csv"))
    if cands: return cands[0]
    raise FileNotFoundError(f"No metadata CSV found under {root}")

def build_image_map(root: Path) -> dict[str, Path]:
    """
    Build {image_id: absolute_path} from:
      - root/images/*.jpg|png (if exists)
      - root/HAM10000_images_part_*/ *.jpg|png
      - root/*.jpg|png (rare)
    """
    exts = ("*.jpg","*.jpeg","*.png")
    pools = []

    img_dir = root / "images"
    if img_dir.exists():
        for ext in exts: pools += list(img_dir.glob(ext))

    for sub in root.glob("HAM10000_images_part_*"):
        if sub.is_dir():
            for ext in exts: pools += list(sub.glob(ext))

    for ext in exts: pools += list(root.glob(ext))

    if not pools:
        raise RuntimeError(
            f"No images found under {root}. Expected 'HAM10000_images_part_*/*.jpg' or 'images/*.jpg'."
        )

    imap: dict[str, Path] = {}
    for p in pools:
        stem = p.stem  # image_id should match this
        imap[stem] = p.resolve()
    return imap

def load_index(root: Path) -> pd.DataFrame:
    meta = infer_metadata_csv(root)
    df = pd.read_csv(meta)

    # Normalize column names
    if "image_id" not in df.columns:
        for alt in ["image","fname","file_name","file","lesion_id"]:  # wide net; 'lesion_id' is NOT image_id usually
            if alt in df.columns:
                df = df.rename(columns={alt:"image_id"})
                break
    if "dx" not in df.columns:
        for alt in ["label","diagnosis","class","target"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"dx"})
                break

    if "image_id" not in df.columns or "dx" not in df.columns:
        raise KeyError(f"metadata CSV must have 'image_id' and 'dx' columns. Found: {list(df.columns)}")

    # filter to known classes
    df = df[df["dx"].isin(CLASS2ID.keys())].copy()
    df["label"] = df["dx"].map(CLASS2ID)

    # Build filesystem map and attach paths
    imap = build_image_map(root)
    paths, keep = [], []
    for _, row in df.iterrows():
        pid = str(row["image_id"])
        p = imap.get(pid)
        if p is not None:
            paths.append(p.as_posix())
            keep.append(True)
        else:
            keep.append(False)

    df = df[keep].copy()
    df["path"] = paths
    df = df[["path","label","dx","image_id"]].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"No images resolved even after scanning folders under {root}")
    return df

def stratified_split(df: pd.DataFrame, train: float, val: float, test: float, seed: int):
    assert abs((train+val+test) - 1.0) < 1e-6, "train+val+test must sum to 1"
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i,row in df.iterrows():
        by_class[int(row["label"])].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for c, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(train * n))
        n_val   = int(round(val * n))
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val  = n - n_train
        train_idx += idxs[:n_train]
        val_idx   += idxs[n_train:n_train+n_val]
        test_idx  += idxs[n_train+n_val:]

    def pack(idxs):
        return [{"path": df.loc[i,"path"], "label": int(df.loc[i,"label"])} for i in idxs]

    return {
        "class_order": CLASS_ORDER,
        "train": pack(train_idx),
        "val":   pack(val_idx),
        "test":  pack(test_idx),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="data/HAM10000")
    ap.add_argument("--out", type=str, required=True, help="external_data/splits/HAM10000")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.2)
    args = ap.parse_args()

    root = Path(args.root)
    out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_index(root)

    for s in args.seeds:
        split = stratified_split(df, args.train, args.val, args.test, seed=s)
        (out / f"seed_{s}.json").write_text(json.dumps(split, indent=2))
        print(f"Wrote {out / f'seed_{s}.json'} "
              f"(n_train={len(split['train'])}, n_val={len(split['val'])}, n_test={len(split['test'])})")

if __name__ == "__main__":
    main()
