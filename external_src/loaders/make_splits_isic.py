#!/usr/bin/env python3
# Run (example):
#   python external_src/loaders/make_splits_isic.py \
#       --root ../../data/ISIC \
#       --out ./data/processed \
#       --splits ../../external_data/splits/ISIC \
#       --val-frac 0.1

import argparse, hashlib, json, math, os, random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image, UnidentifiedImageError, ImageOps

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif",".webp"}
SEEDS = list(range(5))  # seed_0..seed_4

# --- Helpers ---

def fhash(p: Path, algo: str = "md5", chunk: int = 1<<20) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _norm_split_name(name: str) -> str:
    n = name.strip().lower()
    if n in {"train", "training"}: return "train"
    if n in {"val", "valid", "validation"}: return "val"
    if n in {"test", "testing"}: return "test"
    return name  # pass-through


def _find_split_dirs(root: Path) -> Dict[str, Path]:
    # Accept case-insensitive Train/Val/Test folder names
    out = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        s = _norm_split_name(child.name)
        if s in {"train","val","test"}:
            out[s] = child
    # Fallback: Some ISIC dumps only have Train and Test
    return out


# --- Scan ---

def scan(root: Path) -> Tuple[pd.DataFrame, List[str]]:
    split_dirs = _find_split_dirs(root)
    if not split_dirs:
        raise FileNotFoundError(f"No split folders found under {root}. Expect Train[/Val]/Test")

    # Infer classes from any present split (prefer Train, then Test, then Val)
    seed_split = split_dirs.get("train") or split_dirs.get("test") or split_dirs.get("val")
    classes = [d.name for d in sorted(seed_split.iterdir()) if d.is_dir()]
    class2id = {c:i for i,c in enumerate(classes)}

    rows = []
    for sp, sp_dir in split_dirs.items():
        for c in classes:
            d = sp_dir / c
            if not d.exists():
                continue
            for p in d.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                    continue
                w=h=-1
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except (UnidentifiedImageError, OSError):
                    pass
                rows.append({
                    "split_raw": sp,  # as found on disk
                    "label_name": c,
                    "label": class2id[c],
                    "relpath": str(p.relative_to(root)).replace("\\", "/"),
                    "width": w,
                    "height": h,
                    "md5": fhash(p, "md5"),
                    "sha1": fhash(p, "sha1"),
                })
    df = pd.DataFrame(rows).sort_values(["split_raw","label_name","relpath"]).reset_index(drop=True)
    return df, classes


# --- Val split builder when not provided ---

def build_val_membership(relpaths: List[str], labels: List[int], val_frac: float, seed: int) -> List[int]:
    """Return a list of 0/1 flags indicating membership in val given a pool (typically Train). Stratified per class.
    Deterministic for a given seed.
    """
    rnd = list(range(len(relpaths)))
    random.Random(1000 + seed).shuffle(rnd)
    # Group by label
    by_cls = defaultdict(list)
    for idx in rnd:
        by_cls[labels[idx]].append(idx)
    is_val = [0] * len(relpaths)
    for _, idxs in by_cls.items():
        k = max(1, int(round(len(idxs) * val_frac)))
        for i in idxs[:k]:
            is_val[i] = 1
    return is_val


# --- Preview grid ---

def save_preview(df: pd.DataFrame, root: Path, out: Path, split: str, n: int = 64, tag: str = "isic"):
    sub = df[df.split == split]
    if sub.empty:
        return
    sel = sub.sample(min(n, len(sub)), random_state=2025)
    tiles = []
    for rel in sel["relpath"]:
        p = root / rel
        try:
            im = Image.open(p).convert("RGB")
            im = ImageOps.equalize(im)
            im = im.resize((128, 128))
            tiles.append(im)
        except Exception:
            pass
    if not tiles:
        return
    k = int(math.ceil(len(tiles) ** 0.5))
    canvas = Image.new("RGB", (k * 128, k * 128), (0, 0, 0))
    for i, im in enumerate(tiles):
        x = (i % k) * 128
        y = (i // k) * 128
        canvas.paste(im, (x, y))
    out.mkdir(parents=True, exist_ok=True)
    canvas.save(out / f"{tag}_preview_{split}.jpg", quality=92)


# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1, help="If no Val split on disk, carve this fraction from Train per class")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.splits.mkdir(parents=True, exist_ok=True)

    df_raw, classes = scan(args.root)
    class2id = {c:i for i,c in enumerate(classes)}

    # If raw has explicit val, we honor it. Otherwise, build synthetic val for each seed from Train.
    has_val_on_disk = (df_raw["split_raw"] == "val").any()

    # Persist a raw index too (for traceability)
    df_raw.to_parquet(args.out / "isic_index_raw.parquet", index=False)

    # Precompute pools
    train_mask_raw = (df_raw["split_raw"] == "train")
    test_mask_raw = (df_raw["split_raw"] == "test")

    # For seed_0, create a canonical index with 'split' column
    if has_val_on_disk:
        df0 = df_raw.rename(columns={"split_raw": "split"}).copy()
    else:
        df0 = df_raw.copy()
        df0["split"] = df0["split_raw"].values
        # carve val from the train pool only
        pool = df0[train_mask_raw].reset_index(drop=True)
        is_val = build_val_membership(pool["relpath"].tolist(), pool["label"].tolist(), args.val_frac, seed=0)
        val_rel = set([pool.loc[i, "relpath"] for i, f in enumerate(is_val) if f == 1])
        # assign
        df0.loc[df0["relpath"].isin(val_rel), "split"] = "val"

    # Save canonical index
    df0 = df0.sort_values(["split","label_name","relpath"]).reset_index(drop=True)
    df0.to_parquet(args.out / "isic_index.parquet", index=False)

    # Stats (counts by split/class for canonical seed_0)
    counts = (
        df0.groupby(["split","label_name"]).size().rename("n").reset_index().to_dict(orient="records")
    )
    stats = {
        "class_map": {k:int(v) for k,v in class2id.items()},
        "counts": counts,
        "has_val_on_disk": bool(has_val_on_disk),
        "val_frac_if_synth": args.val_frac if not has_val_on_disk else None,
    }
    (args.out / "isic_stats.json").write_text(json.dumps(stats, indent=2))

    # Previews for the canonical seed_0 split view
    for sp in ["train","val","test"]:
        save_preview(df0, args.root, args.out, sp, n=64, tag="isic")

    # Seed manifests: identical membership to df0 if val exists on disk; otherwise, re-draw val per seed.
    for s in SEEDS:
        if has_val_on_disk:
            # simple deterministic shuffle only
            man = {"seed": s, "splits": {}}
            for sp in ["train","val","test"]:
                sub = df0[df0.split == sp]["relpath"].tolist()
                rnd = sub[:]
                random.Random(1000 + s).shuffle(rnd)
                man["splits"][sp] = rnd
        else:
            # re-carve val for each seed from the raw train pool
            man = {"seed": s, "splits": {}}
            train_pool = df_raw[train_mask_raw]
            is_val = build_val_membership(train_pool["relpath"].tolist(), train_pool["label"].tolist(), args.val_frac, seed=s)
            val_rel = set([train_pool.iloc[i]["relpath"] for i, f in enumerate(is_val) if f == 1])
            # Build per split lists
            for sp in ["train","val","test"]:
                if sp == "train":
                    rels = [r for r in train_pool["relpath"].tolist() if r not in val_rel]
                elif sp == "val":
                    rels = sorted(list(val_rel))
                else:  # test
                    rels = df_raw[test_mask_raw]["relpath"].tolist()
                rnd = rels[:]
                random.Random(1000 + s).shuffle(rnd)
                man["splits"][sp] = rnd
        (args.splits / f"seed_{s}.json").write_text(json.dumps(man, indent=2))

    # Dataset card (minimal; filled by Phase-3 pipeline later)
    card = f"""# ISIC Dataset Card (Phase-3)
- Root: {args.root}
- Classes (inferred): {classes}
- Has explicit Val on disk: {bool(has_val_on_disk)}
- Canonical counts (seed_0): {counts}
- Notes: Train/Test discovered from directory names ('Train'/'Test' acceptable). If Val not present, we carve per-class {args.val_frac:.0%} from Train for each seed.
"""
    (args.out / "isic_dataset_card.md").write_text(card)

    print("[OK] ISIC index, stats, previews, and seed manifests emitted.")


if __name__ == "__main__":
    main()
