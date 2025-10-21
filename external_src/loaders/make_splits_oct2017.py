# Run: python make_splits_oct2017.py --root ../../data/OCT2017 --out ./data/processed --splits ../../external_data/splits/OCT2017
import argparse, hashlib, json, math, os, random
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image, UnidentifiedImageError

CLASSES = ["CNV","DME","DRUSEN","NORMAL"]
CLASS2ID = {c:i for i,c in enumerate(CLASSES)}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
SEEDS = list(range(5))  # seed_0..seed_4

def fhash(p, algo="md5", chunk=1<<20):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def scan(root: Path):
    rows=[]
    for split in ["train","val","test"]:
        for c in CLASSES:
            d = root/split/c
            if not d.exists(): continue
            for p in d.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in IMG_EXTS: continue
                w=h=-1
                try:
                    with Image.open(p) as im:
                        w,h = im.size
                except (UnidentifiedImageError, OSError):
                    pass
                rows.append({
                    "split": split,
                    "label_name": c,
                    "label": CLASS2ID[c],
                    "relpath": str(p.relative_to(root)),
                    "width": w, "height": h,
                    "md5": fhash(p,"md5"),
                    "sha1": fhash(p,"sha1")
                })
    return pd.DataFrame(rows)

def save_preview(df, root: Path, out: Path, split: str, n=64):
    from PIL import ImageOps
    sub = df[df.split==split]
    if sub.empty: return
    sel = sub.sample(min(n, len(sub)), random_state=2025)
    tiles=[]
    for rel in sel["relpath"]:
        p = root/rel
        try:
            im = Image.open(p).convert("L")
            im = ImageOps.equalize(im)
            im = im.resize((128,128))
            tiles.append(im)
        except Exception:
            pass
    if not tiles: return
    k = int(math.ceil(len(tiles)**0.5))
    canvas = Image.new("L", (k*128, k*128), 0)
    for i,im in enumerate(tiles):
        x = (i%k)*128; y=(i//k)*128
        canvas.paste(im, (x,y))
    out.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out/f"oct2017_preview_{split}.jpg", quality=92)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.splits.mkdir(parents=True, exist_ok=True)

    df = scan(args.root).sort_values(["split","label_name","relpath"]).reset_index(drop=True)
    # Save index
    df.to_parquet(args.out/"oct2017_index.parquet", index=False)

    # Stats
    counts = (df.groupby(["split","label_name"]).size()
                .rename("n").reset_index().to_dict(orient="records"))
    stats = {"class_map": {k:int(v) for k,v in CLASS2ID.items()}, "counts": counts}
    (args.out/"oct2017_stats.json").write_text(json.dumps(stats, indent=2))

    # Previews
    for sp in ["train","val","test"]:
        save_preview(df, args.root, args.out, sp, n=64)

    # Seed manifests: identical membership, deterministic shuffle
    for s in SEEDS:
        random.seed(1000 + s)
        man = {"seed": s, "splits": {}}
        for sp in ["train","val","test"]:
            sub = df[df.split==sp]["relpath"].tolist()
            rnd = sub[:]  # copy
            random.shuffle(rnd)
            man["splits"][sp] = rnd
        (args.splits/f"seed_{s}.json").write_text(json.dumps(man, indent=2))

    # Dataset card (minimal; filled by pipeline later)
    card = f"""# OCT2017 Dataset Card (Phase-2)
- Classes: {CLASSES}
- Root: {args.root}
- Counts: {counts}
- Notes: Images are effectively grayscale; keep 3-channel replicate to preserve ImageNet init consistency.
"""
    (args.out/"oct2017_dataset_card.md").write_text(card)
    print("[OK] index, stats, previews, and seed manifests emitted.")

if __name__ == "__main__":
    main()
