# external_src/loaders/rebuild_val_oct2017.py
import argparse, shutil, random, time
from pathlib import Path
from collections import defaultdict
from torchvision import datasets

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)   # e.g. data/OCT2017
    ap.add_argument("--ratio", type=float, default=0.10)  # 10% of train -> val
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--move", action="store_true", help="Move files (default). If not set, copies instead.")
    args=ap.parse_args()

    root = args.root
    train_dir, val_dir = root/"train", root/"val"
    assert train_dir.exists(), f"{train_dir} missing"
    (root/"_val_backup").mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if val_dir.exists():
        shutil.move(str(val_dir), str(root/"_val_backup"/f"val_prev_{ts}"))

    base = datasets.ImageFolder(str(train_dir))
    by_cls = defaultdict(list)
    for p,y in base.samples:
        by_cls[y].append(Path(p))

    random.seed(args.seed)
    new_val = []
    for y, lst in by_cls.items():
        random.shuffle(lst)
        k = max(1, int(len(lst)*args.ratio))
        new_val.extend((p,y) for p in lst[:k])

    # create fresh val folders
    val_dir.mkdir(parents=True, exist_ok=True)
    for cname in base.classes:
        (val_dir/cname).mkdir(parents=True, exist_ok=True)

    # move/copy files from train -> val (preserves train shrink)
    op = shutil.move if args.move else shutil.copy2
    for p,y in new_val:
        dst = val_dir/base.classes[y]/p.name
        op(str(p), str(dst))

    print(f"[OK] Built new val with {len(new_val)} images at {val_dir}")
    print(f"[INFO] Previous val backed up under: {root/'_val_backup'}")
    print("[HINT] Re-index next so class weights & previews stay correct.")

if __name__ == "__main__":
    main()
