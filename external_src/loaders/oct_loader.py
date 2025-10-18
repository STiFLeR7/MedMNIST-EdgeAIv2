#!/usr/bin/env python3
"""
Minimal OCT loader smoke test.
Adapt to actual OCT folder layout and preprocessing.
"""
import argparse
from pathlib import Path
from PIL import Image

def inspect_dataset(data_dir):
    data_dir = Path(data_dir)
    imgs = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))
    print(f"Found {len(imgs)} images")
    if imgs:
        sample = imgs[0]
        im = Image.open(sample)
        print("Sample image:", sample, "size:", im.size, "mode:", im.mode)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    inspect_dataset(args.data_dir)
