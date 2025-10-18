#!/usr/bin/env python3
"""
Minimal HAM10000 loader smoke test.
Expects a folder with image files and a CSV metadata file (common HAM10000 layout).
Adapt as needed for your preprocessing pipeline (resize, augment, native-resolution).
"""
import argparse
import csv
from pathlib import Path
from PIL import Image

def inspect_dataset(data_dir):
    data_dir = Path(data_dir)
    csvs = list(data_dir.glob("*.csv"))
    imgs = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
    print(f"Found {len(csvs)} csv(s), {len(imgs)} images")
    if imgs:
        sample = imgs[0]
        im = Image.open(sample)
        print("Sample image:", sample, "size:", im.size, "mode:", im.mode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    inspect_dataset(args.data_dir)
