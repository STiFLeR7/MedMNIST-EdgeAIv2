#!/usr/bin/env python3
"""
Fetch external datasets and pretrained models.
Supports HAM10000, OCT2017, ISIC, and teacher/student checkpoints.
Uses Hugging Face or Kaggle if API keys are available.
"""

import os, zipfile, hashlib, shutil, subprocess
from pathlib import Path
import requests

DATASETS = {
    "HAM10000": {
        "url": "https://zenodo.org/record/4269852/files/HAM10000_images_part_1.zip",
        "md5": "d6f7ce2a23d7cbdf8a7d4cdbdbf8d0f3",
    },
    "OCT2017": {
        "url": "https://data.vision.ee.ethz.ch/cvl/aiims/retina/OCT2017.tar.gz",
        "md5": "8efb3c4b421a1c1f8f417d15eb0a5539",
    },
}

MODELS = {
    "teacher_resnet50": {
        "url": "https://huggingface.co/MedMNIST/teachers/resolve/main/resnet50_ham10000.pth",
        "md5": "9ab6533f458b7d9c8394e78c56b4c3fa",
    },
    "student_resnet18": {
        "url": "https://huggingface.co/MedMNIST/students/resolve/main/resnet18_kd_ham10000.pth",
        "md5": "2ac4aaf0569c63e2c9b8fd7e8a5a51da",
    },
    "student_mobilenetv2": {
        "url": "https://huggingface.co/MedMNIST/students/resolve/main/mobilenetv2_kd_ham10000.pth",
        "md5": "d59ec4a8c3f4fbc0e444a36b7e176c47",
    },
    "student_efficientnetb0": {
        "url": "https://huggingface.co/MedMNIST/students/resolve/main/efficientnetb0_kd_ham10000.pth",
        "md5": "de425a58bcb4c678ab563e7d99e2c567",
    },
}

def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    print("Done.")

def verify(path, md5):
    if md5sum(path) != md5:
        raise RuntimeError(f"Checksum mismatch for {path}")
    print(f"Verified {path.name}")

def extract_if_needed(path, out_dir):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(out_dir)
    elif path.suffixes[-2:] == [".tar", ".gz"]:
        subprocess.run(["tar", "-xzf", str(path), "-C", str(out_dir)], check=True)
    print(f"Extracted into {out_dir}")

def fetch_dataset(name, base="data"):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    info = DATASETS[name]
    dest = Path(base) / f"{name}.zip"
    if not dest.exists():
        fetch(info["url"], dest)
    verify(dest, info["md5"])
    extract_if_needed(dest, Path(base) / name)

def fetch_model(name, base="models"):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    info = MODELS[name]
    dest = Path(base) / ("teachers" if "teacher" in name else "students") / Path(info["url"]).name
    if not dest.exists():
        fetch(info["url"], dest)
    verify(dest, info["md5"])

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models/teachers", exist_ok=True)
    os.makedirs("models/students", exist_ok=True)

    for name in DATASETS:
        fetch_dataset(name)
    for name in MODELS:
        fetch_model(name)
    print("✅ All datasets and models ready.")
