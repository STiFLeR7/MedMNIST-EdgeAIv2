#!/usr/bin/env python3
"""
scripts/fetch_assets.py

Unified downloader for datasets and pretrained models
for MedMNIST-EdgeAIv2 using Kaggle API + Hugging Face.

- Downloads HAM10000, OCT2017, ISIC datasets from Kaggle.
- Downloads teacher/student checkpoints from Hugging Face.
- Extracts archives automatically.
- Shows tqdm progress bars for HTTP downloads.
- Designed for Windows + PowerShell + Kaggle CLI setup.

Author: MedMNIST-EdgeAIv2
"""

import subprocess
import hashlib
import zipfile
import shutil
from pathlib import Path
import requests
import sys
import os

# optional tqdm import
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATASETS = {
    "HAM10000": {"kaggle": "kmader/skin-cancer-mnist-ham10000"},
    "OCT2017": {"kaggle": "paultimothymooney/kermany2018"},
    "ISIC": {"kaggle": "nodoubttome/skin-cancer9-classesisic"},
}

MODELS = {
    "teacher_resnet50": {
        "url": "https://huggingface.co/MedMNIST/teachers/resolve/main/resnet50_ham10000.pth",
        "md5": "9ab6533f458b7d9c8394e78c56b4c3fa",
        "subdir": "teachers",
    },
    "student_resnet18": {
        "url": "https://huggingface.co/MedMNIST/students/resolve/main/resnet18_kd_ham10000.pth",
        "md5": "2ac4aaf0569c63e2c9b8fd7e8a5a51da",
        "subdir": "students",
    },
    "student_mobilenetv2": {
        "url": "https://huggingface.co/MedMNIST/students/resolve/main/mobilenetv2_kd_ham10000.pth",
        "md5": "d59ec4a8c3f4fbc0e444a36b7e176c47",
        "subdir": "students",
    },
    "student_efficientnetb0": {
        "url": "https://huggingface.co/MedMNIST/students/resolve/main/efficientnetb0_kd_ham10000.pth",
        "md5": "de425a58bcb4c678ab563e7d99e2c567",
        "subdir": "students",
    },
}

# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------

def md5sum(path: Path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_md5(path: Path, expected_md5: str) -> bool:
    """Verify file integrity via MD5."""
    if not expected_md5:
        return True
    got = md5sum(path)
    if got.lower() == expected_md5.lower():
        print(f"[OK] {path.name} (MD5 verified)")
        return True
    print(f"[WARN] {path.name} MD5 mismatch: expected {expected_md5}, got {got}")
    return False


def extract_archive(path: Path, out_dir: Path):
    """Extract ZIP or TAR archives into a directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(out_dir)
        elif "".join(path.suffixes[-2:]) in [".tar.gz", ".tgz"]:
            subprocess.run(["tar", "-xzf", str(path), "-C", str(out_dir)], check=True)
        else:
            return
        print(f"[extract] {path.name} → {out_dir}")
    except Exception as e:
        print(f"[extract] Failed to extract {path}: {e}")


def fetch_http(url: str, dest: Path):
    """HTTP(S) download with tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[HTTP] Downloading: {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB
    progress = tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                if progress:
                    progress.update(len(chunk))
    if progress:
        progress.close()
    print(f"[HTTP] Saved → {dest}")


# ---------------------------------------------------------------------
# KAGGLE LOGIC
# ---------------------------------------------------------------------

def check_kaggle_cli():
    """Ensure Kaggle CLI is available and API key exists."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except Exception:
        raise RuntimeError("Kaggle CLI not found. Run: pip install kaggle")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise RuntimeError(
            "Missing Kaggle API token. Please place kaggle.json in ~/.kaggle/"
        )
    print("[Kaggle] CLI detected and API key found.")


def fetch_kaggle(dataset_slug: str, out_dir: Path):
    """Fetch dataset via Kaggle CLI and extract zips."""
    print(f"[Kaggle] Downloading dataset: {dataset_slug}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(out_dir)],
            check=True,
        )
        for z in out_dir.glob("*.zip"):
            extract_archive(z, out_dir / dataset_slug.split("/")[-1])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e}")


# ---------------------------------------------------------------------
# MODEL DOWNLOAD LOGIC
# ---------------------------------------------------------------------

def fetch_model(name: str, base: Path):
    """Fetch pretrained model weights via HTTP."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    info = MODELS[name]
    url, md5, subdir = info["url"], info["md5"], info["subdir"]

    dest = base / subdir / Path(url).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and verify_md5(dest, md5):
        print(f"[cached] {dest.name}")
        return

    fetch_http(url, dest)
    if not verify_md5(dest, md5):
        raise RuntimeError(f"MD5 verification failed for {dest}")
    print(f"[OK] Model ready: {dest}")


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

def main():
    print("=== MedMNIST-EdgeAIv2 Asset Fetcher ===")
    base_data = Path("data")
    base_models = Path("models")
    base_data.mkdir(parents=True, exist_ok=True)
    (base_models / "teachers").mkdir(parents=True, exist_ok=True)
    (base_models / "students").mkdir(parents=True, exist_ok=True)

    # Datasets via Kaggle
    try:
        check_kaggle_cli()
        for name, info in DATASETS.items():
            print(f"\n=== Fetching dataset: {name} ===")
            fetch_kaggle(info["kaggle"], base_data)
    except Exception as e:
        print(f"[ERROR] Dataset fetch failed: {e}")
        print("Tip: ensure Kaggle CLI is installed and kaggle.json is configured.")

    # Models via HTTP
    for name in MODELS:
        print(f"\n=== Fetching model: {name} ===")
        try:
            fetch_model(name, base_models)
        except Exception as e:
            print(f"[ERROR] Model fetch failed for {name}: {e}")

    print("\n✅ All requested datasets and models are ready.")
    print("Check the 'data/' and 'models/' folders.")


if __name__ == "__main__":
    main()
