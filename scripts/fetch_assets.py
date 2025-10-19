#!/usr/bin/env python3
"""
scripts/fetch_assets.py

Model-only downloader for MedMNIST-EdgeAIv2 with Hugging Face authentication.

Usage:
    python scripts/fetch_assets.py
    python scripts/fetch_assets.py --model teacher_resnet50
    python scripts/fetch_assets.py --hf-token <your_token>
"""
from pathlib import Path
import hashlib
import zipfile
import shutil
import subprocess
import requests
import argparse
import sys
import os

# optional tqdm import
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------
# MODELS CONFIG
# ---------------------------------------------------------------------
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

CHUNK_SIZE = 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_md5(path: Path, expected_md5: str) -> bool:
    if not expected_md5:
        return True
    got = md5sum(path)
    ok = got.lower() == expected_md5.lower()
    if ok:
        print(f"[OK] {path.name} (MD5 verified)")
    else:
        print(f"[WARN] {path.name} MD5 mismatch: expected {expected_md5}, got {got}")
    return ok


def extract_if_needed(path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(out_dir)
            print(f"[extract] {path.name} → {out_dir}")
        elif "".join(path.suffixes[-2:]) in [".tar.gz", ".tgz"]:
            subprocess.run(["tar", "-xzf", str(path), "-C", str(out_dir)], check=True)
            print(f"[extract] {path.name} → {out_dir}")
    except Exception as e:
        print(f"[extract] skip/failed for {path}: {e}")


def _locate_hf_token() -> str:
    """Locate HF token from CLI/env/cache files."""
    # 1) environment variable
    env = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HF_HOME")
    if env:
        return env

    # 2) common cache file locations
    candidates = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    for p in candidates:
        if p.exists():
            try:
                txt = p.read_text().strip()
                if txt:
                    return txt
            except Exception:
                pass
    return ""


def download_with_auth(url: str, dest: Path, token: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[HTTP] Downloading: {url}")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with requests.get(url, stream=True, headers=headers) as r:
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # give a friendly hint on auth failure
            if r.status_code == 401:
                raise RuntimeError("401 Unauthorized. Your Hugging Face token may not have permission to access this repo/file.")
            raise

        total = int(r.headers.get("content-length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=Path(url).name) if tqdm else None
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                if progress:
                    progress.update(len(chunk))
        tmp.replace(dest)
        if progress:
            progress.close()
    print(f"[HTTP] Saved → {dest}")


# ---------------------------------------------------------------------
# Model fetch (with HF auth)
# ---------------------------------------------------------------------
def fetch_model(name: str, models_dir: Path, hf_token: str):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    info = MODELS[name]
    url = info["url"]
    md5 = info.get("md5")
    subdir = info.get("subdir", "students")
    dest = models_dir / subdir / Path(url).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and md5 and verify_md5(dest, md5):
        print(f"[cached] {dest}")
        extract_if_needed(dest, models_dir / subdir)
        return

    download_with_auth(url, dest, token=hf_token)
    if md5 and not verify_md5(dest, md5):
        raise RuntimeError(f"MD5 verification failed for {dest}")
    extract_if_needed(dest, models_dir / subdir)
    print(f"[OK] Model ready: {dest}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Download Hugging Face model checkpoints with auth.")
    p.add_argument("--models-dir", default="models", help="Directory to save models")
    p.add_argument("--model", action="append", help="Model name to fetch (repeatable). If omitted, fetch all.")
    p.add_argument("--hf-token", help="Hugging Face token (overrides env/file lookup)")
    return p.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    (models_dir / "teachers").mkdir(parents=True, exist_ok=True)
    (models_dir / "students").mkdir(parents=True, exist_ok=True)

    # token resolution: CLI arg > env/file
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or _locate_hf_token()
    if not hf_token:
        print("ERROR: No Hugging Face token found. Provide with --hf-token or set HUGGINGFACE_TOKEN env or place token into ~/.cache/huggingface/token")
        print("You can also run: huggingface-cli login  (or `huggingface-cli login --token <token>`) to store token locally.")
        sys.exit(1)

    to_fetch = args.model if args.model else list(MODELS.keys())
    for name in to_fetch:
        print(f"\n=== Fetching model: {name} ===")
        try:
            fetch_model(name, models_dir, hf_token=hf_token)
        except Exception as e:
            print(f"[ERROR] {name}: {e}", file=sys.stderr)

    print("\n✅ Models fetch complete. Check the 'models/' folder.")


if __name__ == "__main__":
    main()
