#!/usr/bin/env python3
"""
Log a reproducible environment fingerprint.
Writes python version, pip list, torch/torchvision versions (if available), GPU info.
"""
import json
import platform
import subprocess
import sys

def pip_freeze():
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return out.splitlines()
    except Exception:
        return []

def torch_info():
    try:
        import torch
        return {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda}
    except Exception:
        return {}

def nvidia_smi():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True)
        return out.strip().splitlines()
    except Exception:
        return []

meta = {
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "pip_freeze": pip_freeze(),
    "torch_info": torch_info(),
    "nvidia_smi": nvidia_smi(),
}

print(json.dumps(meta, indent=2))
