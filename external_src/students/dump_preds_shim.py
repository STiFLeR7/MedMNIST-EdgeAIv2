# external_src/students/dump_preds_shim.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------- Dataset (reads seed_X.json test split) --------
class HAMTestDS(Dataset):
    def __init__(self, data_root: str, split_json: str, img_size: int = 224):
        self.root = Path(data_root)
        spec = json.loads(Path(split_json).read_text())
        self.samples = spec["test"]  # list[{path,label}]
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]
        p = Path(rec["path"])
        if not p.is_file():
            p = self.root / rec["path"]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), int(rec["label"])

# -------- Model loader (TorchScript-first; PyTorch fallback) --------
def _try_load_torchscript(ckpt: Path):
    try:
        m = torch.jit.load(str(ckpt), map_location="cpu")
        return m
    except Exception:
        return None

def _is_state_dict(obj) -> bool:
    # Heuristic: dict with tensor leaves; not a scripted module
    if isinstance(obj, dict):
        # state_dict-like if at least one tensor leaf
        for v in obj.values():
            if torch.is_tensor(v) or (isinstance(v, dict) and any(torch.is_tensor(x) for x in v.values())):
                return True
    return False

def load_model(ckpt: Path, device: str = "cuda"):
    """
    Loading order:
      1) torch.jit.load (TorchScript archives, regardless of extension)
      2) torch.load(weights_only=False) → state_dict / full module
            - If 'model_state' in dict: try student's builder
            - Else if looks like a scripted module saved as .pt: try jit load again
    """
    # 1) TorchScript first (handles .pt/.ts that are archives)
    m = _try_load_torchscript(ckpt)
    if m is not None:
        m.eval()
        m.to(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        return m

    # 2) PyTorch object (allow code execution ONLY for trusted checkpoints)
    obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    if hasattr(obj, "eval") and callable(obj.eval):
        # Loaded a full nn.Module
        m = obj
        m.eval()
        m.to(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        return m

    if isinstance(obj, dict) and "model_state" in obj:
        sd = obj["model_state"]
        # Try to construct student model from your training code
        try:
            from external_src.students.train_student_kd import make_student_model
            # Infer num_classes from last linear weight
            num_classes = None
            for k, v in sd.items():
                if k.endswith("weight") and v.ndim == 2:
                    num_classes = int(v.shape[0])
                    break
            if num_classes is None:
                raise RuntimeError("Could not infer num_classes from state_dict.")
            m = make_student_model(num_classes=num_classes)
            m.load_state_dict(sd, strict=False)
            m.eval()
            m.to(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
            return m
        except Exception as e:
            raise RuntimeError(f"Found model_state but failed to rebuild student model for {ckpt}") from e

    if _is_state_dict(obj):
        # As a last resort, try TorchScript again (some .pt are TS but torch.load routed here)
        m = _try_load_torchscript(ckpt)
        if m is not None:
            m.eval()
            m.to(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
            return m
        raise RuntimeError(f"State-dict-like object loaded but no builder available for {ckpt}")

    # If we reach here, it's an unsupported format
    raise RuntimeError(f"Unsupported checkpoint type for {ckpt}")

# -------- Main: run inference and dump Parquet --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--data", required=True, type=str, help="data/HAM10000")
    ap.add_argument("--splits", required=True, type=str, help="external_data/splits/HAM10000/seed_X.json")
    ap.add_argument("--save-dir", required=True, type=str)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    ds = HAMTestDS(args.data, args.splits, img_size=args.imgsz)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    model = load_model(Path(args.ckpt), device=device)

    labels, logits, probs = [], [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            out = model(x)
            if isinstance(out, (tuple, list)): out = out[0]
            p = F.softmax(out, dim=1)
            labels.extend(y.numpy().tolist())
            logits.extend(out.detach().cpu().float().numpy().tolist())
            probs.extend(p.detach().cpu().float().numpy().tolist())

    df = pd.DataFrame({"labels": labels, "probs": probs, "logits": logits})
    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "test_preds.parquet", index=False)
    print("Wrote", outdir / "test_preds.parquet")

if __name__ == "__main__":
    main()
