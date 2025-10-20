#!/usr/bin/env python3
"""
PTQ (FX static) for MobileNetV2 using HAM10000 val calibration.
Default paths and outputs are set for MedMNIST-EdgeAIv2 repo layout.

Save location:
external_src/students/ptq_mobilenetv2_fx_static.py

Example (uses defaults):
python external_src/students/ptq_mobilenetv2_fx_static.py

Or override:
python external_src/students/ptq_mobilenetv2_fx_static.py --checkpoint ./models/students/distilled_mobilenetv2_ham10000/ckpt-best.pth --val-dir ./data/HAM10000/val
"""
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_args():
    p = argparse.ArgumentParser(description="FX static PTQ for MobileNetV2 (HAM10000 calibration)")
    # defaults set to your path
    p.add_argument("--checkpoint", type=str,
                   default="./models/students/distilled_mobilenetv2_ham10000/ckpt-best.pth",
                   help="Path to student checkpoint (.pth)")
    p.add_argument("--val-dir", type=str, default="./data/HAM10000/val",
                   help="Path to HAM10000 val folder (ImageFolder style)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--calib-batches", type=int, default=128, help="Number of batches for calibration")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", type=str, default="./external_src/students/distilled_mobilenetv2_ham10000_int8.pt",
                   help="Output path for quantized model state_dict")
    p.add_argument("--jit-out", type=str, default="./external_src/students/distilled_mobilenetv2_ham10000_traced.pt",
                   help="Output path for TorchScript (jit)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                   help="Device for calibration/convert (FX convert runs on CPU)")
    return p.parse_args()

# ---------- Model utils ----------
def load_mobilenet_v2(checkpoint_path, device="cpu", num_classes=None):
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(pretrained=False)
    if num_classes is not None:
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    sd = torch.load(checkpoint_path, map_location=device)
    # support dict with 'state_dict' or DataParallel prefixes
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[new_k] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    model.eval()
    return model

def fuse_mobilenet_v2(model):
    """
    Best-effort fuse conv+bn(+relu) in MobileNetV2 (works with torchvision implementation).
    """
    for name, module in model.named_children():
        if name == "features":
            for idx in range(len(module)):
                sub = module[idx]
                # fuse simple Conv-BN-ReLU Sequential blocks
                if isinstance(sub, nn.Sequential):
                    names = list(sub._modules.keys())
                    if len(names) >= 2:
                        try:
                            if isinstance(sub._modules[names[0]], nn.Conv2d) and isinstance(sub._modules[names[1]], nn.BatchNorm2d):
                                # include relu if present
                                if len(names) >= 3 and isinstance(sub._modules[names[2]], (nn.ReLU6, nn.ReLU)):
                                    torch.quantization.fuse_modules(sub, [names[0], names[1], names[2]], inplace=True)
                                else:
                                    torch.quantization.fuse_modules(sub, [names[0], names[1]], inplace=True)
                        except Exception:
                            pass
                # handle InvertedResidual blocks
                from torchvision.models.mobilenet import InvertedResidual
                if isinstance(sub, InvertedResidual):
                    try:
                        for _, ssub in sub.conv.named_children():
                            if isinstance(ssub, nn.Sequential):
                                names2 = list(ssub._modules.keys())
                                if len(names2) >= 2 and isinstance(ssub._modules[names2[0]], nn.Conv2d) and isinstance(ssub._modules[names2[1]], nn.BatchNorm2d):
                                    try:
                                        if len(names2) >= 3 and isinstance(ssub._modules[names2[2]], (nn.ReLU6, nn.ReLU)):
                                            torch.quantization.fuse_modules(ssub, [names2[0], names2[1], names2[2]], inplace=True)
                                        else:
                                            torch.quantization.fuse_modules(ssub, [names2[0], names2[1]], inplace=True)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
    return model

# ---------- Dataset / calibration loader ----------
def build_val_loader(val_dir, img_size=224, batch_size=64, num_workers=4):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize
    ])
    ds = ImageFolder(val_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader

# ---------- Main PTQ flow ----------
def main():
    args = get_args()

    print("=> Loading model from:", args.checkpoint)
    model = load_mobilenet_v2(args.checkpoint, device="cpu")
    model.eval()

    print("=> Fusing modules (Conv+BN(+ReLU) where possible)")
    model = fuse_mobilenet_v2(model)

    # set quant engine for CPU (fbgemm recommended on x86)
    try:
        torch.backends.quantized.engine = "fbgemm"
    except Exception:
        pass
    print("Quantized engine:", torch.backends.quantized.engine)

    example_input = torch.randn(1, 3, args.img_size, args.img_size)

    # Build qconfig and prepare FX
    from torch.quantization import get_default_qconfig
    qconfig = get_default_qconfig(torch.backends.quantized.engine)
    qconfig_dict = {"": qconfig}

    print("=> Preparing FX graph (prepare_fx)")
    model_prepared = torch.quantization.quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=example_input)
    model_prepared.eval()

    print("=> Building val loader for calibration from:", args.val_dir)
    val_loader = build_val_loader(args.val_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"=> Calibrating (up to {args.calib_batches} batches)...")
    batches = 0
    with torch.no_grad():
        for images, _ in tqdm(val_loader, total=min(len(val_loader), args.calib_batches), desc="Calibration"):
            model_prepared(images)
            batches += 1
            if batches >= args.calib_batches:
                break
    print(f"=> Calibration completed (batches: {batches})")

    print("=> Converting to quantized model (convert_fx)")
    model_int8 = torch.quantization.quantize_fx.convert_fx(model_prepared)
    model_int8.eval()

    # quick sanity forward
    with torch.no_grad():
        out = model_int8(torch.randn(1, 3, args.img_size, args.img_size))
    print("=> Quantized forward OK, output shape:", out.shape)

    # Save state_dict and optional TorchScript traced model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model_int8.state_dict(),
                "meta": {"img_size": args.img_size, "quantized_engine": torch.backends.quantized.engine}},
               out_path)
    print("=> Saved quantized model state_dict to:", out_path)

    jit_out = Path(args.jit_out)
    jit_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("=> Attempting TorchScript scripting (script)")
        model_scripted = torch.jit.script(model_int8)
        torch.jit.save(model_scripted, str(jit_out))
        print("=> Saved TorchScript scripted model to:", jit_out)
    except Exception as e:
        print("=> scripting failed:", e)
        try:
            print("=> Trying torch.jit.trace fallback")
            traced = torch.jit.trace(model_int8, torch.randn(1, 3, args.img_size, args.img_size))
            torch.jit.save(traced, str(jit_out))
            print("=> Saved TorchScript traced model to:", jit_out)
        except Exception as e2:
            print("=> tracing also failed, skipping TorchScript save. Error:", e2)

    print("PTQ complete.")

if __name__ == "__main__":
    main()
