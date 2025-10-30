# tools/models.py
import os, re, json, glob, torch, torch.nn as nn
from pathlib import Path
import torchvision.models as tv

# ---------- Arch builders ----------
def build_arch(tag: str, num_classes: int):
    tag = tag.lower()
    if tag == "resnet50":
        m = tv.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if tag == "resnet18":
        m = tv.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if tag in ("mbv2", "mobilenet_v2"):
        m = tv.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if tag in ("effb0", "efficientnet_b0"):
        m = tv.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f"Unknown model tag: {tag}")

# ---------- Feature taps ----------
def pick_last_layer(model: nn.Module):
    # Works for torchvision resnet, mobilenet_v2, efficientnet_b0
    if hasattr(model, "layer4"):           # ResNet
        return model.layer4[-1]
    if hasattr(model, "features"):         # MobileNet/EfficientNet
        return list(model.features.children())[-1]
    # fallback: last conv
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d): last_conv = m
    if last_conv is None:
        raise RuntimeError("No conv layer found for Grad-CAM tap")
    return last_conv

@torch.no_grad()
def penultimate(model: nn.Module, x4d: torch.Tensor):
    """Return pre-classifier embedding."""
    model.eval()
    if hasattr(model, "layer4"):   # ResNet
        x = model.conv1(x4d); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x)
        z = torch.flatten(x, 1)    # [B,2048] or [B,512]
        return z
    # MobileNetV2 / EfficientNet
    if hasattr(model, "features"):
        x = model.features(x4d)
        if hasattr(model, "avgpool"):
            x = model.avgpool(x)
        else:
            x = nn.functional.adaptive_avg_pool2d(x, 1)
        z = torch.flatten(x, 1)    # [B,1280] for MBv2/EffB0
        return z
    # fallback: run to logits and return last hidden
    return model(x4d)

# ---------- Checkpoint discovery ----------
def _imagenet_weight_path(tag: str):
    root = Path("models")/"students"
    mapping = {
        "resnet50": "resnet50-0676ba61.pth",
        "resnet18": "resnet18-f37072fd.pth",
        "mbv2":     "mobilenet_v2-b0353104.pth",
        "effb0":    "efficientnet_b0_rwightman-3dd342df.pth",
    }
    fname = mapping.get(tag)
    if fname:
        p = root/fname
        if p.exists(): return p.as_posix()
    # teacher ImageNet weight also allowed under teachers
    if tag == "resnet50":
        p = Path("models")/"teachers"/"resnet50-0676ba61.pth"
        if p.exists(): return p.as_posix()
    return None

def _teacher_ckpt_candidates(dataset: str):
    root = Path("models")/"teachers"
    cand = []
    # dataset-specific fine-tuned teachers you showed
    cand += glob.glob((root/f"{dataset}_resnet50*/ckpt-best.pth").as_posix())
    cand += glob.glob((root/f"{dataset}_resnet50*/ckpt-last.pth").as_posix())
    # general
    cand += glob.glob((root/"**/ckpt-best.pth").as_posix(), recursive=True)
    cand += glob.glob((root/"**/ckpt-last.pth").as_posix(), recursive=True)
    return cand

def _student_ckpt_candidates(dataset: str, tag: str):
    root = Path("models")/"students"
    tag_norm = {
        "resnet18": "resnet18",
        "mbv2": "mobilenetv2",
        "effb0": "efficientnetb0",
    }[tag]
    cand = []
    # distilled_<arch>_<dataset>
    cand += glob.glob((root/f"distilled_{tag_norm}_{dataset}/ckpt-best.pth").as_posix())
    cand += glob.glob((root/f"distilled_{tag_norm}_{dataset}/ckpt-last.pth").as_posix())
    # *_kdat* variants you have (isic_kdat, oct2017_*_kdat_seed0, etc.)
    cand += glob.glob((root/f"*{dataset}*kdat*/ckpt-best.pth").as_posix())
    cand += glob.glob((root/f"*{dataset}*kdat*/ckpt-last.pth").as_posix())
    # generic (last resort)
    cand += glob.glob((root/"**/ckpt-best.pth").as_posix(), recursive=True)
    cand += glob.glob((root/"**/ckpt-last.pth").as_posix(), recursive=True)
    return cand

def _load_state_smart(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
    # strip common prefixes
    new_sd = {}
    for k,v in sd.items():
        nk = k
        nk = re.sub(r"^model\.", "", nk)
        nk = re.sub(r"^module\.", "", nk)
        nk = re.sub(r"^student\.", "", nk)
        new_sd[nk]=v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[load_state] missing keys: {len(missing)} (ok if only classifier)")
    if unexpected:
        print(f"[load_state] unexpected keys: {len(unexpected)}")
    return model

# ---------- Public: load_model_by_tag ----------
def load_model_by_tag(tag: str, dataset: str, device: str = "cuda"):
    from datasets import class_names_for
    tag = tag.lower(); dataset = dataset.lower()
    num_classes = len(class_names_for(dataset))

    model = build_arch(tag, num_classes)

    # 1) prefer fine-tuned teacher/student for dataset
    ckpt = None
    if tag == "resnet50":  # teacher
        t_cands = _teacher_ckpt_candidates(dataset)
        if t_cands:
            ckpt = t_cands[0]
    else:
        s_cands = _student_ckpt_candidates(dataset, tag)
        if s_cands:
            ckpt = s_cands[0]

    # 2) else fall back to ImageNet weights (if present)
    if ckpt is None:
        w = _imagenet_weight_path(tag)
        if w is not None:
            print(f"[info] Loading ImageNet weights for {tag}: {w}")
            state = torch.load(w, map_location="cpu")
            # torchvision weights are raw state_dict
            model = _load_state_smart(model, w)
        else:
            print(f"[warn] no checkpoint found for {tag}/{dataset}; using random init")
    else:
        print(f"[ckpt] {tag}/{dataset} -> {ckpt}")
        model = _load_state_smart(model, ckpt)

    model.to(device)
    model.eval()
    return model
