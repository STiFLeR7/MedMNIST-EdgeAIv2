#!/usr/bin/env python
# D:/MedMNIST-EdgeAIv2/train_and_eval.py
# KD trainer+evaluator with auto val split, sys monitor, robust ckpt loading,
# Grad-CAM export, failure-modes tables, and grid KD orchestration (ablations.json-aware).

import os, sys, time, json, argparse, random
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd

import psutil
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms, datasets

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm  # <-- progress bars

# speed knobs
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")  # allow TF32-friendly kernels
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"

# -----------------------------
# Utils
# -----------------------------
def seed_all(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def mkdir(p): os.makedirs(p, exist_ok=True)

def case_dir(root: str, name: str) -> Optional[str]:
    cand = os.path.join(root, name)
    if os.path.isdir(cand): return cand
    low = name.lower()
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)) and d.lower() == low:
            return os.path.join(root, d)
    return None

# -----------------------------
# Flexible checkpoint loader
# -----------------------------
def _select_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["model_state","state_dict","model","net","weights"]:
            if k in obj and isinstance(obj[k], dict): return obj[k]
        if any(isinstance(v, torch.Tensor) for v in list(obj.values())[:10]):
            return obj
    raise ValueError("No state_dict-like object found in checkpoint")

def _strip_prefix(sd: dict, prefixes=("module.","backbone.","model.")):
    out = {}
    for k,v in sd.items():
        nk=k
        for px in prefixes:
            if nk.startswith(px): nk=nk[len(px):]
        out[nk]=v
    return out

def _head_keys_for_arch(arch: str):
    a = arch.lower()
    if a.startswith("resnet"): return ["fc.weight","fc.bias"]
    if a in ["mobilenet_v2","mbv2","efficientnet_b0","effb0","efficientnet-b0"]:
        return ["classifier.1.weight","classifier.1.bias"]
    return []

def load_checkpoint_flex(model: nn.Module, arch: str, ckpt_path: str, strict_head: bool=False) -> dict:
    dev = next(model.parameters()).device
    raw = torch.load(ckpt_path, map_location=dev)
    sd = _strip_prefix(_select_state_dict(raw))
    head_keys = set(_head_keys_for_arch(arch))
    model_sd = model.state_dict()
    loadable, skipped = {}, {}
    for k,v in sd.items():
        if k not in model_sd: skipped[k]="missing_in_model"; continue
        if model_sd[k].shape != v.shape:
            if (k in head_keys) and (not strict_head):
                skipped[k]=f"shape_mismatch {tuple(v.shape)} -> {tuple(model_sd[k].shape)} (drop head)"; continue
            else:
                skipped[k]=f"shape_mismatch {tuple(v.shape)} -> {tuple(model_sd[k].shape)}"; continue
        loadable[k]=v
    msg = model.load_state_dict(loadable, strict=False)
    return {
        "loaded": len(loadable),
        "skipped": len(skipped),
        "missing_in_ckpt": len([k for k in model_sd.keys() if k not in sd]),
        "skipped_detail": skipped,
        "torch_msg": str(msg),
    }

# -----------------------------
# System monitor
# -----------------------------
class SystemMonitor:
    def __init__(self, out_csv: str, interval: float=1.0):
        self.out_csv = out_csv; self.interval = interval
        self._fh = None; self._stop = False; self._gpu_handles = []
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                for i in range(pynvml.nvmlDeviceGetCount()):
                    self._gpu_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            except Exception: self._gpu_handles = []

    def start(self):
        mkdir(os.path.dirname(self.out_csv))
        self._fh = open(self.out_csv,"w",encoding="utf-8")
        self._fh.write("ts,cpu_pct,ram_used_gb,ram_total_gb,ram_pct,proc_mem_gb,proc_mem_pct,gpu_index,gpu_util,gpu_mem_used_mb,gpu_mem_total_mb\n")

    def step(self):
        if self._stop: return
        try:
            ts = now()
            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_used_gb = vm.used/(1024**3); ram_total_gb = vm.total/(1024**3); ram_pct = vm.percent
            pm = psutil.Process(os.getpid()).memory_info().rss/(1024**3)
            pmp = (pm*(1024**3))/vm.total*100.0
            if self._gpu_handles:
                for idx,h in enumerate(self._gpu_handles):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                        mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
                        row = [ts,cpu_pct,ram_used_gb,ram_total_gb,ram_pct,pm,pmp,idx,util.gpu,mem.used/1024**2,mem.total/1024**2]
                        self._fh.write(",".join(str(x) for x in row)+"\n")
                    except Exception: pass
            else:
                self._fh.write(",".join(str(x) for x in [ts,cpu_pct,ram_used_gb,ram_total_gb,ram_pct,pm,pmp,"","","",""])+"\n")
            self._fh.flush()
        except Exception: pass

    def stop(self):
        self._stop = True
        try:
            if self._fh: self._fh.flush(); self._fh.close()
        except Exception: pass
        if _NVML_OK:
            try: pynvml.nvmlShutdown()
            except Exception: pass

# -----------------------------
# Transforms
# -----------------------------
IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD  = [0.229,0.224,0.225]

def build_transforms(img_size=224):
    return {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "eval": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    }

def denorm_img(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=t.device)[:,None,None]
    std  = torch.tensor(IMAGENET_STD,  device=t.device)[:,None,None]
    x = (t*std + mean).clamp(0,1).detach().cpu().numpy()
    x = np.transpose(x, (1,2,0))
    x = (x*255.0).round().astype(np.uint8)
    return x

# -----------------------------
# Dataset helpers (ImageFolder + HAM CSV)
# -----------------------------
HAM_CLASSES = ["akiec","bcc","bkl","df","nv","mel","vasc"]
HAM_DX_TO_IDX = {k:i for i,k in enumerate(HAM_CLASSES)}

class ImageCSV(Dataset):
    def __init__(self, csv_path: str, root_folders: List[str], transform=None, class_map: Optional[Dict[str,int]]=None, force_ext: Optional[str]=None):
        self.df = pd.read_csv(csv_path)
        self.root_folders = root_folders
        self.transform = transform
        self.force_ext = force_ext
        self.class_map = class_map or {}
        cols = {c.lower(): c for c in self.df.columns}
        if "path" in cols and ("label" in cols or "y" in cols or "target" in cols):
            self.path_col = cols["path"]; self.label_col = cols.get("label", cols.get("y", cols.get("target"))); self.mode = "path_label"
        elif "image" in cols and ("dx" in cols or "label" in cols):
            self.path_col = cols["image"]; self.label_col = cols.get("dx", cols.get("label")); self.mode = "image_dx"
        else:
            raise ValueError(f"CSV {csv_path} must have (path,label) or (image,dx)")
        labs = []
        for v in self.df[self.label_col].tolist():
            if isinstance(v, str):
                if v in self.class_map: labs.append(self.class_map[v])
                elif v.lower() in self.class_map: labs.append(self.class_map[v.lower()])
                elif v in HAM_DX_TO_IDX: labs.append(HAM_DX_TO_IDX[v])
                elif v.lower() in HAM_DX_TO_IDX: labs.append(HAM_DX_TO_IDX[v.lower()])
                else: raise ValueError(f"Unknown label string: {v}")
            else: labs.append(int(v))
        self.labels = np.array(labs, dtype=np.int64)

        self.paths = []
        for raw in self.df[self.path_col].tolist():
            p = str(raw)
            if os.path.exists(p): self.paths.append(p); continue
            name = os.path.basename(p)
            if self.force_ext and not name.lower().endswith(self.force_ext):
                name = f"{name}{self.force_ext}"
            chosen = None
            for r in self.root_folders:
                cand = os.path.join(r, name)
                if os.path.exists(cand): chosen = cand; break
            if not chosen:
                for r in self.root_folders:
                    for root,_,files in os.walk(r):
                        if name in files: chosen = os.path.join(root, name); break
                    if chosen: break
            if not chosen: raise FileNotFoundError(f"Cannot resolve image {raw} under roots {self.root_folders}")
            self.paths.append(chosen)

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, int(self.labels[idx]), self.paths[idx]

class WithPaths(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
        self.is_subset = isinstance(base, Subset)
        self.is_imgcsv = isinstance(base, ImageCSV)
        if self.is_subset and isinstance(base.dataset, datasets.ImageFolder):
            self.kind = "subset_imgfolder"
        elif isinstance(base, datasets.ImageFolder):
            self.kind = "imgfolder"
        elif self.is_imgcsv:
            self.kind = "imagecsv"
        else:
            self.kind = "unknown"

    def __len__(self): return len(self.base)

    def _get_path(self, idx):
        if self.kind == "imgfolder":
            return self.base.samples[idx][0]
        if self.kind == "subset_imgfolder":
            base_idx = self.base.indices[idx]
            return self.base.dataset.samples[base_idx][0]
        if self.kind == "imagecsv":
            return self.base.paths[idx]
        return f"idx_{idx}.png"

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.kind == "imagecsv":
            img, y, p = item; return img, y, p
        else:
            img, y = item; return img, y, self._get_path(idx)

def build_datasets(data_root: str, dataset_key: str, img_size: int, val_ratio: float, val_seed: int):
    ds_root = os.path.join(data_root, dataset_key)
    tfm = build_transforms(img_size)

    if dataset_key.lower() in ["oct2017","medmnist","oct"]:
        tr = case_dir(ds_root,"train"); va = case_dir(ds_root,"val"); te = case_dir(ds_root,"test")
        if tr and te:
            ds_tr_full = datasets.ImageFolder(tr, tfm["train"]); classes = ds_tr_full.classes
            if not va:
                idx = np.arange(len(ds_tr_full))
                y = np.array([ds_tr_full.samples[i][1] for i in idx])
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=val_seed)
                tr_idx, va_idx = next(splitter.split(idx, y))
                ds_tr = Subset(ds_tr_full, tr_idx.tolist())
                ds_va = Subset(datasets.ImageFolder(tr, tfm["eval"]), va_idx.tolist())
            else:
                ds_tr = ds_tr_full; ds_va = datasets.ImageFolder(va, tfm["eval"])
            ds_te = datasets.ImageFolder(te, tfm["eval"])
            return classes, ds_tr, ds_va, ds_te
        else:
            raise RuntimeError(f"OCT2017 expects train/(val)/test in {ds_root}")

    if dataset_key.lower() == "isic":
        tr = case_dir(ds_root,"Train") or case_dir(ds_root,"train")
        te = case_dir(ds_root,"Test")  or case_dir(ds_root,"test")
        if not tr or not te: raise RuntimeError(f"ISIC expects Train/ and Test/ under {ds_root}")
        ds_tr_full = datasets.ImageFolder(tr, tfm["train"]); classes = ds_tr_full.classes
        idx = np.arange(len(ds_tr_full))
        y = np.array([ds_tr_full.samples[i][1] for i in idx])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=val_seed)
        tr_idx, va_idx = next(splitter.split(idx, y))
        ds_tr = Subset(ds_tr_full, tr_idx.tolist())
        ds_va = Subset(datasets.ImageFolder(tr, tfm["eval"]), va_idx.tolist())
        ds_te = datasets.ImageFolder(te, tfm["eval"])
        return classes, ds_tr, ds_va, ds_te

    if dataset_key.lower() == "ham10000":
        tr = case_dir(ds_root,"train"); va = case_dir(ds_root,"val"); te = case_dir(ds_root,"test")
        if tr and te:
            ds_tr_full = datasets.ImageFolder(tr, tfm["train"]); classes = ds_tr_full.classes
            if not va:
                idx = np.arange(len(ds_tr_full))
                y = np.array([ds_tr_full.samples[i][1] for i in idx])
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=val_seed)
                tr_idx, va_idx = next(splitter.split(idx, y))
                ds_tr = Subset(ds_tr_full, tr_idx.tolist())
                ds_va = Subset(datasets.ImageFolder(tr, tfm["eval"]), va_idx.tolist())
            else:
                ds_tr = ds_tr_full; ds_va = datasets.ImageFolder(va, tfm["eval"])
            ds_te = datasets.ImageFolder(te, tfm["eval"])
            return classes, ds_tr, ds_va, ds_te
        else:
            img_roots = [
                os.path.join(ds_root,"HAM10000_images_part_1"),
                os.path.join(ds_root,"HAM10000_images_part_2"),
            ]
            train_csv = os.path.join(ds_root,"train.csv")
            val_csv   = os.path.join(ds_root,"val.csv")
            test_csv  = os.path.join(ds_root,"test.csv")
            if not os.path.exists(train_csv): raise RuntimeError(f"HAM10000: train.csv missing under {ds_root}")
            if not os.path.exists(val_csv):
                df = pd.read_csv(train_csv)
                label_col = None
                for cand in ["label","labels","y","target","dx"]:
                    if cand in df.columns: label_col = cand; break
                if label_col is None: raise RuntimeError("train.csv must have one of label/labels/y/target/dx")
                y = df[label_col]; idx = np.arange(len(df))
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=val_seed)
                tr_idx, va_idx = next(sss.split(idx, y))
                df_tr = df.iloc[tr_idx]; df_va = df.iloc[va_idx]
                tmp_dir = os.path.join(ds_root,"_auto_split"); mkdir(tmp_dir)
                df_va.to_csv(os.path.join(tmp_dir,"val.csv"), index=False)
                df_tr.to_csv(os.path.join(tmp_dir,"train.csv"), index=False)
                train_csv = os.path.join(tmp_dir,"train.csv"); val_csv = os.path.join(tmp_dir,"val.csv")
            class_map = {k:i for i,k in enumerate(HAM_CLASSES)}
            ds_tr = ImageCSV(train_csv, img_roots, transform=build_transforms(img_size)["train"], class_map=class_map, force_ext=".jpg")
            ds_va = ImageCSV(val_csv,   img_roots, transform=build_transforms(img_size)["eval"],  class_map=class_map, force_ext=".jpg")
            ds_te = ImageCSV(test_csv,  img_roots, transform=build_transforms(img_size)["eval"],  class_map=class_map, force_ext=".jpg")
            return HAM_CLASSES, ds_tr, ds_va, ds_te

    raise ValueError(f"Unknown dataset_key: {dataset_key}")

# -----------------------------
# Models + KD/AT
# -----------------------------
def build_model(arch: str, num_classes: int, pretrained: bool=True):
    a = arch.lower()
    if a == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if a == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if a in ["mobilenet_v2","mbv2","mobilenetv2"]:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    if a in ["efficientnet_b0","effb0","efficientnet-b0"]:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    raise ValueError(f"Unknown arch: {arch}")

def extract_attention_maps(model: nn.Module, arch: str, taps: int = 3):
    """Return `taps` feature maps from backbone for AT. Use fewer taps to speed up AT."""
    feats=[]; handles=[]
    def hook(_, __, output): feats.append(output)
    a = arch.lower()
    if a.startswith("resnet"):
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        for layer in layers[-taps:]:
            handles.append(layer.register_forward_hook(hook))
    elif "mobilenet" in a or "efficientnet" in a:
        if hasattr(model,"features"):
            L=len(model.features)
            idxs = list(range(max(0, L - taps), L))
            for idx in idxs:
                handles.append(model.features[idx].register_forward_hook(hook))
    return feats, handles

def attention_transfer_loss(
    student_feats: List[torch.Tensor],
    teacher_feats: List[torch.Tensor],
    beta: float=1000.0,
    device="cpu"
):
    if not student_feats or not teacher_feats or beta <= 0:
        return torch.tensor(0.0, device=device)

    def att(f: torch.Tensor) -> torch.Tensor:
        a = (f.float()**2).mean(dim=1, keepdim=True)     # [N,1,H,W]
        a = a / a.norm(p=2, dim=(2,3), keepdim=True).clamp_min(1e-8)
        return a

    loss = 0.0
    n = min(len(student_feats), len(teacher_feats))
    for i in range(n):
        sa = att(student_feats[i])
        ta = att(teacher_feats[i]).detach()
        _, _, Hs, Ws = sa.shape
        _, _, Ht, Wt = ta.shape
        Hc, Wc = min(Hs, Ht), min(Ws, Wt)
        if (Hs != Hc) or (Ws != Wc):
            sa = F.interpolate(sa, size=(Hc, Wc), mode="bilinear", align_corners=False)
        if (Ht != Hc) or (Wt != Wc):
            ta = F.interpolate(ta, size=(Hc, Wc), mode="bilinear", align_corners=False)
        loss += F.mse_loss(sa, ta)

    return beta * loss / float(n)

# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, arch: str, device: torch.device):
        self.model = model; self.arch = arch; self.device = device
        self.fmap=None; self.grad=None; self.handle_f=None; self.handle_b=None
        target = self._target_module(model, arch)
        self.handle_f = target.register_forward_hook(self._forward_hook)
        def _backward_hook(module, grad_input, grad_output):
            self.grad = grad_output[0]
        self.handle_b = target.register_full_backward_hook(_backward_hook)

    def _target_module(self, model, arch):
        a = arch.lower()
        if a.startswith("resnet"): return model.layer4
        if "mobilenet" in a or a=="mbv2" or a=="mobilenetv2": return model.features[-1]
        if "efficientnet" in a or a=="effb0" or a=="efficientnet-b0": return model.features[-1]
        raise ValueError(f"GradCAM: unsupported arch {arch}")

    def _forward_hook(self, module, inp, out):
        self.fmap = out

    def remove(self):
        try:
            if self.handle_f: self.handle_f.remove()
            if self.handle_b: self.handle_b.remove()
        except Exception: pass

    def generate(self):
        assert self.fmap is not None and self.grad is not None, "Run a forward+backward pass first"
        grad = self.grad.detach()
        fmap = self.fmap.detach()
        weights = grad.mean(dim=(2,3), keepdim=True)
        cam = (weights * fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        N,_,H,W = cam.shape
        cam = cam.view(N,-1)
        cam -= cam.min(dim=1, keepdim=True).values
        denom = cam.max(dim=1, keepdim=True).values + 1e-8
        cam = (cam/denom).view(N,1,H,W)
        return cam

def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float=0.35) -> np.ndarray:
    h,w,_ = rgb.shape
    cam_resized = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_resized, size=(h,w), mode="bilinear", align_corners=False)[0,0].cpu().numpy()
    cmap = plt.get_cmap("jet")
    heat = (cmap(cam_resized)[:,:,:3]*255).astype(np.uint8)
    over = (alpha*heat + (1-alpha)*rgb).astype(np.uint8)
    return over

# -----------------------------
# DataLoader & loops
# -----------------------------
def _wrap_loader_with_paths(ds, batch_size, shuffle, workers):
    ds_wrapped = WithPaths(ds)
    return DataLoader(
        ds_wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=(4 if workers > 0 else None),
    )

def _print_every(loader_len: int) -> int:
    return max(1, loader_len // 10)

@torch.no_grad()
def _eval_pass(model, loader, device, num_classes):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss=0.0; ys=[]; ps=[]; probs=[]
    # tqdm bar for eval
    for x,y,_paths in tqdm(loader, desc="eval", ncols=100):
        x=x.to(device, non_blocking=True)
        if x.is_cuda: x = x.to(memory_format=torch.channels_last)
        y=y.to(device, non_blocking=True)
        logits = model(x); total_loss += ce(logits, y).item()
        ys.append(y.detach().cpu().numpy())
        ps.append(logits.argmax(dim=1).detach().cpu().numpy())
        probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps); p = np.concatenate(probs)
    avg_loss = total_loss/len(loader.dataset)
    acc = float((y_true==y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return y_true, y_pred, p, cm, acc, avg_loss

def _save_metrics_and_confmats(classes, cm, y_true, y_pred, acc, avg_loss, out_tables, out_figs, tag):
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    df_pc = pd.DataFrame({"class": classes, "precision": prec, "recall": rec, "f1": f1, "support": supp})
    mkdir(out_tables); mkdir(out_figs)
    df_pc.to_csv(os.path.join(out_tables, f"perclass_{tag}.csv"), index=False)

    rpt = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    micro_f1 = float(rpt.get("micro avg", {}).get("f1-score", acc))
    df_sum = pd.DataFrame({
        "metric": ["accuracy","macro_f1","macro_precision","macro_recall","weighted_f1","micro_f1","avg_ce_loss"],
        "value": [float(acc),
                  float(rpt["macro avg"]["f1-score"]),
                  float(rpt["macro avg"]["precision"]),
                  float(rpt["macro avg"]["recall"]),
                  float(rpt["weighted avg"]["f1-score"]),
                  micro_f1,
                  float(avg_loss)]
    })
    df_sum.to_csv(os.path.join(out_tables, f"summary_{tag}.csv"), index=False)

    def plot_confmat(cm, classes, out_path, normalize=False, title=""):
        mat = cm.astype(float)
        if normalize:
            rs = mat.sum(axis=1, keepdims=True); rs[rs==0]=1.0; mat = mat/rs
        plt.figure(figsize=(7,6), dpi=180)
        plt.imshow(mat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        if title: plt.title(title)
        plt.colorbar()
        ticks = np.arange(len(classes))
        plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
        fmt = ".2f" if normalize else ".0f"
        thr = mat.max()/2.0 if mat.size else 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i,j]
                plt.text(j, i, format(v, fmt), ha="center", va="center",
                         color="white" if v > thr else "black", fontsize=8)
        plt.ylabel("True"); plt.xlabel("Pred")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight"); plt.close()

    title_base = tag.replace("_"," / ")
    plot_confmat(cm, classes, os.path.join(out_figs, f"confmat_{tag}.png"), False, f"{title_base} — Confusion")
    plot_confmat(cm, classes, os.path.join(out_figs, f"confmat_norm_{tag}.png"), True,  f"{title_base} — Confusion (Row-Norm)")

def _failure_modes_table(classes, cm: np.ndarray, topn: int, out_tables: str, tag: str):
    rows=[]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums==0]=1
    norm = cm/row_sums
    C = cm.shape[0]
    for i in range(C):
        pairs = [(j, int(cm[i,j]), float(norm[i,j])) for j in range(C) if j!=i]
        pairs.sort(key=lambda t: (t[1], t[2]), reverse=True)
        for rank,(j, count, frac) in enumerate(pairs[:topn], start=1):
            rows.append({"true_class": classes[i], "pred_class": classes[j], "count": count, "frac_row": round(frac,6), "rank": rank})
    pd.DataFrame(rows).to_csv(os.path.join(out_tables, f"failure_modes_{tag}.csv"), index=False)

def _collect_misclass_indices(y_true, y_pred, paths, per_class_k):
    wrong = np.where(y_true!=y_pred)[0]
    by_true: Dict[int, List[Tuple[int,str,int,int]]] = {}
    for i in wrong:
        t = int(y_true[i]); p = int(y_pred[i])
        by_true.setdefault(t, []).append((i, paths[i], t, p))
    chosen = {c: lst[:per_class_k] for c, lst in by_true.items()}
    return chosen

def _gradcam_export(model, arch, device, x_batch, cls_targets, paths, out_root_dir):
    model.eval()
    cam = GradCAM(model, arch, device)
    x_batch = x_batch.to(device, non_blocking=True)
    x_batch.requires_grad_(True)
    logits = model(x_batch)
    indices = torch.arange(x_batch.size(0), device=device)
    target_logits = logits[indices, cls_targets.to(device)]
    model.zero_grad(set_to_none=True)
    target_logits.sum().backward(retain_graph=False)
    cams = cam.generate()
    cam.remove()

    os.makedirs(out_root_dir, exist_ok=True)
    for bi in range(x_batch.size(0)):
        rgb = denorm_img(x_batch[bi].detach())
        heat = cams[bi,0].detach().cpu().numpy()
        over = overlay_cam(rgb, heat, alpha=0.35)
        base = os.path.basename(paths[bi])
        save_path = os.path.join(out_root_dir, f"{base}")
        try:
            import imageio
            imageio.imwrite(save_path, over)
        except Exception:
            plt.imsave(save_path, over)

@torch.no_grad()
def evaluate_with_artifacts(model, loader, device, num_classes, classes: List[str],
                            out_tables: str, out_figs: str, tag: str,
                            do_gradcam: bool=False, gradcam_k: int=5, failure_topn: int=3,
                            arch: str="resnet50",
                            monitor: Optional['SystemMonitor']=None, monitor_every: int=10):
    paths_all=[]
    xs=[]; ys=[]; logits_all=[]
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss=0.0
    model.eval()

    for bidx, (x,y,paths) in enumerate(tqdm(loader, desc=f"eval:{tag}", ncols=110)):
        if monitor and (bidx % monitor_every == 0):
            monitor.step()
        x=x.to(device, non_blocking=True)
        if x.is_cuda: x = x.to(memory_format=torch.channels_last)
        y=y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += ce(logits, y).item()
        xs.append(x.detach().cpu()); ys.append(y.detach().cpu()); logits_all.append(logits.detach().cpu())
        paths_all.extend(paths)
    X = torch.cat(xs, dim=0); Y = torch.cat(ys, dim=0).numpy()
    LOG = torch.cat(logits_all, dim=0)
    PRED = LOG.argmax(dim=1).numpy()
    avg_loss = total_loss/len(loader.dataset)
    acc = float((PRED==Y).mean())
    cm = confusion_matrix(Y, PRED, labels=list(range(num_classes)))

    _save_metrics_and_confmats(classes, cm, Y, PRED, acc, avg_loss, out_tables, out_figs, tag)
    _failure_modes_table(classes, cm, failure_topn, out_tables, tag)

    if do_gradcam and gradcam_k>0:
        mis_by_class = _collect_misclass_indices(Y, PRED, paths_all, gradcam_k)
        for true_idx, items in mis_by_class.items():
            if not items: continue
            b_idx   = [it[0] for it in items]
            b_paths = [it[1] for it in items]
            b_pred  = [it[3] for it in items]
            x_batch = X[b_idx]
            cls_targets = torch.tensor(b_pred, dtype=torch.long)
            out_dir = os.path.join(out_figs, f"gradcam_{tag}", f"class_{classes[true_idx]}")
            _gradcam_export(model, arch, device, x_batch, cls_targets, b_paths, out_dir)

    return {"acc": acc, "loss": float(avg_loss)}

def train_epoch(model, loader, device, optimizer, scaler=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    n = 0

    pbar = tqdm(loader, desc="train", ncols=100)

    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = ce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = ce(model(x), y)
            loss.backward()
            optimizer.step()

        total += loss.item() * x.size(0)
        n += x.size(0)

        pbar.set_postfix(loss=loss.item())

    return total / n

def train_epoch_kd(student, teacher, loader, device, alpha=0.5, tau=2.0, beta=0.0, arch_student="resnet18", scaler=None):
    ce = nn.CrossEntropyLoss()
    student.train(); teacher.eval()
    optimizer = getattr(student, "_optimizer")
    total=0.0; n=0
    taps = 2 if beta and beta > 0 else 0

    pbar = tqdm(loader, desc=f"kd({arch_student})", ncols=110)

    for x,y,_ in pbar:
        x=x.to(device, non_blocking=True)
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        y=y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                s_feats, s_hooks = (extract_attention_maps(student, arch_student, taps=taps) if taps>0 else ([],[]))
                t_feats, t_hooks = (extract_attention_maps(teacher, "resnet50", taps=taps) if taps>0 else ([],[]))
                s_logits = student(x)
                with torch.no_grad(): t_logits = teacher(x)
                loss_ce = ce(s_logits, y)
                log_p_s = F.log_softmax(s_logits/tau, dim=1)
                p_t = F.softmax(t_logits/tau, dim=1)
                loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean")*(tau**2)
                loss_at = attention_transfer_loss(s_feats, t_feats, beta=beta, device=device) if taps>0 else torch.tensor(0.0, device=device)
                loss = alpha*loss_ce + (1-alpha)*loss_kd + loss_at
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            s_feats, s_hooks = (extract_attention_maps(student, arch_student, taps=taps) if taps>0 else ([],[]))
            t_feats, t_hooks = (extract_attention_maps(teacher, "resnet50", taps=taps) if taps>0 else ([],[]))
            s_logits = student(x)
            with torch.no_grad(): t_logits = teacher(x)
            loss_ce = ce(s_logits, y)
            log_p_s = F.log_softmax(s_logits/tau, dim=1)
            p_t = F.softmax(t_logits/tau, dim=1)
            loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean")*(tau**2)
            loss_at = attention_transfer_loss(s_feats, t_feats, beta=beta, device=device) if taps>0 else torch.tensor(0.0, device=device)
            loss = alpha*loss_ce + (1-alpha)*loss_kd + loss_at
            loss.backward(); optimizer.step()

        for h in s_hooks + t_hooks:
            try: h.remove()
            except Exception: pass

        total += loss.item()*x.size(0); n += x.size(0)
        pbar.set_postfix(loss=loss.item())

    return total/n

# -----------------------------
# Helpers for orchestration
# -----------------------------
def arch_alias(arch: str) -> str:
    a = arch.lower()
    if a == "mobilenet_v2": return "mobilenetv2"
    if a == "efficientnet_b0": return "efficientnet-b0"
    return a

def build_student(num_classes, arch):
    m = build_model(arch, num_classes, pretrained=False)
    if torch.cuda.is_available():
        m = m.to("cuda")
        m.to(memory_format=torch.channels_last)
    return m

def maybe_load_init_student(net, arch, init_path: Optional[str], logs_dir: str):
    if init_path and os.path.exists(init_path):
        summ = load_checkpoint_flex(net, arch, init_path, strict_head=False)
        if summ.get("skipped_detail"):
            pd.DataFrame([(k,v) for k,v in summ["skipped_detail"].items()], columns=["key","reason"]).to_csv(
                os.path.join(logs_dir, f"student_init_load_skipped_{arch}.csv"), index=False)
        print(f"[{now()}] Loaded student init: {init_path} | loaded={summ['loaded']} skipped={summ['skipped']} missing_in_ckpt={summ['missing_in_ckpt']}")
    else:
        print(f"[{now()}] Student init not found for {arch}: {init_path} (will use random init)")

def run_is_complete(out_root: str, dataset: str, arch: str) -> bool:
    models_dir = os.path.join(out_root, "models")
    if not os.path.isdir(models_dir): return False
    for fname in os.listdir(models_dir):
        if fname.startswith(f"ckpt-best_{dataset}_{arch}_student_kd") and fname.endswith(".pth"):
            return True
    return False

def kd_single_run(
    dataset: str,
    data_root: str,
    arch: str,
    out_root: str,
    teacher_ckpt: Optional[str],
    student_init: Optional[str],
    alpha: float,
    tau: float,
    beta: float,
    epochs: int,
    batch_size: int,
    lr: float,
    img_size: int,
    workers: int,
    seed: int,
    eval_every: int = 2,
    gradcam: bool=False,
    gradcam_k: int=0,
    failure_topn: int=3,
):
    # dataset
    classes, ds_tr, ds_va, ds_te = build_datasets(data_root, dataset, img_size, val_ratio=0.1, val_seed=2025)
    num_classes = len(classes)
    tfm = build_transforms(img_size)

    def mk_loader(ds, shuffle, tfm_kind):
        if isinstance(ds, Subset) and isinstance(ds.dataset, datasets.ImageFolder):
            base = ds.dataset
            new = datasets.ImageFolder(base.root, transform=tfm["train" if tfm_kind=="train" else "eval"])
            ds = Subset(new, ds.indices)
        return _wrap_loader_with_paths(ds, batch_size=batch_size, shuffle=shuffle, workers=workers)

    dl_tr = mk_loader(ds_tr, True,  "train")
    dl_va = mk_loader(ds_va, False, "eval")
    dl_te = mk_loader(ds_te, False, "eval")

    # dirs
    mkdir(out_root)
    figs   = os.path.join(out_root,"figs");   mkdir(figs)
    tables = os.path.join(out_root,"tables"); mkdir(tables)
    models = os.path.join(out_root,"models"); mkdir(models)
    logs   = os.path.join(out_root,"logs");   mkdir(logs)

    # monitor
    tag = f"{dataset}_{arch}_student"
    mon = SystemMonitor(os.path.join(logs, f"{tag}_sysmon.csv"), interval=1.0); mon.start()

    # build teacher + student
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_model(arch, num_classes, pretrained=False).to(device)
    if device.type == "cuda": student.to(memory_format=torch.channels_last)
    maybe_load_init_student(student, arch, student_init, logs)
    teacher = build_model("resnet50", num_classes, pretrained=True).to(device)
    if device.type == "cuda": teacher.to(memory_format=torch.channels_last)
    if teacher_ckpt and os.path.exists(teacher_ckpt):
        summ_t = load_checkpoint_flex(teacher, "resnet50", teacher_ckpt, strict_head=False)
        if summ_t.get("skipped_detail"):
            pd.DataFrame([(k,v) for k,v in summ_t["skipped_detail"].items()], columns=["key","reason"]).to_csv(
                os.path.join(logs, f"ckpt_load_skipped_teacher_{dataset}_resnet50.csv"), index=False)
        print(f"[{now()}] Loaded teacher ckpt: {teacher_ckpt} | loaded={summ_t['loaded']} skipped={summ_t['skipped']}")
    else:
        print(f"[WARN] No teacher ckpt for {dataset}; using ImageNet-pretrained teacher.")

    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    student._optimizer = optimizer; student._scaler = scaler

    best_acc=-1.0; best_path=None; history=[]
    for ep in range(epochs):
        t0=time.time()
        tr_loss = train_epoch_kd(student, teacher, dl_tr, device, alpha=alpha, tau=tau, beta=beta, arch_student=arch, scaler=scaler)

        # throttle eval
        do_eval = ((ep % eval_every) == 0) or (ep == epochs-1)
        if do_eval:
            val_metrics = evaluate_with_artifacts(
                student, dl_va, device, num_classes, classes, tables, figs,
                tag=f"{dataset}_{arch}_student_val",
                do_gradcam=gradcam and (gradcam_k>0), gradcam_k=gradcam_k, failure_topn=failure_topn, arch=arch,
                monitor=mon, monitor_every=10
            )
            val_acc = val_metrics["acc"]; val_loss = val_metrics["loss"]
        else:
            val_acc, val_loss = float("nan"), float("nan")

        dt=time.time()-t0
        history.append({"epoch":ep,"train_loss":tr_loss,"val_acc":val_acc,"val_loss":val_loss,"secs":dt,
                        "alpha":alpha,"tau":tau,"beta":beta})
        print(f"[{now()}] KD ep={ep} a={alpha} t={tau} b={beta} train_loss={tr_loss:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)")
        mon.step()
        if not np.isnan(val_acc) and val_acc>best_acc:
            best_acc=val_acc
            best_path=os.path.join(models, f"ckpt-best_{dataset}_{arch}_student_kd.pth")
            torch.save(student.state_dict(), best_path)

    last_path=os.path.join(models, f"ckpt-last_{dataset}_{arch}_student_kd.pth")
    torch.save(student.state_dict(), last_path)
    pd.DataFrame(history).to_csv(os.path.join(tables, f"trainlog_{dataset}_{arch}_student_kd.csv"), index=False)

    _ = evaluate_with_artifacts(
        student, dl_te, device, num_classes, classes, tables, figs,
        tag=f"{dataset}_{arch}_student_test",
        do_gradcam=gradcam and (gradcam_k>0), gradcam_k=gradcam_k, failure_topn=failure_topn, arch=arch,
        monitor=mon, monitor_every=10
    )
    print(f"[INFO] saved: last={last_path} best={best_path}")
    mon.stop()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Unified teacher/student (KD) with auto val split, Grad-CAM, failure modes + grid KD.")
    ap.add_argument("--mode", choices=["train","eval","kd","grid_kd"], required=True)
    ap.add_argument("--role", choices=["teacher","student"], required=False, default="student")
    ap.add_argument("--dataset", choices=["ham10000","oct2017","isic"], required=False)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_root",  default="RESULTS", help="Base dir for RESULTS/...")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--eval_every", type=int, default=2, help="validate every N epochs (grid_kd uses this)")
    # ckpts
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--strict_head", action="store_true")
    # auto val split
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--val_seed", type=int, default=2025)
    # KD
    ap.add_argument("--teacher_ckpt", default=None)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--tau",   type=float, default=4.0)
    ap.add_argument("--beta",  type=float, default=250.0)
    # Artifacts
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--gradcam_k", type=int, default=0)
    ap.add_argument("--failure_topn", type=int, default=3)
    # Grid KD options
    ap.add_argument("--students", default="mobilenet_v2,efficientnet_b0")
    ap.add_argument("--datasets", default="ham10000,oct2017,isic")
    ap.add_argument("--ablations", default="ablations.json",
                    help="Comma list among softKD,hardKD,ATstrong or JSON path to list of {name,alpha,tau,beta}")

    args = ap.parse_args()
    seed_all(args.seed)

    # teacher & student init maps
    TEACHER_CKPTS = {
        "ham10000": r".\models\teachers\runs_ham10000_resnet50\ckpt-best.pth",
        "oct2017":  r".\models\teachers\oct2017_resnet50\ckpt-best.pth",
        "isic":     r".\models\teachers\isic_resnet50_v2\ckpt-best.pth",
    }
    STUDENT_INIT = {
        "resnet18":        r".\models\students\resnet18-f37072fd.pth",
        "mobilenet_v2":    r".\models\students\mobilenet_v2-b0353104.pth",
        "efficientnet_b0": r".\models\students\efficientnet_b0_rwightman-3dd342df.pth",
    }

    # ---------------- single-run modes ----------------
    if args.mode in ["train","eval","kd"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
        exp_root = os.path.join(args.out_root, f"{args.role}_{args.dataset.lower()}"); mkdir(exp_root)
        figs   = os.path.join(exp_root,"figs");     mkdir(figs)
        tables = os.path.join(exp_root,"tables");   mkdir(tables)
        models = os.path.join(exp_root,"models");   mkdir(models)
        logs   = os.path.join(exp_root,"logs");     mkdir(logs)

        classes, ds_tr, ds_va, ds_te = build_datasets(args.data_root, args.dataset, args.img_size, args.val_ratio, args.val_seed)
        num_classes = len(classes)
        tfm = build_transforms(args.img_size)

        def mk_loader(ds, shuffle, tfm_kind):
            if isinstance(ds, Subset) and isinstance(ds.dataset, datasets.ImageFolder):
                base = ds.dataset
                new = datasets.ImageFolder(base.root, transform=tfm["train" if tfm_kind=="train" else "eval"])
                ds = Subset(new, ds.indices)
            return _wrap_loader_with_paths(ds, batch_size=args.batch_size, shuffle=shuffle, workers=args.workers)

        dl_tr = mk_loader(ds_tr, True,  "train")
        dl_va = mk_loader(ds_va, False, "eval")
        dl_te = mk_loader(ds_te, False, "eval")

        tag = f"{args.dataset}_{args.arch}_{args.role}"
        mon = SystemMonitor(os.path.join(logs, f"{tag}_sysmon.csv"), interval=1.0); mon.start()

        if args.mode in ["train","eval"] and args.role == "teacher":
            net = build_model(args.arch, num_classes, pretrained=True).to(device)
        elif args.mode in ["train","eval"] and args.role == "student":
            net = build_model(args.arch, num_classes, pretrained=True).to(device)
        elif args.mode == "kd":
            net = build_model(args.arch, num_classes, pretrained=False).to(device)
            if device.type == "cuda": net.to(memory_format=torch.channels_last)
            init_map_key = args.arch if args.arch in STUDENT_INIT else ("mobilenet_v2" if args.arch=="mobilenetv2" else "efficientnet_b0" if args.arch=="efficientnet-b0" else args.arch)
            maybe_load_init_student(net, args.arch, STUDENT_INIT.get(init_map_key), logs)
        else:
            raise ValueError("Invalid mode/role")

        if args.ckpt and os.path.exists(args.ckpt):
            summ = load_checkpoint_flex(net, args.arch, args.ckpt, strict_head=args.strict_head)
            if summ.get("skipped_detail"):
                pd.DataFrame([(k,v) for k,v in summ["skipped_detail"].items()], columns=["key","reason"]).to_csv(
                    os.path.join(logs, f"ckpt_load_skipped_{args.dataset}_{args.arch}_{args.role}.csv"), index=False)
            print(f"[{now()}] Loaded ckpt: {args.ckpt} | loaded={summ['loaded']} skipped={summ['skipped']} missing_in_ckpt={summ['missing_in_ckpt']}")

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

        if args.mode == "train":
            best_acc=-1.0; best_path=None; history=[]
            for ep in range(args.epochs):
                t0=time.time()
                tr_loss = train_epoch(net, dl_tr, device, optimizer, scaler)
                val_metrics = evaluate_with_artifacts(
                    net, dl_va, device, num_classes, classes, tables, figs,
                    tag=f"{args.dataset}_{args.arch}_{args.role}_val",
                    do_gradcam=args.gradcam, gradcam_k=args.gradcam_k, failure_topn=args.failure_topn, arch=args.arch,
                    monitor=mon, monitor_every=10
                )
                dt=time.time()-t0
                history.append({"epoch":ep,"train_loss":tr_loss,"val_acc":val_metrics["acc"],"val_loss":val_metrics["loss"],"secs":dt})
                print(f"[{now()}] ep={ep} train_loss={tr_loss:.4f} val_acc={val_metrics['acc']:.4f} ({dt:.1f}s)")
                mon.step()
                if args.save_best and val_metrics["acc"]>best_acc:
                    best_acc=val_metrics["acc"]
                    best_path=os.path.join(models, f"ckpt-best_{args.dataset}_{args.arch}_{args.role}.pth")
                    torch.save(net.state_dict(), best_path)
            last_path=os.path.join(models, f"ckpt-last_{args.dataset}_{args.arch}_{args.role}.pth")
            torch.save(net.state_dict(), last_path)
            pd.DataFrame(history).to_csv(os.path.join(tables, f"trainlog_{args.dataset}_{args.arch}_{args.role}.csv"), index=False)
            _ = evaluate_with_artifacts(
                net, dl_te, device, num_classes, classes, tables, figs,
                tag=f"{args.dataset}_{args.arch}_{args.role}_test",
                do_gradcam=args.gradcam, gradcam_k=args.gradcam_k, failure_topn=args.failure_topn, arch=args.arch,
                monitor=mon, monitor_every=10
            )
            print(f"[INFO] saved: last={last_path} best={best_path}")

        elif args.mode == "eval":
            mon.step()
            _ = evaluate_with_artifacts(
                net, dl_te, device, num_classes, classes, tables, figs,
                tag=f"{args.dataset}_{args.arch}_{args.role}_test",
                do_gradcam=args.gradcam, gradcam_k=args.gradcam_k, failure_topn=args.failure_topn, arch=args.arch,
                monitor=mon, monitor_every=10
            )
            mon.step()

        elif args.mode == "kd":
            teacher = build_model("resnet50", num_classes, pretrained=True).to(device)
            if device.type == "cuda": teacher.to(memory_format=torch.channels_last)
            tpath = args.teacher_ckpt or TEACHER_CKPTS.get(args.dataset)
            if tpath and os.path.exists(tpath):
                summ_t = load_checkpoint_flex(teacher, "resnet50", tpath, strict_head=args.strict_head)
                if summ_t.get("skipped_detail"):
                    pd.DataFrame([(k,v) for k,v in summ_t["skipped_detail"].items()], columns=["key","reason"]).to_csv(
                        os.path.join(logs, f"ckpt_load_skipped_teacher_{args.dataset}_resnet50.csv"), index=False)
                print(f"[{now()}] Loaded teacher ckpt: {tpath} | loaded={summ_t['loaded']} skipped={summ_t['skipped']}")
            else:
                print("[WARN] No teacher_ckpt provided — using ImageNet-pretrained ResNet50.")
            net._optimizer = optimizer; net._scaler = scaler

            best_acc=-1.0; best_path=None; history=[]
            for ep in range(args.epochs):
                t0=time.time()
                tr_loss = train_epoch_kd(net, teacher, dl_tr, device, alpha=args.alpha, tau=args.tau, beta=args.beta, arch_student=args.arch, scaler=scaler)
                val_metrics = evaluate_with_artifacts(
                    net, dl_va, device, num_classes, classes, tables, figs,
                    tag=f"{args.dataset}_{args.arch}_student_val",
                    do_gradcam=args.gradcam, gradcam_k=args.gradcam_k, failure_topn=args.failure_topn, arch=args.arch,
                    monitor=mon, monitor_every=10
                )
                dt=time.time()-t0
                history.append({"epoch":ep,"train_loss":tr_loss,"val_acc":val_metrics["acc"],"val_loss":val_metrics["loss"],"secs":dt,
                                "alpha":args.alpha,"tau":args.tau,"beta":args.beta})
                print(f"[{now()}] KD ep={ep} a={args.alpha} t={args.tau} b={args.beta} train_loss={tr_loss:.4f} val_acc={val_metrics['acc']:.4f} ({dt:.1f}s)")
                mon.step()
                if args.save_best and val_metrics["acc"]>best_acc:
                    best_acc=val_metrics["acc"]
                    best_path=os.path.join(models, f"ckpt-best_{args.dataset}_{args.arch}_student_kd.pth")
                    torch.save(net.state_dict(), best_path)
            last_path=os.path.join(models, f"ckpt-last_{args.dataset}_{args.arch}_student_kd.pth")
            torch.save(net.state_dict(), last_path)
            pd.DataFrame(history).to_csv(os.path.join(tables, f"trainlog_{args.dataset}_{args.arch}_student_kd.csv"), index=False)
            _ = evaluate_with_artifacts(
                net, dl_te, device, num_classes, classes, tables, figs,
                tag=f"{args.dataset}_{args.arch}_student_test",
                do_gradcam=args.gradcam, gradcam_k=args.gradcam_k, failure_topn=args.failure_topn, arch=args.arch,
                monitor=mon, monitor_every=10
            )
            print(f"[INFO] saved: last={last_path} best={best_path}")

        mon.stop()
        return

    # ---------------- grid_kd (ablations.json-aware) ----------------
    def parse_ablations(spec: str):
        spec = spec.strip()
        if spec.endswith(".json") and os.path.exists(spec):
            arr = json.load(open(spec, "r", encoding="utf-8"))
            return [{ "name": a["name"], "alpha": float(a["alpha"]), "tau": float(a["tau"]), "beta": float(a["beta"]) } for a in arr]
        names = [s.strip() for s in spec.split(",") if s.strip()]
        presets = {
            "softKD":   {"alpha":0.6, "tau":6.0, "beta":250.0},
            "hardKD":   {"alpha":0.8, "tau":2.0, "beta":0.0},
            "ATstrong": {"alpha":0.6, "tau":4.0, "beta":1500.0},
        }
        out=[]
        for n in names:
            if n not in presets: raise ValueError(f"Unknown ablation preset: {n}")
            cfg=presets[n].copy(); cfg["name"]=n; out.append(cfg)
        # ensure softKD first to warm-up when present
        out.sort(key=lambda d: 0 if d["name"].lower()=="softkd" else 1)
        return out

    students = [s.strip() for s in args.students.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    abls     = parse_ablations(args.ablations)

    # run OCT2017 first (most time-sensitive); then others
    if "oct2017" in datasets:
        datasets = ["oct2017"] + [d for d in datasets if d != "oct2017"]

    for arch in students:
        arch_key = arch if arch in STUDENT_INIT else ("mobilenet_v2" if arch=="mobilenetv2" else "efficientnet_b0" if arch=="efficientnet-b0" else arch)
        s_alias  = arch_alias(arch_key)
        for abl in abls:
            abl_name = abl["name"]; alpha=abl["alpha"]; tau=abl["tau"]; beta=abl["beta"]
            for ds in datasets:
                out_root = os.path.join(args.out_root, "students", s_alias, abl_name, ds)
                if run_is_complete(out_root, ds, arch_key):
                    print(f"[SKIP] Completed: ds={ds} arch={arch_key} abl={abl_name} -> {out_root}")
                    continue

                t_ckpt = {
                    "ham10000": r".\models\teachers\runs_ham10000_resnet50\ckpt-best.pth",
                    "oct2017":  r".\models\teachers\oct2017_resnet50\ckpt-best.pth",
                    "isic":     r".\models\teachers\isic_resnet50_v2\ckpt-best.pth",
                }[ds]
                s_init = {
                    "resnet18":        r".\models\students\resnet18-f37072fd.pth",
                    "mobilenet_v2":    r".\models\students\mobilenet_v2-b0353104.pth",
                    "efficientnet_b0": r".\models\students\efficientnet_b0_rwightman-3dd342df.pth",
                }[arch_key]

                # --- speed-aware per-dataset overrides ---
                # image size (cap at 192 for OCT & ISIC to cut per-epoch time)
                img_sz_local = args.img_size
                if ds in ["oct2017", "isic"]:
                    img_sz_local = min(img_sz_local, 192)

                # batch sizes tuned for 6GB:
                if arch_key == "mobilenet_v2":
                    bs = 128 if ds != "oct2017" else 96
                elif arch_key == "efficientnet_b0":
                    bs = 96 if ds != "oct2017" else 64
                else:
                    bs = 64

                # LR slightly lower for EN-B0
                lr = 1.5e-3 if arch_key == "efficientnet_b0" else 2e-3

                print(f"\n=== KD RUN: ds={ds} arch={arch_key} abl={abl_name} (alpha={alpha}, tau={tau}, beta={beta}) | img={img_sz_local} bs={bs} eval_every={args.eval_every} ===")
                kd_single_run(
                    dataset=ds,
                    data_root=args.data_root,
                    arch=arch_key,
                    out_root=out_root,
                    teacher_ckpt=t_ckpt,
                    student_init=s_init,
                    alpha=alpha, tau=tau, beta=beta,
                    epochs=args.epochs,
                    batch_size=bs,
                    lr=lr,
                    img_size=img_sz_local,
                    workers=args.workers,
                    seed=args.seed,
                    eval_every=args.eval_every,
                    gradcam=False,
                    gradcam_k=0,
                    failure_topn=3,
                )

if __name__ == "__main__":
    main()
