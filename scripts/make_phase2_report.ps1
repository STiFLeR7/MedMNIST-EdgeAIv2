<# 
make_phase2_report.ps1 — Phase 2 (OCT2017) PDF builder

It assembles:
- Core metrics (val/test)
- Calibration (temperature, ECE/NLL/Brier) + reliability curves
- Robustness (if robustness_table.csv present)
- Efficiency (latency CPU/GPU, memory) tables
- TensorBoard scalars (loss/train, loss/val, acc/train, acc/val, lr)
- Dataset image grids (train/val/test), per-class samples

Usage (from repo root):
  pwsh -File .\scripts\make_phase2_report.ps1 `
    -DataRoot ".\data\OCT2017" `
    -TeacherDir ".\reports\phase2_oct2017\teacher_resnet50" `
    -StudentsDirs ".\reports\phase2_oct2017\student_effb0,.\reports\phase2_oct2017\student_mbv2,.\reports\phase2_oct2017\student_resnet18" `
    -TBRoots ".\models\teachers\oct2017_resnet50_seed0\tb,.\models\students\oct2017_effb0_kdat_seed0\tb,.\models\students\oct2017_mbv2_kdat_seed0\tb,.\models\students\oct2017_resnet18_kdat_seed0\tb" `
    -OutPDF ".\reports\phase2_oct2017\phase2_consolidated.pdf" `
    -Title "MedMNIST-EdgeAI v2 — Phase-2 OCT2017" `
    -SamplesPerClass 4 `
    -ImgSize 224 `
    -ForceRebuild:$false

Notes:
- Requires Python with: reportlab, matplotlib, pillow, numpy, torch, torchvision, tensorboard (for event_accumulator).
- If the Python packer module is missing, this script will write it.
#>

[CmdletBinding()]
param(
  [string]$DataRoot = ".\data\OCT2017",
  [string]$TeacherDir = ".\reports\phase2_oct2017\teacher_resnet50",
  [string]$StudentsDirs = ".\reports\phase2_oct2017\student_effb0,.\reports\phase2_oct2017\student_mbv2,.\reports\phase2_oct2017\student_resnet18",
  [string]$TBRoots = ".\models\teachers\oct2017_resnet50_seed0\tb,.\models\students\oct2017_effb0_kdat_seed0\tb,.\models\students\oct2017_mbv2_kdat_seed0\tb,.\models\students\oct2017_resnet18_kdat_seed0\tb",
  [string]$OutPDF = ".\reports\phase2_oct2017\phase2_consolidated.pdf",
  [string]$Title = "MedMNIST-EdgeAI v2 — Phase-2 OCT2017",
  [int]$SamplesPerClass = 4,
  [int]$ImgSize = 224,
  [switch]$ForceRebuild = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) {
  if (-not $p) { throw "Ensure-Dir got null/empty path" }
  if (-not (Test-Path -LiteralPath $p)) {
    New-Item -ItemType Directory -Path $p | Out-Null
  }
  return (Resolve-Path -LiteralPath $p).Path
}
function Resolve-Existing([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { throw "Path not found: $p" }
  return (Resolve-Path -LiteralPath $p).Path
}
function Py() {
  param([Parameter(ValueFromRemainingArguments=$true)] [string[]]$Args)
  Write-Host ">>> python $($Args -join ' ')" -ForegroundColor Cyan
  & python @Args
  if ($LASTEXITCODE -ne 0) { throw "Python step failed: $($Args -join ' ')" }
}

# --- Paths/Logs/PYTHONPATH ---
$RepoRoot = (Resolve-Path ".").Path
$env:PYTHONPATH = "$RepoRoot;$($env:PYTHONPATH)"
$LogsDir = Ensure-Dir (Join-Path $RepoRoot "logs")
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDir "make_phase2_report_$ts.log"
Start-Transcript -LiteralPath $LogFile | Out-Null

try {
  $DataRoot = Resolve-Existing $DataRoot
  $TeacherDir = Resolve-Existing $TeacherDir

  $StudentDirList = @()
  foreach ($sd in $StudentsDirs.Split(",")) {
    $StudentDirList += (Resolve-Existing $sd.Trim())
  }
  $StudentsCSV = ($StudentDirList -join ",")

  $TBList = @()
  foreach ($tb in $TBRoots.Split(",")) {
    $tb = $tb.Trim()
    if ($tb -and (Test-Path -LiteralPath $tb)) {
      $TBList += (Resolve-Existing $tb)
    }
  }
  $TBCSV = ($TBList -join ",")

  $OutPDFDir = Ensure-Dir (Split-Path -Parent $OutPDF)

  # --- Ensure Python packer exists (write if missing or -ForceRebuild) ---
  $PackerPath = Join-Path $RepoRoot "external_src\report\pack_phase2_oct2017.py"
  $PackerDir  = Split-Path -Parent $PackerPath
  Ensure-Dir $PackerDir
  $NeedWrite = $ForceRebuild -or (-not (Test-Path -LiteralPath $PackerPath))

  if ($NeedWrite) {
    @"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, csv, io, glob, math, time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from torchvision import datasets, transforms

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]

def load_json(p: Path):
    if not p.exists(): return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_rows(p: Path):
    if not p.exists(): return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            rows.append(row)
    return rows

def tb_export_plots(tb_dir: Path, out_dir: Path) -> List[Path]:
    """Export common scalar plots from a TensorBoard logdir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # EventAccumulator config (scalars/images)
    acc = EventAccumulator(str(tb_dir), size_guidance={
        EventAccumulator.SCALARS: 10000,
        EventAccumulator.IMAGES:  256,
    })
    try:
        acc.Reload()
    except Exception:
        return []
    tags = {
        "loss/train": "loss_train",
        "loss/val":   "loss_val",
        "acc/train":  "acc_train",
        "acc/val":    "acc_val",
        "lr":         "lr"
    }
    out_paths = []
    # Scalars
    for tb_tag, out_name in tags.items():
        if tb_tag in acc.Tags().get("scalars", []):
            evs = acc.Scalars(tb_tag)
            xs = [e.step for e in evs]
            ys = [e.value for e in evs]
            plt.figure()
            plt.plot(xs, ys)
            plt.xlabel("step"); plt.ylabel(out_name); plt.title(f"{tb_dir.name}: {tb_tag}")
            outp = out_dir / f"{out_name}.png"
            plt.savefig(outp, bbox_inches="tight"); plt.close()
            out_paths.append(outp)
    # Images (optional)
    if "images" in acc.Tags():
        for tag in acc.Tags()["images"]:
            ims = acc.Images(tag)
            if not ims: continue
            # only take last one
            im = ims[-1]
            arr = np.array(Image.open(io.BytesIO(im.encoded_image_string)))
            outp = out_dir / f"img_{tag.replace('/', '_')}.png"
            Image.fromarray(arr).save(outp)
            out_paths.append(outp)
    return out_paths

def contact_sheet(split_dir: Path, classes: List[str], samples_per_class: int, img_size: int, out_path: Path):
    """Make a grid with samples_per_class per class for split (train/val/test)."""
    paths = []
    for c in classes:
        cdir = split_dir / c
        if not cdir.exists(): continue
        imgs = sorted([p for p in cdir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])
        if not imgs: continue
        pick = imgs[:samples_per_class]
        paths.extend(pick)
    if not paths: return None

    # load + tile
    tiles = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB").resize((img_size,img_size))
            tiles.append(im)
        except Exception:
            pass
    if not tiles: return None

    cols = samples_per_class
    rows = int(math.ceil(len(tiles)/cols))
    W,H = tiles[0].size
    sheet = Image.new("RGB", (cols*W, rows*H), (255,255,255))
    for i, im in enumerate(tiles):
        r = i // cols; c = i % cols
        sheet.paste(im, (c*W, r*H))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path

def draw_header(c: canvas.Canvas, title: str, page_w, page_h):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, page_h-2*cm, title)
    c.setFont("Helvetica", 9)
    c.drawRightString(page_w-2*cm, page_h-2*cm, time.strftime("%Y-%m-%d %H:%M:%S"))

def draw_kv(c: canvas.Canvas, x, y, key, val):
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, f"{key}:")
    c.setFont("Helvetica", 10)
    c.drawString(x+3.8*cm, y, str(val))

def place_image(c: canvas.Canvas, img_path: Path, x, y, max_w, max_h):
    if (not img_path) or (not img_path.exists()): return 0,0
    im = Image.open(img_path)
    w,h = im.size
    scale = min(max_w/w, max_h/h)
    w2,h2 = w*scale, h*scale
    c.drawImage(ImageReader(im), x, y, width=w2, height=h2, preserveAspectRatio=True, mask='auto')
    return w2,h2

def add_page_metrics(c: canvas.Canvas, title: str, dir_path: Path, page_w, page_h):
    c.showPage()
    draw_header(c, f"{title} — Metrics", page_w, page_h)
    # load jsons
    met = load_json(dir_path/"metrics.json") or {}
    cal = load_json(dir_path/"calibration_temperature.json") or {}
    y = page_h - 3*cm
    draw_kv(c, 2*cm, y, "Dir", dir_path); y -= 0.6*cm
    if "val" in met:
        draw_kv(c, 2*cm, y, "Val Acc", met["val"].get("acc","n/a")); y -= 0.6*cm
        draw_kv(c, 2*cm, y, "Val Loss", met["val"].get("loss","n/a")); y -= 0.6*cm
    if "test" in met:
        draw_kv(c, 2*cm, y, "Test Acc", met["test"].get("acc","n/a")); y -= 0.6*cm
        draw_kv(c, 2*cm, y, "Test Loss", met["test"].get("loss","n/a")); y -= 0.6*cm
    if cal:
        y -= 0.4*cm
        draw_kv(c, 2*cm, y, "Temp (T)", cal.get("temperature","n/a")); y -= 0.6*cm
        for split in ["val","test"]:
            if split in cal:
                d=cal[split]
                draw_kv(c, 2*cm, y, f"{split.upper()} ECE raw/cal", f"{d.get('ece_raw','-')} / {d.get('ece_cal','-')}"); y -= 0.6*cm
                draw_kv(c, 2*cm, y, f"{split.upper()} NLL raw/cal", f"{d.get('nll_raw','-')} / {d.get('nll_cal','-')}"); y -= 0.6*cm
                draw_kv(c, 2*cm, y, f"{split.upper()} Brier raw/cal", f"{d.get('brier_raw','-')} / {d.get('brier_cal','-')}"); y -= 0.6*cm

    # reliability images
    y2 = page_h - 13*cm
    imgs = [
        ("Val Reliability (Raw)", dir_path/"reliability_val_raw.png"),
        ("Val Reliability (Cal)", dir_path/"reliability_val_cal.png"),
        ("Test Reliability (Raw)", dir_path/"reliability_test_raw.png"),
        ("Test Reliability (Cal)", dir_path/"reliability_test_cal.png"),
    ]
    x = 2*cm
    for idx,(label,p) in enumerate(imgs):
        if idx==2: x=2*cm; y2 = page_h - 22*cm
        if p.exists():
            c.setFont("Helvetica", 9)
            c.drawString(x, y2+7.6*cm, label)
            place_image(c, p, x, y2, 8*cm, 7.2*cm)
            x += 9*cm

def add_page_robust_eff(c: canvas.Canvas, title: str, dir_path: Path, page_w, page_h):
    c.showPage()
    draw_header(c, f"{title} — Robustness & Efficiency", page_w, page_h)
    y = page_h - 3*cm
    rob = dir_path/"robustness_table.csv"
    if rob.exists():
        rows = load_csv_rows(rob)
        c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Robustness summary (first 12 rows)"); y -= 0.7*cm
        c.setFont("Helvetica", 8)
        for r in rows[:12]:
            c.drawString(2*cm, y, ", ".join(map(str,r))); y -= 0.45*cm
    else:
        c.setFont("Helvetica-Oblique", 10); c.drawString(2*cm, y, "No robustness_table.csv"); y -= 0.7*cm

    # Efficiency
    eff_cpu = dir_path/"efficiency_latency_cpu.csv"
    eff_gpu = dir_path/"efficiency_latency_gpu.csv"
    eff_mem = dir_path/"efficiency_memory.csv"
    for cap, p in [("CPU Latency", eff_cpu), ("GPU Latency", eff_gpu), ("Memory", eff_mem)]:
        y -= 0.4*cm
        c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, cap); y -= 0.6*cm
        if p.exists():
            rows = load_csv_rows(p)
            c.setFont("Helvetica", 8)
            for r in rows[:10]:
                c.drawString(2*cm, y, ", ".join(map(str,r))); y -= 0.45*cm
        else:
            c.setFont("Helvetica-Oblique", 10); c.drawString(2*cm, y, "(missing)"); y -= 0.6*cm

def add_page_tb(c: canvas.Canvas, title: str, tb_dir: Path, tmp_dir: Path, page_w, page_h):
    c.showPage()
    draw_header(c, f"{title} — TensorBoard", page_w, page_h)
    out_imgs = tb_export_plots(tb_dir, tmp_dir/tb_dir.name)
    if not out_imgs:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(2*cm, page_h-3*cm, f"No scalars/images found in {tb_dir}")
        return
    x, y = 2*cm, page_h - 4*cm
    wslot, hslot = 8*cm, 6*cm
    for i, img in enumerate(sorted(out_imgs)):
        if i>0 and i%2==0:
            x = 2*cm; y -= (hslot + 1.2*cm)
        c.setFont("Helvetica", 9)
        c.drawString(x, y+ hslot + 0.2*cm, img.name)
        place_image(c, img, x, y, wslot, hslot)
        x += (wslot + 1.5*cm)

def add_page_dataset_grids(c: canvas.Canvas, data_root: Path, samples_per_class: int, img_size: int, page_w, page_h):
    c.showPage()
    draw_header(c, "Dataset Grids (train/val/test)", page_w, page_h)
    # discover classes
    classes = sorted([p.name for p in (data_root/"train").iterdir() if p.is_dir()])
    tmp = data_root / "_report_tmp"
    grids = []
    for split in ["train", "val", "test"]:
        outp = tmp / f"grid_{split}.png"
        contact_sheet(data_root/split, classes, samples_per_class, img_size, outp)
        grids.append( (split, outp) )

    x = 2*cm; y = page_h - 4*cm
    wslot, hslot = 8*cm, 8*cm
    for i,(split,img) in enumerate(grids):
        if i==1:
            x = 2*cm; y -= (hslot + 1.5*cm)
        elif i==2:
            x = 2*cm; y -= (hslot + 1.5*cm)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y + hslot + 0.2*cm, split.upper())
        place_image(c, img, x, y, wslot, hslot)
        x += (wslot + 1.5*cm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--students-dirs", required=True, help="Comma-separated list of student eval dirs")
    ap.add_argument("--tb-root-list", default="", help="Comma-separated TB log roots (will try plotting common scalars)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="MedMNIST-EdgeAI v2 — Phase-2 OCT2017")
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    data_root = Path(args.dataset_root)
    teacher_dir = Path(args.teacher_dir)
    student_dirs = [Path(x.strip()) for x in args.students_dirs.split(",") if x.strip()]
    tb_dirs = [Path(x.strip()) for x in args.tb_root_list.split(",") if x.strip()]
    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    page_w, page_h = A4

    # Cover
    draw_header(c, args.title, page_w, page_h)
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, page_h-3*cm, "Phase-2 OCT2017 consolidated evidence")
    c.drawString(2*cm, page_h-3.8*cm, f"Teacher dir: {teacher_dir}")
    y = page_h-4.6*cm
    for sd in student_dirs:
        c.drawString(2*cm, y, f"Student dir: {sd}")
        y -= 0.6*cm
    c.showPage()

    # Teacher pages
    add_page_metrics(c, "Teacher", teacher_dir, page_w, page_h)
    add_page_robust_eff(c, "Teacher", teacher_dir, page_w, page_h)

    # Students pages
    for sd in student_dirs:
        title = f"Student — {sd.name}"
        add_page_metrics(c, title, sd, page_w, page_h)
        add_page_robust_eff(c, title, sd, page_w, page_h)

    # TensorBoard pages
    tmp_dir = out_pdf.parent / "_tb_plots_tmp"
    for tb in tb_dirs:
        add_page_tb(c, tb.name, tb, tmp_dir, page_w, page_h)

    # Dataset grids
    add_page_dataset_grids(c, data_root, args.samples_per_class, args.img_size, page_w, page_h)

    c.save()
    print("Wrote PDF:", out_pdf)

if __name__ == "__main__":
    main()
"@ | Set-Content -LiteralPath $PackerPath -Encoding UTF8
    Write-Host "Wrote packer module: $PackerPath" -ForegroundColor DarkGreen
  } else {
    Write-Host "Using existing packer: $PackerPath" -ForegroundColor DarkGreen
  }

  # --- Build PDF ---
  Write-Host "`n[1/1] Building Phase-2 PDF" -ForegroundColor Yellow
  Py -m external_src.report.pack_phase2_oct2017 `
    --dataset-root $DataRoot `
    --teacher-dir $TeacherDir `
    --students-dirs $StudentsCSV `
    --tb-root-list $TBCSV `
    --out $OutPDF `
    --title $Title `
    --samples-per-class $SamplesPerClass `
    --img-size $ImgSize

  Write-Host "`nDone. PDF:" -ForegroundColor Green
  Write-Host (" - " + (Resolve-Path $OutPDF).Path)

} catch {
  Write-Error $_.Exception.Message
  throw
} finally {
  Stop-Transcript | Out-Null
  Write-Host "Log saved to: $LogFile" -ForegroundColor DarkGray
}
