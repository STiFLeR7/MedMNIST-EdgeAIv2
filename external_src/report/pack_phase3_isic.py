#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-3 ISIC (HAM-mode) PDF packer — A4 friendly

What it packs from each eval dir (teacher + students):
- metrics.json (val/test acc/loss)
- calibration_temperature.json (ECE/NLL/Brier raw vs calibrated, val+test)
- reliability_* PNGs (val/test, raw/cal)
- robustness_table.csv (optional, if --do-robust was used)
- efficiency_latency_{cpu,gpu}.csv, efficiency_memory.csv (optional)

Extras:
- Optional TensorBoard pages from provided TB roots
- Dataset contact sheets (train/val/test), k samples per class

Numbers are trimmed to fixed precision for clean A4 tables.
Structure mirrors the Phase-2 packer so downstream tooling stays consistent.
"""

import argparse, json, csv, io, math, time
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
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None  # handled gracefully

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]

# ------------------------- IO helpers -------------------------
def load_json(p: Path):
    if not p.exists(): return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_header_rows(p: Path):
    if not p.exists(): return [], []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, [])
        rows = [row for row in r]
    return header, rows

# ------------------------- TensorBoard export -------------------------
def _tb_size_guidance():
    return {"scalars": 10000, "images": 256}

def tb_export_plots(tb_dir: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if EventAccumulator is None:
        return []
    try:
        acc = EventAccumulator(str(tb_dir), size_guidance=_tb_size_guidance())
        acc.Reload()
    except Exception:
        return []

    tag_map = acc.Tags() if hasattr(acc, "Tags") else {}
    scalar_tags = tag_map.get("scalars", []) if isinstance(tag_map, dict) else []
    image_tags  = tag_map.get("images", [])  if isinstance(tag_map, dict) else []

    wanted = {
        "loss/train": "loss_train",
        "loss/val":   "loss_val",
        "acc/train":  "acc_train",
        "acc/val":    "acc_val",
        "lr":         "lr"
    }

    outs: List[Path] = []

    if hasattr(acc, "Scalars"):
        for tb_tag, out_name in wanted.items():
            if tb_tag in scalar_tags:
                try:
                    evs = acc.Scalars(tb_tag)
                    xs = [e.step for e in evs]
                    ys = [e.value for e in evs]
                    if not xs: 
                        continue
                    plt.figure()
                    plt.plot(xs, ys)
                    plt.xlabel("step"); plt.ylabel(out_name); plt.title(f"{tb_dir.name}: {tb_tag}")
                    outp = out_dir / f"{out_name}.png"
                    plt.savefig(outp, bbox_inches="tight"); plt.close()
                    outs.append(outp)
                except Exception:
                    pass

    if hasattr(acc, "Images"):
        for tag in image_tags:
            try:
                ims = acc.Images(tag)
                if not ims: continue
                im = ims[-1]
                arr = np.array(Image.open(io.BytesIO(im.encoded_image_string)))
                outp = out_dir / f"img_{tag.replace('/', '_')}.png"
                Image.fromarray(arr).save(outp)
                outs.append(outp)
            except Exception:
                pass

    return outs

# ------------------------- Dataset grids -------------------------
def contact_sheet(split_dir: Path, classes: List[str], k: int, img_size: int, out_path: Path):
    paths = []
    for c in classes:
        cdir = split_dir / c
        if not cdir.exists(): 
            continue
        imgs = sorted([p for p in cdir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])
        if not imgs: continue
        paths.extend(imgs[:k])

    tiles = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB").resize((img_size,img_size))
            tiles.append(im)
        except Exception:
            pass
    if not tiles:
        return None

    cols = k
    rows = int(math.ceil(len(tiles)/cols))
    W,H = tiles[0].size
    sheet = Image.new("RGB", (cols*W, rows*H), (255,255,255))
    for i, im in enumerate(tiles):
        r = i // cols; c = i % cols
        sheet.paste(im, (c*W, r*H))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path

# ------------------------- Drawing helpers -------------------------
def draw_header(c: canvas.Canvas, title: str, page_w, page_h):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, page_h-2*cm, title)
    c.setFont("Helvetica", 9)
    c.drawRightString(page_w-2*cm, page_h-2*cm, time.strftime("%Y-%m-%d %H:%M:%S"))

def draw_kv(c: canvas.Canvas, x, y, key, val):
    c.setFont("Helvetica-Bold", 10); c.drawString(x, y, f"{key}:")
    c.setFont("Helvetica", 10);      c.drawString(x+3.8*cm, y, str(val))

def place_image(c: canvas.Canvas, img_path: Path, x, y, max_w, max_h):
    if (not img_path) or (not img_path.exists()): 
        return 0,0
    im = Image.open(img_path)
    w,h = im.size
    scale = min(max_w/w, max_h/h)
    w2,h2 = w*scale, h*scale
    c.drawImage(ImageReader(im), x, y, width=w2, height=h2, preserveAspectRatio=True, mask='auto')
    return w2,h2

# ------------------------- Table formatting -------------------------
NUMERIC_HEADERS = {"acc","nll","ece","brier","n","level","lat_ms","latency_ms","mem_mb","batch","throughput"}

def _fmt_num(val: str, decimals: int = 4):
    try:
        if float(val).is_integer():
            return str(int(round(float(val))))
    except Exception:
        pass
    try:
        f = float(val)
        return f"{f:.{decimals}f}"
    except Exception:
        return val

def _format_rows_by_header(header: List[str], rows: List[List[str]], decimals: int = 4):
    h_lower = [h.strip().lower() for h in header]
    numeric_idx = [i for i,h in enumerate(h_lower) if h in NUMERIC_HEADERS]
    fmt_rows = []
    for r in rows:
        rr = list(r)
        for i in numeric_idx:
            if i < len(rr):
                rr[i] = _fmt_num(rr[i], decimals)
        fmt_rows.append(rr)
    return fmt_rows, numeric_idx

def _table_style(base_font=8, numeric_cols: List[int] = None):
    st = TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), base_font),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 2),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ])
    if numeric_cols:
        for col in numeric_cols:
            st.add("ALIGN", (col,1), (col,-1), "RIGHT")
    st.add("ALIGN", (0,1), (0,-1), "LEFT")
    return st

def _desired_widths_for(header: List[str], avail_w: float) -> List[float]:
    h = [x.strip().lower() for x in header]
    base_cm = []
    for name in h:
        if name in ("corruption","type","arch","model"): base_cm.append(3.2)
        elif name in ("level","batch"):                 base_cm.append(1.6)
        elif name in ("acc","nll","ece","brier"):       base_cm.append(2.4)
        elif name in ("lat_ms","latency_ms"):           base_cm.append(2.6)
        elif name in ("mem_mb","n","throughput"):       base_cm.append(1.8)
        else:                                           base_cm.append(2.0)
    widths_pt = [w*cm for w in base_cm]
    total = sum(widths_pt)
    if total <= 0:
        return [avail_w/len(header)]*len(header)
    scale = avail_w / total
    return [w*scale for w in widths_pt]

def _draw_paginated_table(c: canvas.Canvas, title: str, header: List[str], rows: List[List[str]],
                          page_w, page_h,
                          col_widths: List[float] = None,
                          decimals: int = 4,
                          max_font=8,
                          top_margin=3*cm, bottom_margin=2*cm, side_margin=2*cm):
    avail_w = page_w - 2*side_margin
    avail_h = page_h - top_margin - bottom_margin

    if not header: header = []
    rows = rows or []
    if header and rows:
        rows, numeric_cols = _format_rows_by_header(header, rows, decimals=decimals)
    else:
        numeric_cols = []

    col_count = max(len(header), max((len(r) for r in rows), default=0))
    if len(header) < col_count:
        header = header + [""]*(col_count-len(header))
    if col_widths is None:
        col_widths = _desired_widths_for(header, avail_w)

    idx = 0
    n = len(rows)
    while idx < n or (idx==0 and n==0):
        c.showPage()
        c.setTitle(title)
        draw_header(c, title, page_w, page_h)
        y_top = page_h - top_margin

        data_page = [header]
        j = idx
        while j < n:
            data_page.append(rows[j])
            tbl = Table(data_page, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
            tbl.setStyle(_table_style(base_font=max_font, numeric_cols=numeric_cols))
            w,h = tbl.wrapOn(c, avail_w, avail_h)
            if h > avail_h:
                data_page.pop()
                break
            j += 1

        if len(data_page) == 1 and idx < n:
            data_page.append(rows[idx])
            j = idx + 1
            tbl = Table(data_page, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
            tbl.setStyle(_table_style(base_font=max_font, numeric_cols=numeric_cols))
            w,h = tbl.wrapOn(c, avail_w, avail_h)

        tbl.drawOn(c, side_margin, y_top - h)
        idx = j if n>0 else 1

def add_table_from_csv(c: canvas.Canvas, title: str, csv_path: Path, page_w, page_h,
                       decimals: int = 4,
                       fallback_msg="(missing)"):
    header, rows = load_csv_header_rows(csv_path)
    if not rows:
        c.showPage(); draw_header(c, title, page_w, page_h)
        c.setFont("Helvetica-Oblique", 10); c.drawString(2*cm, page_h-3*cm, f"{fallback_msg}: {csv_path}")
        return
    avail_w = page_w - 4*cm
    col_widths = _desired_widths_for(header, avail_w)
    _draw_paginated_table(c, title, header, rows, page_w, page_h,
                          col_widths=col_widths, decimals=decimals, max_font=8)

# ------------------------- Page builders -------------------------
def add_page_metrics(c: canvas.Canvas, title: str, dir_path: Path, page_w, page_h):
    c.showPage()
    draw_header(c, f"{title} — Metrics", page_w, page_h)
    met = load_json(dir_path/"metrics.json") or {}
    cal = load_json(dir_path/"calibration_temperature.json") or {}

    y = page_h - 3*cm
    def _fmt4(x):
        try:
            xf = float(x)
            return f"{xf:.4f}"
        except Exception:
            return x

    draw_kv(c, 2*cm, y, "Dir", dir_path); y -= 0.6*cm
    if "val" in met:
        draw_kv(c, 2*cm, y, "Val Acc", _fmt4(met["val"].get("acc","n/a"))); y -= 0.6*cm
        draw_kv(c, 2*cm, y, "Val Loss", _fmt4(met["val"].get("loss","n/a"))); y -= 0.6*cm
    if "test" in met:
        draw_kv(c, 2*cm, y, "Test Acc", _fmt4(met["test"].get("acc","n/a"))); y -= 0.6*cm
        draw_kv(c, 2*cm, y, "Test Loss", _fmt4(met["test"].get("loss","n/a"))); y -= 0.6*cm

    if cal:
        y -= 0.4*cm
        draw_kv(c, 2*cm, y, "Temp (T)", _fmt4(cal.get("temperature","n/a"))); y -= 0.6*cm
        for split in ["val","test"]:
            if split in cal:
                d=cal[split]
                draw_kv(c, 2*cm, y, f"{split.upper()} ECE raw/cal", f"{_fmt4(d.get('ece_raw','-'))} / {_fmt4(d.get('ece_cal','-'))}"); y -= 0.6*cm
                draw_kv(c, 2*cm, y, f"{split.upper()} NLL raw/cal", f"{_fmt4(d.get('nll_raw','-'))} / {_fmt4(d.get('nll_cal','-'))}"); y -= 0.6*cm
                draw_kv(c, 2*cm, y, f"{split.upper()} Brier raw/cal", f"{_fmt4(d.get('brier_raw','-'))} / {_fmt4(d.get('brier_cal','-'))}"); y -= 0.6*cm

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

def add_pages_robust_eff(c: canvas.Canvas, title: str, dir_path: Path, page_w, page_h):
    add_table_from_csv(c, f"{title} — Robustness", dir_path/"robustness_table.csv", page_w, page_h,
                       decimals=4, fallback_msg="No robustness table")
    add_table_from_csv(c, f"{title} — Efficiency (CPU latency)", dir_path/"efficiency_latency_cpu.csv", page_w, page_h,
                       decimals=3, fallback_msg="No CPU latency table")
    add_table_from_csv(c, f"{title} — Efficiency (GPU latency)", dir_path/"efficiency_latency_gpu.csv", page_w, page_h,
                       decimals=3, fallback_msg="No GPU latency table")
    add_table_from_csv(c, f"{title} — Efficiency (Memory)", dir_path/"efficiency_memory.csv", page_w, page_h,
                       decimals=1, fallback_msg="No Memory table")

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
    # ISIC folders can have TitleCase (Train/Val/Test) — normalize
    def _norm(name): 
        for cand in [name, name.lower(), name.capitalize(), name.upper(), name.title()]:
            p = data_root / cand
            if p.exists(): return p
        return data_root / name
    train_dir = _norm("train")
    val_dir   = _norm("val")
    test_dir  = _norm("test")

    classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()]) if train_dir.exists() else []
    tmp = data_root / "_report_tmp"
    grids = []
    for split_dir, split in [(train_dir,"train"),(val_dir,"val"),(test_dir,"test")]:
        outp = tmp / f"grid_{split}.png"
        if split_dir.exists():
            contact_sheet(split_dir, classes, samples_per_class, img_size, outp)
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

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--students-dirs", required=True, help="Comma-separated list of student eval dirs")
    ap.add_argument("--tb-root-list", default="", help="Comma-separated TB log roots (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="MedMNIST-EdgeAI v2 — Phase-3 ISIC (HAM-mode)")
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
    c.drawString(2*cm, page_h-3*cm, "Phase-3 ISIC consolidated evidence (HAM-mode)")
    c.drawString(2*cm, page_h-3.8*cm, f"Teacher dir: {teacher_dir}")
    y = page_h-4.6*cm
    for sd in student_dirs:
        c.drawString(2*cm, y, f"Student dir: {sd}")
        y -= 0.6*cm
    c.showPage()

    # Teacher pages
    add_page_metrics(c, "Teacher", teacher_dir, page_w, page_h)
    add_pages_robust_eff(c, "Teacher", teacher_dir, page_w, page_h)

    # Students pages
    for sd in student_dirs:
        title = f"Student — {sd.name}"
        add_page_metrics(c, title, sd, page_w, page_h)
        add_pages_robust_eff(c, title, sd, page_w, page_h)

    # Optional TensorBoard pages
    tmp_dir = out_pdf.parent / "_tb_plots_tmp"
    for tb in tb_dirs:
        add_page_tb(c, tb.name, tb, tmp_dir, page_w, page_h)

    # Dataset grids
    add_page_dataset_grids(c, data_root, args.samples_per_class, args.img_size, page_w, page_h)

    c.save()
    print("Wrote PDF:", out_pdf)

if __name__ == "__main__":
    main()
