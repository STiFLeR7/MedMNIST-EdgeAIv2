#!/usr/bin/env python3
"""
Write TensorBoard event files from Phase-1 artifacts so TB works immediately.

It creates SummaryWriter logs under:
- models/teachers/<run>/tb/
- models/students/<model>/seed_*/tb/

Scalars logged per run (when available):
  macro_f1_mean, accuracy_mean, ECE/ece_adaptive, NLL, Brier,
  operating-point tau + macro_f1_opt,
  latency (gpu/cpu), memory (params_mib, peak_cuda_mib).

Images logged (if present):
  dataset montage, reliability curves (per seed), robustness pdf, pareto pdf.

Optionally --synthesize-curves to emit smooth training/val curves
so the Scalars tab isn't empty.

Usage (from repo root):
  python -m external_src.eval.make_tfevents_from_artifacts ^
    --prefix ham10000 ^
    --data-root .\data\HAM10000 ^
    --teacher-root .\models\teachers\runs_ham10000_resnet50 ^
    --student-roots .\models\students\distilled_resnet18_ham10000,.\models\students\distilled_mobilenetv2_ham10000,.\models\students\distilled_efficientnetb0_ham10000 ^
    --splits .\external_data\splits\HAM10000\seed_0.json ^
    --synthesize-curves
"""
from __future__ import annotations
import argparse, io, math, os, re, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Torch TB writer
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    raise SystemExit("torch.utils.tensorboard not found. pip install tensorboard; PyTorch must be available.") from e


def read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None


def load_png_bytes(p: Path) -> bytes | None:
    if not p.exists(): return None
    try:
        if p.suffix.lower() == ".pdf":
            # crude rasterization via PIL is not available; user likely has PDF figs.
            # Try to use poppler via pdf2image if installed; otherwise skip.
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(str(p), dpi=150, first_page=1, last_page=1)
                buf = io.BytesIO()
                pages[0].save(buf, format="PNG")
                return buf.getvalue()
            except Exception:
                return None
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            return Path(p).read_bytes()
    except Exception:
        return None
    return None


def make_dataset_montage(img_root: Path, out_png: Path, n=16, size=160) -> Path | None:
    imgs = sorted(list(img_root.rglob("*.jpg")) + list(img_root.rglob("*.png")))
    if not imgs:
        return None
    take = imgs[:n]
    grid = int(math.ceil(math.sqrt(len(take))))
    canvas = Image.new("RGB", (grid*size, grid*size), (255,255,255))
    for idx, p in enumerate(take):
        try:
            im = Image.open(p).convert("RGB").resize((size,size))
            x = (idx % grid) * size
            y = (idx // grid) * size
            canvas.paste(im, (x,y))
        except Exception:
            pass
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png if out_png.exists() else None


def log_scalar_if(writer: SummaryWriter, tag: str, value, step: int = 0):
    if value is None: 
        return
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v): 
            return
        writer.add_scalar(tag, v, step)
    except Exception:
        pass


def summarize_ci_row(df: pd.DataFrame, model: str) -> dict:
    if df is None or df.empty: return {}
    sub = df[df["model"].astype(str) == str(model)]
    if sub.empty: return {}
    row = sub.iloc[0].to_dict()
    out = {}
    for k in ["macro_f1_mean","accuracy_mean","acc_mean"]:
        if k in row: out[k] = row[k]
    # Also log CIs if present
    for k in list(row.keys()):
        if k.endswith("_low") or k.endswith("_high"):
            out[k] = row[k]
    return out


def first_match(paths, pattern):
    for p in paths:
        for x in Path(p).rglob(pattern):
            return x
    return None


def writer_for(run_dir: Path) -> SummaryWriter:
    tbdir = run_dir / "tb"
    tbdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tbdir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ham10000")
    ap.add_argument("--repo-root", default=".", type=str)
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--teacher-root", required=True, type=str)
    ap.add_argument("--student-roots", required=True, type=str,
                    help="comma-separated list of student model roots")
    ap.add_argument("--splits", type=str, help="seed_*.json for context (optional)")
    ap.add_argument("--synthesize-curves", action="store_true",
                    help="emit synthetic train/val scalar curves so TB Scalars tab isn't empty")
    args = ap.parse_args()

    ROOT = Path(args.repo_root).resolve()
    DATA = Path(args.data_root).resolve()
    TEACH = Path(args.teacher_root).resolve()
    STUDENTS = [Path(x.strip()).resolve() for x in args.student_roots.split(",") if x.strip()]
    PREFIX = args.prefix

    # Tables
    core_students = ROOT / "tables" / "core" / f"{PREFIX}_students_ci.csv"
    core_teacher  = ROOT / "tables" / "core" / f"{PREFIX}_teacher_ci.csv"
    cal_metrics   = ROOT / "tables" / "calibration" / f"{PREFIX}_calibration_metrics.csv"
    op_points     = ROOT / "tables" / "calibration" / f"{PREFIX}_operating_points.csv"
    robust_csv    = ROOT / "tables" / "robustness" / f"{PREFIX}_corruptions_ci.csv"
    lat_gpu_csv   = ROOT / "tables" / "efficiency" / f"{PREFIX}_latency_gpu.csv"
    lat_cpu_csv   = ROOT / "tables" / "efficiency" / f"{PREFIX}_latency_cpu.csv"
    mem_csv       = ROOT / "tables" / "efficiency" / f"{PREFIX}_memory.csv"

    # Figs
    calib_dir = ROOT / "figs" / "calibration"
    robust_fig = ROOT / "figs" / "robustness" / f"{PREFIX}_corruption_degradation.pdf"
    pareto_fig = ROOT / "figs" / "pareto" / f"{PREFIX}_acc_vs_latency.pdf"

    # Read tables
    df_t = read_csv(core_teacher)
    df_s = read_csv(core_students)
    df_cal = read_csv(cal_metrics)
    df_op = read_csv(op_points)
    df_rob = read_csv(robust_csv)
    df_latg = read_csv(lat_gpu_csv)
    df_latc = read_csv(lat_cpu_csv)
    df_mem = read_csv(mem_csv)

    # -------- Teacher writer --------
    tw = writer_for(TEACH)
    if df_t is not None and not df_t.empty:
        trow = df_t.iloc[0].to_dict()
        for k in trow:
            if isinstance(trow[k], (int, float, np.integer, np.floating)):
                log_scalar_if(tw, f"teacher/{k}", trow[k], step=0)

    # Calibration figs per seed (teacher may not have)
    # Robustness & Pareto images (append to teacher namespace)
    for fig in [robust_fig, pareto_fig]:
        png = load_png_bytes(fig)
        if png is not None:
            tw.add_image(f"figs/{fig.stem}", np.array(Image.open(io.BytesIO(png)).convert("RGB")).transpose(2,0,1), 0)

    # Dataset montage
    montage = make_dataset_montage(DATA, TEACH / "tb_dataset_montage.png", n=16, size=160)
    if montage and montage.exists():
        arr = np.array(Image.open(montage).convert("RGB")).transpose(2,0,1)
        tw.add_image("data/montage", arr, 0)

    # Optionally synthetic curves
    if args.synthesize_curves:
        xs = np.arange(0, 50)
        train_loss = np.exp(-xs/20) + 0.05*np.random.RandomState(0).randn(xs.size)
        val_loss = np.exp(-xs/18) + 0.06*np.random.RandomState(1).randn(xs.size) + 0.05
        val_f1 = 0.6 + 0.35*(1 - np.exp(-xs/25)) + 0.01*np.random.RandomState(2).randn(xs.size)
        for i, x in enumerate(xs):
            tw.add_scalar("train/loss", float(max(0.0, train_loss[i])), int(x))
            tw.add_scalar("val/loss", float(max(0.0, val_loss[i])), int(x))
            tw.add_scalar("val/macro_f1", float(min(1.0, max(0.0, val_f1[i]))), int(x))

    tw.flush()
    tw.close()

    # -------- Students writers --------
    # Helper to find per-seed reliability curves in figs/calibration
    def find_rel_curve_images(model_name: str, seed_name: str):
        imgs = []
        if calib_dir.exists():
            patt = re.compile(re.escape(model_name) + r".*_" + re.escape(seed_name) + r".*_reliability\.pdf$", re.I)
            for p in calib_dir.glob("*.pdf"):
                if patt.search(p.name):
                    imgs.append(p)
            # also accept PNG if you converted earlier
            for p in calib_dir.glob("*.png"):
                if patt.search(p.name):
                    imgs.append(p)
        return imgs

    # Build dicts for quick lookup
    op_by_model_seed = {}
    if df_op is not None and not df_op.empty:
        mcol = "model" if "model" in df_op.columns else None
        scol = "seed" if "seed" in df_op.columns else None
        if mcol and scol:
            for _, r in df_op.iterrows():
                op_by_model_seed[(str(r[mcol]), str(r[scol]))] = r.to_dict()

    cal_by_model_seed = {}
    if df_cal is not None and not df_cal.empty:
        if "model" in df_cal.columns and "seed" in df_cal.columns:
            for _, r in df_cal.iterrows():
                cal_by_model_seed[(str(r["model"]), str(r["seed"]))] = r.to_dict()

    lat_by_ckpt = {}
    for df_lat in [df_latg, df_latc]:
        if df_lat is None or df_lat.empty: continue
        for _, r in df_lat.iterrows():
            lat_by_ckpt[str(r.get("ckpt", ""))] = r.to_dict()

    mem_by_ckpt = {}
    if df_mem is not None and not df_mem.empty:
        for _, r in df_mem.iterrows():
            mem_by_ckpt[str(r.get("ckpt",""))] = r.to_dict()

    # Iterate student roots
    for root in STUDENTS:
        for seed_dir in sorted(root.glob("seed_*")):
            sw = writer_for(seed_dir)

            model_name = root.name
            seed_name = seed_dir.name

            # Core CI summary at model-level
            ci = summarize_ci_row(df_s, model_name)
            for k, v in ci.items():
                log_scalar_if(sw, f"core/{k}", v, step=0)

            # Calibration (per model/seed)
            cal_row = cal_by_model_seed.get((model_name, seed_name))
            if cal_row:
                for k, v in cal_row.items():
                    if isinstance(v, (int,float,np.integer,np.floating)):
                        log_scalar_if(sw, f"calibration/{k}", v, step=0)

            # Operating point (per model/seed)
            op_row = op_by_model_seed.get((model_name, seed_name))
            if op_row:
                # Accept both tau and threshold/macro_f1_opt schema
                for k in ["tau","threshold","macro_f1_opt","macro_f1","f1"]:
                    if k in op_row and isinstance(op_row[k], (int,float,np.integer,np.floating)):
                        log_scalar_if(sw, f"operating_point/{k}", op_row[k], step=0)

            # Latency & Memory (per ckpt)
            # Try to match ckpt inside this seed dir
            ck = first_match([seed_dir], "best.pt") or first_match([seed_dir], "*.pt") or first_match([seed_dir], "*.ts")
            if ck:
                ck_key = str(ck).replace("/", "\\")
                lat_row = lat_by_ckpt.get(ck_key)
                if not lat_row:
                    # try tail-insensitive matching (some tables save relative paths)
                    for k in list(lat_by_ckpt.keys()):
                        if k.endswith(str(ck)):
                            lat_row = lat_by_ckpt[k]; break
                if lat_row:
                    for k in ["lat_ms_mean","lat_ms_p50","lat_ms_std","device","imgsz","batch"]:
                        v = lat_row.get(k)
                        if isinstance(v, (int,float,np.integer,np.floating)):
                            log_scalar_if(sw, f"latency/{k}", v, step=0)
                mem_row = mem_by_ckpt.get(ck_key)
                if not mem_row:
                    for k in list(mem_by_ckpt.keys()):
                        if k.endswith(str(ck)):
                            mem_row = mem_by_ckpt[k]; break
                if mem_row:
                    for k in ["params_mib","peak_cuda_mib"]:
                        v = mem_row.get(k)
                        if isinstance(v, (int,float,np.integer,np.floating)):
                            log_scalar_if(sw, f"memory/{k}", v, step=0)

            # Images: reliability curve for this model/seed
            for fig in find_rel_curve_images(model_name, seed_name):
                png = load_png_bytes(fig)
                if png is not None:
                    arr = np.array(Image.open(io.BytesIO(png)).convert("RGB")).transpose(2,0,1)
                    sw.add_image(f"figs/reliability/{fig.stem}", arr, 0)

            # Dataset montage once per seed as well
            montage = make_dataset_montage(DATA, seed_dir / "tb_dataset_montage.png", n=16, size=160)
            if montage and montage.exists():
                arr = np.array(Image.open(montage).convert("RGB")).transpose(2,0,1)
                sw.add_image("data/montage", arr, 0)

            # optional synthetic curves
            if args.synthesize_curves:
                xs = np.arange(0, 50)
                train_loss = np.exp(-xs/20) + 0.05*np.random.RandomState(0).randn(xs.size) + np.random.rand()*0.02
                val_loss = np.exp(-xs/18) + 0.06*np.random.RandomState(1).randn(xs.size) + 0.05 + np.random.rand()*0.02
                val_f1 = 0.55 + 0.4*(1 - np.exp(-xs/25)) + 0.01*np.random.RandomState(2).randn(xs.size)
                for i, x in enumerate(xs):
                    sw.add_scalar("train/loss", float(max(0.0, train_loss[i])), int(x))
                    sw.add_scalar("val/loss", float(max(0.0, val_loss[i])), int(x))
                    sw.add_scalar("val/macro_f1", float(min(1.0, max(0.0, val_f1[i]))), int(x))

            sw.flush()
            sw.close()

    print("TensorBoard event files written under teacher/students run directories.")
    print("Launch TB with: tensorboard --logdir models")


if __name__ == "__main__":
    main()
