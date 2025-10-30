# tools/efficiency_profile.py
import argparse, os, glob, json, torch, pandas as pd
from torchvision import models
from thop import profile

ARCH_MAP = {
  "resnet50": lambda: models.resnet50(num_classes=1000),
  "resnet18": lambda: models.resnet18(num_classes=1000),
  "mbv2":     lambda: models.mobilenet_v2(num_classes=1000),
  "effb0":    lambda: models.efficientnet_b0(num_classes=1000)
}

def params_flops(arch, imgsz=224):
    m = ARCH_MAP[arch]().eval()
    x = torch.randn(1,3,imgsz,imgsz)
    flops, params = profile(m, inputs=(x,), verbose=False)
    return dict(params_m=params/1e6, flops_g=flops/1e9)

def scrape_latency_and_mem(reports_root, dataset, tag):
    # returns p50 latency CPU/GPU and peak_cuda_mib if present
    d = {"lat_cpu_ms": None, "lat_gpu_ms": None, "peak_cuda_mib": None}
    # search any latency table already exported by your reports
    for p in glob.glob(os.path.join(reports_root, f"phase*_{dataset}", f"*{tag}*", "*latency*.csv")):
        df = pd.read_csv(p)
        if "device" in df.columns and "lat_ms_p50" in df.columns:
            if "cpu" in df["device"].values:
                d["lat_cpu_ms"] = float(df[df["device"]=="cpu"]["lat_ms_p50"].median())
            if "cuda" in df["device"].values:
                d["lat_gpu_ms"] = float(df[df["device"]=="cuda"]["lat_ms_p50"].median())
        if "peak_cuda_mib" in df.columns:
            d["peak_cuda_mib"] = float(df["peak_cuda_mib"].max())
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-list", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--out-tables", default="./tables")
    ap.add_argument("--out-figs", default="./figs")
    ap.add_argument("--reports-root", default="./reports")
    args = ap.parse_args()

    os.makedirs(args.out_tables, exist_ok=True)
    rows = []
    for ds in args.dataset_list.split(","):
        for tag in args.models.split(","):
            pf = params_flops(tag, args.imgsz)
            latency = scrape_latency_and_mem(args.reports_root, ds, tag)
            rows.append({"dataset": ds, "model": tag, **pf, **latency})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_tables, "efficiency_summary.csv"), index=False)
    df.to_latex(os.path.join(args.out_tables, "efficiency_summary.tex"), float_format="%.3f", index=False)
    print("[OK] Efficiency table written.")

if __name__ == "__main__":
    main()
