# tools/ablate_kd.py
import argparse, os, glob, re, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt

HP_REGEX = re.compile(r".*alpha(?P<alpha>[0-9.]+)_tau(?P<tau>[0-9.]+)_beta(?P<beta>[0-9.]+)")

def find_runs(reports_root, dataset):
    roots = sorted(glob.glob(os.path.join(reports_root, f"phase*_{dataset}", "student_*")))
    runs = []
    for r in roots:
        m = HP_REGEX.match(r.replace("\\","/"))
        if not m: 
            # also check for metrics.json carrying hp
            met = glob.glob(os.path.join(r, "metrics.json"))
            hp = dict(alpha=None,tau=None,beta=None)
            if met:
                try:
                    js = json.load(open(met[0],"r"))
                    hp["alpha"]=js.get("kd_alpha"); hp["tau"]=js.get("kd_tau"); hp["beta"]=js.get("at_beta")
                except: pass
            # pick scalars if present
            test = glob.glob(os.path.join(r,"test_summary.json")) + glob.glob(os.path.join(r,"metrics.json"))
            acc=f1=ece=lat=None
            if test:
                try:
                    js=json.load(open(test[0],"r"))
                    acc=js.get("test_acc") or js.get("acc"); f1=js.get("macro_f1"); ece=js.get("ece")
                except: pass
            lat = _lat_from_csv(r)
            runs.append({**hp,"path":r,"acc":acc,"macro_f1":f1,"ece":ece,"lat_ms":lat})
            continue

        hp = {k: float(v) for k,v in m.groupdict().items()}
        test = glob.glob(os.path.join(r,"test_summary.json")) + glob.glob(os.path.join(r,"metrics.json"))
        acc=f1=ece=lat=None
        if test:
            try:
                js=json.load(open(test[0],"r"))
                acc=js.get("test_acc") or js.get("acc"); f1=js.get("macro_f1"); ece=js.get("ece")
            except: pass
        lat = _lat_from_csv(r)
        runs.append({**hp,"path":r,"acc":acc,"macro_f1":f1,"ece":ece,"lat_ms":lat})
    return pd.DataFrame(runs)

def _lat_from_csv(run_dir):
    latcsv = glob.glob(os.path.join(run_dir, "latency.csv"))
    if not latcsv: return None
    try:
        df = pd.read_csv(latcsv[0])
        if "lat_ms_p50" in df.columns: 
            return float(df[df["batch"]==1]["lat_ms_p50"].median())
        return float(df["lat_ms_mean"].median())
    except: 
        return None

def heatmap(df, xkey, ykey, zkey, out):
    piv = df.pivot_table(index=ykey, columns=xkey, values=zkey, aggfunc="max")
    fig=plt.figure(figsize=(6,5)); plt.imshow(piv.values, aspect="auto")
    plt.xticks(range(len(piv.columns)), [str(x) for x in piv.columns], rotation=45, ha="right")
    plt.yticks(range(len(piv.index)), [str(y) for y in piv.index])
    plt.colorbar(); plt.title(f"{zkey} vs {xkey},{ykey}")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)

def lines(df, key, out):
    fig=plt.figure(figsize=(7,5))
    for grouping, g in df.groupby(["beta"]):
        g=g.sort_values(key)
        plt.plot(g[key], g["macro_f1"], label=f"β={grouping}")
    plt.xlabel(key); plt.ylabel("Macro-F1"); plt.legend()
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)

def pareto(df, out):
    fig=plt.figure(figsize=(6,5))
    plt.scatter(df["lat_ms"], df["macro_f1"])
    plt.xlabel("Latency p50 (ms, batch=1)"); plt.ylabel("Macro-F1")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="Comma list")
    ap.add_argument("--reports-root", default="./reports")
    ap.add_argument("--out-figs", default="./figs")
    ap.add_argument("--out-tables", default="./tables")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    for ds in args.datasets.split(","):
        df = find_runs(args.reports_root, ds)
        if df.empty:
            print(f"[INFO] No ablation runs detected under {args.reports_root} for {ds}.")
            continue
        tab_dir = os.path.join(args.out_tables, ds); os.makedirs(tab_dir, exist_ok=True)
        fig_dir = os.path.join(args.out_figs, ds); os.makedirs(fig_dir, exist_ok=True)

        df.to_csv(os.path.join(tab_dir, "ablation_grid.csv"), index=False)
        df.to_latex(os.path.join(tab_dir, "ablation_grid.tex"), float_format="%.3f", index=False)

        for z in ["macro_f1","acc","ece"]:
            heatmap(df.dropna(subset=["alpha","tau",z]), "alpha","tau",z, os.path.join(fig_dir, f"ablation_heatmap_{z}.png"))
        lines(df.dropna(subset=["alpha","macro_f1","beta"]), "alpha", os.path.join(fig_dir,"ablation_lines_vs_alpha.png"))
        lines(df.dropna(subset=["tau","macro_f1","beta"]), "tau", os.path.join(fig_dir,"ablation_lines_vs_tau.png"))
        pareto(df.dropna(subset=["macro_f1","lat_ms"]), os.path.join(fig_dir, "ablation_pareto_f1_vs_latency.png"))
        print(f"[OK] Ablation aggregation & plots done for {ds}.")

if __name__ == "__main__":
    main()
