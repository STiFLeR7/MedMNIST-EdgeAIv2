import argparse, os, numpy as np, pandas as pd, torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def to_scalar(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return int(np.array(y).squeeze())

def orthogonal_procrustes(A, B, scale=True):
    """
    Align A -> B in least-squares sense (both Nx2 after t-SNE).
    Returns A_aligned with optional isotropic scaling.
    """
    A0 = A - A.mean(0, keepdims=True)
    B0 = B - B.mean(0, keepdims=True)
    if scale:
        sA = np.sqrt((A0**2).sum()); sB = np.sqrt((B0**2).sum())
        A0 = A0 / (sA + 1e-12); B0 = B0 / (sB + 1e-12)
    C = A0.T @ B0
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt
    A_hat = A0 @ R
    if scale:
        A_hat *= (sB / (sA + 1e-12))
    A_hat += B.mean(0, keepdims=True)
    return A_hat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reports-root", default="./reports")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--students", required=True)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-figs", default="./figs")
    ap.add_argument("--out-tables", default="./tables")
    args = ap.parse_args()

    from datasets import get_dataset_split, class_names_for
    from models import load_model_by_tag, penultimate
    classes = class_names_for(args.dataset)
    testset = get_dataset_split(args.dataset, "test")

    n = len(testset)
    k = min(args.samples, n)
    rng = np.random.RandomState(0)
    idxs = rng.choice(n, size=k, replace=False)

    os.makedirs(os.path.join(args.out_figs,args.dataset), exist_ok=True)
    os.makedirs(os.path.join(args.out_tables,args.dataset), exist_ok=True)

    tags = [args.teacher] + args.students.split(",")
    feats = {}
    labels = np.array([to_scalar(testset[i][1]) for i in idxs], dtype=np.int64)

    # 1) Extract features per model (dims may differ across models)
    for tag in tags:
        m = load_model_by_tag(tag, args.dataset, device=args.device).eval()
        Z = []
        for i in idxs:
            x, _ = testset[i]
            if isinstance(x, torch.Tensor):
                x_t = x.unsqueeze(0).to(args.device)
            else:
                import torchvision.transforms as T
                x_t = T.Compose([
                    T.Resize((224,224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])(x).unsqueeze(0).to(args.device)
            z = penultimate(m, x_t).detach().cpu().numpy()
            Z.append(z[0])
        feats[tag] = np.vstack(Z)  # shape: [k, D_tag]

    # 2) Run t-SNE **independently per model** to 2D
    coords = {}
    for tag in tags:
        ts = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0)
        coords[tag] = ts.fit_transform(feats[tag])  # [k,2]

    # 3) Align students to teacher with Procrustes so plots share a frame
    T2 = coords[tags[0]]
    for tag in tags[1:]:
        coords[tag] = orthogonal_procrustes(coords[tag], T2, scale=True)

    # 4) Plot teacher vs each student
    for tag in tags[1:]:
        fig=plt.figure(figsize=(6,5))
        plt.scatter(T2[:,0], T2[:,1], s=8, alpha=0.35, label=f"{tags[0]}")
        plt.scatter(coords[tag][:,0], coords[tag][:,1], s=8, alpha=0.35, label=f"{tag}")
        plt.title(f"t-SNE: {args.dataset} — {tags[0]} vs {tag}")
        plt.legend()
        fig.savefig(os.path.join(args.out_figs,args.dataset,f"tsne_{tags[0]}_vs_{tag}.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)

    # 5) Centroid alignment distances (after alignment)
    rows=[]
    C = len(classes)
    for tag in tags[1:]:
        for c in range(C):
            mask = labels==c
            if mask.sum() == 0:
                continue
            mu_T = T2[mask].mean(axis=0)
            mu_S = coords[tag][mask].mean(axis=0)
            d = float(np.linalg.norm(mu_T - mu_S))
            rows.append({"model": tag, "class": classes[c], "tsne_centroid_dist": d})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_tables,args.dataset,"tsne_centroid_alignment.csv"), index=False)
    df.to_latex(os.path.join(args.out_tables,args.dataset,"tsne_centroid_alignment.tex"), float_format="%.3f", index=False)
    print("[OK] t-SNE & centroid alignment exported.")

if __name__ == "__main__":
    main()
