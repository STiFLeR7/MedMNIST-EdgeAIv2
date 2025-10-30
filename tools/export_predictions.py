import argparse, os, numpy as np, torch, sys
from pathlib import Path

def npz_path(reports_root, dataset, model_tag):
    def model_dir_name(dataset, model_tag):
        return f"teacher_{model_tag}_{dataset}" if model_tag == "resnet50" else f"student_{model_tag}_{dataset}"
    candidates = [
        Path(reports_root)/f"phase1_{dataset}"/model_dir_name(dataset, model_tag),
        Path(reports_root)/f"phase2_{dataset}"/model_dir_name(dataset, model_tag),
        Path(reports_root)/f"phase3_{dataset}"/model_dir_name(dataset, model_tag),
        Path(reports_root)/dataset/model_dir_name(dataset, model_tag),
    ]
    for c in candidates:
        if c.exists():
            (c/"inference_cache").mkdir(parents=True, exist_ok=True)
            return c/"inference_cache"/"test_predictions.npz"
    outdir = Path(reports_root)/dataset/model_dir_name(dataset, model_tag)/"inference_cache"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir/"test_predictions.npz"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ham10000","medmnist","isic","oct2017"])
    ap.add_argument("--model", required=True, choices=["resnet50","resnet18","mbv2","effb0"])
    ap.add_argument("--out-root", default="./reports")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-missing-only", action="store_true")
    args = ap.parse_args()

    out_npz = npz_path(args.out_root, args.dataset, args.model)
    if args.force_missing_only and out_npz.exists():
        print(f"[SKIP] Predictions exist: {out_npz}")
        return

    from datasets import get_dataset_split
    from models import load_model_by_tag
    ds = get_dataset_split(args.dataset, split="test")
    n = len(ds)
    m = load_model_by_tag(args.model, args.dataset, device=args.device).eval()

    def to_scalar(y):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        return int(np.array(y).squeeze())

    bs = 64
    y_true = np.zeros(n, dtype=np.int64)
    probs_list = []

    bad = 0
    with torch.inference_mode():
        i = 0
        while i < n:
            xs, ys = [], []
            # fill a batch robustly
            while len(xs) < bs and i < n:
                try:
                    x, y = ds[i]
                    xs.append(x)
                    ys.append(to_scalar(y))
                except Exception as e:
                    bad += 1
                    # Skip unreadable/slow sample and continue
                finally:
                    i += 1
            if not xs:
                continue
            x_t = torch.stack(xs, dim=0).to(args.device, non_blocking=True)
            logits = m(x_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
            start = (i - len(xs))
            y_true[start:start+len(xs)] = np.array(ys, dtype=np.int64)
            if i % (bs*20) == 0:
                print(f"[{args.dataset}/{args.model}] {i}/{n} processed (skipped {bad})")

    if not probs_list:
        print(f"[ERROR] No predictions produced for {args.dataset}/{args.model}.")
        sys.exit(2)

    y_prob = np.concatenate(probs_list, axis=0)
    # y_true might have trailing zeros if we skipped tail-only samples; trim to y_prob length
    y_true = y_true[:y_prob.shape[0]]

    np.savez_compressed(out_npz, y_true=y_true, y_prob=y_prob)
    print(f"[OK] wrote {out_npz} :: N={y_prob.shape[0]}, C={y_prob.shape[1]} (skipped {bad})")

if __name__ == "__main__":
    main()
