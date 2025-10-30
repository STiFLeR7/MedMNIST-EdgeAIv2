import argparse, os, json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
import torch.nn.functional as F

def ece_score(probs, y, n_bins=15):
    # probs: [N,C], y: [N]
    conf = probs.max(1)
    preds = probs.argmax(1)
    bins = torch.linspace(0,1,n_bins+1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.sum() == 0: continue
        acc = (preds[m]==y[m]).float().mean()
        conf_m = conf[m].mean()
        ece += (m.float().mean()) * (conf_m - acc).abs()
    return ece.item()

def kd_loss(student_logits, teacher_logits, y, alpha, tau):
    ce = F.cross_entropy(student_logits, y)
    if alpha == 0.0: return ce
    # KL div with temperature
    p = F.log_softmax(student_logits / tau, dim=1)
    q = F.softmax(teacher_logits / tau, dim=1)
    kl = F.kl_div(p, q, reduction="batchmean") * (tau*tau)
    return (1 - alpha) * ce + alpha * kl

def at_loss(student_feat, teacher_feat):
    # Attention Transfer (Zagoruyko & Komodakis): match normalized spatial maps
    def attn(x):  # [B,C,H,W] -> [B,1,H,W]
        a = x.pow(2).mean(dim=1, keepdim=True)
        a = a / (a.norm(p=2, dim=(2,3), keepdim=True) + 1e-8)
        return a
    return F.mse_loss(attn(student_feat), attn(teacher_feat))

def pick_hook_layer(m):
    # same logic as pick_last_layer()
    if hasattr(m, "layer4"): return m.layer4[-1]
    if hasattr(m, "features"): return list(m.features.children())[-1]
    return next(reversed([mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ham10000","medmnist","isic"])
    ap.add_argument("--teacher", required=True, choices=["resnet50"])
    ap.add_argument("--student", required=True, choices=["resnet18","mbv2","effb0"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default="./reports")
    ap.add_argument("--save-tag", default=None)
    args = ap.parse_args()

    from datasets import get_dataset_split, class_names_for
    from models import load_model_by_tag, build_arch

    device = args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu"
    classes = class_names_for(args.dataset); C = len(classes)

    # data
    trainset = get_dataset_split(args.dataset, "train")
    valset   = get_dataset_split(args.dataset, "val")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # models
    t = load_model_by_tag(args.teacher, args.dataset, device=device).eval()
    s = build_arch(args.student, C).to(device).train()

    # attention taps (optional)
    if args.beta > 0:
        tl = pick_hook_layer(t); sl = pick_hook_layer(s)
        feat_t, feat_s = {}, {}
        tl.register_forward_hook(lambda m,i,o: feat_t.setdefault("z", o))
        sl.register_forward_hook(lambda m,i,o: feat_s.setdefault("z", o))

    opt = torch.optim.AdamW(s.parameters(), lr=3e-4, weight_decay=1e-4)
    out_dir = Path(args.out_root)/f"ablation_{args.dataset}"/(args.save_tag or f"KD_a{args.alpha}_t{args.tau}_b{args.beta}_{args.student}_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_path = -1.0, None
    metrics_path = out_dir/"metrics.jsonl"

    for ep in range(1, args.epochs+1):
        s.train()
        for x,y in train_loader:
            x = x.to(device); y = y.long().to(device)
            with torch.no_grad():
                t_logits = t(x)
            s_logits = s(x)
            loss = kd_loss(s_logits, t_logits, y, args.alpha, args.tau)
            if args.beta > 0:
                # forward already captured features
                loss = loss + args.beta * at_loss(feat_s["z"], feat_t["z"])
                feat_t.clear(); feat_s.clear()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        # validate
        s.eval()
        y_true, y_prob = [], []
        with torch.inference_mode():
            for x,y in val_loader:
                x = x.to(device); y = y.long().to(device)
                logits = s(x)
                probs = torch.softmax(logits, dim=1)
                y_true.append(y.cpu()); y_prob.append(probs.cpu())
        y_true = torch.cat(y_true).numpy()
        y_prob = torch.cat(y_prob).numpy()
        y_pred = y_prob.argmax(1)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        ece = ece_score(torch.from_numpy(y_prob), torch.from_numpy(y_true))

        rec = {"epoch": ep, "macro_f1": float(macro_f1), "ece": float(ece),
               "alpha": args.alpha, "tau": args.tau, "beta": args.beta,
               "student": args.student, "dataset": args.dataset}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec)+"\n")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_path = out_dir/f"model_best.pth"
            torch.save(s.state_dict(), best_path)

    print(f"[OK] KD trained: {out_dir} best_f1={best_f1:.3f} ckpt={best_path}")

if __name__ == "__main__":
    main()
