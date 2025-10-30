import argparse, os, numpy as np, torch, cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF

def to_scalar(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return int(np.array(y).squeeze())

def denorm_to_uint8(x):
    """x: tensor [C,H,W], normalized ImageNet -> uint8 HxWx3"""
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
    x = x.detach().cpu()*std + mean
    x = (x.clamp(0,1)*255.0).byte()
    return x.permute(1,2,0).numpy()

def as_4d(x, device):
    if isinstance(x, torch.Tensor):
        if x.ndim == 3: x = x.unsqueeze(0)
        return x.to(device)
    # PIL -> tensor
    x = transforms.ToTensor()(x)
    # normalize
    x = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(x)
    return x.unsqueeze(0).to(device)

def gradcam(model, x4d, target_layer, class_idx):
    model.eval()
    features = {}
    grads = {}
    def fwd_hook(m,i,o): features['feat']=o
    def bwd_hook(m,gi,go): grads['grad']=go[0]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x4d)               # [1,C]
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=True)

    A = features['feat'].detach()     # [1,K,H,W]
    G = grads['grad'].detach()        # [1,K,H,W]
    weights = G.flatten(2).mean(dim=2)  # [1,K]
    cam = (weights[:,:,None,None] * A).sum(dim=1)   # [1,H,W]
    cam = torch.relu(cam)[0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-6)
    h1.remove(); h2.remove()
    return cam, logits.detach()

def overlay(rgb_u8, cam, alpha=0.35):
    # Always operate at 224x224
    H, W = 224, 224
    if rgb_u8.shape[:2] != (H, W):
        rgb_u8 = cv2.resize(rgb_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_up = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255*cam_up), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (1-alpha)*rgb_u8 + alpha*heat
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reports-root", default="./reports")
    ap.add_argument("--models", required=True)    # students: "resnet18,mbv2,effb0"
    ap.add_argument("--teacher", default="resnet50")
    ap.add_argument("--samples-per-class", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-figs", default="./figs")
    args = ap.parse_args()

    from datasets import get_dataset_split, class_names_for
    from models import load_model_by_tag, pick_last_layer

    classes = class_names_for(args.dataset)
    C = len(classes)
    testset = get_dataset_split(args.dataset, "test")

    # Collect up to K indices per class
    idxs_per_class = {c:[] for c in range(C)}
    for i in range(len(testset)):
        _, y = testset[i]
        y = to_scalar(y)
        if 0 <= y < C and len(idxs_per_class[y]) < args.samples_per_class:
            idxs_per_class[y].append(i)
        if all(len(v) >= args.samples_per_class for v in idxs_per_class.values()):
            break

    # Fix number of columns to the minimum available across classes
    n_cols = min(len(v) for v in idxs_per_class.values())
    if n_cols == 0:
        print(f"[WARN] Not enough samples to render Grad-CAM for {args.dataset}.")
        return

    os.makedirs(os.path.join(args.out_figs, args.dataset), exist_ok=True)
    tags = [args.teacher] + args.models.split(",")

    TILE_W = TILE_H = 224

    for tag in tags:
        model = load_model_by_tag(tag, args.dataset, device=args.device)
        tlayer = pick_last_layer(model)

        rows = []
        for c in range(C):
            row_tiles = []
            # Use exactly n_cols indices per class
            for i in idxs_per_class[c][:n_cols]:
                x, _ = testset[i]
                x4d = as_4d(x, args.device)
                cam, logits = gradcam(model, x4d, tlayer, c)

                # base RGB (force 224x224)
                if isinstance(x, torch.Tensor):
                    rgb = denorm_to_uint8(x)
                else:
                    rgb = np.array(x)
                rgb = cv2.resize(rgb, (TILE_W, TILE_H), interpolation=cv2.INTER_LINEAR)

                over = overlay(rgb, cam)
                row_tiles.append(over)

            # Safety: if a row is short for any reason, right-pad with blanks
            while len(row_tiles) < n_cols:
                row_tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

            rows.append(np.concatenate(row_tiles, axis=1))  # same width for every row

        panel = np.concatenate(rows, axis=0)  # stack rows -> consistent width
        outp = os.path.join(args.out_figs, args.dataset, f"{tag}_gradcam_panel.png")
        cv2.imwrite(outp, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print("[OK] Grad-CAM panel:", outp)

if __name__ == "__main__":
    main()
