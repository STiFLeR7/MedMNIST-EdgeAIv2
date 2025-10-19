# scripts/verify_and_sanity_test.py
import torch, torchvision
from pathlib import Path
import json
import sys

project_root = Path(__file__).resolve().parents[1]
manifest = project_root / "models" / "checkpoints-manifest.json"
if not manifest.exists():
    print("Manifest missing. Run fetch_checkpoints.ps1 first.", file=sys.stderr)
    sys.exit(2)

data = json.loads(manifest.read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

mapping = {
    "resnet50-0676ba61.pth": ("resnet50", torchvision.models.resnet50),
    "resnet18-f37072fd.pth": ("resnet18", torchvision.models.resnet18),
    "mobilenet_v2-b0353104.pth": ("mobilenet_v2", torchvision.models.mobilenet_v2),
    "efficientnet_b0_rwightman-3dd342df.pth": ("efficientnet_b0", torchvision.models.efficientnet_b0),
}

for entry in data:
    fname = Path(entry["path"]).name
    if fname not in mapping:
        print("Skipping unknown:", fname); continue
    name, ctor = mapping[fname]
    print(f"\n--- Testing {fname} as {name} ---")
    model = ctor(weights=None)   # create architecture without auto-download
    sd = torch.load(entry["path"], map_location="cpu")
    # if saved as a state_dict with 'state_dict' key, handle both cases:
    if isinstance(sd, dict) and 'state_dict' in sd and len(sd) > 1:
        sd = sd['state_dict']
    # convert keys if they have 'module.' prefix from DataParallel
    new_sd = {}
    for k,v in sd.items():
        nk = k
        if k.startswith('module.'):
            nk = k[len('module.'):]
        new_sd[nk] = v
    try:
        model.load_state_dict(new_sd)
        model = model.to(device)
        model.eval()
        # count params
        total = sum(p.numel() for p in model.parameters())
        print(f"Loaded OK. Params: {total:,}")
        # tiny forward
        b = torch.randn(1,3,224,224).to(device)
        with torch.no_grad():
            out = model(b)
        print("Forward OK. Output shape:", tuple(out.shape))
    except Exception as e:
        print("ERROR loading/forward for", fname, e)
