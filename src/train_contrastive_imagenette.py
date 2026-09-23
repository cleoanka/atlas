"""Unsupervised contrastive encoder for ImageNette (SimCLR / NT-Xent, ResNet-18, pure torch).

    python src/train_contrastive_imagenette.py --epochs 120 --img 128 --batch 256

No labels are used: the only signal is "two augmentations of the same photo embed together".
Trains on the 9469 ImageNette train images, exports an L2-normalised embedding of the 3925
validation images -> data/imagenette_emb.npz (labels attached only for evaluation).
Device auto: MPS -> CUDA -> CPU.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imagenette_data as D  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
DEV = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ ResNet-18 (pure torch)
class Block(nn.Module):
    def __init__(s, cin, cout, stride=1):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(cout)
        s.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(cout)
        s.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            s.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(s, x):
        y = F.relu(s.b1(s.c1(x))); y = s.b2(s.c2(y))
        return F.relu(y + s.sc(x))


class ResNet18(nn.Module):
    def __init__(s, dim=512):
        super().__init__()
        # real-ResNet stem: 7×7 s2 conv + 3×3 s2 maxpool → 4× downsample before the blocks
        # (skipping this is what OOM'd MPS: full-res 128×128×64 activations at batch 256).
        s.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1))
        cfg = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
               (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
        s.blocks = nn.Sequential(*[Block(a, b, st) for a, b, st in cfg])
        s.dim = dim

    def forward(s, x):
        x = s.blocks(s.stem(x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)      # (B, 512)


# ------------------------------------------------------------------ GPU-side SimCLR augmentation
def augment(x, out):
    """x: (B,3,H,W) float[0,1] -> a differently-augmented (B,3,out,out): RRC+flip + colour jitter."""
    B = x.shape[0]
    dev = x.device
    # random-resized-crop + horizontal flip via a random affine grid
    scale = 0.5 + torch.rand(B, device=dev) * 0.5                 # crop covers 50-100%
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - scale)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - scale)
    zero = torch.zeros(B, device=dev)
    th = torch.stack([torch.stack([scale * flip, zero, tx], 1),
                      torch.stack([zero, scale, ty], 1)], 1)
    grid = F.affine_grid(th, (B, 3, out, out), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
    # colour jitter: brightness, contrast, saturation (per-image factors)
    br = (1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.4)
    x = (x * br).clamp(0, 1)
    ct = (1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.4)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * ct + mean).clamp(0, 1)
    gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    sat = (1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.4)
    x = (gray + (x - gray) * sat).clamp(0, 1)
    # random grayscale (whole image) with prob 0.2
    g = (torch.rand(B, 1, 1, 1, device=dev) < 0.2).float()
    x = x * (1 - g) + gray.expand_as(x) * g
    return x


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1); N = z1.shape[0]
    sim = z @ z.T / tau
    sim = sim - torch.eye(2 * N, device=z.device) * 1e9
    tgt = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def load_train(base):
    """Preload all train images resized to base×base as a uint8 tensor (cached)."""
    cache = os.path.join(_ROOT, "data", f"imagenette_train_{base}.npz")
    if os.path.exists(cache):
        return np.load(cache)["X"]
    paths, _ = D.train_paths()
    X = np.zeros((len(paths), base, base, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        X[i] = D._load_resized(p, base)
        if i % 1000 == 0:
            print(f"  train cache {i}/{len(paths)}", flush=True)
    np.savez_compressed(cache, X=X)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    print(f"device={DEV}  img={a.img}  batch={a.batch}  epochs={a.epochs}", flush=True)

    base = int(a.img * 1.15)                                   # a little headroom for random crop
    Xtr_np = load_train(base)
    Xtr = torch.from_numpy(Xtr_np).to(DEV).permute(0, 3, 1, 2).contiguous()  # uint8 (N,3,base,base)
    n = Xtr.shape[0]
    print(f"train tensor {tuple(Xtr.shape)} on {DEV}", flush=True)

    enc = ResNet18().to(DEV)
    head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), a.lr, weight_decay=1e-6)
    steps = a.epochs * (n // a.batch)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)

    enc.train(); head.train()
    for ep in range(a.epochs):
        perm = torch.randperm(n, device=DEV); t0 = time.time(); tot = 0.0; nb = 0
        for i in range(0, n - a.batch + 1, a.batch):
            xb = Xtr[perm[i:i + a.batch]].float() / 255.0
            v1 = augment(xb, a.img); v2 = augment(xb, a.img)
            loss = nt_xent(head(enc(v1)), head(enc(v2)))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item(); nb += 1
        print(f"  ep{ep+1}/{a.epochs} loss={tot/nb:.4f} {time.time()-t0:.0f}s", flush=True)

    # embed the validation pool (no aug), L2-normalised
    enc.eval()
    Xev_np, yev = D.eval_cache(a.img)
    Xev = torch.from_numpy(Xev_np).to(DEV).permute(0, 3, 1, 2).float() / 255.0
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xev), 256):
            outs.append(F.normalize(enc(Xev[i:i + 256]), dim=1))
    emb = torch.cat(outs).cpu().numpy().astype("float32")
    out = os.path.join(_ROOT, "data", "imagenette_emb.npz")
    np.savez_compressed(out, emb=emb, y=yev)
    print(f"done -> imagenette_emb.npz {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
