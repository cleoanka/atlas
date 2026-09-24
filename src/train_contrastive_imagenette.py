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


# ------------------------------------------------------------------ ResNet (pure torch, 18 & 50)
class Block(nn.Module):
    """BasicBlock (ResNet-18/34)."""
    exp = 1

    def __init__(s, cin, width, stride=1):
        super().__init__()
        cout = width * s.exp
        s.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(cout)
        s.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(cout)
        s.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            s.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(s, x):
        y = F.relu(s.b1(s.c1(x))); y = s.b2(s.c2(y))
        return F.relu(y + s.sc(x))


class Bottleneck(nn.Module):
    """Bottleneck block (ResNet-50/101)."""
    exp = 4

    def __init__(s, cin, width, stride=1):
        super().__init__()
        cout = width * s.exp
        s.c1 = nn.Conv2d(cin, width, 1, bias=False); s.b1 = nn.BatchNorm2d(width)
        s.c2 = nn.Conv2d(width, width, 3, stride, 1, bias=False); s.b2 = nn.BatchNorm2d(width)
        s.c3 = nn.Conv2d(width, cout, 1, bias=False); s.b3 = nn.BatchNorm2d(cout)
        s.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            s.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(s, x):
        y = F.relu(s.b1(s.c1(x))); y = F.relu(s.b2(s.c2(y))); y = s.b3(s.c3(y))
        return F.relu(y + s.sc(x))


class ResNet(nn.Module):
    """Generalised ResNet. real-ResNet stem (7×7 s2 + 3×3 s2 maxpool) → 4× downsample first
    (skipping it is what OOM'd MPS: full-res activations). out_dim = last channels."""

    def __init__(s, block, layers, widths=(64, 128, 256, 512)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
                               nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        cin, blocks = 64, []
        for i, (w, n) in enumerate(zip(widths, layers)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(block(cin, w, stride)); cin = w * block.exp
        s.blocks = nn.Sequential(*blocks)
        s.out_dim = cin

    def forward(s, x):
        x = s.blocks(s.stem(x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def make_backbone(name="resnet18"):
    if name == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3])     # out_dim 2048
    return ResNet(Block, [2, 2, 2, 2])              # out_dim 512      # (B, 512)


# ------------------------------------------------------------------ GPU-side SimCLR augmentation
def _gaussian_blur(x, p=0.5, k=9):
    """SimCLR blur: depthwise Gaussian conv on a random p-fraction of the batch (random sigma)."""
    B, C, H, W = x.shape
    dev = x.device
    sigma = 0.1 + torch.rand(1, device=dev).item() * 1.9
    ax = torch.arange(k, device=dev) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    ker = (g[:, None] * g[None, :]).view(1, 1, k, k).repeat(C, 1, 1, 1)
    xb = F.conv2d(x, ker, padding=k // 2, groups=C)
    m = (torch.rand(B, 1, 1, 1, device=dev) < p).float()
    return x * (1 - m) + xb * m


def augment(x, out):
    """x: (B,3,H,W) float[0,1] -> a differently-augmented (B,3,out,out): RRC+flip + colour jitter + blur."""
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
    # gaussian blur (SimCLR): random half of the batch
    x = _gaussian_blur(x, p=0.5)
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
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--tau", type=float, default=0.2, help="NT-Xent temperature (lower=harder negatives, e.g. 0.1)")
    a = ap.parse_args()
    use_amp = (DEV == "cuda")                                  # mixed precision: big speed/VRAM win on NVIDIA
    print(f"device={DEV}  backbone={a.backbone}  img={a.img}  batch={a.batch}  "
          f"epochs={a.epochs}  amp={use_amp}", flush=True)

    base = int(a.img * 1.15)                                   # a little headroom for random crop
    Xtr_np = load_train(base)
    Xtr = torch.from_numpy(Xtr_np).to(DEV).permute(0, 3, 1, 2).contiguous()  # uint8 (N,3,base,base)
    n = Xtr.shape[0]
    print(f"train tensor {tuple(Xtr.shape)} on {DEV}", flush=True)

    enc = make_backbone(a.backbone).to(DEV)
    dim = enc.out_dim
    head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 128)).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), a.lr, weight_decay=1e-6)
    steps = a.epochs * (n // a.batch)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dev = "cuda" if DEV == "cuda" else "cpu"

    ckpt = os.path.join(_ROOT, "models", "imagenette_encoder.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    def save_ckpt(ep_done):
        torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                    "arch": a.backbone, "img": a.img, "epochs": a.epochs,
                    "epochs_done": ep_done, "dim": dim}, ckpt)

    def embed(Xu8):
        """Embed uint8 images (center region, no aug), L2-normalised. Assumes enc in eval mode."""
        X = torch.from_numpy(Xu8).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                outs.append(F.normalize(enc(X[i:i + 256]), dim=1))
        return torch.cat(outs).cpu().numpy().astype("float32")

    # Validation pool, cached once. We export its embedding at EVERY checkpoint (not just at the
    # end) so the optimum epoch can be read off downstream — SSL contrastive loss never shows an
    # overfitting bump, so "when to stop" is a downstream question (edge purity / few-label acc vs
    # epoch), answered by figures/make_imagenette_epoch_scan.py on these per-epoch embeddings.
    Xev_np, yev = D.eval_cache(a.img)
    data_dir = os.path.join(_ROOT, "data")

    def export_val_emb(path):
        was_training = enc.training
        enc.eval()
        np.savez_compressed(path, emb=embed(Xev_np), y=yev)
        if was_training:
            enc.train()

    enc.train(); head.train()
    for ep in range(a.epochs):
        perm = torch.randperm(n, device=DEV); t0 = time.time(); tot = 0.0; nb = 0
        for i in range(0, n - a.batch + 1, a.batch):
            xb = Xtr[perm[i:i + a.batch]].float() / 255.0
            v1 = augment(xb, a.img); v2 = augment(xb, a.img)
            opt.zero_grad()
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                loss = nt_xent(head(enc(v1)), head(enc(v2)), tau=a.tau)
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sch.step(); tot += loss.item(); nb += 1
        print(f"  ep{ep+1}/{a.epochs} loss={tot/nb:.4f} {time.time()-t0:.0f}s", flush=True)
        if (ep + 1) % 25 == 0:                      # periodic checkpoint — survive a preemption
            save_ckpt(ep + 1)
            export_val_emb(os.path.join(data_dir, f"imagenette_emb_ep{ep+1}.npz"))
            print(f"  [ckpt @ ep{ep+1}  +val-emb]", flush=True)

    # final encoder weights (local keepsake; .pt is gitignored)
    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                "arch": a.backbone, "img": a.img, "epochs": a.epochs, "dim": dim}, ckpt)
    print(f"saved -> models/imagenette_encoder.pt", flush=True)

    # validation pool → the evaluation embedding (shipped)
    enc.eval()
    np.savez_compressed(os.path.join(data_dir, "imagenette_emb.npz"),
                        emb=embed(Xev_np), y=yev)
    print(f"done -> imagenette_emb.npz ({len(yev)}, 512)", flush=True)
    # train pool (center-crop to img) → embeddings for the proper full-train linear-probe ceiling
    _, ytr = D.train_paths()
    c = (base - a.img) // 2
    Xtr_crop = Xtr_np[:, c:c + a.img, c:c + a.img, :]
    np.savez_compressed(os.path.join(_ROOT, "data", "imagenette_train_emb.npz"),
                        emb=embed(Xtr_crop), y=ytr)
    print(f"done -> imagenette_train_emb.npz ({len(ytr)}, 512)", flush=True)


if __name__ == "__main__":
    main()
