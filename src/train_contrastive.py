"""Unsupervised contrastive encoder (SimCLR / NT-Xent). Device auto: MPS → CUDA → CPU.

    python src/train_contrastive.py 40      # ~5-10 min on an Apple-Silicon GPU

Writes contrastive_emb.npz (L2-normalised embedding of the 10k evaluation split) and
eval_split.npz (the fixed split indices + labels). Both are shipped in the repo so you do NOT
need a GPU to reproduce the headline — this script only matters if you want to retrain.

No labels are used anywhere here: the only signal is "two augmentations of the same image
should embed close together". Labels enter only at evaluation time (src/evaluate.py).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
DEV = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_ROOT, "mnist.npz")


def load_mnist():
    if not os.path.exists(CACHE):
        from sklearn.datasets import fetch_openml
        print("downloading MNIST (once)...", flush=True)
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        np.savez_compressed(CACHE, X=d.data.astype("float32") / 255.0, y=d.target.astype(int))
    d = np.load(CACHE)
    return d["X"], d["y"]


def augment(x):
    B = x.shape[0]
    ang = (torch.rand(B, device=x.device) * 2 - 1) * (18 * np.pi / 180)
    sc = 1 + (torch.rand(B, device=x.device) * 2 - 1) * 0.18
    tx = (torch.rand(B, device=x.device) * 2 - 1) * 0.22
    ty = (torch.rand(B, device=x.device) * 2 - 1) * 0.22
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    o = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                      align_corners=False, padding_mode="zeros")
    r0 = torch.randint(0, 21, (B,), device=x.device); r1 = torch.randint(0, 21, (B,), device=x.device)
    ar = torch.arange(28, device=x.device)
    m = ~(((ar[None] >= r0[:, None]) & (ar[None] < r0[:, None] + 8))[:, :, None] &
          ((ar[None] >= r1[:, None]) & (ar[None] < r1[:, None] + 8))[:, None, :])
    return (o * m[:, None].float() + torch.randn_like(o) * 0.08).clamp(0, 1)


class Enc(nn.Module):
    def __init__(s, dim=128):
        super().__init__()
        s.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 4 * 4, dim))

    def forward(s, x):
        return s.f(x)


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    N = z1.shape[0]
    sim = z @ z.T / tau
    sim = sim - torch.eye(2 * N, device=z.device) * 1e9
    tgt = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def main(epochs: int = 40):
    print("device:", DEV, flush=True)
    X, y = load_mnist()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    ev_idx = perm[30000:40000]                       # fixed 10k evaluation split
    mask = np.ones(len(X), bool); mask[ev_idx] = False
    Xtr = torch.tensor(X[mask]).view(-1, 1, 28, 28).to(DEV)
    Xev = torch.tensor(X[ev_idx]).view(-1, 1, 28, 28).to(DEV)
    np.savez_compressed(os.path.join(_ROOT, "eval_split.npz"), idx=ev_idx, y=y[ev_idx])
    print(f"unlabeled train={tuple(Xtr.shape)}  eval={tuple(Xev.shape)}", flush=True)

    BS = 512
    enc = Enc().to(DEV)
    head = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64)).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 2e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Xtr) // BS),
                                              pct_start=0.1)
    for ep in range(epochs):
        p = torch.randperm(len(Xtr), device=DEV); t0 = time.time(); tot = 0.0; nb = 0
        for i in range(0, len(Xtr) - BS + 1, BS):
            x = Xtr[p[i:i + BS]]
            loss = nt_xent(head(enc(augment(x))), head(enc(augment(x))))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item(); nb += 1
        print(f"  ep{ep+1}/{epochs} loss={tot/nb:.4f} {time.time()-t0:.0f}s", flush=True)

    enc.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xev), 1000):
            o.append(enc(Xev[i:i + 1000]))
    emb = F.normalize(torch.cat(o), dim=1).cpu().numpy().astype("float32")
    out = os.path.join(_ROOT, "contrastive_emb.npz")
    np.savez_compressed(out, emb=emb, y=y[ev_idx])
    print(f"done -> contrastive_emb.npz {emb.shape}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
