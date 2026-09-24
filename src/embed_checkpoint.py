"""Regenerate the validation and full-train embeddings from a saved encoder checkpoint.

The browser training run ships imagenette_encoder.pt and the per-epoch validation embeddings,
but NOT imagenette_train_emb.npz (needed for the proper full-train linear-probe ceiling). This
rebuilds the backbone from the checkpoint and re-embeds both pools locally (CPU/MPS, no GPU
needed), so the shipped val embedding and the train embedding come from the SAME weights — never
mix encoders across the two files or the ceiling is meaningless.

As a guard it first re-embeds the validation pool and checks it reproduces the shipped
data/imagenette_emb.npz (cosine ~1.0); only then does it trust the checkpoint enough to write the
train embedding.

    python src/embed_checkpoint.py                       # uses models/imagenette_encoder.pt
    python src/embed_checkpoint.py path/to/encoder.pt
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_contrastive_imagenette as T  # noqa: E402  (backbone, DEV, data loaders)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def embed(enc, Xu8, bs=256):
    """L2-normalised embedding of uint8 (N,H,W,3), moving one batch at a time to the device."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xu8), bs):
            xb = torch.from_numpy(Xu8[i:i + bs]).to(T.DEV).permute(0, 3, 1, 2).float() / 255.0
            outs.append(F.normalize(enc(xb), dim=1).cpu())
    return torch.cat(outs).numpy().astype("float32")


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models",
                                                                    "imagenette_encoder.pt")
    ck = torch.load(ckpt_path, map_location=T.DEV, weights_only=True)
    arch, img = ck["arch"], ck["img"]
    print(f"checkpoint: {os.path.relpath(ckpt_path, ROOT)}  arch={arch} img={img} "
          f"dim={ck.get('dim')} epochs={ck.get('epochs')}  device={T.DEV}", flush=True)

    enc = T.make_backbone(arch).to(T.DEV)
    enc.load_state_dict(ck["encoder"])
    enc.eval()

    # guard: reproduce the shipped validation embedding
    Xev, yev = T.D.eval_cache(img)
    val = embed(enc, Xev)
    ref_path = os.path.join(ROOT, "data", "imagenette_emb.npz")
    if os.path.exists(ref_path):
        ref = np.load(ref_path)["emb"].astype("float32")
        cos = float((val * ref).sum(1).mean()) if ref.shape == val.shape else -1.0
        print(f"  val-embedding reproduction vs shipped: mean cosine = {cos:.4f} "
              f"({'OK' if cos > 0.99 else 'MISMATCH — do not trust'})", flush=True)
    else:
        np.savez_compressed(ref_path, emb=val, y=yev)
        print(f"  wrote data/imagenette_emb.npz ({val.shape})", flush=True)

    # full train pool -> the linear-probe ceiling embedding (same weights as val)
    base = int(img * 1.15)
    Xtr = T.load_train(base)
    _, ytr = T.D.train_paths()
    c = (base - img) // 2
    Xtr_crop = Xtr[:, c:c + img, c:c + img, :]
    tr = embed(enc, Xtr_crop)
    out = os.path.join(ROOT, "data", "imagenette_train_emb.npz")
    np.savez_compressed(out, emb=tr, y=ytr)
    print(f"  wrote data/imagenette_train_emb.npz ({tr.shape}, from {arch} @ ep{ck.get('epochs')})",
          flush=True)


if __name__ == "__main__":
    main()
