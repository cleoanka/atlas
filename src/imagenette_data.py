"""ImageNette data — real 10-class ImageNet photos (fast.ai). Loaded with PIL, no torchvision.

Two things the rest of the pipeline needs:
  * `train_paths()`  — the 9469 unlabelled training images (for contrastive training).
  * `eval_cache(sz)` — the 3925 validation images resized to sz×sz, cached as uint8 in
                       data/imagenette_eval_{sz}.npz (the transductive evaluation pool).
Labels enter only at evaluation time, one per class — exactly as on MNIST.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(_ROOT, "data", "imagenette2-160")
# fast.ai wnid -> human label
CLASSES = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}
WNIDS = sorted(CLASSES)                      # fixed class order 0..9
LABEL_NAMES = [CLASSES[w] for w in WNIDS]


def _split_paths(split: str):
    """Return (list[path], np.array[label]) for 'train' or 'val'."""
    base = os.path.join(ROOT, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} missing — download imagenette2-160 into data/ first.")
    paths, labels = [], []
    for ci, w in enumerate(WNIDS):
        d = os.path.join(base, w)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".jpeg"):
                paths.append(os.path.join(d, fn)); labels.append(ci)
    return paths, np.asarray(labels, dtype=int)


def train_paths():
    return _split_paths("train")


def _load_resized(path: str, sz: int) -> np.ndarray:
    """Load an image, resize short side to sz, centre-crop sz×sz, return uint8 (sz,sz,3)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = sz / min(w, h)
    im = im.resize((max(sz, round(w * s)), max(sz, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - sz) // 2, (h - sz) // 2
    return np.asarray(im.crop((left, top, left + sz, top + sz)), dtype=np.uint8)


def eval_cache(sz: int = 128, rebuild: bool = False):
    """Return (X uint8 [N,sz,sz,3], y [N]) for the validation pool; cache to npz."""
    cache = os.path.join(_ROOT, "data", f"imagenette_eval_{sz}.npz")
    if os.path.exists(cache) and not rebuild:
        d = np.load(cache)
        return d["X"], d["y"]
    paths, y = _split_paths("val")
    X = np.zeros((len(paths), sz, sz, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        X[i] = _load_resized(p, sz)
        if i % 500 == 0:
            print(f"  eval cache {i}/{len(paths)}", flush=True)
    np.savez_compressed(cache, X=X, y=y)
    print(f"wrote {cache}  {X.shape}", flush=True)
    return X, y


if __name__ == "__main__":
    sz = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    tp, ty = train_paths()
    print(f"train images: {len(tp)}  |  classes: {len(WNIDS)}")
    X, y = eval_cache(sz)
    print(f"eval pool: {X.shape}  labels: {np.bincount(y)}")
