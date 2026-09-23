"""Representations for MNIST and ImageNette: raw pixels, PCA-50, diffusion map, contrastive.

The single idea of this repo lives here: the *metric* (how you measure distance between
points) is where the hard work is. Each function produces a different metric space for the
SAME evaluation pool; `metrics.py` scores each one with the same protocol (edge purity,
1-label-per-class Euclid-1NN vs. graph diffusion).

All representations are UNSUPERVISED (no labels used to build them) — labels only enter at
evaluation time, one per class. Pick the dataset with the `dataset=` argument.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST = os.path.join(_HERE, "mnist.npz")
EVAL_SPLIT = os.path.join(_HERE, "eval_split.npz")
CONTRASTIVE = os.path.join(_HERE, "contrastive_emb.npz")
IMAGENETTE_EMB = os.path.join(_HERE, "data", "imagenette_emb.npz")

DATASETS = ("mnist", "imagenette")


def _mnist_pixels():
    if not os.path.exists(EVAL_SPLIT):
        raise FileNotFoundError("eval_split.npz missing — run `python src/train_contrastive.py`.")
    if not os.path.exists(MNIST):                     # gitignored (21 MB) → fetch once on demand
        from sklearn.datasets import fetch_openml
        print("downloading MNIST (once)...", flush=True)
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        np.savez_compressed(MNIST, X=d.data.astype("float32") / 255.0, y=d.target.astype(int))
    d = np.load(MNIST)
    sp = np.load(EVAL_SPLIT)
    return np.ascontiguousarray(d["X"][sp["idx"]], "float32"), sp["y"].astype(int)


def _imagenette_pixels(raw_size: int = 32):
    """Validation pool as flattened low-res RGB (the naive 'raw pixel' metric for real images).

    The cached eval images (128×128) are block-mean-pooled to raw_size×raw_size and flattened
    to 3·raw_size² dims in [0,1] — a fair, cheap pixel baseline (full-res L2 is hopeless AND slow).
    """
    import imagenette_data as ID
    X8, y = ID.eval_cache(128)                        # (N,128,128,3) uint8
    f = 128 // raw_size
    X = (X8.reshape(len(X8), raw_size, f, raw_size, f, 3).mean(axis=(2, 4)) / 255.0)
    return X.reshape(len(X8), -1).astype("float32"), y.astype(int)


def load_eval_pixels(dataset: str = "mnist"):
    """Return (X_pixels, y) for the fixed evaluation pool of the chosen dataset."""
    if dataset == "mnist":
        return _mnist_pixels()
    if dataset == "imagenette":
        return _imagenette_pixels()
    raise KeyError(f"unknown dataset '{dataset}'; choices: {DATASETS}")


def raw_pixels(dataset: str = "mnist"):
    """Raw pixel space (the naive metric: L2 distance between images)."""
    return load_eval_pixels(dataset)


def pca50(dataset: str = "mnist", dim: int = 50):
    """Linear PCA to `dim` dimensions — a cheap, standard denoising of pixel space."""
    X, y = load_eval_pixels(dataset)
    return PCA(n_components=dim, random_state=0).fit_transform(X).astype("float32"), y


def diffusion_map(dataset: str = "mnist", n_components: int = 50, k: int = 10):
    """Coifman-Lafon diffusion map: nonlinear coordinates from the graph Laplacian."""
    X, y = load_eval_pixels(dataset)
    A = kneighbors_graph(X, k, mode="distance", include_self=False)
    A = A.maximum(A.T)
    sig = np.median(A.data)
    W = A.copy()
    W.data = np.exp(-(W.data ** 2) / (2.0 * sig ** 2))
    W = csr_matrix(W)
    dd = np.asarray(W.sum(1)).ravel()
    dd[dd == 0] = 1.0
    S = (diags(dd ** -0.5) @ W @ diags(dd ** -0.5)).tocsr()
    vals, vecs = eigsh(S, k=n_components + 1, which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order][1:], vecs[:, order][:, 1:]
    return (vecs * vals[None, :]).astype("float32"), y


def contrastive(dataset: str = "mnist"):
    """Load the unsupervised contrastive embedding (L2-normalised). Shipped in the repo."""
    path = CONTRASTIVE if dataset == "mnist" else IMAGENETTE_EMB
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.basename(path)} missing — run the matching train_contrastive script.")
    d = np.load(path)
    return np.ascontiguousarray(d["emb"], "float32"), d["y"].astype(int)


# Registry: name -> (callable, human label). Order = table order.
REGISTRY = {
    "raw": (raw_pixels, "raw pixels"),
    "pca50": (pca50, "PCA-50"),
    "diffusion_map": (diffusion_map, "diffusion map"),
    "contrastive": (contrastive, "contrastive"),
}


def get(name: str, dataset: str = "mnist"):
    """Return (X, y) for a named representation on the chosen dataset."""
    if name not in REGISTRY:
        raise KeyError(f"unknown representation '{name}'; choices: {list(REGISTRY)}")
    return REGISTRY[name][0](dataset)
