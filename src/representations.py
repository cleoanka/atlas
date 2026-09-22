"""Representations of MNIST: raw pixels, PCA-50, diffusion map, contrastive embedding.

The single idea of this repo lives here: the *metric* (how you measure distance between
points) is where the hard work is. Each function below produces a different metric space
for the SAME 10k evaluation digits; `metrics.py` then scores each one with the same
downstream protocol (edge purity, 1-label-per-class Euclid-1NN vs. graph diffusion).

All representations are UNSUPERVISED (no labels used to build them) — labels only enter at
evaluation time, one per class.
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


def load_eval_pixels():
    """Return (X_pixels[10k,784] in [0,1], y[10k]) — the fixed evaluation split.

    Uses eval_split.npz (indices + labels) produced by train_contrastive.py, so every
    representation is scored on EXACTLY the same 10k digits as the contrastive embedding.
    """
    if not os.path.exists(EVAL_SPLIT):
        raise FileNotFoundError(
            "eval_split.npz missing — it is shipped in the repo; run "
            "`python src/train_contrastive.py` to regenerate it.")
    if not os.path.exists(MNIST):                     # gitignored (21 MB) → fetch once on demand
        from sklearn.datasets import fetch_openml
        print("downloading MNIST (once)...", flush=True)
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        np.savez_compressed(MNIST, X=d.data.astype("float32") / 255.0, y=d.target.astype(int))
    d = np.load(MNIST)
    X = d["X"]
    sp = np.load(EVAL_SPLIT)
    idx = sp["idx"]
    return np.ascontiguousarray(X[idx], "float32"), sp["y"].astype(int)


def raw_pixels():
    """Raw 784-dim pixel space (the naive metric: L2 distance between images)."""
    X, y = load_eval_pixels()
    return X, y


def pca50(dim: int = 50):
    """Linear PCA to `dim` dimensions — a cheap, standard denoising of pixel space."""
    X, y = load_eval_pixels()
    return PCA(n_components=dim, random_state=0).fit_transform(X).astype("float32"), y


def diffusion_map(n_components: int = 50, k: int = 10, seed: int = 0):
    """Coifman-Lafon diffusion map: nonlinear coordinates from the graph Laplacian.

    Build a symmetric kNN affinity, form the normalized operator S = D^-1/2 W D^-1/2, take
    its top eigenvectors (dropping the trivial constant one) scaled by their eigenvalues.
    This bends the metric along the data manifold BEFORE any labels are seen.
    """
    X, y = load_eval_pixels()
    A = kneighbors_graph(X, k, mode="distance", include_self=False)
    A = A.maximum(A.T)
    sig = np.median(A.data)
    W = A.copy()
    W.data = np.exp(-(W.data ** 2) / (2.0 * sig ** 2))
    W = csr_matrix(W)
    dd = np.asarray(W.sum(1)).ravel()
    dd[dd == 0] = 1.0
    S = (diags(dd ** -0.5) @ W @ diags(dd ** -0.5)).tocsr()
    # top n_components+1 eigenpairs (largest algebraic); drop the trivial leading one
    vals, vecs = eigsh(S, k=n_components + 1, which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order][1:], vecs[:, order][:, 1:]
    emb = (vecs * vals[None, :]).astype("float32")            # diffusion coordinates (t=1)
    return emb, y


def contrastive():
    """Load the pretrained contrastive embedding (contrastive_emb.npz, 64-dim, L2-normed).

    Produced by src/train_contrastive.py (unsupervised NT-Xent). Shipped in the repo so the
    headline result reproduces without a GPU.
    """
    if not os.path.exists(CONTRASTIVE):
        raise FileNotFoundError(
            "contrastive_emb.npz missing — run `python src/train_contrastive.py 40` "
            "(needs a GPU/MPS; the file is shipped in the repo so you usually don't).")
    d = np.load(CONTRASTIVE)
    return np.ascontiguousarray(d["emb"], "float32"), d["y"].astype(int)


# Registry: name -> (callable, human label). Order = table order.
REGISTRY = {
    "raw": (raw_pixels, "raw pixels"),
    "pca50": (pca50, "PCA-50"),
    "diffusion_map": (diffusion_map, "diffusion map"),
    "contrastive": (contrastive, "contrastive (40ep)"),
}


def get(name: str):
    """Return (X, y) for a named representation."""
    if name not in REGISTRY:
        raise KeyError(f"unknown representation '{name}'; choices: {list(REGISTRY)}")
    return REGISTRY[name][0]()
