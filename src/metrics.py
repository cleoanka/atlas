"""The evaluation protocol — identical for every representation.

Given a metric space X (any representation) and labels y:
  * `knn_graph`      : symmetric Gaussian kNN affinity W (the graph the metric induces).
  * `edge_purity`    : fraction of graph edges whose endpoints share a class. This is the
                       single quantity that predicts whether few-label propagation works.
  * `euclid_1nn`     : 1-label-per-class nearest-neighbour accuracy (the naive baseline).
  * `diffusion_spread`: 1-label-per-class graph diffusion / label spreading (Zhou 2004).
  * `linear_upper_bound`: fully-supervised linear probe (the ceiling).

Everything is transductive and seeded, so results reproduce bit-for-bit.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, csgraph
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph

K_DEFAULT = 10
ALPHA = 0.99
ITERS = 60


def knn_graph(X: np.ndarray, k: int = K_DEFAULT):
    """Symmetric Gaussian-weighted kNN graph. sigma = median edge distance (self-tuning)."""
    A = kneighbors_graph(X, k, mode="distance", include_self=False)
    A = A.maximum(A.T)
    sig = np.median(A.data)
    W = A.copy()
    W.data = np.exp(-(W.data ** 2) / (2.0 * sig ** 2))
    return csr_matrix(W)


def edge_purity(W: csr_matrix, y: np.ndarray) -> float:
    """Fraction of (upper-triangular) edges connecting same-class nodes. In [0,1]."""
    co = W.tocoo()
    m = co.row < co.col
    same = y[co.row[m]] == y[co.col[m]]
    return float(same.mean())


def bridge_edges(W: csr_matrix, y: np.ndarray) -> int:
    """Number of cross-class ("bridge") edges — the leaks diffusion can flow through."""
    co = W.tocoo()
    m = co.row < co.col
    return int((y[co.row[m]] != y[co.col[m]]).sum())


def n_components(W: csr_matrix) -> int:
    return int(csgraph.connected_components(W)[0])


def _normalized_operator(W: csr_matrix) -> csr_matrix:
    dd = np.asarray(W.sum(1)).ravel()
    dd[dd == 0] = 1.0
    return (diags(dd ** -0.5) @ W @ diags(dd ** -0.5)).tocsr()


def euclid_1nn(X: np.ndarray, labeled_idx: np.ndarray, labeled_y: np.ndarray,
               y: np.ndarray) -> float:
    """Nearest-labeled-point accuracy in the raw metric of X (no graph)."""
    d = -2 * X @ X[labeled_idx].T + (X[labeled_idx] ** 2).sum(1)[None]
    pred = labeled_y[d.argmin(1)]
    return float((pred == y).mean())


def diffusion_spread(S: csr_matrix, labeled_idx: np.ndarray, labeled_y: np.ndarray,
                     y: np.ndarray, n_classes: int, alpha: float = ALPHA,
                     iters: int = ITERS) -> float:
    """Label spreading on the graph: F <- alpha*S@F + (1-alpha)*Y, iterated. Accuracy."""
    n = len(y)
    Y = np.zeros((n, n_classes))
    Y[labeled_idx, labeled_y] = 1.0
    F = Y.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * Y
    return float((F.argmax(1) == y).mean())


def one_label_per_class(X, W, y, trials: int = 60, seed: int = 1):
    """Average Euclid-1NN vs diffusion accuracy over `trials` random 1-label-per-class draws.

    Returns dict with means/stds and the graph diagnostics (purity, bridges, components).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    C = len(classes)
    S = _normalized_operator(W)
    e1, h1 = [], []
    for _ in range(trials):
        li = np.array([rng.choice(np.where(y == c)[0]) for c in classes])
        ly = y[li]
        e1.append(euclid_1nn(X, li, ly, y))
        h1.append(diffusion_spread(S, li, ly, y, C))
    return {
        "purity": edge_purity(W, y),
        "bridges": bridge_edges(W, y),
        "components": n_components(W),
        "euclid_1nn": float(np.mean(e1)),
        "euclid_1nn_std": float(np.std(e1)),
        "diffusion": float(np.mean(h1)),
        "diffusion_std": float(np.std(h1)),
    }


def linear_upper_bound(X: np.ndarray, y: np.ndarray, train_frac: float = 0.7,
                       seed: int = 1) -> float:
    """Fully-supervised linear probe accuracy (70/30 split of one pool) — the ceiling."""
    rng = np.random.default_rng(seed)
    pm = rng.permutation(len(y))
    sp = int(train_frac * len(y))
    clf = LogisticRegression(max_iter=1500).fit(X[pm[:sp]], y[pm[:sp]])
    return float(clf.score(X[pm[sp:]], y[pm[sp:]]))


def linear_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> float:
    """Fully-supervised linear probe trained on a SEPARATE train set, tested on the eval pool."""
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))
