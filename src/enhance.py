"""Does a better metric / smarter seeds squeeze more from the same 10 labels? — measured.

Tests the geometric upgrades discussed in the design notes, each as an honest ablation on the
SHIPPED embeddings (no retraining, no GPU):

  metric (the Riemannian idea — a locally-adaptive / anisotropic ruler instead of one global σ):
    * euclid       : isotropic Gaussian, σ = median edge distance                 (baseline)
    * cosine       : L2-normalise first (no-op for already-normalised contrastive)
    * self_tuning  : per-point scale σ_i = dist to k-th neighbour (Zelnik-Manor–Perona)
    * anisotropic  : Coifman-Lafon α=1 density normalisation → Laplace–Beltrami operator
    * mahalanobis  : local inverse-covariance metric tensor g(x) (PCA-reduced; Singer-Coifman)

  seed (the "which labels?" idea — not all chairs are equally chair):
    * random       : one random point per class, 60-trial mean                    (baseline)
    * density      : the highest-degree (most typical / prototypical) point per class
    * medoid       : the class medoid (min mean within-class distance)

    python src/enhance.py --dataset mnist
    python src/enhance.py --dataset imagenette
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "src"))
import metrics as M            # noqa: E402
import representations as R    # noqa: E402


# ---------------------------------------------------------------- affinities (the metric)
def _sym_knn(X, k):
    A = kneighbors_graph(X, k, mode="distance", include_self=False)
    return A.maximum(A.T)


def affinity(X, k=10, scale="median"):
    """Symmetric Gaussian kNN affinity. scale='median' (isotropic) or 'self_tuning' (per-point)."""
    A = _sym_knn(X, k)
    if scale == "self_tuning":
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        d, _ = nn.kneighbors(X)
        sig = d[:, -1] + 1e-9                       # σ_i = distance to k-th neighbour
        co = A.tocoo()
        w = np.exp(-(co.data ** 2) / (sig[co.row] * sig[co.col]))
        return csr_matrix((w, (co.row, co.col)), shape=A.shape)
    s = np.median(A.data)
    W = A.copy(); W.data = np.exp(-(W.data ** 2) / (2.0 * s ** 2))
    return csr_matrix(W)


def maha_affinity(X, k=10, dim=40, n_local=30, reg=1e-2):
    """Local-Mahalanobis (Riemannian metric tensor) affinity: d²=½(xi-xj)ᵀ(Cᵢ⁻¹+Cⱼ⁻¹)(xi-xj).

    Cᵢ = local covariance from n_local neighbours (PCA-reduced to `dim` for tractability). This
    is the literal position-dependent metric tensor g(x)=C(x)⁻¹ — 'each vector's own weights'.
    """
    Z = PCA(n_components=min(dim, X.shape[1]), random_state=0).fit_transform(X).astype("float64")
    n = len(Z)
    nn = NearestNeighbors(n_neighbors=n_local + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    P = np.eye(Z.shape[1]) * reg
    Cinv = np.empty((n, Z.shape[1], Z.shape[1]))
    for i in range(n):
        d = Z[idx[i, 1:]] - Z[i]
        Cinv[i] = np.linalg.inv(d.T @ d / n_local + P)
    # build symmetric kNN edges with the averaged local metric
    A = _sym_knn(Z, k).tocoo()
    diff = Z[A.row] - Z[A.col]
    G = 0.5 * (Cinv[A.row] + Cinv[A.col])
    d2 = np.einsum("ij,ijk,ik->i", diff, G, diff)   # bench-side: einsum is fine (not core learning)
    s = np.median(d2) + 1e-12
    w = np.exp(-d2 / s)
    return csr_matrix((w, (A.row, A.col)), shape=(n, n))


def snn_reweight(X, W, knn_for_snn=20):
    """Shared-nearest-neighbour edge weighting (the 'processing' lever): multiply each affinity by
    the fraction of neighbours its two endpoints share. Bridge edges across a gap share few
    neighbours → suppressed; edges inside a dense cluster survive. Directly down-weights the
    atypical/bridge nodes label mass leaks through, and combats kNN hubness in high dimensions.
    """
    nn = NearestNeighbors(n_neighbors=knn_for_snn + 1).fit(X)
    _, idx = nn.kneighbors(X)
    nbr = [set(r[1:]) for r in idx]
    co = W.tocoo()
    shared = np.fromiter((len(nbr[i] & nbr[j]) for i, j in zip(co.row, co.col)),
                         dtype=float, count=co.nnz)
    w = co.data * (shared + 1e-6) / knn_for_snn
    return csr_matrix((w, (co.row, co.col)), shape=W.shape)


def operator(W, alpha=0.0):
    """Normalised diffusion operator. alpha>0 = Coifman anisotropic (α=1 → Laplace–Beltrami)."""
    d = np.asarray(W.sum(1)).ravel(); d[d == 0] = 1.0
    if alpha > 0:
        Wa = csr_matrix(diags(d ** (-alpha)) @ W @ diags(d ** (-alpha)))
    else:
        Wa = W
    dd = np.asarray(Wa.sum(1)).ravel(); dd[dd == 0] = 1.0
    return (diags(dd ** -0.5) @ Wa @ diags(dd ** -0.5)).tocsr(), Wa


# ---------------------------------------------------------------- seeds (which labels)
def seeds_density(W, y, classes):
    deg = np.asarray(W.sum(1)).ravel()             # graph degree = local density / typicality
    return np.array([cls_members[np.argmax(deg[cls_members])]
                     for cls_members in (np.where(y == c)[0] for c in classes)])


def seeds_medoid(X, y, classes):
    out = []
    for c in classes:
        m = np.where(y == c)[0]
        Xm = X[m]
        d = -2 * Xm @ Xm.T + (Xm ** 2).sum(1)[:, None] + (Xm ** 2).sum(1)[None]
        out.append(m[np.argmin(d.mean(1))])
    return np.array(out)


def acc_from_seeds(S, seeds, y, classes):
    return M.diffusion_spread(S, seeds, y[seeds], y, len(classes))


def eval_config(X, y, k=10, scale="median", alpha=0.0, metric="euclid", seed="random",
                processing="none", trials=60, rng_seed=1):
    classes = np.unique(y)
    Xu = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12) if metric == "cosine" else X
    if metric == "mahalanobis":
        W = maha_affinity(Xu, k)
    else:
        W = affinity(Xu, k, scale)
    if processing == "snn":
        W = snn_reweight(Xu, W)
    S, _ = operator(W, alpha)
    if seed == "random":
        rng = np.random.default_rng(rng_seed)
        accs = [acc_from_seeds(S, np.array([rng.choice(np.where(y == c)[0]) for c in classes]),
                               y, classes) for _ in range(trials)]
        return float(np.mean(accs)), float(np.std(accs)), M.edge_purity(W, y)
    sd = seeds_density(W, y, classes) if seed == "density" else seeds_medoid(Xu, y, classes)
    return acc_from_seeds(S, sd, y, classes), 0.0, M.edge_purity(W, y)


def main(dataset):
    X, y = R.contrastive(dataset)
    print(f"=== enhance ablation · {dataset} · contrastive · 1 label/class ===")
    rows = []

    def run(tag, **kw):
        a, s, pur = eval_config(X, y, **kw)
        rows.append({"config": tag, **kw, "acc": a, "std": s, "purity": pur})
        print(f"  {tag:<34} acc={a*100:5.1f}%{'' if s==0 else f' ±{s*100:.1f}'}   purity={pur*100:.1f}%")

    run("baseline (euclid, median, random)")
    run("cosine", metric="cosine")
    run("self_tuning scale", scale="self_tuning")
    run("anisotropic α=1 (Laplace-Beltrami)", alpha=1.0)
    run("local Mahalanobis g(x)", metric="mahalanobis")
    run("processing: SNN bridge-kill", processing="snn")
    run("seed: density-peak", seed="density")
    run("seed: medoid", seed="medoid")
    run("medoid + SNN processing", seed="medoid", processing="snn")
    run("BEST: self_tuning+α=1+density", scale="self_tuning", alpha=1.0, seed="density")

    # k / alpha grid (quick), baseline metric, random seeds
    grid = []
    for k in (5, 10, 20, 40):
        for al in (0.0, 0.5, 1.0):
            a, _, _ = eval_config(X, y, k=k, alpha=al, trials=20)
            grid.append({"k": k, "alpha": al, "acc": a})
    best = max(grid, key=lambda g: g["acc"])
    print(f"  k/α grid best: k={best['k']} α={best['alpha']} → {best['acc']*100:.1f}%")

    out = {"dataset": dataset, "rows": rows, "grid": grid, "grid_best": best}
    with open(os.path.join(_HERE, "results", f"{dataset}_enhance.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote results/{dataset}_enhance.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=R.DATASETS)
    main(ap.parse_args().dataset)
