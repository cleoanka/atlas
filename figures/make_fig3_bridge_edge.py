"""fig3 — one bridge edge is enough to leak a label across a boundary. Left: clean graph,
the label stays in its region. Right: a single cross-class edge added, the label floods the
other side. This is why the condition is binary: purity, not average distance, decides."""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402


def diffuse(W, seeds, seed_y, n, C=2, alpha=0.99, iters=60):
    dd = np.asarray(W.sum(1)).ravel(); dd[dd == 0] = 1
    S = (diags(dd ** -0.5) @ W @ diags(dd ** -0.5)).tocsr()
    Y = np.zeros((n, C)); Y[seeds, seed_y] = 1; F = Y.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * Y
    return F.argmax(1)


def build(X, k=6):
    A = kneighbors_graph(X, k, mode="distance", include_self=False)
    A = A.maximum(A.T)
    sig = np.median(A.data); W = A.copy()
    W.data = np.exp(-(W.data ** 2) / (2 * sig ** 2))
    return csr_matrix(W)


def draw_edges(ax, X, W):
    co = W.tocoo()
    for i, j in zip(co.row, co.col):
        if i < j:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="#dfe2ea", alpha=0.7,
                    lw=0.6, zorder=1)


def main():
    st.setup()
    rng = np.random.default_rng(0)
    n0 = 150
    x0 = rng.uniform(-5.0, -1.0, n0); x1 = rng.uniform(1.0, 5.0, n0)
    b0 = np.c_[x0, rng.standard_normal(n0) * 0.35]
    b1 = np.c_[x1, rng.standard_normal(n0) * 0.35]
    X = np.vstack([b0, b1]); y = np.r_[np.zeros(n0), np.ones(n0)].astype(int)
    n = len(y)
    W = build(X, k=6)
    seeds = np.array([int(b0[:, 0].argmax()), int(n0 + b1[:, 0].argmax())])
    seed_y = y[seeds]
    pred_clean = diffuse(W, seeds, seed_y, n); acc_clean = (pred_clean == y).mean()

    a = int(b0[:, 0].argmax()); b = int(n0 + b1[:, 0].argmin())
    Wb = W.tolil(); w = float(np.median(W.data)); Wb[a, b] = w; Wb[b, a] = w
    Wb = Wb.tocsr()
    pred_bridge = diffuse(Wb, seeds, seed_y, n); acc_bridge = (pred_bridge == y).mean()

    cols = [st.BLUE, st.ORANGE]
    fig = plt.figure(figsize=(12.4, 5.2))
    st.header(fig, "One wrong edge floods the boundary",
              "two labels, graph diffusion — a single bridge edge leaks a class across the gap",
              accent=st.RED)
    axes = [fig.add_axes([0.03, 0.05, 0.45, 0.66]), fig.add_axes([0.52, 0.05, 0.45, 0.66])]
    for ax, W_, pred, acc, title, bridge, ok in [
        (axes[0], W, pred_clean, acc_clean, "Clean graph", None, True),
        (axes[1], Wb, pred_bridge, acc_bridge, "One bridge edge added", (a, b), False)]:
        draw_edges(ax, X, W_)
        for c in (0, 1):
            m = pred == c
            ax.scatter(X[m, 0], X[m, 1], s=36, c=cols[c], zorder=2, edgecolors="white",
                       linewidths=0.4)
        if bridge is not None:
            ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], color=st.RED, lw=3.6, zorder=4)
            ax.scatter(X[[a, b], 0], X[[a, b], 1], s=80, c=st.RED, zorder=5)
        ax.scatter(X[seeds, 0], X[seeds, 1], s=380, marker="*", c=st.INK,
                   edgecolors="white", linewidths=1.3, zorder=6)
        ax.set_title(title, fontsize=15, color=st.INK, fontweight="bold", pad=8)
        ax.text(0.5, -0.02, f"accuracy {acc*100:.0f}%", transform=ax.transAxes, ha="center",
                va="top", fontsize=17, fontweight="bold", color=st.GREEN if ok else st.RED)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    st.save(fig, "fig3_bridge_edge.png")


if __name__ == "__main__":
    main()
