"""banner.png — the repo hero. Title block on the left, the two-moons diffusion motif on the
right (label flowing from 2 seeds along each moon). Wide, calm, and readable as a thumbnail."""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csgraph
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402


def moons_diffused():
    X, y = make_moons(n_samples=520, noise=0.075, random_state=1)
    X = (X - X.mean(0)) / X.std(0)
    A = kneighbors_graph(X, 8, mode="distance", include_self=False)
    A = A.maximum(A.T)
    rng = np.random.default_rng(4)
    seeds = [int(rng.choice(np.where(y == 0)[0])), int(rng.choice(np.where(y == 1)[0]))]
    # colour = nearest-seed by graph geodesic (clean two-tone fill)
    dist = csgraph.dijkstra(A, indices=seeds, min_only=True)
    src = csgraph.dijkstra(A, indices=seeds, min_only=True, return_predecessors=False)  # noqa
    d0 = csgraph.dijkstra(A, indices=[seeds[0]])[0]
    d1 = csgraph.dijkstra(A, indices=[seeds[1]])[0]
    cls = (d1 < d0).astype(int)
    co = A.tocoo()
    segs = [[X[i], X[j]] for i, j in zip(co.row, co.col) if i < j]
    return X, cls, seeds, segs


def main():
    st.setup()
    X, cls, seeds, segs = moons_diffused()

    fig = plt.figure(figsize=(12.8, 4.2))
    fig.patch.set_facecolor(st.PAPER)

    # ---- right: the art ----
    ax = fig.add_axes([0.52, 0.02, 0.46, 0.96])
    ax.add_collection(LineCollection(segs, colors="#dfe2ea", linewidths=0.7, zorder=1))
    cols = np.where(cls == 1, st.ORANGE, st.BLUE)
    ax.scatter(X[:, 0], X[:, 1], s=42, c=cols, edgecolors="white", linewidths=0.5, zorder=2)
    ax.scatter(X[seeds, 0], X[seeds, 1], s=520, marker="*",
               c=[st.BLUE, st.ORANGE], edgecolors=st.INK, linewidths=1.8, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.spines[:].set_visible(False); ax.margins(0.04)
    ax.set_facecolor(st.PAPER)

    # ---- left: the words ----
    fig.text(0.056, 0.955, "FEW-LABEL LEARNING", fontsize=11.5, fontweight="bold",
             color=st.BLUE, va="top")
    fig.text(0.055, 0.90, "atlas", fontsize=52, fontweight="bold", color=st.INK, va="top")
    fig.text(0.058, 0.585, "chart the whole manifold from a few names",
             fontsize=17.5, color=st.BLUE, va="top", fontweight="bold")
    fig.text(0.058, 0.45,
             "The hard part of classification is the metric, not the classifier.\n"
             "Fix the representation and one label per class propagates across\n"
             "the whole dataset along a similarity graph.",
             fontsize=13, color=st.SUBINK, va="top", linespacing=1.5)
    # headline chips
    fig.text(0.058, 0.135, "  10 labels → 94.7%  ", fontsize=15, fontweight="bold",
             color="white", va="center",
             bbox=dict(boxstyle="round,pad=0.5", fc=st.ORANGE, ec="none"))
    fig.text(0.27, 0.135, "  7000 labels → 98.6%  ", fontsize=15, fontweight="bold",
             color="white", va="center",
             bbox=dict(boxstyle="round,pad=0.5", fc=st.MUTED, ec="none"))
    st.save(fig, "banner.png")


if __name__ == "__main__":
    main()
