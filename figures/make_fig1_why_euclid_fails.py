"""fig1 — why Euclidean distance fails on a manifold. Two spirals, one point, two neighborhoods.

Left:  the Euclidean ball around a point reaches across the gap onto the OTHER arm.
Right: the graph (kNN, manifold-following) neighborhood of the same point stays on its own arm.
One glance: the definition of "near" decides everything.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402


def two_spirals(n=520, noise=0.28, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.random(n)) * 3.2 * np.pi
    r = t
    x1 = np.c_[r * np.cos(t), r * np.sin(t)] + rng.standard_normal((n, 2)) * noise
    x2 = np.c_[r * np.cos(t + np.pi), r * np.sin(t + np.pi)] + rng.standard_normal((n, 2)) * noise
    X = np.vstack([x1, x2])
    lab = np.r_[np.zeros(n), np.ones(n)].astype(int)
    return X, lab


def main():
    st.setup()
    X, lab = two_spirals()
    arm0 = np.where(lab == 0)[0]
    r = np.linalg.norm(X, axis=1)
    p = arm0[np.argmin(np.abs(r[arm0] - 6.5))]

    R = 3.2
    eucl = np.linalg.norm(X - X[p], axis=1) <= R
    G = kneighbors_graph(X, 8, mode="distance", include_self=False)
    G = G.maximum(G.T)
    dist = csgraph.dijkstra(G, indices=p)
    budget = np.percentile(dist[np.isfinite(dist)], 12)
    graph_nb = dist <= budget

    fig = plt.figure(figsize=(12.4, 6.6))
    st.header(fig, "The metric decides who counts as a neighbour",
              "same point, same neighbourhood size — only the notion of distance changes",
              accent=st.BLUE)
    axes = [fig.add_axes([0.04, 0.03, 0.44, 0.72]), fig.add_axes([0.53, 0.03, 0.44, 0.72])]
    for ax, sel, title, note, ok in [
        (axes[0], eucl, "Euclidean “near”", "the ball reaches the other arm", False),
        (axes[1], graph_nb, "Graph “near”", "the neighbourhood follows the arm", True)]:
        for c, col in [(0, st.BLUE), (1, st.ORANGE)]:
            m = lab == c
            ax.scatter(X[m, 0], X[m, 1], s=16, c=col, alpha=0.40, zorder=1)
        ax.scatter(X[sel, 0], X[sel, 1], s=48, c=st.GREEN if ok else st.RED,
                   edgecolors="white", linewidths=0.7, zorder=3)
        wrong = int(((lab != lab[p]) & sel).sum())
        ax.scatter([X[p, 0]], [X[p, 1]], s=360, marker="*", c=st.INK,
                   edgecolors="white", linewidths=1.3, zorder=5)
        if not ok:
            ax.add_patch(plt.Circle((X[p, 0], X[p, 1]), R, fill=False, ls=(0, (5, 4)),
                                    ec=st.RED, lw=2.0, zorder=4))
        ax.set_title(title, fontsize=15, color=st.INK, fontweight="bold", pad=8)
        ax.text(0.5, -0.02, f"{note}  —  {wrong} neighbours on the other arm",
                transform=ax.transAxes, ha="center", va="top", fontsize=12.5,
                color=st.GREEN if ok else st.RED, fontweight="bold")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    st.save(fig, "fig1_why_euclid_fails.png")


if __name__ == "__main__":
    main()
