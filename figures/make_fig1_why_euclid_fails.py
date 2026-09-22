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
    # anchor: a point on arm 0, out along the arm where the other arm passes close
    arm0 = np.where(lab == 0)[0]
    r = np.linalg.norm(X, axis=1)
    p = arm0[np.argmin(np.abs(r[arm0] - 6.5))]

    # Euclidean neighborhood: ball of radius R (sized so it clearly bites the other arm)
    R = 3.2
    eucl = np.linalg.norm(X - X[p], axis=1) <= R

    # graph neighborhood: shortest path on kNN graph within a geodesic budget
    G = kneighbors_graph(X, 8, mode="distance", include_self=False)
    G = G.maximum(G.T)
    dist = csgraph.dijkstra(G, indices=p)
    budget = np.percentile(dist[np.isfinite(dist)], 12)   # comparable neighborhood size
    graph_nb = dist <= budget

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, sel, title, note in [
        (axes[0], eucl, "Euclidean “near”", "the ball reaches the other arm"),
        (axes[1], graph_nb, "Graph “near”", "the neighborhood follows the arm")]:
        # base points, distinct by arm
        for c, col in [(0, st.OK["sky"]), (1, st.OK["orange"])]:
            m = lab == c
            ax.scatter(X[m, 0], X[m, 1], s=16, c=col, alpha=0.45, zorder=1)
        # highlighted neighborhood — solid, ringed so it pops on either arm colour
        ax.scatter(X[sel, 0], X[sel, 1], s=46, c=st.OK["vermillion"],
                   edgecolors="white", linewidths=0.7, zorder=3)
        # how many of the neighbors are on the WRONG arm
        wrong = int(((lab != lab[p]) & sel).sum())
        ax.scatter([X[p, 0]], [X[p, 1]], s=340, marker="*", c=st.OK["black"],
                   edgecolors="white", linewidths=1.2, zorder=5)
        if ax is axes[0]:
            circ = plt.Circle((X[p, 0], X[p, 1]), R, fill=False, ls="--",
                              ec=st.OK["vermillion"], lw=2.0, zorder=4)
            ax.add_patch(circ)
        ax.set_title(title)
        ax.text(0.5, -0.06, f"{note}\n({wrong} neighbors on the other arm)",
                transform=ax.transAxes, ha="center", va="top", fontsize=13,
                color=st.OK["vermillion"] if wrong else st.OK["green"], fontweight="bold")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
    fig.suptitle("Same point, same neighborhood size — the metric decides who is a neighbor",
                 fontsize=17, fontweight="bold", y=1.02)
    st.save(fig, "fig1_why_euclid_fails.png")


if __name__ == "__main__":
    main()
