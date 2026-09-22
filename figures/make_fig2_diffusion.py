"""fig2 — the cover: label diffusion from just 2 seeds on a two-moons graph (animated GIF).

Two interleaving moons, one labeled point each. Colour spreads along the graph edges as the
diffusion iterates — each moon fills with its seed's colour without crossing the gap. This is
the whole idea in motion: given a manifold-following metric, two labels colour the world.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy.sparse import diags, csgraph
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402

import matplotlib.colors as _mc

C0 = np.array(_mc.to_rgb(st.BLUE))    # class 0
C1 = np.array(_mc.to_rgb(st.ORANGE))  # class 1
CG = np.array([0.86, 0.87, 0.90])     # grey  (wavefront not arrived yet)


def colors(reached, p1):
    """Grey where the wavefront hasn't arrived; else blue/orange by the diffusion's argmax.

    The label-spread MASS decays exponentially with graph distance (far nodes have vanishing
    mass yet a perfectly correct argmax), so we reveal nodes by geodesic order and colour them
    by the converged label — an honest picture: the label flows along the graph and every node
    it reaches converges to the right class.
    """
    hue = np.where(p1[:, None] >= 0.5, C1, C0)
    r = reached[:, None].astype(float)
    return CG[None, :] * (1 - r) + hue * r


def main():
    st.setup()
    X, y = make_moons(n_samples=600, noise=0.075, random_state=0)
    X = (X - X.mean(0)) / X.std(0)
    A = kneighbors_graph(X, 8, mode="distance", include_self=False)
    A = A.maximum(A.T)
    sig = np.median(A.data)
    W = A.copy(); W.data = np.exp(-(W.data ** 2) / (2 * sig ** 2))
    dd = np.asarray(W.sum(1)).ravel(); dd[dd == 0] = 1
    S = (diags(dd ** -0.5) @ W @ diags(dd ** -0.5)).tocsr()

    rng = np.random.default_rng(3)
    seeds = [int(rng.choice(np.where(y == 0)[0])), int(rng.choice(np.where(y == 1)[0]))]
    Y = np.zeros((len(y), 2)); Y[seeds[0], 0] = 1; Y[seeds[1], 1] = 1
    alpha = 0.99
    F = Y.copy()
    for _ in range(60):
        F = alpha * (S @ F) + (1 - alpha) * Y
    p1 = np.where(F.sum(1) > 1e-30, F[:, 1] / np.maximum(F.sum(1), 1e-30), 0.5)  # converged argmax
    acc = float((F.argmax(1) == y).mean())

    # wavefront = geodesic order from the nearest seed (mass decays exponentially, so distance —
    # not mass — is the honest "when did the label arrive" clock).
    dist = csgraph.dijkstra(A, indices=seeds, min_only=True)
    finite = dist[np.isfinite(dist)]
    reveal = np.linspace(0, finite.max() * 1.02, 34)          # thresholds swept over the animation

    co = A.tocoo()
    segs = [[X[i], X[j]] for i, j in zip(co.row, co.col) if i < j]
    seed_cols = [C0, C1]

    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.add_collection(LineCollection(segs, colors="#dddddd", linewidths=0.5, zorder=1))
    scat = ax.scatter(X[:, 0], X[:, 1], s=34, c=colors(dist <= 0, p1),
                      edgecolors="white", linewidths=0.3, zorder=2)
    ax.scatter(X[seeds, 0], X[seeds, 1], s=430, marker="*", c=seed_cols,
               edgecolors="black", linewidths=1.5, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.spines[:].set_visible(False); ax.margins(0.03)
    title = ax.set_title("", fontsize=15)
    ax.text(0.5, -0.02, "colour flows ALONG each moon, never across the gap",
            transform=ax.transAxes, ha="center", va="top", fontsize=13, color=st.OK["grey"])

    def update(i):
        k = min(i, len(reveal) - 1)
        reached = dist <= reveal[k]
        scat.set_color(colors(reached, p1))
        title.set_text(f"2 labels  ·  spreading {reached.mean()*100:3.0f}%  ·  "
                       f"{acc*100:.0f}% correct")
        return scat, title

    total = 4 + len(reveal) + 8    # hold at start and end for a smooth loop
    anim = FuncAnimation(fig, update, frames=total, interval=100, blit=False)
    anim.save(os.path.join(st.HERE, "fig2_diffusion.gif"), writer=PillowWriter(fps=10))
    plt.close(fig)
    print("  wrote figures/fig2_diffusion.gif")

    # static final-state PNG (README fallback / thumbnail)
    fig2, ax2 = plt.subplots(figsize=(8, 6.4))
    ax2.add_collection(LineCollection(segs, colors="#dddddd", linewidths=0.5, zorder=1))
    ax2.scatter(X[:, 0], X[:, 1], s=34, c=colors(np.ones(len(y), bool), p1),
                edgecolors="white", linewidths=0.3, zorder=2)
    ax2.scatter(X[seeds, 0], X[seeds, 1], s=430, marker="*", c=seed_cols,
                edgecolors="black", linewidths=1.5, zorder=5)
    ax2.set_title(f"2 labels → {acc*100:.0f}% of both moons, correctly", fontsize=18)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_aspect("equal")
    ax2.spines[:].set_visible(False); ax2.margins(0.03)
    st.save(fig2, "fig2_diffusion_final.png")


if __name__ == "__main__":
    main()
