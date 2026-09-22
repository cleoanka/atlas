"""fig6 — the contrastive embedding, in 2D. Left: true labels. Right: labels PREDICTED from
just 10 seeds (one per class, black stars) by graph diffusion; errors in red. The point:
10 labels paint almost the whole map correctly."""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402
_ROOT = os.path.dirname(st.HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
import metrics as M            # noqa: E402
import representations as R    # noqa: E402


def main():
    st.setup()
    X, y = R.contrastive()
    C = len(np.unique(y))
    W = M.knn_graph(X)
    S = M._normalized_operator(W)
    rng = np.random.default_rng(1)
    seeds = np.array([rng.choice(np.where(y == c)[0]) for c in range(C)])
    Yv = np.zeros((len(y), C)); Yv[seeds, y[seeds]] = 1; F = Yv.copy()
    for _ in range(M.ITERS):
        F = M.ALPHA * (S @ F) + (1 - M.ALPHA) * Yv
    pred = F.argmax(1)
    acc = (pred == y).mean()

    # 2D layout for display (subsample for a clean, fast t-SNE)
    sub = rng.choice(len(y), 3000, replace=False)
    sub = np.unique(np.concatenate([sub, seeds]))            # keep the seeds in view
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X[sub])
    ys, ps = y[sub], pred[sub]
    seed_pos = {int(s): i for i, s in enumerate(sub) if s in set(seeds)}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    # left: true labels
    for c in range(C):
        m = ys == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=10, c=st.DIGITS[c], alpha=0.8, label=str(c))
    axes[0].set_title("True labels")
    axes[0].legend(loc="upper right", markerscale=2, ncol=2, fontsize=10, frameon=False,
                   title="digit")
    # right: predicted from 10 labels; errors red; seeds as stars
    err = ps != ys
    for c in range(C):
        m = (ps == c) & ~err
        axes[1].scatter(Z[m, 0], Z[m, 1], s=10, c=st.DIGITS[c], alpha=0.8)
    axes[1].scatter(Z[err, 0], Z[err, 1], s=16, c=st.OK["vermillion"], alpha=0.95,
                    label=f"errors ({err.mean()*100:.1f}%)", zorder=3)
    star_idx = [i for s, i in seed_pos.items()]
    axes[1].scatter(Z[star_idx, 0], Z[star_idx, 1], s=340, marker="*", c="black",
                    edgecolors="white", linewidths=1.2, zorder=5, label="the 10 labels")
    axes[1].set_title(f"Predicted from 10 labels — {acc*100:.1f}% correct")
    axes[1].legend(loc="upper right", markerscale=1, fontsize=11, frameon=False)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    fig.suptitle("Ten labels paint the whole manifold", fontsize=18, fontweight="bold", y=1.0)
    st.save(fig, "fig6_embedding.png")


if __name__ == "__main__":
    main()
