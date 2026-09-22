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

    sub = rng.choice(len(y), 3000, replace=False)
    sub = np.unique(np.concatenate([sub, seeds]))
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X[sub])
    ys, ps = y[sub], pred[sub]
    star_idx = [i for i, s in enumerate(sub) if s in set(seeds)]

    fig = plt.figure(figsize=(13.2, 7.0))
    st.header(fig, "Ten labels paint the whole manifold",
              "contrastive embedding (t-SNE view) · left: truth · right: diffused from 10 seeds",
              accent=st.GREEN)
    axL = fig.add_axes([0.02, 0.03, 0.46, 0.72]); axR = fig.add_axes([0.52, 0.03, 0.46, 0.72])

    for c in range(C):
        m = ys == c
        axL.scatter(Z[m, 0], Z[m, 1], s=14, c=st.DIGITS[c], alpha=0.85, label=str(c),
                    edgecolors="none")
    axL.set_title("True labels", fontsize=15, fontweight="bold", color=st.INK, pad=8)
    axL.legend(loc="center left", bbox_to_anchor=(0.98, 0.5), markerscale=2.2, ncol=1,
               fontsize=11, frameon=False, title="digit", handletextpad=0.2, columnspacing=0.8)

    err = ps != ys
    for c in range(C):
        m = (ps == c) & ~err
        axR.scatter(Z[m, 0], Z[m, 1], s=14, c=st.DIGITS[c], alpha=0.85, edgecolors="none")
    axR.scatter(Z[err, 0], Z[err, 1], s=22, c=st.RED, alpha=0.95, zorder=3,
                label=f"errors ({err.mean()*100:.1f}%)")
    axR.scatter(Z[star_idx, 0], Z[star_idx, 1], s=380, marker="*", c=st.INK,
                edgecolors="white", linewidths=1.3, zorder=5, label="the 10 labels")
    axR.set_title(f"Predicted from 10 labels — {acc*100:.1f}% correct", fontsize=15,
                  fontweight="bold", color=st.INK, pad=8)
    axR.legend(loc="lower right", fontsize=11.5, frameon=False)
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    st.save(fig, "fig6_embedding.png")


if __name__ == "__main__":
    main()
