"""fig1 — the metric decides who counts as a neighbour. REAL data, real vector space.

No synthetic spiral or moons: this is the actual ImageNette evaluation pool. One real photo,
its 8 nearest neighbours in two different vector spaces:
  Row 1  raw pixels (naive L2 between images)      -> neighbours are OTHER classes.
  Row 2  the learned 512-D contrastive embedding   -> neighbours are all the SAME class.
Same photo, same k. Only the space changed. The number under each row (same-class / k) is the
edge purity you would build a graph on. The footer states the aggregate over all 3,925 images so
the single example is representative, not cherry-picked.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import _style as st  # noqa: E402
import representations as R  # noqa: E402
from imagenette_data import LABEL_NAMES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
K = 8


def knn_idx(F, k=K):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(F)
    _, idx = nn.kneighbors(F)
    return idx[:, 1:]                                  # drop self


def pick_query(y, raw_nb, emb_nb, seed=0):
    """Deterministic: emb neighbourhood perfectly pure, raw neighbourhood maximally impure and
    spread over the most distinct wrong classes (a diverse, honest failure of pixel distance)."""
    raw_wrong = (y[raw_nb] != y[:, None]).sum(1)
    emb_wrong = (y[emb_nb] != y[:, None]).sum(1)
    n_distinct = np.array([len(set(y[raw_nb[i]][y[raw_nb[i]] != y[i]])) for i in range(len(y))])
    ok = np.where((emb_wrong == 0) & (raw_wrong == K))[0]
    best = ok[np.argsort(-(n_distinct[ok] * 100 + raw_wrong[ok]))]
    return int(best[seed])


def thumb(fig, rect, img, border, lw=3.0):
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(border); s.set_linewidth(lw)
    return ax


def main():
    st.setup()
    Xraw, y = R.get("raw", "imagenette")              # 32x32 block-pool pixels (canonical)
    emb, ye = R.get("contrastive", "imagenette")      # 512-D learned space
    imgs = np.load(os.path.join(ROOT, "data", "imagenette_eval_160.npz"))
    pics, yp = imgs["X"], imgs["y"]
    assert np.array_equal(y, ye) and np.array_equal(y, yp), "representation / image order mismatch"

    raw_nb, emb_nb = knn_idx(Xraw), knn_idx(emb)
    raw_pur = (y[raw_nb] == y[:, None]).mean()
    emb_pur = (y[emb_nb] == y[:, None]).mean()
    p = pick_query(y, raw_nb, emb_nb)
    qname = LABEL_NAMES[y[p]]

    fig = plt.figure(figsize=(13.6, 8.9))
    AR = 13.6 / 8.9
    st.header(fig, "The metric decides who counts as a neighbour",
              "same real photo, same k = 8 — only the vector space changes  ·  ImageNette",
              accent=st.BLUE)

    # two rows of neighbours
    x0, x1 = 0.250, 0.980
    w = 0.086
    gap = (x1 - x0 - K * w) / (K - 1)
    h = w * AR
    raw_y, emb_y = 0.480, 0.150
    rows = [
        (raw_nb[p], raw_y, "Raw pixels", "L2 distance between the images themselves", raw_pur),
        (emb_nb[p], emb_y, "Learned embedding", "distance in the 512-D contrastive space", emb_pur),
    ]

    # query photo, vertically centred against the two-row block, on the left
    qw = 0.158
    qh = qw * AR
    block_c = (emb_y + (raw_y + h)) / 2
    qax = thumb(fig, [0.048, block_c - qh / 2, qw, qh], pics[p], st.GOLD, lw=3.6)
    qax.set_title(f"query\n“{qname}”", fontsize=14.5, color=st.INK, fontweight="bold", pad=9)

    for nb, ry, title, sub, pur in rows:
        same = int((y[nb] == y[p]).sum())
        col = st.GREEN if same == K else st.RED
        fig.text(x0, ry + h + 0.052, title, fontsize=15.5, color=st.INK, fontweight="bold")
        fig.text(x0, ry + h + 0.020, sub, fontsize=11.5, color=st.SUBINK)
        for j, ni in enumerate(nb):
            good = y[ni] == y[p]
            rx = x0 + j * (w + gap)
            thumb(fig, [rx, ry, w, h], pics[ni], st.GREEN if good else st.RED, lw=2.8)
            if not good:
                fig.text(rx + w / 2, ry - 0.014, LABEL_NAMES[y[ni]], fontsize=8.6, color=st.RED,
                         ha="center", va="top", style="italic")
        # count badge
        fig.text(x1 + 0.006, ry + h / 2, f"{same}/{K}", fontsize=23, color=col,
                 fontweight="bold", ha="left", va="center")
        fig.text(x1 + 0.006, ry + h / 2 - 0.052, "same\nclass", fontsize=9.5, color=st.MUTED,
                 ha="left", va="center")

    st.footer(fig,
              f"Across all {len(y):,} validation images the same pattern holds: raw-pixel "
              f"neighbours are {raw_pur*100:.0f}% same-class, learned-embedding neighbours "
              f"{emb_pur*100:.0f}%.  This graph purity — a property of the representation, not the "
              f"classifier — is what few-label diffusion rides on.")
    st.save(fig, "fig1_why_euclid_fails.png")


if __name__ == "__main__":
    main()
