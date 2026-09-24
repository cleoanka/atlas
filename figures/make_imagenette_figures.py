"""ImageNette figures — the same story on real photos. Run after:
    python src/train_contrastive_imagenette.py ...   (-> data/imagenette_emb.npz)
    python src/evaluate.py --dataset imagenette      (-> results/imagenette_results.json)
    python src/sweep.py --dataset imagenette         (-> results/imagenette_sweep.json)
Produces figures/imagenette_*.png (representation bar, headline, embedding, phase, budget)."""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402
_ROOT = os.path.dirname(st.HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
KICK = "ATLAS · IMAGENETTE"


def _res():
    return json.load(open(os.path.join(_ROOT, "results", "imagenette_results.json")))


def representation():
    res = _res()
    order = ["raw", "pca50", "diffusion_map", "contrastive"]
    rows = {r["name"]: r for r in res["rows"]}
    labels = ["raw\npixels", "PCA-50", "diffusion\nmap", "contrastive\nembedding"]
    acc = [rows[n]["diffusion"] * 100 for n in order]
    ceiling = res["full_label_linear_upper_bound"] * 100
    cols = [st.MUTED, st.MUTED, st.MUTED, st.ORANGE]
    fig = plt.figure(figsize=(9.2, 6.6)); ax = fig.add_axes([0.11, 0.11, 0.85, 0.585])
    st.header(fig, "The label was never the hard part",
              "ImageNette · same 10 labels every time — only the representation changes",
              kicker=KICK, accent=st.ORANGE)
    x = np.arange(4); bars = ax.bar(x, acc, width=0.66, color=cols, zorder=3)
    ax.axhline(ceiling, ls=(0, (4, 3)), color=st.INK, lw=1.5, zorder=2)
    ax.text(-0.38, ceiling + 1.2, f"fully-supervised ceiling · {ceiling:.1f}%", ha="left",
            va="bottom", fontsize=11.5, color=st.INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("accuracy from 10 labels (%)"); ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    st.grid(ax, axis="y")
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=18, fontweight="bold", color=st.INK)
    gap = acc[3] - max(acc[:3])
    ax.annotate(f"+{gap:.0f} pts", xy=(2.66, acc[3] - 4), xytext=(2.16, min(acc[:3]) - 8),
                fontsize=16, color=st.ORANGE, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=st.ORANGE, lw=2.6,
                                connectionstyle="arc3,rad=-0.2"))
    st.save(fig, "imagenette_representation.png")


def headline():
    res = _res()
    ten = next(r for r in res["rows"] if r["name"] == "contrastive")["diffusion"] * 100
    full = res["full_label_linear_upper_bound"] * 100
    nfull = res["n_labels_full"]
    fig = plt.figure(figsize=(8.4, 6.4)); ax = fig.add_axes([0.13, 0.10, 0.82, 0.60])
    st.header(fig, "Ten labels, real photographs",
              "ImageNette · contrastive representation · graph diffusion · 60-trial mean",
              kicker=KICK, accent=st.ORANGE)
    bars = ax.bar([0, 1], [ten, full], color=[st.ORANGE, st.MUTED], width=0.6, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["10 labels\n(1 per class)",
                                               f"~{nfull} labels\n(fully supervised)"], fontsize=14)
    ax.set_ylabel("ImageNette accuracy (%)"); ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    st.grid(ax, axis="y")
    for b, v in zip(bars, [ten, full]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=24, fontweight="bold", color=st.INK)
    ax.annotate("", xy=(0.7, full), xytext=(0.3, ten),
                arrowprops=dict(arrowstyle="<->", color=st.INK, lw=2.0))
    ax.text(0.5, (ten + full) / 2 + 4, f"{full-ten:.1f}\npts", ha="center", va="bottom",
            fontsize=13.5, color=st.INK, fontweight="bold")
    st.save(fig, "imagenette_headline.png")


def embedding():
    from sklearn.manifold import TSNE
    import metrics as M
    import representations as R
    X, y = R.contrastive("imagenette")
    C = len(np.unique(y))
    W = M.knn_graph(X); S = M._normalized_operator(W)
    rng = np.random.default_rng(1)
    seeds = np.array([rng.choice(np.where(y == c)[0]) for c in range(C)])
    Yv = np.zeros((len(y), C)); Yv[seeds, y[seeds]] = 1; F = Yv.copy()
    for _ in range(M.ITERS):
        F = M.ALPHA * (S @ F) + (1 - M.ALPHA) * Yv
    pred = F.argmax(1); acc = (pred == y).mean()
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X)
    star = list(seeds)
    fig = plt.figure(figsize=(13.2, 7.0))
    st.header(fig, "Ten labels paint the whole manifold",
              "ImageNette contrastive embedding (t-SNE) · left: truth · right: diffused from 10 seeds",
              kicker=KICK, accent=st.GREEN)
    axL = fig.add_axes([0.02, 0.03, 0.46, 0.72]); axR = fig.add_axes([0.52, 0.03, 0.46, 0.72])
    try:
        import imagenette_data as ID
        names = ID.LABEL_NAMES
    except Exception:
        names = [str(i) for i in range(C)]
    for c in range(C):
        m = y == c
        axL.scatter(Z[m, 0], Z[m, 1], s=12, c=st.DIGITS[c], alpha=0.85, label=names[c], edgecolors="none")
    axL.set_title("True labels", fontsize=15, fontweight="bold", color=st.INK, pad=8)
    axL.legend(loc="center left", bbox_to_anchor=(0.99, 0.5), markerscale=2.2, fontsize=9,
               frameon=False, handletextpad=0.2)
    err = pred != y
    for c in range(C):
        m = (pred == c) & ~err
        axR.scatter(Z[m, 0], Z[m, 1], s=12, c=st.DIGITS[c], alpha=0.85, edgecolors="none")
    axR.scatter(Z[err, 0], Z[err, 1], s=18, c=st.RED, alpha=0.9, zorder=3, label=f"errors ({err.mean()*100:.1f}%)")
    axR.scatter(Z[star, 0], Z[star, 1], s=360, marker="*", c=st.INK, edgecolors="white",
                linewidths=1.3, zorder=5, label="the 10 labels")
    axR.set_title(f"Predicted from 10 labels — {acc*100:.1f}% correct", fontsize=15,
                  fontweight="bold", color=st.INK, pad=8)
    axR.legend(loc="lower right", fontsize=11, frameon=False)
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    st.save(fig, "imagenette_embedding.png")


def phase():
    p = os.path.join(_ROOT, "results", "imagenette_sweep.json")
    if not os.path.exists(p):
        print("  (skip phase: run sweep --dataset imagenette first)"); return
    sw = sorted(json.load(open(p))["purity_sweep"], key=lambda d: d["purity"])
    pur = np.array([d["purity"] for d in sw]) * 100
    euc = np.array([d["euclid_1nn"] for d in sw]) * 100
    dif = np.array([d["diffusion"] for d in sw]) * 100
    fig = plt.figure(figsize=(9, 6.6)); ax = fig.add_axes([0.11, 0.11, 0.85, 0.60])
    st.header(fig, "Metric quality has a phase transition",
              "ImageNette · contrastive embedding degraded with Gaussian noise · crossover ≈ 62% purity",
              kicker=KICK, accent=st.BLUE)
    ax.plot(pur, euc, "-o", color=st.MUTED, mfc="white", mew=2, ms=8, label="Euclid 1-NN (naive)")
    ax.plot(pur, dif, "-o", color=st.ORANGE, mfc="white", mew=2, ms=8, label="graph diffusion")
    st.grid(ax); ax.set_xlabel("edge purity  —  % of graph edges within a class")
    ax.set_ylabel("1-label-per-class accuracy (%)"); ax.legend(loc="upper left", frameon=False)
    st.save(fig, "imagenette_phase.png")


def main():
    representation(); headline(); embedding(); phase()
    # experiment figures (skip gracefully if their result JSONs aren't present yet)
    import make_encoder_comparison
    import make_imagenette_epoch_scan
    make_encoder_comparison.main()
    make_imagenette_epoch_scan.main()


if __name__ == "__main__":
    main()
