"""fig8 — the money shot. The label budget is fixed at 10; ONLY the representation changes.
Three representations sit at ~72-75%; the contrastive metric leaps to 95%. The 23-point jump
is 100% representation — the label never was the hard part."""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402
_ROOT = os.path.dirname(st.HERE)


def main():
    st.setup()
    res = json.load(open(os.path.join(_ROOT, "results", "results.json")))
    order = ["raw", "pca50", "diffusion_map", "contrastive"]
    rows = {r["name"]: r for r in res["rows"]}
    labels = ["raw\npixels", "PCA-50", "diffusion\nmap", "contrastive\nembedding"]
    acc = [rows[n]["diffusion"] * 100 for n in order]
    ceiling = res["full_label_linear_upper_bound"] * 100
    colors = [st.MUTED, st.MUTED, st.MUTED, st.ORANGE]

    fig = plt.figure(figsize=(9.2, 6.6))
    ax = fig.add_axes([0.11, 0.11, 0.85, 0.585])
    st.header(fig, "The label was never the hard part",
              "same 10 labels every time — only the representation changes", accent=st.ORANGE)

    x = np.arange(4)
    bars = ax.bar(x, acc, width=0.66, color=colors, zorder=3)
    ax.axhline(ceiling, ls=(0, (4, 3)), color=st.INK, lw=1.5, zorder=2)
    ax.text(-0.38, ceiling + 1.0, f"fully-supervised ceiling · {ceiling:.1f}%",
            ha="left", va="bottom", fontsize=11.5, color=st.INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("accuracy from 10 labels (%)")
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    st.grid(ax, axis="y")
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=18, fontweight="bold", color=st.INK)
    # the jump — a compact tag; the title already carries the sentence
    ax.annotate(f"+{acc[3]-acc[2]:.0f} pts", xy=(2.66, acc[3] - 4), xytext=(2.16, 62),
                fontsize=16, color=st.ORANGE, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=st.ORANGE, lw=2.6,
                                connectionstyle="arc3,rad=-0.2"))
    st.save(fig, "fig8_representation.png")


if __name__ == "__main__":
    main()
