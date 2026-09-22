"""fig7 — the headline bar: 10 labels vs 7000 labels. Built to stand alone on social media."""
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402

_ROOT = os.path.dirname(st.HERE)


def main():
    st.setup()
    with open(os.path.join(_ROOT, "results", "results.json")) as fh:
        res = json.load(fh)
    contr = next(r for r in res["rows"] if r["name"] == "contrastive")
    ten = contr["diffusion"] * 100
    full = res["full_label_linear_upper_bound"] * 100

    fig = plt.figure(figsize=(8.4, 6.4))
    ax = fig.add_axes([0.13, 0.10, 0.82, 0.60])
    st.header(fig, "One label per class is almost enough",
              "MNIST · contrastive representation · graph diffusion · 60-trial mean",
              accent=st.ORANGE)

    bars = ax.bar([0, 1], [ten, full], color=[st.ORANGE, st.MUTED], width=0.6, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["10 labels\n(1 per class)", "7000 labels\n(fully supervised)"], fontsize=14)
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    st.grid(ax, axis="y")
    for b, v in zip(bars, [ten, full]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=26, fontweight="bold", color=st.INK)
    ax.text(0, 46, "700×\nfewer\nlabels", ha="center", va="center", fontsize=19,
            fontweight="bold", color="white")
    ax.annotate("", xy=(0.7, full), xytext=(0.3, ten),
                arrowprops=dict(arrowstyle="<->", color=st.INK, lw=2.0))
    ax.text(0.5, (ten + full) / 2 + 4, f"only {full-ten:.1f}\npts lower", ha="center", va="bottom",
            fontsize=13.5, color=st.INK, fontweight="bold")
    st.save(fig, "fig7_headline.png")


if __name__ == "__main__":
    main()
