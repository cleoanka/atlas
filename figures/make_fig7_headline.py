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

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar([0, 1], [ten, full],
                  color=[st.OK["vermillion"], st.OK["grey"]], width=0.62, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["10 labels\n(1 per class)", "7000 labels\n(fully supervised)"],
                       fontsize=15)
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for b, v in zip(bars, [ten, full]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=22, fontweight="bold")
    # white callout INSIDE the orange bar (no overlap with anything)
    ax.text(0, 47, "700×\nfewer\nlabels", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")
    # gap callout: connect the two bar tops, "only N pts lower" above the gap
    ax.annotate("", xy=(0.69, full), xytext=(0.31, ten),
                arrowprops=dict(arrowstyle="<->", color=st.OK["vermillion"], lw=2.2))
    ax.text(0.5, (ten + full) / 2 + 3.5, f"only {full-ten:.1f}\npts lower", ha="center",
            va="bottom", fontsize=14, color=st.OK["vermillion"], fontweight="bold")
    ax.set_title("With a good enough metric, one label per class is almost enough",
                 fontsize=15.5, pad=14)
    fig.text(0.5, -0.02, "MNIST · contrastive representation · graph diffusion · 60-trial mean",
             ha="center", fontsize=11, color=st.OK["grey"])
    st.save(fig, "fig7_headline.png")


if __name__ == "__main__":
    main()
