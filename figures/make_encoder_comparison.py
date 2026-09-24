"""imagenette_encoder_comparison — does a BIGGER encoder help few-label? ResNet-18 vs ResNet-50.

The bigger 2048-D encoder raises the linear-probe ceiling yet LOWERS few-label diffusion at every
fair (medoid-seed) regime. Left: few-label accuracy per regime, both encoders, with each encoder's
ceiling as a dashed line. Right: the diagnostics that explain it — the 2048-D space concentrates
distances, so its kNN graph is less pure and carries more cross-class bridges. Reads
results/imagenette_encoder_comparison.json."""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402
ROOT = os.path.dirname(st.HERE)

REGIMES = [("seed_random", "random\nseed"), ("medoid_plain", "medoid"),
           ("medoid_mahalanobis", "+ Mahalanobis"), ("medoid_snn", "+ SNN"),
           ("medoid_mahalanobis_snn", "+ both")]


def main():
    path = os.path.join(ROOT, "results", "imagenette_encoder_comparison.json")
    if not os.path.exists(path):
        print("  (run `python src/encoder_comparison.py` first)"); return
    d = json.load(open(path))
    r18, r50 = d["resnet18"], d["resnet50"]
    v18 = [r18["regimes"][k] for k, _ in REGIMES]
    v50 = [r50["regimes"][k] for k, _ in REGIMES]

    st.setup()
    fig = plt.figure(figsize=(12.6, 6.9))
    st.header(fig, "A bigger encoder raises the ceiling — and lowers few-label",
              "ResNet-18 (512-D) vs ResNet-50 (2048-D) · same contrastive recipe · 1 label/class, medoid seed",
              accent=st.ORANGE)

    ax = fig.add_axes([0.065, 0.11, 0.58, 0.62])
    x = np.arange(len(REGIMES)); w = 0.38
    st.glow_bars(ax, x - w / 2, v18, [st.BLUE] * len(x), width=w)
    st.glow_bars(ax, x + w / 2, v50, [st.ORANGE] * len(x), width=w)
    for xi, a, b in zip(x, v18, v50):
        ax.text(xi - w / 2, a + 0.6, f"{a:.1f}", ha="center", va="bottom", fontsize=10,
                color=st.BLUE, fontweight="bold")
        ax.text(xi + w / 2, b + 0.6, f"{b:.1f}", ha="center", va="bottom", fontsize=10,
                color=st.ORANGE, fontweight="bold")
    # ceilings as dashed lines
    ax.axhline(r18["ceiling"], color=st.BLUE, ls=(0, (5, 4)), lw=1.8, alpha=0.9)
    ax.axhline(r50["ceiling"], color=st.ORANGE, ls=(0, (5, 4)), lw=1.8, alpha=0.9)
    ax.text(len(x) - 0.5, r50["ceiling"] + 0.3, f"RN-50 ceiling {r50['ceiling']:.1f}%",
            ha="right", va="bottom", fontsize=9.5, color=st.ORANGE)
    ax.text(len(x) - 0.5, r18["ceiling"] - 1.7, f"RN-18 ceiling {r18['ceiling']:.1f}%",
            ha="right", va="bottom", fontsize=9.5, color=st.BLUE)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in REGIMES], fontsize=11)
    ax.set_ylabel("1-label diffusion accuracy (%)")
    ax.set_ylim(52, 88)
    st.grid(ax, axis="y")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=st.BLUE),
               plt.Rectangle((0, 0), 1, 1, color=st.ORANGE)],
              ["ResNet-18  (512-D)", "ResNet-50  (2048-D)"], loc="lower right", frameon=False)

    # right: the "why" diagnostics
    hx = 0.70
    fig.text(hx, 0.70, "why the bigger space is a noisier graph", fontsize=13,
             color=st.INK, fontweight="bold")
    h18, h50 = r18["hubness"], r50["hubness"]
    diag = [
        ("linear-probe ceiling", f"{r18['ceiling']:.1f}%", f"{r50['ceiling']:.1f}%", True),
        ("edge purity", f"{r18['regimes']['purity']:.3f}", f"{r50['regimes']['purity']:.3f}", False),
        ("cross-class bridges", f"{r18['regimes']['bridges']}", f"{r50['regimes']['bridges']}", False),
        ("dist. concentration\n(std/mean, higher=better)", f"{h18['dist_concentration']:.3f}",
         f"{h50['dist_concentration']:.3f}", False),
    ]
    yv = 0.60
    fig.text(hx + 0.145, 0.645, "RN-18", fontsize=10, color=st.BLUE, fontweight="bold", ha="center")
    fig.text(hx + 0.235, 0.645, "RN-50", fontsize=10, color=st.ORANGE, fontweight="bold", ha="center")
    for name, a, b, up in diag:
        fig.text(hx, yv, name, fontsize=10.5, color=st.SUBINK, va="top")
        fig.text(hx + 0.145, yv, a, fontsize=11, color=st.BLUE, ha="center", va="top", fontweight="bold")
        fig.text(hx + 0.235, yv, b, fontsize=11, color=st.ORANGE, ha="center", va="top", fontweight="bold")
        yv -= 0.085
    fig.text(hx, 0.135,
             "More linear info, but distances concentrate →\nlower kNN purity, more bridges → "
             "few-label leaks.\nMahalanobis + SNN repair it most on RN-50,\nyet it stays ~1 pt behind.",
             fontsize=10, color=st.MUTED, va="top")

    st.save(fig, "imagenette_encoder_comparison.png")


if __name__ == "__main__":
    main()
