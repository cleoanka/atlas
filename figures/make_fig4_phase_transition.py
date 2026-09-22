"""fig4 — the phase transition. As the metric degrades (edge purity falls), graph diffusion
collapses steeply while Euclid-1NN barely moves; below a crossover purity the graph method
does WORSE than the naive baseline. Drawn from results/sweep.json (real noise sweep)."""
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
    with open(os.path.join(_ROOT, "results", "sweep.json")) as fh:
        sw = json.load(fh)["purity_sweep"]
    pts = sorted(sw, key=lambda d: d["purity"])
    pur = np.array([d["purity"] for d in pts]) * 100
    euc = np.array([d["euclid_1nn"] for d in pts]) * 100
    dif = np.array([d["diffusion"] for d in pts]) * 100

    diff = dif - euc
    xover = None
    for i in range(len(pur) - 1):
        if diff[i] <= 0 <= diff[i + 1]:
            f = -diff[i] / (diff[i + 1] - diff[i])
            xover = pur[i] + f * (pur[i + 1] - pur[i]); break

    fig = plt.figure(figsize=(9, 6.6))
    ax = fig.add_axes([0.11, 0.11, 0.85, 0.60])
    st.header(fig, "Metric quality has a phase transition",
              "one label per class · PCA-50 degraded with Gaussian noise", accent=st.BLUE)

    if xover is not None:
        ax.axvspan(pur.min() - 3, xover, color=st.RED, alpha=0.06, zorder=0)
        ax.axvline(xover, ls=(0, (4, 3)), color=st.MUTED, lw=1.6, zorder=1)
    ax.plot(pur, euc, "-o", color=st.MUTED, mfc="white", mew=2, ms=8,
            label="Euclid 1-NN (naive)", zorder=3)
    ax.plot(pur, dif, "-o", color=st.ORANGE, mfc="white", mew=2, ms=8,
            label="graph diffusion", zorder=4)
    st.grid(ax)
    if xover is not None:
        ax.text(xover - 2, 74, "diffusion HURTS\nworse than naive", ha="right", va="top",
                fontsize=12.5, color=st.RED, fontweight="bold")
        ax.text(xover + 2, 74, "diffusion WINS\ngood metric", ha="left", va="top",
                fontsize=12.5, color=st.GREEN, fontweight="bold")
        ax.text(xover + 0.5, 8, f"crossover ≈ {xover:.0f}%", ha="left", va="bottom",
                fontsize=11.5, color=st.MUTED)
    ax.set_xlabel("edge purity  —  % of graph edges that stay within a class")
    ax.set_ylabel("1-label-per-class accuracy (%)")
    ax.set_ylim(5, 82)
    ax.legend(loc="upper left", frameon=False)
    st.save(fig, "fig4_phase_transition.png")


if __name__ == "__main__":
    main()
