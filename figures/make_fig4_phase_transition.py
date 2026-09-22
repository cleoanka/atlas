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

    # crossover purity where diffusion overtakes euclid (linear interp on the difference)
    diff = dif - euc
    xover = None
    for i in range(len(pur) - 1):
        if diff[i] <= 0 <= diff[i + 1]:
            f = -diff[i] / (diff[i + 1] - diff[i])
            xover = pur[i] + f * (pur[i + 1] - pur[i])
            break

    fig, ax = plt.subplots(figsize=(9, 6.3))
    ax.plot(pur, euc, "-o", color=st.OK["blue"], label="Euclid 1-NN (naive)", zorder=3)
    ax.plot(pur, dif, "-o", color=st.OK["vermillion"], label="graph diffusion", zorder=3)
    if xover is not None:
        ax.axvline(xover, ls="--", color=st.OK["grey"], lw=1.8, zorder=1)
        ax.axvspan(pur.min() - 2, xover, color=st.OK["grey"], alpha=0.10, zorder=0)
        ax.text(xover - 1.5, 66, "diffusion HURTS\n(worse than naive)", ha="right",
                va="top", fontsize=13, color=st.OK["grey"], fontweight="bold")
        ax.text(xover + 1.5, 66, "diffusion WINS\n(good metric)", ha="left",
                va="top", fontsize=13, color=st.OK["vermillion"], fontweight="bold")
        ax.text(xover, ax.get_ylim()[0] - 0, f"  crossover ≈ {xover:.0f}% purity",
                ha="left", va="bottom", fontsize=12, color=st.OK["grey"])
    ax.set_xlabel("edge purity  (%  of graph edges that stay within a class)")
    ax.set_ylabel("1-label-per-class accuracy (%)")
    ax.set_title("Metric quality has a phase transition")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.22)
    ax.set_ylim(5, 82)
    fig.text(0.5, -0.02, "MNIST · PCA-50 degraded with Gaussian noise · results/sweep.json",
             ha="center", fontsize=11, color=st.OK["grey"])
    st.save(fig, "fig4_phase_transition.png")


if __name__ == "__main__":
    main()
