"""fig5 — labels scale with MODES, not classes. Cluster PCA-50 into N groups and name each
with one label; accuracy climbs toward the per-cluster majority ceiling. The gap between
'1 label/class' and '1 label/cluster' is the naming budget. From results/sweep.json."""
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
        lb = json.load(fh)["label_budget"]
    pts = lb["points"]
    n = np.array([d["clusters"] for d in pts])
    one = np.array([d["one_per_cluster"] for d in pts]) * 100
    maj = np.array([d["majority"] for d in pts]) * 100
    per_class = lb["per_class_diffusion"] * 100

    fig = plt.figure(figsize=(9, 6.6))
    ax = fig.add_axes([0.11, 0.11, 0.85, 0.60])
    st.header(fig, "Labels scale with modes, not classes",
              "PCA-50 · KMeans · one label per cluster (hard assignment)", accent=st.GREEN)

    ax.fill_between(n, one, maj, color=st.MUTED, alpha=0.13, zorder=0)
    ax.axhline(per_class, ls=(0, (4, 3)), color=st.BLUE, lw=2.2,
               label=f"1 label / class (10 labels): {per_class:.0f}%")
    ax.plot(n, maj, "-o", color=st.GREEN, mfc="white", mew=2, ms=8,
            label="majority / cluster (ceiling)", zorder=4)
    ax.plot(n, one, "-o", color=st.ORANGE, mfc="white", mew=2, ms=8,
            label="1 label / cluster", zorder=4)
    st.grid(ax)

    j = min(range(len(n)), key=lambda i: abs(n[i] - 30))
    ax.annotate("naming budget\npenalty of 1 random\nlabel vs. majority",
                xy=(n[j], (one[j] + maj[j]) / 2), xytext=(102, 55),
                fontsize=12, color=st.SUBINK, ha="center",
                arrowprops=dict(arrowstyle="->", color=st.MUTED, lw=1.6))
    ax.set_xlabel("number of clusters  —  the number of names you give")
    ax.set_ylabel("MNIST accuracy (%)")
    ax.legend(loc="lower right", frameon=False)
    st.save(fig, "fig5_label_budget.png")


if __name__ == "__main__":
    main()
