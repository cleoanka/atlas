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

    fig, ax = plt.subplots(figsize=(9, 6.3))
    ax.axhline(per_class, ls="--", color=st.OK["blue"], lw=2.2,
               label=f"1 label / class  (10 labels): {per_class:.0f}%")
    ax.plot(n, one, "-o", color=st.OK["vermillion"], label="1 label / cluster", zorder=3)
    ax.plot(n, maj, "-o", color=st.OK["green"], label="majority / cluster (ceiling)", zorder=3)
    ax.fill_between(n, one, maj, color=st.OK["grey"], alpha=0.12, zorder=0)

    # annotate the naming-budget gap — arrow to a mid gap, text in the empty lower-right
    j = min(range(len(n)), key=lambda i: abs(n[i] - 30))
    ax.annotate("naming budget\n(penalty of 1 random\nlabel vs. majority)",
                xy=(n[j], (one[j] + maj[j]) / 2),
                xytext=(105, 56), fontsize=12, color=st.OK["grey"], fontweight="bold",
                ha="center", arrowprops=dict(arrowstyle="->", color=st.OK["grey"], lw=1.6))
    ax.set_xlabel("number of clusters  (= number of names you give)")
    ax.set_ylabel("MNIST accuracy (%)")
    ax.set_title("Labels scale with the number of modes, not classes")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    ax.grid(alpha=0.22)
    fig.text(0.5, -0.02, "MNIST · PCA-50 · KMeans · results/sweep.json",
             ha="center", fontsize=11, color=st.OK["grey"])
    st.save(fig, "fig5_label_budget.png")


if __name__ == "__main__":
    main()
