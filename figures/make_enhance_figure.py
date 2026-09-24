"""fig9 — what actually helps? Δ accuracy of each geometric upgrade vs the baseline, both
datasets. Metric upgrades (a locally-adaptive / Riemannian ruler) barely move an already-good
representation; choosing the *most typical* point as the label is the big lever — hugely so on
hard data. Reads results/{mnist,imagenette}_enhance.json."""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as st  # noqa: E402
_ROOT = os.path.dirname(st.HERE)

LABELS = {
    "cosine": "cosine metric",
    "self_tuning scale": "self-tuning scale",
    "anisotropic α=1 (Laplace-Beltrami)": "anisotropic α=1  (Laplace–Beltrami)",
    "local Mahalanobis g(x)": "local Mahalanobis  g(x)  ← Riemannian",
    "processing: SNN bridge-kill": "bridge-kill (SNN)  ← processing",
    "seed: density-peak": "seed: density-peak",
    "seed: medoid": "seed: medoid  ← most typical",
    "medoid + SNN processing": "medoid + bridge-kill  ← seed + processing",
}


def _deltas(dataset):
    d = json.load(open(os.path.join(_ROOT, "results", f"{dataset}_enhance.json")))
    base = d["rows"][0]["acc"]
    out = {}
    for r in d["rows"]:
        if r["config"] in LABELS:
            out[r["config"]] = (r["acc"] - base) * 100
    return out, base * 100


def main():
    for ds in ("mnist", "imagenette"):
        if not os.path.exists(os.path.join(_ROOT, "results", f"{ds}_enhance.json")):
            print(f"  (skip fig9: run `python src/enhance.py --dataset {ds}` first)")
            return
    st.setup()
    dm, bm = _deltas("mnist")
    di, bi = _deltas("imagenette")
    keys = list(LABELS)
    y = np.arange(len(keys))[::-1]
    vm = [dm[k] for k in keys]
    vi = [di[k] for k in keys]

    fig = plt.figure(figsize=(10.8, 7.8))
    ax = fig.add_axes([0.36, 0.09, 0.60, 0.63])
    st.header(fig, "What actually helps the same 10 labels",
              "Δ accuracy vs baseline · metric ≈ flat · bridge-killing helps a little · the typical seed is the lever",
              accent=st.GREEN)
    h = 0.38
    ax.barh(y + h / 2, vm, height=h, color=st.BLUE, alpha=0.25, zorder=2)
    ax.barh(y + h / 2, vm, height=h, color=st.BLUE, zorder=3, label="MNIST")
    ax.barh(y - h / 2, vi, height=h, color=st.ORANGE, alpha=0.25, zorder=2)
    ax.barh(y - h / 2, vi, height=h, color=st.ORANGE, zorder=3, label="ImageNette")
    ax.axvline(0, color=st.SUBINK, lw=1.2, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k in keys], fontsize=11.5)
    ax.set_xlabel("Δ accuracy vs random-seed baseline (points)")
    ax.set_xlim(-4, 20)
    st.grid(ax, axis="x")
    for yi, v in zip(y + h / 2, vm):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, color=st.BLUE, fontweight="bold")
    for yi, v in zip(y - h / 2, vi):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, color=st.ORANGE, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.text(0.34, 0.045, f"baselines: MNIST {bm:.1f}%  ·  ImageNette {bi:.1f}%   (contrastive, 1 label/class)",
             fontsize=10.5, color=st.MUTED)
    st.save(fig, "fig9_enhance.png")


if __name__ == "__main__":
    main()
