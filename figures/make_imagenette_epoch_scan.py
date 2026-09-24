"""imagenette_epoch_scan — the honest answer to "how many epochs?", read off downstream.

SSL contrastive loss never shows an overfitting bump, and its value is a poor proxy for what
atlas actually needs (edge purity / few-label accuracy). So we do NOT early-stop on val loss.
Instead the trainer exports the validation embedding at every 25-epoch checkpoint
(data/imagenette_emb_ep{N}.npz); this script evaluates each one — GPU-free — with the SAME
few-label protocol as the rest of the repo (1 label/class, graph diffusion) and plots the metric
that matters against epoch. The optimum epoch is where few-label accuracy plateaus, not where a
loss curve bottoms out.

Run:  python figures/make_imagenette_epoch_scan.py   (after a training run with the updated script)
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import _style as st  # noqa: E402
import metrics as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_OUT = os.path.join(ROOT, "results", "imagenette_epoch_scan.json")
TRIALS = 40


def scan():
    files = glob.glob(os.path.join(ROOT, "data", "imagenette_emb_ep*.npz"))
    pairs = []
    for f in files:
        m = re.search(r"_ep(\d+)\.npz$", f)
        if m:
            pairs.append((int(m.group(1)), f))
    pairs.sort()
    rows = []
    for ep, f in pairs:
        d = np.load(f)
        emb, y = d["emb"].astype("float64"), d["y"]
        W = M.knn_graph(emb)
        r = M.one_label_per_class(emb, W, y, trials=TRIALS)
        r["epoch"] = ep
        rows.append(r)
        print(f"  ep{ep:>3}  purity={r['purity']:.3f}  1-label diffusion={r['diffusion']*100:5.1f}%"
              f"  (euclid-1NN {r['euclid_1nn']*100:4.1f}%)")
    if rows:                                            # persist so the figure is reproducible
        with open(JSON_OUT, "w") as fh:                #   without the heavy per-epoch embeddings
            json.dump(rows, fh, indent=2)
    return rows


def main():
    rows = scan()
    if not rows and os.path.exists(JSON_OUT):           # per-epoch npz are gitignored/local; fall
        rows = json.load(open(JSON_OUT))                #   back to the shipped scan numbers
        print(f"  (loaded {len(rows)} checkpoints from {os.path.relpath(JSON_OUT, ROOT)})")
    if not rows:
        print("no data/imagenette_emb_ep*.npz and no results/imagenette_epoch_scan.json — run "
              "training with the updated train_contrastive_imagenette.py first.")
        return
    ep = np.array([r["epoch"] for r in rows])
    dif = np.array([r["diffusion"] for r in rows])
    difs = np.array([r["diffusion_std"] for r in rows])
    pur = np.array([r["purity"] for r in rows])
    e1 = np.array([r["euclid_1nn"] for r in rows])
    best = int(np.argmax(dif))

    if len(rows) < 2:
        print(f"only one checkpoint (ep{ep[0]}) — need >=2 for a scan plot. "
              f"1-label diffusion {dif[0]*100:.1f}%.")
        return

    st.setup()
    fig = plt.figure(figsize=(11.0, 6.8))
    st.header(fig, "How many epochs? Read it off the few-label metric, not the loss",
              "ImageNette · 1 label/class, graph diffusion — the number atlas actually optimises",
              accent=st.GREEN)
    ax = fig.add_axes([0.085, 0.13, 0.87, 0.62])

    st.glow_line(ax, ep, e1 * 100, st.BLUE, lw=2.4, label="Euclid-1NN (baseline)")
    st.glow_line(ax, ep, pur * 100, st.GOLD, lw=2.4, label="edge purity")
    ax.fill_between(ep, (dif - difs) * 100, (dif + difs) * 100, color=st.GREEN, alpha=0.12, zorder=2)
    st.glow_line(ax, ep, dif * 100, st.GREEN, lw=3.2, label="1-label diffusion (the metric)")

    ax.scatter([ep[best]], [dif[best] * 100], s=240, marker="*", c=st.GREEN,
               edgecolors="white", linewidths=1.2, zorder=8)
    ax.annotate(f"optimum ≈ ep {ep[best]}\n{dif[best]*100:.1f}%",
                (ep[best], dif[best] * 100), textcoords="offset points", xytext=(10, -34),
                fontsize=12.5, color=st.GREEN, fontweight="bold")

    ax.set_xlabel("training epoch"); ax.set_ylabel("accuracy / purity  (%)")
    ax.set_xlim(ep.min() - 5, ep.max() + 5)
    ax.grid(True, color=st.HAIR, alpha=1.0); ax.set_axisbelow(True)
    ax.spines["left"].set_color(st.HAIR); ax.spines["bottom"].set_color(st.HAIR)
    leg = ax.legend(loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(st.SUBINK)
    st.footer(fig, "SSL contrastive loss keeps falling and never flags overfitting; the few-label "
                   "metric plateaus, and that plateau — not a loss curve — is where to stop.")
    st.save(fig, "imagenette_epoch_scan.png")


if __name__ == "__main__":
    main()
