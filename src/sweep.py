"""Two stress tests that draw the boundary of the idea.

  1. PURITY SWEEP — degrade the PCA-50 metric with noise until edge purity collapses, and
     watch the phase transition: diffusion accuracy tracks purity steeply, Euclid-1NN barely
     moves, and below a crossover purity the "clever" graph method does WORSE than the naive
     baseline. This is the "when it doesn't work" evidence, made quantitative.

  2. LABEL BUDGET — labels scale with the number of MODES, not classes. Cluster PCA-50 into
     N groups; one label per cluster approaches the per-cluster majority ceiling. The gap
     between "1 label/class" and "1 label/cluster" is the naming budget.

    python src/sweep.py     # writes results/sweep.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import KMeans

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "src"))

import metrics as M            # noqa: E402
import representations as R    # noqa: E402


def purity_sweep(dataset="mnist", base_rep="pca50",
                 noise_levels=(0.0, 0.4, 0.7, 0.9, 1.1, 1.4, 1.9, 2.8),
                 k=M.K_DEFAULT, trials=20, seed=0):
    """Add isotropic Gaussian noise (× feature std) to a base representation; record purity/acc.

    base_rep starts ABOVE the transition so degrading it crosses the threshold (mnist: PCA-50;
    imagenette: contrastive, since pixel spaces there are already near-chance)."""
    X0, y = R.get(base_rep, dataset)
    std = float(X0.std())
    rng = np.random.default_rng(seed)
    pts = []
    for nl in noise_levels:
        X = X0 + rng.standard_normal(X0.shape).astype("float32") * (nl * std)
        W = M.knn_graph(X, k)
        res = M.one_label_per_class(X, W, y, trials=trials)
        pts.append({"noise": float(nl), "purity": res["purity"],
                    "euclid_1nn": res["euclid_1nn"], "diffusion": res["diffusion"]})
        print(f"  noise={nl:4.1f}  purity={res['purity']*100:5.1f}%  "
              f"euclid={res['euclid_1nn']*100:5.1f}%  diffusion={res['diffusion']*100:5.1f}%",
              flush=True)
    return pts


def label_budget(dataset="mnist", base_rep="pca50",
                 cluster_counts=(10, 20, 30, 40, 60, 80, 120, 160), trials=25, seed=2):
    """PCA-50 KMeans into N clusters; name each cluster with ONE label (hard assignment).

    Directly measures "how many names do you need": each point gets its cluster's label.
      * majority : cluster labelled by its majority class → the CEILING for this clustering
                   (monotone in N; how well K modes carve the classes).
      * 1/cluster: cluster labelled by ONE random member → the realistic budget (below ceiling
                   by the sampling penalty).
    Reference: '1 label/class' (10 labels) via graph diffusion — a flat line to beat.
    """
    X, y = R.get(base_rep, dataset)
    W = M.knn_graph(X)
    per_class = M.one_label_per_class(X, W, y, trials=trials)["diffusion"]
    rng = np.random.default_rng(seed)
    pts = []
    for n in cluster_counts:
        km = KMeans(n_clusters=n, n_init=10, random_state=0).fit(X)
        lab = km.labels_
        members = [np.where(lab == c)[0] for c in range(n)]
        pred_major = np.empty(len(y), int)
        for c in range(n):
            mm = members[c]
            if mm.size:
                vals, cnts = np.unique(y[mm], return_counts=True)
                pred_major[mm] = vals[cnts.argmax()]
        maj = float((pred_major == y).mean())
        one = []
        for _ in range(trials):
            pred = np.empty(len(y), int)
            for c in range(n):
                mm = members[c]
                if mm.size:
                    pred[mm] = y[rng.choice(mm)]
            one.append((pred == y).mean())
        pts.append({"clusters": int(n), "one_per_cluster": float(np.mean(one)),
                    "majority": maj})
        print(f"  clusters={n:4d}  1/cluster={np.mean(one)*100:5.1f}%  "
              f"majority(ceil)={maj*100:5.1f}%", flush=True)
    return {"per_class_diffusion": float(per_class), "points": pts}


def run(dataset="mnist"):
    base = "contrastive" if dataset == "imagenette" else "pca50"
    print(f"=== purity sweep · {dataset} ({base}, noise-degraded metric) ===")
    t0 = time.time()
    sweep = purity_sweep(dataset, base_rep=base)
    print(f"\n=== label budget · {dataset} ({base}, KMeans clusters) ===")
    budget = label_budget(dataset, base_rep=base)
    out = {"purity_sweep": sweep, "label_budget": budget}
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    name = "sweep.json" if dataset == "mnist" else f"{dataset}_sweep.json"
    with open(os.path.join(_HERE, "results", name), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/{name}  [{time.time()-t0:.0f}s]")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=R.DATASETS)
    a = ap.parse_args()
    run(dataset=a.dataset)
