"""Does a BIGGER encoder help few-label learning? — ResNet-18 (512-D) vs ResNet-50 (2048-D).

Both encoders are unsupervised contrastive, evaluated with the identical few-label protocol. The
honest finding: the bigger encoder has a marginally HIGHER linear-probe ceiling but a LOWER
few-label diffusion accuracy, at every fair regime. This script measures the fair-regime table
(typicality/medoid seed, not random) and the hubness diagnostics that explain WHY the bigger
2048-D space is a slightly noisier kNN graph. Writes results/imagenette_encoder_comparison.json.

    python src/encoder_comparison.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import skew
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "src"))
import metrics as M       # noqa: E402
import enhance as E       # noqa: E402

RN18 = os.path.join(_HERE, "data", "imagenette_emb.npz")            # shipped headline encoder
RN50 = os.path.join(_HERE, "data", "imagenette_emb_resnet50.npz")   # experiment encoder
RN50_TR = os.path.join(_HERE, "data", "imagenette_train_emb_resnet50.npz")


def fair_regimes(X, y):
    """Few-label diffusion under the FAIR (typicality/medoid) seed, isolating each lever."""
    classes = np.unique(y)
    med = E.seeds_medoid(X, y, classes)

    def acc(W):
        S, _ = E.operator(W, 0.0)
        return M.diffusion_spread(S, med, y[med], y, len(classes)) * 100

    W = E.affinity(X, k=10)
    Wm = E.maha_affinity(X, k=10)
    Ws = E.snn_reweight(X, W)
    Wms = E.snn_reweight(X, Wm)
    # random-seed baseline for reference (60-draw mean)
    S0, _ = E.operator(W, 0.0)
    accs = []
    rng = np.random.default_rng(1)
    for _ in range(60):
        sd = np.array([rng.choice(np.where(y == c)[0]) for c in classes])
        accs.append(M.diffusion_spread(S0, sd, y[sd], y, len(classes)))
    return {
        "seed_random": float(np.mean(accs) * 100),
        "medoid_plain": acc(W),
        "medoid_mahalanobis": acc(Wm),
        "medoid_snn": acc(Ws),
        "medoid_mahalanobis_snn": acc(Wms),
        "purity": M.edge_purity(W, y),
        "bridges": M.bridge_edges(W, y),
    }


def hubness(X, k=10):
    """kNN hubness diagnostics. Skew of the k-occurrence distribution N_k(i) = how many points
    have i among their k neighbours; a long right tail (a few 'hub' points neighbouring everyone)
    is the high-dimensional pathology that manufactures cross-class bridge edges. Also the
    relative distance concentration std/mean (smaller = distances collapse together)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, idx = nn.kneighbors(X)
    occ = np.bincount(idx[:, 1:].ravel(), minlength=len(X))
    # sample pairwise distance concentration on a subset (memory-efficient: O(n²) not O(n²·d))
    rng = np.random.default_rng(0)
    s = rng.choice(len(X), size=min(1000, len(X)), replace=False)
    D = pairwise_distances(X[s])
    off = D[~np.eye(len(s), dtype=bool)]
    return {
        "kocc_skew": float(skew(occ)),
        "hub_max_occ": int(occ.max()),
        "dist_concentration": float(off.std() / (off.mean() + 1e-12)),
        "dim": int(X.shape[1]),
    }


def ceiling(train_path, X, y):
    if not os.path.exists(train_path):
        return None
    d = np.load(train_path)
    return M.linear_probe(d["emb"], d["y"].astype(int), X, y) * 100


def main():
    d18 = np.load(RN18); X18, y = d18["emb"].astype("float64"), d18["y"]
    d50 = np.load(RN50); X50, y50 = d50["emb"].astype("float64"), d50["y"]
    assert np.array_equal(y, y50)

    # RN-18 ceiling from its shipped results (proper full-train probe)
    r18_results = json.load(open(os.path.join(_HERE, "results", "imagenette_results.json")))
    ceil18 = r18_results.get("full_label_linear_upper_bound", 0.0) * 100

    out = {
        "resnet18": {"regimes": fair_regimes(X18, y), "hubness": hubness(X18),
                     "ceiling": ceil18},
        "resnet50": {"regimes": fair_regimes(X50, y), "hubness": hubness(X50),
                     "ceiling": ceiling(RN50_TR, X50, y)},
    }
    with open(os.path.join(_HERE, "results", "imagenette_encoder_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    for name in ("resnet18", "resnet50"):
        r, h, c = out[name]["regimes"], out[name]["hubness"], out[name]["ceiling"]
        print(f"\n=== {name}  ({h['dim']}-D) ===")
        print(f"  ceiling(full-label probe) = {c:.1f}%   purity={r['purity']:.3f}   "
              f"bridges={r['bridges']}")
        print(f"  few-label: random={r['seed_random']:.1f}  medoid={r['medoid_plain']:.1f}  "
              f"+maha={r['medoid_mahalanobis']:.1f}  +snn={r['medoid_snn']:.1f}  "
              f"+both={r['medoid_mahalanobis_snn']:.1f}")
        print(f"  hubness: k-occ skew={h['kocc_skew']:.2f}  max-occ={h['hub_max_occ']}  "
              f"dist-concentration(std/mean)={h['dist_concentration']:.3f}")
    print("\nwrote results/imagenette_encoder_comparison.json")


if __name__ == "__main__":
    main()
