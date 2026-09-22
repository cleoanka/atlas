"""Produce the headline table: purity + 1-label-per-class accuracy for every representation.

    python src/evaluate.py            # all representations, 60 trials, writes results/results.json

The single claim, made measurable: with a good enough metric, ONE label per class recovers
~95% of 7000 labels' worth of accuracy — and with a bad metric the same graph method does
WORSE than doing nothing clever (Euclid-1NN). Both halves are in this table.
"""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "src"))

import metrics as M            # noqa: E402
import representations as R    # noqa: E402


def run(trials: int = 60, k: int = M.K_DEFAULT):
    rows = []
    upper = None
    for name, (_, label) in R.REGISTRY.items():
        t0 = time.time()
        X, y = R.get(name)
        W = M.knn_graph(X, k)
        res = M.one_label_per_class(X, W, y, trials=trials)
        res["name"] = name
        res["label"] = label
        rows.append(res)
        if name == "contrastive":                       # ceiling from the best representation
            upper = M.linear_upper_bound(X, y)
        print(f"  {label:<20} purity={res['purity']*100:5.1f}%  "
              f"euclid={res['euclid_1nn']*100:5.1f}%  "
              f"diffusion={res['diffusion']*100:5.1f}% (±{res['diffusion_std']*100:.1f})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    out = {
        "config": {"dataset": "MNIST", "n_eval": int(len(y)), "labels": "1 per class",
                   "trials": trials, "k": k, "alpha": M.ALPHA, "iters": M.ITERS},
        "rows": rows,
        "full_label_linear_upper_bound": upper,
        "n_labels_full": int(0.7 * len(y)),
    }
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    with open(os.path.join(_HERE, "results", "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nfull-label linear upper bound: {upper*100:.1f}%  "
          f"(~{out['n_labels_full']} labels)")
    print("wrote results/results.json")
    return out


if __name__ == "__main__":
    tr = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"=== few-label-manifold: headline table ({tr} trials) ===")
    run(trials=tr)
