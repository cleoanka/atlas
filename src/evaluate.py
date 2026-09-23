"""Produce the headline table: purity + 1-label-per-class accuracy for every representation.

    python src/evaluate.py                        # MNIST      -> results/results.json
    python src/evaluate.py --dataset imagenette   # ImageNette -> results/imagenette_results.json

The single claim, made measurable: with a good enough metric, ONE label per class recovers
most of a fully-supervised model — and with a bad metric the same graph method does WORSE
than doing nothing clever (Euclid-1NN). Both halves are in this table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "src"))

import metrics as M            # noqa: E402
import representations as R    # noqa: E402


def _outfile(dataset):
    name = "results.json" if dataset == "mnist" else f"{dataset}_results.json"
    return os.path.join(_HERE, "results", name)


def run(dataset: str = "mnist", trials: int = 60, k: int = M.K_DEFAULT):
    rows = []
    upper = None
    for name, (_, label) in R.REGISTRY.items():
        t0 = time.time()
        X, y = R.get(name, dataset)
        W = M.knn_graph(X, k)
        res = M.one_label_per_class(X, W, y, trials=trials)
        res["name"] = name; res["label"] = label
        rows.append(res)
        if name == "contrastive":
            upper = M.linear_upper_bound(X, y)
        print(f"  {label:<20} purity={res['purity']*100:5.1f}%  "
              f"euclid={res['euclid_1nn']*100:5.1f}%  "
              f"diffusion={res['diffusion']*100:5.1f}% (±{res['diffusion_std']*100:.1f})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    out = {
        "config": {"dataset": dataset, "n_eval": int(len(y)),
                   "n_classes": int(len(set(y.tolist()))), "labels": "1 per class",
                   "trials": trials, "k": k, "alpha": M.ALPHA, "iters": M.ITERS},
        "rows": rows,
        "full_label_linear_upper_bound": upper,
        "n_labels_full": int(0.7 * len(y)),
    }
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    with open(_outfile(dataset), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nfull-label linear upper bound: {upper*100:.1f}%  (~{out['n_labels_full']} labels)")
    print(f"wrote {os.path.relpath(_outfile(dataset), _HERE)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=R.DATASETS)
    ap.add_argument("--trials", type=int, default=60)
    a = ap.parse_args()
    print(f"=== atlas: headline table · {a.dataset} ({a.trials} trials) ===")
    run(dataset=a.dataset, trials=a.trials)
