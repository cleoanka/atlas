<p align="center"><img src="figures/banner.png" width="100%"></p>

# atlas

> **few-label learning on a manifold** — *(Türkçe: [README.tr.md](README.tr.md))*

**One label per class classifies MNIST at 94.7%.** Not because the labels are clever — because the label was never the hard part. Give a good representation to a plain graph-diffusion rule and ten labels do the work of seven thousand. Swap in a bad representation and the *exact same rule* collapses to 71%.

<p align="center"><img src="figures/fig8_representation.png" width="70%"></p>

The whole 23-point spread is the **metric** — how you decide which points are "near". The label only puts a name on a cluster the geometry already found. This repo makes that precise, measurable, and shows exactly where it breaks.

<p align="center"><img src="figures/fig2_diffusion.gif" width="460"></p>

---

## Three claims, each measured

1. **The metric is the whole game.** Same 10 labels, same algorithm: raw pixels → 71%, a learned contrastive metric → **94.7%** (fully-supervised ceiling: 98.6%). The classifier didn't change; the geometry did.
2. **Labels count modes, not classes.** Once the geometry has grouped the data, a label is just a *name* for a group. You need roughly one label per **mode**, and a class is a union of modes — so the label budget tracks the data's shape, not its number of categories.
3. **It's a phase transition, not a trade-off.** There is a sharp edge-purity threshold. Above it, one label floods its cluster correctly. Below it, propagating labels is *actively worse than doing nothing*. This is the load-bearing fact, not a footnote.

---

## Why it works

**Euclidean distance lies on a manifold.** Two digits can be close in pixel space yet belong to different classes, because the straight line between them cuts across the gap between two folds. Nearest-neighbour in that metric is wrong wherever the folds come close.

<p align="center"><img src="figures/fig1_why_euclid_fails.png" width="80%"></p>

**A graph fixes the metric.** Connect each point to its k nearest neighbours and let a label *flow along the edges* instead of jumping in a straight line. Now "near" means *reachable along the data*. From one seed per class, that flow paints the whole dataset — as long as edges mostly connect same-class points. The single number that decides everything is **edge purity**.

---

## The mechanism — the math

**The graph.** Embed each of the $n$ points with a representation $\phi$, join every point to its $k$ nearest neighbours, weight each edge by a Gaussian kernel, symmetrise ($W=W^{\top}$), and normalise with the degree matrix $D=\operatorname{diag}(\sum_j W_{ij})$:

$$W_{ij} = \exp\!\left(-\frac{\lVert \phi_i-\phi_j\rVert^2}{2\sigma^2}\right),\qquad S = D^{-1/2}\,W\,D^{-1/2}.$$

**Label spreading.** Put the few labels in a one-hot seed matrix $Y$ (nonzero only on seeds) and iterate to a fixed point:

$$F^{(t+1)} = \alpha\,S\,F^{(t)} + (1-\alpha)\,Y \;\;\xrightarrow{\;\alpha<1\;}\;\; F^{\star} = (1-\alpha)\,(I-\alpha S)^{-1}\,Y = (1-\alpha)\sum_{t=0}^{\infty}(\alpha S)^{t}\,Y,$$

then predict $\hat y_i=\arg\max_c F^{\star}_{ic}$. That series is the point: $(\alpha S)^{t}$ is the length-$t$ walk operator, so a point's label is the **discounted sum of label mass reaching it over walks of every length** — the label flows along edges, short walks weighted most. Equivalently $F^\star$ is the unique minimiser of

$$\tfrac12\sum_{i,j} W_{ij}\left\lVert \tfrac{F_i}{\sqrt{D_{ii}}}-\tfrac{F_j}{\sqrt{D_{jj}}}\right\rVert^2 \;+\; \tfrac{1-\alpha}{\alpha}\sum_i \lVert F_i-Y_i\rVert^2,$$

labels that vary slowly across strong edges while staying near the seeds.

**Why purity is the whole game.** Split $S=S_{\text{in}}+S_{\text{cross}}$ into within-class and cross-class edges. The walk series carries a seed's label correctly through $S_{\text{in}}$; every term touching $S_{\text{cross}}$ is a channel leaking the *wrong* label across a boundary. With edge purity $\pi=\tfrac{\text{within-class edge weight}}{\text{total edge weight}}$: as $\pi\to1$, $S$ is block-diagonal by class and diffusion stays home; as $\pi$ drops, cross-class conductance grows until, past a threshold, the wrong-label mass at a node overtakes the right and the $\arg\max$ **flips**. A cliff, not a slope.

**The representation that lifts $\pi$.** A contrastive encoder, trained with **no labels**, pulls two augmentations of an image together and pushes everything else apart (NT-Xent, temperature $\tau$):

$$\mathcal{L}_i = -\log\frac{\exp(\langle z_i,z_i^{+}\rangle/\tau)}{\sum_{j\ne i}\exp(\langle z_i,z_j\rangle/\tau)}.$$

This bends the metric so same-class images become neighbours; edge purity jumps from 88% (raw) to 97% (contrastive) — exactly what the transition rewards.

---

## The catch that makes it real: a phase transition

Degrade the metric and edge purity falls. Graph diffusion tracks it **steeply**; naive 1-NN barely moves. Below a crossover (~65% purity on MNIST) the "clever" method is *worse than the baseline* — a single heavy "bridge" edge is enough to flood a whole region with the wrong label.

<p align="center">
  <img src="figures/fig4_phase_transition.png" width="49%">
  <img src="figures/fig3_bridge_edge.png" width="49%">
</p>

This is the useful part of the result: it tells you *when not to bother*. If your representation gives a low-purity graph, label propagation will cost you accuracy, not save you labels.

---

## How many labels does a dataset actually need?

Cluster the representation into $N$ groups and name each with one label. Accuracy climbs toward the per-cluster majority ceiling as $N$ grows — you buy accuracy with *names*, and the currency is **modes, not classes**. Ten labels (one per class) is just the coarsest point on this curve.

<p align="center"><img src="figures/fig5_label_budget.png" width="64%"></p>

---

## Ten labels, drawn on the map

The contrastive embedding, coloured by truth (left) and by the label diffused from just 10 seeds (right, black stars; errors in red). Ten names, the whole map.

<p align="center"><img src="figures/fig6_embedding.png" width="88%"></p>

---

## Reproduce

```bash
pip install -r requirements.txt
python src/evaluate.py          # the table          -> results/results.json
python src/sweep.py             # purity + budget    -> results/sweep.json
python figures/make_figures.py  # every figure incl. the cover GIF
```

The contrastive embedding and eval split ship in the repo, so every number reproduces **without a GPU** (MNIST downloads itself on first use). Retrain the encoder with `pip install torch && python src/train_contrastive.py 40` (MPS / CUDA / CPU auto). A 5-cell tour is in [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

**Numbers** (MNIST, 10k eval, 1 label/class, 60 draws, k=10) — `results/*.json` is the source of truth:

| representation | edge purity | Euclid 1-NN | graph diffusion |
|---|--:|--:|--:|
| raw pixels | 88.4% | 42.3% | 71.4% |
| PCA-50 | 89.9% | 43.6% | 74.6% |
| diffusion map | 91.3% | 37.6% | 72.8% |
| **contrastive** | **96.6%** | **62.9%** | **94.7%** |

---

## What this is

A clean, honest **measurement** of a classic primitive, not a new algorithm: label spreading is Zhou et al. (2004), diffusion maps are Coifman & Lafon (2006). What atlas adds is the anatomy on a legible dataset — the edge-purity phase transition, the representation-is-everything gap, and the modes-not-classes label law — each reproduced from one command. MIT licensed.
