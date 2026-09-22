<p align="center"><img src="figures/banner.png" width="100%"></p>

# few-label-manifold

**With a good enough representation, one label per class recovers ~95% of what 7000 labels buy — and with a bad representation the same trick does *worse* than nothing.**

*(Türkçe: [README.tr.md](README.tr.md))*

<p align="center"><img src="figures/fig2_diffusion.gif" width="480"></p>

The hard part of classification is not the classifier. It is the **metric** — how you decide which points are "near" each other. Fix the metric and labels become almost free: you are no longer *teaching* the machine what a digit is, you are just *naming* clusters it already found.

---

## The idea

**Why Euclidean distance fails.** On real data, classes live on curved, tangled manifolds. Two points can be close in straight-line (pixel) distance while belonging to different classes, because the straight line cuts across the gap between two folds of the manifold. A nearest-neighbour rule in that metric is fooled everywhere the folds come close.

<p align="center"><img src="figures/fig1_why_euclid_fails.png" width="720"></p>

**What the graph does.** Build a k-nearest-neighbour graph and let a label *flow along the edges* instead of jumping in a straight line. Now "near" means *reachable along the data*, so a label spreads within a fold and stops at the gap. Starting from a single labelled point per class, this graph diffusion paints almost the whole dataset — **if** the graph's edges mostly connect same-class points. The one number that predicts success is **edge purity**: the fraction of edges that stay within a class.

<p align="center"><img src="figures/fig2_diffusion_final.png" width="460"></p>

**Why a label only *names*.** Once the representation has already grouped the data, a label does not carve out a class — it just attaches a human name to a group the geometry discovered on its own. So the number of labels you need scales with the number of **modes** (distinct blobs) in the data, not with the number of classes.

---

## How it works — the math

**The graph.** Embed each of the $n$ points with a representation $\phi$, connect every point to its $k$ nearest neighbours, and weight each edge with a Gaussian kernel

$$W_{ij} = \exp\!\left(-\frac{\lVert \phi_i-\phi_j\rVert^2}{2\sigma^2}\right),\qquad \sigma=\text{median edge distance},$$

symmetrised so $W=W^{\top}$. With the degree matrix $D=\operatorname{diag}(\sum_j W_{ij})$, form the symmetric normalized operator

$$S = D^{-1/2}\,W\,D^{-1/2}.$$

**Label spreading.** Put the (very few) labels in a one-hot seed matrix $Y\in\{0,1\}^{n\times C}$ — nonzero only on the handful of labelled seeds. Iterate

$$F^{(t+1)} = \alpha\,S\,F^{(t)} + (1-\alpha)\,Y,\qquad \alpha\in(0,1),$$

and predict $\hat y_i=\arg\max_c F_{ic}$. Since $\lVert \alpha S\rVert_2=\alpha<1$ the map is a contraction and converges to the closed form

$$F^{\star} = (1-\alpha)\,(I-\alpha S)^{-1}\,Y = (1-\alpha)\sum_{t=0}^{\infty}(\alpha S)^{t}\,Y.$$

**Why this is diffusion.** That series *is* the idea: $(\alpha S)^{t}$ is the length-$t$ walk operator on the graph, discounted by $\alpha^{t}$. So a point's label is the **discounted sum of label mass reaching it over graph walks of every length**, short walks weighted most — the label flows *along edges*, and "near" means *reachable along the data*, not straight-line. Equivalently $F^{\star}$ minimises

$$\tfrac12\sum_{i,j} W_{ij}\left\lVert \tfrac{F_i}{\sqrt{D_{ii}}}-\tfrac{F_j}{\sqrt{D_{jj}}}\right\rVert^2 \;+\; \tfrac{1-\alpha}{\alpha}\sum_i \lVert F_i-Y_i\rVert^2,$$

i.e. *labels vary slowly across strong edges* while staying near the seeds.

**Why edge purity is the whole game.** Split $S=S_{\text{in}}+S_{\text{cross}}$ into within-class and between-class edges. The walk series carries a seed's label correctly through $S_{\text{in}}$; every term involving $S_{\text{cross}}$ is a channel that leaks the *wrong* label across a boundary. Define **edge purity**

$$\pi=\frac{\text{within-class edge weight}}{\text{total edge weight}}.$$

As $\pi\to1$, $S$ is nearly block-diagonal by class and diffusion stays in-class. As $\pi$ falls, cross-class conductance grows until — past a threshold — the wrong-label mass reaching a node exceeds the right-label mass and the $\arg\max$ **flips**. That is why the accuracy curve is a *phase transition*, not a gentle slope (fig4), and why a single heavy bridge edge can flip a whole region (fig3).

**The naive baseline.** Euclidean 1-NN sets $\hat y_i = y_{\,\text{nearest seed to }i}$ in the ambient metric: it consults a *single* point and trusts the raw distance, so it breaks wherever the manifold folds (fig1). Diffusion integrates over the entire graph instead — a strength when $\pi$ is high, a liability when it is low. Hence the crossover.

**The representation that makes $\pi$ high.** A contrastive encoder is trained with **no labels** to pull two augmentations of the same image together and push everything else apart (NT-Xent, temperature $\tau$):

$$\mathcal{L}_i = -\log\frac{\exp(\langle z_i,z_i^{+}\rangle/\tau)}{\sum_{j\ne i}\exp(\langle z_i,z_j\rangle/\tau)}.$$

This bends the metric so same-class images become neighbours, and edge purity jumps (raw pixels $88\%\to$ contrastive $97\%$) — exactly what the phase transition rewards.

**Why labels scale with modes, not classes.** Diffusion colours each pure connected component with whichever seed lands in it; a component with **no** seed is undetermined. If the data is a mixture of $M$ well-separated modes, you need $\gtrsim 1$ seed per mode to colour everything. Since a class is a *union* of modes, the label budget scales with $M\ge C$, not with the number of classes $C$ (fig5).

> **Diffusion maps** (one of the table rows) use the *same* operator differently: take the top eigenvectors of $S$, scale them by their eigenvalues, and use those as coordinates. Euclidean distance there equals *diffusion distance* on the graph — a representation, not a classifier.

---

## Results

MNIST, 10 000 evaluation digits, **one label per class**, 60 random draws, k = 10.

| representation | edge purity | Euclid 1-NN | graph diffusion |
|---|--:|--:|--:|
| raw pixels | 88.4% | 42.3% | 71.4% |
| PCA-50 | 89.9% | 43.6% | 74.6% |
| diffusion map | 91.3% | 37.6% | 72.8% |
| **contrastive (40 ep)** | **96.6%** | **62.9%** | **94.7%** |

Fully-supervised linear ceiling (≈7000 labels): **98.6%**.
So **10 labels → 94.7%**, **7000 labels → 98.6%**: 700× fewer labels, 3.9 points lower.

Two things jump out. Diffusion beats naive nearest-neighbour by a huge margin *only* on a high-purity graph (contrastive: 94.7% vs 62.9%). And the representation, not the label budget, is what moves the needle.

### The phase transition

Degrade the metric (add noise to PCA-50) and edge purity falls. Graph diffusion tracks purity *steeply*; Euclid-1NN barely moves. Below a crossover (~65% purity here) the "clever" method does **worse** than the naive baseline.

<p align="center"><img src="figures/fig4_phase_transition.png" width="640"></p>

### Labels scale with modes, not classes

Cluster PCA-50 into *N* groups and give each **one** label. Accuracy climbs toward the per-cluster majority ceiling as *N* grows — you are buying accuracy with *names*, and the currency is modes, not classes.

<p align="center"><img src="figures/fig5_label_budget.png" width="640"></p>

### Ten labels, drawn on the map

Left: the contrastive embedding coloured by true label. Right: coloured by the label diffused from just **10 seeds** (black stars); errors in red.

<p align="center"><img src="figures/fig6_embedding.png" width="760"></p>

---

## When it does **not** work

This is the important half, so it is not buried:

- **Below the purity threshold, the manifold method actively hurts.** One bad ("bridge") edge lets a label flood across a class boundary; enough of them and diffusion is *worse* than ignoring the graph (see the phase-transition figure, and `figures/fig3_bridge_edge.png`: one edge takes a toy problem from 100% to 82%). If your representation gives a low-purity graph, do not reach for graph diffusion.
- **It is transductive.** The label spreads over a *fixed* set of unlabelled points you have in hand. Classifying a genuinely new point means rebuilding/extending the graph — this is not a trained model you can ship and call on one sample.
- **Labels scale with modes, not classes.** If a class is multi-modal (many distinct sub-clusters), one label per *class* misses most modes. You need roughly one label per *mode*, and finding the modes is the clustering problem again.
- **The reproduced numbers are from this pipeline, not a leaderboard.** They depend on the seed, k, and the exact graph. One row differs from the original cloud run: the *diffusion-map* Euclid-1NN came out at **37.6%** here vs. 46.6% previously (the diffusion-map embedding's scaling changes local 1-NN a lot). Everything is measured by the scripts below; treat `results/*.json` as the source of truth, not the prose.

<p align="center"><img src="figures/fig3_bridge_edge.png" width="720"></p>

---

## Run it

```bash
pip install -r requirements.txt
python src/evaluate.py          # headline table  -> results/results.json
python src/sweep.py             # purity sweep + label budget -> results/sweep.json
python figures/make_figures.py  # regenerate every figure (incl. the cover GIF)
```

The contrastive embedding (`contrastive_emb.npz`) and the evaluation split are shipped, so the headline reproduces **without a GPU**. MNIST itself (21 MB) is downloaded on first use. To retrain the encoder from scratch (Apple-Silicon / CUDA / CPU auto-detected):

```bash
pip install torch
python src/train_contrastive.py 40
```

A 5-cell walkthrough is in [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

---

## Related work

- **Zhu, Ghahramani & Lafferty (2003)** — semi-supervised learning with Gaussian fields / harmonic functions. The label-propagation objective this repo diffuses; here it is a demo of *when* it works, framed around edge purity.
- **Zhou, Bousquet, Lal, Weston & Schölkopf (2004)** — "Learning with local and global consistency" (label spreading). The exact normalized-Laplacian iteration used in `src/metrics.py`; this repo stresses it to its failure point rather than proposing it.
- **Coifman & Lafon (2006)** — diffusion maps. Used here as one *representation* (the "diffusion map" table row), not as the classifier.
- **Douze, Szlam, Hariharan & Jégou (2018)** — large-scale label propagation for low-shot learning. Same idea at ImageNet scale with learned features; this repo isolates the single-label-per-class regime and the purity phase transition on a small, legible dataset.
- **Google Expander** — production label propagation on massive graphs. Industrial engineering of the same primitive; this repo is the 200-line pedagogical opposite, focused on the boundary of the idea.

## License

MIT — see [LICENSE](LICENSE).
