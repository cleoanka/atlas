# few-label-manifold

**With a good enough representation, one label per class recovers ~95% of what 7000 labels buy — and with a bad representation the same trick does *worse* than nothing.**

*(Türkçe: [README.tr.md](README.tr.md))*

<p align="center">
  <img src="figures/fig7_headline.png" width="560"><br>
  <img src="figures/fig2_diffusion.gif" width="460">
</p>

The hard part of classification is not the classifier. It is the **metric** — how you decide which points are "near" each other. Fix the metric and labels become almost free: you are no longer *teaching* the machine what a digit is, you are just *naming* clusters it already found.

---

## The idea

**Why Euclidean distance fails.** On real data, classes live on curved, tangled manifolds. Two points can be close in straight-line (pixel) distance while belonging to different classes, because the straight line cuts across the gap between two folds of the manifold. A nearest-neighbour rule in that metric is fooled everywhere the folds come close.

<p align="center"><img src="figures/fig1_why_euclid_fails.png" width="720"></p>

**What the graph does.** Build a k-nearest-neighbour graph and let a label *flow along the edges* instead of jumping in a straight line. Now "near" means *reachable along the data*, so a label spreads within a fold and stops at the gap. Starting from a single labelled point per class, this graph diffusion paints almost the whole dataset — **if** the graph's edges mostly connect same-class points. The one number that predicts success is **edge purity**: the fraction of edges that stay within a class.

<p align="center"><img src="figures/fig2_diffusion_final.png" width="460"></p>

**Why a label only *names*.** Once the representation has already grouped the data, a label does not carve out a class — it just attaches a human name to a group the geometry discovered on its own. So the number of labels you need scales with the number of **modes** (distinct blobs) in the data, not with the number of classes.

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
