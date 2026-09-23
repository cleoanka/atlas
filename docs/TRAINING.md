# Training guide — Mac (MPS) & NVIDIA (CUDA)

atlas ships the trained embeddings, so **you never need to train to reproduce the results**.
Train only if you want to change the encoder (longer, bigger, higher-res) or reproduce it
yourself. Training is fully unsupervised (SimCLR / NT-Xent) — no labels.

What a run produces (in `data/`, all gitignored except the shipped `*_emb.npz`):
- `imagenette_emb.npz` — L2-normalised embedding of the 3925 val images (what evaluation uses).
- `imagenette_train_emb.npz` — embedding of the 9469 train images (for the honest linear-probe ceiling).
- `models/imagenette_encoder.pt` — the encoder weights (`{encoder, head, arch, img, dim}`), kept locally.

The device is auto-detected: **MPS → CUDA → CPU**. Mixed precision (AMP) turns on automatically on CUDA.

---

## Faster vs. better — read this first

- **A faster GPU does not make the model more accurate by itself.** The same config gives ~the same
  accuracy on any device.
- **What a fast GPU buys you is a *heavier config*** — more epochs, larger batch (SimCLR loves large
  batches: more in-batch negatives), higher resolution, a bigger backbone (ResNet-50). *Those* raise
  accuracy. On slow/throttled MPS a heavy config is simply too slow to run.
- So on the RTX box the move is not "same run, faster" — it's "run the config MPS couldn't afford."

---

## Mac (Apple Silicon / MPS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch          # torch ships with MPS support on Apple Silicon
python -c "import torch; print('mps:', torch.backends.mps.is_available())"   # -> True

# ImageNette (downloads ~95 MB on first run, then caches)
python src/train_contrastive_imagenette.py --epochs 300 --img 160 --batch 512
# MNIST
python src/train_contrastive.py 40
```

Notes:
- **24 GB unified memory** lets you use a large batch (512 @ 160px fits). This is MPS's one advantage
  over an 8 GB laptop GPU.
- MPS is **much slower than CUDA for training** and has no AMP here, so expect ~25–90 s/epoch, and far
  more in **Low-Power Mode** (it throttles clocks hard — turn it off for training speed).
- MPS is fine for a proof run; use CUDA for the heavy config.

---

## NVIDIA (CUDA) — including RTX 50-series (Blackwell)

### 1. Install PyTorch with the right CUDA

RTX 50-series (5070/5080/5090, compute capability **sm_120 / Blackwell**) needs a **CUDA 12.8**
build — older wheels won't run on Blackwell.

```bash
python -m venv .venv && source .venv/bin/activate     # (Linux/WSL2)  or  py -m venv .venv (Windows)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
# if cu128 stable isn't available for your Python yet, use nightly:
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify the GPU is seen:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expect: ... True   NVIDIA GeForce RTX 5070 Laptop GPU   (12, 0)
```
If `is_available()` is `False` or you see an `sm_120 not compatible` warning, your torch build is too
old — reinstall with the cu128 (or nightly cu128) index above.

### 2. Train

```bash
python src/train_contrastive_imagenette.py --epochs 800 --img 160 --batch 512 --backbone resnet18
```
AMP (mixed precision) is enabled automatically on CUDA — roughly **2× faster and ~half the VRAM**.

### 3. VRAM guidance — the RTX 5070 Laptop has only **8 GB**

8 GB is *less* than the Mac's 24 GB, so the Mac's `batch 512 @ 160px` may OOM here. With AMP on,
sensible starting points (drop batch if you hit `CUDA out of memory`):

| backbone | img | batch (8 GB, AMP) | note |
|---|--:|--:|---|
| resnet18 | 128 | 512 | fast, safe |
| resnet18 | 160 | 256–384 | good default for the RTX 5070 mobile |
| resnet50 | 128 | 192–256 | more capacity → higher ceiling |
| resnet50 | 224 | 96–128 | best quality; watch VRAM |

Bigger batch = more SimCLR negatives = better representation, so **use the largest batch that fits**.
On CUDA even the heavy configs finish in tens of minutes, not hours.

**Recommended "best result" run on the RTX 5070 (8 GB):**
```bash
python src/train_contrastive_imagenette.py --backbone resnet50 --img 160 --batch 192 --epochs 800
```
Expect a higher linear-probe ceiling and a higher few-label number than the shipped 100-epoch
ResNet-18 baseline (which is intentionally modest — see the README's honest-scope note).

### 4. Windows notes
- **WSL2 (Ubuntu) is the smoothest path**: install the NVIDIA Windows driver, then inside WSL2
  `pip install torch --index-url .../cu128`. `nvidia-smi` should list the 5070 inside WSL2.
- Native Windows works too (`py -m venv`, same pip). Avoid the CPU-only default wheel — always pass
  the `cu128` index.

---

## After training

Re-run the evaluation and figures against the freshly written embedding:
```bash
python src/evaluate.py --dataset imagenette          # -> results/imagenette_results.json
python src/sweep.py    --dataset imagenette           # -> results/imagenette_sweep.json
python figures/make_imagenette_figures.py             # -> figures/imagenette_*.png
```
Everything downstream is representation-agnostic, so a better encoder flows straight into the table,
the phase-transition curve, and the embedding map with no other changes.

---

## Pushing further (retrain levers, best on a fast GPU)

The shipped encoders are deliberately modest. To raise the ceiling (and with it the few-label
number), these are the levers, in rough order of bang-for-buck — all need retraining, so run them
on the CUDA box:

- **Bigger batch** (`--batch`) — the cheapest real win for SimCLR: more in-batch negatives. Use the
  largest that fits VRAM (with AMP on CUDA).
- **Lower temperature** (`--tau 0.1`, default 0.2) — sharper separation of hard negatives; pairs
  well with a cosine-decay LR. Already wired: `--tau`.
- **Bigger backbone** (`--backbone resnet50`) — more capacity → higher ceiling. Already wired.
- **Longer** (`--epochs 800`) — SimCLR keeps improving for many hundreds of epochs; cheap on CUDA.
- **Non-linear projection head** — *already implemented*: a 2-layer MLP head is trained and then
  discarded; evaluation uses the raw backbone output. (This is worth ~+10 pts and is on by default.)

Two heavier upgrades not yet in the code (documented so you can add them on the GPU box):

- **Multi-crop (SwAV):** instead of 2 global crops, use 2 global + 4–6 small local crops and match
  local↔global. Strong gains from forcing part↔whole consistency; ~1.5–2× the forward cost.
- **Momentum encoder + memory bank (MoCo):** a second, momentum-updated key encoder feeding a queue
  of tens of thousands of negatives — decouples "number of negatives" from batch size, ideal for an
  8 GB GPU. A ~150-line addition (key encoder, EMA update, queue); best implemented and tested
  directly on the CUDA box.

After any retrain, re-run `evaluate.py --dataset imagenette` + `make_imagenette_figures.py`; the
numbers flow straight through. And note the separate finding in the README's "Squeezing more"
section: once the encoder is good, **choosing the most-typical point to label** (`src/enhance.py`,
medoid/density seed) buys more than any metric tweak.
