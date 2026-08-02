# Phase 0 — LanCE baseline reproduction (CUB / CUB-Painting)

**Status: PASS.** This gates every later phase — if this hadn't matched, nothing built on top of it would be trustworthy.

## Setup

- Model: `clip_cbm` (CLIP-CBM), CLIP ViT-L/14, human-style concept bank (`cub_concepts.txt`, 311 concepts)
- Dataset: CUB-200-2011 (source/train, official Caltech release) → CUB-200-Painting (target/OOD test, Google Drive release linked from LanCE's README)
- 50 epochs, batch size 64, Adam, lr 1e-4 — same hyperparameters as `joeyz0z/LanCE`'s README-documented training command
- Two runs: `--alpha 0` (baseline, no DDO) and `--alpha 1` (+DDO)

## Result vs. paper's Table 1 (CLIP-CBM/human row for CUB-Painting)

| | Baseline (α=0) | +DDO (α=1) |
|---|---|---|
| Paper (Table 1) | 50.54% | 55.53% |
| Our reproduction | **50.64%** | **57.04%** |
| Delta | +0.10 | +1.51 |

Both land within the plan's ~3–5 point pass bar — baseline essentially exact, +DDO slightly exceeding the paper's reported gain (+6.40 points in our run vs. +4.99 in theirs, same direction and comparable magnitude). Full epoch-by-epoch logs: `external/LanCE/logs_baseline_run.log`, `external/LanCE/logs_ddo_run.log`.

## Bugs found and fixed in `joeyz0z/LanCE`'s released code to get here

None of these touch the method itself (DDO loss, model architecture) — all are plumbing bugs that silently blocked the README's own documented commands from running:

1. **`data/__init__.py`** — missing `import os` despite using `os.path.join` throughout.
2. **`data/__init__.py`** — stray leading `/` in the CUB-Painting path join (`"/CUB/CUB-200-Painting/images"`), which breaks `os.path.join`'s path resolution. Also, the actual downloaded archive has no `images/` subdirectory (class folders sit directly under `CUB-200-Painting/`), so the path segment needed correcting, not just de-slashing.
3. **`args.py`** — `--batch_size` was typed `str` and `--epochs` was typed `float`; both broke as soon as you pass either flag on the command line (`range()` needs an int, `DataLoader` rejects a non-int batch size).
4. **`data/CUB/cub_data.py`** — `attr_label` was hardcoded to a dummy `torch.tensor([0]*77)` in both `Processed_CUB_Dataset` and `Processed_CUBP_Dataset`, but the actual concept bank has 311 concepts. Irrelevant to training outcome (`--beta`, the concept-supervision weight, defaults to 0), but `BCEWithLogitsLoss` still requires matching shapes to even compute the (discarded) loss value, so training crashed before a single step. Fixed to size dynamically off `len(self.concept2id)`.
5. **`data/__init__.py`** — the CUB branch passed `train_dataset.classname2id` for *both* the `classname2id` and `concept2id` arguments when constructing the target/painting test set, instead of the actual `train_dataset.concept2id`. Copy-paste bug; caused the same shape-mismatch crash during evaluation.

## Known slowdown (not a bug, just infrastructure)

Their code recomputes CLIP image embeddings from scratch every epoch, even though CLIP is frozen and the embeddings never change. This makes each epoch far slower than necessary (~7–14 minutes on an RTX 5060 8GB for ~6k CUB training images). Planned fix before Phase A/B/C: cache embeddings once, train the (tiny) linear classifier on cached features.
