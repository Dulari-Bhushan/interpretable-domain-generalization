# Component 2 — self-diagnosing domain grounding

**Status: ⚠️ Done — partial/mixed result.** The diagnostic half works exactly as designed: run on a cheap 20-images/class probe, it correctly and confidently flags the photo→Midjourney v6 domain shift as untrustworthy (mean alignment 0.0123, far below the calibrated threshold 0.1431). The fallback half — replacing the text-only `domain_diffs` with a single image-measured direction from that same probe — does **not** recover DDO's lost benefit. It does worse than both the (known-weak) text-only DDO and the plain baseline.

## One-line summary

The self-diagnosis correctly detects when DDO's text-only domain guess can't be trusted, but the specific "swap in one real-image-measured direction" fallback tested here makes DDO's target accuracy *worse* than doing nothing (best target 73.81% vs. baseline's 74.27%), not better — a real, honest dead end for this exact design, not for the diagnostic capability itself.

## Origin

`planning/04-component2-self-diagnosing-domain-grounding-plan.md`, following `docs/new_methodology_report.md` §1's Component 2 row.

## The issue this targets

DDO's orthogonality regularizer (`clip_cbm_orth`) is built entirely from text — `domain_diffs[i] = normalize(text_embedding(target_template) − text_embedding(source_template))` — and never checks whether that text-predicted domain shift matches the real one. Phase F1/F3/F4 measured this guess to be unreliable outside CLIP's comfort zone (alignment 0.04–0.32 against the paper's own claimed 0.90–0.99), and Phase E2 measured the downstream cost directly: DDO's benefit over a plain baseline collapses from +6.40 points (CUB→Painting, good guess) to +0.68 points (photo→Midjourney v6, bad guess) under an identical training protocol. Component 2 asks whether the model can catch this itself, cheaply, and do something about it.

## Why we tried this approach specifically

The alignment-score methodology was already validated three times (F1/F3/F4) — reusing it as a live diagnostic, rather than inventing a new trust metric, keeps the diagnostic claim tightly scoped to something already shown to work. The fallback direction (image-measured `domain_diffs`) was chosen because it's architecturally free: `clip_cbm_orth`'s regularizer only ever consumes a `(num_directions, num_classes, feat_dim)` tensor and never inspects where it came from — a single image-measured direction per class is a drop-in replacement needing zero model changes.

## Method

1. **Diagnostic** (`external/LanCE/model/domain_grounding.py::compute_alignment_score`): for a small probe of real images per class from source and target domains, `visual_shift = normalize(mean_img_emb(target) − mean_img_emb(source))`; `textual_shift = normalize(text_emb(target_template) − text_emb(source_template))`; `alignment = cosine(visual_shift, textual_shift)`, averaged over classes. Identical formula to Phase F3 (the cleanest of the three F1/F3/F4 variants — real matched photos on both sides, no text-as-photo-proxy confound), generalized to run at probe scale (20/class here vs. F3's 1,500/source).
2. **Threshold calibration** (`experiments/component2_alignment_calibration.py`): rather than trust the paper's claimed 0.90–0.99, we measured our own diagnostic on a domain shift already known trustworthy — PACS's photo→art_painting/cartoon/sketch, the domains Component 1's own core result already confirms sit inside CLIP's comfort zone. Threshold = midpoint of PACS's measured mean and Phase F3's cited photo→Midjourney number.
3. **Fallback** (`build_image_grounded_domain_diffs`): when alignment < threshold, replace the 204-direction text-only `domain_diffs` with a single direction per class — `normalize(mean(target probe embeddings) − mean(source probe embeddings))` — built from the same probe images already used for diagnosis, shape `(1, num_classes, feat_dim)`.
4. **Validation** (`experiments/component2_defactify_grounding_ddo.py`): rerun Phase E2's exact baseline-vs-+DDO protocol (50 epochs, batch 64, AdamW, lr 1e-4, weight_decay 1e-4, `clip_cbm_orth`, ViT-L/14) on Defactify (photo→Midjourney v6), adding a third condition — +DDO with the self-diagnosed, image-grounded `domain_diffs` — so all three conditions are trained and evaluated under one identical split rather than reusing Phase E2's saved numbers against a different target-test set (the probe had to be held out from `target_test`, shrinking it slightly from Phase E2's).

## Dataset(s) used, and why

- **PACS** (photo→art_painting/cartoon/sketch, train split): calibration only, no training — the trust-positive anchor, chosen because it's already established (Component 1) to be a domain CLIP handles well.
- **Defactify/MS-COCO-AI** (photo→Midjourney v6): the exact domain shift Phase F3 (alignment) and Phase E2 (trained-model cost) already measured, so Component 2's result connects directly to both without introducing a new unknown into the comparison. Downloaded directly on the server for this run (not previously transferred); required installing the `datasets` package into the server's `mlgpu` conda env.

## Code

- [`external/LanCE/model/domain_grounding.py`](../external/LanCE/model/domain_grounding.py) — diagnostic + fallback.
- [`external/LanCE/experiments/component2_alignment_calibration.py`](../external/LanCE/experiments/component2_alignment_calibration.py) — PACS threshold calibration.
- [`external/LanCE/experiments/component2_defactify_grounding_ddo.py`](../external/LanCE/experiments/component2_defactify_grounding_ddo.py) — main 3-condition validation.
- [`results/component2_alignment_calibration.json`](component2_alignment_calibration.json), [`results/component2_defactify_grounding_ddo.json`](component2_defactify_grounding_ddo.json) — raw results.

## Results

### Calibration ([`component2_alignment_calibration.json`](component2_alignment_calibration.json))

| Domain shift | Mean alignment (this project's diagnostic, real images both sides) |
|---|---|
| photo → art_painting (PACS) | 0.326 |
| photo → cartoon (PACS) | 0.222 |
| photo → sketch (PACS) | 0.200 |
| **PACS overall mean** | **0.249** |
| photo → Midjourney v6 (Defactify, cited from Phase F3, same formula) | 0.037 |
| **Calibrated threshold (midpoint)** | **0.143** |

**An honest finding worth stating plainly, not burying:** none of these — including PACS's own "trustworthy" domains — come anywhere near the paper's claimed 0.90–0.99 alignment range under this real-matched-photo formula. The paper's Fig. 2 number apparently doesn't replicate at the per-class, small-probe scale this diagnostic uses. What *does* hold up is the **relative** separation: PACS domains score 5.4–8.8x higher than the known-bad Defactify shift, which is what a calibrated (not absolute) threshold actually needs.

### Main experiment ([`component2_defactify_grounding_ddo.json`](component2_defactify_grounding_ddo.json))

**Diagnosis on the Defactify probe (20 images/class, 460 total):** mean alignment **0.0123**, threshold 0.1431 → correctly flagged **untrustworthy**, triggering the image-grounded fallback. (Consistent with, and even lower than, Phase F3's larger-sample 0.037 — the probe-scale measurement agrees in direction and margin.)

| Condition | Best target (Midjourney) acc | Best source (photo) acc | Gain over baseline |
|---|---|---|---|
| Baseline (α=0) | 74.27% | 73.00% | — |
| +DDO, text-only `domain_diffs` (α=1) | 75.03% | 72.76% | **+0.76** |
| +DDO, image-grounded `domain_diffs` (α=1) | 73.81% | 71.31% | **−0.46** |

*(Reference, Phase E2's separately-measured number on a very slightly different target_test split: +0.68. Our text-DDO condition's +0.76 is a close, independent reproduction — a useful cross-check that this rerun's setup matches Phase E2's.)*

## What this means

The diagnostic did exactly its job: it measured a real, near-zero alignment on a tiny, cheap probe and correctly decided not to trust the text-only guess — no ambiguity in that part of the result. But the specific thing it fell back to made matters worse, not better. **Plausible explanation:** the text-only regularizer draws on 204 diverse (if individually imperfect) directions, which — even when none of them describes Midjourney specifically — may act as a broad, noise-averaged regularizer that mildly helps generalization almost by volume. The image-grounded fallback replaces this with exactly *one* direction, estimated from only 20 real images per class. That single direction is far noisier (no averaging over diverse phrasings) and, being derived directly from the small target-domain sample, risks encoding sample-specific quirks of those 20 images rather than a stable "what does this domain look like" signal — the orthogonality loss then pushes the classifier away from directions that don't generalize, actively hurting rather than helping. This is a different, more specific failure than "grounding in real images doesn't work" — it's "one noisy direction from a small sample is worse than 204 imperfect-but-diverse text directions," which is a narrower and more useful thing to know.

## Verdict

**Partially solved.** The self-diagnosis capability — cheaply and correctly detecting when a text-only domain guess shouldn't be trusted — works and is validated, matching Phase F3's own independently-measured number. The specific fallback mechanism tested (a single image-measured direction) is a **dead end as implemented**: it does not recover DDO's lost benefit and in fact costs more than doing nothing. This is not evidence that grounding in real images can't work in principle — Phase F2 already showed a *fully trained* classifier reaches strong accuracy on this exact domain — only that this particular, minimal way of injecting real images into DDO's specific regularizer mechanism doesn't help. No literature search was done specifically for this fallback design (out of scope for a single-session validation); `docs/new_methodology_report.md` §6's existing note on Idea 3 (self-diagnosing domain grounding as an underexplored *continual-DG-specific* setting, as opposed to domain adaptation with target data) still applies to the diagnostic half.

## What's next

This is a dead end for the naive single-direction fallback specifically — not for self-diagnosing grounding as a component. Concrete next steps, most promising first:

1. **Preserve directional diversity in the fallback**, rather than collapsing the probe to one mean direction. E.g., bootstrap-resample the probe into several sub-means (giving the regularizer multiple noisy-but-averaged directions, closer in spirit to the text pool's 204), or keep the probe images as individual per-image directions rather than one class mean.
2. **Increase probe size** and check whether the fallback's harm shrinks as the single direction becomes less noisy — 20/class was chosen for cheapness, not tuned; a probe-size sweep (10/20/50/100) would show whether this is a sample-size artifact or a structural problem with single-direction grounding.
3. **Blend rather than replace**: keep the text-only 204 directions and add the image-grounded direction(s) as extra terms, instead of swapping one for the other — tests whether the failure is specifically about *losing* the text pool's diversity, not about the image direction being actively bad.
4. **EuroSAT as a second validation domain** (already downloaded, alignment already known from Phase F1 at 0.324 — worth remeasuring with this project's own probe-scale diagnostic for a 3-point PACS/EuroSAT/Defactify calibration curve) — not yet run, would show whether this result is Defactify-specific or general.
5. Once any fallback variant above shows a real recovery, this component becomes eligible for the combined/ablation-testing stage flagged in `docs/session_handoff.md` §5 (Component 1 + Component 2 together) — not yet applicable while Component 2's own fix isn't working.
