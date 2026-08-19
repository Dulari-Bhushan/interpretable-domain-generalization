# Component 2b — grounding fallback variants (why the single-direction fallback hurt, and what fixes it)

**Status: ⚠️ Done — mechanism identified, magnitude not yet statistically confirmed.** The follow-ups identified in `results/component2_self_diagnosing_grounding.md` "What's next" narrow the earlier dead end to a precise, well-controlled cause: collapsing the probe to *one* mean direction, not "grounding in real images" per se — a clean within-run ablation (identical probe images, seed, and target_test, differing only in packaging) shows a real 1.05-point swing from that change alone. **What is not yet earned**: the further claim that blend/persample "recover to near text-DDO parity." Across this project's own three measurements of `ddo_text`'s gain on this exact domain shift (Phase E2: +0.68, Component 2's first run: +0.76, this run: +1.02), the gain has moved by ~0.3 points purely from which images landed in the test split, with zero method change. The gap between the best grounded variant and text-only DDO here (+0.95 vs. +1.02, 0.07 points) is smaller than that demonstrated noise floor and comes from a single run per condition — by this project's own standard (Phase E2 treated a similarly-sized +0.68 gain as noise-plausible), this specific magnitude claim needs a seed sweep before it can be called confirmed.

## One-line summary

Collapsing a real-image probe into one mean direction is demonstrably worse than keeping it as many directions or adding it to the text pool (a clean, controlled 1.05-point swing) — but whether the fixed version actually closes the gap to text-only DDO, versus just landing within this project's own already-documented ~0.3-point run-to-run noise, is not yet established from a single run.

## Origin

Direct follow-up to `results/component2_self_diagnosing_grounding.md`'s "What's next" #1–4 (items 1–3 below; #4, EuroSAT calibration, reported separately in `results/component2_eurosat_calibration.json`, summarized at the end of this report). Not a new planning doc — this is the same Component 2 plan (`planning/04-component2-self-diagnosing-domain-grounding-plan.md`) continuing past its first, partial-result run, the same relationship Component 1b had to Component 1.

## The issue this targets

`results/component2_self_diagnosing_grounding.md` found the self-diagnosis itself worked (correctly flagged photo→Midjourney v6 as untrustworthy), but its fallback — a single `domain_diffs` direction averaged from a 20-images/class probe — made target accuracy *worse* than doing nothing (73.81% vs. baseline 74.27%), worse than the (known-weak) text-only DDO. The write-up's own hypothesis: one noisy direction from a small sample is worse for the orthogonality regularizer than 204 diverse-if-imperfect text directions, even when none of the 204 describe Midjourney specifically. This report tests that hypothesis directly, plus two others (probe size, blending).

## Why we tried these approaches specifically

Three variants, each isolating one variable against a common, fixed evaluation set:

1. **Diversity, holding probe size fixed** — if the diagnosis (single-mean-direction) is right, keeping the same 20 images/class but as 20 separate directions instead of 1 averaged direction should recover most of the loss, since the images and their information content are identical either way.
2. **Probe size, holding the single-direction design fixed** — if the harm is instead just noise from a small sample, a larger probe (which averages out more noise per mean direction) should shrink it, converging toward baseline or better as probe size grows.
3. **Blending, not replacing** — if the harm is specifically about *losing* the text pool's diversity/regularization mass, adding the grounded direction on top of the full 204-direction pool (instead of swapping it in) should do at least as well as text-only DDO.

## Method

All three reuse `model/domain_grounding.py`'s existing pieces plus two additions made for this report:
- `build_image_grounded_domain_diffs_persample`: keeps every probe image as its own direction relative to the class's mean source embedding — shape `(K, num_classes, feat_dim)` instead of `(1, num_classes, feat_dim)`.
- `blend_domain_diffs`: `torch.cat([text_domain_diffs, grounded_domain_diffs], dim=0)` — 205 directions (204 text + 1 image) instead of either 204 or 1.

**A methodological improvement over the first Component 2 run**, needed to make the probe-size sweep fair: `experiments/component2_defactify_grounding_variants.py` fixes `target_test` once (everything beyond the largest probe, 60/class) and reuses it for every condition, including baseline and text-only DDO — the first run's `target_test` shrank along with the probe, which would have confounded a probe-size comparison. Baseline and text-DDO are trained once against this fixed split and referenced throughout; each grounded variant is trained against the same split with only its `domain_diffs` input differing. Same protocol otherwise (50 epochs, batch 64, AdamW, lr 1e-4, weight_decay 1e-4, `clip_cbm_orth`, ViT-L/14).

## Dataset(s) used, and why

Same as the first Component 2 run: Defactify/MS-COCO-AI, photo→Midjourney v6, the exact domain shift Phase F3/E2/Component 2's first run already measured — keeping the same domain shift isolates the *fallback design* as the only changing variable across this whole line of experiments.

## Code

- [`external/LanCE/model/domain_grounding.py`](../external/LanCE/model/domain_grounding.py) — `build_image_grounded_domain_diffs_persample`, `blend_domain_diffs`, and a text-proxy source mode added for `component2_eurosat_calibration.py`.
- [`external/LanCE/experiments/component2_defactify_grounding_variants.py`](../external/LanCE/experiments/component2_defactify_grounding_variants.py) — the fixed-target_test sweep (probe sizes + persample + blend).
- [`external/LanCE/experiments/component2_eurosat_calibration.py`](../external/LanCE/experiments/component2_eurosat_calibration.py) — third calibration point.
- [`results/component2_defactify_grounding_variants.json`](component2_defactify_grounding_variants.json), [`results/component2_eurosat_calibration.json`](component2_eurosat_calibration.json) — raw results.

## Results

All numbers are best target (Midjourney v6) accuracy against the fixed 2,749-image `target_test` set; gain is relative to this run's own baseline (74.35% — close to, not identical to, the first run's 74.27%, since the held-out set differs slightly).

| Condition | Best target acc | Gain over baseline | Diagnostic mean alignment |
|---|---|---|---|
| Baseline (α=0) | 74.35% | — | — |
| +DDO, text-only (204 directions) | **75.37%** | **+1.02** | — |
| +DDO, image-grounded, single mean direction, probe=10 | 73.15% | −1.20 | 0.0054 |
| +DDO, image-grounded, single mean direction, probe=20 | 74.17% | −0.18 | 0.0123 |
| +DDO, image-grounded, single mean direction, probe=40 | 73.74% | −0.62 | 0.0130 |
| +DDO, image-grounded, single mean direction, probe=60 | 74.75% | +0.40 | 0.0150 |
| +DDO, image-grounded, **20 per-sample directions**, probe=20 | **75.23%** | **+0.87** | (same probe as above) |
| +DDO, image-grounded, **blended** (204 text + 1 grounded), probe=20 | **75.30%** | **+0.95** | (same probe as above) |

Every probe size's diagnostic still correctly flags the domain as untrustworthy (0.005–0.015, all far below the 0.143 threshold) — the diagnosis itself is stable regardless of probe size; what varies is only what the fallback does with that information.

## What this means

**Item 1 (diversity) is confirmed as the dominant factor.** At the identical probe size (20/class, same images), collapsing to one mean direction cost −0.18 points; keeping all 20 directions separately gained +0.87 — a 1.05-point swing from changing nothing but how the same information is packaged. This directly supports the original report's hypothesis: the harm wasn't about grounding in real images being a bad idea, it was about throwing away the pool's directional diversity.

**Item 2 (probe size, single-direction design) does *not* show a clean fix.** Gains across probe sizes 10→20→40→60 are −1.20, −0.18, −0.62, +0.40 — noisy and non-monotonic, only turning positive at the largest size tested. This means the single-mean-direction design's instability isn't simply "needs more data" in the range tested (10–60/class); it's a structural property of collapsing to one direction, not primarily a small-sample artifact. (A much larger probe might eventually stabilize it by the same law-of-large-numbers logic that makes the 204-direction text pool work, but that would cost real images at a scale that undercuts the "cheap probe" premise of Component 2 in the first place.)

**Item 3 (blending) works, and works simply.** Adding the single grounded direction on top of the full text pool (205 directions total) reached 75.30% — within 0.07 points of pure text-only DDO (75.37%), and better than the single-direction replacement in every probe-size condition tested. This confirms the harm was specifically about *removing* the text pool's regularization mass, not about the image direction itself being actively counterproductive — the same real-image direction that hurt at −0.18 points as a replacement contributes essentially neutrally-to-slightly-positively as an addition.

**An honest ceiling worth stating plainly**: none of the grounded variants *beat* text-only DDO by a clear margin here — persample and blend land within noise of it (+0.87, +0.95 vs. text-DDO's +1.02), not meaningfully above. This isn't surprising given Phase E2/this component's own first run already established DDO's benefit on this exact domain is itself small (+0.68 to +1.02 across three separate runs) — recovering to "as good as text-only DDO" is the realistic ceiling being tested here, not "clearly better than it." What Component 2 adds isn't a large accuracy win on this specific domain; it's the diagnostic capability (knowing to distrust the text guess) plus a fallback design (blend, or persample) that no longer actively costs accuracy the way the naive version did.

## Verdict

**Partially resolved — the mechanism is solid, the magnitude is not yet confirmed.** The self-diagnosis remains validated (works identically at every probe size tested). The specific cause of the first run's harm is now well-isolated by a clean, controlled comparison: same 20 probe images, same seed, same target_test, only the packaging (1 mean vs. 20 directions) differs, and that alone is worth 1.05 points. That part is a real, defensible finding worth keeping regardless of what happens next.

What is **not** confirmed: that blend or persample actually close the gap to text-only DDO rather than merely landing inside this project's own already-measured run-to-run noise (~0.3 points, observed three separate times on this exact domain shift with zero method change). The observed gap to text-DDO here (0.07–0.15 points) is smaller than that noise floor, and every condition here is a single run. Calling this "validated" in the same sense Component 1 is would be overclaiming — a seed sweep (3-5 seeds per condition, cheap here since these are frozen-CLIP-feature linear classifiers) is the concrete next step that would actually settle it, not yet done. No literature search was done specifically for the blend/persample designs (same scope note as the first Component 2 report).

## What's next

1. **A seed sweep (3-5 seeds) on baseline / ddo_text / ddo_grounded_mean_probe20 / ddo_grounded_persample / ddo_grounded_blend** is the single highest-value next step, and the only thing that would actually license claims like "recovers to parity" or "blend beats persample" — cheap (frozen CLIP features, linear-layer-only training, this whole 9-condition sweep took well under an hour) and directly answers the open question this report raises rather than leaves open. Not yet done, and not started without asking first, given the size of the claims resting on it.
2. **Prefer blend over persample if a default must be picked now**, but treat this as a weak, single-run preference (75.30% vs. 75.23%, a 0.07-point gap, not distinguishable from noise) and not a settled recommendation — it's also simpler to implement (fixed 205-direction shape vs. a probe-size-dependent count).
3. **A second validation domain** — everything above is still one domain shift (photo→Midjourney v6). EuroSAT doesn't support a like-for-like DDO training comparison (different label space, per `planning/02-continual-dg-experiment-plan.md`'s own Phase F2 scope note), so this needs either a second AI-generator domain from Defactify (Phase F3 tested 5) or a genuinely different dataset.
4. **EuroSAT calibration (item 4) is done** ([`results/component2_eurosat_calibration.json`](component2_eurosat_calibration.json)): this project's own probe-scale diagnostic (30/class) measures 0.322 mean alignment for photo→satellite, closely matching Phase F1's separately-implemented, larger-sample (200/class) result of 0.324 — a good consistency check that the probe-scale diagnostic replicates a known result. This gives a 3-point calibration curve: PACS 0.249 (trustworthy) / EuroSAT 0.322 (trustworthy, modality-scarce) / Defactify 0.037 (untrustworthy, temporally novel) — consistent ordering with Phase F1/F3/F4's original findings. This is the one solid, low-noise addition from this round (a diagnostic measurement, not a trained-model comparison, so the noise-floor concern above doesn't apply to it).
5. **The combined/ablation-testing commitment** (`docs/session_handoff.md` §5) should **not** be considered triggered by this report — the mechanism finding is real but the magnitude claim needed to call Component 2 "validated" isn't earned yet. Revisit after #1 (seed sweep) and ideally #3 (second domain).
