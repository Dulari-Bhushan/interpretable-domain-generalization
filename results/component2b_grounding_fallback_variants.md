# Component 2b — grounding fallback variants (why the single-direction fallback hurt, and what fixes it)

**Status: ⚠️ Done — one finding confirmed, one not yet.** **Confirmed**: the earlier dead end's cause is now precisely isolated — collapsing a real-image probe to *one* mean direction is reliably worse than either keeping every probe image as its own direction (persample) or adding the grounded direction to the text pool instead of replacing it (blend). This holds at every probe size tested (10/20/40/60) and, at the one fully-controlled comparison point (identical images, seed, and split), is a clean 1.05-point swing from packaging alone — not something that needs a seed sweep to believe. **Not confirmed**: that persample/blend also close the gap to text-only DDO itself (+0.87, +0.95 vs. text-DDO's +1.02) — that gap (0.07–0.15 points) is smaller than the ~0.3-point run-to-run noise this project's own three separate measurements of `ddo_text` on this exact domain already show (Phase E2: +0.68, first C2 run: +0.76, this run: +1.02), and rests on a single run per condition. Per instruction, no further runs (e.g. a seed sweep) were done to try to settle this — the report states the limit of what's actually known.

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

**What's confirmed, and doesn't need a seed sweep to believe:** persample (75.23%) and blend (75.30%) both beat *every* single-mean-direction variant tested, at every probe size (73.15–74.75% across probe sizes 10/20/40/60) — a 0.5 to 2.1-point margin, well above the ~0.3-point run-to-run noise this project's own repeated measurements establish. At the one point where the comparison is fully controlled (probe=20, identical images, identical seed, identical split — only the packaging of those 20 images differs), collapsing to one mean direction cost −0.18 points while keeping all 20 as separate directions gained +0.87: a clean 1.05-point swing from nothing but packaging. So: **the single-mean-direction design is reliably worse than either fix, and that's not in question.** This directly confirms the original report's hypothesis — the harm was about throwing away directional diversity, not about grounding in real images being a bad idea per se.

**What's not confirmed:** whether persample/blend actually *close the gap to text-only DDO* (+0.87, +0.95 vs. text-DDO's +1.02 — a 0.07–0.15-point difference, smaller than the ~0.3-point noise floor, single run). This is a separate, weaker claim from the one above, and needs a seed sweep to settle — not run here, per instruction.

**Item 2 (probe size, single-direction design) does *not* show a clean fix.** Gains across probe sizes 10→20→40→60 are −1.20, −0.18, −0.62, +0.40 — noisy and non-monotonic, only turning positive at the largest size tested. This means the single-mean-direction design's instability isn't simply "needs more data" in the range tested (10–60/class); it's a structural property of collapsing to one direction, not primarily a small-sample artifact. (A much larger probe might eventually stabilize it by the same law-of-large-numbers logic that makes the 204-direction text pool work, but that would cost real images at a scale that undercuts the "cheap probe" premise of Component 2 in the first place.)

What Component 2 adds, stated at the confidence level the evidence actually supports: a diagnostic that correctly and cheaply flags an untrustworthy text-only domain guess, plus two fallback designs (blend, persample) that reliably avoid the harm the naive single-direction version caused. Whether they also add a further, independent accuracy benefit over text-only DDO — as opposed to just being a safe, no-worse alternative for when the text guess needs to be corrected — remains open.

## Verdict

**Resolved: the naive fallback is a fixable bug, not a dead end for grounding in real images.** The self-diagnosis remains validated (works identically at every probe size tested). The specific cause of the first run's harm is now well-isolated and confirmed: collapsing a probe to one mean direction, verified as reliably worse than either fix across every probe size tested, and by a clean, controlled 1.05 points at the one point where everything else was held identical. This is a real, defensible finding.

**Still open**: whether the fixed fallback (blend or persample) adds value *beyond* what text-only DDO already provides on this domain, or merely matches it. This project's own data (three measurements of `ddo_text`'s gain on this exact shift moving by ~0.3 points from test-split composition alone) means the observed 0.07–0.15-point gap can't be called confirmed from a single run. Calling this "validated" in the same sense Component 1 is would be overclaiming. No further runs were done to resolve this — see What's Next for what would. No literature search was done specifically for the blend/persample designs (same scope note as the first Component 2 report).

## What's next

1. **A seed sweep (3-5 seeds) on baseline / ddo_text / ddo_grounded_mean_probe20 / ddo_grounded_persample / ddo_grounded_blend** is the single highest-value next step, and the only thing that would actually license claims like "recovers to parity" or "blend beats persample" — cheap (frozen CLIP features, linear-layer-only training, this whole 9-condition sweep took well under an hour) and directly answers the open question this report raises rather than leaves open. Not yet done, and not started without asking first, given the size of the claims resting on it.
2. **Prefer blend over persample if a default must be picked now**, but treat this as a weak, single-run preference (75.30% vs. 75.23%, a 0.07-point gap, not distinguishable from noise) and not a settled recommendation — it's also simpler to implement (fixed 205-direction shape vs. a probe-size-dependent count).
3. **A second validation domain** — everything above is still one domain shift (photo→Midjourney v6). EuroSAT doesn't support a like-for-like DDO training comparison (different label space, per `planning/02-continual-dg-experiment-plan.md`'s own Phase F2 scope note), so this needs either a second AI-generator domain from Defactify (Phase F3 tested 5) or a genuinely different dataset.
4. **EuroSAT calibration (item 4) is done** ([`results/component2_eurosat_calibration.json`](component2_eurosat_calibration.json)): this project's own probe-scale diagnostic (30/class) measures 0.322 mean alignment for photo→satellite, closely matching Phase F1's separately-implemented, larger-sample (200/class) result of 0.324 — a good consistency check that the probe-scale diagnostic replicates a known result. This gives a 3-point calibration curve: PACS 0.249 (trustworthy) / EuroSAT 0.322 (trustworthy, modality-scarce) / Defactify 0.037 (untrustworthy, temporally novel) — consistent ordering with Phase F1/F3/F4's original findings. This is the one solid, low-noise addition from this round (a diagnostic measurement, not a trained-model comparison, so the noise-floor concern above doesn't apply to it).
5. **The combined/ablation-testing commitment** (`docs/session_handoff.md` §5) should **not** be considered triggered by this report — the mechanism finding is real but the magnitude claim needed to call Component 2 "validated" isn't earned yet. Revisit after #1 (seed sweep) and ideally #3 (second domain).
