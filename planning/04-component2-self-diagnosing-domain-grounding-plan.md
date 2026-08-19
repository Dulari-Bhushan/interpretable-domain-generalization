# Component 2 — self-diagnosing domain grounding

## Origin

`docs/new_methodology_report.md` §1, row 2 of the 5-component table: *"Self-diagnosing domain grounding — DDO predicts a new domain's look purely from text, with measured near-zero reliability for domains that matter (Phase F1/F3/F4)."* First component to be attempted after Component 1 (exact classifier update).

## The issue this targets

DDO's orthogonality regularizer (`clip_cbm_orth` in `external/LanCE/model/cbm_models.py`) is built entirely from **text**: `domain_diffs[i] = normalize(text_embedding(target_template) − text_embedding(source_template))`, computed once, per class, from the fixed descriptor pool (`prompts/prompt200new.py`, 204 entries) — never touching a single real image of the domain it's supposedly anticipating. Phase F1/F3/F4 measured how good that text-only guess actually is, using an alignment score (cosine similarity between the *real* image-measured domain-shift direction and the *text-predicted* one):

| Phase | Domain shift | Photo reference | Mean alignment | vs. paper's own claimed 0.90–0.99 |
|---|---|---|---|---|
| (paper's own Fig. 2/8) | sketch/painting/sculpture/clipart | — | 0.90–0.99 | — |
| F1 | photo → EuroSAT (satellite) | CLIP text proxy | 0.324 | ~⅓ of the low end |
| F4 | photo → Midjourney v6 (GenImage) | CLIP text proxy | 0.232 | ~¼ of the low end |
| F3 | photo → 5 AI generators (Defactify) | **real matched photos** | **0.037–0.051** | near-zero |

F4's own write-up flagged that the two text-proxy numbers (F1, F4) likely *overstate* real alignment relative to F3's real-photo version — F3 is the cleanest number in the project. Separately, Phase E2 showed what this costs in trained-model terms: DDO's benefit over a plain baseline collapses from +6.40 points (CUB→Painting, good text guess) to +0.68 points (photo→Midjourney, bad text guess) under an identical protocol.

**What's missing, and what Component 2 is:** none of F1/F3/F4/E2 gave the *model* any way to know, on its own, that its text-only guess for a given domain is untrustworthy. They're external measurements the project made after the fact. Component 2 builds that measurement *into the method* — a cheap, automatic check run the moment a new domain arrives, using a small probe of real images (not a full training set), that decides whether to trust the text-only `domain_diffs` (current LanCE behavior, zero real images needed) or fall back to measuring `domain_diffs` directly from the probe images instead.

## Why this approach specifically

The alignment-score methodology already exists and is already validated (F1/F3/F4) — reusing it as a live diagnostic rather than inventing a new trust metric keeps this component's claim tightly scoped to something already measured to work. The fallback (image-measured domain_diffs) is the natural complement: `clip_cbm_orth`'s regularizer only ever consumes a `(num_directions, num_classes, feat_dim)` tensor of domain-shift directions — nothing in the architecture requires those directions come from text. A single image-measured direction per class (`mean(target probe embeddings) − mean(source probe embeddings)`, normalized) is a drop-in replacement, needing no model changes, only a different way to build one input tensor.

## Method

1. **Diagnostic (`compute_alignment_score`)**: for a small probe set of real images per class from both source and target domains (tens, not hundreds, per class): `visual_shift = normalize(mean_img_emb(target) − mean_img_emb(source))`; `textual_shift = normalize(text_emb(target_template) − text_emb(source_template))`; `alignment = cosine(visual_shift, textual_shift)`, averaged over classes. Identical formula to Phase F3 (the cleanest, real-photo version), generalized to run at probe scale.
2. **Threshold**: calibrated empirically (not guessed) by running the diagnostic itself on a domain shift already known trustworthy (PACS's photo→art/cartoon/sketch — CLIP's comfort zone) and one already known untrustworthy (Defactify's photo→Midjourney v6, F3's 0.037 result) — both measured with this project's own diagnostic code, at the same probe scale, so the threshold is grounded in a real, apples-to-apples separation rather than borrowed from the paper's differently-computed 0.90–0.99 claim.
3. **Fallback (`build_image_grounded_domain_diffs`)**: when alignment falls below threshold, replace the text-only `domain_diffs` tensor with a single image-measured direction per class from the same probe images already used for diagnosis (no extra data collection beyond the probe) — shape `(1, num_classes, feat_dim)` instead of `(204, num_classes, feat_dim)`, which `clip_cbm_orth`'s forward pass accepts unchanged (the descriptor-count dimension is just averaged over in the orthogonality loss).
4. **Validation**: rerun Phase E2's exact baseline-vs-+DDO protocol on Defactify (photo→Midjourney v6), adding a third condition — +DDO with the self-diagnosed, image-grounded `domain_diffs` — under an identical training setup, to see whether grounding recovers some of the +6.40-vs-+0.68 gap Phase E2 measured.

## Dataset(s) used, and why

- **PACS** (photo→art_painting/cartoon/sketch): already on the server, already confirmed CLIP-comfort-zone domains (Component 1's own core result). Used only for the diagnostic's trust-positive calibration point — no training here, this isn't a forgetting test.
- **Defactify/MS-COCO-AI** (photo→Midjourney v6): the exact domain shift Phase F3 (cleanest alignment number) and Phase E2 (the trained-model cost) already measured — reusing it lets Component 2's result connect directly to both without introducing a new unknown dataset into the comparison. Needs downloading directly on the server (not yet transferred, per `docs/session_handoff.md`).

## Code (planned)

- `external/LanCE/model/domain_grounding.py` — diagnostic + fallback module.
- `external/LanCE/experiments/component2_alignment_calibration.py` — PACS (+ reuses Defactify probe) threshold calibration.
- `external/LanCE/experiments/component2_defactify_grounding_ddo.py` — 3-condition training comparison.

## Status

Not yet run. This file records intent; results go in `results/component2_self_diagnosing_grounding.md` once actually executed, per `docs/component_report_template.md`.
