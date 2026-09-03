# Plan 07: does a directly-trained CBM on a strong pretrained backbone beat CLIP-zero-shot concept activations?

**Stages 1 and 2 below are done — results in [`results/concept_source_backbone_comparison.md`](../results/concept_source_backbone_comparison.md).** Stage 3 (§7) is a new proposal written up after Stage 2's result, not yet run.

## 0. Where this comes from, and what it is

This is a direct instance of **Variant B** from [`planning/03-detector-grounded-concept-extraction-plan.md`](03-detector-grounded-concept-extraction-plan.md) §2, requested explicitly by the project owner while looking at the architecture diagram: instead of getting the image-side concept activation via CLIP's similarity trick (`E_I(x) · E_T(c_i)`, the green-circled step in the diagram — `self.clip_model.encode_image(images)` inside [`clip_cbm_orth`](../external/LanCE/model/cbm_models.py:171)), swap in a **directly-trained concept classifier** sitting on top of a strong, non-CLIP pretrained backbone (DINOv2), trained on real per-image concept labels. Everything downstream of the concept-activation vector (the linear classifier, DDO where applicable) is unchanged in spirit — this plan only replaces *how a concept score is produced*.

Plan 03 already scoped this exact variant and flagged its binding constraint (§2): a directly-trained concept classifier needs real, per-image concept **labels** to supervise it, and **CUB is the only dataset in this project that has them** (312 real human-labeled attributes/image, from the original CBM paper's own annotation effort). This plan is the concrete, staged version of that variant, CUB-only, matching Plan 03's own suggested order of work (§7: "Variant B ... optional, lower priority ... since Variant C already answers the main question without the labeling cost" — the project owner asked for it directly this time, so it's promoted).

## 1. The two variants being compared

| Variant | How it decides a concept is present | Backbone | Needs labels? |
|---|---|---|---|
| **A — CLIP zero-shot (existing, baseline)** | `image_embedding · text_embedding(concept)`, no training | CLIP ViT-L/14 image + text towers (frozen) | No |
| **B — Directly-trained concept classifier (new)** | A trained linear probe per concept, supervised on real attribute labels | DINOv2 ViT-B/14 and ViT-L/14 (frozen backbone, linear head trained) | Yes — CUB's real 312 attributes |

DINOv2 (not a CLIP variant, not ImageNet-supervised ResNet-50) was picked because the project owner explicitly said fidelity to the original 2020 CBM paper's ResNet-50 doesn't matter here — the point is "use a CBM instead of CLIP," with a strong modern pretrained backbone. DINOv2 is a reasonable current choice: self-supervised, not trained against any text/label signal that could make its features suspiciously well-aligned with the task by construction (unlike CLIP, which is already text-image aligned), so a strong result here is a genuine test of "do good general visual features + real supervision beat zero-shot CLIP similarity," not an artifact of CLIP-family features appearing on both sides. Both ViT-B/14 and ViT-L/14 are run, so the result also separates "does supervision help" from "does backbone scale help."

Both backbones are kept **frozen** (linear probe only, no fine-tuning) — this matches how CLIP is used everywhere else in this project (`for p in self.clip_model.parameters(): p.requires_grad = False`, every model class in `cbm_models.py`) and keeps the comparison about *which frozen feature space + concept-scoring mechanism* is better, not about how much fine-tuning budget was spent. If Stage 1 is promising, fine-tuning the backbone is a natural (but not yet committed) follow-up.

## 2. What happens to DDO

Per Plan 03 §1, once concept activations stop coming from CLIP similarity, DDO's text-only domain-shift trick no longer lives in the same space as the new concept activations. **Stage 1 below doesn't touch DDO at all** — it's a concept-level agreement check, no classifier training, no domain shift, matching Plan 03's own Stage 1 design. If Stage 2 (downstream classification, below) is picked up later, it runs **without DDO**, comparing plain classification accuracy only — same handling Plan 03 already specifies.

## 3. Stage 1 — concept-level agreement (this is what gets run now)

**Question:** on CUB, does variant B's concept activation agree with the real 312-attribute ground truth better than variant A's zero-shot similarity does?

**Protocol:**
1. For each backbone (CLIP ViT-L/14 zero-shot; DINOv2 ViT-B/14 linear probe; DINOv2 ViT-L/14 linear probe), produce a continuous concept-activation score per (test image, concept) pair.
   - Variant A: cosine similarity between the CLIP image embedding and the CLIP text embedding of each of CUB's 312 concept names (`data/CUB/cub_concepts.txt`) — no training, direct zero-shot score.
   - Variant B (×2 backbones): a `LayerNorm → Linear(feat_dim, 312)` head, trained with `BCEWithLogitsLoss` against CUB's real per-image binary attribute labels (`data/CUB/CUBpath2attr.pkl`) on the train split; sigmoid output on the held-out test split is the concept-activation score.
2. Score every variant against the same ground truth (CUB's real per-image attribute labels, test split) using per-concept **AUROC** (threshold-free, works identically for a cosine-similarity score and a trained sigmoid probability, so it's a fair metric across variants without picking an arbitrary cutoff). Concepts with only one class present in the test split are skipped (AUROC undefined) and the skip count is reported honestly.
3. Report: mean/median AUROC per variant, how many of the 312 concepts each variant "wins" on, and the specific concepts where variant B helps most/least — this last part matters because Plan 03 §6 already flags that detector/classifier-style approaches are expected to do better on localizable concepts (a beak shape) than diffuse whole-image ones (glossy texture); Stage 1's per-concept breakdown is what would actually show that pattern, not just an aggregate number.

**What would make this worth continuing to Stage 2:** variant B's mean AUROC needs to be real and not noise — comfortably above variant A's, not a coin-flip-sized difference — on both DINOv2 sizes, or at least one of them by a clear margin. If it's a wash or variant A wins, that's a legitimate, useful stopping point per this project's standing honesty commitment (`docs/component_report_template.md` §12 — no padded next steps).

**Code:** [`external/LanCE/experiments/concept_source_cub_stage1.py`](../external/LanCE/experiments/concept_source_cub_stage1.py).

## 4. Stage 2 — concept-level result + downstream classification accuracy (documented here, deferred — run only if time permits)

**Not started. This section exists so the idea is recorded precisely rather than left as a vague "and also check accuracy" — per the project owner's explicit instruction to write this down now and pick it up later if there's time.**

**What it would add on top of Stage 1:** train a classifier on top of each variant's concept-activation vector (matching `clip_cbm`'s own downstream design: `LayerNorm(n_concepts) → Linear(n_concepts, n_classes)`, no DDO term per §2 above) and compare final CUB→CUB-Painting classification accuracy across variants — mirroring Phase 0's own baseline protocol (`results/phase0_cub_reproduction.md`), so the number is stated the same way as every other accuracy result in this project.

**Why this is a separate, later stage rather than folded into Stage 1:** Stage 1 answers "are the concept scores themselves more accurate" — a direct, cheap, per-concept check against real labels. Stage 2 answers a different, more expensive question — "does that translate into better final classification" — which requires training a full classifier per variant and evaluating on the CUB-Painting domain shift, and depends on Stage 1's result being real before it's worth spending the extra GPU time (per §3's continuation gate above).

**Protocol, once picked up:**
1. Reuse Stage 1's cached concept-activation vectors (train + test) for each variant as the classifier's input features — no need to re-run backbone inference.
2. Train `LayerNorm → Linear(n_concepts, 200)` per variant on CUB train, evaluate on CUB test (in-domain) and CUB-Painting test (the domain-shift check, same target set Phase 0 used).
3. Report accuracy per variant, both in-domain and on the CUB-Painting shift, directly next to Phase 0's existing baseline number (50.64% in-domain baseline, `results/phase0_cub_reproduction.md`) so it's comparable, not a new number invented in isolation.

**Code (not yet written):** would extend `concept_source_cub_stage1.py` or add a sibling `concept_source_cub_stage2_downstream.py`, following the same file-naming convention.

## 5. Metrics summary

| Stage | Metric(s) | Matches the convention from |
|---|---|---|
| 1 (this run) | Per-concept AUROC vs. real CUB attribute labels; mean/median; win count | New to this plan — Plan 03 §5 anticipated this metric for its own Stage 1 but never ran it |
| 2 (deferred) | Classification accuracy, in-domain and CUB→CUB-Painting shift | Phase 0 |

## 6. Risks and open questions

- **Class imbalance per concept.** Many of CUB's 312 attributes are rare (present in a small minority of images) — AUROC is reasonably robust to this but not immune; the per-concept breakdown in the write-up should flag any concept where the positive count is small enough that the AUROC number is noisy (e.g. under ~10 positive test examples).
- **Frozen linear probe may understate Variant B's true ceiling** — fine-tuning DINOv2 end-to-end (closer to the original CBM paper's own training recipe) would likely score higher still; deliberately not done in Stage 1 to keep the comparison about frozen-feature quality and keep the run fast, per §1.
- **This only tests CUB.** Plan 03 §2 already flags that Variant B can't run anywhere else in this project without new concept labels — so a strong Stage 1 result doesn't by itself say anything about PACS/Office-Home/EuroSAT, where this project's actual domain-shift claims live. That gap stays open regardless of Stage 1's outcome.

## 7. Stage 3 (proposed, not yet run) — combined DINOv2 concept source + grounded domain-shift regularizer

**Where this comes from:** the one "what's next" item that cleared the 90% bar in `results/concept_source_backbone_comparison.md` §11, once Stage 2 showed DINOv2's concept source wins on CUB→CUB-Painting (67.15% ViT-L/14 vs. 50.34% CLIP zero-shot) with **no** domain-shift-specific machinery on either side. Open question: does adding a DDO-style regularizer on top of the DINOv2 concept source push that gap further, or has DINOv2's concept space already captured whatever DDO was compensating for on the CLIP pipeline?

**Why DDO can't just be reattached as-is** (§2 above already established this in general; restated precisely here): `clip_cbm_orth`'s regularizer computes `self.classifier[1:](self.diffs @ self.concept_embeddings.T)` (`model/cbm_models.py:204`) — it needs two CLIP-specific objects. `self.diffs` is a domain-shift vector living in **CLIP's text embedding space**, built from GPT-3.5-written domain descriptor prompts (`utils.get_domain_text_embs`, `prompts/prompt200new.py`'s `target_text_prompts`). `self.concept_embeddings` is **CLIP's own text-embedding matrix** for the concept names. Neither exists for the DINOv2 pipeline: DINOv2 has no text encoder at all, and Stage 1/2's DINOv2 concept space is defined by a *trained linear probe's weight matrix* (`Linear(feat_dim, 312)`), not a text-embedding lookup. So DDO's exact formula doesn't apply — a domain-shift regularizer for this pipeline needs a different, if analogous, construction.

**The concrete adaptation, reusing machinery this project already built and validated (Component 2):**
1. **Domain-shift vector, in DINOv2's own feature space instead of CLIP's.** `model/domain_grounding.py`'s `build_image_grounded_domain_diffs` / `_persample` functions already compute a domain-shift direction from real probe images rather than text (`mean(target_probe_embeddings) − mean(source_probe_embeddings)`, per class) — validated on Defactify/DALL-E 3 (`results/component2c_seed_sweep_and_second_domain.md`, blend variant: +1.17 mean gain across seeds/domains). The formula itself isn't CLIP-specific, only the current implementation is (`_mean_image_embeddings_by_class` hardcodes `clip_model.encode_image`) — swapping in the DINOv2 backbone's forward pass is the only change needed there.
2. **Projecting that shift into concept space.** CLIP's version projects via `@ self.concept_embeddings.T` (CLIP's text embeddings of the 312 concept names). The DINOv2 analog: project through the **trained probe head's own linear layer** instead (its `Linear(feat_dim, 312)` weight matrix) — the natural feature-space-to-concept-space map for this pipeline, since that's literally what turns a DINOv2 feature vector into a concept-activation vector everywhere in Stages 1/2.
3. **The regularizer term.** Feed that concept-space domain-shift vector through the downstream classifier's own `LayerNorm → Linear` (mirroring `self.classifier[1:](...)` exactly) and penalize its magnitude the same way DDO already does (`torch.abs(reg).mean()`, every training script in this project) — same orthogonality *idea* (a classifier insensitive to the direction concepts move under a domain shift), applied to DINOv2's own concept space instead of CLIP's.

**What this would answer that Stage 1/2 didn't:** whether DINOv2's large, regularizer-free domain-shift gain is because its concept space is already close to domain-invariant (an added regularizer would then do little), or whether there's still a domain-specific-concept problem a grounded regularizer would catch and fix on top (accuracy should then rise further, in the same direction as Component 2's own +1.17-point mean gain on Defactify).

**Scope, honestly — this is new engineering, not a rerun:** (1) a probe-image loader for CUB-Painting (Component 2's pattern needs a small `source_images_by_class`/`target_images_by_class` probe — CUB-Painting's existing 200-class structure supports this directly, no new data collection), (2) a DINOv2-backbone version of `_mean_image_embeddings_by_class`, (3) a small `clip_cbm_orth`-style wrapper around Stage 2's DINOv2 classifier that adds the regularizer term Stage 2's plain classifier doesn't have. Not started.
