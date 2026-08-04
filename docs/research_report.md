# Do concept bottleneck models survive when domains show up one at a time?

**A hands-on investigation into whether LanCE — a strong, recent method for making CLIP-based image classifiers robust to new visual domains — actually holds up once those domains stop arriving all at once, and start arriving continually, over time.**

- **Base method:** LanCE (Zeng et al., CVPR 2025) — [arXiv:2503.18483](https://arxiv.org/abs/2503.18483) · [original code](https://github.com/joeyz0z/LanCE)
- **Report sections:** 9, grounded in 11 real experiment runs
- **Datasets used:** 6
- **Report date:** 4 Aug 2026

This project starts from **LanCE**, a CVPR 2025 method that makes CLIP-based image classifiers more robust to domain shift (e.g. training on photos, testing on paintings) by using AI-generated text descriptions of visual styles to scrub domain-specific bias out of the model, once, during a single training run. That "once" is the whole story of this report. Real deployments don't get every domain up front — a hospital adopts a new scanner, a new AI image generator appears, a satellite mission updates its sensors. This project asks two separate questions about what happens next, runs eleven measured experiments to answer them (grouped into nine sections below — a couple were folded together where they tested the same underlying question), and reports every result exactly as it came out — including the ones that didn't match the original prediction.

> **Two frozen components, not one.** LanCE actually depends on **two** things that are each frozen once and never revisited: the **CLIP backbone** (does it visually understand a new domain?) and the **GPT-3.5 descriptor list** that LanCE's DDO mechanism regularizes against (does it even have the right *words* for a new domain?). Phases F1/F3/F4 test the first. Phase F3's descriptor-pool check and Phase E2, added after this report's first pass, test the second directly — including a real trained-model accuracy comparison, not just a representation-level score. See §4 for the full breakdown.

---

## §1 Why two separate pillars

LanCE has two moving parts, and each one can fail in continual deployment for a different reason. Testing only one would leave half the argument unmeasured.

**Pillar 1 — Forgetting.** LanCE trains a small classifier on top of frozen features, once, on whatever domains are available that day. If a new domain shows up later and you fine-tune on it without replaying old data, does the classifier forget what it learned about earlier domains — the textbook continual-learning failure mode?

**Pillar 2 — Backbone coverage.** Even a perfect fix to Pillar 1 can't help if the frozen CLIP vision-language backbone itself never learned to represent a domain properly in the first place — e.g. satellite imagery, or an AI art style invented after CLIP's training data was collected. This failure lives below the classifier, in the backbone LanCE never updates.

Both pillars had to be tested because they are independent: a continual-learning fix (Pillar 1) cannot repair a backbone blind spot (Pillar 2), and a well-covered backbone doesn't stop the classifier from forgetting (Pillar 1). The report is organized around this split throughout.

---

## §2 Every hypothesis, at a glance

| Phase | Hypothesis (plain English) | Dataset | Result | Status |
|---|---|---|---|---|
| **0** (trust gate) | Can we reproduce the paper's own reported numbers before building anything on top of this codebase? | CUB-200-2011 → CUB-Painting | Reproduced: 50.64% baseline / 57.04% +DDO, vs. paper's 50.54% / 55.53% | ✅ Pass |
| **B** | Training PACS's 4 domains one at a time, with no replay, causes forgetting of earlier domains | PACS (4 domains, 7 classes) | Real forgetting (BWT −8.30) in 1 of 3 orderings; other 2 stayed near the joint upper bound | ⚠️ Partial |
| **B.1** (follow-up to B) | Standard, off-the-shelf continual-learning fixes already solve Phase B's forgetting | PACS | Cumulative DDO / replay closed the gap almost fully (−8.30 → −0.07 / −0.96); EWC worked but cost 7–12 accuracy points | ✅ Mostly resolved |
| **D** | A harder, more class-crowded benchmark reveals forgetting more consistently than PACS did | Office-Home (4 domains, 65 classes) | All 3 orderings showed real negative BWT (−0.7 to −4.7); fixes only partially closed the gap | ✅ Strongest evidence |
| **F1** | CLIP's image–text alignment breaks down for a domain it rarely saw in training (satellite imagery) | EuroSAT (10 classes) | Mean alignment 0.32, vs. paper's own 0.90–0.99 range — no overlap | ✅ Confirmed |
| **F2** | A trained CBM classifier is also capped near CLIP's weak zero-shot accuracy on that same domain | EuroSAT (in-distribution) | No — trained CBM reached 90.9%, near the 98.1% linear-probe ceiling, far above the ~60% zero-shot ceiling | ❌ Hypothesis wrong — reframed |
| **F3** | Alignment collapses further for post-2021 domains — *and* the frozen descriptor list never had the words for them either | Defactify / MS-COCO-AI (5 post-2021 AI generators, real photo pairs) + LanCE's own shipped descriptor pool | Mean alignment 0.05 (0.037 per-class) — lowest score in the project; separately, 0/20 AI-generation terms found anywhere in the actual 204-descriptor pool | ✅ Strongest evidence |
| **F4** | A third, independent post-2021 dataset confirms Phase F3's finding | GenImage / Midjourney (155 of 1,000 classes, partial download) | Mean alignment 0.23 — confirms the direction, reveals a methodology nuance (see §4) | ✅ Confirmed, with caveat |
| **E2** | A trained baseline-vs-+DDO run shows how much the descriptor-list gap actually costs in accuracy, not just in a proxy score | Defactify / MS-COCO-AI (real photos → Midjourney v6, 23 classes, unmodified descriptor pool) | DDO gain collapses to +0.68 pts (vs. Phase 0's +6.40 pts) — but trained accuracy on the new domain (74.7%) isn't depressed at all | ⚠️ Confirmed, nuanced |

*A side-check not listed as its own row: removing the most target-relevant descriptors from the pool (a continuous extension of the paper's own relevant/irrelevant ablation) left accuracy flat (55.9–57.1%) on CUB→CUB-Painting — a null result. It didn't disprove the closed-world critique, it relocated it: within a family of art-style descriptors the exact one barely matters; the critique bites harder at "wrong category of domain entirely," which is exactly what Pillar 2 tests. Full write-up: [`results/phase_a_descriptor_coverage.md`](../results/phase_a_descriptor_coverage.md).*

---

## §3 Pillar 1 — Does LanCE survive domains arriving over time?

### Phase 0 — Baseline reproduction
**CUB-200-2011 → CUB-Painting · trust gate for every later phase**

Every later phase reuses this codebase's model, training loop, and DDO loss unchanged. If the reproduction hadn't matched the paper, nothing built on top of it — five more phases of work — would have been trustworthy. This is a gate, not a hypothesis about domain generalization itself.

Trained the paper's own CLIP-CBM architecture (CLIP ViT-L/14, a 311-concept human-written concept bank) on CUB-200-2011 and evaluated on CUB-Painting, the paper's own primary benchmark. Two runs: without the DDO regularizer (baseline) and with it.

| | Baseline (α=0) | +DDO (α=1) |
|---|---|---|
| Paper, Table 1 | 50.54% | 55.53% |
| Our reproduction | **50.64%** | **57.04%** |

Both numbers land inside the project's pre-set 3–5 point tolerance. This depended on finding and fixing six silent bugs in the paper's released code first (§6) — none of which touch the method itself.

Figures: [`results/figures/phase0_target_accuracy.png`](../results/figures/phase0_target_accuracy.png), [`phase0_all_accuracy_curves.png`](../results/figures/phase0_all_accuracy_curves.png). Full write-up: [`results/phase0_cub_reproduction.md`](../results/phase0_cub_reproduction.md).

### Phase B — Sequential domain-incremental forgetting
**PACS · 4 domains, 7 classes, ~9,991 images**

This is the core question of Pillar 1: LanCE's classifier is trained once, jointly, on every available domain. Nothing in the method defines what happens when a new domain shows up after deployment. Standard continual-learning theory — and prior work on closely related concept-bottleneck architectures (CONCIL, CI-CBM) that had to build dedicated anti-forgetting machinery for exactly this reason — predicts that naively fine-tuning on new domains without replaying old data causes catastrophic forgetting. LanCE has no analog of that machinery.

**Why PACS:** the smallest standard 4-domain generalization benchmark, cheap to iterate on while building the continual-learning harness from scratch for the first time.

Trained one model per domain ordering, moving through PACS's 4 domains one at a time with a fresh optimizer at each stage and no replay, evaluating on all 4 domains after every stage. Compared against a joint/oracle model trained on all domains pooled at once (the non-continual upper bound). Repeated across 3 domain orderings.

**Joint/oracle upper bound: 98.29% ACC** (photo 100%, art painting 99.0%, cartoon 98.7%, sketch 95.4%)

| Domain order | Final ACC | BWT |
|---|---|---|
| photo → art → cartoon → sketch | 92.16% | **−8.30** |
| sketch → cartoon → art → photo | 97.81% | −0.54 |
| art → sketch → photo → cartoon | 97.96% | −0.26 |

> **BWT (backward transfer):** how much an earlier domain's accuracy changes after training on later domains. BWT = 0 means no forgetting. More negative = worse forgetting. Positive BWT (rare) means later training actually helped the earlier domain.

Only the **photo-first ordering** shows real forgetting: photo accuracy holds at 99–100% through the first two later stages, then drops to 87.4% by the end — a genuine ~12-point fall below its own diagonal value. The other two orderings stay within 1–2 points of their diagonal values and close to the oracle ceiling throughout.

**Why:** PACS's 7 broad, visually distinct classes give the classifier enormous slack — near-ceiling accuracy (95–100%) leaves little room for a forgetting signal to appear. The one ordering that does show forgetting demonstrates the effect is real and order-dependent, exactly as continual-learning theory predicts — but PACS's ease means most orderings don't expose it. This didn't retract the core claim; it meant PACS alone was too soft a test, which is why Phase D became a priority rather than an optional stretch goal.

Figures: [`phase_b_bwt_comparison.png`](../results/figures/phase_b_bwt_comparison.png), [`phase_b_acc_heatmap_photo_art_cartoon_sketch.png`](../results/figures/phase_b_acc_heatmap_photo_art_cartoon_sketch.png). Full write-up: [`results/phase_b_domain_il.md`](../results/phase_b_domain_il.md).

### B.1 — Do standard fixes already solve it? *(a follow-up to Phase B, not its own numbered phase)*

Phase B found forgetting, but only in one of three orderings. The question here: does that gap need anything new, or does it already close under a standard, off-the-shelf continual-learning fix layered on the exact same harness? A gap that closes trivially is a weaker argument for "this needs new research" than a gap that resists the field's default toolkit.

Three fixes, all layered on the identical harness, hyperparameters, and domain orderings as Phase B, reseeded identically so each starts from the same point as the baseline:

1. **Cumulative DDO** — LanCE's orthogonality penalty normally regularizes only against its fixed 204-descriptor pool, unchanged at every stage. Here, at each new stage the classifier is *additionally* pushed to stay orthogonal to the average feature direction of every domain trained on so far — not just the current one. Costs nothing extra to store (reuses cached embeddings).
2. **Cached-embedding replay** — the classic continual-learning trick: mix a small sample (100 cached feature vectors) from every prior domain into the current domain's training batches. Unlike cumulative DDO, this has a real memory cost.
3. **EWC (Elastic Weight Consolidation)** — after each domain, estimate which classifier parameters mattered most for it (a Fisher-information approximation), then penalize moving those specific parameters too far away while training later domains. λ=1000, an untuned default.

| Domain order | Baseline BWT | Cumulative DDO | Replay | EWC |
|---|---|---|---|---|
| photo → art → cartoon → sketch | **−8.30** | −0.07 | −0.96 | −0.16 |
| sketch → cartoon → art → photo | −0.54 | −1.24 | −0.15 | +0.17 |
| art → sketch → photo → cartoon | −0.26 | −0.29 | +0.08 | −0.13 |

| Domain order | Baseline ACC | Cumulative DDO | Replay | EWC |
|---|---|---|---|---|
| photo → art → cartoon → sketch | 92.16% | 97.41% | 97.81% | 90.31% |
| sketch → cartoon → art → photo | 97.81% | 97.03% | 98.18% | 85.55% |
| art → sketch → photo → cartoon | 97.96% | 97.73% | 98.21% | 94.70% |

(Joint/oracle reference: 98.29% ACC.)

**Cumulative DDO and replay hold BWT within ~1.3 points of zero in every ordering, and match or beat baseline accuracy in 5 of 6 comparisons** — close to a clean win, at little to no cost. **EWC also controls forgetting** (never worse than −0.17) **but costs 7–12 accuracy points** in two orderings — the classic stability–plasticity tradeoff: at this untuned penalty strength, it prevents drift so effectively that it also partly prevents learning the new domain.

This does not mean LanCE has a hidden built-in continual-learning mechanism — every fix here is externally bolted on. What it means is that Phase B's forgetting, where it appeared at all, was shallow enough that a cheap intervention closed almost all of it. Combined with Phase B's own finding that forgetting showed up in only 1 of 3 orderings, this reinforced the working theory that PACS's easy, near-ceiling task doesn't strain the architecture enough to make its lack of a built-in update rule bite hard — which made Phase D's harder benchmark more important, not less.

Figures: [`phase_c_bwt_comparison.png`](../results/figures/phase_c_bwt_comparison.png), [`phase_c_acc_comparison.png`](../results/figures/phase_c_acc_comparison.png), [`phase_c_photo_first_forgetting.png`](../results/figures/phase_c_photo_first_forgetting.png). Full write-up: [`results/phase_c_remediation.md`](../results/phase_c_remediation.md).

### Phase D — A harder benchmark (Office-Home)
**Office-Home · 4 domains, 65 classes, ~15,588 images · promoted from optional stretch goal**

The working theory after Phases B and B.1 was that PACS's near-ceiling accuracy (98.3% joint) was masking a real forgetting effect rather than indicating genuine robustness. Office-Home was chosen specifically because it's harder along exactly that axis: 65 everyday object classes instead of 7, with substantially less data per class per domain, same 4-domain structure so the comparison to PACS is apples-to-apples.

Identical protocol, hyperparameters, and code to Phases B and B.1, generalized to a different dataset, with a newly written 257-concept bank for Office-Home's 65 classes.

**Joint/oracle: 90.78% ACC** (art 90.7%, clipart 82.1%, product 96.9%, real world 93.5%) — meaningfully below PACS's 98.3% ceiling, exactly as the harder-benchmark theory predicted.

| Domain order | Baseline BWT | Cumulative DDO | Replay | EWC |
|---|---|---|---|---|
| art → clipart → product → real world | **−0.68** | +0.33 | −0.37 | +0.15 |
| real world → product → clipart → art | **−2.39** | −2.31 | −1.63 | +0.08 |
| clipart → real world → art → product | **−4.68** | −3.30 | −3.06 | +0.14 |

**Every single Office-Home ordering shows negative baseline BWT** — not the one-out-of-three pattern PACS showed. It's also not simply "the same effect, bigger everywhere": the art-first ordering is milder on Office-Home than PACS's worst case, but the other two orderings are clearly worse. The more important property isn't magnitude — it's consistency: this removes the "maybe that one PACS ordering was a fluke" caveat Phase B had to carry.

The remediations that nearly fully fixed PACS are now only partial fixes: cumulative DDO and replay reduce BWT in every ordering but leave real forgetting behind in the harder two (−2.31/−3.30 and −1.63/−3.06 respectively). EWC again drives BWT close to zero, but at a bigger accuracy cost than on PACS (80.2–86.9% vs. baseline's own 87.8–89.6%).

This is the strongest evidence in the project for Pillar 1's core claim. It confirms the theory directly: PACS's forgetting wasn't absent because LanCE has hidden resilience — it was masked by a task too easy to expose the failure mode cleanly. That said, this still isn't a dramatic, catastrophic collapse — the worst case (−4.68 BWT) still leaves the model at 87.8% accuracy, only about 3 points below the joint/oracle ceiling.

Figures: [`phase_d_vs_phase_b_bwt.png`](../results/figures/phase_d_vs_phase_b_bwt.png), [`phase_d_bwt_comparison.png`](../results/figures/phase_d_bwt_comparison.png), [`phase_d_acc_comparison.png`](../results/figures/phase_d_acc_comparison.png). Full write-up: [`results/phase_d_officehome.md`](../results/phase_d_officehome.md).

---

## §4 Pillar 2 — Does the frozen backbone have a shelf life?

> **What "alignment score" actually means.** LanCE's trick for handling a new domain *without any real images from it*: take the text embedding of "a photo of X" and "a sketch of X," compute the difference, and assume that direction points the same way as the *real* difference between actual photo and sketch images in CLIP's embedding space. **Alignment score = how well that assumption holds** — the cosine similarity between the text-predicted direction and the real, measured direction. 1.0 means the words perfectly predict where real images of that domain sit; 0 means no relationship at all. The paper's own well-behaved domains score 0.90–0.99. A low score means DDO's entire text-only simulation of "what would this domain look like" has nothing reliable to work from — but on its own, a low score doesn't say how much that costs in real trained-model accuracy. That's a proxy limitation stated honestly below, and it's exactly what Phase E2 was built to close.

### Phase F1 — Modality scarcity (EuroSAT)

LanCE's entire mechanism depends on one empirical claim: that CLIP's shared embedding space aligns visual domain shifts with textual domain descriptions. The paper demonstrates this only for domains heavily represented in CLIP's web-scraped training data — reporting alignment scores of 0.90–0.99 for those. The authors themselves admit a limitation (Appendix G): *"these models are limited in application to some professional fields like medical treatments."*

**Why EuroSAT:** OpenAI's own CLIP paper already publishes an independent anchor number for it — 59.6% zero-shot accuracy vs. 98.1% linear-probe accuracy — giving a ceiling to compare against that wasn't produced by this project itself.

For each of EuroSAT's 10 land-cover classes, measured cosine similarity between a visual domain-shift direction (satellite images vs. a photo reference) and a textual domain-shift direction ("a satellite image of X" vs. "a photo of X"). No matched real-photo dataset was available without turning this into a second data-collection project, so CLIP's own text embedding of "a photo of X" stood in for the photo reference.

**Mean alignment: 0.324** — every one of the 10 classes lands between 0.28 and 0.35, with zero overlap with the paper's own reported 0.90–0.99 range.

As an independent second signal, our own zero-shot classification accuracy on EuroSAT was 64.05%, close to OpenAI's published 59.6% — both far below the 98.1% linear-probe ceiling.

Figure: [`phase_f1_alignment_per_class.png`](../results/figures/phase_f1_alignment_per_class.png). Full write-up: [`results/phase_f1_eurosat_alignment.md`](../results/phase_f1_eurosat_alignment.md).

### Phase F2 — Concept-activation ceiling test

> **Three ways to get a class prediction out of CLIP.** *Zero-shot*: no training data at all — just compare an image's embedding to each class name's text embedding and pick the closest. *Linear probe*: train one simple classifier layer directly on CLIP's raw 768-dimensional image embedding, using real labeled examples — full access to whatever visual information CLIP's image encoder captured, no language involved. *CBM (this project's architecture)*: sits in between — also trains on real labeled data, but funnels the image through a small set of hand-written concept phrases first. On EuroSAT: zero-shot ≈ 60%, linear probe ≈ 98% (OpenAI's own published numbers).

Given F1's weak alignment score, the natural next question was whether that weakness caps a trained classifier's accuracy too — if concept activations inherit CLIP's alignment weakness, a trained CBM should be stuck near CLIP's ~60% zero-shot ceiling rather than reaching the ~98% a linear probe achieves.

Trained a plain CLIP-CBM (DDO turned off, isolating the concept-bottleneck architecture itself) directly on EuroSAT, in-distribution — a random train/test split of the same domain, not a source-to-target shift. This is a precondition check, not a domain-generalization test.

**Best test accuracy: 90.89%** — far above both zero-shot anchors and much closer to the 98.1% linear-probe ceiling.

The literal hypothesis is directly falsified — but working through why sharpens the argument rather than weakening it. A linear probe has full access to the raw embedding and trains on it directly. Zero-shot uses no training data at all — it takes CLIP's frozen text-image match at face value. The CBM sits architecturally in between: it projects the image into a concept-activation space and then *trains* a linear layer on top of that projection, on real labeled data. Training recovers most of the class-discriminative signal even though each individual concept's alignment may be imperfect.

**The sentence that survives, more precise than the original hypothesis:** the visual representation is fine; what's specifically broken is *zero-shot, training-free* alignment. LanCE's DDO mechanism is architecturally exactly that — a zero-target-data, text-only simulation of what a new domain will look like, with no access to the recovery path (real target-domain training) that made this 90.9% possible. So the ceiling that matters for LanCE specifically isn't a trained-CBM ceiling — it's what DDO's text-only mechanism alone can achieve, which is precisely what F1's weak alignment score already speaks to.

Figure: [`phase_f2_ceiling_curve.png`](../results/figures/phase_f2_ceiling_curve.png). Full write-up: [`results/phase_f2_eurosat_ceiling.md`](../results/phase_f2_eurosat_ceiling.md).

### Phase F3 — Temporal novelty (alignment **and** descriptor coverage)

Phase F1 tested *modality scarcity* — satellite imagery is rare in ordinary captioned photos, but it also predates CLIP (2017–19 imagery vs. CLIP's Jan 2021 release), so CLIP could in principle have seen some of it. This phase tests the sharper, more clearly "continual" sub-case directly: domains that emerged *after* the frozen backbone had already finished training. That failure can happen at two different points, and this phase checks both:

1. Using text prompts we hand-wrote ourselves ("a Midjourney-generated image of a {}."), does **CLIP** connect those words to real images of the domain?
2. Does LanCE's own frozen, GPT-3.5-written descriptor list contain *any* such words in the first place, independent of what CLIP could do with them if it did?

**Dataset:** Defactify/MS-COCO-AI ([`Rajarshi-Roy-research/Defactify_Image_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset), Hugging Face) — 96,000 images: 48,000 real COCO photos plus 48,000 AI-generated images split evenly across 5 generators, all captioned. Every generator postdates both cutoffs:

| Generator | Released | Margin past GPT-3.5 cutoff (~Sept 2021) |
|---|---|---|
| Stable Diffusion 2.1 | Dec 2022 | +15 months |
| Stable Diffusion XL | Jul 2023 | +22 months |
| DALL-E 3 | Oct 2023 | +25 months |
| Midjourney v6 | Dec 2023 | +27 months |
| Stable Diffusion 3 | 2024 | +2.5+ years |

**Part 1 — does CLIP connect the words to the images?** Two independently-computed alignment scores: a *global* score mixing all content together per generator, and a *per-class-controlled* score restricted to 24 matched COCO categories (a check that the global number wasn't an artifact of mixing unrelated content together).

| Generator | Global alignment | Per-class alignment |
|---|---|---|
| Stable Diffusion 2.1 | 0.044 | 0.033 |
| Stable Diffusion XL | 0.049 | 0.045 |
| Stable Diffusion 3 | 0.050 | 0.048 |
| DALL-E 3 | 0.017 | 0.023 |
| Midjourney v6 | 0.096 | 0.034 |
| **Mean** | **0.051** | **0.037** |

Both versions agree closely (0.05 vs. 0.037) — the per-class control didn't change the conclusion. Every generator lands below 0.1, under a third of Phase F1's already-large-margin result, and nowhere near the paper's claimed 0.90–0.99 range.

**Part 2 — does the frozen descriptor list even have the right words?** GPT-3.5-turbo's training data cuts off around September 2021 — before any of the 5 generators above existed. Rather than pay for a fresh API call to re-ask a GPT-3.5-class model, we searched the pool that's actually shipped and actually used in every other phase — the more faithful test of the two: [`external/LanCE/prompts/prompt200new.py`](../external/LanCE/prompts/prompt200new.py)'s actual 204-entry `target_text_prompts`.

| Check | Result |
|---|---|
| Direct AI-generation terms found (of 20 checked) | **0 / 20** |
| The 5 generators above, named anywhere in the pool | **0 / 5** |
| Conceptually-adjacent terms found (of 10 checked, e.g. "digital art", "CGI render") | 10 / 10 |

The pool contains terms in the same general neighborhood ("a digital art of a {}.", "a CGI render of a {}.", "a cyberpunk illustration of a {}.") but none describe photorealistic AI-generated imagery mimicking a real photograph.

**Interpretation:** the two results compound rather than overlap. Even if you supply the right words, CLIP barely connects them to real images (alignment ≈ 0.037–0.05) — *and* LanCE's own automatic descriptor-writer would never have supplied those words to begin with (0/20, 0/5). A future CLIP with perfect alignment couldn't fix this alone, because DDO's regularizer only ever sees the 204 fixed phrases — adding a new one needs a human to manually re-prompt the LLM and retrain from scratch. What this doesn't establish is how much either gap costs in real, trained-model accuracy — that's Phase E2, next.

Figures: [`phase_f3_alignment_temporal.png`](../results/figures/phase_f3_alignment_temporal.png), [`phase_e1_term_check.png`](../results/figures/phase_e1_term_check.png). Full write-ups: [`results/phase_f3_temporal_novelty.md`](../results/phase_f3_temporal_novelty.md), [`results/phase_e1_descriptor_staleness.md`](../results/phase_e1_descriptor_staleness.md).

### Phase F4 — Independent replication

One dataset, however clean, is one data point. GenImage was the dataset originally identified as the ideal temporal-novelty test during planning, but was dropped early for data-access reasons before Phase F3's dataset was found. Once a usable subset was located and downloaded directly, it became worth revisiting as an independent third confirmation.

**How F4 differs from F3, precisely:** same target (a post-cutoff generator), same alignment-score formula — the one thing that changes is what stands in for "photo." F3's dataset (Defactify) ships real matched COCO photos. F4's dataset (GenImage) only survived as a partial download with no usable real-photo split (a multi-part Google Drive archive; only the final part was retrievable — 928 Midjourney images spanning 155 of 1,000 ImageNet classes extracted successfully), so it falls back to the same text-proxy substitute F1 used.

**Mean alignment: 0.232** across 155 classes — well below the paper's range, consistent in direction with F1 and F3, but not as extreme as F3's near-zero result.

| Dataset | Sub-case | Photo reference | Mean alignment |
|---|---|---|---|
| EuroSAT (F1) | Modality scarcity | CLIP text (proxy) | 0.324 |
| GenImage/Midjourney (F4) | Temporal novelty | CLIP text (proxy) | 0.232 |
| Defactify (F3) | Temporal novelty | Real matched photos | **0.037** |

All three land far below the paper's claimed range. But the two datasets using CLIP *text* as a stand-in for "photo" (F1, F4) cluster together around 0.2–0.3, while the one dataset with *real* matched photos (F3) sits a full order of magnitude lower. That's a real methodological signal, not noise: the text-proxy anchor is itself an imperfect stand-in for where real photos actually sit in CLIP's embedding space. F3's number should be weighted as the most trustworthy of the three.

Figure: [`phase_f4_three_way_comparison.png`](../results/figures/phase_f4_three_way_comparison.png). Full write-up: [`results/phase_f4_genimage_alignment.md`](../results/phase_f4_genimage_alignment.md).

### Phase E2 — A real training run: what does this cost?

Phase F3 measures an alignment *score* — a representation-level proxy — and its own write-up flags the gap directly: no training run was attempted. This phase closes it: an actual baseline-vs-+DDO training run, source = real photos, target = Midjourney v6 images never seen in training.

**Setup:** Defactify/MS-COCO-AI, captions keyword-tagged against the 80 standard COCO categories (F3's own heuristic); kept every category with ≥90 tagged samples on both the real-photo side and the Midjourney v6 side, across all three dataset splits combined — **23 categories qualified** (bus, train, toilet, giraffe, car, motorcycle, bench, bird, sink, airplane, person, cat, sheep, fire hydrant, dog, stop sign, traffic light, bicycle, bowl, truck, clock, oven, chair), capped at 200 images/class/domain. Wrote a 76-concept hand-written concept bank (4/class). Real photos split 80/20 → 3,303 train / 826 source-test images; 4,129 Midjourney v6 images held out entirely as target (OOD) test.

**The critical detail:** domain differences for DDO's regularizer were computed against the **actual, unmodified 204-entry descriptor pool** — no Midjourney-specific descriptor added. That's the whole point: it reproduces exactly what happens when a model trained today, using LanCE's existing pool, meets a domain nobody curated the pool for. Protocol matched Phase 0 exactly: 50 epochs, batch 64, AdamW, lr 1e-4, weight_decay 1e-4.

| | Baseline (α=0) | +DDO (α=1) |
|---|---|---|
| Source (real photo) test accuracy | 73.00% | 72.76% |
| **Target (Midjourney v6) test accuracy** | **74.06%** | **74.74%** |
| DDO gain over baseline (target) | — | **+0.68 points** |
| *Phase 0 reference (CUB → Painting)* | *50.64%* | *57.04%* |
| *Phase 0 DDO gain* | — | ***+6.40 points*** |

Both conditions climb from near-chance (23 classes, ≈4.3% chance) into the low-mid 70s over 50 epochs, tracking each other closely — the final gap (+0.68 points) is roughly a **tenth** the size of Phase 0's own +6.40-point gain, measured under an identical protocol.

**A second, unplanned finding:** target (Midjourney) accuracy is not lower than source (real photo) accuracy in either condition — if anything it's slightly higher (74.06% vs. 73.00% baseline; 74.74% vs. 72.76% +DDO), despite Phase F3 measuring a near-zero alignment score (0.037) for this exact shift. A plausible explanation: Midjourney's outputs for common object categories tend to be clean, centered, prototypical renderings — possibly easier to classify than the visual clutter typical of real COCO photos.

**Honest interpretation:** this does not mean the model fails badly on a genuinely new domain — it doesn't; both conditions reach 72–75% on 23-way classification, far above chance. It means something narrower and more precise: **DDO's specific value-add over a plain classifier collapses almost entirely when its fixed descriptor pool has no term for the target domain.** Phase 0 showed DDO is worth +6.40 points when the pool covers the shift well. Phase E2 shows that same mechanism, run under identical conditions, is worth only +0.68 points when the pool has zero coverage (Phase F3/E1) — roughly a tenth of its value. This connects F3's two findings and this phase into one story: the descriptor list has no words for this domain → the words CLIP would need don't align well with real images of it either (0.037) → and the practical cost, measured directly rather than inferred, is that DDO stops adding meaningful value. It also reinforces Phase F2's earlier finding: a *trained* classifier still reaches strong absolute accuracy on this domain — training recovers the signal. What's specifically broken is DDO's zero-target-data, text-only mechanism for adding value *beyond* that trained baseline, not the model's ability to classify the new domain at all.

Figures: [`phase_e2_target_accuracy.png`](../results/figures/phase_e2_target_accuracy.png), [`phase_e2_ddo_gain_comparison.png`](../results/figures/phase_e2_ddo_gain_comparison.png), [`phase_e2_source_vs_target.png`](../results/figures/phase_e2_source_vs_target.png). Full write-up: [`results/phase_e2_defactify_ddo_training.md`](../results/phase_e2_defactify_ddo_training.md). Training script: [`external/LanCE/experiments/phase_e2_defactify_ddo_training.py`](../external/LanCE/experiments/phase_e2_defactify_ddo_training.py).

---

## §5 Datasets used, and why

| Dataset | Used in | Size | Released | Why this one |
|---|---|---|---|---|
| CUB-200-2011 → CUB-Painting | Phase 0 | 5,990 / 5,790 / 3,047 imgs | CUB: 2011 | The paper's own primary benchmark — the only dataset it reports numbers for directly |
| PACS | Phase B, B.1 | 9,991 imgs, 4 domains, 7 classes | 2017 | Smallest standard 4-domain benchmark — cheap to iterate on while building the harness |
| Office-Home | Phase D | 15,588 imgs, 4 domains, 65 classes | 2017 | Deliberately harder than PACS — built to test whether PACS's weak signal was real or ceiling-masked |
| EuroSAT | Phase F1, F2 | 27,000 imgs, 10 classes | 2017–2019 | OpenAI's own CLIP paper already publishes an independent anchor number for it |
| Defactify / MS-COCO-AI | Phase F3, E2 | 96,000 imgs (real + 5 generators) | Generators: Dec 2022–2024 | All 5 generators postdate CLIP's and GPT-3.5's training cutoffs by 15+ months, with real matched photo images |
| GenImage / Midjourney | Phase F4 | 928 imgs, 155 classes (partial) | Midjourney images: 2023 | Originally the ideal temporal-novelty test; revisited once a usable subset was located |

Two candidates were investigated during planning and rejected: **LADA-Sculpture** (hidden Google Drive / Baidu Netdisk dependency) and **AWA2** (13GB download disproportionate to what it would have tested). Full detail in [`planning/02-continual-dg-experiment-plan.md`](../planning/02-continual-dg-experiment-plan.md).

---

## §6 Engineering integrity: bugs found along the way

None of the following touch LanCE's actual method (the DDO loss or model architecture) — all are plumbing bugs in the paper's released code that silently blocked its own documented commands from running:

1. Missing `import os` in `data/__init__.py`
2. A stray leading `/` breaking a path join for the CUB-Painting target set
3. `--batch_size` / `--epochs` CLI arguments typed as `str` / `float` instead of `int`
4. A hardcoded dummy concept-label shape (77) that didn't match the actual concept bank (312)
5. A copy-paste bug passing the wrong dictionary when building the target test set
6. The dataset loader's return tuple silently returning a `DataLoader` in the slot meant for the raw `Dataset` object

A seventh bug was found in this project's own new code, not the paper's: the embedding-caching data loader's `num_workers=8` setting silently produced identical cached embeddings for every image once a split needed enough worker-dispatched batches — caught before Phase B's results were finalized. All affected caches were rebuilt and verified row-by-row.

Separately, the paper's own code recomputes CLIP image embeddings from scratch every epoch even though CLIP is frozen throughout training. Precomputing and caching those embeddings once cut epoch time from roughly 7–14 minutes down to about 3 seconds — this is the pipeline used for every experiment from Phase A onward ([`cache_utils.py`](../external/LanCE/cache_utils.py), [`train_cached.py`](../external/LanCE/train_cached.py)).

---

## §7 Limitations that apply across the whole project

- **No seed sweeps.** Every phase reports a single run per condition. Where BWT differences between conditions are small (e.g. −0.07 vs. −1.24 in B.1), run-to-run noise hasn't been ruled out as an explanation.
- **Partial domain-ordering coverage.** Phases B, B.1, and D each tested 3 of the 24 possible orderings for a 4-domain sequence — enough to establish the phenomenon is order-dependent, not enough to characterize its full distribution.
- **First-pass concept banks.** The PACS, Office-Home, EuroSAT, and Defactify concept banks are hand-written, first-draft lists (roughly 4 concepts per class) rather than curated or validated against alternatives.
- **EWC's penalty strength was never tuned** (λ=1000 throughout) — its accuracy cost relative to cumulative DDO and replay may partly be a hyperparameter artifact.
- **Two phases (F1, F4) substitute CLIP text for a real photo reference** where a matched photo dataset wasn't readily available — Phase F4's own three-way comparison shows this likely inflates the measured alignment score relative to Phase F3's real-photo version.
- **Phase F4's dataset was only partially recoverable** (155 of 1,000 classes, ~6 images/class) due to a multi-part archive download issue.
- **Phase E1's pool check inspects the already-generated, checked-in descriptor pool** rather than a fresh GPT-3.5-turbo API call — deliberate (see §4), but it trusts that file is representative of what GPT-3.5 would produce if re-prompted today.
- **Phase E2 tested one target generator** (Midjourney v6) and both training curves were still rising slightly at epoch 50 — kept fixed to match Phase 0's protocol exactly rather than training longer, which would have broken the direct comparison.

---

## §8 What's genuinely still open

- **A complete GenImage download.** Phase E2 ran the real baseline-vs-+DDO training test on Defactify instead; a full GenImage download would still be useful as a second, independent dataset for the same comparison.
- ~~A descriptor-set staleness test (planned, never run)~~ — **done**, as Phase F3's pool inspection (0/20 AI-generation terms found) and Phase E2 (the real training-accuracy cost: DDO's gain shrinks from +6.40 to +0.68 points). Remaining open extension: test generators beyond Midjourney v6, and re-verify against a fresh GPT-3.5 API call rather than the checked-in pool alone.
- **Failure Mode 4** (static concept bank, fixed output layer) — deliberately deferred throughout, to keep the domain-only forgetting results uncontaminated by a simultaneously changing class set.
- **A tuned EWC sweep on Office-Home** — already run once at an untuned λ=1000; a swept value could change how it compares to cumulative-DDO and replay.
- **A literature-updated pass** — the literature-gap table in [`docs/lance_continual_dg_failure_analysis.md`](lance_continual_dg_failure_analysis.md) §3 wasn't re-searched during this round.

---

## §9 Bottom line

**Pillar 1:** LanCE has no built-in mechanism to survive domains arriving over time. Forgetting is real, and a harder benchmark (Office-Home) shows it consistently across every domain ordering tested, not just an occasional one — the easier PACS benchmark had been masking the effect, not disproving it. Off-the-shelf continual-learning fixes help substantially but don't fully close the gap on the harder benchmark.

**Pillar 2:** Independent of forgetting, LanCE leans on two separately-frozen things, and both have real gaps. The **CLIP backbone** has a large-margin coverage gap for domains it wasn't trained to represent, which gets dramatically worse for domains that didn't exist yet when CLIP finished training (Phases F1/F3/F4, alignment collapsing to as low as 0.037). Separately, the **GPT-3.5-written descriptor list** DDO regularizes against contains zero words for any post-cutoff AI-generation domain at all (Phase F3/E1) — and a real trained-model comparison shows this has a direct, measurable cost: DDO's benefit over a plain classifier shrinks from +6.40 points (a domain the list covers) to +0.68 points (one it doesn't), roughly a tenth (Phase E2). Importantly, this is a narrower failure than "the model breaks" — trained accuracy on the new domain stayed strong (74.7%) in both experiments; what specifically disappears is DDO's own added value over training alone.

Together, these results support the project's starting argument: LanCE is a strong method for the single-training-run setting it was built and evaluated for, but none of its three frozen components — the classifier, the CLIP backbone, or the descriptor-generating LLM — has a way to keep up once domains stop arriving all at once. Three null or reframed results along the way (Phase A, Phase F2, and the "target isn't depressed" nuance in Phase E2) were reported honestly rather than adjusted to fit, and each one sharpened the argument rather than weakening it by locating more precisely where each failure mode actually bites.

---

*Every number in this report is drawn directly from the corresponding `results/phase_*.md` write-up and its accompanying `results/*.json` data — no figures were estimated or extrapolated. Base method: Zeng et al., "Explaining Domain Shifts in Language: Concept Erasing for Interpretable Image Classification" (LanCE), CVPR 2025, [arXiv:2503.18483](https://arxiv.org/abs/2503.18483).*
