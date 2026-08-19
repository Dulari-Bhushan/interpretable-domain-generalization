# Plan 03: does it matter where concept scores come from?

## 0. Where this comes from, and what it is

This plan came out of a direct question raised while reviewing the architecture diagram: right now, "does this image show concept X" is answered by comparing a CLIP image embedding to a CLIP text embedding of the concept's name — a similarity trick, not a direct look. The proposal: replace that step with something that looks at the image and answers the question more directly, either (a) a classic, directly-trained concept classifier, or (b) a pretrained open-vocabulary detector that's told the concept list and points at what it finds. The concept list itself stays exactly as it already is (hand-written or LLM-generated) — only *how the model decides a concept is present* changes.

**Read `docs/new_methodology_report.md` first if you haven't** — this plan assumes Component 1 (the exact incremental classifier) already exists and works, and reuses it unchanged. This plan does not replace that report; it's a new, parallel thread investigating a different piece of the architecture (concept activation), not the classifier update.

**Honest framing, stated up front, matching how every other piece of this project has been framed:** neither variant here is new machinery. Directly-trained CBMs are the original 2020 concept-bottleneck-model design (Koh et al.), predating CLIP-based CBMs entirely. Detector-grounded concept checking already exists too — [VLG-CBM](https://proceedings.neurips.cc/paper_files/paper/2024/file/90043ebd68500f9efe84fedf860a64f3-Paper-Conference.pdf) pairs each concept with a bounding box from an open-vocabulary detector for exactly this reason (stopping a model from claiming a concept is present when it isn't actually visible). What's being tested here is whether either approach changes anything *in our specific setting* — continual domain-incremental arrival, on top of Component 1's exact classifier, on datasets where CLIP's own similarity trick has already been measured to struggle (Pillar 2 of the original diagnosis).

---

## 1. The critical technical wrinkle this plan has to handle: what happens to DDO

This has to be addressed before any experiment design, because it changes what's possible at each stage.

DDO's whole mechanism — simulating what a new domain would do to concept scores, using only text, with zero real images — works *because* real image-derived concept activations and text-derived "simulated" activations are computed the same way, through the same CLIP concept embeddings, and therefore live in the same space. That's not incidental; it's the entire reason the base method can say anything about a domain it's never seen a picture of.

**The moment real concept activations stop coming from CLIP's similarity trick, that shared space breaks.** A directly-trained classifier's concept scores and a detector's confidence scores don't live in the same geometry as CLIP-text-derived "simulated domain shift" vectors — comparing them wouldn't mean anything.

So, for any experiment below where concept activations come from something other than CLIP similarity, DDO has to be handled one of two ways:
- **Drop it entirely** for that run, and compare plain classification accuracy (with Component 1's exact update, no DDO term) against the plain-baseline (no-DDO) numbers already on record from Phase 0/E2 — a fair, like-for-like comparison.
- **Replace its input** with a real, measured domain-shift vector computed in the *new* concept space (this is exactly what Idea 3 — self-diagnosing grounding — already does: measure the real shift from a small batch of images instead of trusting a text-only guess). This keeps an orthogonality-style regularizer in play, just grounded in reality instead of text.

Every stage below states explicitly which of these two applies.

---

## 2. The three concept-activation variants being compared

| Variant | How it decides a concept is present | Needs labeled training data? | Status |
|---|---|---|---|
| **A — CLIP similarity (existing, baseline)** | Compare CLIP image embedding to CLIP text embedding of the concept name | No | Already built, already the basis of every prior result in this project |
| **B — Directly-trained CBM** | A classifier (ResNet-50, ImageNet-pretrained, matching the original 2020 CBM paper's own setup) fine-tuned to predict each concept from real labeled examples | **Yes** — real per-image concept labels | Not built. Blocked on labeled data for most of our datasets (see §4) |
| **C — Detector-grounded** | A pretrained open-vocabulary detector (Grounding DINO or OWL-ViT) given the concept list as text prompts, scored by detection confidence | No — pretrained, zero-shot on our data | Not built. The more practical variant — no labeling blocker |

Variant C is the one to build first: it removes the labeled-data blocker that stalls Variant B on every dataset except CUB, and it's the version closest to what was actually proposed (a model good at picking out distinct elements in an image, not a from-scratch trained classifier).

---

## 3. The train/test protocol — answering "do we train on everything?"

Directly answering the question that prompted this plan: **no.** Training on every domain up front would defeat the point — real deployments never get that option either, since a domain that doesn't exist yet obviously can't be pre-trained on.

**The protocol, precisely, matching how Component 1 was already validated:**

1. Pick a domain order. Train sequentially, one domain at a time, no replay of earlier domains' raw data (Component 1's exact update means "no replay" doesn't cost anything here — see `docs/new_methodology_report.md`).
2. After every stage, evaluate on two categorically different kinds of domains:
   - **Domains already trained on** — this is the forgetting check (BWT), exactly as Phase B/D measured it.
   - **A domain never trained on at all, held out entirely** — this is the honest generalization check. This is where a domain like medical imaging belongs *before* it's ever incorporated: point the current model at it cold and see what happens, no training on it yet.
3. **The trigger point:** if step 2's held-out-domain result is bad (matching what Pillar 2's diagnosis already found for CLIP similarity), that's exactly when Components 2/3 (the trust check, the growing vocabulary) are supposed to activate — the model notices it's out of its depth, and only *then* does that domain get incorporated into training, becoming a "seen" domain for every stage after.

This protocol doesn't change based on which concept-activation variant (A/B/C) is being used — it's the same experimental scaffold either way. What changes is which numbers come out of it.

---

## 4. Staged experiment plan

### Stage 1 — Does the concept extraction even work? (CUB, no domain shift)

**Why CUB specifically:** it's the only dataset in this project with real, human-labeled concept ground truth (312 attributes/image) — not a CLIP guess, not something we made up. It's also the base LanCE paper's own primary benchmark (Phase 0), already fully set up and cached.

**What this stage is *not*:** it is not a domain-shift test. No continual learning, no BWT, no multiple domains. It's a calibration step — does Variant C's output agree with reality at all — before spending effort on the harder stages.

**Protocol:**
1. Run the detector (Grounding DINO or OWL-ViT) over CUB's images, using CUB's real 312 concept names as the text prompts.
2. For each image, record the detector's confidence score per concept.
3. Compare against CUB's real labels: precision/recall per concept, and overall agreement.
4. (Optional, only if Stage 1 alone looks promising and there's appetite for the labeling cost) Also train Variant B (a directly-supervised ResNet-50 CBM) on CUB's real labels, as a second comparison point — this is the one dataset where Variant B is actually possible without new annotation work.

**No DDO here at all** — this stage has no domain shift, so the wrinkle in §1 doesn't apply yet.

**What would make this worth continuing:** the detector's concept scores need to correlate meaningfully with the real labels. If they don't, Variant C isn't a viable replacement anywhere, and this whole plan stops here rather than compounding a bad concept extractor into more expensive experiments.

### Stage 2 — Does it change anything under domain shift? (PACS, Office-Home)

**Why these two:** the exact benchmarks Component 1 was already validated on — reusing them means every number here is directly comparable to results already on record, not a new baseline to establish from scratch.

**Protocol:**
1. Swap Variant C in for Variant A as the concept-activation source, keeping everything else identical: same domain orderings as Phase B/D, same classes, same Component 1 exact-update classifier.
2. Per §1, since Variant C's concept space no longer matches CLIP-text-derived domain-shift vectors: run this **without DDO**, comparing plain accuracy/BWT against the existing no-DDO baseline numbers (Phase 0/B's `alpha=0` runs), not the +DDO numbers.
3. Compare directly against the existing Variant A + Component 1 results (`results/component1_pacs_results.json`, `results/component1_officehome_results.json`): does BWT change? Does final accuracy change?

**What a meaningful result looks like either way:** if accuracy matches or beats Variant A's no-DDO baseline, detector-grounded concepts are at least as good as CLIP similarity on domains CLIP already understands well (PACS/Office-Home are all pre-2021 art styles — CLIP's comfort zone) — a reasonable, unsurprising outcome, useful mainly as a sanity check that swapping the mechanism didn't break anything. If it's meaningfully *worse*, that's evidence CLIP's similarity trick, weak as it is on some domains, is still hard to beat when it's already comfortable — worth knowing either way.

### Stage 3 — Does it help exactly where CLIP was shown to struggle?

**This is the stage that actually tests the motivating idea.** Everything before this is groundwork.

**Datasets, in order of readiness:**

| Dataset | Status | Why it's here | What CLIP-similarity scored (from the original diagnosis) |
|---|---|---|---|
| EuroSAT | Have, ready now | Modality-scarce case (satellite imagery) | Alignment 0.32 vs. paper's 0.90–0.99 range; zero-shot accuracy ~60–64% |
| Defactify / MS-COCO-AI | Have, ready now | Temporal-novelty case (post-2021 AI generators) | Alignment 0.037 — the lowest score measured anywhere in this project |
| Camelyon17 (medical) | Blocked — see `docs/new_methodology_report.md` | The case the base paper's own authors admit was never tested | Not yet measured (never tested — that's the point) |

**Protocol, per dataset:**
1. **Cold, held-out test first** (per §3's protocol) — a model trained on PACS/Office-Home-style domains, never trained on this one, evaluated on it directly. Run this with both Variant A (CLIP similarity — reproducing/extending Phase F1-style measurement) and Variant C (detector-grounded), so there's a direct, matched comparison rather than comparing against an old number computed a different way.
2. **Then incorporate it** — per the §3 trigger-point logic, treat it as a newly-arriving domain and run it through the same sequential-training protocol as Stage 2, again both with Variant A and Variant C.
3. **Report both accuracy and the DDO-style gain** where applicable — mirroring Phase E2's exact comparison (baseline vs. +something, on the same held-out target), so the "how much did this help, precisely" number is stated the same way as every other result in this project, not a new metric invented just for this.

**What a meaningful result looks like:** if Variant C's cold, held-out accuracy on EuroSAT/Defactify/Camelyon17 is measurably higher than Variant A's, that's real evidence the detector-grounding approach helps exactly where it was predicted to — the actual point of doing any of this. If it's the same or worse, that's also a real, useful, reportable finding (per this project's running commitment to report null results honestly, same as Phase A and the "target isn't depressed" nuance in Phase E2).

---

## 5. Metrics, per stage — so results stay comparable to everything already on record

| Stage | Metric(s) | Matches the convention from |
|---|---|---|
| 1 (CUB) | Precision/recall per concept vs. real labels | New to this plan — no prior convention to match, since no earlier phase had real concept labels to check against |
| 2 (PACS/Office-Home) | ACC, BWT, max-diff-from-joint | Phase B/D, Component 1 |
| 3, cold-test | Alignment-style score / zero-shot-equivalent accuracy | Phase F1/F3/F4 |
| 3, incorporated | Accuracy gain over baseline (with vs. without the regularizer term) | Phase E2 |

---

## 6. Risks and open questions

- **Detectors are built for localizable things** — a beak, a tail, a wing — and weaker on whole-image texture or behavior ("glossy fur," "a wagging tail"). Some fraction of any concept list will suit this approach better than others; Stage 1's per-concept precision/recall breakdown should surface which kinds fail, not just an overall number.
- **Confidence-score-to-activation-magnitude mapping is undefined so far.** A detector returns a confidence per phrase; the existing pipeline expects a continuous concept-activation vector. Whether to use raw confidence, a calibrated probability, or a thresholded presence/absence needs a decision before Stage 1 can run, and different choices could change the results meaningfully — worth trying more than one and reporting which was used.
- **Dropping DDO for Stage 2/3 (per §1) means losing the specific "does the descriptor pool cover this domain" story** that Components 3/4 are built around — those components assume CLIP-text-based domain descriptors are still in play. If Variant C ends up being the preferred concept-activation source going forward, Components 3/4 would need rethinking, not just a drop-in swap. Flagging this now so it doesn't get discovered as a surprise later.
- **This entire plan could come back negative at Stage 1** — if the detector's concept scores don't track CUB's real labels well, that's a legitimate, useful stopping point, not a failure of the plan itself.

---

## 7. Suggested order of work

1. **Stage 1 (CUB)** — cheapest, fastest, and it's a real gate: if this doesn't show the detector's concepts are trustworthy, nothing downstream is worth running yet.
2. **Decide the confidence-to-activation mapping** (§6) — needed before Stage 1's numbers can be trusted, not after.
3. **Stage 2 (PACS/Office-Home)** — sanity check that swapping the mechanism doesn't break anything on domains CLIP already handles well.
4. **Stage 3, cold-test first, on EuroSAT and Defactify** — both already downloaded, no blockers, and this is where the actual motivating question gets answered.
5. **Stage 3, Camelyon17** — once the dataset download is unblocked (`docs/new_methodology_report.md` §2.2) or a substitute is in hand.
6. **Variant B (directly-trained CBM) on CUB only**, as a secondary comparison point — optional, lower priority than the above, since Variant C already answers the main question without the labeling cost.
