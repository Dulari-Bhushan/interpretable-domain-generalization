# Why LanCE won't survive continual domain generalization

Source paper: Zeng, Su, Sun, Wen, Zhang, Wang, Chen, Liu, Ma — *"Explaining Domain Shifts in Language: Concept Erasing for Interpretable Image Classification"* (LanCE), CVPR 2025. [arXiv:2503.18483](https://arxiv.org/abs/2503.18483) · [code](https://github.com/joeyz0z/LanCE)

This document supports the presentation for **"CBMs for Continual Domain Generalization."** It was originally written as a paper-analysis document with predicted failure modes, each flagged as a prediction to verify. **That verification is now done** — Phases 0 through D (Pillar 1: forgetting) and Phases F1/F2 (Pillar 2: backbone coverage) have run, and every "what to verify empirically next" item below has been replaced with an actual measured result, traced to a specific file under `results/`. Where a prediction turned out wrong or only partially right, that's reported here exactly as honestly as it is in the underlying results docs — several did.

---

## 1. What LanCE actually does

LanCE is a **plug-in regularizer** for CLIP-based concept bottleneck models (CBMs). A CBM maps an image to a vector of concept activations `a` (cosine similarity between CLIP image embedding and CLIP text embeddings of human/LLM-written concepts like "red color," "waxy texture"), then a linear layer `W_F` maps `a → ŷ` (Eq. 5–8).

LanCE's insight (Sec. 3): CLIP's embedding space aligns *visual* domain shifts with *textual* domain descriptions — e.g. the embedding difference between a sketch of an apple and a photo of an apple lands close to the text embedding of "a sketch of an apple" minus "a photo of an apple." They exploit this two ways:

1. **Generate domain descriptors once.** Prompt GPT-3.5-turbo one time: *"Please list visual domains in short phrases as much as possible"* → a fixed list of 200 descriptors `P` (sketch, clipart, painting, sculpture, 3D model, watercolor, pixel art, ... — Appendix C.1, Fig. 9). This happens **before training** and is never revisited.
2. **Erase domain-specific concepts.** For each descriptor and class, simulate a "domain-specific concept activation" `a_sp(p,y)` (Eq. 12) from the language-guided domain shift, then add a loss term (**DDO**, Eq. 13) that pushes `W_F` toward orthogonality with all these simulated activations. Final objective: `L = L_CE + λ·L_DDO` (Eq. 14), optimized **once**, jointly, with Adam, on a single training domain `D_train` (Sec. 4.1).

The entire pipeline — concept bank, domain descriptor set, CLIP backbone, classifier weights — is fixed after this one training run. There is no notion of time, sequence, or update in the method as described.

---

## 2. Five failure modes under continual DG — now with measured evidence

### Failure Mode 1 — Closed-world, frozen domain-descriptor set
**Mechanism:** `P` (200 descriptors) is generated once (Sec. 4.3) and used identically for every dataset. The DDO loss (Eq. 13) is an expectation over `(p_i, y) ~ P × Y` — it can only erase influence from domains *representable* in `P`.

**Why it breaks continual DG:** In a continual setting, domains arrive as a stream, including ones nobody could enumerate at t=0. Since `P` is frozen at training time, any genuinely new domain gets no orthogonality protection unless someone manually re-prompts the LLM and retrains from scratch.

**Evidence (the paper's own ablation):** Table 2 / Table 8 split the 200 descriptors into "domain-relevant" vs. "domain-irrelevant" and show `+DDO(IR)` consistently underperforms `+DDO` (full/relevant set) — e.g. LADA-Sculpture OOD: 77.70 (IR) vs 80.00 (full). Fig. 5 shows accuracy gains diminish sharply past ~100–200 descriptors.

**Our own empirical test (Phase A, `results/phase_a_descriptor_coverage.md`): the predicted dose-response didn't materialize.** We extended their binary relevant/irrelevant split into a continuous one — ranking all 204 descriptors by similarity to the true "painting" domain shift, then progressively excluding the most-similar ones. Accuracy stayed flat (55.9%–57.1%) whether the pool contained a near-exact descriptor match or only the 24 *least* similar ones. Likely explanation: DDO's regularizer operates in 311-dim concept-activation space, not raw embedding space, which may equalize descriptors that look different as text but land similarly once projected through the shared concept vocabulary. This doesn't mean the closed-world assumption is false — it means it bites at a different level than we tested (see Failure Mode 3's update below, not "the exact right style is missing").

**Strength: Confirmed by paper's own numbers, but our own extension gave a null result.** The closed-world critique survives at the "wrong modality entirely" level (Failure Mode 3), not clearly at the "wrong specific style within a covered family" level (this test).

---

### Failure Mode 2 — No incremental update path; retraining needs the original data
**Mechanism:** Eq. 14 is a single joint objective solved once over `D_train`. Nothing in Sec. 4 defines an update rule for after deployment.

**Why it breaks continual DG:** To add a newly-arrived domain without replaying old data, you get **catastrophic forgetting** of the orthogonality constraints learned for earlier domains — a well-documented failure mode in this exact model family (CONCIL, CI-CBM built dedicated anti-forgetting machinery for precisely this reason). LanCE has no analog of this machinery.

**Our own empirical test (Phases B/C/D — the core experiment of this whole project):**
- **Phase B (PACS, 4 domains, 7 classes, `results/phase_b_domain_il.md`): real but narrow.** Naive sequential fine-tuning showed clear forgetting (BWT −8.30) in only 1 of 3 domain orderings tested; the other two stayed within ~1 point of the joint/oracle upper bound (98.29% ACC). PACS's near-ceiling accuracy on 7 easy, visually-distinct classes left too little headroom for forgetting to show up reliably.
- **Phase C (same PACS harness, three remediations, `results/phase_c_remediation.md`): textbook fixes work — mostly.** Cumulative DDO and cached-embedding replay closed the one real forgetting case almost entirely (BWT −8.30 → −0.07 and −0.96) at little-to-no accuracy cost. EWC also drove BWT near zero but cost 7–12 points of accuracy in two orderings — the classic stability-plasticity tradeoff.
- **Phase D (Office-Home, 4 domains, 65 classes, `results/phase_d_officehome.md`): the strongest evidence.** On a harder, less-saturated benchmark (90.78% joint ACC, not 98.3%), **every one of the 3 domain orderings tested showed real, consistent negative BWT (−0.68 to −4.68)** — not just one out of three. The same "near-free" remediations that nearly fully fixed PACS only *partially* closed the gap here (cumulative DDO/replay left −1.6 to −3.3 BWT behind in the harder orderings).

**Strength: Confirmed empirically, with the effect size and consistency scaling directly with how much headroom the benchmark leaves.** PACS understated the effect (masked by an easy task); Office-Home reveals it consistently. It's real, moderate (not catastrophic near-chance collapse — the worst case still leaves the model at 87.8% ACC, a few points below oracle), and increasingly resistant to the cheapest fixes as the benchmark gets harder. This is now the best-evidenced failure mode in the whole analysis, not the "top candidate for follow-up" it was before — it's been followed up.

---

### Failure Mode 3 — Frozen CLIP breaks the core premise outside its pretraining distribution
**Mechanism:** Everything rests on the Sec. 3 empirical claim that CLIP's embedding space aligns visual and textual domain shifts. This was demonstrated (Fig. 2, Fig. 8) only on domains heavily represented in CLIP's web-scraped pretraining corpus. The 200-item descriptor list (Fig. 9) is *entirely* conventional art/media styles — no scientific or industrial imaging modality appears.

**Evidence — the authors admit this themselves.** Appendix G, verbatim: *"Our method highly depends on pre-trained VLMs like CLIP and LLMs like GPT-3.5. However, these models are limited in application to some professional fields like medical treatments."*

**Our own empirical test (Phases F1–F4): confirmed with a large margin across four datasets, refined once, and refined again on methodology.**
- **Phase F1 (`results/phase_f1_eurosat_alignment.md`): mean domain-shift alignment score = 0.32** for a photo→satellite shift on EuroSAT, vs. the paper's own reported 0.90–0.99 range — no overlap at all. Important caveat discovered while running this: EuroSAT (2017–19) actually *predates* CLIP (2021), so this specifically tests **modality scarcity** (satellite imagery rare in captioned web photos), not temporal novelty. Computed the same way the paper's own Fig. 2 does, with one documented adaptation (CLIP text as photo-domain proxy, since no matched real-photo dataset was used).
- **Phase F2 (`results/phase_f2_eurosat_ceiling.md`), the more interesting finding: the hypothesis, as literally stated, was wrong.** We predicted a trained concept-bottleneck classifier would be capped near CLIP's zero-shot ceiling (~60%). Instead it reached **90.9%** — far closer to the 98.1% linear-probe ceiling. The reason clarifies rather than undermines the failure mode: the *visual representation* is fine (training recovers most of the signal), but *zero-shot, training-free* alignment is what's specifically broken — and DDO's mechanism (Eq. 12) is exactly a zero-target-data, text-only simulation with no access to the recovery path a trained classifier has.
- **Phase F3 (`results/phase_f3_temporal_novelty.md`), the genuine temporal-novelty test: confirmed even more dramatically.** Dataset: Defactify/MS-COCO-AI, 5 generators (Stable Diffusion 2.1/XL/3, DALL-E 3, Midjourney v6) all released 15 months to 2.5+ years after **both** CLIP's (~2020/2021) and GPT-3.5's (~Sept 2021, what LanCE's 200-descriptor list was generated from) training cutoffs. Mean alignment: **0.05** (mixed-content) / **0.037** (per-class-controlled — both agree, ruling out a methodology artifact) — below even F1's already-large-margin EuroSAT result, effectively near zero. Also closes F1's adaptation gap: this dataset provides real matched photo images, not a text stand-in.
- **Phase F4 (`results/phase_f4_genimage_alignment.md`), a third independent dataset, revealing a methodology-dependent nuance.** GenImage/Midjourney (only partially downloadable — a multi-part archive, documented in full in the results file — 155 of 1,000 ImageNet classes recovered). Mean alignment: **0.23**, well below the paper's range but *not* as low as Phase F3. Comparing all three: the two tests using CLIP text as a photo-domain stand-in (F1: 0.32, F4: 0.23) cluster together, while the one test with real matched photos (F3: 0.037) is an order of magnitude lower — meaning the text-proxy methodology itself likely inflates the score, and F3's result should be weighted as the most trustworthy of the three, not averaged naively with the other two.

**Strength: Confirmed empirically, large margin, on two independent sub-cases (modality scarcity and temporal novelty), with the mechanism precisely located (F2) rather than just asserted.** This remains the most *fundamental* failure mode — not fixable by more descriptors or better continual-learning tricks to the classifier, because the ceiling is set by what DDO's text-only mechanism can predict, not by classifier capacity. Temporal novelty (F3) breaks that prediction even harder than modality scarcity (F1) does.

---

### Failure Mode 4 — Static concept bank and fixed-size output layer
**Mechanism:** The concept set `C = {c_i}_{i=1}^M` is built once; `W_F` is a fixed `M → N_y` linear map.

**Why it breaks continual DG:** If new domains bring new relevant concepts, or continual DG is coupled with new classes appearing over time, the fixed `M` and `N_y` can't absorb them without architecture surgery + retraining — reopening Failure Mode 2's forgetting problem.

**Status: not empirically tested this round.** This was explicitly deferred in `planning/02-continual-dg-experiment-plan.md` ("Joint Domain-IL + Class-IL... mixing two axes of failure right now would blur which one causes what") to keep Phase B/C/D's forgetting results clean and attributable to domain shift alone, not confounded with a simultaneously-changing label space. Still architecturally true by inspection and corroborated by CI-CBM's own existence as a paper built to solve exactly this problem — but remains a prediction, not a measured result, and should be presented as such.

**Strength: Moderate — architectural, not paper-tested, deliberately not tested by us either (documented scope decision, not an oversight).**

---

### Failure Mode 5 — Structural domain gaps persist even with full descriptor coverage
**Mechanism/Evidence:** Sec. 5.3, verbatim: *"all models' generalization performance to sculpture and 3D model images on the LADA-Sculpture and LADV-3D benchmark is limited... even with DDO."* (e.g. LADV-3D: 99.93% ID vs 68.01% OOD).

**Why it matters for continual DG:** If a well-covered, explicitly-anticipated domain shift still leaves a ~32-point gap, degradation should compound for domains with zero descriptor coverage and zero CLIP pretraining exposure — reinforcing Failure Modes 1 and 3.

**Status: not independently re-tested, but corroborated in spirit by our own results** — Phase D showed a harder, less-saturated benchmark reveals gaps PACS's easier task hid, and Phase F1 showed an out-of-distribution modality gap far larger than anything the paper itself reports even for its own worst-case (2D→3D) domain.

**Strength: Strong (direct paper quote + numbers), corroborating evidence for #1/#3 rather than independently re-verified here.**

---

## 3. Literature check — confirming the gap is real (not assumed)

Verified via search + abstract checks (unchanged from the original analysis — no new literature search was part of the executed experimental plan):

| Cluster | Representative work | What it actually solves | Why it's not "CBM for continual DG" |
|---|---|---|---|
| CBM + continual learning | [Language Guided CBMs for Interpretable CL](https://arxiv.org/abs/2503.23283) (CVPR'25), [CONCIL](https://arxiv.org/abs/2411.17471), [CI-CBM](https://arxiv.org/pdf/2604.14519) | New **classes/concepts** arriving over time, within one visual domain | No domain shift at all — same domain throughout |
| CBM + domain generalization | LanCE (this paper), PCBM, LaBO | Generalizing to unseen domains, **single-shot** train-once | No streaming/sequential aspect — literally a one-time fit |
| CBM + temporal drift | [Tree of Concepts](https://arxiv.org/pdf/2604.17089) | Non-stationary **tabular clinical** data over time | Not visual, not discrete style-domains, single continuously-drifting distribution rather than distinct domains |
| Continual domain-incremental (no CBM) | [Domain Generalizable Continual Learning (DGCL)](https://arxiv.org/abs/2510.16914) (Oct 2025) | Sequential single-domain tasks, generalize across all encountered domains | Closest structural analog, but black-box (no interpretability, no CLIP-language descriptor mechanism) |
| Concept-based domain adaptation | CUDA (ICML'25) | Aligning concepts using **target-domain data** | Adaptation, not generalization to *future unseen* domains — an easier problem (target data available) |

**Conclusion:** No existing work sits at the intersection of (concept-based interpretability) × (generalization to *unseen future* domains) × (continual/streaming domain arrival without forgetting) × (dynamic, language-guided descriptor updating). That intersection is the white space this project's results now support with measured evidence, not just a literature-absence argument.

---

## 4. What we verified empirically

The four items originally listed as "predictions to test" — here's what actually happened, each traced to its results file:

1. **Coverage-gap test (extended Table 2 into a continuous dose-response). Result: null.** `results/phase_a_descriptor_coverage.md`. Predicted accuracy would collapse as the descriptor pool got less similar to the true domain; it stayed flat (55.9–57.1%) even with the pool stripped to the 24 least-similar descriptors. Reprioritized rather than confirmed — pushed the closed-world critique toward Failure Mode 3 instead.
2. **Out-of-CLIP-distribution test. Result: confirmed on four datasets, with a refinement, and a methodology caveat.** `results/phase_f1_eurosat_alignment.md`, `results/phase_f2_eurosat_ceiling.md`, `results/phase_f3_temporal_novelty.md`, `results/phase_f4_genimage_alignment.md`. Domain-shift alignment for photo→satellite (EuroSAT, modality scarcity) scored 0.32 vs. the paper's 0.90–0.99 range. A trained concept-bottleneck classifier reached 90.9%, not the ~60% we predicted — clarifying that it's zero-shot alignment specifically, not the visual backbone, that's the bottleneck. Testing temporal novelty directly on two independent datasets: Defactify/MS-COCO-AI (5 generators released after both CLIP's and GPT-3.5's cutoffs, real matched photos) scored 0.037 — near zero; GenImage/Midjourney (155 ImageNet classes, CLIP-text photo proxy) scored 0.23. All four numbers confirm the qualitative finding, but the two text-proxy tests (EuroSAT, GenImage) cluster well above the one real-photo test (Defactify) — the text-proxy methodology itself appears to inflate the score, so Defactify's near-zero result should be weighted as the most trustworthy.
3. **Sequential-domain forgetting test. Result: confirmed, with effect size scaling with benchmark difficulty.** `results/phase_b_domain_il.md`, `results/phase_c_remediation.md`, `results/phase_d_officehome.md`. Weak/order-dependent on PACS (1 of 3 orderings), consistent and only partially remediable on Office-Home (3 of 3 orderings). The clearest, best-evidenced result of the whole project.
4. **Descriptor-set staleness test. Result: not run.** No experiment tested whether `P` (frozen at GPT-3.5-turbo's training-time knowledge) fails against domain descriptions a newer LLM would generate but GPT-3.5 wouldn't. Still an open prediction, out of scope for this round — flagged honestly rather than silently dropped.

---

## 5. Presentation framing

With measured results in hand, lead with the two best-evidenced findings, not the paper's own numbers: **Phase D's Office-Home result** (real, consistent forgetting across every domain ordering tested, only partially fixed by textbook remediations) as the continual-learning core claim, and **Phase F1's 0.32-vs-0.90–0.99 alignment gap** as the fundamental-ceiling claim, sharpened by **Phase F2's** precise localization of exactly which sub-mechanism breaks (zero-shot alignment, not raw visual representation). Use Failure Mode 1's own paper-cited numbers (Table 2/8) as corroborating context, not the headline — our own Phase A extension of that test came back null, so lean on the paper's original ablation there rather than our replication. Close by being upfront about what didn't hold up as predicted (Phase A's flat dose-response, Phase F2's wrong initial hypothesis) — the honest arc ("we predicted dramatic failure everywhere; what we found is narrower, more precise, and in the cases that matter most, still real") is a stronger, more defensible story for a CL-literature-savvy audience than a version that only reports the confirmations.
