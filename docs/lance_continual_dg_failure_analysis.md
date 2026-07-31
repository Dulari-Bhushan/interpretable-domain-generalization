# Why LanCE won't survive continual domain generalization

Source paper: Zeng, Su, Sun, Wen, Zhang, Wang, Chen, Liu, Ma — *"Explaining Domain Shifts in Language: Concept Erasing for Interpretable Image Classification"* (LanCE), CVPR 2025. [arXiv:2503.18483](https://arxiv.org/abs/2503.18483) · [code](https://github.com/joeyz0z/LanCE)

This document supports the first presentation for **"CBMs for Continual Domain Generalization."** Every claim below is traced to a specific equation, section, table, or quoted sentence from the paper (I extracted and read the full 17-page PDF, including all appendices), or to an abstract-verified external paper. Where I predict a failure rather than cite a measured one, it's flagged explicitly — those are the things to verify empirically next, not conclusions to present as fact.

---

## 1. What LanCE actually does

LanCE is a **plug-in regularizer** for CLIP-based concept bottleneck models (CBMs). A CBM maps an image to a vector of concept activations `a` (cosine similarity between CLIP image embedding and CLIP text embeddings of human/LLM-written concepts like "red color," "waxy texture"), then a linear layer `W_F` maps `a → ŷ` (Eq. 5–8).

LanCE's insight (Sec. 3): CLIP's embedding space aligns *visual* domain shifts with *textual* domain descriptions — e.g. the embedding difference between a sketch of an apple and a photo of an apple lands close to the text embedding of "a sketch of an apple" minus "a photo of an apple." They exploit this two ways:

1. **Generate domain descriptors once.** Prompt GPT-3.5-turbo one time: *"Please list visual domains in short phrases as much as possible"* → a fixed list of 200 descriptors `P` (sketch, clipart, painting, sculpture, 3D model, watercolor, pixel art, ... — Appendix C.1, Fig. 9). This happens **before training** and is never revisited.
2. **Erase domain-specific concepts.** For each descriptor and class, simulate a "domain-specific concept activation" `a_sp(p,y)` (Eq. 12) from the language-guided domain shift, then add a loss term (**DDO**, Eq. 13) that pushes `W_F` toward orthogonality with all these simulated activations. Final objective: `L = L_CE + λ·L_DDO` (Eq. 14), optimized **once**, jointly, with Adam, on a single training domain `D_train` (Sec. 4.1).

The entire pipeline — concept bank, domain descriptor set, CLIP backbone, classifier weights — is fixed after this one training run. There is no notion of time, sequence, or update in the method as described.

---

## 2. Five failure modes under continual DG

### Failure Mode 1 — Closed-world, frozen domain-descriptor set
**Mechanism:** `P` (200 descriptors) is generated once (Sec. 4.3) and used identically for every dataset. The DDO loss (Eq. 13) is an expectation over `(p_i, y) ~ P × Y` — it can only erase influence from domains *representable* in `P`.

**Why it breaks continual DG:** In a continual setting, domains arrive as a stream, including ones nobody could enumerate at t=0. Since `P` is frozen at training time, any genuinely new domain gets no orthogonality protection unless someone manually re-prompts the LLM and retrains from scratch.

**Evidence (the paper's own ablation, not speculation):** Table 2 / Table 8 split the 200 descriptors into "domain-relevant" vs. "domain-irrelevant" (relative to the actual test domain) and show `+DDO(IR)` (irrelevant only) consistently underperforms `+DDO` (full/relevant set):
- LADA-Sculpture OOD: 77.70 (IR) vs 80.00 (full)
- LADV-3D OOD: 65.46 (IR) vs 68.01 (full)
- DomainNet→sketch: 66.35→67.74 (IR) vs → 69.04 (full)

This is a direct, quantified demonstration that the benefit is coverage-dependent. Fig. 5 shows accuracy gains diminish sharply past ~100–200 descriptors — i.e., the method was tuned around "enumerate what's imaginable," not "handle what's unforeseen."

**Strength:** Strong — grounded directly in the paper's own numbers.

---

### Failure Mode 2 — No incremental update path; retraining needs the original data
**Mechanism:** Eq. 14 is a single joint objective solved once over `D_train`. Nothing in Sec. 4 defines an update rule for after deployment.

**Why it breaks continual DG:** To add a newly-arrived domain, you would need to (a) regenerate `P` to include it, (b) recompute `a_sp` for the enlarged descriptor set, and (c) retrain `W_F` via Eq. 14 — which requires the *original* `D_train` still being available. Standard continual-learning setups explicitly restrict or forbid this (bounded memory, data-retention/privacy limits). If you instead fine-tune `W_F` only on the new domain without replaying old data, you get **catastrophic forgetting** of the orthogonality constraints learned for earlier domains — a well-documented failure mode in this exact model family (see §3 below: CONCIL and CI-CBM had to build dedicated anti-forgetting machinery — recursive linear-regression updates, pseudo-concept generation — specifically because naively fine-tuning a CBM's bottleneck/final layer under new data forgets old concepts/classes). LanCE has no analog of this machinery.

**Strength:** Moderate — architecturally clear from how the method is specified, but not something the paper measures (they never do sequential/multi-stage training). This is the top candidate for your empirical follow-up: simulate two sequential domains and measure domain-1 accuracy after adapting to domain-2.

---

### Failure Mode 3 — Frozen CLIP breaks the core premise outside its pretraining distribution
**Mechanism:** Everything rests on the Sec. 3 empirical claim that CLIP's embedding space aligns visual and textual domain shifts. This was demonstrated (Fig. 2, Fig. 8) only on domains heavily represented in CLIP's web-scraped pretraining corpus: sketch, painting, sculpture, clipart, 3D-render/CAD, cartoon. The full 200-item descriptor list (Fig. 9) is *entirely* conventional art/media styles — not one scientific or industrial imaging modality appears (no X-ray, MRI, satellite/SAR, thermal, microscopy, hyperspectral, etc.).

**Why it breaks continual DG:** Two concrete sub-cases:
- **Temporal drift of CLIP itself.** CLIP ViT-L/14 (Radford et al. 2021) is frozen and trained on data collected up to ~2021. Visual styles that emerged after that (specific diffusion-model aesthetics — Midjourney, SDXL, Flux-style renders — new platform-specific filters, synthetic/deepfake domains) were not in its training distribution, so the image–text alignment the whole method depends on is unverified for them. Continual DG by definition must eventually face domains the backbone predates.
- **Modalities scarce in web image-text pairs.** Domains like medical imaging, remote sensing, thermal/IR are known (BiomedCLIP, PubMedCLIP literature) to have weak CLIP zero-shot alignment because they're underrepresented in web-scraped pretraining data.

**Evidence — the authors admit this themselves.** Appendix G, verbatim: *"Our method highly depends on pre-trained VLMs like CLIP and LLMs like GPT-3.5. However, these models are limited in application to some professional fields like medical treatments. We think further integration of an extra knowledge base and task-specific fine-tuning of these pre-trained models is a potential solution."* They name the problem and propose a one-off fix (fine-tuning), not a continual one.

**Strength:** Strong — the authors' own stated limitation, plus well-known CLIP domain-coverage literature. This is the most *fundamental* failure mode: it's not fixable by adding more descriptors, because the shared embedding space itself is the weak point, not just the descriptor set's coverage.

---

### Failure Mode 4 — Static concept bank and fixed-size output layer
**Mechanism:** The concept set `C = {c_i}_{i=1}^M` is built once (human-written or LLM-generated, Sec. 4.1); `W_F` is a fixed `M → N_y` linear map.

**Why it breaks continual DG:** If new domains bring new relevant concepts, or continual DG is coupled with new classes appearing over time (a realistic joint setting), the fixed `M` and `N_y` can't absorb them without architecture surgery + retraining — which reopens Failure Mode 2's forgetting problem.

**Strength:** Moderate — architectural, not paper-tested. Confirmed as a real, hard problem by the fact that dedicated papers exist purely to solve *this* (CI-CBM: "concept-incremental and class-incremental learning" as a "novel continual learning task for CBMs" specifically because "existing CBMs typically assume static datasets").

---

### Failure Mode 5 — Structural domain gaps persist even with full descriptor coverage
**Mechanism/Evidence:** Sec. 5.3, verbatim: *"all models' generalization performance to sculpture and 3D model images on the LADA-Sculpture and LADV-3D benchmark is limited, indicating generalization from 2D to 3D remains challenging."* This is despite "3D model," "CGI render," "low-poly model," "holographic image" all being explicitly present in the 200-descriptor list — i.e., this is a domain LanCE was *designed* to handle, and the OOD numbers still land far below ID (e.g. LADV-3D: 99.93% ID vs 68.01% OOD even with DDO).

**Why it matters for continual DG:** If a well-covered, explicitly-anticipated domain shift still leaves a ~32-point gap, the degradation should be expected to compound for domains with zero descriptor coverage and zero CLIP pretraining exposure — reinforcing Failure Modes 1 and 3 rather than standing alone.

**Strength:** Strong (direct paper quote + numbers), but best used as corroborating evidence for #1/#3, not a standalone continual-specific failure.

---

## 3. Literature check — confirming the gap is real (not assumed)

You suspected most adjacent work is domain adaptation or non-continual DG. Verified via search + abstract checks:

| Cluster | Representative work | What it actually solves | Why it's not "CBM for continual DG" |
|---|---|---|---|
| CBM + continual learning | [Language Guided CBMs for Interpretable CL](https://arxiv.org/abs/2503.23283) (CVPR'25), [CONCIL](https://arxiv.org/abs/2411.17471), [CI-CBM](https://arxiv.org/pdf/2604.14519) | New **classes/concepts** arriving over time, within one visual domain | No domain shift at all — same domain throughout |
| CBM + domain generalization | LanCE (this paper), PCBM, LaBO | Generalizing to unseen domains, **single-shot** train-once | No streaming/sequential aspect — literally a one-time fit |
| CBM + temporal drift | [Tree of Concepts](https://arxiv.org/pdf/2604.17089) | Non-stationary **tabular clinical** data over time | Not visual, not discrete style-domains, single continuously-drifting distribution rather than distinct domains |
| Continual domain-incremental (no CBM) | [Domain Generalizable Continual Learning (DGCL)](https://arxiv.org/abs/2510.16914) (Oct 2025) | Sequential single-domain tasks, generalize across all encountered domains | Closest structural analog, but black-box (no interpretability, no CLIP-language descriptor mechanism) |
| Concept-based domain adaptation | CUDA (ICML'25) | Aligning concepts using **target-domain data** | Adaptation, not generalization to *future unseen* domains — an easier problem (target data available) |

**Conclusion:** No existing work sits at the intersection of (concept-based interpretability) × (generalization to *unseen future* domains) × (continual/streaming domain arrival without forgetting) × (dynamic, language-guided descriptor updating). That intersection is your white space — defensible from this literature check, not asserted from absence of a quick search.

---

## 4. What to verify empirically next (predictions, not results)

Ranked by how directly they follow from paper evidence vs. how much new experimentation they need:

1. **Coverage-gap test (extends Table 2 directly).** Take a domain with *zero* semantic overlap to any of the 200 stock descriptors (not just "domain-irrelevant" within their set, but genuinely absent — e.g. a specific post-2021 synthetic-art style) and measure OOD accuracy. Prediction: gains collapse further than their own `+DDO(IR)` numbers, toward baseline CLIP-CBM.
2. **Out-of-CLIP-distribution test.** Run LanCE (using their public checkpoints/code) on a domain scarce in CLIP's pretraining — e.g. a medical imaging or remote-sensing style adapted from an existing attribute-annotated dataset. Prediction: both baseline and DDO-augmented accuracy drop sharply, and the *relative* DDO benefit shrinks or vanishes, since the Sec. 3 alignment premise itself weakens.
3. **Sequential-domain forgetting test.** Train on domain A with DDO, then continue training/fine-tuning `W_F` on domain B (without replaying A). Measure domain-A OOD accuracy before vs. after. Prediction: significant drop, demonstrating catastrophic forgetting LanCE was never built to resist.
4. **Descriptor-set staleness test.** Freeze `P` as generated by GPT-3.5-turbo (2023-era knowledge), then test against a domain description that a newer LLM would generate but GPT-3.5 wouldn't (a domain/style that didn't exist or wasn't named at GPT-3.5's training time). Prediction: `a_sp` simulation is poor for it because the descriptor was never in the pool to begin with.

---

## 5. Presentation framing suggestion

Lead with **Failure Mode 3** (frozen CLIP + authors' own admitted limitation) as the "fundamental ceiling," then **Failure Mode 1** (their own ablation numbers) as the "closed-world" problem, then **Failure Mode 2** (no update path) as the concrete continual-learning gap, using Failure Mode 5's 2D→3D numbers as a visceral "even their best case has a 32-point gap" hook. Close with the literature-gap table to establish that "continual DG for CBMs" is unclaimed territory — then hand off to the empirical-verification list as "next steps," framed honestly as predictions to be tested, not settled conclusions.
