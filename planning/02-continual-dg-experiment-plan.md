# Full plan, restated: two pillars, continuality first

## Context

You asked two things: restate the whole plan, and confirm whether we're still covering the scenario where a frozen CLIP simply doesn't know how to represent domains outside its training data — showing the paper's approach can't survive in the long run even if someone fixed its forgetting problem. Short answer: **yes, we're bringing that back in**, as an explicit second pillar — it got pushed to a one-line footnote last round in the name of staying lean, but you're right that it's a distinct, real part of "why continual DG won't work here," not noise. It stays clearly secondary to the forgetting story, not equal billing.

**Pillar 1 (primary, first-and-foremost): LanCE has no mechanism to update as domains arrive over time.** Even within the kinds of domains its frozen CLIP backbone understands perfectly well (ordinary photos, paintings, cartoons, sketches), the model still breaks the moment it has to learn domain 2 without forgetting domain 1. This is the core continuality claim your advisor will judge the proposal on.

**Pillar 2 (secondary, reinforcing): even a perfectly-solved forgetting problem wouldn't save it in the long run.** LanCE's entire mechanism is built on CLIP's frozen, 2021-era image-text alignment. A continual system is, by definition, going to keep meeting new domains indefinitely — and eventually some of those domains will be things CLIP's frozen snapshot simply never learned to align with language for at all (not just "a style it wasn't told about," but "a kind of image its embedding space doesn't represent well"). No amount of clever continual-updating of the classifier fixes that, because the foundation underneath it never updates. This is the "why this doesn't just need a smarter classifier, it needs a fundamentally different foundation-model story" argument — useful for motivating where your actual proposed method goes next.

**Why these stay two separate experiments instead of one merged one:** Pillar 1 needs a multi-domain dataset where all domains are things CLIP already understands well — that's what makes it a clean test of *forgetting* specifically, not confounded by CLIP just being bad at some domain. PACS (photo/art/cartoon/sketch — all pre-2021, all art styles CLIP saw constantly) is exactly this. But that same property — CLIP understands PACS well — makes PACS structurally unable to test Pillar 2 at all. A real Pillar 2 test needs a domain CLIP genuinely doesn't understand. I looked for one that could plug into the *same* PACS class sequence (so it'd be one unified experiment) — a "post-2021 AI-generated" domain via the GenImage dataset, using the same object classes — and it's a dead end for now: the accessible version has no class labels, and the full version's per-class downloadability couldn't be confirmed. Forcing a merge here would burn real time chasing a dataset that might not even be gettable. So Pillar 2 runs as its own small, separate, low-risk experiment (EuroSAT) instead — same conclusion, cheaper and safer to actually obtain.

You have an RTX 5060, 8GB VRAM — everything below keeps CLIP frozen (inference-only); only a small linear layer ever trains, so nothing here is GPU-constrained.

## Continual-learning vocabulary this plan is built around (for the proposal write-up)

- **Domain-IL** (van de Ven & Tolias's standard three-scenario taxonomy): same task, shifting input domain, task identity unknown at test time. PACS with its fixed 7-class label set and 4 changing domains is a clean Domain-IL setup — the setting LanCE was never evaluated in.
- **ACC / BWT** (Lopez-Paz & Ranzato): Average Accuracy across all domains after the final increment, and Backward Transfer — how much earlier-domain performance changes after learning later domains. Negative BWT = forgetting. Standard CL reporting, not ad hoc percentages.
- **Closed-world vs. open-world continual learning**: closed-world assumes the future is enumerable in advance; open-world doesn't. LanCE's 200-descriptor list, written once before training, is a textbook closed-world assumption.
- **EWC** (Kirkpatrick et al.): the standard baseline anti-forgetting technique, used in Phase C as a remediation attempt.

## Two things already investigated and ruled out this session (documented so they don't get re-chased)

1. **GenImage** (post-2021 AI-generated images, same classes as ImageNet/AWA2) — conceptually the ideal Pillar 2 dataset, since it could theoretically share a class taxonomy with a Domain-IL sequence. Practically a dead end: the small "Tiny-GenImage" version has no per-class labels; the full labeled version's per-class download granularity couldn't be confirmed. Dropped.
2. **AWA2/AwA2-clipart as the Domain-IL testbed** (instead of PACS) — already has a loader in LanCE's code, zero new engineering. Rejected anyway: AWA2's photo domain alone is a ~13GB download vs. PACS's ~174MB. PACS stays.

## What we're building on top of

LanCE's own released code (`github.com/joeyz0z/LanCE`) has `main.py`, the DDO loss, the CLIP-CBM model, and LaBO-style LLM concept generation — but only 2-domain benchmarks (CUB/CUB-Painting, AWA2/AwA2-clipart, LAD_animal/LADA-Sculpture, LAD_vehicle/LADV-3D, RIVAL10), no PACS loader, and obviously no EuroSAT support. We reuse their model/DDO-loss/training code as-is everywhere; we add one PACS loader, one small EuroSAT script, and new experiment-driver scripts around them.

## Design decisions across every phase

- **LaBO-style CLIP-CBM throughout** — their most complete model, LLM-generated so it needs no human attribute annotation for PACS/EuroSAT, and it's what they use for their own multi-domain (Fig. 4) and ablation (Table 2) results. One model family, comparable results everywhere.
- **Only `W_F` (the final linear layer) is ever trained/fine-tuned** — not our simplification, the *only* learnable piece of their architecture. CLIP and the concept bank are frozen by design.

---

## PILLAR 1 — Continuality (primary)

**Phase 0 — Reproduce the baseline (control/gate, not evidence itself)**
**[Corrected during implementation]** Switched from LADA-Sculpture to **CUB + CUB-Painting**. Reading LanCE's actual data-loading code surfaced two problems with LADA-Sculpture: (1) the HuggingFace `LADA-Sculpture` release is only the *target* (sculpture) domain — the *source* (real photo) domain is a separate download from Google Drive/Baidu Netdisk with no confirmed size, a real access-risk on the level of the GenImage dead-end; (2) the code's default LADA path is hardcoded to `LAD_animal_conceptNet_concepts.txt`, which corresponds to Table 1's **PCBM/ConceptNet row (76.69 → 79.74 OOD)**, not the LaBO row I'd originally cited. CUB is hosted directly by Caltech (1.1GB, confirmed, no registration) and CUB-Painting is a single Google Drive link — lower risk than chasing LADA's source domain blind. CUB's default code path (`Processed_CUB_Dataset`) loads `cub_concepts.txt` (311 human-style attributes), matching Table 1's **CLIP-CBM/human row (50.54 → 55.53 OOD)**. **Pass bar: within ~3–5 points of that row.** Nothing downstream is trusted until this passes. ViT-B/32 is the fallback if VRAM is tight (it won't be — CLIP stays frozen/inference-only).

**Phase A — Closed-world descriptor assumption test**
On CUB/CUB-Painting: instead of their binary relevant/irrelevant descriptor split (Table 2/8), measure actual CLIP-embedding similarity between each of the 200 descriptors and the true test domain, then sweep DDO training on progressively less-similar descriptor subsets — a dose-response curve. Shows DDO's benefit depends on the descriptor list having anticipated the domain — the textbook closed-world assumption continual/open-world settings violate by construction. Caveat to state honestly: even the least-similar subset is still one of their own 200 art-style descriptors, so this shows degradation *within* the anticipated family, not a domain totally outside it (that's what LADA can't test — see Pillar 2).

**Phase B — Domain-IL sequential protocol on PACS (the core experiment)**
Dataset: **PACS** — 4 domains (photo/art_painting/cartoon/sketch), 7 classes (dog, elephant, giraffe, guitar, horse, house, person), ~9,991 images, ~174MB, official source [sketchx.eecs.qmul.ac.uk](http://sketchx.eecs.qmul.ac.uk/). Chosen because it's the smallest, cheapest, most standard multi-domain benchmark with enough domains (4) to actually simulate a sequence, and all its domains are things CLIP knows well — which is exactly what isolates *forgetting* as the cause of any failure, not CLIP's general unfamiliarity with the domain (that's Pillar 2's job).

1. Write `data/pacs_data.py` (mirrors `cub_data.py`'s interface).
2. Generate PACS's 7-class concept bank using their exact concept-prompt template (Fig. 9) — done directly, no external LLM dependency. Their 200 domain descriptors are dataset-agnostic and reused as-is.
3. `experiments/domain_il.py`: **(a) Joint/oracle** — train once on all 4 domains pooled (upper bound). **(b) Naive sequential** — train domain *i* with CE+DDO, eval all 4, move to domain *i+1* fine-tuning only `W_F`, no replay. Direct forgetting probe.
4. **Mechanism-level measurement, not just accuracy.** At every stage, recompute the DDO loss value (`|W_F · a_sp(p,y)|`) against *domain-1's* descriptors using the *current* `W_F`. Plain forgetting is true of any sequentially fine-tuned classifier — not LanCE-specific, and a sharp advisor will say so. What's LanCE-specific is whether the orthogonality property DDO explicitly trained for survives, or silently erodes once nothing is actively enforcing it.
5. Report in **ACC / BWT**, the standard CL metrics.
6. Repeat across 2–3 domain orderings — order-sensitivity is a known CL confound, controlling for it is expected practice.

**Phase C — Remediation attempts: does a textbook CL fix already solve it?**
Same PACS harness, three standard-toolkit fixes: **(1) cumulative DDO** (near-free — DDO needs no raw images, so just keep applying every prior domain's descriptor terms, not only the newest); **(2) cached-embedding replay** (store small compressed summaries of prior domains, mix into later training — the realistic version, with a real memory cost); **(3) EWC-style regularization on `W_F`** (Kirkpatrick et al. — anchor the classifier against drifting too far from earlier domains' solution). If none of these fully fix it, that's the strongest possible motivation for a new method. If one does, report that honestly too — it still shows the architecture has no *built-in* fix, only an externally bolted-on one.

**Phase D — Stretch, optional:** repeat Phase B/C on Office-Home (65 classes) to confirm the PACS result isn't a one-dataset fluke, only if there's time after Pillar 1's core result is solid.

---

## PILLAR 2 — Long-term viability (secondary, reinforcing)

**Phase F1 — Domain-shift alignment check on EuroSAT (cheap, ~1–2 hours, minimal setup)**
Dataset: **EuroSAT** — satellite land-use imagery, 10 classes (forest, river, highway, residential, industrial, pasture, crop, sea/lake, etc.), ~89,000 images total but we only need a small sample, ~89MB via `torchvision.datasets.EuroSAT`. Chosen because it gives a precise, already-published, unimpeachable number: OpenAI's own CLIP paper reports **ViT-L/14-336 zero-shot on EuroSAT: 59.6%**, vs. a **linear probe on the identical features: 98.1%**. That 38-point gap proves CLIP's *visual* representation of satellite imagery is fine — the failure is specifically in *image-text alignment*, the one mechanism LanCE's concept activations and DDO loss depend on entirely.

What we're hoping to show: EuroSAT's classes (forest, river, highway...) are generic land-cover words with ordinary ground-photo equivalents, so "a photo of a forest" vs. "a satellite image of a forest" is a legitimate same-concept domain-shift pair — the same shape of test as their own Fig. 2 (which reports 0.90–0.99 alignment scores for sketch/sculpture/painting shifts). We compute the same kind of alignment for a photo→satellite shift and expect it to land well below their reported range — a direct falsification of Sec. 3's premise for a domain outside CLIP's comfort zone, no training run required.

**Phase F2 — Concept-activation ceiling test on EuroSAT (if F1 result justifies the extra step)**
Build a plain CLIP-CBM (no DDO) on EuroSAT's own 10 classes, in-distribution, and compare its accuracy to the two anchors above (59.6% zero-shot, 98.1% linear-probe ceiling). **Important, state explicitly:** this is *not* a domain-generalization test — EuroSAT's classes aren't in the same label space as PACS/LADA. It's a precondition check: does concept-based classification work at all in a modality CLIP aligns poorly with. If CBM accuracy tracks near 59.6% rather than 98.1%, that shows concept-bottleneck models inherit CLIP's alignment weakness even where the visual information is demonstrably present in the representation — a "long-run ceiling" that no amount of continual-learning fixes to the classifier can lift, because the ceiling is set by the frozen backbone underneath.

---

## Phase G — Write-up
Fold every measured result (both pillars) into `docs/lance_continual_dg_failure_analysis.md` and the artifact, replacing predictions with actual numbers, each traced to a specific run/log.

## Deliberately deferred to future work (not this round)

- **Joint Domain-IL + Class-IL**: whether `W_F`'s fixed shape can absorb new classes arriving alongside new domains. Real and likely the natural next step, but mixing two failure axes right now would blur which one causes what — save it as "where this goes next" in the proposal's close.

## Verification

- Phase 0 is the hard gate for all of Pillar 1 (and Phase F2, since it reuses the same CBM training code).
- Phase B/C/D report exact ACC/BWT numbers, domain order, and config — no cherry-picking.
- Phase C is reported honestly either way — a partial fix is still a real, useful finding.
- Phase F1/F2 are clearly labeled as testing a *different* claim than Pillar 1 (backbone coverage vs. forgetting) so the two never get conflated in the write-up.
- Everything reuses LanCE's existing model/DDO-loss code rather than reimplementing it.

**Nothing runs until you say start.**
