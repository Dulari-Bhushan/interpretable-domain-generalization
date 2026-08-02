# Coding plan: prove LanCE has no continual-DG mechanism, via multiple CL protocols

## Context

Your advisor specializes in continual learning. That makes **proving LanCE has no continual-learning mechanism the single first-and-foremost goal of this round** — every phase below exists to serve that one claim, from an angle your advisor will recognize from the CL literature, not a single accuracy number. Nothing else (domain coverage, CLIP's modality gaps, temporal staleness) gets active engineering time unless it directly strengthens the continuality argument.

You confirmed: stay on **LanCE only** (no second CBM paper to reproduce), probe it with **multiple continual-learning protocols** rather than multiple methods, and keep the plan lean — don't chase side-quests that cost setup time without directly serving the continuality claim. That discipline already paid off once this session, on two counts:

1. **We checked whether a "post-2021 AI-generated domain" (via the GenImage dataset) would be a better stand-in than EuroSAT for a *temporal*-staleness argument.** Conceptually it's sound (Midjourney/SD-generated images of ImageNet/AWA2-overlapping classes are genuinely images CLIP could never have seen). Practically it's a dead end right now: the easy-to-get "Tiny-GenImage" version has no per-class labels at all (only real/fake + generator labels — useless for us), and for the full class-labeled version we could not confirm that individual classes are downloadable without pulling a much larger per-generator archive. Chasing this further would burn setup time on a side-quest. **Verdict: dropped, not pursued.**
2. **We re-checked whether AWA2/AwA2-clipart (which already has a loader in LanCE's code, so zero new engineering) should replace PACS as the Domain-IL testbed.** It shouldn't: AWA2's photo domain alone is a ~13GB download vs. PACS's ~174MB. Writing one small PACS loader (my time) is cheaper than a 13GB download (your time/bandwidth/disk). **PACS stays.**

Documenting both here so neither gets re-investigated later. If GenImage-style evidence is ever worth revisiting, the design would be: AWA2 (photo) → AwA2-clipart → a small GenImage-Midjourney slice restricted to AWA2-overlapping classes (tiger, zebra, elephant, etc.), reusing LanCE's existing AWA2 loader for the first two stages — but only once someone confirms a real way to download a small class-filtered slice.

This refines the earlier plan (LADA-Sculpture reproduction + coverage-gap ablation still stand; the PACS domain-incremental experiment is the centerpiece, expanded into distinct CL-literature-standard protocols instead of one forgetting run). You have an RTX 5060, 8GB VRAM — CLIP stays frozen throughout, only `W_F` (a small linear layer) ever trains, so none of this is GPU-constrained.

## Grounding this in standard continual-learning vocabulary (for the proposal write-up)

Your advisor will map anything you show onto the field's standard framework, so the experiments are designed to speak that language directly rather than needing translation:

- **Domain-IL vs. Class-IL vs. Task-IL** (van de Ven & Tolias's standard three-scenario taxonomy for continual learning): PACS with a fixed 7-class label set and a changing domain per increment is a clean **Domain-IL** setting — same task, shifting input distribution, task identity not given at test time. This is the setting LanCE was never evaluated in (Sec. 4.1 trains on exactly one domain, once).
- **ACC / BWT** (standard metrics from the continual-learning evaluation literature, e.g. Lopez-Paz & Ranzato's GEM paper): Average Accuracy across all domains after the final increment, and Backward Transfer — how much performance on domain *i* changes after training on later domains. Negative BWT = forgetting. Reporting these instead of ad hoc "accuracy dropped" numbers is what makes results legible to a CL-literature reviewer.
- **Closed-world vs. open-world continual learning** (a standard distinction in CL surveys): a closed-world learner assumes the full set of future classes/domains is enumerable in advance; open-world settings assume it isn't. LanCE's descriptor set `P` is a textbook closed-world assumption (Sec. 4.3 enumerates domains once, via one LLM prompt, before training).
- **EWC** (Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks") is the standard baseline anti-forgetting technique we'll use as a remediation attempt in Phase C below — if even a textbook CL fix doesn't rescue LanCE, that's a strong argument a bespoke method is needed.

## What we're building on top of

LanCE's own released code (`github.com/joeyz0z/LanCE`: `main.py`, DDO loss, CLIP-CBM model, LaBO-style LLM concept generation) has only 2-domain benchmarks (CUB/CUB-Painting, AWA2/AwA2-clipart, LAD_animal/LADA-Sculpture, LAD_vehicle/LADV-3D, RIVAL10) and no PACS/OfficeHome/DomainNet loader. Everything below reuses their model/DDO-loss/training code as-is; we add one new PACS data loader and new experiment-driver scripts around it.

## Design decisions that apply across every phase

- **Standardize on LaBO-style CLIP-CBM throughout.** It's the paper's most complete model, used for their PACS/OfficeHome/DomainNet results (Fig. 4) and their Table 2 ablations, and it's LLM-generated so it applies unmodified to PACS (no human attribute annotations needed). One consistent model family = results comparable across every phase.
- **Only `W_F` (the final linear layer) is ever trained/fine-tuned.** Not a simplification we're choosing — it's the *only* learnable component in the architecture (CLIP and the concept bank `C` are frozen by design, Sec. 4.1). This is worth stating explicitly in the proposal: "the paper gives us nothing else to update" is itself part of the finding.
- **PACS is the primary testbed for all Domain-IL experiments** — 4 domains (photo/art_painting/cartoon/sketch), 7 classes, ~9,991 images, ~174MB, official source [sketchx.eecs.qmul.ac.uk](http://sketchx.eecs.qmul.ac.uk/). LADA-Sculpture (HF: `JoeyZoZ/LADA-Sculpture`, ~15.4k images) stays as the Phase 0 reproduction/control dataset and the Phase A closed-world ablation dataset — neither of those depends on the EuroSAT critique, both are continual-learning-relevant (closed-world assumption testing), so both stay in scope.

## Phase-by-phase plan

**Phase 0 — Reproduce the baseline (control/gate, not itself evidence)**
Clone `joeyz0z/LanCE`, `pip install -r requirements.txt`, download LADA-Sculpture, run `main.py --dataset LADA --alpha 0` and `--alpha 1` (LaBO concepts, CLIP ViT-L/14). **Pass bar: within ~3–5 points of Table 1's LaBO row (baseline 74.56 → +DDO 80.00 OOD).** If we can't reproduce their own numbers, nothing downstream can be trusted. ViT-B/32 is the fallback if VRAM is tight.

**Phase A — Closed-world descriptor assumption test (extends Table 2/8; frames Failure Mode 1 in CL terms)**
On LADA-Sculpture: compute CLIP-embedding similarity between each of the 200 descriptors' domain-shift embeddings and the true test domain's shift embedding, then sweep DDO training on the bottom-10%/25%/50%/100%-similarity subsets — a dose-response curve instead of their single relevant/irrelevant split. **Framing for the proposal:** this is a closed-world-assumption test — DDO's benefit is shown to depend on how well the *pre-enumerated* descriptor set anticipates the *actual* domain, which is precisely the assumption open-world/continual settings violate by construction (you cannot enumerate a stream in advance). **Honest caveat:** even the least-similar subset is still drawn from their own 200 GPT-3.5 descriptors, so this shows degradation *within* the anticipated family, not total breakdown outside it — say this plainly, don't overclaim.

**Phase B — Domain-IL sequential protocol on PACS (the core experiment — direct forgetting probe)**
1. Write `data/pacs_data.py` (mirrors `cub_data.py`'s interface: image paths + labels + domain field).
2. Generate a concept bank for PACS's 7 classes (dog, elephant, giraffe, guitar, horse, house, person) using their exact concept-prompt template (Fig. 9) — generated directly, no external LLM API dependency. Their 200 domain descriptors are dataset-agnostic and reused as-is.
3. New script `experiments/domain_il.py`, reusing their model/DDO-loss code:
   - **(a) Joint/oracle** — train once on all 4 domains pooled (non-continual upper bound).
   - **(b) Naive sequential (Domain-IL)** — train on domain *i* with CE+DDO, eval on all 4 domains, move to domain *i+1* fine-tuning only `W_F`, no replay. The direct forgetting probe.
4. **Mechanism-level measurement, not just accuracy.** At every stage, recompute the DDO loss value itself (`|W_F · a_sp(p,y)|`, Eq. 13) against **domain-1's** descriptors using the **current** `W_F`. Plain sequential forgetting is true of any linear classifier and isn't LanCE-specific — a sharp advisor will say so immediately. What's specific to LanCE is whether the *orthogonality property DDO explicitly trained for* survives later updates, or silently erodes once nothing is actively regularizing against it. That's the real claim.
5. Report using **ACC and BWT** (defined above), not ad hoc percentages.
6. Repeat across 2–3 domain orderings — order-sensitivity is a documented confound in the CL literature, so controlling for it here is expected practice, not extra credit.

**Phase C — Remediation attempts: does a textbook CL fix already solve it? (this is the phase your advisor will ask about first)**
Three different standard-toolkit fixes, all applied on top of the same Phase B harness, to test whether LanCE's forgetting needs a bespoke solution or a known one already works:
1. **Cumulative DDO (near-free ablation).** DDO's `a_sp` simulation (Eq. 12) is purely descriptor-based — it needs no raw images. So "replaying domain 1" costs nothing: keep applying *all* previously-seen domains' DDO terms (not just the current domain's) at every subsequent stage. If this alone rescues domain-1 accuracy, that's an important, honest finding — it would mean LanCE's forgetting has a nearly-free fix, which changes (but doesn't erase) the pitch: it shows the *architecture* still has no principled update rule, that we discovered the fix ourselves through direct experimentation, not the authors.
2. **Cached-embedding replay.** Store a small buffer of concept-activation vectors (not raw images — cheap) from each prior domain, mix into every subsequent training stage. Tests the more realistic version of "just keep old data around," with a real memory cost this time.
3. **EWC-style regularization on `W_F`.** After domain-*i*, anchor `W_F` with an importance-weighted penalty (Fisher-information-based, per Kirkpatrick et al.) against moving away from its post-domain-*i* values while training domain *i+1*. This is the standard CL-toolkit answer to catastrophic forgetting — if it doesn't work here, that's the strongest possible motivation for a new method, because it shows the problem resists the field's default solution, not just a naive baseline.

**Phase D — Stretch, optional:** repeat Phase B/C on Office-Home (65 classes, larger concept bank) — a second Domain-IL dataset to confirm the PACS result isn't a one-dataset fluke — if Phase B/C results are clean and there's time left.

**Phase E — Write-up:** fold measured numbers into `docs/lance_continual_dg_failure_analysis.md` and the artifact, replacing predictions with results, each traced to a specific run/log.

## Deliberately deferred to future work (not part of this round)

- **Joint Domain-IL + Class-IL** (previously "Phase D"): testing whether `W_F`'s fixed `M → N_y` shape can even structurally absorb new classes arriving alongside new domains. Real, and probably the natural next step once the pure domain-incremental case is nailed down — but mixing two axes of failure right now would blur which one is causing what. Save it as the "here's where this goes next" close of the current proposal.
- **EuroSAT/CLIP-alignment angle** (Failure Mode 3): a domain-*representation* argument, not a continual-*learning* one — off-target for a CL-focused advisor as a headline result. One-line footnote only, not a core experiment.

## Verification

- Phase 0 is the gate: nothing downstream is trusted until baseline reproduction is within ~3–5 points of Table 1.
- Phase B/C/D results reported with exact ACC/BWT numbers, domain order, and config — no cherry-picking a single favorable order or condition.
- Phase C is reported honestly either way: if a remediation works, say so — it sharpens rather than weakens the proposal ("the architecture has no principled fix, but we found an ad hoc one; a real method should do better than our ad hoc patch").
- Everything reuses LanCE's existing model/DDO-loss code rather than reimplementing it, to avoid a subtly-wrong reimplementation producing misleading "failures."

**Nothing runs until you say start.**
