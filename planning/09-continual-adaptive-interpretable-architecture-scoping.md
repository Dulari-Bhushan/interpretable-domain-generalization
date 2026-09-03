# Plan 09 (scoping only, no code): the continual/domain-adaptive/interpretable vision spec — what it would take to build

**This is a scoping document, not an experiment plan.** The project owner supplied a full, self-contained architecture spec ("Continual, Domain-Adaptive, Interpretable Vision Architecture") for a system combining class-incremental learning, continuous test-time domain adaptation, concept-embedding interpretability with a leakage guarantee, and calibrated drift detection. Per explicit instruction, nothing here gets built until this scoping is reviewed — this document exists to answer "how big is this, in what order, and what does it conflict with in what's already here," honestly, before any GPU time or implementation effort is spent.

The full spec is reproduced nowhere in this repo (it was supplied inline in conversation, not as a file) — this document references its section numbers (§4.1-§4.8, §6-§9) as given, and anyone picking this up cold should ask the project owner for the original spec text if the section references aren't self-explanatory enough.

---

## 1. The two scope conflicts that need a decision before anything else

These aren't implementation details — they're decisions about what this project's methodology *is*, and building around them without resolving them first would waste real effort.

### 1.1 Class-incremental learning was explicitly paused by decision

`docs/new_methodology_report.md` §1 states, verbatim: **"every component stays on the domain axis. The class set stays fixed throughout; class-incremental behavior is paused by decision."** The new spec's §3.1, §4.4, and §6 are built around class-incremental learning as a first-class requirement (new classifier weight vectors per new class, herding per (class, domain) cluster, the classifier head "incrementally extended per new class"). This is not a small extension of the existing 5-component methodology — it reverses a scope boundary that was set deliberately, not by omission. **Needs an explicit decision:** does this new work supersede that boundary (i.e., is class-incremental learning back in scope now), or does the new spec get scoped down to domain-incremental-only to stay consistent with everything else in this project? The rest of this document assumes this is still an open question and flags every place it matters.

### 1.2 The prototype exemplar bank (§4.5) directly conflicts with Component 4's own privacy motivation

Component 4 (`planning/06-component4-domain-memory-plan.md`, in progress) exists *specifically* to solve the problem the new spec's §4.5 reintroduces. Component 4's own framing: real per-sample replay buffers are "keep a linkable per-patient record around indefinitely, just in embedding form instead of pixel form" — a real governance problem for domains like Camelyon17 (medical) or CheXpert, which is why Component 4 replaces per-sample storage with a per-class Gaussian summary (mean + diagonal variance) that never retains any individually-linkable record. The new spec's §4.5 explicitly calls for "real (never generated) stored exemplars" via herding selection — the exact mechanism Component 4 was built to avoid. **Needs a decision:** adopt Component 4's Gaussian-summary approach instead of §4.5's real-exemplar bank (loses the anchor loss's most direct grounding — an anchor toward a synthetic Gaussian mean is weaker evidence than an anchor toward a real stored example, this is a real tradeoff not a free substitution), keep §4.5 as specified for non-sensitive domains only (with the two mechanisms coexisting as a configurable choice), or treat this new track as deliberately out of scope for privacy-sensitive data and accept real-exemplar storage as a stated limitation. Not a decision this document should make unilaterally.

---

## 2. Module-by-module: what exists, what's new, rough effort

| Spec module | What exists in this repo today | What's genuinely new | Effort (rough) |
|---|---|---|---|
| §4.1 Frozen backbone | CLIP ViT-L/14 and DINOv2 ViT-B/14/L/14 both already integrated and validated (Plan 07) | Nothing — direct reuse | **None** |
| §4.2 Concept embedding bottleneck (CEM vectors + leakage adversary) | Scalar concept-activation mechanisms exist (CLIP zero-shot similarity, DINOv2 trained probe) and concept-bank generation (human, LLM — Plan 08) exists | Vector-valued concepts (`c_k+`/`c_k-` per concept, not one probability), the leakage adversary + gradient-reversal training loop, MI-based concept filtering | **M** — closest existing work to build from; realistically prototype-able against CUB alone in isolation |
| §4.3 Prompt pool (continual learning) | Nothing — LanCE has no mechanism for injecting learnable tokens into backbone blocks, nor for freezing old task parameters while training new ones this way | Prompt injection into specific ViT blocks (needs backbone-forward hooks or a wrapper), query/key retrieval + soft blending, the diversity loss, the freeze-old/train-new update discipline | **L** — genuinely new subsystem, no code to build from; this is essentially implementing L2P/DualPrompt/CODA-Prompt-style continual learning from scratch |
| §4.4 Cosine/NCM classifier head | `clip_cbm`'s classifier is a plain `Linear`, fixed-size at init — assumes the concept count and class count are both known upfront | A cosine-similarity head with incrementally-appendable per-class weight vectors, sized against a *growing* `K·d_c` concept-embedding dimension (current code has nothing like a growing input dimension anywhere) | **S-M** — small in isolation, but only meaningful once §4.2 and §4.3 exist |
| §4.5 Prototype exemplar bank | Nothing exists; **Component 4 exists as the opposite design philosophy** (see §1.2) | Herding selection, Mahalanobis-based cluster merging, FAISS retrieval at scale | **L** — and blocked on §1.2's decision before it's worth building either variant |
| §4.6 Test-time adaptation (entropy-free) | Nothing — every training loop in this codebase is one-shot offline supervised training on a fixed dataset; there is no streaming/online training path at all | The entire online update loop, the consistency + anchor loss, stochastic parameter restoration, careful parameter-group isolation (frozen backbone/old-prompts must be structurally excluded from the optimizer) | **L** |
| §4.7 Drift/uncertainty monitor | Loosely related to `docs/new_methodology_report.md`'s flagged (not-started) stretch idea, "detecting a new domain arrived, without being told" — conceptually adjacent, not built | Mahalanobis scoring against the prototype bank (depends on §4.5's resolution), temperature-scaling calibration refresh, the hard gradient-isolation requirement from §4.6's TTA loss | **M** — depends on §4.5 existing first |
| §4.8 Concept vocabulary expansion | **Component 3 already built and tested** (`results/component3_self_growing_vocabulary.md`) — VLM-candidate generation, trust-threshold validation, growth/no-growth decision logic, on Defactify | Porting/adapting Component 3's validated machinery to trigger off residual variance (§4.2's leakage signal) instead of Component 3's own trigger, and to actually add new `c_k±` vectors + extend the classifier head (currently untested — Component 3's real run never actually grew the vocabulary, since 0/10 candidates cleared its trust threshold) | **S-M** — best-covered module in the whole spec, real code to start from |
| §4.9 Output bundle / eval protocol (§5, §8) | Nothing — no existing harness produces this bundle shape, and §8 admits no benchmark exists | The full metrics suite (ECE, OOD AUROC, concept-intervention accuracy, leakage-over-time, prototype scaling) *and*, first, the benchmark itself | **XL** — see §3 below, this is likely the single largest cost item in the whole undertaking |

**Reading the table:** §4.2 and §4.8 are the two modules with real, validated code to build from. §4.3, §4.5, §4.6, and the benchmark (§4.9) are each substantial, independent systems-engineering efforts with nothing in this repo to reuse. This is not "extend the CBM work" — past §4.1/§4.2, it is closer to a new codebase.

---

## 3. The missing benchmark (spec's own §8, taken seriously)

The spec is honest that no existing public dataset combines: a task/domain sequence with repeated classes across a shift, per-image concept labels, and a held-out true-novelty domain never trained or adapted on. What's on hand in this project doesn't close that gap either:

- **CUB** has real per-image concept labels (312 attributes) — but no domain-incremental sequence at all (one source domain, one fixed target, no repeated classes across multiple shifts).
- **PACS / Office-Home / DomainNet** have exactly the domain-incremental sequence structure needed (Component 1 already validated exact-match classifier updates across PACS/Office-Home/DomainNet orderings) — but none of them have real per-image concept labels; their concept banks are hand-written first drafts (or, for DomainNet, template-generated and explicitly flagged as "not real concepts").
- **A held-out true-novelty domain never trained on** — closest existing asset is Defactify's 5 AI-image generators (Phase F1/F3/F4's own alignment-score work already characterizes several of these as genuinely low-CLIP-alignment), but Defactify has no concept labels either and isn't currently organized as a class-incremental sequence.

**What building this benchmark actually requires:** picking one dataset family (Office-Home or DomainNet are the strongest domain-incremental candidates, per Component 1's own validated harness) and either (a) hand-writing or LLM-generating concept banks for it (Plan 08's machinery is directly reusable here — the LLM-bank generation process and `--concept_file` plumbing both already exist) with no real per-image ground truth to check them against (so §4.2's MI-filtering step and any concept-level evaluation would run against soft VLM pseudo-labels only, not real labels — a real limitation to state up front, not discover later), or (b) finding/licensing a dataset that already has both, which the spec itself says doesn't exist publicly in this combination.

This is worth being blunt about: **the benchmark-construction sub-project is comparable in scope to several of the model modules above**, and nothing else in this document can be empirically validated end-to-end without it. Module-level validation (§4.2 alone on CUB, §4.8 reusing Component 3's harness) can happen without it; the full system's own evaluation protocol (§8's metrics table) cannot.

---

## 4. Recommended build order (if this proceeds)

Ordered to front-load the modules with existing validated code, defer the two unresolved scope decisions as long as possible without blocking real progress, and treat full-system integration as the last, largest, and most uncertain step rather than the starting point.

1. **§4.2 alone, on CUB.** Vector CEM + leakage adversary, tested against Plan 07/08's already-measured scalar concept-activation numbers (mean AUROC, downstream accuracy) as the baseline to beat. Self-contained, uses an existing dataset and existing evaluation convention, no dependency on any other new module. This is the natural first slice — real signal on whether CEM + leakage-removal is worth carrying into the rest of the system, before building anything that depends on it.
2. **§4.8, adapted from Component 3.** Port the existing trust-threshold/candidate-validation logic to trigger off §4.2's leakage residual instead of Component 3's original trigger. Cheap relative to everything else here, and the one place a real prior validated result already exists to build from.
3. **§4.3 prompt pool, on Office-Home or DomainNet (domain-incremental only, sidestepping §1.1 for now).** The largest genuinely-new subsystem with no scope-conflict blocking it — can be built and validated (does soft prompt retrieval + freeze-old/train-new actually prevent forgetting, measured against Component 1's own exact-match BWT numbers as a reference point) without resolving the class-incremental question, since Component 1's existing domain-IL harness already has everything needed except the prompt-pool code itself.
4. **§1.1 and §1.2 decisions, revisited here** — by this point, §4.2/§4.3/§4.8 would be real, working, validated pieces, which is a much better position to decide "is class-incremental learning worth reopening" and "which exemplar-storage philosophy" from than deciding it upfront with nothing built yet.
5. **§4.6 TTA module** — depends on §4.3 (prompt pool is what it adapts) and needs the streaming/online training path this codebase has never had. Real, isolated engineering risk: parameter-group isolation (never touching frozen backbone or old prompts) needs to be structurally enforced, not just intended, given this project's own track record of catching subtle-but-real bugs in adjacent code (Phase 0's six documented bugs in the original LanCE release).
6. **§4.5 prototype bank + §4.7 drift monitor**, whichever variant §1.2 resolved to. These two are coupled (the drift monitor scores against the bank) and both depend on knowing the exemplar-storage philosophy first.
7. **Full-system integration + the missing benchmark (§3).** Last, not first — the point in the literature where "components each work in isolation" and "the full pipeline works end-to-end, with all the interaction effects that implies" diverge is exactly the point most vulnerable to silent bugs, and this project's own history (Phase 0, Component 1's L1→L2 substitution, several of Component 2's early overclaims later corrected) is a consistent argument for validating pieces before trusting the whole.

---

## 5. What this document is not

This is not a commitment to build any of it. Per the project owner's own instruction, this is the scoping pass requested before any code gets written — the next decision is whether to proceed, and if so, starting where (§4 above gives a recommended order, not a mandate), and how §1.1/§1.2 get resolved.
