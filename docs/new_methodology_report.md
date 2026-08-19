# New methodology: our work on fixing what the diagnosis found

**Status: Component 1 done and validated. Component 2's diagnostic validated, fallback mechanism identified but not yet statistically confirmed. Component 3 done — mechanism validated as safe, not (yet) effective. Components 4–5 not started. Report updated as work continues.**

This document picks up exactly where [`docs/research_report.md`](research_report.md) leaves off. That report is the *diagnosis*: eleven experiments establishing that LanCE forgets earlier domains once a harder benchmark is used (Pillar 1), and that its frozen CLIP backbone and frozen descriptor list both have real, measured coverage gaps (Pillar 2). This document is the *treatment* — an actual proposed method, why each piece exists, what it's shown so far, and an honest account of what's proven versus what's still just well-motivated.

---

## 1. The plan

The plan was developed and approved as a standalone planning artifact before any of it was implemented; its content is reproduced here so this report is self-contained and stays in sync with what's actually been built.

**One method, five components, plus one flagged stretch novelty — not a patch list.**

> A concept bottleneck model that updates its classifier exactly as new domains arrive, and automatically knows when its own text-only guess about a new domain can't be trusted — switching to real examples instead, growing and pruning its own vocabulary, and falling back to direct supervision only when nothing else is trustworthy.

| # | Component | Failure it addresses | Status |
|---|---|---|---|
| 1 | An exact, no-forgetting classifier update | Naive fine-tuning forgets earlier domains (Phase B/D); every remedy tested (cumulative DDO, replay, EWC) is an approximation | ✅ **Done, validated** |
| 2 | Self-diagnosing domain grounding | DDO predicts a new domain's look purely from text, with measured near-zero reliability for domains that matter (Phase F1/F3/F4) | ⚠️ **Diagnostic validated; fallback mechanism identified, not yet statistically confirmed** (single run, effect size below this project's own demonstrated noise floor) — see `results/component2b_grounding_fallback_variants.md` |
| 3 | A vocabulary that grows and prunes itself | The 204-phrase descriptor list is frozen at t=0; zero phrases match AI-generated imagery (Phase E1); DDO's benefit collapses ~10x without coverage (Phase E2) | ⚠️ **Partial/mixed result** — filter correctly rejected all 10 candidates (none cleared the trust threshold), so growth was a safe no-op, not a fix, on photo→Midjourney v6 — see `results/component3_self_growing_vocabulary.md` |
| 4 | Domain memory that never stores raw images | Remedies that work well (replay) need to keep real examples from old domains around — a real problem for sensitive domains like medical imaging | Not started |
| 5 | Knowing when to stop trusting itself | Some domains (sculpture/3D, per the base paper's own numbers) stay hard even with full descriptor coverage — a structural wall, not a coverage gap | Not started |
| — | *Stretch: detecting a new domain arrived, without being told* | Every experiment so far assumes a domain boundary is handed to the method; real deployments often don't get that | Not started |

**Scope boundary, carried through unchanged:** every component stays on the domain axis. The class set stays fixed throughout; class-incremental behavior is paused by decision.

**One honest note carried through from the plan itself:** only Component 1 was ever claimed as provably certain — a theorem, not a trend. Components 2–5 are strongly motivated by evidence already collected in Pillars 1–2, but "strongly motivated" and "proven" are different things until each is actually run.

---

## 2. What we've done so far

### 2.1 Component 1: implementation

Built [`external/LanCE/model/analytic_classifier.py`](../external/LanCE/model/analytic_classifier.py) — `AnalyticDomainIncrementalClassifier`. The derivation, in short (full version in the module's own docstring):

- With CLIP and the concept embeddings frozen, the *only* thing LanCE's `clip_cbm_orth` ever trains is `Sequential(Flatten, LayerNorm(M), Linear(M, N))`. LayerNorm's normalization step has no learnable parameters — it's a fixed, per-sample transform. Its learnable affine step, composed with the following Linear layer, collapses algebraically into a single affine map from normalized concept activations to class logits. So the entire trainable pipeline, regardless of how it's parameterized, is exactly one linear regression.
- A linear regression fit by least squares has an exact incremental form: keep running sums `R = Σ φφᵀ` and `Q = Σ φyᵀ` (φ = normalized activations plus a bias term), and solve `W = R⁻¹Q`. Because matrix addition is commutative, the classifier after N domains — in *any* order — is numerically identical to solving on all N domains pooled at once.
- **The one deliberate, stated adaptation:** DDO's orthogonality penalty (the term that pushes domain-descriptor directions toward predicting nothing) is originally an L1 (mean-absolute) term in the base method. L1 penalties don't have a closed-form incremental solution. This module substitutes the L2 (mean-square) version, which folds cleanly into `R` as one more additive ridge-style term. This is flagged, not hidden — see §5 for what's still unverified about this substitution.
- A second implementation detail worth recording: `domain_diffs` (the per-descriptor, per-anchor-class text-embedding differences) is a 3D tensor `(num_descriptors, num_anchor_classes, feature_dim)`, not 2D as first assumed — caught by a runtime shape error on the first run, fixed by flattening the leading two dimensions before projecting through the concept embeddings, which is exactly what the original model's `classifier[1:]` does implicitly via broadcasting.

Built two validation harnesses, [`experiments/component1_analytic_domain_il.py`](../external/LanCE/experiments/component1_analytic_domain_il.py) (PACS) and [`experiments/component1_analytic_domain_il_officehome.py`](../external/LanCE/experiments/component1_analytic_domain_il_officehome.py) (Office-Home), both reusing `DomainILSession`'s existing cached CLIP embeddings from Phases B/D — no new data pass, no new CLIP encoding, so results are directly comparable to the original numbers.

### 2.2 Dataset acquisition

Checked the Downloads folder and existing `data/` directories first (nothing dataset-related sitting unused). Then:

| Dataset | Outcome |
|---|---|
| AWA2 (13GB) | Downloaded from the official host (`cvml.ist.ac.at`), extracted, verified — image paths resolve against the existing (previously unused) loader code. Hit and fixed a Windows case-insensitive-filesystem collision that nested images one level too deep. |
| DomainNet (6 zips, ~18GB) | Downloaded from the official host (`csr.bu.edu`), extracted — 345 classes confirmed present in each of the 6 domains. No loader written yet (separate task, see §6). |
| Camelyon17-WILDS (~10GB) | Swapped in as the medical dataset (replacing the originally planned chest-X-ray combination) once found to be pip-installable, no-registration, and purpose-built for multi-hospital domain shift. Blocked: its host (`worksheets.codalab.org`) returns HTTP 500 on direct verification, confirmed independently of the `wilds` downloader — a server-side outage, not something a retry fixes. |
| GenImage (full) | Investigated, not downloaded. The only fuller mirror found is an unofficial 607GB community copy of unconfirmed label fidelity; the official version is Baidu-Netdisk-only. Judged a bad trade for a supplementary confirmation dataset — flagged rather than downloaded silently. |
| LADA-Sculpture, CheXpert, MIMIC-CXR | Not downloaded — each requires registration/credentialing under the project owner's own identity. Cannot be automated. |

---

## 3. Why we did this, and what issue it traces back to

Phase B found forgetting, but only in 1 of 3 PACS domain orderings — PACS's near-ceiling accuracy (98.3% joint) left too little headroom for the effect to show reliably. Phase D, on the harder Office-Home benchmark, found the real version of the problem: **every single domain ordering tested showed real, consistent negative backward transfer (−0.68 to −4.68)** — not an occasional fluke, a structural gap.

Phase B.1/C already tried the standard toolkit against this: cumulative DDO, cached-embedding replay, and EWC. On PACS's one real case, cumulative DDO and replay closed it almost fully (−8.30 → −0.07 / −0.96). On Office-Home, the same fixes only got partway — cumulative DDO/replay left −1.6 to −3.3 BWT behind in the harder orderings, and EWC, while driving BWT near zero, cost 7–12 accuracy points (the classic stability–plasticity tradeoff of a soft regularizer that's fighting learning as hard as it's fighting forgetting).

**The issue Component 1 was built to solve, precisely:** every one of those three remedies is an *approximation* — a penalty term, a partial replay buffer, a Fisher-information estimate — and approximations have residual error that gets worse as the benchmark gets harder. The question Component 1 asks is narrower and sharper: given that the trainable part of this specific architecture is mathematically just a linear regression (once CLIP and the concept embeddings are frozen, which they always are in this method), is there a fix with *zero* residual error, not a smaller one?

---

## 4. What the results show

### PACS ([`results/component1_pacs_results.json`](../results/component1_pacs_results.json))

| Domain order | Original SGD BWT | Component 1 BWT | Component 1 max\|diff from joint\| |
|---|---|---|---|
| photo → art → cartoon → sketch | **−8.30** | −0.15 | **0.0000** |
| sketch → cartoon → art → photo | −0.54 | −0.18 | **0.0000** |
| art → sketch → photo → cartoon | −0.26 | −0.03 | **0.0000** |

### Office-Home ([`results/component1_officehome_results.json`](../results/component1_officehome_results.json)) — the benchmark that mattered

| Domain order | Original SGD BWT | Original best remedy (BWT) | Component 1 BWT | Component 1 max\|diff from joint\| |
|---|---|---|---|---|
| art → clipart → product → real world | −0.68 | +0.33 (cumulative DDO) | −0.49 | **0.0000** |
| real world → product → clipart → art | −2.39 | −1.63 (replay) | −0.34 | **0.0000** |
| clipart → real world → art → product | −4.68 | −3.06 (replay) | −1.45 | **0.0000** |

**"max\|diff from joint\|" is the number that matters most.** It's the largest absolute difference, across every domain and every ordering, between (a) the classifier you get training sequentially, one domain at a time, and (b) the classifier you get training on all domains pooled at once — the textbook non-continual upper bound. A value of exactly 0.0000 means the sequential classifier isn't *close to* the joint one, it's *identical* to it, down to floating-point solve precision. Final accuracy in every ordering, on both datasets, lands at exactly the same number (98.51% on PACS, 89.03% on Office-Home) — order stops mattering entirely.

The residual BWT values that remain (−0.03 to −1.45) are **not forgetting** in the sense Phase B/D measured. BWT compares a domain's accuracy right when it finished training against its final accuracy after later domains join. Since the classifier is being incrementally refit toward the *true joint optimum* as each new domain's data arrives, a domain's own accuracy naturally shifts a little as more data joins the fit — the same way a running average shifts as new numbers are added. That shift is bounded and exactly explained; it isn't decay of previously-learned information, which is what the original SGD BWT numbers were measuring.

---

## 5. Did it solve the problem? How much, and why

**Yes, on the specific problem Phase B/D measured (naive sequential fine-tuning forgets earlier domains), and completely, not partially — on both benchmarks tested, including the one that resisted the standard fixes.**

Why it's complete rather than partial: the standard remedies (cumulative DDO, replay, EWC) all work by making forgetting *smaller* — a better penalty, a bigger buffer, a stronger constraint. Component 1 doesn't make the residual error smaller; it removes the mechanism that produces residual error in the first place, because the update is derived to be mathematically identical to joint retraining rather than an approximation of it. That's the entire reason the empirical result on Office-Home (the hard case) is as clean as the result on PACS (the easy case) — the guarantee doesn't depend on how much headroom the benchmark leaves, which is precisely the property none of Phase C's three remedies had (their Office-Home numbers were all worse than their PACS numbers).

**What this result does *not* claim to have solved:**
- It doesn't touch Pillar 2's backbone-coverage or descriptor-staleness problems (Components 2–3's job, not started).
- It doesn't establish that the L2-surrogate DDO term produces the *same trained model* as the original L1 version — only that, whichever version is used, the incremental update matches joint retraining of that same version exactly. Whether the L2 substitution changes real classification accuracy compared to the original L1-trained SGD baseline is untested. This is the most important open question about Component 1 specifically (see §7).
- It's validated on 4-domain, 7–65 class benchmarks. The plan's own dataset roster calls for confirming this holds at DomainNet's scale (6 domains, 345 classes) before treating "holds at any scale" as established rather than expected.

---

## 6. Literature check — what's already published, and what isn't

Before treating any of this as a contribution, we searched for prior work matching each piece. Short version: **the general technique underneath every component already exists, in some form, in the published literature.** What (if anything) is genuinely unclaimed is a narrower, specific combination — and even that narrower claim isn't confirmed absent, only not found in the searches done so far.

**Component 1 (exact classifier update).** The update rule itself — closed-form ridge regression instead of gradient descent, to get an update provably identical to joint retraining — is an active area called analytic continual learning (e.g. ACIL and its descendants). More directly relevant: [**CONCIL**](https://arxiv.org/abs/2411.17471) (Nov 2024, now published at ACM MM 2025) already applies exactly this closed-form approach *to concept bottleneck models specifically*, to eliminate catastrophic forgetting without gradient descent. What we haven't found anywhere: CONCIL targets concept-incremental/class-incremental learning (new concepts/classes arriving, same visual domain throughout — the same distinction this project's own original literature review in `docs/lance_continual_dg_failure_analysis.md` §3 already flagged) and has no analog of DDO's language-guided domain-orthogonality penalty. The narrower, unconfirmed claim that remains: applying this closed-form approach to *domain*-incremental arrival specifically, with DDO folded into the same solve. Component 1's measured result (exact match on PACS/Office-Home) is real and reproducible regardless of this — what changes is only the novelty framing, not the finding itself.

**Idea 2 (augmentation-expanded domains, the guide's suggestion).** Style-transfer/corruption-based domain synthesis is standard in domain generalization (MixStyle, AdaIN-based style transfer), and has specifically been used before to build longer synthetic domain sequences for studying forgetting — a benchmark called DomainCIFAR-100, from ["Make Domain Shift a Catastrophic Forgetting Alleviator in Class-Incremental Learning"](https://arxiv.org/html/2501.00237v1) (Jan 2025). This should be presented as an experiment applying an established technique to our setting, not a contribution.

**Idea 3 (self-diagnosing domain grounding).** Confidence/alignment-gated fallback for CLIP under distribution shift is an active area — but specifically in domain *adaptation*, where some target-domain data is generally assumed available to adapt with (e.g. ["Towards Fine-Grained Adaptation of CLIP via a Self-Trained Alignment Score"](https://arxiv.org/pdf/2507.09615)). Not found: applied to continual domain *generalization*, where the backbone stays frozen and no target data exists until a domain arrives, with the decision made live. Worth testing whether a domain-adaptation idea transplants into this more restricted setting.

**Idea 4 (self-growing vocabulary).** Dynamic, autonomously-discovered concept sets already exist for CBMs — Caption Bottleneck Models, Flexible CBM (a hypernetwork that dynamically incorporates new concepts). Not found: applied to a domain-descriptor pool specifically (as opposed to class-describing concepts), in a continual domain-generalization setting.

**What this means for how to present the work:** Component 1, 3, and 4 are best framed as *"we adapted an existing, established technique to a specific architecture and setting that, as far as we can find, hasn't been tested this way"* — not as inventing new machinery. Idea 2 is best framed plainly as an experiment with an existing technique. This is not a weaker story than claiming novelty outright — reviewers check exactly this, and a precise, defensible scope claim holds up better than an overstated one.

**This check is not exhaustive.** It reflects a handful of targeted searches, not a systematic literature review — analytic continual learning especially is publishing multiple new papers a month as of this writing. Treat every "not found" above as "not found in this pass," not as "confirmed absent."

---

## 7. Improvements that could be made on top of this

1. **Directly compare L1 (original, SGD-trained) vs. L2 (this module's) DDO on identical data.** Right now Component 1 is only compared against the original SGD baseline's *forgetting* behavior, not its *absolute accuracy* under matched conditions. A clean ablation — same domains, same order, L1-SGD vs. L2-analytic, both evaluated on final accuracy and on the actual orthogonality property DDO is meant to enforce — would close the one real gap in the "exact fix" story.
2. **Run it on DomainNet once the loader exists.** 345 classes and 6 domains is a meaningfully different regime (the R matrix is `(M+1)×(M+1)` regardless of class count, so this is a cheap experiment, not a scaling risk — but "cheap in theory" and "confirmed in practice" are different things).
3. **The synthetic long-domain-sequence stress test from the plan (15–20+ domains via style-transfer/corruption variations) hasn't been run.** This is where the plan expected Component 1's claim to be pushed hardest, and where Component 3's later pruning logic needs a testbed with known ground-truth boundaries anyway — worth doing before Component 3 starts, not after.
4. **Numerical conditioning at scale.** The module solves `R⁻¹Q` from scratch at every stage in float64 rather than using a rank-1 (Sherman-Morrison) update chain, specifically to avoid precision drift — worth a stress test with many more domains/stages to confirm this holds, rather than assuming it from the derivation alone.

---

## 8. Suggestions for what's next

Ranked by what unblocks the most downstream work per unit of effort:

1. **Run the L1-vs-L2 DDO ablation (§7.1) before treating Component 1 as fully closed.** Cheap (reuses existing cached embeddings and the existing SGD training script), and it's the one open question that could change the L2-substitution's framing from "a deliberate, low-risk adaptation" to "a change worth reconsidering." — ✅ done, see [`results/component1b_l1_vs_l2_ablation.md`](../results/component1b_l1_vs_l2_ablation.md).
2. **Write the DomainNet loader.** Unblocks Component 1's strongest scale test and doesn't depend on anything else being finished first. — ✅ done; the scale-test run itself is in progress (see `docs/session_handoff.md`).
3. **Retry Camelyon17 (or download it manually from `wilds.stanford.edu/downloads`).** It's the dataset every one of Components 2, 3, and 4 is designed around most directly (the no-raw-image-storage argument in Component 4 specifically needs a sensitive-data domain to be a meaningful claim, not just a hypothetical one). — still blocked (host outage), superseded for Components 2–3 by Defactify/PACS/EuroSAT, which turned out sufficient for both.
4. **Start Component 2** — ✅ done, see [`results/component2_self_diagnosing_grounding.md`](../results/component2_self_diagnosing_grounding.md) and its follow-up [`results/component2b_grounding_fallback_variants.md`](../results/component2b_grounding_fallback_variants.md) (diagnostic + a diversity-preserving/blended fallback both validated, on one domain shift).
5. **Start Component 3** — in progress, see §9.

---

## 9. Combined / ablation testing across components — planned, not yet run

**This section exists because the project owner raised it directly and asked that it not get lost:** since each component fixes a *different* problem (Component 1: forgetting; Component 2: untrustworthy text-only domain grounding; Component 3: a frozen descriptor vocabulary), when do we test them **combined**, not just individually? The natural comparison is a real ablation table — original baseline, each validated component alone, validated components combined, eventually all of them together — not a set of isolated single-component results presented as if they were independent proof points for one method.

**Status: not started as an experiment.** This section is the "it's now a real, written part of the plan" step the commitment asked for — the run itself still needs Component 3 (and ideally Component 4) to reach at least a first result before a combined condition is meaningful. With Component 1 done and Component 2's diagnostic validated (fallback mechanism identified, not yet statistically confirmed), there is already enough validated to define what the table looks like:

| Condition | What it tests |
|---|---|
| Original SGD baseline (no components) | The paper's own untouched method, for reference |
| Component 1 alone | Exact incremental update, no domain-grounding or vocabulary changes |
| Component 2 alone | Self-diagnosing grounding on top of ordinary SGD training (not the exact classifier) |
| Component 3 alone | Growing/pruning vocabulary on top of ordinary SGD training |
| Component 1 + Component 2 | Exact update *and* grounded domain_diffs together |
| Component 1 + Component 3 | Exact update *and* a growing vocabulary together |
| All validated components combined | The full proposed method, as far as it's been built |

**Design questions still open, to resolve before this is run rather than during it:** which benchmark(s) — Office-Home for the forgetting axis (Component 1's own hard case) and Defactify for the grounding/vocabulary axis (Components 2/3's shared testbed) may need to be combined into one multi-domain sequence rather than run separately, since Component 1's contribution only shows up across a *sequence* of domains while Components 2/3's contributions show up on a *single* new-domain arrival; and whether Component 2 and Component 3 (both modify `domain_diffs`/the descriptor pool DDO regularizes against) compose cleanly or need an explicit merge rule. Revisit once Component 3 has a first result.
