# Component 3 — a vocabulary that grows and prunes itself

## Origin

`docs/new_methodology_report.md` §1, row 3 of the 5-component table: *"A vocabulary that grows and prunes itself — the 204-phrase descriptor list is frozen at t=0; zero phrases match AI-generated imagery (Phase E1); DDO's benefit collapses ~10x without coverage (Phase E2)."* Third component, building directly on Component 2's diagnostic machinery.

## The issue this targets

DDO's orthogonality regularizer (`clip_cbm_orth` in `external/LanCE/model/cbm_models.py`) computes `domain_diffs` from a **fixed, 204-entry pool of hand-authored text templates** (`external/LanCE/prompts/prompt200new.py`'s `target_text_prompts`) — written once by GPT-3.5-turbo before the method ever runs, never updated afterward. Phase E1 found this pool contains **zero** terms for any post-cutoff AI-image-generation domain (0/20 direct terms, 0/5 named generators checked) even though it has plenty of conceptually-adjacent terms ("a digital art of a {}.", "a CGI render of a {}."). Phase E2 measured the real cost: DDO's benefit over a plain classifier on photo→Midjourney v6 is +0.68 points, versus +6.40 points on a domain the pool actually covers (CUB→Painting) — roughly a tenth.

**What's missing:** nothing in the architecture requires this pool to be a static, hand-curated, one-time artifact. `clip_cbm_orth.forward` only ever consumes `self.diffs @ self.concept_embeddings.T`, and the leading `num_directions` dimension is averaged away in the orthogonality loss (`torch.abs(reg).mean()`) — so the pool can grow (add directions) or shrink (drop directions) between training runs with no model change, exactly the same way Component 2 showed the *source* of a direction can change (text vs. image) with no model change. Component 3 is the vocabulary-side counterpart: automatically add descriptor phrases for a domain the pool doesn't cover, and prune the pool so it doesn't grow without bound as more domains arrive.

## Why this approach specifically

Two design constraints, both taken directly from how this project has treated similar problems so far:

- **No live LLM API call.** Phase E1 deliberately inspected the already-generated, checked-in pool rather than pay for a fresh GPT-3.5-turbo call, and flagged that choice openly. Component 3 keeps the same discipline: growth uses cheap, deterministic **string templating** off the arriving domain's own name (something the method already has the moment a new domain arrives, before any images are labeled) — not a paid generative call. This is a real, stated substitution for "have an LLM write new descriptors," not a hidden shortcut, and it is flagged as such (see Method and the report's own §5/§11 once written).
- **Reuse Component 2's already-calibrated trust threshold and probe machinery**, rather than inventing a second one. Candidate descriptors are only worth adding if they'd actually raise the model's grounding of the new domain — and Component 2 already built and calibrated (`results/component2_alignment_calibration.json`, threshold 0.1431) exactly the alignment-score check needed to test that, on the same probe images, with no new data collection.

## Method

1. **Growth.** When a new domain arrives (named, e.g. `"Midjourney v6"`), generate a small set of candidate descriptor templates by filling a fixed list of generic patterns with the domain's name (`model/vocabulary_growth.py`'s `CANDIDATE_TEMPLATE_PATTERNS`, e.g. `"a {domain} image of a {}."`, `"an AI-generated {domain} image of a {}."`). Score each candidate with Component 2's `compute_alignment_score` formula (visual shift from the probe vs. textual shift from the candidate template), using the *same* probe images Component 2's diagnostic already draws — no extra data. Keep only candidates whose alignment meets or exceeds the calibrated threshold (0.1431), so a candidate is added because it's measured to actually connect to the arriving domain's real images, not because it was merely generated. Append survivors (deduplicated against the existing pool) to `target_text_prompts`, permanently, for this and all later domains.
2. **Pruning.** As the pool grows across domains, redundant phrases accumulate (e.g. a grown `"an AI-generated image of a {}."` sitting near the existing `"a digital art of a {}."` in embedding space). `prune_redundant_descriptors` embeds every pool entry (template filled with a neutral placeholder noun), computes pairwise cosine similarity, and greedily drops later-added entries whose similarity to an earlier-kept entry exceeds a redundancy threshold (0.97) — bounding pool growth without needing per-domain human review. Originally-shipped (paper) entries are always preferred over grown ones when a redundant pair is found, so growth never silently displaces the base pool.
3. **Validation.** Rerun Component 2/Phase E2's exact Defactify (photo→Midjourney v6) baseline-vs-+DDO protocol, adding a fourth condition: +DDO with the **grown** pool (original 204 + validated new Midjourney-specific phrases) in place of the untouched 204-entry pool. Compares against baseline (α=0), +DDO-text (original pool, reproduces E2/C2's +0.68), and — since the probe is already being drawn — also reports Component 2's own grounded-fallback condition for a complete 4-way picture on one consistent split. The empirical question: does growing the vocabulary recover more of the +6.40-vs-+0.68 gap than Component 2's image-grounded fallback did (which made things *worse*, not better — 73.81% vs. baseline 74.27%)?

## Dataset(s) used, and why

**Defactify/MS-COCO-AI** (photo→Midjourney v6): identical to Component 2's and Phase E2's setup, deliberately — this keeps the new condition directly comparable to two already-measured numbers (+0.68 text-DDO gain, and Component 2's -0.46-point grounded-fallback result) rather than introducing a new, harder-to-compare benchmark for a component whose whole claim is about *this exact* coverage gap.

## Code (planned)

- `external/LanCE/model/vocabulary_growth.py` — candidate generation, alignment-based filtering (growth), redundancy-based pruning.
- `external/LanCE/experiments/component3_defactify_growing_vocab_ddo.py` — 4-condition training comparison (baseline, +DDO-text, +DDO-grounded, +DDO-grown), reusing Component 2's data pipeline.

## Status

Not yet run. This file records intent; results go in `results/component3_self_growing_vocabulary.md` once actually executed, per `docs/component_report_template.md`.
