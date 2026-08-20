# Component 4 — domain memory that never stores raw images (or per-sample data)

## Origin

`docs/new_methodology_report.md` §1, row 4 of the 5-component table: *"Domain memory that never stores raw images — remedies that work well (replay) need to keep real examples from old domains around, a real problem for sensitive domains like medical imaging."* Fourth component.

## The issue this targets

Phase B.1/C found that **replay** (`ReplaySession` in `external/LanCE/experiments/remediation.py`) is one of the two remedies that actually helps against forgetting — nearly closes PACS's one real forgetting case (−8.30 → −0.96 BWT) and helps substantially on Office-Home (−2.39→−1.63, −4.68→−3.06). But its mechanism, checked directly in the code (`ReplaySession._build_stage_loader`, lines 89–104): for every prior domain, keep a fixed-size buffer (`REPLAY_BUFFER_SIZE_PER_DOMAIN = 100`) of **individually sampled, real cached CLIP embeddings**, and mix them into every later training stage, unchanged, for the rest of the run.

This is not raw pixels — but each stored 768-dim vector still traces back 1:1 to one specific real training example. For an ordinary photo domain that's a non-issue. For a domain like Camelyon17 (hospital tissue slides) or CheXpert (patient X-rays), it's still "keep a linkable per-patient record around indefinitely," just in embedding form instead of pixel form — the governance problem (data retention, re-identification risk, right-to-be-forgotten obligations) doesn't meaningfully change just because the record is a vector instead of an image.

**What Component 4 asks:** can most of replay's forgetting-reduction benefit be recovered from something that never retains *any* per-sample, per-record data past the moment a domain finishes training — only a domain-level, class-conditional statistical summary?

## Why we tried this approach specifically

The obvious, minimal change that actually addresses the stated privacy concern (rather than a smaller buffer, which is still per-record data, just less of it): replace `ReplaySession`'s per-sample buffer with a **per-domain, per-class Gaussian summary** (mean + diagonal variance in CLIP-embedding space) computed once from a domain's own cached training features, and generate *synthetic* replay samples by drawing from that Gaussian at every later stage — no individual real example, nor its embedding, needs to survive past the moment its domain's summary is computed. This is not a novel invention: class-conditional Gaussian feature-space replay without real exemplars is an established technique in exemplar-free class-incremental learning (e.g. the "prototype + Gaussian augmentation" family — PASS, CVPR 2021, is the closest published match). The honest framing, consistent with this project's own literature-check convention (`docs/new_methodology_report.md` §6): this is an existing technique, applied here to a domain-incremental setting and framed explicitly around the privacy/data-retention motivation, not a new algorithm.

## Method

1. **`GaussianReplaySession`** (subclasses `DomainILSession`, same seam `ReplaySession` uses — `_build_stage_loader`): the first time a prior domain `d` is needed for replay at some later stage, compute `{class_id: (mean, std)}` from `self.caches[(d, "train")]` — per-class mean and per-dimension standard deviation (floored at 1e-3 to avoid degenerate near-zero-variance dimensions), memoized so it's computed once. At every stage needing replay from `d`, draw synthetic feature vectors `mean + noise * std` (`noise ~ N(0, I)`), split evenly across `d`'s classes, totaling the same `REPLAY_BUFFER_SIZE_PER_DOMAIN` budget `ReplaySession` uses, for a fair per-domain memory-budget comparison.
2. **One flagged design choice**, stated plainly per this project's convention: **diagonal covariance, not full covariance.** A full 768×768 covariance per class per domain is both expensive and severely under-determined from the ~dozens-to-low-hundreds of samples typically available per class per domain (far fewer samples than dimensions). Diagonal covariance is the standard simplification in the literature this borrows from (PASS and similar). It will understate any real correlation structure between CLIP feature dimensions — flagged as the one place this mechanism is a simplification of the ideal, not a hidden shortcut.
3. **Second flagged difference from `ReplaySession`:** `ReplaySession` samples its 100-example buffer uniformly at random across a domain's *pooled* training examples (so class representation follows the domain's natural class balance). `GaussianReplaySession` splits its budget *evenly* across classes by construction. Both are legitimate design choices, not a bug — but it means the two mechanisms' synthetic/real buffers aren't class-distribution-identical, worth noting rather than silently assuming equivalence.
4. **Validation**: reuse Phase B.1/C's and Phase D's own harness and metrics exactly (`domain_il.py`'s `compute_acc_bwt`, same domain orderings, same 50-epochs/stage protocol) on **both PACS and Office-Home** — mirroring Component 1's own two-benchmark depth (the easy case where forgetting barely shows, and the hard case where it's consistent across every ordering) since both harnesses already exist and reuse the same cached embeddings, at effectively no extra cost. Three conditions per benchmark per ordering: naive sequential (no memory), real per-sample replay (existing `ReplaySession`), Gaussian-summary replay (new). The empirical question: does BWT recovery from Gaussian-summary replay come close to real replay's, or does discarding per-sample identity cost real forgetting-resistance?

## Dataset(s) used, and why

**PACS and Office-Home**, both already fully set up for this exact comparison (Phase B/C and Phase D/C already ran naive-sequential and real-replay on both, with cached CLIP embeddings already on disk) — reusing them keeps Component 4's numbers directly comparable to Phase B.1/C/D's own already-published BWT figures, the cleanest possible baseline. No new dataset needed; the actual sensitive-data domain this component is ultimately motivated by (Camelyon17 or equivalent) remains blocked (host outage) or requires the project owner's own credentialed registration — the mechanism is validated here on an architectural, dataset-independent question (does a Gaussian summary recover what a per-sample buffer recovers), with the medical-domain motivation stated as the reason this matters, not as something empirically confirmed on medical data in this run.

## Code (planned)

- `external/LanCE/experiments/component4_gaussian_replay.py` — `GaussianReplaySession` class + PACS 3-condition comparison (naive / real-replay / Gaussian-replay).
- `external/LanCE/experiments/component4_gaussian_replay_officehome.py` — same comparison on Office-Home, importing `GaussianReplaySession` from the file above (mirrors `remediation_officehome.py`'s own import-from-`remediation.py` pattern).

## Status

Not yet run. This file records intent; results go in `results/component4_domain_memory.md` once actually executed, per `docs/component_report_template.md`.
