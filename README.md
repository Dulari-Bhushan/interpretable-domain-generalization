# CBMs for Continual Domain Generalization

Research project investigating whether **concept bottleneck models (CBMs) for domain generalization survive a continual-learning setting** — i.e. domains arriving one at a time, over time, rather than all at once during a single training run.

## Starting point: LanCE

The most relevant prior work is **LanCE** ([Zeng et al., CVPR 2025](https://arxiv.org/abs/2503.18483), [original code](https://github.com/joeyz0z/LanCE)) — a CLIP-based CBM that erases domain-specific concepts from its final classifier using a language-guided "Domain Descriptor Orthogonality" (DDO) loss. It gets strong domain-generalization results, but it trains once, on one domain, and never updates again. This project argues (and is in the process of measuring) that this design cannot survive domains arriving continually — see [`docs/lance_continual_dg_failure_analysis.md`](docs/lance_continual_dg_failure_analysis.md) and the rendered brief in [`presentation/lance_failure_brief.html`](presentation/lance_failure_brief.html) for the full argument.

## Repo layout

| Path | What's there |
|---|---|
| [`docs/`](docs) | [`research_report.md`](docs/research_report.md) — the full, consolidated research report (start here). Also: the failure-mode analysis of LanCE (five failure modes, each traced to a specific equation/table in the paper) — **updated throughout with every phase's actual measured result**, not just the original predictions |
| [`planning/`](planning) | The two research planning documents developed for this project, kept as a live record including corrections made along the way (datasets considered and rejected, scope refinements, etc.) — not just the final plan |
| [`external/LanCE/`](external/LanCE) | Vendored copy of the LanCE reference implementation, with bug fixes applied (see below), plus every new dataset loader and experiment driver script built for this project (see "Experiment code map" below) |
| [`results/`](results) | Write-up (`.md`), raw numbers (`.json`), and figures (`figures/*.png`) for every completed phase |
| [`presentation/`](presentation) | Rendered HTML artifact summarizing the failure-mode analysis for slide-building — also updated with real numbers |

## New methodology: fixing what Pillars 1 & 2 found (in progress)

Everything above (Phases 0–D, F1–F4, E1–E2) is the **diagnosis**: it establishes that LanCE forgets earlier domains once a harder benchmark is used, and that its frozen CLIP backbone + frozen descriptor list both have real, measured coverage gaps. The project's second half is the **treatment**: an actual proposed method — not a patch list — with each piece addressing one measured failure. Full plan, reasoning, and status: [`docs/new_methodology_report.md`](docs/new_methodology_report.md) (start there for the self-contained "what we did, why, did it work" narrative).

Status so far:
- ✅ **Component 1 — exact, no-forgetting classifier update:** implemented and validated on both PACS and Office-Home using the existing cached embeddings. Result: **max difference from the joint/oracle fit = 0.0000** on every domain ordering tested, on both datasets — including Office-Home, the benchmark where the original SGD-trained baseline showed real forgetting (BWT −0.68 to −4.68) that the standard remediations only partially fixed. See `results/component1_pacs_results.json`, `results/component1_officehome_results.json`, and the full writeup in `docs/new_methodology_report.md`.
- ⏳ Components 2–5 (self-diagnosing domain grounding, self-growing/pruning vocabulary, no-raw-image domain memory, confidence-gated fallback): not started — waiting on the medical dataset and a DomainNet loader.
- **New datasets:** AWA2 (13GB) and DomainNet (6 domains, 345 classes, ~18GB) are downloaded and, for AWA2, verified working end-to-end. Camelyon17-WILDS (the planned medical dataset — 5 real hospitals, histopathology, no registration needed) is blocked by a server-side outage on its host (`worksheets.codalab.org` returning HTTP 500); LADA-Sculpture, CheXpert, and MIMIC-CXR all need registration under your own identity and can't be automated. Full breakdown in `docs/new_methodology_report.md`.
- **Literature check:** the mechanism behind each component already exists in some form in published work — most notably [CONCIL](https://arxiv.org/abs/2411.17471), which already applies closed-form/analytic continual learning to concept bottleneck models, just for concept/class-incremental learning rather than domain-incremental. See `docs/new_methodology_report.md` §6 for the full breakdown of what's already published versus what (narrowly) isn't confirmed anywhere yet.
- ⏳ **A parallel thread, not yet started:** does it matter *where* concept scores come from — CLIP similarity (current), a directly-trained classifier, or a pretrained open-vocabulary detector (Grounding DINO/OWL-ViT) grounded on the same concept list? Full staged plan (CUB sanity check → PACS/Office-Home → the domains CLIP was shown to struggle on) in [`planning/03-detector-grounded-concept-extraction-plan.md`](planning/03-detector-grounded-concept-extraction-plan.md).

## Current status

**Pillar 1 (primary): does LanCE have any mechanism to survive domains arriving over time?**
- ✅ **Phase 0 — baseline reproduction (CUB → CUB-Painting):** PASS. See [`results/phase0_cub_reproduction.md`](results/phase0_cub_reproduction.md). Our reproduction (baseline 50.64%, +DDO 57.04%) lands within tolerance of the paper's own Table 1 numbers (50.54% → 55.53%), confirming the codebase — after fixing five plumbing bugs in the released code — is trustworthy for everything built on top of it.
- ✅ **Phase A — closed-world descriptor assumption test:** done, but the predicted dose-response didn't materialize — see [`results/phase_a_descriptor_coverage.md`](results/phase_a_descriptor_coverage.md) for the honest null result and what it means for the rest of the plan (short version: it makes Pillar 2 more important, not less).
- ✅ **Phase B — Domain-IL sequential protocol (PACS):** done, weaker/more conditional than predicted — see [`results/phase_b_domain_il.md`](results/phase_b_domain_il.md). Naive sequential fine-tuning only shows real forgetting (BWT −8.3) for one of three domain orderings tested; the other two stay within ~1 point of the joint/oracle upper bound (BWT −0.5, −0.3). Real, order-dependent forgetting exists, but PACS's near-ceiling accuracy (98.3% joint ACC on 7 easy classes) leaves little room to see it clearly — raises the case for a harder benchmark (Phase D) and, alongside Phase A, for Pillar 2.
- ✅ **Phase C — remediation attempts:** done, and yes — mostly. See [`results/phase_c_remediation.md`](results/phase_c_remediation.md). Cumulative DDO and cached-embedding replay close Phase B's one real forgetting case (BWT −8.3 → −0.07 and −0.96) at little-to-no accuracy cost; EWC also controls BWT but costs 7–12 points of accuracy in two orderings (classic stability–plasticity tradeoff, likely an untuned λ). Textbook fixes work here — reinforces rather than weakens the case for Phase D and Pillar 2 carrying more of the overall argument, since PACS's easy task didn't stress the architecture enough to make its lack of a built-in update rule bite hard.
- ✅ **Phase D — repeat on Office-Home (65 classes), promoted from optional stretch:** done, and this is the strongest evidence yet. See [`results/phase_d_officehome.md`](results/phase_d_officehome.md). All 3 domain orderings show real, consistent negative BWT (−0.7 to −4.7) — unlike PACS, where 2 of 3 were near zero. The remediations that nearly fully fixed PACS's forgetting only partially close the gap here (cumulative DDO/replay leave −1.6 to −3.3 BWT behind in the harder orderings; EWC still zeroes BWT but at a bigger accuracy cost). Confirms PACS's near-ceiling accuracy (98.3% joint ACC) was masking a real, harder-to-fix effect, not indicating robustness.

**Pillar 1 (Phases 0–D) is done. Pillar 2 (F1–F4) is done.**

**Pillar 2 (secondary): does the frozen CLIP backbone itself have a shelf life, independent of the forgetting problem?**
- ✅ **Phase F1 — domain-shift alignment check on EuroSAT (modality scarcity):** PASS, large margin. See [`results/phase_f1_eurosat_alignment.md`](results/phase_f1_eurosat_alignment.md). Mean photo→satellite alignment score: **0.32**, vs. the paper's own reported 0.90–0.99 range for domains it handles well. Note: EuroSAT (2017–19) actually predates CLIP (2021) — this tests modality scarcity (satellite imagery rare in captioned web photos), not temporal novelty.
- ✅ **Phase F2 — concept-activation ceiling test on EuroSAT:** done, and the literal hypothesis was wrong — the more interesting finding. See [`results/phase_f2_eurosat_ceiling.md`](results/phase_f2_eurosat_ceiling.md). A trained CBM (no DDO) reaches **90.9%**, far closer to the 98.1% linear-probe ceiling than the ~60–64% zero-shot ceiling — the opposite of what "CBM inherits CLIP's alignment weakness" predicted. Reframed finding: the *visual representation* is fine; what's specifically broken is *zero-shot, training-free* alignment — and DDO's mechanism is exactly a zero-target-data, text-only simulation with no way to recover the way a trained classifier does.
- ✅ **Phase F3 — domain-shift alignment check on genuinely post-cutoff generators (temporal novelty):** PASS, even more dramatically than F1. See [`results/phase_f3_temporal_novelty.md`](results/phase_f3_temporal_novelty.md). Dataset: Defactify/MS-COCO-AI (5 generators, all released 15–30+ months after both CLIP's and GPT-3.5's training cutoffs: SD 2.1, SDXL, SD3, DALL-E 3, Midjourney v6). Mean alignment: **0.05** (global) / **0.037** (per-class-controlled, both agree) — below even F1's EuroSAT result, essentially near zero. Uses real matched photo images this time (no text-as-photo-proxy needed), closing the adaptation gap F1 flagged.
- ✅ **Phase F4 — domain-shift alignment check on GenImage/Midjourney, a third independent dataset:** PASS, and reveals an important methodological nuance. See [`results/phase_f4_genimage_alignment.md`](results/phase_f4_genimage_alignment.md). Mean alignment: **0.23** across 155 ImageNet classes (partial download — only some of a multi-part Google Drive archive was recoverable; documented in full in the results file) — well below the paper's range, but *not* as low as F3's near-zero. Comparing all three: the two tests using CLIP text as a photo-domain stand-in (F1: 0.32, F4: 0.23) cluster together, while the one test with real matched photos (F3: 0.037) is an order of magnitude lower — suggesting the text-proxy methodology itself inflates the score, and F3's real-photo result is the most trustworthy of the three.
- ✅ **Phase E1 — descriptor-set staleness check (the "originally planned item 4" below, now run):** PASS, unambiguously. See [`results/phase_e1_descriptor_staleness.md`](results/phase_e1_descriptor_staleness.md). LanCE's actual shipped 204-descriptor pool (`prompts/prompt200new.py`) contains **0 of 20** direct AI-generation terms and names **0 of Phase F3's 5 generators** — distinct from F1/F3/F4, which test CLIP's alignment given words we supplied ourselves; this tests whether LanCE's own frozen, GPT-3.5-written descriptor list would ever have produced the right words in the first place.
- ✅ **Phase E2 — real baseline-vs-+DDO training run, real photos → Midjourney v6:** PASS, and more nuanced than a simple pass/fail. See [`results/phase_e2_defactify_ddo_training.md`](results/phase_e2_defactify_ddo_training.md). Using Defactify/MS-COCO-AI (23 COCO categories) and the *unmodified* 204-descriptor pool, DDO's gain over baseline collapses from Phase 0's **+6.40 points** (CUB→Painting, a domain the pool covers) to **+0.68 points** here (a domain it doesn't, per E1) — roughly a tenth. But trained accuracy on the new domain (74.7%) is *not* depressed relative to in-domain accuracy, despite F3's near-zero alignment score for this exact shift — training still recovers the signal (echoing Phase F2); what specifically disappears is DDO's own added value over that trained baseline.

**Both pillars are done. Phase G (folding every measured result into `docs/lance_continual_dg_failure_analysis.md` and the presentation artifact) is done. Phases E1–E2, the descriptor-staleness follow-up, are also done** — a full research report consolidating everything, including E1/E2, is at [`docs/research_report.md`](docs/research_report.md) (start there for the complete, self-contained write-up). See "Further plans" below for what's genuinely still open.

Full detail on why the plan is shaped this way — including datasets that were investigated and rejected (LADA-Sculpture's hidden Google-Drive/Baidu-Netdisk dependency, AWA2's 13GB download) — is in [`planning/02-continual-dg-experiment-plan.md`](planning/02-continual-dg-experiment-plan.md).

## Datasets used, and why

| Dataset | Used in | Size | Released | Why this one |
|---|---|---|---|---|
| **CUB-200-2011 → CUB-Painting** | Phase 0, Phase A | 5,990 / 5,790 / 3,047 imgs | CUB: 2011 | The paper's own primary benchmark — cleanest public download, needed first as a trust-gate before building anything on top |
| **PACS** | Phase B, C | 9,991 imgs, 4 domains, 7 classes | 2017 | Smallest standard 4-domain benchmark, cheap to iterate on — turned out to have a near-ceiling ~98% ceiling that masked forgetting, which is itself a finding (see Phase D) |
| **Office-Home** | Phase D | 15,588 imgs, 4 domains, 65 classes | 2017 | Deliberately harder than PACS (far more classes, far less data/class/domain) — built specifically to test whether PACS's weak forgetting signal was real or ceiling-masked |
| **EuroSAT** | Phase F1, F2 | 27,000 imgs, 10 classes | 2017–2019 | OpenAI's own CLIP paper already publishes an anchor number for it (59.6% zero-shot vs. 98.1% linear probe) — an independently-sourced ceiling to compare against. **Note**: predates CLIP, so this tests modality scarcity, not temporal novelty |
| **Defactify/MS-COCO-AI** | Phase F3 | 96,000 imgs (real COCO + 5 generators) | Generators: Dec 2022–2024 | All 5 generators postdate both CLIP's (~2020/21) and GPT-3.5's (~Sept 2021) training cutoffs by 15+ months — a genuine temporal-novelty test, with real matched photo images (not a text stand-in) |
| **GenImage/Midjourney** | Phase F4 | 928 imgs, 155 ImageNet classes (partial — see below) | Midjourney images: 2023 | The dataset originally identified as the ideal temporal-novelty test, dropped early on for access reasons, revisited once a Midjourney subset was located and downloaded directly |
| **AWA2** | Component 1 (planned) | 37,322 imgs, 50 classes | 2017 (v2) | A loader already existed in the vendored codebase from before this project (unused); downloaded now that dataset scale isn't a constraint — a third, ~free confirmation of Component 1 beyond PACS/Office-Home |
| **DomainNet** | Component 1 (planned) | ~0.6M imgs, 6 domains, 345 classes | 2019 | Downloaded (all 6 domain zips, ~18GB); a data loader (prep script + `*_data.py` + concept bank, following the existing per-dataset pattern) is still needed before it can be used in an experiment — the standard hard benchmark this subfield expects to see Component 1 tested against |
| **Camelyon17-WILDS** | Component 2 (planned) | ~10GB, 5 hospitals, 2 classes | 2019 | Multi-hospital histopathology domain-shift benchmark, purpose-built for exactly this kind of study, no registration required (`pip install wilds`) — currently blocked by a server-side outage on its host, not yet downloaded |
| **LADA-Sculpture, CheXpert, MIMIC-CXR** | Component 5 / Component 2 (planned) | — | — | Not downloaded — each requires registration/credentialing under the project owner's own identity (Google Drive/Baidu access, a Stanford AIMI account, and PhysioNet CITI training + a signed data-use agreement respectively); can't be automated |

Datasets are gitignored (large, and either publicly redistributable or regenerable). PACS/Office-Home/GenImage needed manual download (Google Drive); EuroSAT and Defactify download directly via `torchvision`/`datasets` with no manual step. See each phase's `results/phase_*.md` for exact download sources and any access complications encountered (GenImage's download in particular was a partial multi-part archive — documented in full in `results/phase_f4_genimage_alignment.md`).

## Experiment code map

All new code lives under `external/LanCE/`, reusing LanCE's own model/DDO-loss/training code as-is per the project's design principle (never reimplement what the paper already provides).

**Per-dataset loaders** (`data/<Dataset>/`), each following the same pattern (mirrors `data/CUB/cub_data.py`): a `prepare_*.py` one-time script that scans raw images and writes flat split manifests, a `*_data.py` PyTorch `Dataset` class, and a hand-written concept bank (`*_concepts.txt`):
- `data/PACS/` — `pacs_data.py`, `prepare_pacs_dataset.py`, `pacs_concepts.txt` (70 concepts, 7 classes)
- `data/OfficeHome/` — `office_home_data.py`, `prepare_office_home_dataset.py`, `office_home_concepts.txt` (257 concepts, 65 classes)
- `data/EuroSAT/` — `eurosat_concepts.txt` (40 concepts, 10 classes; loader is inline in `experiments/phase_f2_eurosat_ceiling.py`, dataset itself comes from `torchvision.datasets.EuroSAT`)
- `data/GenImage/` — `extract_genimage_subset.py` (pulls whichever classes' Midjourney images actually survived the partial multi-part archive download — 155/1,000 in practice), `imagenet_class_index.json` (standard class-index reference, verified empirically before trusting it)
- `data/Defactify/` — `defactify_concepts.txt` (76 concepts, 23 classes); dataset itself loads directly from Hugging Face (`Rajarshi-Roy-research/Defactify_Image_Dataset`) inline in `experiments/phase_e2_defactify_ddo_training.py`, no separate loader script needed

**Continual-learning harness** (`experiments/`), built for Phase B and reused unchanged everywhere after:
- `domain_il.py` — `DomainILSession`: joint/oracle training, naive-sequential Domain-IL training, ACC/BWT computation, the DDO-erosion mechanism metric. Generalized (domains/loader/cache-key are constructor params) so the same class runs PACS or Office-Home without modification.
- `remediation.py` — three subclasses of `DomainILSession` (`CumulativeDDOSession`, `ReplaySession`, `EWCSession`), each overriding only the hook points `DomainILSession` was built with (`between_stage_hooks`, `_compute_loss`, `_build_stage_loader`) — fully dataset-agnostic, no PACS-specific code.
- `domain_il_officehome.py`, `remediation_officehome.py` — thin driver scripts configuring the same classes for Office-Home instead of PACS.

**One-off analysis scripts** (`experiments/`), no training run, direct representation-level measurement:
- `phase_a_descriptor_coverage.py` — descriptor dose-response test (Phase A)
- `phase_f1_eurosat_alignment.py`, `phase_f2_eurosat_ceiling.py` — EuroSAT alignment score + trained-CBM ceiling test
- `phase_f3_defactify_alignment.py` — Defactify alignment score (global + per-class-controlled)
- `phase_f4_genimage_alignment.py` — GenImage/Midjourney alignment score
- `phase_e1_descriptor_staleness_check.py` — scans LanCE's shipped 204-descriptor pool for AI-generation terms (Phase E1)
- `phase_e2_defactify_ddo_training.py` — real baseline-vs-+DDO training run, real photos → Midjourney v6, using the unmodified descriptor pool (Phase E2, this one *does* train a model)

**New methodology** (`model/`, `experiments/`), built after Pillars 1 & 2's diagnosis was complete:
- `model/analytic_classifier.py` — Component 1: `AnalyticDomainIncrementalClassifier`, the exact incremental replacement for `clip_cbm_orth`'s trained classifier. Full derivation (why LayerNorm+Linear collapses to one linear map, why DDO's L1 penalty is substituted with an L2 surrogate to keep the update closed-form) is in the module's own docstring.
- `experiments/component1_analytic_domain_il.py`, `experiments/component1_analytic_domain_il_officehome.py` — validation harnesses reusing `DomainILSession`'s existing cached embeddings; compare the analytic classifier's sequential fit against a joint/oracle fit on the same data, per domain ordering.

**Shared infrastructure fix**: `external/LanCE/cache_utils.py` — found and fixed a real bug (`num_workers=8` in the caching `DataLoader` silently produced identical cached embeddings for every image once a split needed enough worker-dispatched batches on this environment; fixed to `num_workers=0`). Affected every phase from B onward until caught — full incident writeup in `results/phase_b_domain_il.md`.

**Every phase's own write-up** (`results/phase_*.md`) states its hypothesis, exact method, dataset, numbers, an honest interpretation (including when the literal hypothesis was wrong), and explicit limitations — read those directly for full methodological detail rather than relying on this summary.

## Further plans (genuinely open, not yet done)

- **A DomainNet data loader.** Data is downloaded (§ Datasets above); still needs a `prepare_domainnet_dataset.py`, `domainnet_data.py`, and a hand-written concept bank following the existing PACS/Office-Home pattern before Component 1 can be run against it.
- **Camelyon17 download**, once its host recovers (or via manual download from `wilds.stanford.edu/downloads`) — the planned dataset for Components 2–4.
- **Components 2–5** of the new methodology (self-diagnosing domain grounding, self-growing/pruning descriptor vocabulary, no-raw-image domain memory, confidence-gated fallback) — not started; see `docs/new_methodology_report.md` for the full design and what each one needs.
- **A complete GenImage download.** Phase F4 only recovered 155/1,000 ImageNet classes and no real labeled training photos (partial multi-part archive — see `results/phase_f4_genimage_alignment.md`). Phase E2 ran the real baseline-vs-+DDO training test on Defactify instead (real photos → Midjourney v6); a full GenImage download would still be useful as a second, independent dataset for the same comparison.
- ~~Descriptor-set staleness test (originally planned item 4, never run)~~ — **done**, as Phase E1 (pool inspection: 0/20 AI-generation terms found in LanCE's shipped descriptor list) and Phase E2 (the real training-accuracy cost: DDO's gain shrinks from +6.40 to +0.68 points on a domain the list doesn't cover). Remaining open extension: test generators beyond Midjourney v6, and re-verify E1 against a fresh GPT-3.5 API call rather than the checked-in pool alone.
- **Failure Mode 4 (static concept bank / fixed output layer)**: deliberately deferred throughout — mixing class-incremental with domain-incremental forgetting would have blurred Phase B–D's results. A natural next experiment once the domain-only case is fully closed out.
- **Phase C's remediations on Office-Home**: already run (see Phase D status above) — but EWC's λ=1000 was never tuned; a swept value could change how it compares to cumulative-DDO/replay.
- **A literature-updated pass**: the literature-gap table in `docs/lance_continual_dg_failure_analysis.md` §3 wasn't re-searched during this round — worth a fresh check before any external write-up.

## Bugs fixed in LanCE's released code

None of these touch the method (DDO loss, model architecture) — all are plumbing bugs that silently blocked the README's own documented commands from running. Full detail in [`results/phase0_cub_reproduction.md`](results/phase0_cub_reproduction.md):

1. Missing `import os` in `data/__init__.py`
2. A stray leading `/` breaking a path join for the CUB-Painting target set
3. `--batch_size`/`--epochs` CLI args typed as `str`/`float` instead of `int`
4. A hardcoded dummy concept-label shape (77) that didn't match the actual concept bank (312)
5. A copy-paste bug passing the wrong dict (`classname2id` twice instead of `concept2id`) when building the target test set
6. The dataset-loader's return tuple silently returned a `DataLoader` in the slot meant for the raw `Dataset` object

## Performance note: cached training

LanCE's released code recomputes CLIP image embeddings from scratch every epoch, even though CLIP is frozen throughout training (embeddings never change). `external/LanCE/cache_utils.py` + `train_cached.py` precompute embeddings once per dataset split and train on the cached features instead — verified to produce numerically identical results to the original pipeline (50.64% baseline target accuracy either way), while cutting epoch time from ~7–14 minutes to **~3 seconds**. Use `train_cached.py` in place of `main.py` for any new experiment; it takes the same CLI arguments.

## Setup

```bash
cd external/LanCE
pip install -r requirements.txt
pip install "git+https://github.com/openai/CLIP.git@main" wandb ftfy gdown pydantic
```

Datasets are gitignored (large, and either publicly redistributable or regenerable — see `external/LanCE/README.md` for download links and `planning/` for exact sizes/sources per dataset).
