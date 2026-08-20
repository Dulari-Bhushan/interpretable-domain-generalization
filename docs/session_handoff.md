# Session handoff — read this first in a new chat

**Why this file exists:** the chat session that did Component 1's work ran low on context. This is a self-contained briefing so a fresh chat (working on Component 2 or anything else) doesn't need that conversation's history to know exactly where things stand and how to continue.

---

## 1. The remote server — how to use it

A GPU server is already set up and reachable. Full detail belongs here rather than being re-discovered:

- **Connect:** `ssh lab-server` (an SSH config alias already exists on this machine, key-based auth, no password needed).
- **Server specs:** 3× NVIDIA RTX A5000 (24GB each), shared with other users.
- **Hard rule: use at most 2 of the 3 GPUs at a time, and check `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` before starting anything** — other people use this machine too.
- **Python environment:** conda env `mlgpu` at `/data/ai25mtech14009/miniconda3` — activate with `source /data/ai25mtech14009/miniconda3/bin/activate mlgpu`. Has torch (cu121), CLIP, and the project's requirements installed (note: `mmcv` from `requirements.txt` fails to build there and is skipped deliberately — nothing in this codebase actually imports it, confirmed by grep).
- **Repo location on server:** `/data/ai25mtech14009/repo` — a normal git clone of this GitHub repo. **Sync by `git pull origin main`** on the server after pushing from wherever you're working, not by copying files by hand.
- **Disk quota:** 200GB, well within budget for everything downloaded so far.
- **Datasets are gitignored, same as everywhere else in this project** — code and manifests sync via git, but raw images have to be fetched separately on the server (either downloaded directly there, or transferred). PACS, Office-Home, EuroSAT, and the CLIP embeddings cache are already on the server (transferred from this machine). DomainNet is being downloaded directly on the server (see §3).

**Important lesson learned the hard way, worth not re-learning:** any command run directly over a live SSH pipe that takes longer than a minute or so is at real risk of dying from "Connection reset by peer" — this happened mid-run. **Anything long-running must go inside a detached `tmux` session on the server** (`tmux new-session -d -s <name> '<command> > logfile 2>&1'`), so it survives a dropped connection. Check on it later with fresh, short SSH calls that just tail the log file or run `tmux list-sessions` — don't keep one connection open for the whole duration.

Also worth knowing: when running a background command through this session's own tool, **never add a manual trailing `&`** on top of the tool's own backgrounding — doing both together causes the tool to report "done" instantly while silently orphaning and killing the real process. This bug bit the dataset transfer once already (see git history / Component 1's own troubleshooting).

---

## 2. Component 1 — exact status

**Fully done, including the scale test. Nothing further needed unless the L1-vs-L2-on-DomainNet follow-up below gets picked up.**
- [`results/component1_exact_classifier.md`](../results/component1_exact_classifier.md) — the full report. PACS, Office-Home, **and now DomainNet (6 domains, 345 classes, ~586K images)** all show max-difference-from-joint = 0.0000 on every domain ordering tested — the exact-match property doesn't degrade at scale.
- [`results/component1b_l1_vs_l2_ablation.md`](../results/component1b_l1_vs_l2_ablation.md) — the L1-vs-L2 DDO ablation, also done (PACS/Office-Home only). Real, mixed result: free on PACS, costs 1.75 accuracy points on Office-Home for a much stronger orthogonality property.
- The DomainNet loader (`external/LanCE/data/DomainNet/`) is built, tested, and pushed; its concept bank is template-generated rather than hand-written — deliberate, documented in `generate_domainnet_concepts.py`'s own docstring, fine for the exactness test but would need replacing with real concepts if DomainNet classification *accuracy* ever needs to mean something on its own.
- Applying the report template's 90% bar honestly (see `results/component1_exact_classifier.md` §12): the previously-listed "longer synthetic sequences" and "numerical conditioning at scale" follow-ups were dropped as low-value now that exactness is confirmed at three real scales with zero deviation. The one item that does clear the bar and is genuinely still open: **running the L1-vs-L2 ablation on DomainNet too** — 1b's own finding was dataset-dependent, not universal, so whether that pattern holds, reverses, or changes at DomainNet's very different scale is a real open question, not yet run.

**This chat's job is done unless something about Component 1 needs revisiting.** Don't duplicate this work in a new chat — if the L1-vs-L2-on-DomainNet follow-up gets picked up, it can go in whichever chat has room, following `external/LanCE/experiments/component1_l1_vs_l2_ablation.py` as the pattern (same script, swap in `get_domainnet_datasets`/`DOMAINNET_DOMAINS`).

---

## 3. What's already true about the rest of the project (context, not new work)

- **The plan and its status:** [`docs/new_methodology_report.md`](new_methodology_report.md) — 5 components + 1 stretch idea, literature-checked against what's already published (§6). Component 1 is the only one done. Components 2-5 not started.
- **The reporting convention:** [`docs/component_report_template.md`](component_report_template.md) — every component gets a report in this exact structure once it's actually been run, dead ends included, following it word for word. `results/component1_exact_classifier.md` is the worked example.
- **A parallel thread, separate from the 5 components:** [`planning/03-detector-grounded-concept-extraction-plan.md`](../planning/03-detector-grounded-concept-extraction-plan.md) — testing whether concept scores should come from CLIP similarity (current), a directly-trained classifier, or a pretrained open-vocabulary detector (Grounding DINO/OWL-ViT). Not started. Its own staged plan: CUB sanity check first, then PACS/Office-Home, then the domains CLIP struggles on.
- **Datasets on hand:** PACS, Office-Home, EuroSAT, Defactify, GenImage (partial, 155 classes), AWA2, DomainNet (loader ready, images downloading). **Still blocked:** Camelyon17 (medical - host outage, retry `wilds.download_datasets` periodically or download manually from wilds.stanford.edu/downloads), LADA-Sculpture/CheXpert/MIMIC-CXR (need the project owner's own credentialed registration, can't be automated).

---

## 4. Component 2 — status (started 2026-08-19, this session)

Self-diagnosing domain grounding — checking whether DDO's text-only domain-shift guess is trustworthy, falling back to a real-image-measured `domain_diffs` when it isn't. Plan: [`planning/04-component2-self-diagnosing-domain-grounding-plan.md`](../planning/04-component2-self-diagnosing-domain-grounding-plan.md).

**Built:**
- [`external/LanCE/model/domain_grounding.py`](../external/LanCE/model/domain_grounding.py) — the diagnostic (Phase F3's real-matched-photo alignment-score formula, generalized to a small probe) + the image-grounded `domain_diffs` fallback (drop-in replacement, `clip_cbm_orth` needs no changes).
- [`external/LanCE/experiments/component2_alignment_calibration.py`](../external/LanCE/experiments/component2_alignment_calibration.py) — threshold calibration on PACS.
- [`external/LanCE/experiments/component2_defactify_grounding_ddo.py`](../external/LanCE/experiments/component2_defactify_grounding_ddo.py) — main validation: reruns Phase E2's baseline-vs-+DDO protocol on Defactify (photo→Midjourney v6) with a third condition, +DDO using the self-diagnosed/grounded `domain_diffs`.

**Calibration run: done.** PACS (photo→art_painting/cartoon/sketch, real images both sides, this project's own diagnostic code) gives mean alignment 0.20–0.33 per domain (0.2493 overall) — **not** the paper's claimed 0.90–0.99, an honest finding worth keeping in the write-up (the diagnostic still separates known-good from known-bad domains by ~6-9x; it just doesn't hit the paper's absolute range under this formula). Defactify (photo→Midjourney v6, cited from Phase F3): 0.037. Calibrated threshold = midpoint = **0.1431**. Result: `results/component2_alignment_calibration.json`.

**Main experiment (first run): done** (2026-08-19 ~19:58-20:15 IST, GPU 1, tmux `c2_defactify`). Diagnostic worked; the single-mean-direction fallback made things worse than baseline. Written up at [`results/component2_self_diagnosing_grounding.md`](component2_self_diagnosing_grounding.md).

**Follow-up run: done** (2026-08-19 ~22:00-22:30 IST, GPU 1, tmux sessions `c2_eurosat` then `c2_variants`, both exited cleanly). Isolated the specific cause of the first run's harm via a clean, controlled ablation (same 20 probe images, same seed, same target_test, only the packaging differs): collapsing to 1 mean direction cost −0.18 points; keeping all 20 as separate directions gained +0.87 — a real, defensible 1.05-point swing. **Caught and corrected an overclaim on first write-up**: I initially marked this "confirmed/validated," but this project's own three measurements of `ddo_text`'s gain on this exact domain (Phase E2 +0.68, first C2 run +0.76, this run +1.02) already show ~0.3 points of pure run-to-run noise from test-split composition alone — and the gap between the best grounded variant and text-DDO here (+0.95 vs +1.02) is smaller than that. So the *mechanism* (diversity matters) is solid; the *magnitude* claim ("recovers to near-parity with text-DDO") is not yet earned without a seed sweep, which hasn't been run. EuroSAT calibration (3rd point) confirmed Phase F1's number almost exactly (0.322 vs. 0.324) — that one's a clean diagnostic measurement, not subject to the same noise concern. Full write-up (now correctly hedged): [`results/component2b_grounding_fallback_variants.md`](component2b_grounding_fallback_variants.md).

**Seed sweep + second domain: done** (2026-08-19 ~23:07-23:38 IST, GPU 1, tmux `c2_seedsweep`, ran seed=1 and seed=2 reruns of the variants script back-to-back with the DALL-E 3 validation, exited cleanly). Gated on a literature check first (per instruction) — converged on 3 independent lines of work (CLIP's own prompt-ensembling result; Tip-Adapter/CLIP-Adapter/Proto-Adapter's blend-few-shot-with-text approach, with blend ratio scaled by domain gap; Task Arithmetic's "adding vectors works" finding) that all point the same direction as `component2b`'s findings, which is what justified proceeding to more runs. **Result: this substantially strengthens and corrects the earlier picture.** The true same-split training-seed noise floor for `ddo_text` is only ~0.10 points (not the ~0.3 estimated from mixing different train/test splits in component2b) - and against that tighter floor: **blend is now confirmed** (mean gain 1.17 across 4 measurements, essentially identical to text-DDO's own 1.17, tight std 0.14); **persample shows a real, promising edge** (mean gain 1.74, beat text-DDO in 3 of 4 measurements) not yet nailed down to high confidence given its own variance; **the single-mean-direction design's flaw is now correctly understood as instability** (std 0.90, ~8-10x text-DDO's own noise), not simply "worse on average" as first thought. DALL-E 3 (Phase F3's most distinct generator from Midjourney) confirms the same pattern on a second domain, and its diagnostic (0.0216) closely matches Phase F3's own independently-measured number (0.017-0.023). Full write-up: [`results/component2c_seed_sweep_and_second_domain.md`](component2c_seed_sweep_and_second_domain.md).

Component 2 tracker status in `docs/new_methodology_report.md` now reflects this: "diagnostic + blend fallback validated across 3 seeds x 2 domains; persample promising, not yet fully confirmed."

Other GPUs at the time this was worked: GPU 0 was Component 1's DomainNet work (a different chat session). GPU 1 is this component's; GPU 2 was left free throughout (kept total project usage at 2 of 3 GPUs, per the shared-server rule).

**Still open:** more seeds specifically on persample (its own variance, not whether grounding works at all, is the remaining question); a weighted-blend variant (Proto-Adapter-motivated: scale the grounded direction's weight by how far below threshold the alignment falls, rather than a flat 1-in-205). See `results/component2c_seed_sweep_and_second_domain.md` "What's next". **The combined/ablation-testing commitment (§5 below) can now reasonably include Component 2** (diagnostic + blend) - this is a real change from the prior "not yet" status.

---

## 5. An open commitment, raised but not yet acted on — don't lose this

The project owner asked directly: since each component fixes a *different* problem, when do we test them **combined**, not just individually? E.g. Component 1 + Component 3 together, and eventually all validated components running as one system, compared against the original baseline and against each component alone — a real ablation table, not just isolated single-component results. **Agreed this is necessary and not yet built into the plan.** There's nothing to combine yet with only Component 1 done, but **as soon as a second component is validated, this combined/ablation testing stage needs to become a real, written part of the plan** (a new section in `docs/new_methodology_report.md`, most likely) — not an afterthought at the very end. Flagging this here so it survives the context handoff.
