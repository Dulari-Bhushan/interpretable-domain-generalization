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

**Core validation: done, confirmed, nothing further needed there.**
- [`results/component1_exact_classifier.md`](../results/component1_exact_classifier.md) — the full report. PACS and Office-Home both show max-difference-from-joint = 0.0000 on every domain ordering tested.
- [`results/component1b_l1_vs_l2_ablation.md`](../results/component1b_l1_vs_l2_ablation.md) — the L1-vs-L2 DDO ablation, also done. Real, mixed result: free on PACS, costs 1.75 accuracy points on Office-Home for a much stronger orthogonality property.

**In progress: the DomainNet scale test.** This is the one piece of Component 1 not yet finished.
- The loader (`external/LanCE/data/DomainNet/domainnet_data.py`, `prepare_domainnet_dataset.py`, `generate_domainnet_concepts.py`, plus the wiring in `data/__init__.py`) is built, tested, and already pushed.
- The validation script (`external/LanCE/experiments/component1_analytic_domain_il_domainnet.py`) is built and pushed, mirroring the PACS/Office-Home harnesses exactly.
- **What's blocking it right now:** DomainNet's ~18GB of raw images are being downloaded directly on the server (not transferred from this machine — faster and more reliable). Running inside a detached tmux session called `domainnet_dl`, logging to `/data/ai25mtech14009/domainnet_dl.log`. As of this handoff (19 Aug 2026, ~19:37 IST), it was partway through (clipart done, infograph in progress) at roughly 1-4MB/s depending on the moment — could reasonably take a few more hours from when you're reading this.

**To pick this up in the new chat:**
1. Check download status: `ssh lab-server "tail -50 /data/ai25mtech14009/domainnet_dl.log"` — look for `DOMAINNET_DOWNLOAD_DONE` at the end.
2. Once done, run the actual experiment **inside another tmux session** (per §1's lesson):
   ```bash
   ssh lab-server "cd /data/ai25mtech14009/repo/external/LanCE && tmux new-session -d -s domainnet_run 'source /data/ai25mtech14009/miniconda3/bin/activate mlgpu && CUDA_VISIBLE_DEVICES=0 python -m experiments.component1_analytic_domain_il_domainnet > /data/ai25mtech14009/domainnet_run.log 2>&1'"
   ```
   This will first build a CLIP embedding cache for ~586K images (the slow part, one-time), then run the joint fit and 3 sequential orderings.
3. Once it writes `results/component1_domainnet_results.json` on the server, copy it back and write up the result following [`docs/component_report_template.md`](component_report_template.md) — likely updating `results/component1_exact_classifier.md` §12 (mark the DomainNet item done, same pattern as the L1-vs-L2 ablation) rather than a whole new report, since it's extending Component 1's own existing report, not answering a separate question.
4. **A design decision already made and worth knowing:** the DomainNet concept bank (`domainnet_concepts.txt`, 1380 entries) is template-generated, not hand-written like PACS/Office-Home's — deliberately, because Component 1's exactness claim is a mathematical property that doesn't depend on concept quality. This is fine for the scale test; it would need replacing with real concepts first if a future experiment needed DomainNet *classification accuracy* to be meaningful, not just the exactness property.

**This chat is staying on Component 1** in case the DomainNet run needs debugging or follow-up questions come up — don't duplicate that work in a new chat.

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

**Follow-up run: done** (2026-08-19 ~22:00-22:30 IST, GPU 1, tmux sessions `c2_eurosat` then `c2_variants`, both exited cleanly). Resolved the dead end: it was specifically the single-direction collapse, not grounding-in-images itself. A diversity-preserving fallback (20 per-sample directions instead of 1 mean) reached +0.87 gain, and blending the grounded direction into the full 204-direction text pool reached +0.95 — both within ~0.1 point of text-only DDO's own +1.02, up from the single-direction design's −0.18 to −1.20 across probe sizes 10-60. EuroSAT calibration (3rd point) confirmed Phase F1's number almost exactly (0.322 vs. 0.324). Full write-up: [`results/component2b_grounding_fallback_variants.md`](component2b_grounding_fallback_variants.md). **Honest ceiling stated there**: the best grounded variants match, not clearly beat, text-only DDO on this one domain — DDO's own benefit here is itself small (+0.68 to +1.02 across three separate runs), so "recovers to parity" is the realistic result, not "clearly better."

Component 2 tracker status updated in `docs/new_methodology_report.md`'s component table.

Other GPUs at the time this was worked: GPU 0 was Component 1's DomainNet work (a different chat session; that tmux session had exited by 22:00, status unknown to this chat - check with the other session or `results/component1_domainnet_results.json` for whether it finished). GPU 1 is this component's; GPU 2 was left free throughout (kept total project usage at 2 of 3 GPUs, per the shared-server rule).

**Still open, not started:** a second validation domain for Component 2 (everything so far is one domain shift, photo→Midjourney v6, single run, no seed sweep) — see `results/component2b_grounding_fallback_variants.md` "What's next" #2.

---

## 5. An open commitment, raised but not yet acted on — don't lose this

The project owner asked directly: since each component fixes a *different* problem, when do we test them **combined**, not just individually? E.g. Component 1 + Component 3 together, and eventually all validated components running as one system, compared against the original baseline and against each component alone — a real ablation table, not just isolated single-component results. **Agreed this is necessary and not yet built into the plan.** There's nothing to combine yet with only Component 1 done, but **as soon as a second component is validated, this combined/ablation testing stage needs to become a real, written part of the plan** (a new section in `docs/new_methodology_report.md`, most likely) — not an afterthought at the very end. Flagging this here so it survives the context handoff.
