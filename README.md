# CBMs for Continual Domain Generalization

Research project investigating whether **concept bottleneck models (CBMs) for domain generalization survive a continual-learning setting** — i.e. domains arriving one at a time, over time, rather than all at once during a single training run.

## Starting point: LanCE

The most relevant prior work is **LanCE** ([Zeng et al., CVPR 2025](https://arxiv.org/abs/2503.18483), [original code](https://github.com/joeyz0z/LanCE)) — a CLIP-based CBM that erases domain-specific concepts from its final classifier using a language-guided "Domain Descriptor Orthogonality" (DDO) loss. It gets strong domain-generalization results, but it trains once, on one domain, and never updates again. This project argues (and is in the process of measuring) that this design cannot survive domains arriving continually — see [`docs/lance_continual_dg_failure_analysis.md`](docs/lance_continual_dg_failure_analysis.md) and the rendered brief in [`presentation/lance_failure_brief.html`](presentation/lance_failure_brief.html) for the full argument.

## Repo layout

| Path | What's there |
|---|---|
| [`docs/`](docs) | The original failure-mode analysis of LanCE (five failure modes, each traced to a specific equation/table in the paper, plus a literature-gap check against adjacent CBM/continual-learning work) |
| [`planning/`](planning) | The two research planning documents developed for this project, kept as a live record including corrections made along the way (datasets considered and rejected, scope refinements, etc.) — not just the final plan |
| [`external/LanCE/`](external/LanCE) | Vendored copy of the LanCE reference implementation, with bug fixes applied (see below) and two new scripts added (`train_cached.py`, `cache_utils.py`) for fast iteration |
| [`results/`](results) | Write-ups and figures for each experiment phase, as they complete |
| [`presentation/`](presentation) | Rendered HTML artifact summarizing the failure-mode analysis for slide-building |

## Current status

**Pillar 1 (primary): does LanCE have any mechanism to survive domains arriving over time?**
- ✅ **Phase 0 — baseline reproduction (CUB → CUB-Painting):** PASS. See [`results/phase0_cub_reproduction.md`](results/phase0_cub_reproduction.md). Our reproduction (baseline 50.64%, +DDO 57.04%) lands within tolerance of the paper's own Table 1 numbers (50.54% → 55.53%), confirming the codebase — after fixing five plumbing bugs in the released code — is trustworthy for everything built on top of it.
- ⏳ **Phase A — closed-world descriptor assumption test:** next up.
- ⏳ **Phase B — Domain-IL sequential protocol (PACS):** the core forgetting experiment.
- ⏳ **Phase C — remediation attempts:** does a textbook continual-learning fix (replay, EWC) already solve it?

**Pillar 2 (secondary): does the frozen CLIP backbone itself have a shelf life, independent of the forgetting problem?**
- ⏳ Not started — EuroSAT-based alignment test, deferred until Pillar 1's core result is in hand.

Full detail on why the plan is shaped this way — including datasets that were investigated and rejected (LADA-Sculpture's hidden Google-Drive/Baidu-Netdisk dependency, AWA2's 13GB download, GenImage's missing class labels) — is in [`planning/02-continual-dg-experiment-plan.md`](planning/02-continual-dg-experiment-plan.md).

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
