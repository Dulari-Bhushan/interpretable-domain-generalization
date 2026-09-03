# Plan 08: does an LLM-generated concept bank beat the human-written one?

**Done and replicated — results in [`results/llm_concept_bank_comparison.md`](../results/llm_concept_bank_comparison.md).** The LLM-generated bank won on every metric (in-domain and CUB-Painting shift, with and without DDO) on two independent draws — draw 1 by 1.6-3.2 points, draw 2 by 2.0-4.7 points.

## 0. Where this comes from, and what it is

The project's own architecture diagram has two branches feeding the concept bank: **"Human or LLM"** → "what are useful visual features for identifying an {class}" → concept bank `C = {c_i}`. Every dataset actually run in this project so far took the Human branch: CUB's 312 concepts are the original CUB-200-2011 paper's real, human-annotated attribute names; PACS/Office-Home/EuroSAT/Defactify's concept banks are "hand-written, first-draft lists (roughly 4 concepts per class)... rather than curated or validated against alternatives" (`docs/research_report.md` §_first-pass concept banks_); DomainNet's is template-generated, not written by anyone at all (`data/DomainNet/generate_domainnet_concepts.py`).

**This is a different axis from Plan 07.** Plan 07 asked "given a fixed, human-written concept bank, which mechanism produces better concept *activations* — CLIP zero-shot similarity or a trained probe?" This plan asks "regardless of activation mechanism, does the concept bank's *content* itself matter — human-written vs. LLM-generated?" — the two questions are independent and this plan changes nothing about how concept activations are produced.

**What's actually different from every prior run in this project:** GPT-3.5 was already used once in this codebase, but only for the *domain descriptor* prompts that feed DDO (`prompts/prompt200new.py`'s `target_text_prompts`, e.g. "a sketch of a {class}") — never for the concept bank itself. This plan is the first time an LLM-generated concept bank is actually built and tested.

## 1. The comparison

| | Concept bank source | Concept activation mechanism | Everything else |
|---|---|---|---|
| **Existing baseline** (Phase 0, `results/phase0_cub_reproduction.md`) | Human-written (CUB's real 312 attributes) | CLIP ViT-L/14 zero-shot similarity | `clip_cbm_orth`, DDO, 50 epochs, batch 64, lr 1e-4 |
| **New (this plan)** | **LLM-generated** (312 phrases, generated fresh from general ornithological knowledge — see §3) | CLIP ViT-L/14 zero-shot similarity (unchanged) | Identical — same architecture, same hyperparameters, same DDO |

**This is the cleanest possible ablation of concept-bank content alone.** Unlike Plan 07 (which swapped the image encoder and therefore *had* to drop DDO, since DDO's concept-activation path no longer existed for DINOv2 — see plan 07 §7's correction), this experiment touches nothing about the image encoder, the text encoder, or DDO's machinery. DDO's regularizer (`self.classifier[1:](self.diffs @ self.concept_embeddings.T)`, `model/cbm_models.py:217`) is built entirely from CLIP-text objects that recompute automatically from whatever `concept_names` the model is given — swap the concept bank file, and both the concept-activation path *and* DDO's regularizer path pick up the new concept embeddings for free, with zero code changes to the model itself. So **both runs — baseline (α=0) and +DDO (α=1) — are done here**, directly comparable number-for-number against Phase 0's existing 50.64%/57.04%.

## 2. The issue this targets

Doesn't map to an existing failure hypothesis. It's a direct answer to a question asked while reviewing this project's own architecture diagram: the diagram depicts concept-bank authorship as a live design choice ("Human or LLM"), and this project has only ever exercised one branch of it. `docs/research_report.md`'s own limitations section already flags the non-CUB concept banks as "first-pass... rather than curated or validated against alternatives" — this plan is the first attempt to actually test whether that matters, on the one dataset (CUB) where a real, human-curated concept bank exists to compare against.

## 3. Method — how the LLM concept bank was built

Generated directly (no external API call needed or used — the acting assistant *is* the LLM referenced in the architecture diagram's "Human or LLM" branch), prompted conceptually the same way LaBo/GPT-CBM-style papers do it: "what visual features would help distinguish this class from the other 199 CUB classes?", pooled across all 200 CUB class names into one shared vocabulary, generated **without reading CUB's real attribute file or the existing `cub_concepts.txt`** — an honest independent generation, not a rephrasing of the ground truth it's being compared against.

- **Exactly 312 phrases** — matched to the existing human-written bank's count deliberately, so nothing about `attr_label`'s dummy-zero tensor width (`Processed_CUB_Dataset.__getitem__`, always `torch.tensor([0]*len(self.concept2id))` — never real supervision, `beta` defaults to 0 everywhere in this project) or any cached tensor shape needs to change; the two concept banks are swappable via one new `--concept_file` CLI flag alone.
- **Style:** short, atomic phrases ("a curved bill", "a rufous coloured crown"), matching the existing bank's phrasing convention so the *only* variable that changes between the two runs is content, not format/tokenization style.
- **Coverage:** bill shape/size/color, per-body-part color and pattern (crown, nape, throat, breast, back, wing, tail, leg, eye), body size, and habitat/behavior cues — the same categories of feature the real CUB attribute ontology covers, since both are trying to solve the same 200-way classification problem.
- File: [`external/LanCE/data/CUB/cub_concepts_llm.txt`](../external/LanCE/data/CUB/cub_concepts_llm.txt). (Caught during the first sanity-check training run: `wc -l` on the human-written `cub_concepts.txt` reports 311 because its last line has no trailing newline — the file actually parses to 312 concepts via Python's `readlines()`, which is what every script in this project actually uses, and what Stage 1's own run logged: `n_concepts=312`. The LLM bank was generated at 311 lines against the wrong `wc -l` count and corrected to exactly 312 with one added phrase, "a tufted crest", before training.)

**A limitation stated plainly, not hidden:** this is one LLM's (Claude's) one-shot generation, not GPT-3.5/4 (which is what the rest of this project's LLM-touched content — the domain descriptors — came from), and not validated against a second LLM or a human review pass. It is exactly as first-draft as the hand-written banks this project already flagged as unvalidated (`docs/research_report.md`) — that symmetry is deliberate: this experiment compares two first-draft concept banks, human vs. LLM, not a curated human bank against a curated LLM bank.

## 4. Code changes (small, backward-compatible)

A concept bank filename was hardcoded (`cub_concepts.txt`) in three places; made parameterizable via a new `--concept_file` CLI arg (default unchanged, so every existing invocation of this codebase behaves identically):
- [`args.py`](../external/LanCE/args.py) — new `--concept_file` argument, default `cub_concepts.txt`.
- [`data/CUB/cub_data.py`](../external/LanCE/data/CUB/cub_data.py) — `Processed_CUB_Dataset.__init__` takes a `concept_file` parameter (falls back to `args.concept_file`).
- [`data/__init__.py`](../external/LanCE/data/__init__.py) — CUB branch passes `concept_file=args.concept_file` through to both the train and (source) test dataset constructors, so concept ordering/indexing stays consistent between them. (`Processed_CUBP_Dataset`, the CUB-Painting target set, needs no change — it already receives `concept2id` directly from `train_dataset`, never reads the file itself.)

Nothing in `model/cbm_models.py`, `train_cached.py`, or `cache_utils.py` needed to change — concept text embeddings and DDO's regularizer both recompute fresh from `concept_names` on every run already (never cached), and the cached CLIP *image* embeddings (`embeddings_cache/CUB_ViT-L-14_*.pt`) are reused as-is since they don't depend on the concept bank at all.

## 5. Protocol

Reuses `train_cached.py` exactly as Phase 0 used it (`results/phase0_cub_reproduction.md`) — same architecture (`clip_cbm_orth`), same hyperparameters (50 epochs, batch size 64, Adam, lr 1e-4), same two runs:

```
python train_cached.py --dataset CUB --alpha 0 --epochs 50 --batch_size 64 --class_avg_concept --CBM_type clip_cbm --concept_file cub_concepts_llm.txt
python train_cached.py --dataset CUB --alpha 1 --epochs 50 --batch_size 64 --class_avg_concept --CBM_type clip_cbm --concept_file cub_concepts_llm.txt
```

Report: CUB in-domain (source test) and CUB-Painting (target test) accuracy, both α=0 and α=1, directly next to Phase 0's 50.64%/57.04% — same convention as every other accuracy number in this project.

## 6. Dataset used, and why

CUB → CUB-Painting only, for the same reason Plan 07 is CUB-only: it's the one dataset in this project with an existing human-written concept bank and an established baseline number to compare against. A strong or weak result here says nothing yet about whether LLM-generated concept banks would help the *hand-written* banks used for PACS/Office-Home/EuroSAT/Defactify — extending this to those datasets is a real, cheap follow-up once this result is in (no real-attribute-label constraint blocks it there, unlike Plan 07's DINOv2 axis, since this experiment only needs concept *names*, not per-image concept labels).

## 7. Risks and open questions

- **One-shot, one-LLM generation** (§3's limitation, restated) — a different LLM, a different prompt, or multiple sampled-and-merged generations could produce a meaningfully different (better or worse) bank; this result characterizes *one* draw, not "LLM-generated concept banks" as a category.
- **No independent concept-level check.** Unlike Plan 07 (which could score concept activations against CUB's real per-image attribute labels directly), the LLM concept bank's phrases don't line up 1:1 with the real attribute ontology, so there's no equivalent per-concept AUROC check here — only the downstream classification number is measurable.
- **Same DDO-decoupling insight as Plan 07 §7 applies in reverse here, worth stating explicitly:** since nothing about the image encoder changes in this experiment, DDO's own domain-shift assumption (that CLIP-text-described "photo→painting" reasoning is a useful prior) is untouched by which concept bank is in use — so if DDO's gain (α=1 vs α=0) looks different here than Phase 0's own +6.40 points, that difference is attributable to the concept bank's content changing what DDO has to work with, not to any change in DDO's own mechanism.

## 8. Code

- Concept bank: [`external/LanCE/data/CUB/cub_concepts_llm.txt`](../external/LanCE/data/CUB/cub_concepts_llm.txt)
- Modified: [`external/LanCE/args.py`](../external/LanCE/args.py), [`external/LanCE/data/CUB/cub_data.py`](../external/LanCE/data/CUB/cub_data.py), [`external/LanCE/data/__init__.py`](../external/LanCE/data/__init__.py)
- Training: [`external/LanCE/train_cached.py`](../external/LanCE/train_cached.py) (unmodified, reused as-is per §5)
