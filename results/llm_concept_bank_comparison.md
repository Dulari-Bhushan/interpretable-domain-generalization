# LLM-generated vs. human-written CUB concept bank

**Status: ✅ Done — result confirmed.**

## 1. One-line summary

Swapping CUB's real, human-written 312-concept bank for a 312-phrase bank generated independently by an LLM (with the image encoder, DDO, and every hyperparameter held identical) improves classification accuracy on every metric measured — in-domain and under the CUB→CUB-Painting domain shift, with and without DDO — by 1.6 to 3.2 points.

## 2. Origin

[`planning/08-llm-generated-concept-bank-plan.md`](../planning/08-llm-generated-concept-bank-plan.md), the plan's only stage.

## 3. The issue this targets

Doesn't map to an existing failure hypothesis. It answers a question raised while reviewing this project's own architecture diagram: the diagram depicts concept-bank authorship as a live design choice ("Human or LLM" → concept bank), and every dataset actually run in this project so far — CUB included — took the Human branch. `docs/research_report.md` already flagged the non-CUB concept banks as unvalidated first drafts; this is the first time the Human-vs-LLM axis was actually tested, on the one dataset with a real human-curated bank to compare against.

## 4. Why we tried this approach specifically

CUB is the only dataset in this project with a genuine, human-curated concept bank (real per-image attribute labels underpin it) rather than a first-draft hand-written list — so it's the one place a fair "curated human list vs. LLM list" comparison is even possible. Critically, this experiment doesn't touch the image encoder, so — unlike Plan 07's DINOv2 swap, which had to drop DDO because DDO's *concept-activation* path (not its regularizer) is CLIP-specific — DDO applies here completely unmodified. That makes this the cleanest possible test of concept-bank content alone: everything else in `clip_cbm_orth` (image encoder, text encoder, DDO's domain-shift machinery, training hyperparameters) is held bit-for-bit identical to Phase 0's own reproduction run.

## 5. Method

**Concept bank generation:** 312 phrases, generated directly by the acting assistant (Claude) — the LLM referenced in the architecture diagram's own "Human or LLM" branch — prompted conceptually the same way LaBo/GPT-CBM-style papers do it ("what visual features would help distinguish this class from the other 199 CUB classes?"), pooled across all 200 CUB class names into one shared vocabulary. Generated **without reading** CUB's real attribute file or the existing `cub_concepts.txt`, so it's an honest independent draft, not a rephrasing of the ground truth it's being compared against. Style matched to the existing bank (short atomic phrases, e.g. "a curved bill") so format/tokenization isn't a confound. File: [`external/LanCE/data/CUB/cub_concepts_llm.txt`](../external/LanCE/data/CUB/cub_concepts_llm.txt).

A count bug was caught and fixed before training: `wc -l` reports 311 for the human-written bank because its last line has no trailing newline, but every script in this project actually loads it with Python's `readlines()`, which correctly returns 312 (confirmed by Stage 1's own logged `n_concepts=312`, `results/concept_source_cub_stage1.json`). The LLM bank was generated at 311 lines against the wrong count, caught by a 1-epoch sanity-check run (`BCEWithLogitsLoss` shape mismatch), and fixed to exactly 312 with one added phrase ("a tufted crest").

**Code change:** the concept bank filename was hardcoded in three places; made parameterizable via a new `--concept_file` CLI flag (default unchanged, so every existing invocation of this codebase behaves identically) — see [`args.py`](../external/LanCE/args.py), [`data/CUB/cub_data.py`](../external/LanCE/data/CUB/cub_data.py), [`data/__init__.py`](../external/LanCE/data/__init__.py). No changes anywhere else — concept text embeddings and DDO's regularizer both already recompute fresh from `concept_names` on every run, and the cached CLIP *image* embeddings are reused as-is since they don't depend on the concept bank.

**Training:** `train_cached.py`, unmodified — the exact script and hyperparameters Phase 0 used (`clip_cbm_orth`, CLIP ViT-L/14, 50 epochs, batch size 64, AdamW, lr 1e-4, weight decay 1e-4), two runs (α=0 baseline, α=1 +DDO), the only difference being `--concept_file cub_concepts_llm.txt`.

## 6. Dataset used, and why

CUB → CUB-Painting, matching Phase 0's own baseline exactly, for direct numerical comparability — the whole point of this experiment is a same-pipeline, same-hyperparameter, concept-bank-only ablation against an established number.

## 7. Code

- Concept bank: [`external/LanCE/data/CUB/cub_concepts_llm.txt`](../external/LanCE/data/CUB/cub_concepts_llm.txt)
- Modified: [`external/LanCE/args.py`](../external/LanCE/args.py), [`external/LanCE/data/CUB/cub_data.py`](../external/LanCE/data/CUB/cub_data.py), [`external/LanCE/data/__init__.py`](../external/LanCE/data/__init__.py)
- Training (unmodified, reused as-is): [`external/LanCE/train_cached.py`](../external/LanCE/train_cached.py)
- Figure generator: [`results/generate_llm_concept_bank_figures.py`](generate_llm_concept_bank_figures.py)
- Second concept bank (replication): [`external/LanCE/data/CUB/cub_concepts_llm2.txt`](../external/LanCE/data/CUB/cub_concepts_llm2.txt)
- Logs: [`external/LanCE/logs_plan08_llm_alpha0_cached.log`](../external/LanCE/logs_plan08_llm_alpha0_cached.log), [`external/LanCE/logs_plan08_llm_alpha1_cached.log`](../external/LanCE/logs_plan08_llm_alpha1_cached.log) (LLM draw 1); [`external/LanCE/logs_plan08_llm2_alpha0_cached.log`](../external/LanCE/logs_plan08_llm2_alpha0_cached.log), [`external/LanCE/logs_plan08_llm2_alpha1_cached.log`](../external/LanCE/logs_plan08_llm2_alpha1_cached.log) (LLM draw 2, replication); [`external/LanCE/logs_baseline_cached.log`](../external/LanCE/logs_baseline_cached.log), [`external/LanCE/logs_ddo_cached.log`](../external/LanCE/logs_ddo_cached.log) (human bank, Phase 0's own runs, reused for comparison)

## 8. Results

Best-epoch numbers (selected by target/CUB-Painting accuracy, same convention as every other accuracy result in this project — the paired source/in-domain number is whatever it was on that same epoch, not independently maximized):

| Concept bank | DDO | In-domain (CUB test) | CUB-Painting (shift) |
|---|---|---|---|
| Human-written (Phase 0) | α=0 | 77.81% | 50.64% |
| **LLM-generated** | α=0 | **79.43%** (+1.62) | **53.82%** (+3.18) |
| Human-written (Phase 0) | α=1 | 79.52% | 57.04% |
| **LLM-generated** | α=1 | **81.24%** (+1.72) | **59.07%** (+2.03) |

![Human vs. LLM concept bank, baseline and +DDO](figures/llm_concept_bank_comparison.png)

**DDO's own gain, within each bank:** human bank +6.40 points (50.64%→57.04%, matches Phase 0's report exactly); LLM bank +5.25 points (53.82%→59.07%). DDO helps both banks by a similar, large margin; it's slightly smaller in absolute points on the LLM bank, but starts from and ends at a higher number.

### Replication — a second, independent LLM-generated bank

The one open item from §11 below: does the win replicate on a second, independent LLM-generated draw? A second bank (`cub_concepts_llm2.txt`, generated fresh, without reading the real attributes, the first LLM draft, or looking at this section) was built and run through the identical protocol.

| Concept bank | DDO | In-domain | CUB-Painting (shift) |
|---|---|---|---|
| Human-written (Phase 0) | α=0 | 77.81% | 50.64% |
| LLM draw 1 | α=0 | 79.43% (+1.62) | 53.82% (+3.18) |
| **LLM draw 2** | α=0 | **79.84%** (+2.03) | **55.33%** (+4.69) |
| Human-written (Phase 0) | α=1 | 79.52% | 57.04% |
| LLM draw 1 | α=1 | 81.24% (+1.72) | 59.07% (+2.03) |
| **LLM draw 2** | α=1 | **81.93%** (+2.41) | **61.40%** (+4.36) |

Both independent LLM draws beat the human-written bank on every metric — the effect replicates, and draw 2 is if anything slightly stronger than draw 1 (both draws land within 1.5-2.5 points of each other, well inside a coherent, repeatable win rather than one draw being a fluke and the other a wash). This meaningfully upgrades the confidence level from §10/§11's original "one draw" caveat: two independent generations, same direction, comparable magnitude, is real evidence for a genuine effect of LLM-generated content here — not proof the effect holds for every possible LLM/prompt/dataset combination, but no longer a single, unreplicated data point either.

## 9. What this means

The LLM-generated concept bank wins on **every single metric measured** — not a mixed result. The domain-shift gain (+3.18 without DDO, +2.03 with DDO) is larger than the in-domain gain (+1.62 / +1.72) in the no-DDO case, echoing (at smaller magnitude) the pattern Plan 07 found for DINOv2's concept source: changing what produces the concept representation tends to help more under distribution shift than in-domain. One plausible reading: the LLM-generated phrases may lean more generic/visually salient ("a black mask", "a bright plumage") than CUB's real attribute ontology, which was built by human annotators for maximum fine-grained discriminability within CUB specifically (e.g. very specific bill-length comparisons) — a bank that's slightly less overfit to CUB's exact photo-domain quirks could transfer a bit better to CUB-Painting's different visual style. This is a plausible mechanism, not a confirmed one — nothing here directly measures concept-level generality the way Plan 07's Stage 1 measured concept-level accuracy.

**A genuinely different kind of result from Plan 07's:** Plan 07 found a large, unambiguous backbone effect (DINOv2 vs. CLIP zero-shot, 6.5+ points). Here the effect is real but smaller (1.6-3.2 points) — worth taking seriously, but also the kind of margin where a single-run, single-generation result deserves real caution before calling it a settled finding (§10-11).

## 10. Verdict

**Answers the question this plan asked, on CUB, and now replicated across two independent draws.** Both an initial LLM-generated concept bank and a second, independently-generated one beat the same human-written bank on every metric (in-domain, domain shift, with and without DDO) — a coherent, repeatable effect, not a single lucky draw. It does **not** establish that "LLM-generated concept banks are generally better than human-written ones" as a universal category claim — two draws from one LLM (Claude), one dataset, one pipeline is real evidence, not proof across LLMs/prompts/datasets. Literature check: LaBo and similar GPT-generated-concept-bank CBM papers already establish that LLM-generated concepts are a viable, competitive concept source in general; this project's specific contribution is a controlled, same-pipeline, same-hyperparameter, *replicated* A/B against a real, human-curated (not just a plausible-sounding hand-written) concept bank, with and without a domain-shift regularizer — a comparison the literature checked for `docs/new_methodology_report.md` §6 didn't turn up in this exact form.

## 11. What's next

**The single-draw caveat is resolved — replication is done.** What's left, honestly assessed against the 90% bar:

- **Extending to PACS/Office-Home/EuroSAT/Defactify's hand-written banks** now clears the bar that the original write-up said it didn't yet — the CUB result no longer rests on one draw. This is a real, cheap follow-up (`--concept_file` already supports it; each dataset just needs its own LLM-generated bank, same generation process as here) and would test whether the effect generalizes beyond CUB, where every other domain-shift claim in this project actually lives.
- **A third LLM or a different prompting strategy** (e.g. explicitly asking for class-discriminative rather than generic descriptive phrases) is a reasonable future check but doesn't clear the 90% bar right now — two same-LLM draws already answered "is this reproducible for this generation process," and the marginal information from a third same-style draw is low; a genuinely different LLM/prompting axis is a different, lower-priority question than finishing the cross-dataset extension above.
