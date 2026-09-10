# Plan 09: how well can a VLM generate concepts, given real images?

**Status: not started.**

## 0. Where this comes from, and what it is

New problem statement suggested by the advisor: instead of (or alongside) improving the CBM pipeline itself, directly study **how good different VLMs are at the concept-generation step** — treat concept-bank quality as the research question, not a fixed input. LanCE's own pipeline (human-written or LLM-written concept bank → CLIP zero-shot activation → DDO training) becomes the **baseline** to compare a VLM-driven concept-generation pipeline against.

**This is a different axis from Plan 08**, which already tested "LLM-generated vs. human-written concept bank" — but Plan 08's generation was **text-only**: one LLM (Claude), prompted with nothing but the 200 CUB class names, using general world knowledge, never looking at an actual image. This plan's distinguishing feature is **image-grounding**: show the VLM real training images and ask what visual concepts *it actually sees*, then compare that against Plan 08's text-only result to isolate what vision contributes. It also broadens from one LLM to multiple SOTA VLMs, compared against each other.

## 1. The comparison

| | Concept source | Sees real images? | Everything else |
|---|---|---|---|
| **Baseline A** (Phase 0) | Human-written (CUB's real 312 attributes) | N/A (ground truth) | `clip_cbm_orth`, DDO, existing numbers |
| **Baseline B** (Plan 08) | Claude, text-only, class names only | No | Existing numbers, directly on record |
| **New — VLM, image-grounded** | Claude / Gemini / open-source VLM, each shown real per-class images | Yes | Same downstream pipeline, when run |

Three VLMs, each run the same way, so the comparison is apples-to-apples:
1. **Claude** (this session, multimodal) — directly comparable to Baseline B since it's the same model family, isolating what showing images adds over Plan 08's text-only prompt.
2. **Gemini** (2.5 Pro or Flash, via API — user has a key) — a second, independent SOTA VLM.
3. **An open-source VLM**, run locally on the lab GPU server (`docs/session_handoff.md` §1) — default **Qwen2-VL-7B-Instruct** (fits one A5000, strong open-weight vision-language performance); swap for a different one if preferred.

## 2. Method — how each VLM's concept bank is built

For each of CUB's 200 classes:
- Sample a small fixed number of real training images (e.g. 5) for that class.
- Prompt the VLM with the images + class name: "what visual features distinguish this bird from other species — describe only what you can see in these images."
- Pool per-class outputs into one shared vocabulary (same convention as Plan 08 §3), deduplicating near-identical phrases.

Each VLM's run must use the **same images per class** across all three VLMs, so any quality difference is attributable to the model, not to which images happened to be sampled.

Deliberately **not** matched to exactly 312 phrases this time (unlike Plan 08, which matched the human bank's count to keep tensor shapes identical) — the point here is partly to see how many *and which* concepts each VLM naturally proposes; count itself is a data point, not a constraint. If a fixed count turns out to be needed for downstream training compatibility, revisit then.

## 3. Evaluation

**Primary: concept-level quality against CUB's real 312 human attributes.**
- Embed both the generated concept phrases and the real attribute names with CLIP's text encoder (same encoder already used throughout this project, so nothing new needs validating).
- **Coverage**: % of real attributes that have some generated concept above a similarity threshold.
- **Precision**: % of generated concepts that match some real attribute above threshold.
- Report both, per VLM, plus for Baseline B (Plan 08's existing text-only bank) as a retroactive comparison point — this metric was never computed for Plan 08 at the time, so it needs to be run once for that bank too, to make the vision-vs-no-vision comparison real. Threshold value needs picking (start with something informed by the CLIP text encoder's normal similarity range, then sanity-check by hand on a sample of the matches before trusting the metric — don't just accept whatever an unvalidated threshold produces).

**Secondary (stretch, not blocking): downstream accuracy.** If concept-level results look interesting, reuse Plan 08's exact training protocol (`train_cached.py`, same hyperparameters, `--concept_file` flag) to get CUB / CUB-Painting accuracy for each VLM's bank, directly next to Phase 0 (50.64/57.04) and Plan 08's existing numbers. Only worth the compute if the concept-level metric shows real separation between VLMs.

## 4. Secondary pipeline: DINOv2 → CLIP concept discovery (non-VLM baseline)

A genuinely different mechanism, worth one comparison point in the same write-up: no language model involved at all.
- Extract DINOv2 image embeddings per class (already used elsewhere in this project — see Plan 07).
- Cluster embeddings within a class (or across classes, TBD) to find recurring visual sub-structure.
- Label each cluster using CLIP's open-vocabulary matching against a large candidate phrase set (or against the same generated-concept vocabularies from §2, to see whether the clusters land on the same concepts a VLM would name).

Scoped as secondary because it's a different kind of system (embedding clustering + open-vocab labeling, not a VLM being prompted) — useful as a contrast case for "does concept quality even need language generation," but not the main comparison the advisor asked for.

## 5. Dataset

**CUB only, first** — the one dataset with real human attribute labels to score concept quality against, and existing Phase 0 / Plan 08 numbers to sit next to. Given Plan 08 found the LLM-bank effect *reversed* between CUB and PACS, this result should be treated as CUB-specific until (if useful) extended — not assumed to generalize.

## 6. Risks and open questions

- **Threshold-dependence of the CLIP-similarity coverage/precision metric** — needs hand-validation on a sample before the numbers are trusted, same caution as any unvalidated automatic metric in this project.
- **Image-sampling variance**: which 5 images get shown per class could itself affect what a VLM notices; not sweeping this initially (fixed sample, documented), but worth flagging as a limitation, not hiding it.
- **Gemini API cost/quota**: 200 classes × however many images-per-call could hit free-tier rate limits — check AI Studio quota before running the full sweep, do a small pilot (e.g. 10 classes) first.
- **Open-source VLM choice**: Qwen2-VL-7B-Instruct is a default, not a commitment — worth checking current (2026) SOTA open-weight VLM leaderboards before committing, since this space moves fast.
- **Apples-to-apples with Plan 08**: Baseline B's text-only bank was generated with a different, less specific prompt than this plan's image-grounded one — worth prompting Claude here with as close to the *same* instruction as Plan 08 used (minus the images-vs-no-images difference) so the comparison isolates vision, not prompt wording.

## 7. Code (not yet written)

Planned additions under `external/LanCE/`, following the existing pattern:
- `data/CUB/generate_concepts_vlm.py` (or similar) — per-VLM concept generation script, image sampling, pooling.
- `data/CUB/cub_concepts_claude_grounded.txt`, `cub_concepts_gemini.txt`, `cub_concepts_<opensource>.txt` — output concept banks.
- `experiments/component_vlm_concept_quality.py` — CLIP-similarity coverage/precision scoring against real attributes.
- Reuse `train_cached.py` unmodified (as Plan 08 did) if/when the downstream-accuracy stretch goal is run.
