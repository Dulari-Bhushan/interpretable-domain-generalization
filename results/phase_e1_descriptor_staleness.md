# Phase E1 — descriptor-set staleness check (Pillar 2)

**Status: confirmed, unambiguously.** The frozen 204-entry descriptor pool that every phase in this project trains against contains **zero** terms naming an AI-generated-image domain, and **none** of the 5 specific generators Phase F3 tested (Stable Diffusion 2.1/XL/3, DALL-E 3, Midjourney v6) are named anywhere in it.

## What this tests, and how it's different from F1/F3/F4

Phases F1/F3/F4 test whether CLIP's *image-text alignment* holds up for a domain outside its training distribution — using text prompts we hand-wrote ourselves (e.g. `"a Midjourney-generated image of a {}."`). Those experiments assume the right words are available and ask whether CLIP can connect them to the right images.

This phase asks a narrower, separate question, flagged in [README.md:81](../README.md) as planned but never run: **does LanCE's own frozen descriptor pool — 204 phrases generated once by GPT-3.5-turbo before training and never revisited — contain the right words in the first place?** GPT-3.5-turbo's training data has a cutoff around September 2021. Every generator Phase F3 tested shipped after that (Stable Diffusion 2.1: Dec 2022, through Stable Diffusion 3: 2024) — so GPT-3.5 could not have named any of them when the pool was generated, independent of anything CLIP can or can't do visually.

## What we did

Rather than pay for a fresh API call to a GPT-3.5-class model (whose live-queried output can't be verified as an exact match to what LanCE's authors actually used, and would cost real money for a question this codebase can already answer directly), we inspected the actual, already-generated, already-deployed pool: `external/LanCE/prompts/prompt200new.py`'s `target_text_prompts`, the literal 204 phrases every training run in this project (Phase 0 through F2) has been regularizing against. This is a more faithful test than a new sample would be — it's the exact artifact LanCE ships, not a fresh draw that could differ by model snapshot.

Searched all 204 entries (case-insensitive) for: (a) 20 direct AI-generation terms — "AI-generated," "Stable Diffusion," "DALL-E," "Midjourney," "diffusion model," "GAN," "synthetic image," "deepfake," and similar; (b) each of Phase F3's 5 specific generator names individually; (c) 10 broader, conceptually-adjacent terms already known to exist in the pool from a first read, to characterize what *is* there instead.

## Result

| Check | Result |
|---|---|
| Direct AI-generation terms found (of 20 checked) | **0 / 20** |
| Specific generators named (of Phase F3's 5) | **0 / 5** |
| Conceptually-adjacent terms found (of 10 checked) | **10 / 10** |

The pool does contain terms in the same general neighborhood — `"a digital art of a {}."`, `"a CGI render of a {}."`, `"a vector graphic of a {}."`, `"a VR model of a {}."`, `"a hologram of a {}."` / `"a 3D hologram of a {}."`, `"a sci-fi style of a {}."`, `"a futuristic concept art of a {}."`, `"a cyberpunk illustration of a {}."`, `"an augmented reality filter of a {}."` — but none of these describe *photorealistic AI-generated imagery mimicking a real photograph*, which is what Stable Diffusion/DALL-E/Midjourney actually produce and what the Defactify dataset (Phase F3/E2) pairs against real photos to test. "Digital art" and "CGI render" name a *style* (something visibly art-like or rendered); modern diffusion-model output is often designed to look indistinguishable from a real photo — a different domain than anything in this list describes.

## Interpretation

This is a clean, unambiguous confirmation of the gap flagged as open in the README. It's also mechanistically distinct from what F1/F3/F4 measured: those three showed that even when you *supply the right words yourself*, CLIP's embedding space doesn't reliably connect them to real images of that domain (alignment scores of 0.05–0.32 against the paper's 0.90–0.99). This phase shows a failure one step earlier in the pipeline — LanCE's own automatic descriptor-generation mechanism would never have produced the right words to begin with, for any domain that emerged after GPT-3.5's training cutoff. The two failures compound rather than overlap: even a hypothetical future CLIP with perfect alignment for new domains couldn't help LanCE here, because DDO's regularizer only ever sees the 204 phrases in this fixed list — it has no mechanism to add "a Midjourney-generated image of a {}." to its own vocabulary without a human manually re-prompting the LLM and retraining from scratch.

**What this changes for the overall argument:** it completes Failure Mode 1 and Failure Mode 3's picture together. Phase A already showed the closed-world assumption doesn't bite at the "wrong specific style within a covered family" level (removing similar-to-painting descriptors didn't hurt accuracy). This phase shows where it does bite: an entire *category* of domain — anything born after the descriptor-generating LLM's cutoff — has no representation in the pool at all, not even a distant one. Phase E2 (below) tests what this actually costs in trained-model accuracy, not just in descriptor-list coverage.

## Honest limitations of this experiment

- This checks LanCE's own already-generated pool, not a fresh GPT-3.5-turbo API call — a deliberate choice (see above), but it means we're trusting that the checked-in `prompt200new.py` file is representative of what GPT-3.5 would produce if re-prompted today under the same instructions. It very likely is (nothing suggests the file was hand-edited after generation), but this wasn't independently re-verified against a live model.
- Keyword matching only checks for literal term presence, not semantic coverage — a descriptor could name a related concept using different words we didn't think to check. The 10 adjacent-term matches found here (digital art, CGI, etc.) show the pool does contain some semantically-nearby entries; whether DDO's concept-activation-space mechanism could partially generalize from those (the same equalizing effect Phase A's null result suggested) is exactly what Phase E2 tests empirically rather than assuming either way.
- Only the 5 generators from Phase F3 were checked individually; the broader question ("would GPT-3.5 name *any* post-cutoff visual domain, not just AI generators") is not fully answered by one category of domain.
