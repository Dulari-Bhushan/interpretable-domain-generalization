# Component 1b — L1 vs. L2 DDO ablation

## 1. Status

⚠️ **Done — result confirmed, and it's genuinely mixed.** Both models trained and evaluated for real, on identical data. The result isn't a clean pass — it cuts differently on the two benchmarks tested, and that difference is itself the finding.

## 2. One-line summary

Swapping DDO's original L1 penalty for the L2 surrogate (needed to make Component 1's exact update possible) costs nothing on PACS and even improves both accuracy and the orthogonality property — but on Office-Home it trades 1.75 points of accuracy for a much stronger (5–26x smaller) orthogonality signal.

## 3. Origin

Directly from [`results/component1_exact_classifier.md`](component1_exact_classifier.md) §12, improvement #1 — flagged there as "the most important open question about Component 1 specifically."

## 4. The issue this targets

Not a new failure hypothesis — a validity question about Component 1 itself. Component 1's exact incremental classifier had to substitute DDO's original L1 (mean-absolute) orthogonality penalty for an L2 (mean-square) surrogate to get a closed-form solution (see `results/component1_exact_classifier.md` §6). That substitution was flagged honestly but never actually checked against the original L1-trained model under matched conditions — same data, same domains, same descriptor pool, both evaluated the same way. This experiment closes that gap.

## 5. Why we tried this approach specifically

The two models being compared aren't hypothetical — one is exactly Phase 0/B/D's own unmodified SGD training loop (`DomainILSession._train_loop`, `clip_cbm_orth`, `alpha=1.0`), the other is Component 1's own analytic classifier (`AnalyticDomainIncrementalClassifier`, `ddo_lambda=1.0`). Running both on the same pooled joint data, with the same descriptor pool, isolates the L1-vs-L2 question from every other variable.

## 6. Method

For each dataset:
1. Train the original model exactly as Phase 0/B/D did: `clip_cbm_orth`, cross-entropy + `alpha * mean(|regularizer|)` (the L1 penalty), AdamW, 50 epochs, batch 64, lr 1e-4, weight decay 1e-4, on all domains pooled.
2. Fit Component 1's analytic classifier on the same pooled data (`ridge_lambda=1.0`, `ddo_lambda=1.0`).
3. For both, measure final accuracy per domain and overall.
4. For both, measure the orthogonality property DDO is actually meant to enforce — how close to zero the classifier's response is to every (descriptor, anchor-class) domain-shift direction — reported two ways on the *same* tensor for both models: `mean(|.|)` (the L1 model's own training objective, and the metric `domain_il.py` already tracks as "ddo_erosion") and `mean(.²)` (the L2 model's own training objective). Reporting both scales for both models means neither model is only ever measured on the metric it was optimized for.

## 7. Dataset(s) used, and why

PACS and Office-Home — the same two benchmarks Component 1 was already validated on, so this result is directly comparable to, and reuses, the existing setup (`DomainILSession`, cached embeddings). No new dataset needed for this question.

## 8. Code

- [`external/LanCE/experiments/component1_l1_vs_l2_ablation.py`](../external/LanCE/experiments/component1_l1_vs_l2_ablation.py) — the full ablation script.
- Reuses, unmodified: `external/LanCE/experiments/domain_il.py` (`DomainILSession`), `external/LanCE/experiments/component1_analytic_domain_il.py` (`_build_classifier`), `external/LanCE/model/cbm_models.py` (`clip_cbm_orth`), `external/LanCE/model/analytic_classifier.py` (`AnalyticDomainIncrementalClassifier`).

Run on the lab server (GPU 0 only, per the 2-GPU sharing limit), conda env `mlgpu`.

## 9. Results

Full raw numbers: [`results/component1b_l1_vs_l2_ablation.json`](component1b_l1_vs_l2_ablation.json).

**PACS**

| | L1-SGD (original) | L2-analytic (Component 1) |
|---|---|---|
| ACC | 98.27% | **98.51%** (+0.25) |
| Orthogonality, mean\|.\| | 0.3375 | **0.3337** (lower) |
| Orthogonality, mean(.²) | 0.2037 | **0.1499** (lower) |

**Office-Home**

| | L1-SGD (original) | L2-analytic (Component 1) |
|---|---|---|
| ACC | **90.78%** | 89.03% (−1.75) |
| Orthogonality, mean\|.\| | 0.4686 | **0.0969** (~4.8x lower) |
| Orthogonality, mean(.²) | 0.3903 | **0.0151** (~26x lower) |

## 10. What this means

**On PACS, the substitution is free — arguably a small win.** L2-analytic matches or beats the original on every number measured, including the L1 model's own preferred metric (mean\|.\|). No evidence of a cost here.

**On Office-Home, there's a real, honest trade-off.** The L2-analytic classifier gives up 1.75 accuracy points, but in exchange it suppresses domain-specific signal dramatically more than the original ever did — not a small difference, roughly 5x smaller on the same scale the original model was trained on, and 26x smaller on the L2 model's own scale. That's not noise; that's the classifier behaving as if the L2 penalty is being enforced far more aggressively than the L1 version ever managed to enforce its own penalty.

**A plausible explanation, not yet confirmed:** Office-Home has 65 classes and much less data per class per domain than PACS's 7 classes — far less slack for a classifier to both fit the data *and* stay orthogonal to every domain direction. The L2 penalty, being smooth and quadratic everywhere (vs. L1's constant-magnitude gradient), may push harder on directions the original L1 penalty left comparatively alone, trading some class-discriminative signal for a stronger orthogonality guarantee, specifically in the low-data-per-class regime. This is a plausible mechanism, not a proven one — it would need a targeted follow-up (e.g. checking whether the accuracy gap concentrates in specific classes with the least data) to confirm.

## 11. Verdict

**Answers the open question, and the answer is "it depends on the benchmark," not a clean yes or no.** The L1→L2 substitution is not a free lunch in general — Component 1's exact-forgetting guarantee still holds unconditionally (that's a separate, already-proven property, unaffected by this result), but the specific accuracy Component 1 delivers on a hard, class-crowded benchmark is measurably lower than what the original L1-trained model achieves, in exchange for much stronger orthogonality. Component 1's exact-match claim from `results/component1_exact_classifier.md` is unaffected by this — it was always a claim about matching *joint retraining of the L2 objective*, not matching the *L1 objective's own accuracy*. This result is what makes that distinction concrete rather than theoretical.

## 12. What's next

Not a dead end — a real, answerable follow-up:

1. **Find out whether the accuracy gap is recoverable by tuning `ddo_lambda` down** on Office-Home specifically — right now it's set to 1.0 to match the original `alpha=1.0`, untuned. If a smaller `ddo_lambda` closes most of the accuracy gap while keeping orthogonality meaningfully better than the L1 baseline, that's a strictly better operating point, still with the exact-update guarantee intact (the guarantee holds for any fixed `ddo_lambda`).
2. **Check whether the accuracy gap concentrates in specific classes** (per the mechanism hypothesis in §10) rather than spreading evenly — this would confirm or rule out the low-data-per-class explanation directly.
3. Carry this same ablation forward to any future dataset Component 1 gets tested on (DomainNet, per `results/component1_exact_classifier.md` §12) so the L1-vs-L2 trade-off is characterized at scale too, not just on these two benchmarks.
