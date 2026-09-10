"""Plan 09 Sec.3: score a generated concept bank's quality against CUB's real
312 human attributes, using CLIP ViT-L/14 text-embedding similarity (same
encoder the rest of this project already uses) rather than exact string
match.

Reports two threshold-free numbers (the primary metric, since any hard
threshold on CLIP-text cosine similarity is itself unvalidated — see Plan 09
Sec.6):
  - coverage: for each real attribute, the similarity to its single best-
    matching generated concept, averaged over all 312 attributes.
  - precision: for each generated concept, the similarity to its single
    best-matching real attribute, averaged over all generated concepts.
Also reports thresholded coverage/precision at a couple of threshold values,
clearly labeled as threshold-dependent, plus a sample of best/worst matched
pairs so the numbers can be sanity-checked by eye rather than trusted blind.

Must run where CLIP + torch are available (the lab GPU server, not the local
machine) — see docs/session_handoff.md for SSH details.

Usage:
    python vlm_concepts_eval_quality.py --bank cub_concepts_qwen2vl_grounded.txt
    python vlm_concepts_eval_quality.py --bank cub_concepts_llm.txt --bank cub_concepts_llm2.txt
"""
import argparse
import json
from pathlib import Path

CUB_DIR = Path(__file__).parent
REFERENCE_FILE = CUB_DIR / "cub_concepts.txt"
THRESHOLDS = [0.85, 0.90]
N_SAMPLE_PAIRS = 10


def load_phrases(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def embed(clip_model, device, phrases, clip_module):
    import torch

    tokens = clip_module.tokenize(phrases, truncate=True).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens).float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", required=True, help="concept bank file(s) to score, repeatable")
    parser.add_argument("--reference", default=str(REFERENCE_FILE))
    parser.add_argument("--clip-model", default="ViT-L/14")
    args = parser.parse_args()

    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()

    reference_phrases = load_phrases(args.reference)
    ref_emb = embed(clip_model, device, reference_phrases, clip)

    results = {}
    for bank_path in args.bank:
        candidate_phrases = load_phrases(bank_path)
        cand_emb = embed(clip_model, device, candidate_phrases, clip)

        # [n_candidates, n_reference]
        sim = cand_emb @ ref_emb.T

        # Coverage: per reference attribute, best-matching candidate.
        coverage_best, coverage_idx = sim.max(dim=0)
        # Precision: per candidate concept, best-matching reference attribute.
        precision_best, precision_idx = sim.max(dim=1)

        coverage_mean = coverage_best.mean().item()
        precision_mean = precision_best.mean().item()

        thresholded = {}
        for t in THRESHOLDS:
            thresholded[t] = {
                "coverage_at_t": (coverage_best >= t).float().mean().item(),
                "precision_at_t": (precision_best >= t).float().mean().item(),
            }

        # Sample pairs for manual sanity-check: best and worst precision matches.
        order = torch.argsort(precision_best)
        worst_idx = order[:N_SAMPLE_PAIRS].tolist()
        best_idx = order[-N_SAMPLE_PAIRS:].tolist()
        sample_pairs = {
            "best_precision_matches": [
                {
                    "candidate": candidate_phrases[i],
                    "matched_reference": reference_phrases[precision_idx[i].item()],
                    "similarity": precision_best[i].item(),
                }
                for i in best_idx
            ],
            "worst_precision_matches": [
                {
                    "candidate": candidate_phrases[i],
                    "matched_reference": reference_phrases[precision_idx[i].item()],
                    "similarity": precision_best[i].item(),
                }
                for i in worst_idx
            ],
        }

        bank_name = Path(bank_path).stem
        results[bank_name] = {
            "n_candidates": len(candidate_phrases),
            "n_reference": len(reference_phrases),
            "coverage_mean_best_similarity": coverage_mean,
            "precision_mean_best_similarity": precision_mean,
            "thresholded": {str(t): v for t, v in thresholded.items()},
            "sample_pairs": sample_pairs,
        }

        print(f"\n=== {bank_name} ({len(candidate_phrases)} concepts vs {len(reference_phrases)} real attributes) ===")
        print(f"  coverage (mean best-match similarity, GT->candidate): {coverage_mean:.4f}")
        print(f"  precision (mean best-match similarity, candidate->GT): {precision_mean:.4f}")
        for t in THRESHOLDS:
            print(f"  @ threshold {t}: coverage={thresholded[t]['coverage_at_t']:.4f}  precision={thresholded[t]['precision_at_t']:.4f}")
        print("  worst precision matches (sanity-check the metric itself):")
        for pair in sample_pairs["worst_precision_matches"][:5]:
            print(f"    [{pair['similarity']:.3f}] '{pair['candidate']}' -> '{pair['matched_reference']}'")
        print("  best precision matches:")
        for pair in sample_pairs["best_precision_matches"][:5]:
            print(f"    [{pair['similarity']:.3f}] '{pair['candidate']}' -> '{pair['matched_reference']}'")

    out_path = CUB_DIR / "vlm_concepts" / "quality_eval_results.json"
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
