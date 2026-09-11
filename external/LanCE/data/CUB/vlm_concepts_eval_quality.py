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

Size-normalized comparison (--normalize, on by default when >1 bank is
given): raw coverage is structurally biased toward larger banks — with more
candidate phrases, the single best match to any given reference attribute
tends to be closer just from having more draws, regardless of whether the
bank's concepts are actually any good (an extreme-value-statistics effect,
not a quality signal). To compare banks of very different sizes fairly, each
bank larger than the smallest one is randomly subsampled down to that size,
repeated over many trials, and reported as mean +/- std — this uses each
bank's own concepts throughout (no cherry-picking), so a bank that's simply
larger-but-not-better will not be artificially rewarded for size alone.

Must run where CLIP + torch are available (the lab GPU server, not the local
machine) — see docs/session_handoff.md for SSH details.

Usage:
    python vlm_concepts_eval_quality.py --bank cub_concepts_qwen2vl_grounded.txt
    python vlm_concepts_eval_quality.py --bank cub_concepts_llm.txt --bank cub_concepts_llm2.txt
    python vlm_concepts_eval_quality.py --bank a.txt --bank b.txt --normalize-size 263 --trials 50
"""
import argparse
import json
from pathlib import Path

CUB_DIR = Path(__file__).parent
REFERENCE_FILE = CUB_DIR / "cub_concepts.txt"
THRESHOLDS = [0.85, 0.90]
N_SAMPLE_PAIRS = 10
DEFAULT_TRIALS = 50


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


def run_normalized_comparison(sims_by_bank, normalize_size, trials, seed):
    import torch

    if normalize_size is None:
        normalize_size = min(sim.shape[0] for sim, _ in sims_by_bank.values())

    generator = torch.Generator().manual_seed(seed)
    normalized = {}
    for bank_name, (sim, candidate_phrases) in sims_by_bank.items():
        n = sim.shape[0]
        if n < normalize_size:
            print(f"WARNING: {bank_name} has only {n} concepts, fewer than the normalize size {normalize_size} -- using all {n} as-is (not a fair comparison against subsampled banks)")

        if n <= normalize_size:
            # Nothing to subsample -- the bank IS the sample; report its own
            # full-bank numbers with zero variance rather than fabricate trials.
            coverage_mean = sim.max(dim=0).values.mean().item()
            precision_mean = sim.max(dim=1).values.mean().item()
            normalized[bank_name] = {
                "original_size": n,
                "coverage_mean": coverage_mean,
                "coverage_std": 0.0,
                "precision_mean": precision_mean,
                "precision_std": 0.0,
                "n_trials": 1,
            }
            continue

        coverages, precisions = [], []
        for _ in range(trials):
            idx = torch.randperm(n, generator=generator)[:normalize_size]
            sub = sim[idx]  # [normalize_size, n_reference]
            coverages.append(sub.max(dim=0).values.mean().item())
            precisions.append(sub.max(dim=1).values.mean().item())

        coverages = torch.tensor(coverages)
        precisions = torch.tensor(precisions)
        normalized[bank_name] = {
            "original_size": n,
            "coverage_mean": coverages.mean().item(),
            "coverage_std": coverages.std().item(),
            "precision_mean": precisions.mean().item(),
            "precision_std": precisions.std().item(),
            "n_trials": trials,
        }

    return normalize_size, normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", required=True, help="concept bank file(s) to score, repeatable")
    parser.add_argument("--reference", default=str(REFERENCE_FILE))
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--normalize-size", type=int, default=None, help="subsample every bank down to this many concepts (default: the smallest bank given)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="random subsampling trials for the size-normalized comparison")
    parser.add_argument("--no-normalize", action="store_true", help="skip the size-normalized comparison")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()

    reference_phrases = load_phrases(args.reference)
    ref_emb = embed(clip_model, device, reference_phrases, clip)

    results = {}
    sims_by_bank = {}  # bank_name -> (sim tensor, candidate_phrases) for the normalized pass below
    for bank_path in args.bank:
        candidate_phrases = load_phrases(bank_path)
        cand_emb = embed(clip_model, device, candidate_phrases, clip)

        # [n_candidates, n_reference]
        sim = cand_emb @ ref_emb.T
        sims_by_bank[Path(bank_path).stem] = (sim, candidate_phrases)

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

    if not args.no_normalize and len(sims_by_bank) > 1:
        size_used, normalized = run_normalized_comparison(sims_by_bank, args.normalize_size, args.trials, args.seed)
        for bank_name, stats in normalized.items():
            results[bank_name]["size_normalized"] = stats
        print(f"\n=== size-normalized comparison (subsampled to {size_used} concepts, {args.trials} trials) ===")
        for bank_name, stats in normalized.items():
            print(
                f"  {bank_name} (n={stats['original_size']}): "
                f"coverage={stats['coverage_mean']:.4f}+/-{stats['coverage_std']:.4f}  "
                f"precision={stats['precision_mean']:.4f}+/-{stats['precision_std']:.4f}"
            )

    out_path = CUB_DIR / "vlm_concepts" / "quality_eval_results.json"
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
