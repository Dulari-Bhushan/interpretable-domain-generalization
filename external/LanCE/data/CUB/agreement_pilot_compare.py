"""Per-image LanCE vs VLM agreement pilot - final comparison.

Joins LanCE's per-image predictions/concepts (agreement_pilot_lance.json)
against one or more VLMs' per-image predictions/concepts on the exact same
images, and reports:
  - classification agreement: LanCE vs ground truth, VLM vs ground truth,
    LanCE vs VLM (do the two systems land on the same class independently of
    whether either is right), broken down by confidence.
  - concept agreement: CLIP-similarity overlap between LanCE's top-K
    concepts (per-concept contribution to the predicted class's own logit -
    see agreement_pilot_extract_lance.py - not raw image similarity, which
    only reflects "what's visible in the photo," true for any system looking
    at the same image regardless of whether their decisions actually agree)
    and the VLM's freshly-generated per-image concepts.

    Reported two ways, because a first pass at this (mean best-match
    similarity) turned out to be flat (~0.66) whether the two systems agreed
    on the class or not - CLIP similarity between short bird-attribute
    phrases has a high floor just from being in the same narrow domain,
    independent of real content overlap:
      - threshold-based overlap (fraction of concepts with a close, >=0.85,
        match) - a stricter, more decisive criterion than an averaged score.
      - a shuffled-pairing control: the SAME metric computed against a
        different, randomly-paired image's VLM concepts instead of the
        image's own match. This is the noise floor a real pairing needs to
        exceed to mean anything; if real and shuffled scores are
        indistinguishable, the metric isn't detecting real agreement.

Must run where CLIP + torch are available (the lab GPU server).

Usage:
    python agreement_pilot_compare.py --vlm qwen2vl
    python agreement_pilot_compare.py --vlm gemini
"""
import argparse
import json
import random

from agreement_pilot_common import load_lance_pilot

THRESHOLD = 0.85


def score_pair(sim):
    """sim: [n_lance, n_vlm] cosine similarity matrix -> overlap stats."""
    lance_to_vlm = sim.max(dim=1).values
    vlm_to_lance = sim.max(dim=0).values
    return {
        "mean": ((lance_to_vlm.mean() + vlm_to_lance.mean()) / 2).item(),
        "at_threshold": (((lance_to_vlm >= THRESHOLD).float().mean() + (vlm_to_lance >= THRESHOLD).float().mean()) / 2).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True, help="name matching agreement_pilot_<vlm>.json")
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()

    lance_pilot = load_lance_pilot()
    lance_by_path = {r["image_path"]: r for r in lance_pilot["results"]}

    with open(f"agreement_pilot_{args.vlm}.json", encoding="utf-8") as f:
        vlm_results = json.load(f)

    def embed(phrases):
        if not phrases:
            return None
        tokens = clip.tokenize(phrases, truncate=True).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).float()
        return emb / emb.norm(dim=-1, keepdim=True)

    items = []
    for vlm_r in vlm_results:
        lance_r = lance_by_path.get(vlm_r["image_path"])
        if lance_r is None:
            continue
        lance_concepts = [c["concept"] for c in lance_r["lance_top_concepts"]]
        lance_emb = embed(lance_concepts)
        vlm_emb = embed(vlm_r["vlm_concepts"])
        items.append({"lance_r": lance_r, "vlm_r": vlm_r, "lance_emb": lance_emb, "vlm_emb": vlm_emb})

    # Fixed derangement (no image paired with itself) for the shuffled control.
    n = len(items)
    rng = random.Random(args.shuffle_seed)
    shuffle_idx = list(range(n))
    while True:
        rng.shuffle(shuffle_idx)
        if all(i != j for i, j in enumerate(shuffle_idx)):
            break

    rows = []
    for i, item in enumerate(items):
        lance_r, vlm_r = item["lance_r"], item["vlm_r"]
        real_overlap = shuf_overlap = None
        if item["lance_emb"] is not None and item["vlm_emb"] is not None:
            real_overlap = score_pair(item["lance_emb"] @ item["vlm_emb"].T)
        shuf_vlm_emb = items[shuffle_idx[i]]["vlm_emb"]
        if item["lance_emb"] is not None and shuf_vlm_emb is not None:
            shuf_overlap = score_pair(item["lance_emb"] @ shuf_vlm_emb.T)

        rows.append(
            {
                "image_path": vlm_r["image_path"],
                "true_class": lance_r["true_class"],
                "lance_pred": lance_r["lance_pred_class"],
                "lance_confidence": lance_r["lance_confidence"],
                "lance_correct": lance_r["lance_correct"],
                "vlm_pred": vlm_r["vlm_pred_class"],
                "vlm_confidence": vlm_r["vlm_confidence"],
                "vlm_correct": vlm_r["vlm_correct"],
                "lance_vlm_agree": lance_r["lance_pred_class"] == vlm_r["vlm_pred_class"],
                "concept_overlap": real_overlap,
                "concept_overlap_shuffled": shuf_overlap,
            }
        )

    n_rows = len(rows)
    lance_acc = sum(r["lance_correct"] for r in rows) / n_rows
    vlm_acc = sum(r["vlm_correct"] for r in rows) / n_rows
    agree_rate = sum(r["lance_vlm_agree"] for r in rows) / n_rows

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    real_means = [r["concept_overlap"]["mean"] for r in rows if r["concept_overlap"]]
    shuf_means = [r["concept_overlap_shuffled"]["mean"] for r in rows if r["concept_overlap_shuffled"]]
    real_thresh = [r["concept_overlap"]["at_threshold"] for r in rows if r["concept_overlap"]]
    shuf_thresh = [r["concept_overlap_shuffled"]["at_threshold"] for r in rows if r["concept_overlap_shuffled"]]

    agree_real_means = [r["concept_overlap"]["mean"] for r in rows if r["lance_vlm_agree"] and r["concept_overlap"]]
    disagree_real_means = [r["concept_overlap"]["mean"] for r in rows if not r["lance_vlm_agree"] and r["concept_overlap"]]
    agree_real_thresh = [r["concept_overlap"]["at_threshold"] for r in rows if r["lance_vlm_agree"] and r["concept_overlap"]]
    disagree_real_thresh = [r["concept_overlap"]["at_threshold"] for r in rows if not r["lance_vlm_agree"] and r["concept_overlap"]]

    lance_conf_correct = [r["lance_confidence"] for r in rows if r["lance_correct"]]
    lance_conf_wrong = [r["lance_confidence"] for r in rows if not r["lance_correct"]]
    vlm_conf_correct = [r["vlm_confidence"] for r in rows if r["vlm_correct"] and r["vlm_confidence"] is not None]
    vlm_conf_wrong = [r["vlm_confidence"] for r in rows if not r["vlm_correct"] and r["vlm_confidence"] is not None]

    summary = {
        "n_images": n_rows,
        "lance_accuracy": lance_acc,
        f"{args.vlm}_accuracy": vlm_acc,
        "lance_vlm_agreement_rate": agree_rate,
        "concept_overlap_mean_real": avg(real_means),
        "concept_overlap_mean_shuffled": avg(shuf_means),
        "concept_overlap_at_threshold_real": avg(real_thresh),
        "concept_overlap_at_threshold_shuffled": avg(shuf_thresh),
        "concept_overlap_mean_when_agree": avg(agree_real_means),
        "concept_overlap_mean_when_disagree": avg(disagree_real_means),
        "concept_overlap_at_threshold_when_agree": avg(agree_real_thresh),
        "concept_overlap_at_threshold_when_disagree": avg(disagree_real_thresh),
        "lance_confidence_when_correct": avg(lance_conf_correct),
        "lance_confidence_when_wrong": avg(lance_conf_wrong),
        f"{args.vlm}_confidence_when_correct": avg(vlm_conf_correct),
        f"{args.vlm}_confidence_when_wrong": avg(vlm_conf_wrong),
        "rows": rows,
    }

    out_path = f"agreement_pilot_comparison_{args.vlm}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== LanCE vs {args.vlm}, {n_rows} pilot images ===")
    print(f"  LanCE accuracy:          {lance_acc:.1%}")
    print(f"  {args.vlm} accuracy:     {vlm_acc:.1%}")
    print(f"  LanCE-{args.vlm} agree:  {agree_rate:.1%}  (independent of ground truth)")
    print(f"  concept overlap (mean):       real={avg(real_means):.3f}  shuffled={avg(shuf_means):.3f}")
    print(f"  concept overlap (>={THRESHOLD}): real={avg(real_thresh):.3f}  shuffled={avg(shuf_thresh):.3f}")
    print(f"  mean overlap   when agree/disagree:       {avg(agree_real_means):.3f} / {avg(disagree_real_means):.3f}")
    print(f"  threshold overlap when agree/disagree:    {avg(agree_real_thresh):.3f} / {avg(disagree_real_thresh):.3f}")
    print(f"  LanCE confidence correct/wrong: {avg(lance_conf_correct):.3f} / {avg(lance_conf_wrong):.3f}")
    print(f"  {args.vlm} confidence correct/wrong: {avg(vlm_conf_correct)} / {avg(vlm_conf_wrong)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
