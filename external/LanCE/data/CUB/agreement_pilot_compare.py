"""Per-image LanCE vs VLM agreement pilot - final comparison.

Joins LanCE's per-image predictions/concepts (agreement_pilot_lance.json)
against one or more VLMs' per-image predictions/concepts on the exact same
images, and reports:
  - classification agreement: LanCE vs ground truth, VLM vs ground truth,
    LanCE vs VLM (do the two systems land on the same class independently of
    whether either is right), broken down by confidence.
  - concept agreement: CLIP-similarity overlap between LanCE's top-K
    activated concepts and the VLM's freshly-generated per-image concepts,
    same coverage/precision-style scoring as vlm_concepts_eval_quality.py,
    applied per image instead of per whole concept bank.

Must run where CLIP + torch are available (the lab GPU server).

Usage:
    python agreement_pilot_compare.py --vlm qwen2vl
    python agreement_pilot_compare.py --vlm gemini
"""
import argparse
import json

from agreement_pilot_common import load_lance_pilot


def concept_overlap(clip_model, device, clip_module, lance_concepts, vlm_concepts):
    """Symmetric coverage/precision-style overlap between two per-image concept
    lists, same mean-best-match-similarity convention as
    vlm_concepts_eval_quality.py, just applied to one image's two concept
    lists instead of two whole concept banks."""
    import torch

    if not lance_concepts or not vlm_concepts:
        return None

    def embed(phrases):
        tokens = clip_module.tokenize(phrases, truncate=True).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).float()
        return emb / emb.norm(dim=-1, keepdim=True)

    lance_emb = embed(lance_concepts)
    vlm_emb = embed(vlm_concepts)
    sim = lance_emb @ vlm_emb.T  # [n_lance, n_vlm]
    lance_to_vlm = sim.max(dim=1).values.mean().item()  # each LanCE concept's best VLM match
    vlm_to_lance = sim.max(dim=0).values.mean().item()  # each VLM concept's best LanCE match
    return {"lance_to_vlm": lance_to_vlm, "vlm_to_lance": vlm_to_lance, "mean": (lance_to_vlm + vlm_to_lance) / 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True, help="name matching agreement_pilot_<vlm>.json")
    parser.add_argument("--clip-model", default="ViT-L/14")
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

    rows = []
    for vlm_r in vlm_results:
        lance_r = lance_by_path.get(vlm_r["image_path"])
        if lance_r is None:
            continue
        lance_concepts = [c["concept"] for c in lance_r["lance_top_concepts"]]
        overlap = concept_overlap(clip_model, device, clip, lance_concepts, vlm_r["vlm_concepts"])

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
                "concept_overlap": overlap,
            }
        )

    n = len(rows)
    lance_acc = sum(r["lance_correct"] for r in rows) / n
    vlm_acc = sum(r["vlm_correct"] for r in rows) / n
    agree_rate = sum(r["lance_vlm_agree"] for r in rows) / n
    overlaps = [r["concept_overlap"]["mean"] for r in rows if r["concept_overlap"] is not None]
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else None

    agree_overlaps = [r["concept_overlap"]["mean"] for r in rows if r["lance_vlm_agree"] and r["concept_overlap"]]
    disagree_overlaps = [r["concept_overlap"]["mean"] for r in rows if not r["lance_vlm_agree"] and r["concept_overlap"]]

    lance_conf_correct = [r["lance_confidence"] for r in rows if r["lance_correct"]]
    lance_conf_wrong = [r["lance_confidence"] for r in rows if not r["lance_correct"]]
    vlm_conf_vals = [r["vlm_confidence"] for r in rows if r["vlm_confidence"] is not None]
    vlm_conf_correct = [r["vlm_confidence"] for r in rows if r["vlm_correct"] and r["vlm_confidence"] is not None]
    vlm_conf_wrong = [r["vlm_confidence"] for r in rows if not r["vlm_correct"] and r["vlm_confidence"] is not None]

    def avg(vals):
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_images": n,
        "lance_accuracy": lance_acc,
        f"{args.vlm}_accuracy": vlm_acc,
        "lance_vlm_agreement_rate": agree_rate,
        "mean_concept_overlap": mean_overlap,
        "mean_concept_overlap_when_agree": avg(agree_overlaps),
        "mean_concept_overlap_when_disagree": avg(disagree_overlaps),
        "lance_confidence_when_correct": avg(lance_conf_correct),
        "lance_confidence_when_wrong": avg(lance_conf_wrong),
        f"{args.vlm}_confidence_mean": avg(vlm_conf_vals),
        f"{args.vlm}_confidence_when_correct": avg(vlm_conf_correct),
        f"{args.vlm}_confidence_when_wrong": avg(vlm_conf_wrong),
        "rows": rows,
    }

    out_path = f"agreement_pilot_comparison_{args.vlm}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== LanCE vs {args.vlm}, {n} pilot images ===")
    print(f"  LanCE accuracy:          {lance_acc:.1%}")
    print(f"  {args.vlm} accuracy:     {vlm_acc:.1%}")
    print(f"  LanCE-{args.vlm} agree:  {agree_rate:.1%}  (independent of ground truth)")
    print(f"  mean concept overlap:    {mean_overlap:.3f}" if mean_overlap is not None else "  mean concept overlap: n/a")
    print(f"    ...when they agree:    {avg(agree_overlaps):.3f}" if agree_overlaps else "    ...when they agree: n/a")
    print(f"    ...when they disagree: {avg(disagree_overlaps):.3f}" if disagree_overlaps else "    ...when they disagree: n/a")
    print(f"  LanCE confidence  correct/wrong: {avg(lance_conf_correct):.3f} / {avg(lance_conf_wrong):.3f}" if lance_conf_wrong else f"  LanCE confidence correct: {avg(lance_conf_correct)}")
    print(f"  {args.vlm} confidence correct/wrong: {avg(vlm_conf_correct)} / {avg(vlm_conf_wrong)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
