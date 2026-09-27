"""Per-image LanCE vs VLM agreement pilot - constrained-vocabulary comparison.

Companion to agreement_pilot_compare.py, for the --constrained VLM query
mode where the VLM selects concepts from LanCE's own 312-phrase vocabulary
instead of writing free text. Since both sides now draw from the identical
discrete vocabulary, overlap is exact set membership (Jaccard/recall) - no
CLIP embedding needed at all, and no fuzzy-similarity floor to worry about.

Still includes the same shuffled-pairing control as the free-form version:
LanCE's concepts for image A scored against a different, randomly-paired
image's VLM selections, as the noise floor a real pairing needs to beat.

Usage:
    python agreement_pilot_compare_constrained.py --vlm qwen2vl
    python agreement_pilot_compare_constrained.py --vlm gemini
"""
import argparse
import json
import random

from agreement_pilot_common import load_lance_pilot


def set_overlap(lance_concepts, vlm_concepts):
    a, b = set(lance_concepts), set(vlm_concepts)
    if not a or not b:
        return None
    inter = a & b
    union = a | b
    return {
        "jaccard": len(inter) / len(union) if union else 0.0,
        "recall": len(inter) / len(a),
        "precision": len(inter) / len(b),
        "n_intersection": len(inter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True, help="name matching agreement_pilot_<vlm>_constrained.json")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    lance_pilot = load_lance_pilot()
    lance_by_path = {r["image_path"]: r for r in lance_pilot["results"]}

    with open(f"agreement_pilot_{args.vlm}_constrained.json", encoding="utf-8") as f:
        vlm_results = json.load(f)

    items = []
    for vlm_r in vlm_results:
        lance_r = lance_by_path.get(vlm_r["image_path"])
        if lance_r is None:
            continue
        lance_concepts = [c["concept"] for c in lance_r["lance_top_concepts"]]
        items.append({"lance_r": lance_r, "vlm_r": vlm_r, "lance_concepts": lance_concepts})

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
        real = set_overlap(item["lance_concepts"], vlm_r["vlm_concepts"])
        shuf = set_overlap(item["lance_concepts"], items[shuffle_idx[i]]["vlm_r"]["vlm_concepts"])
        rows.append(
            {
                "image_path": vlm_r["image_path"],
                "true_class": lance_r["true_class"],
                "lance_pred": lance_r["lance_pred_class"],
                "lance_correct": lance_r["lance_correct"],
                "vlm_pred": vlm_r["vlm_pred_class"],
                "vlm_correct": vlm_r["vlm_correct"],
                "lance_vlm_agree": lance_r["lance_pred_class"] == vlm_r["vlm_pred_class"],
                "vlm_n_invalid_concepts": vlm_r.get("vlm_n_invalid_concepts"),
                "n_vlm_concepts": len(vlm_r["vlm_concepts"]),
                "overlap_real": real,
                "overlap_shuffled": shuf,
            }
        )

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    n_rows = len(rows)
    lance_acc = sum(r["lance_correct"] for r in rows) / n_rows
    vlm_acc = sum(r["vlm_correct"] for r in rows) / n_rows
    agree_rate = sum(r["lance_vlm_agree"] for r in rows) / n_rows

    real_jaccard = [r["overlap_real"]["jaccard"] for r in rows if r["overlap_real"]]
    shuf_jaccard = [r["overlap_shuffled"]["jaccard"] for r in rows if r["overlap_shuffled"]]
    agree_jaccard = [r["overlap_real"]["jaccard"] for r in rows if r["lance_vlm_agree"] and r["overlap_real"]]
    disagree_jaccard = [r["overlap_real"]["jaccard"] for r in rows if not r["lance_vlm_agree"] and r["overlap_real"]]
    real_recall = [r["overlap_real"]["recall"] for r in rows if r["overlap_real"]]
    shuf_recall = [r["overlap_shuffled"]["recall"] for r in rows if r["overlap_shuffled"]]

    summary = {
        "n_images": n_rows,
        "lance_accuracy": lance_acc,
        f"{args.vlm}_accuracy": vlm_acc,
        "lance_vlm_agreement_rate": agree_rate,
        "mean_n_valid_concepts_selected": avg([r["n_vlm_concepts"] for r in rows]),
        "mean_n_invalid_selections": avg([r["vlm_n_invalid_concepts"] for r in rows]),
        "jaccard_real": avg(real_jaccard),
        "jaccard_shuffled": avg(shuf_jaccard),
        "jaccard_when_agree": avg(agree_jaccard),
        "jaccard_when_disagree": avg(disagree_jaccard),
        "recall_real": avg(real_recall),
        "recall_shuffled": avg(shuf_recall),
        "rows": rows,
    }

    out_path = f"agreement_pilot_comparison_{args.vlm}_constrained.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print(f"\n=== LanCE vs {args.vlm} (constrained vocab), {n_rows} pilot images ===")
    print(f"  LanCE accuracy: {lance_acc:.1%}   {args.vlm} accuracy: {vlm_acc:.1%}   agree: {agree_rate:.1%}")
    print(f"  mean valid concepts selected: {fmt(avg([r['n_vlm_concepts'] for r in rows]))}   mean invalid (hallucinated) selections: {fmt(avg([r['vlm_n_invalid_concepts'] for r in rows]))}")
    print(f"  Jaccard overlap: real={fmt(avg(real_jaccard))}  shuffled={fmt(avg(shuf_jaccard))}")
    print(f"  Jaccard when agree/disagree: {fmt(avg(agree_jaccard))} / {fmt(avg(disagree_jaccard))}")
    print(f"  recall (LanCE's own top concepts also picked by VLM): real={fmt(avg(real_recall))}  shuffled={fmt(avg(shuf_recall))}")
    n_empty_vlm_sets = sum(1 for r in rows if r["n_vlm_concepts"] == 0)
    if n_empty_vlm_sets:
        print(f"  WARNING: {n_empty_vlm_sets}/{n_rows} images had ZERO valid (vocabulary-matched) VLM concept selections")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
