"""Per-image LanCE vs VLM agreement pilot (following up on Plan 09) - LanCE
side. For a sampled set of CUB source-test images, extracts LanCE's own
per-image top-K activated concepts, predicted class, and confidence, using
LanCE's canonical config (human-written concept bank, clip_cbm_orth, alpha=1
+DDO - Phase 0's own setup). Written to a JSON file so it can be compared
against a VLM shown the *same raw images* (agreement_pilot_query_vlm.py).

This is a genuinely different question from the rest of Plan 09: not "does
swapping a VLM-generated concept *bank* into training change accuracy" but
"does a VLM's live per-image judgement (concepts + predicted class) agree
with LanCE's own already-trained per-image judgement, image by image."

No checkpoint existed to load (this project doesn't commit model
checkpoints - gitignored, regenerable), so this trains LanCE's canonical
model fresh (fast: cached embeddings, ~5 min) and reloads the actual
best-epoch checkpoint (train_cached.py's --save_model, on by default, saves
the best-target-acc epoch's weights - the in-memory model after run()
finishes holds the *last* epoch's weights, not necessarily the best one).

Usage (run from external/LanCE/, same environment as train_cached.py):
    python agreement_pilot_extract_lance.py --n-images 25 --top-k 8
"""
import argparse
import json
import random
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=25)
    parser.add_argument("--sample-seed", type=int, default=42, help="seed for which test images get sampled (separate from training's own --seed)")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default="agreement_pilot_lance.json")
    cli = parser.parse_args()

    # Train LanCE's own canonical model (human bank, alpha=1/+DDO) - the
    # exact config Phase 0 and every comparison in this project measures
    # against. --save_model defaults to True in this project's args.py.
    sys.argv = [
        "train_cached.py", "--dataset", "CUB", "--alpha", "1", "--epochs", "50",
        "--batch_size", "64", "--class_avg_concept", "--CBM_type", "clip_cbm",
        "--concept_file", "cub_concepts.txt",
    ]
    from args import get_args
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from train_cached import CachedTrainingSession

    session = CachedTrainingSession(args)
    session.run()

    run_tag = f"{args.dataset}_{args.CBM_type}_alpha{args.alpha}"
    ckpt_path = f"checkpoints/best_{run_tag}.pth"
    session.model.load_state_dict(torch.load(ckpt_path, map_location=session.device))
    session.model.eval()

    concept_names = [
        line.rstrip("\n") for line in open("data/CUB/cub_concepts.txt", encoding="utf-8") if line.strip()
    ]
    with open("data/CUB/CUB_200_2011/classes.txt", encoding="utf-8") as f:
        class_names = [x.split(" ")[1][4:].replace("_", " ").lower().rstrip() for x in f.readlines()]

    with open("data/CUB/cub_test.txt", encoding="utf-8") as f:
        test_rows = [line.strip().split(",") for line in f.readlines()]
    test_paths = [r[0] for r in test_rows]
    test_labels = [int(r[1]) - 1 for r in test_rows]

    # Cached source-test features were built with shuffle=False (cache_utils.py),
    # so cache index i lines up exactly with cub_test.txt line i - verified below
    # rather than just assumed.
    cached = torch.load("embeddings_cache/CUB_ViT-L-14_source_test.pt", map_location=session.device)
    feats = cached["features"]
    cached_labels = cached["labels"].tolist()
    if cached_labels != test_labels:
        raise RuntimeError("Cache order does not match cub_test.txt order - the index-alignment assumption this script depends on is wrong.")

    rng = random.Random(cli.sample_seed)
    indices = rng.sample(range(len(test_paths)), cli.n_images)

    results = []
    with torch.no_grad():
        for idx in indices:
            feat = feats[idx : idx + 1].to(session.device)
            concept_activations, cls_preds, _ = session.model.forward_cached(feat)
            # concept_activations is raw image<->concept-text cosine similarity
            # (see model/cbm_models.py forward_cached) - same kind of quantity
            # used throughout this project's other CLIP-similarity comparisons,
            # so no sigmoid squashing here (it would only rescale, not reorder).
            concept_scores = concept_activations[0]
            cls_probs = torch.softmax(cls_preds, dim=-1)[0]
            pred_class_id = cls_probs.argmax().item()
            confidence = cls_probs[pred_class_id].item()
            topk = torch.topk(concept_scores, cli.top_k)
            top_concepts = [
                {"concept": concept_names[j], "score": concept_scores[j].item()} for j in topk.indices.tolist()
            ]
            results.append(
                {
                    "image_path": test_paths[idx],
                    "true_class": class_names[test_labels[idx]],
                    "lance_pred_class": class_names[pred_class_id],
                    "lance_confidence": confidence,
                    "lance_correct": pred_class_id == test_labels[idx],
                    "lance_top_concepts": top_concepts,
                }
            )

    with open(cli.out, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "results": results}, f, indent=2)
    n_correct = sum(r["lance_correct"] for r in results)
    print(f"Wrote {len(results)} pilot images -> {cli.out} (LanCE correct: {n_correct}/{len(results)})")


if __name__ == "__main__":
    main()
